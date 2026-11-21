//! Genetic path planning on graphs and grids: a genome is a simple path from start to goal,
//! initialised by randomised depth-first search, recombined at shared vertices, mutated by
//! rerouting a stretch, and repaired by cutting loops.
//!
//! ```
//! use evolutionary_planning::graph_path::{Config, Grid, dijkstra, solve};
//! use evolutionary_planning::{Limits, SplitMix64};
//!
//! // A 10 x 10 grid with a wall that has a gap at the bottom.
//! let mut grid = Grid::open(10, 10, false);
//! for y in 0..9 {
//!     grid.block(5, y);
//! }
//! let (start, goal) = (grid.node(0, 0), grid.node(9, 0));
//! let (path, cost) = solve(&grid, start, goal, &Config::default(), &mut Limits::evaluations(20_000), &mut SplitMix64::new(1)).expect("reachable");
//! assert_eq!(Some(cost), dijkstra(&grid, start, goal).map(|(_, c)| c));
//! assert_eq!((path[0], path[path.len() - 1]), (start, goal));
//! ```

use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::cmp::Ordering;

use rand_core::Rng;

use crate::ga::{Config as GaConfig, Ga, Operators, Scheme};
use crate::operators::selection::Tournament;
use crate::operators::{Crossover, Initializer, Mutation, Repair};
use crate::rng::{below, shuffle, two_sorted_indices};
use crate::{Budget, Objective, Stats};

/// A directed graph with non-negative edge costs over vertices `0..node_count()`.
pub trait Graph {
    /// The number of vertices.
    fn node_count(&self) -> usize;

    /// Appends the out-edges `(target, cost)` of `node` to `out`; costs are non-negative and
    /// finite.
    fn neighbours(&self, node: usize, out: &mut Vec<(usize, f64)>);

    /// The cost of the edge `from -> to`, if it exists (the cheapest of parallel edges).
    fn edge_cost(&self, from: usize, to: usize) -> Option<f64> {
        let mut out = Vec::new();
        self.neighbours(from, &mut out);

        out.into_iter()
            .filter(|&(v, _)| v == to)
            .map(|(_, c)| c)
            .min_by(f64::total_cmp)
    }
}

/// A graph given by adjacency lists.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AdjacencyList {
    /// `edges[v]` lists the out-edges `(target, cost)` of `v`.
    pub edges: Vec<Vec<(usize, f64)>>,
}

impl Graph for AdjacencyList {
    fn node_count(&self) -> usize {
        self.edges.len()
    }

    fn neighbours(&self, node: usize, out: &mut Vec<(usize, f64)>) {
        out.extend_from_slice(&self.edges[node]);
    }
}

/// A 4- or 8-connected grid with blocked cells; straight moves cost `1`, diagonal moves
/// `√2` (and may not cut blocked corners). Vertex `y * width + x` is cell `(x, y)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grid {
    /// The number of columns.
    width: usize,
    /// The number of rows.
    height: usize,
    /// Whether each cell is blocked, row-major.
    blocked: Vec<bool>,
    /// Whether diagonal moves are allowed.
    diagonal: bool,
}

impl Grid {
    /// A grid without obstacles.
    #[must_use]
    pub fn open(width: usize, height: usize, diagonal: bool) -> Self {
        Self {
            width,
            height,
            blocked: alloc::vec![false; width * height],
            diagonal,
        }
    }

    /// Blocks cell `(x, y)`.
    ///
    /// # Panics
    ///
    /// Panics if the cell is outside the grid.
    pub fn block(&mut self, x: usize, y: usize) {
        assert!(x < self.width && y < self.height, "cell outside the grid");
        self.blocked[y * self.width + x] = true;
    }

    /// The vertex of cell `(x, y)`.
    #[must_use]
    pub fn node(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    /// The cell `(x, y)` of a vertex.
    #[must_use]
    pub fn cell(&self, node: usize) -> (usize, usize) {
        (node % self.width, node / self.width)
    }

    /// Whether cell `(x, y)` is free; cells outside the grid are not.
    fn free(&self, x: isize, y: isize) -> bool {
        let (Ok(x), Ok(y)) = (usize::try_from(x), usize::try_from(y)) else {
            return false;
        };

        x < self.width && y < self.height && !self.blocked[y * self.width + x]
    }
}

impl Graph for Grid {
    fn node_count(&self) -> usize {
        self.width * self.height
    }

    fn neighbours(&self, node: usize, out: &mut Vec<(usize, f64)>) {
        let (x, y) = self.cell(node);
        let (Ok(x), Ok(y)) = (isize::try_from(x), isize::try_from(y)) else {
            return;
        };

        if !self.free(x, y) {
            return;
        }

        for (dx, dy) in [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ] {
            let diagonal = dx != 0 && dy != 0;
            if diagonal && !(self.diagonal && self.free(x + dx, y) && self.free(x, y + dy)) {
                continue;
            }

            if self.free(x + dx, y + dy) {
                let target = usize::try_from(
                    (y + dy) * isize::try_from(self.width).unwrap_or(isize::MAX) + x + dx,
                )
                .expect("a free cell has a valid index");
                out.push((
                    target,
                    if diagonal {
                        core::f64::consts::SQRT_2
                    } else {
                        1.0
                    },
                ));
            }
        }
    }
}

/// The cost of `path` if every consecutive pair is an edge; `None` otherwise (or if empty).
#[must_use]
pub fn path_cost<G: Graph + ?Sized>(graph: &G, path: &[usize]) -> Option<f64> {
    if path.is_empty() {
        return None;
    }

    path.windows(2).map(|w| graph.edge_cost(w[0], w[1])).sum()
}

/// A shortest path by Dijkstra's algorithm, with its cost; `None` if `goal` is unreachable.
/// The reference for tests and the baseline for benches.
///
/// # Panics
///
/// Panics if `start` or `goal` is not a vertex of the graph.
#[must_use]
pub fn dijkstra<G: Graph + ?Sized>(
    graph: &G,
    start: usize,
    goal: usize,
) -> Option<(Vec<usize>, f64)> {
    /// A heap entry ordered by smallest distance first.
    #[derive(PartialEq)]
    struct Entry(f64, usize);
    impl Eq for Entry {}
    impl PartialOrd for Entry {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Entry {
        fn cmp(&self, other: &Self) -> Ordering {
            other.0.total_cmp(&self.0).then(other.1.cmp(&self.1))
        }
    }

    let n = graph.node_count();
    let mut distance = alloc::vec![f64::INFINITY; n];
    let mut parent = alloc::vec![usize::MAX; n];
    let mut heap = BinaryHeap::new();
    let mut edges = Vec::new();
    distance[start] = 0.0;
    heap.push(Entry(0.0, start));
    while let Some(Entry(d, v)) = heap.pop() {
        if v == goal {
            break;
        }
        if d > distance[v] {
            continue;
        }

        edges.clear();
        graph.neighbours(v, &mut edges);
        for &(w, c) in &edges {
            if d + c < distance[w] {
                distance[w] = d + c;
                parent[w] = v;
                heap.push(Entry(d + c, w));
            }
        }
    }

    if !distance[goal].is_finite() {
        return None;
    }

    let mut path = alloc::vec![goal];
    while *path.last().expect("non-empty") != start {
        path.push(parent[*path.last().expect("non-empty")]);
    }
    path.reverse();

    Some((path, distance[goal]))
}

/// A randomised depth-first search from `from` to `to` avoiding `forbidden` vertices,
/// expanding at most `budget` vertices; returns the path (both ends included).
fn random_dfs<G: Graph + ?Sized, R: Rng + ?Sized>(
    graph: &G,
    from: usize,
    to: usize,
    forbidden: &[bool],
    budget: usize,
    rng: &mut R,
) -> Option<Vec<usize>> {
    let mut visited = forbidden.to_vec();
    let mut stack: Vec<(usize, Vec<usize>)> = Vec::new();
    let mut edges = Vec::new();
    let mut path = alloc::vec![from];
    visited[from] = true;
    let mut expansions = 0;
    let frontier = |v: usize, visited: &[bool], rng: &mut R, edges: &mut Vec<(usize, f64)>| {
        edges.clear();
        graph.neighbours(v, edges);
        let mut next: Vec<usize> = edges
            .iter()
            .map(|e| e.0)
            .filter(|&w| w == to || !visited[w])
            .collect();
        shuffle(rng, &mut next);
        next
    };
    stack.push((from, frontier(from, &visited, rng, &mut edges)));
    while let Some((_, options)) = stack.last_mut() {
        let Some(next) = options.pop() else {
            stack.pop();
            path.pop();
            continue;
        };

        if next == to {
            path.push(to);
            return Some(path);
        }
        if visited[next] {
            continue;
        }

        expansions += 1;
        if expansions > budget {
            return None;
        }

        visited[next] = true;
        path.push(next);
        let options = frontier(next, &visited, rng, &mut edges);
        stack.push((next, options));
    }

    None
}

/// Random simple paths from `start` to `goal` by randomised depth-first search. If the goal
/// is unreachable the genome is just `[start]`, which [`PathLength`] scores `+∞`.
#[derive(Debug, Clone, Copy)]
pub struct RandomPath<'g, G: ?Sized> {
    /// The graph.
    pub graph: &'g G,
    /// The first vertex.
    pub start: usize,
    /// The last vertex.
    pub goal: usize,
}

impl<G: Graph + ?Sized> Initializer<Vec<usize>> for RandomPath<'_, G> {
    fn initialize<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Vec<usize> {
        let forbidden = alloc::vec![false; self.graph.node_count()];
        random_dfs(
            self.graph,
            self.start,
            self.goal,
            &forbidden,
            usize::MAX,
            rng,
        )
        .unwrap_or_else(|| alloc::vec![self.start])
    }
}

/// Crossover at a shared vertex: both parents are cut at a vertex they have in common and
/// the tails exchanged; loops are cut afterwards. Parents without a shared interior vertex
/// are returned unchanged.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CommonVertex;

impl Crossover<Vec<usize>> for CommonVertex {
    fn crossover<R: Rng + ?Sized>(
        &mut self,
        a: &Vec<usize>,
        b: &Vec<usize>,
        rng: &mut R,
    ) -> [Vec<usize>; 2] {
        let shared: Vec<(usize, usize)> = a
            .iter()
            .enumerate()
            .skip(1)
            .take(a.len().saturating_sub(2))
            .filter_map(|(i, v)| {
                b.iter()
                    .position(|w| w == v)
                    .filter(|&j| j > 0 && j + 1 < b.len())
                    .map(|j| (i, j))
            })
            .collect();
        if shared.is_empty() {
            return [a.clone(), b.clone()];
        }

        let (i, j) = shared[below(rng, shared.len())];
        let mut c1: Vec<usize> = a[..=i].iter().chain(&b[j + 1..]).copied().collect();
        let mut c2: Vec<usize> = b[..=j].iter().chain(&a[i + 1..]).copied().collect();
        remove_loops(&mut c1);
        remove_loops(&mut c2);

        [c1, c2]
    }
}

/// Replaces the stretch between two path vertices by a random detour (a bounded randomised
/// depth-first search that avoids the rest of the path).
#[derive(Debug, Clone, Copy)]
pub struct Reroute<'g, G: ?Sized> {
    /// The graph.
    pub graph: &'g G,
    /// The largest number of vertices one detour search may expand.
    pub max_expansions: usize,
}

impl<G: Graph + ?Sized> Mutation<Vec<usize>> for Reroute<'_, G> {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut Vec<usize>, rng: &mut R) {
        if genome.len() < 2 {
            return;
        }

        let (i, j) = two_sorted_indices(rng, genome.len());
        let mut forbidden = alloc::vec![false; self.graph.node_count()];
        for &v in genome[..i].iter().chain(&genome[j + 1..]) {
            forbidden[v] = true;
        }

        if let Some(detour) = random_dfs(
            self.graph,
            genome[i],
            genome[j],
            &forbidden,
            self.max_expansions,
            rng,
        ) {
            genome.splice(i..=j, detour);
        }
    }
}

/// Cuts every loop: when a vertex reappears, the stretch between its occurrences goes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RemoveLoops;

impl Repair<Vec<usize>> for RemoveLoops {
    fn repair<R: Rng + ?Sized>(&mut self, genome: &mut Vec<usize>, _rng: &mut R) {
        remove_loops(genome);
    }
}

/// Removes loops in place, keeping the first occurrence of each vertex.
fn remove_loops(path: &mut Vec<usize>) {
    let mut out: Vec<usize> = Vec::with_capacity(path.len());
    for &v in path.iter() {
        if let Some(k) = out.iter().position(|&w| w == v) {
            out.truncate(k + 1);
        } else {
            out.push(v);
        }
    }

    *path = out;
}

/// Path length as an objective: the sum of edge costs, `+∞` for paths that are not walks in
/// the graph or do not run from `start` to `goal`.
#[derive(Debug, Clone, Copy)]
pub struct PathLength<'g, G: ?Sized> {
    /// The graph.
    pub graph: &'g G,
    /// The required first vertex.
    pub start: usize,
    /// The required last vertex.
    pub goal: usize,
}

impl<G: Graph + ?Sized> Objective<Vec<usize>> for PathLength<'_, G> {
    fn evaluate(&mut self, genome: &Vec<usize>) -> f64 {
        if genome.first() != Some(&self.start) || genome.last() != Some(&self.goal) {
            return f64::INFINITY;
        }

        path_cost(self.graph, genome).unwrap_or(f64::INFINITY)
    }
}

/// Parameters of [`solve`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// The GA population size.
    pub population_size: usize,
    /// The largest number of vertices a reroute may expand.
    pub max_expansions: usize,
}

impl Default for Config {
    /// 50 paths, reroutes of at most 64 expansions.
    fn default() -> Self {
        Self {
            population_size: 50,
            max_expansions: 64,
        }
    }
}

/// Searches a short path with a generational GA (elitism two, tournament three, shared-vertex
/// crossover, reroute mutation, loop removal); returns the best path and its cost, or `None`
/// if no evaluated path reached the goal.
pub fn solve<G, B, R>(
    graph: &G,
    start: usize,
    goal: usize,
    config: &Config,
    budget: &mut B,
    rng: &mut R,
) -> Option<(Vec<usize>, f64)>
where
    G: Graph + ?Sized,
    B: Budget<Stats> + ?Sized,
    R: Rng + ?Sized,
{
    let mut ga = Ga::new(
        GaConfig {
            population_size: config.population_size,
            elitism: 2,
            crossover_rate: 0.7,
            mutation_rate: 0.8,
            scheme: Scheme::Generational,
        },
        Operators {
            init: RandomPath { graph, start, goal },
            selection: Tournament { size: 3 },
            crossover: CommonVertex,
            mutation: Reroute {
                graph,
                max_expansions: config.max_expansions,
            },
            repair: RemoveLoops,
        },
    );
    let mut objective = PathLength { graph, start, goal };
    let outcome = ga.run(&mut objective, budget, &mut (), rng);

    outcome
        .best
        .filter(|b| b.cost.is_finite())
        .map(|b| (b.genome, b.cost))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Limits, SplitMix64};

    /// A grid with a few walls, 4- or 8-connected.
    fn maze(diagonal: bool) -> Grid {
        let mut grid = Grid::open(16, 12, diagonal);
        for y in 0..9 {
            grid.block(4, y);
        }
        for y in 3..12 {
            grid.block(9, y);
        }
        for x in 11..15 {
            grid.block(x, 6);
        }

        grid
    }

    /// The GA finds an optimal path on 4- and 8-connected mazes, and every path it returns
    /// is a valid simple walk.
    #[test]
    fn matches_dijkstra() {
        for diagonal in [false, true] {
            let grid = maze(diagonal);
            let (start, goal) = (grid.node(0, 0), grid.node(15, 11));
            let (_, optimum) = dijkstra(&grid, start, goal).expect("reachable");
            let (path, cost) = solve(
                &grid,
                start,
                goal,
                &Config::default(),
                &mut Limits::evaluations(30_000),
                &mut SplitMix64::new(2),
            )
            .expect("found");

            assert_eq!(path_cost(&grid, &path), Some(cost));
            assert!(
                (cost - optimum).abs() < 1e-9,
                "diagonal {diagonal}: {cost} vs {optimum}"
            );
            let mut sorted = path.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), path.len(), "the path is simple");
        }
    }

    /// Operators keep paths valid from start to goal.
    #[test]
    fn operators_keep_paths_valid() {
        let grid = maze(true);
        let (start, goal) = (grid.node(0, 0), grid.node(15, 11));
        let mut rng = SplitMix64::new(5);
        let mut init = RandomPath {
            graph: &grid,
            start,
            goal,
        };
        let mut reroute = Reroute {
            graph: &grid,
            max_expansions: 50,
        };
        for _ in 0..100 {
            let a = init.initialize(&mut rng);
            let b = init.initialize(&mut rng);
            for mut child in CommonVertex.crossover(&a, &b, &mut rng) {
                reroute.mutate(&mut child, &mut rng);
                assert_eq!((child[0], child[child.len() - 1]), (start, goal));
                assert!(path_cost(&grid, &child).is_some());
            }
        }
    }

    /// An unreachable goal yields no path.
    #[test]
    fn unreachable_goal() {
        let mut grid = Grid::open(5, 5, false);
        for y in 0..5 {
            grid.block(2, y);
        }

        assert!(dijkstra(&grid, 0, 4).is_none());
        assert!(
            solve(
                &grid,
                0,
                4,
                &Config::default(),
                &mut Limits::evaluations(500),
                &mut SplitMix64::new(1)
            )
            .is_none()
        );
    }
}
