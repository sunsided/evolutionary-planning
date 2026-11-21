//! A multi-robot intersection benchmark after Parque (2024): every robot crosses a four-way
//! intersection on a lattice roadmap built around its reference path, the lattice of robot
//! `i` is parameterised by its base `b_i` and height `h_i`, and the cost of a parameter
//! vector `x = (b_1, h_1, …, b_N, h_N)` is the total length of the coordinated, collision-free
//! paths (equation (1) of the paper), so any gradient-free optimiser (RADES, DE, CMA-ES) can
//! search the roadmap configurations.
//!
//! The paper coordinates the robots with dRRT* in the composite roadmap. This benchmark uses
//! prioritised space-time search instead: robots plan in index order on their own roadmaps,
//! each avoiding the already planned ones (positions at every time step and edge midpoint
//! at least `safety` apart; a robot leaves the scene at its goal). One roadmap edge takes one
//! time step and waiting in place is allowed. A robot that finds no plan adds a penalty of
//! `1000` plus its straight-line distance, so infeasible configurations still compare.
//!
//! ```
//! use evolutionary_planning::de::rades::{Config, Rades};
//! use evolutionary_planning::problems::intersection::Intersection;
//! use evolutionary_planning::{Limits, SplitMix64};
//!
//! let mut scene = Intersection::crossing(4);
//! let mut rades = Rades::new(Config::default(), scene.bounds());
//! // The paper's budget: 300 function evaluations.
//! let outcome = rades.run(&mut scene, &mut Limits::evaluations(300), &mut (), &mut SplitMix64::new(1));
//! let plan = scene.plan(outcome.best_genome().expect("evaluated")).expect("feasible");
//! assert_eq!(scene.collisions(&plan), 0);
//! ```

use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::cmp::Ordering;

use crate::Objective;
use crate::operators::real::Bounds;

/// A point in the plane.
pub type Point = (f64, f64);

/// One robot: where it enters and where it leaves, and its reference path (a polyline from
/// start to goal through the intersection).
#[derive(Debug, Clone, PartialEq)]
pub struct Robot {
    /// The reference polyline; its first point is the start, its last the goal.
    pub reference: Vec<Point>,
}

/// An axis-aligned rectangular obstacle `[x0, x1] × [y0, y1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Block {
    /// The lower-left corner.
    pub min: Point,
    /// The upper-right corner.
    pub max: Point,
}

impl Block {
    /// Whether the segment `a → b` touches the block (sampled densely; the blocks are large
    /// compared to the sampling step).
    fn hits(&self, from: Point, to: Point) -> bool {
        (0..=32).any(|k| {
            let t = f64::from(k) / 32.0;
            let (x, y) = (from.0 + t * (to.0 - from.0), from.1 + t * (to.1 - from.1));
            (self.min.0..=self.max.0).contains(&x) && (self.min.1..=self.max.1).contains(&y)
        })
    }
}

/// The intersection scene.
#[derive(Debug, Clone, PartialEq)]
pub struct Intersection {
    /// The robots, in planning priority order.
    pub robots: Vec<Robot>,
    /// The buildings at the corners.
    pub blocks: Vec<Block>,
    /// The minimum distance between two robots.
    pub safety: f64,
    /// The bounds of each `(b_i, h_i)` pair: the paper's `b ∈ [1, 6]`, `h ∈ [0.5, 5]`.
    pub parameter_bounds: [(f64, f64); 2],
}

/// A roadmap: vertex positions and forward edges.
struct Roadmap {
    /// The vertex positions; vertex `0` is the start, vertex `1` the goal.
    points: Vec<Point>,
    /// The out-edges of every vertex: `(target, length)`.
    edges: Vec<Vec<(usize, f64)>>,
}

impl Intersection {
    /// A four-way crossing of roads 8 wide with buildings at the corners and up to eight
    /// robots driving straight across on the right-hand lanes: robots 0–3 come from the west,
    /// east, south and north; robots 4–7 follow them on the same lanes, 8 units behind.
    ///
    /// # Panics
    ///
    /// Panics if `robots` is not in `1..=8`.
    #[must_use]
    pub fn crossing(robots: usize) -> Self {
        assert!(
            (1..=8).contains(&robots),
            "the crossing scene has one to eight robots"
        );

        let lanes: [(Point, Point); 4] = [
            ((-18.0, -2.0), (18.0, -2.0)),
            ((18.0, 2.0), (-18.0, 2.0)),
            ((2.0, -18.0), (2.0, 18.0)),
            ((-2.0, 18.0), (-2.0, -18.0)),
        ];
        let robots = (0..robots)
            .map(|i| {
                let (start, goal) = lanes[i % 4];
                let behind = if i < 4 { 0.0 } else { 8.0 };
                let direction = unit((goal.0 - start.0, goal.1 - start.1));
                let shift = |p: Point, d: f64| (p.0 - d * direction.0, p.1 - d * direction.1);
                Robot {
                    reference: alloc::vec![shift(start, behind), shift(goal, -behind)],
                }
            })
            .collect();
        let corner = |sx: f64, sy: f64| Block {
            min: (
                if sx > 0.0 { 4.0 } else { -40.0 },
                if sy > 0.0 { 4.0 } else { -40.0 },
            ),
            max: (
                if sx > 0.0 { 40.0 } else { -4.0 },
                if sy > 0.0 { 40.0 } else { -4.0 },
            ),
        };

        Self {
            robots,
            blocks: alloc::vec![
                corner(1.0, 1.0),
                corner(-1.0, 1.0),
                corner(1.0, -1.0),
                corner(-1.0, -1.0)
            ],
            safety: 1.5,
            parameter_bounds: [(1.0, 6.0), (0.5, 5.0)],
        }
    }

    /// The search box of the parameter vector `(b_1, h_1, …, b_N, h_N)`.
    #[must_use]
    pub fn bounds(&self) -> Bounds {
        let [(b0, b1), (h0, h1)] = self.parameter_bounds;
        let lower = self.robots.iter().flat_map(|_| [b0, h0]).collect();
        let upper = self.robots.iter().flat_map(|_| [b1, h1]).collect();

        Bounds::new(lower, upper)
    }

    /// The coordinated trajectories (one position per time step, until each robot's goal)
    /// for parameters `x`, or `None` if some robot found no collision-free plan.
    #[must_use]
    pub fn plan(&self, x: &[f64]) -> Option<Vec<Vec<Point>>> {
        let mut planned: Vec<Vec<Point>> = Vec::with_capacity(self.robots.len());
        for (i, robot) in self.robots.iter().enumerate() {
            let roadmap = self.roadmap(
                robot,
                self.parameter(x, 2 * i, 0),
                self.parameter(x, 2 * i + 1, 1),
            );
            planned.push(self.space_time(&roadmap, &planned)?);
        }

        Some(planned)
    }

    /// The number of time steps (and midpoints) at which two robots are closer than `safety`.
    #[must_use]
    pub fn collisions(&self, trajectories: &[Vec<Point>]) -> usize {
        let horizon = trajectories.iter().map(Vec::len).max().unwrap_or(0);
        let mut count = 0;
        for t in 0..horizon {
            for (i, a) in trajectories.iter().enumerate() {
                for b in &trajectories[i + 1..] {
                    count += usize::from(conflict(a, b, t, self.safety));
                }
            }
        }

        count
    }

    /// Reads parameter `index` of `x`, clamped to its bounds (`kind` 0: base, 1: height).
    fn parameter(&self, x: &[f64], index: usize, kind: usize) -> f64 {
        let (low, high) = self.parameter_bounds[kind];

        x.get(index).copied().unwrap_or(low).clamp(low, high)
    }

    /// The lattice roadmap of `robot` with base `base` and height `height`: points on the
    /// reference path every `base` units and offset by `±height`, with the forward edges of
    /// the paper's equation (4) that avoid the buildings.
    fn roadmap(&self, robot: &Robot, base: f64, height: f64) -> Roadmap {
        let middle = resample(&robot.reference, base);
        let r = middle.len();
        let mut points = Vec::with_capacity(3 * r);
        // Vertex 0 is the start and 1 the goal; then the interior middle points and both rows.
        points.push(middle[0]);
        points.push(middle[r - 1]);
        let mut index = alloc::vec![[usize::MAX; 3]; r];
        index[0][0] = 0;
        index[r - 1][0] = 1;
        for j in 0..r {
            let normal = normal_at(&middle, j);
            for (row, offset) in [(0, 0.0), (1, height), (2, -height)] {
                if index[j][row] != usize::MAX || ((j == 0 || j == r - 1) && row != 0) {
                    continue;
                }
                index[j][row] = points.len();
                points.push((
                    middle[j].0 + offset * normal.0,
                    middle[j].1 + offset * normal.1,
                ));
            }
        }

        let mut edges = alloc::vec![Vec::new(); points.len()];
        let mut connect = |from: usize, to: usize, points: &[Point]| {
            if from == usize::MAX || to == usize::MAX {
                return;
            }
            let (a, b) = (points[from], points[to]);
            if !self.blocks.iter().any(|block| block.hits(a, b)) {
                edges[from].push((to, distance(a, b)));
            }
        };
        for j in 0..r {
            let [q, p, s] = index[j];
            connect(q, p, &points);
            connect(q, s, &points);
            if j + 1 < r {
                let [q1, p1, s1] = index[j + 1];
                for (from, to) in [
                    (q, q1),
                    (p, q1),
                    (s, q1),
                    (p, p1),
                    (s, s1),
                    (q, p1),
                    (q, s1),
                ] {
                    connect(from, to, &points);
                }
            }
        }

        Roadmap { points, edges }
    }

    /// Prioritised space-time Dijkstra on `roadmap` avoiding the `planned` trajectories;
    /// minimises path length, then time (waiting costs `1e-3` per step).
    fn space_time(&self, roadmap: &Roadmap, planned: &[Vec<Point>]) -> Option<Vec<Point>> {
        /// A search entry ordered by smallest cost first.
        #[derive(PartialEq)]
        struct Entry(f64, usize, usize);
        impl Eq for Entry {}
        impl PartialOrd for Entry {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for Entry {
            fn cmp(&self, other: &Self) -> Ordering {
                other
                    .0
                    .total_cmp(&self.0)
                    .then(other.2.cmp(&self.2))
                    .then(other.1.cmp(&self.1))
            }
        }

        let horizon = 2 * roadmap.points.len() + planned.iter().map(Vec::len).max().unwrap_or(0);
        let n = roadmap.points.len();
        let mut best = alloc::vec![f64::INFINITY; n * (horizon + 1)];
        let mut parent = alloc::vec![usize::MAX; n * (horizon + 1)];
        let mut heap = BinaryHeap::new();
        if !self.free(roadmap.points[0], roadmap.points[0], 0, planned) {
            return None;
        }

        best[0] = 0.0;
        heap.push(Entry(0.0, 0, 0));
        while let Some(Entry(cost, v, t)) = heap.pop() {
            if v == 1 {
                return Some(trace(&roadmap.points, &parent, n, v, t));
            }
            if cost > best[t * n + v] || t == horizon {
                continue;
            }

            let wait = core::iter::once((v, 1e-3));
            for (w, length) in wait.chain(roadmap.edges[v].iter().copied()) {
                let next = (t + 1) * n + w;
                let total = cost + length;
                if total < best[next]
                    && self.free(roadmap.points[v], roadmap.points[w], t + 1, planned)
                {
                    best[next] = total;
                    parent[next] = t * n + v;
                    heap.push(Entry(total, w, t + 1));
                }
            }
        }

        None
    }

    /// Whether moving `from → to` arriving at time `t` keeps the safety distance to every planned
    /// robot still in the scene (at `t` and at the midpoint of the move).
    fn free(&self, from: Point, to: Point, t: usize, planned: &[Vec<Point>]) -> bool {
        let middle = midpoint(from, to);
        planned.iter().all(|other| {
            let at = |time: usize| other.get(time).copied();
            let clear =
                |p: Point, q: Option<Point>| q.is_none_or(|q| distance(p, q) >= self.safety);
            let previous = t.checked_sub(1).and_then(at);
            let midpoint = match (previous, at(t)) {
                (Some(before), Some(now)) => Some(midpoint(before, now)),
                _ => None,
            };
            clear(to, at(t)) && clear(middle, midpoint)
        })
    }
}

impl Objective<Vec<f64>> for Intersection {
    /// The total path length of the coordinated plan; failed robots add `1000` plus their
    /// straight-line distance.
    fn evaluate(&mut self, x: &Vec<f64>) -> f64 {
        let mut planned: Vec<Vec<Point>> = Vec::with_capacity(self.robots.len());
        let mut cost = 0.0;
        for (i, robot) in self.robots.iter().enumerate() {
            let roadmap = self.roadmap(
                robot,
                self.parameter(x, 2 * i, 0),
                self.parameter(x, 2 * i + 1, 1),
            );
            if let Some(path) = self.space_time(&roadmap, &planned) {
                cost += path.windows(2).map(|w| distance(w[0], w[1])).sum::<f64>();
                planned.push(path);
            } else {
                let (start, goal) = (
                    robot.reference[0],
                    robot.reference[robot.reference.len() - 1],
                );
                cost += 1000.0 + distance(start, goal);
            }
        }

        cost
    }
}

/// Whether trajectories `a` and `b` are closer than `safety` at time `t` or at the midpoint of
/// the step into `t`.
fn conflict(a: &[Point], b: &[Point], t: usize, safety: f64) -> bool {
    let (Some(&pa), Some(&pb)) = (a.get(t), b.get(t)) else {
        return false;
    };
    if distance(pa, pb) < safety {
        return true;
    }

    match (
        t.checked_sub(1).and_then(|s| a.get(s)),
        t.checked_sub(1).and_then(|s| b.get(s)),
    ) {
        (Some(&qa), Some(&qb)) => distance(midpoint(pa, qa), midpoint(pb, qb)) < safety,
        _ => false,
    }
}

/// The midpoint of two points.
fn midpoint(a: Point, b: Point) -> Point {
    (f64::midpoint(a.0, b.0), f64::midpoint(a.1, b.1))
}

/// Follows the parent pointers back from `(v, t)`.
fn trace(points: &[Point], parent: &[usize], n: usize, v: usize, t: usize) -> Vec<Point> {
    let mut states = alloc::vec![t * n + v];
    while parent[*states.last().expect("non-empty")] != usize::MAX {
        states.push(parent[*states.last().expect("non-empty")]);
    }
    states.reverse();

    states.into_iter().map(|s| points[s % n]).collect()
}

/// Points along `polyline` every `step` units of arc length, both ends included (the last
/// interval may be shorter): `R = 1 + ⌈L / b⌉` points, the paper's (2) rounded up.
fn resample(polyline: &[Point], step: f64) -> Vec<Point> {
    let mut points = alloc::vec![polyline[0]];
    let mut carried = 0.0;
    for w in polyline.windows(2) {
        let length = distance(w[0], w[1]);
        let mut s = step - carried;
        while s < length - 1e-9 {
            let t = s / length;
            points.push((
                w[0].0 + t * (w[1].0 - w[0].0),
                w[0].1 + t * (w[1].1 - w[0].1),
            ));
            s += step;
        }
        carried = length - (s - step);
    }

    points.push(polyline[polyline.len() - 1]);
    points
}

/// The left normal of the path at point `j` (from the neighbouring points).
fn normal_at(points: &[Point], j: usize) -> Point {
    let a = points[j.saturating_sub(1)];
    let b = points[(j + 1).min(points.len() - 1)];
    let (dx, dy) = unit((b.0 - a.0, b.1 - a.1));

    (-dy, dx)
}

/// The unit vector of `v` (zero stays zero).
fn unit(v: Point) -> Point {
    let length = libm::hypot(v.0, v.1);
    if length > 0.0 {
        (v.0 / length, v.1 / length)
    } else {
        (0.0, 0.0)
    }
}

/// The Euclidean distance.
fn distance(a: Point, b: Point) -> f64 {
    libm::hypot(a.0 - b.0, a.1 - b.1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cmaes::{CmaEs, Config as CmaConfig};
    use crate::de::rades::{Config, Rades};
    use crate::de::{Config as DeConfig, De};
    use crate::{Limits, SplitMix64};

    /// Every optimiser finds a collision-free coordinated plan for four robots (RADES also
    /// for eight), and the reported cost equals the planned lengths.
    #[test]
    fn coordinated_plans_are_collision_free() {
        for robots in [4, 8] {
            let mut scene = Intersection::crossing(robots);
            let bounds = scene.bounds();
            let mut rng = SplitMix64::new(3);
            let budget = if robots == 4 { 300 } else { 100 };
            let mut outcomes = alloc::vec![Rades::new(Config::default(), bounds.clone()).run(
                &mut scene,
                &mut Limits::evaluations(budget),
                &mut (),
                &mut rng
            )];
            if robots == 4 {
                outcomes.push(
                    De::new(
                        DeConfig {
                            population_size: 10,
                            ..DeConfig::default()
                        },
                        bounds.clone(),
                    )
                    .run(
                        &mut scene,
                        &mut Limits::evaluations(300),
                        &mut (),
                        &mut rng,
                    ),
                );
                outcomes.push(
                    CmaEs::new(
                        bounds.center(),
                        1.0,
                        CmaConfig {
                            bounds: Some(bounds.clone()),
                            ..CmaConfig::default()
                        },
                    )
                    .minimize(
                        &mut scene,
                        &mut Limits::evaluations(300),
                        &mut (),
                        &mut rng,
                    ),
                );
            }
            for outcome in outcomes {
                let x = outcome.best_genome().expect("evaluated");
                let plan = scene.plan(x).expect("feasible");
                let length: f64 = plan
                    .iter()
                    .map(|p| p.windows(2).map(|w| distance(w[0], w[1])).sum::<f64>())
                    .sum();
                assert_eq!(scene.collisions(&plan), 0, "{robots} robots");
                assert!((length - outcome.best_cost()).abs() < 1e-9);
                assert!(
                    plan.iter()
                        .zip(&scene.robots)
                        .all(|(p, r)| p[p.len() - 1] == r.reference[r.reference.len() - 1])
                );
            }
        }
    }

    /// Lattice resampling and roadmap structure.
    #[test]
    fn lattice_structure() {
        let points = resample(&[(0.0, 0.0), (10.0, 0.0)], 3.0);
        assert_eq!(
            points,
            [(0.0, 0.0), (3.0, 0.0), (6.0, 0.0), (9.0, 0.0), (10.0, 0.0)]
        );

        let scene = Intersection::crossing(1);
        let roadmap = scene.roadmap(&scene.robots[0], 4.0, 1.0);
        // The straight middle row is always available: the start reaches the goal.
        assert!(roadmap.edges[0].iter().any(|&(w, _)| w >= 2));
        assert!(roadmap.edges.iter().flatten().all(|&(_, l)| l > 0.0));
    }
}
