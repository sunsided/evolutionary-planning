//! Operators for permutations `Vec<usize>` of `0..n`: swap, insertion, inversion and scramble
//! mutation; partially mapped (PMX), order (OX), cycle (CX) and edge recombination (ERX)
//! crossover. Every operator maps permutations to permutations (property-tested).
//!
//! Jakob (2021) recommends operators that change gene order, including segment inversion and
//! order-preserving crossover, for combinatorial tasks; these are they.

use alloc::vec::Vec;

use rand_core::Rng;

use super::{Crossover, Initializer, Mutation};
use crate::rng::{below, shuffle, two_sorted_indices};

/// Uniform random permutations of `0..n`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RandomPermutation {
    /// The number of elements.
    pub n: usize,
}

impl Initializer<Vec<usize>> for RandomPermutation {
    fn initialize<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Vec<usize> {
        let mut p: Vec<usize> = (0..self.n).collect();
        shuffle(rng, &mut p);

        p
    }
}

/// Swaps two distinct positions. Works for any `Vec<T>`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Swap;

impl<T> Mutation<Vec<T>> for Swap {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut Vec<T>, rng: &mut R) {
        if genome.len() >= 2 {
            let (i, j) = two_sorted_indices(rng, genome.len());
            genome.swap(i, j);
        }
    }
}

/// Removes one element and reinserts it at another position. Works for any `Vec<T>`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Insertion;

impl<T> Mutation<Vec<T>> for Insertion {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut Vec<T>, rng: &mut R) {
        if genome.len() >= 2 {
            let from = below(rng, genome.len());
            let item = genome.remove(from);
            let to = below(rng, genome.len() + 1);
            genome.insert(to, item);
        }
    }
}

/// Reverses a random segment (2-opt move for tours). Works for any `Vec<T>`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Inversion;

impl<T> Mutation<Vec<T>> for Inversion {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut Vec<T>, rng: &mut R) {
        if genome.len() >= 2 {
            let (i, j) = two_sorted_indices(rng, genome.len());
            genome[i..=j].reverse();
        }
    }
}

/// Shuffles a random segment. Works for any `Vec<T>`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Scramble;

impl<T> Mutation<Vec<T>> for Scramble {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut Vec<T>, rng: &mut R) {
        if genome.len() >= 2 {
            let (i, j) = two_sorted_indices(rng, genome.len());
            shuffle(rng, &mut genome[i..=j]);
        }
    }
}

/// Partially mapped crossover (Goldberg and Lingle 1985): each child takes a segment of one
/// parent and fills the rest from the other, resolving conflicts through the segment's
/// mapping.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Pmx;

impl Crossover<Vec<usize>> for Pmx {
    fn crossover<R: Rng + ?Sized>(
        &mut self,
        a: &Vec<usize>,
        b: &Vec<usize>,
        rng: &mut R,
    ) -> [Vec<usize>; 2] {
        if a.len() < 2 {
            return [a.clone(), b.clone()];
        }

        let (i, j) = two_sorted_indices(rng, a.len());

        [pmx_child(a, b, i, j), pmx_child(b, a, i, j)]
    }
}

/// The PMX child with `donor[i..=j]` and the other positions from `filler`.
fn pmx_child(donor: &[usize], filler: &[usize], i: usize, j: usize) -> Vec<usize> {
    let n = donor.len();
    let mut child = alloc::vec![usize::MAX; n];
    let mut position_in_donor = alloc::vec![usize::MAX; n];
    for (k, &v) in donor.iter().enumerate() {
        position_in_donor[v] = k;
    }

    child[i..=j].copy_from_slice(&donor[i..=j]);
    for k in (0..i).chain(j + 1..n) {
        let mut value = filler[k];
        // Follow the mapping donor[p] -> filler[p] until the value is outside the segment.
        while (i..=j).contains(&position_in_donor[value]) {
            value = filler[position_in_donor[value]];
        }

        child[k] = value;
    }

    child
}

/// Order crossover OX1 (Davis 1985): each child keeps a segment of one parent and fills the
/// remaining positions, starting after the segment, with the other parent's elements in their
/// relative order.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Ox;

impl Crossover<Vec<usize>> for Ox {
    fn crossover<R: Rng + ?Sized>(
        &mut self,
        a: &Vec<usize>,
        b: &Vec<usize>,
        rng: &mut R,
    ) -> [Vec<usize>; 2] {
        if a.len() < 2 {
            return [a.clone(), b.clone()];
        }

        let (i, j) = two_sorted_indices(rng, a.len());

        [ox_child(a, b, i, j), ox_child(b, a, i, j)]
    }
}

/// The OX child with `donor[i..=j]` and the rest from `filler` in cyclic order after `j`.
fn ox_child(donor: &[usize], filler: &[usize], i: usize, j: usize) -> Vec<usize> {
    let n = donor.len();
    let mut used = alloc::vec![false; n];
    let mut child = alloc::vec![usize::MAX; n];
    for k in i..=j {
        child[k] = donor[k];
        used[donor[k]] = true;
    }

    let mut write = (j + 1) % n;
    for offset in 0..n {
        let value = filler[(j + 1 + offset) % n];
        if !used[value] {
            child[write] = value;
            used[value] = true;
            write = (write + 1) % n;
        }
    }

    child
}

/// Cycle crossover (Oliver, Smith and Holland 1987): positions are partitioned into cycles;
/// alternate cycles come from alternate parents, so every element keeps a parent's absolute
/// position.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Cx;

impl Crossover<Vec<usize>> for Cx {
    fn crossover<R: Rng + ?Sized>(
        &mut self,
        a: &Vec<usize>,
        b: &Vec<usize>,
        _rng: &mut R,
    ) -> [Vec<usize>; 2] {
        let n = a.len();
        let mut position_in_a = alloc::vec![0; n];
        for (k, &v) in a.iter().enumerate() {
            position_in_a[v] = k;
        }

        let mut c1 = b.clone();
        let mut c2 = a.clone();
        let mut visited = alloc::vec![false; n];
        let mut take_from_a = true;
        for start in 0..n {
            if visited[start] {
                continue;
            }

            let mut k = start;
            while !visited[k] {
                visited[k] = true;
                if take_from_a {
                    c1[k] = a[k];
                    c2[k] = b[k];
                }
                k = position_in_a[b[k]];
            }

            take_from_a = !take_from_a;
        }

        [c1, c2]
    }
}

/// Edge recombination crossover (Whitley, Starkweather and Fuquay 1989) for tours: children
/// are built from the union of both parents' adjacency lists, always continuing with the
/// neighbour that has the fewest remaining neighbours. Preserves edges, which matter more than
/// positions in routing problems.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EdgeRecombination;

impl Crossover<Vec<usize>> for EdgeRecombination {
    fn crossover<R: Rng + ?Sized>(
        &mut self,
        a: &Vec<usize>,
        b: &Vec<usize>,
        rng: &mut R,
    ) -> [Vec<usize>; 2] {
        [erx_child(a, b, rng), erx_child(b, a, rng)]
    }
}

/// One edge recombination child, starting from `first[0]`.
fn erx_child<R: Rng + ?Sized>(first: &[usize], second: &[usize], rng: &mut R) -> Vec<usize> {
    let n = first.len();
    if n < 2 {
        return first.to_vec();
    }

    let mut edges: Vec<Vec<usize>> = alloc::vec![Vec::with_capacity(4); n];
    for tour in [first, second] {
        for k in 0..n {
            let (v, next) = (tour[k], tour[(k + 1) % n]);
            add_edge(&mut edges, v, next);
            add_edge(&mut edges, next, v);
        }
    }

    let mut used = alloc::vec![false; n];
    let mut child = Vec::with_capacity(n);
    let mut current = first[0];
    loop {
        child.push(current);
        used[current] = true;
        for list in &mut edges {
            list.retain(|&v| v != current);
        }

        if child.len() == n {
            return child;
        }

        current = next_erx_vertex(&edges, &used, current, rng);
    }
}

/// Adds the undirected adjacency `v -> w` once.
fn add_edge(edges: &mut [Vec<usize>], v: usize, w: usize) {
    if !edges[v].contains(&w) {
        edges[v].push(w);
    }
}

/// The neighbour of `current` with the fewest remaining edges (ties uniformly), or a uniform
/// unused vertex if `current` has no neighbours left.
fn next_erx_vertex<R: Rng + ?Sized>(
    edges: &[Vec<usize>],
    used: &[bool],
    current: usize,
    rng: &mut R,
) -> usize {
    let neighbours = &edges[current];
    if neighbours.is_empty() {
        let unused: Vec<usize> = (0..used.len()).filter(|&v| !used[v]).collect();
        return unused[below(rng, unused.len())];
    }

    let fewest = neighbours
        .iter()
        .map(|&v| edges[v].len())
        .min()
        .unwrap_or(0);
    let candidates: Vec<usize> = neighbours
        .iter()
        .copied()
        .filter(|&v| edges[v].len() == fewest)
        .collect();

    candidates[below(rng, candidates.len())]
}

/// Returns `true` if `p` is a permutation of `0..p.len()`.
#[must_use]
pub fn is_permutation(p: &[usize]) -> bool {
    let mut seen = alloc::vec![false; p.len()];
    p.iter()
        .all(|&v| v < seen.len() && !core::mem::replace(&mut seen[v], true))
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::SplitMix64;

    proptest! {
        /// Every operator maps permutations to permutations.
        #[test]
        fn operators_keep_permutations(n in 1usize..30, seed in any::<u64>()) {
            let mut rng = SplitMix64::new(seed);
            let mut init = RandomPermutation { n };
            let a = init.initialize(&mut rng);
            let b = init.initialize(&mut rng);
            for child in Pmx.crossover(&a, &b, &mut rng)
                .into_iter()
                .chain(Ox.crossover(&a, &b, &mut rng))
                .chain(Cx.crossover(&a, &b, &mut rng))
                .chain(EdgeRecombination.crossover(&a, &b, &mut rng))
            {
                prop_assert!(is_permutation(&child), "{:?}", child);
            }

            let mut m = a.clone();
            Swap.mutate(&mut m, &mut rng);
            Insertion.mutate(&mut m, &mut rng);
            Inversion.mutate(&mut m, &mut rng);
            Scramble.mutate(&mut m, &mut rng);
            prop_assert!(is_permutation(&m));
        }
    }

    /// The textbook PMX example (Eiben and Smith, Fig. 4.10).
    #[test]
    fn pmx_textbook_example() {
        let a = [0, 1, 2, 3, 4, 5, 6, 7, 8];
        // 1 2 3 4 5 6 7 8 9 and 9 3 7 8 2 6 5 1 4 with segment 4..=7 give 9 3 2 4 5 6 7 1 8.
        let b = [8, 2, 6, 7, 1, 5, 4, 0, 3];
        assert_eq!(pmx_child(&a, &b, 3, 6), [8, 2, 1, 3, 4, 5, 6, 0, 7]);
    }

    /// Cycle crossover keeps every element at a parent's position.
    #[test]
    fn cycle_crossover_keeps_positions() {
        let a = alloc::vec![0, 1, 2, 3, 4, 5, 6, 7, 8];
        let b = alloc::vec![8, 2, 5, 6, 3, 1, 7, 4, 0];
        let [c1, c2] = Cx.crossover(&a, &b, &mut SplitMix64::new(0));
        for k in 0..9 {
            assert!(c1[k] == a[k] || c1[k] == b[k]);
            assert!(c2[k] == a[k] || c2[k] == b[k]);
        }

        assert!(is_permutation(&c1) && is_permutation(&c2));
    }
}
