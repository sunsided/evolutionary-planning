//! GLEAM action chains (Blume and Jakob; Jakob 2021): variable-length chromosomes of typed,
//! parameterised actions, with gene-level and chain-level mutations and crossovers that
//! accept parents of different lengths.
//!
//! Two layers:
//!
//! - **Chain level**, generic over any `Vec<T>`: shifting a gene or a segment, inverting a
//!   segment, inserting, deleting, duplicating or replacing genes and segments, and
//!   cut-and-splice, n-point and segment-exchange crossover. A new gene comes from any
//!   [`Initializer<T>`], so a SAS+ plan `Vec<OpId>` uses these unchanged.
//! - **Gene level**, on [`Action`]s described by an [`ActionModel`]: each action type has its
//!   own integer or real parameters with bounds that the mutations respect (Jakob's rule
//!   that explicit restrictions belong in the operators).
//!
//! GLEAM applies one operator per offspring, drawn from a weighted operator mix; build such a
//! mix with [`mix`](super::mix).
//!
//! ```
//! use evolutionary_planning::operators::chain::{Action, ActionKind, ActionModel, ParamSpec, RandomChain};
//! use evolutionary_planning::operators::Initializer;
//! use evolutionary_planning::SplitMix64;
//!
//! // Two action types: "move" with a real distance and "turn" with an integer angle.
//! let model = ActionModel::new(vec![
//!     ActionKind::new(vec![ParamSpec::real(0.0, 10.0)]),
//!     ActionKind::new(vec![ParamSpec::integer(-180.0, 180.0)]),
//! ]);
//! let mut init = RandomChain { genes: model.clone(), min_len: 2, max_len: 6 };
//! let chain: Vec<Action> = init.initialize(&mut SplitMix64::new(1));
//! assert!((2..=6).contains(&chain.len()));
//! assert!(chain.iter().all(|action| model.is_valid(action)));
//! ```

use alloc::vec::Vec;

use rand_core::Rng;

use super::{Crossover, Initializer, Mutation};
use crate::operators::sequence::splice;
use crate::rng::{below, uniform, unit_f64};

/// The bounds and kind of one action parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParamSpec {
    /// The smallest value, finite.
    pub low: f64,
    /// The largest value, finite, `>= low`.
    pub high: f64,
    /// Whether the parameter takes integer values only (stored as whole `f64`s).
    pub integer: bool,
}

impl ParamSpec {
    /// A real parameter in `[low, high]`.
    #[must_use]
    pub const fn real(low: f64, high: f64) -> Self {
        Self {
            low,
            high,
            integer: false,
        }
    }

    /// An integer parameter in `[low, high]` (both rounded towards the inside).
    #[must_use]
    pub const fn integer(low: f64, high: f64) -> Self {
        Self {
            low,
            high,
            integer: true,
        }
    }

    /// Draws a uniform value; integers uniformly among the whole numbers in range.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        if self.integer {
            let low = libm::ceil(self.low);
            let high = libm::floor(self.high);
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "the count of whole numbers in a finite parameter range"
            )]
            let count = (high - low) as usize + 1;
            #[expect(
                clippy::cast_precision_loss,
                reason = "parameter ranges are far below 2^52"
            )]
            let offset = below(rng, count.max(1)) as f64;
            low + offset
        } else {
            uniform(rng, self.low, self.high).min(self.high)
        }
    }

    /// Clamps and, for integers, rounds `value` into range.
    fn fit(&self, value: f64) -> f64 {
        if self.integer {
            libm::round(value).clamp(libm::ceil(self.low), libm::floor(self.high))
        } else {
            value.clamp(self.low, self.high)
        }
    }

    /// Returns `true` if `value` is in range (and whole, for integers).
    fn admits(&self, value: f64) -> bool {
        (self.low..=self.high).contains(&value)
            && (!self.integer || libm::round(value).total_cmp(&value).is_eq())
    }
}

/// One action type: its parameter list.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionKind {
    /// The parameters every action of this type carries, in order.
    pub params: Vec<ParamSpec>,
}

impl ActionKind {
    /// An action type with the given parameters.
    #[must_use]
    pub fn new(params: Vec<ParamSpec>) -> Self {
        Self { params }
    }
}

/// One gene of a GLEAM chain: an action type and its parameter values.
#[derive(Debug, Clone, PartialEq)]
pub struct Action {
    /// Index into [`ActionModel::kinds`].
    pub kind: usize,
    /// One value per parameter of the kind, within its [`ParamSpec`].
    pub params: Vec<f64>,
}

/// The action types a chain may contain. Implements [`Initializer<Action>`]: a uniform kind
/// with uniform parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionModel {
    /// The action types; never empty.
    kinds: Vec<ActionKind>,
}

impl ActionModel {
    /// Creates a model.
    ///
    /// # Panics
    ///
    /// Panics if `kinds` is empty or a parameter range is empty or not finite.
    #[must_use]
    pub fn new(kinds: Vec<ActionKind>) -> Self {
        assert!(
            !kinds.is_empty(),
            "an action model needs at least one action type"
        );
        for spec in kinds.iter().flat_map(|k| &k.params) {
            assert!(
                spec.low.is_finite() && spec.high.is_finite() && spec.low <= spec.high,
                "invalid parameter range [{}, {}]",
                spec.low,
                spec.high
            );
        }

        Self { kinds }
    }

    /// The action types.
    #[must_use]
    pub fn kinds(&self) -> &[ActionKind] {
        &self.kinds
    }

    /// Returns `true` if `action` has a known kind and admissible parameters.
    #[must_use]
    pub fn is_valid(&self, action: &Action) -> bool {
        self.kinds.get(action.kind).is_some_and(|kind| {
            kind.params.len() == action.params.len()
                && kind
                    .params
                    .iter()
                    .zip(&action.params)
                    .all(|(spec, &v)| spec.admits(v))
        })
    }

    /// Draws new parameters for an action of type `kind`.
    fn sample_params<R: Rng + ?Sized>(&self, kind: usize, rng: &mut R) -> Vec<f64> {
        self.kinds[kind]
            .params
            .iter()
            .map(|spec| spec.sample(rng))
            .collect()
    }
}

impl Initializer<Action> for ActionModel {
    fn initialize<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Action {
        let kind = below(rng, self.kinds.len());
        let params = self.sample_params(kind, rng);

        Action { kind, params }
    }
}

/// Random chains of uniform length in `min_len..=max_len`, genes from `genes`.
#[derive(Debug, Clone, PartialEq)]
pub struct RandomChain<S> {
    /// The gene sampler, for example an [`ActionModel`].
    pub genes: S,
    /// The shortest chain.
    pub min_len: usize,
    /// The longest chain, `>= min_len`.
    pub max_len: usize,
}

impl<T, S: Initializer<T>> Initializer<Vec<T>> for RandomChain<S> {
    fn initialize<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Vec<T> {
        let span = self.max_len.saturating_sub(self.min_len);
        let len = self.min_len + below(rng, span + 1);

        (0..len).map(|_| self.genes.initialize(rng)).collect()
    }
}

/// Picks a random segment `[i, j)` of a chain of length `n >= 1`, at most `max_len` long
/// (`None`: unbounded).
fn segment<R: Rng + ?Sized>(rng: &mut R, n: usize, max_len: Option<usize>) -> (usize, usize) {
    let start = below(rng, n);
    let longest = (n - start).min(max_len.unwrap_or(usize::MAX)).max(1);
    let len = 1 + below(rng, longest);

    (start, start + len)
}

/// Moves one gene to another position (GLEAM "gene shift").
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GeneShift;

impl<T> Mutation<Vec<T>> for GeneShift {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut Vec<T>, rng: &mut R) {
        super::permutation::Insertion.mutate(genome, rng);
    }
}

/// Moves a random segment to another position (GLEAM "segment shift").
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SegmentShift {
    /// The longest segment moved; `None` means unbounded.
    pub max_segment: Option<usize>,
}

impl<T> Mutation<Vec<T>> for SegmentShift {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut Vec<T>, rng: &mut R) {
        if genome.len() < 2 {
            return;
        }

        let (start, end) = segment(rng, genome.len(), self.max_segment);
        let moved: Vec<T> = genome.drain(start..end).collect();
        let at = below(rng, genome.len() + 1);
        let tail = genome.split_off(at);
        genome.extend(moved);
        genome.extend(tail);
    }
}

/// Reverses a random segment (GLEAM "segment inversion").
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SegmentInversion {
    /// The longest segment reversed; `None` means unbounded.
    pub max_segment: Option<usize>,
}

impl<T> Mutation<Vec<T>> for SegmentInversion {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut Vec<T>, rng: &mut R) {
        if genome.len() >= 2 {
            let (start, end) = segment(rng, genome.len(), self.max_segment);
            genome[start..end].reverse();
        }
    }
}

/// Inserts one new random gene at a random position, unless the chain has `max_len` genes.
#[derive(Debug, Clone, PartialEq)]
pub struct InsertGene<S> {
    /// The gene sampler.
    pub genes: S,
    /// The longest chain allowed.
    pub max_len: usize,
}

impl<T, S: Initializer<T>> Mutation<Vec<T>> for InsertGene<S> {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut Vec<T>, rng: &mut R) {
        if genome.len() < self.max_len {
            let at = below(rng, genome.len() + 1);
            genome.insert(at, self.genes.initialize(rng));
        }
    }
}

/// Inserts a segment of new random genes (length `1..=max_segment`), within `max_len`.
#[derive(Debug, Clone, PartialEq)]
pub struct InsertSegment<S> {
    /// The gene sampler.
    pub genes: S,
    /// The longest segment inserted, at least `1`.
    pub max_segment: usize,
    /// The longest chain allowed.
    pub max_len: usize,
}

impl<T, S: Initializer<T>> Mutation<Vec<T>> for InsertSegment<S> {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut Vec<T>, rng: &mut R) {
        let room = self
            .max_len
            .saturating_sub(genome.len())
            .min(self.max_segment.max(1));
        if room == 0 {
            return;
        }

        let len = 1 + below(rng, room);
        let at = below(rng, genome.len() + 1);
        let tail = genome.split_off(at);
        for _ in 0..len {
            genome.push(self.genes.initialize(rng));
        }

        genome.extend(tail);
    }
}

/// Deletes one random gene, unless the chain has only `min_len` genes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeleteGene {
    /// The shortest chain allowed.
    pub min_len: usize,
}

impl<T> Mutation<Vec<T>> for DeleteGene {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut Vec<T>, rng: &mut R) {
        if genome.len() > self.min_len && !genome.is_empty() {
            let at = below(rng, genome.len());
            genome.remove(at);
        }
    }
}

/// Deletes a random segment, keeping at least `min_len` genes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeleteSegment {
    /// The shortest chain allowed.
    pub min_len: usize,
    /// The longest segment deleted; `None` means unbounded.
    pub max_segment: Option<usize>,
}

impl<T> Mutation<Vec<T>> for DeleteSegment {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut Vec<T>, rng: &mut R) {
        let removable = genome.len().saturating_sub(self.min_len);
        if removable == 0 {
            return;
        }

        let cap = Some(self.max_segment.unwrap_or(usize::MAX).min(removable));
        let (start, end) = segment(rng, genome.len(), cap);
        genome.drain(start..end);
    }
}

/// Duplicates one random gene in place (the copy follows the original), within `max_len`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DuplicateGene {
    /// The longest chain allowed.
    pub max_len: usize,
}

impl<T: Clone> Mutation<Vec<T>> for DuplicateGene {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut Vec<T>, rng: &mut R) {
        if !genome.is_empty() && genome.len() < self.max_len {
            let at = below(rng, genome.len());
            genome.insert(at + 1, genome[at].clone());
        }
    }
}

/// Replaces one random gene by a new random gene (GLEAM "new action").
#[derive(Debug, Clone, PartialEq)]
pub struct ReplaceGene<S> {
    /// The gene sampler.
    pub genes: S,
}

impl<T, S: Initializer<T>> Mutation<Vec<T>> for ReplaceGene<S> {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut Vec<T>, rng: &mut R) {
        if !genome.is_empty() {
            let at = below(rng, genome.len());
            genome[at] = self.genes.initialize(rng);
        }
    }
}

/// Picks a random parameter of a random gene that has parameters: `(gene, parameter)`.
fn random_parameter<R: Rng + ?Sized>(genome: &[Action], rng: &mut R) -> Option<(usize, usize)> {
    let with_params: Vec<usize> = (0..genome.len())
        .filter(|&g| !genome[g].params.is_empty())
        .collect();
    if with_params.is_empty() {
        return None;
    }

    let gene = with_params[below(rng, with_params.len())];

    Some((gene, below(rng, genome[gene].params.len())))
}

/// Gives one parameter of one gene a new uniform value within its bounds (GLEAM "parameter
/// new value").
#[derive(Debug, Clone, PartialEq)]
pub struct ParamNewValue {
    /// The action model holding the parameter bounds.
    pub model: ActionModel,
}

impl Mutation<Vec<Action>> for ParamNewValue {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut Vec<Action>, rng: &mut R) {
        if let Some((gene, param)) = random_parameter(genome, rng) {
            let spec = self.model.kinds[genome[gene].kind].params[param];
            genome[gene].params[param] = spec.sample(rng);
        }
    }
}

/// Changes one parameter of one gene by a small amount (GLEAM "parameter shift").
///
/// The step is `±width · 10^(-3u)` with `u ~ U(0, 1)`: log-uniform between the whole range
/// and a thousandth of it, so small changes are as likely per decade as large ones (a
/// reconstruction of GLEAM's ten change classes, which the guide does not spell out). Integer
/// parameters move by at least one; the result is clamped to the bounds.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamShift {
    /// The action model holding the parameter bounds.
    pub model: ActionModel,
}

impl Mutation<Vec<Action>> for ParamShift {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut Vec<Action>, rng: &mut R) {
        let Some((gene, param)) = random_parameter(genome, rng) else {
            return;
        };

        let spec = self.model.kinds[genome[gene].kind].params[param];
        let mut step = (spec.high - spec.low) * libm::pow(10.0, -3.0 * unit_f64(rng));
        if spec.integer {
            step = step.max(1.0);
        }

        let sign = if rng.next_u32() & 1 == 0 { -1.0 } else { 1.0 };
        let value = &mut genome[gene].params[param];
        *value = spec.fit(*value + sign * step);
    }
}

/// Gives every parameter of one gene new uniform values (GLEAM "action new parameters").
#[derive(Debug, Clone, PartialEq)]
pub struct ActionNewParams {
    /// The action model holding the parameter bounds.
    pub model: ActionModel,
}

impl Mutation<Vec<Action>> for ActionNewParams {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut Vec<Action>, rng: &mut R) {
        if !genome.is_empty() {
            let gene = below(rng, genome.len());
            genome[gene].params = self.model.sample_params(genome[gene].kind, rng);
        }
    }
}

/// Cut-and-splice crossover: independent cut points in each parent, tails exchanged. Children
/// lengths vary, which lets chain length evolve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CutAndSplice {
    /// Children longer than this are truncated.
    pub max_len: usize,
}

impl<T: Clone> Crossover<Vec<T>> for CutAndSplice {
    fn crossover<R: Rng + ?Sized>(&mut self, a: &Vec<T>, b: &Vec<T>, rng: &mut R) -> [Vec<T>; 2] {
        let cut_a = below(rng, a.len() + 1);
        let cut_b = below(rng, b.len() + 1);
        let mut c1 = splice(a, b, cut_a, cut_b);
        let mut c2 = splice(b, a, cut_b, cut_a);
        c1.truncate(self.max_len);
        c2.truncate(self.max_len);

        [c1, c2]
    }
}

/// GLEAM n-point crossover on chains of different lengths: `points` cut fractions are drawn
/// once and mapped to each parent's length, and the children alternate parents segment by
/// segment, so corresponding regions of the parents are exchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NPoint {
    /// The number of cut points, at least `1`.
    pub points: usize,
}

impl<T: Clone> Crossover<Vec<T>> for NPoint {
    fn crossover<R: Rng + ?Sized>(&mut self, a: &Vec<T>, b: &Vec<T>, rng: &mut R) -> [Vec<T>; 2] {
        let mut fractions: Vec<f64> = (0..self.points.max(1)).map(|_| unit_f64(rng)).collect();
        fractions.sort_by(f64::total_cmp);
        let cuts_a = scaled_cuts(&fractions, a.len());
        let cuts_b = scaled_cuts(&fractions, b.len());

        [
            alternate(a, b, &cuts_a, &cuts_b),
            alternate(b, a, &cuts_b, &cuts_a),
        ]
    }
}

/// Maps sorted fractions to cut positions `0..=len`, keeping the sentinels `0` and `len`.
fn scaled_cuts(fractions: &[f64], len: usize) -> Vec<usize> {
    let mut cuts = Vec::with_capacity(fractions.len() + 2);
    cuts.push(0);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "a fraction in [0, 1) of a chain length"
    )]
    cuts.extend(
        fractions
            .iter()
            .map(|f| libm::round(f * len as f64) as usize),
    );
    cuts.push(len);

    cuts
}

/// Concatenates segment `k` of `first` for even `k` and of `second` for odd `k`.
fn alternate<T: Clone>(
    first: &[T],
    second: &[T],
    cuts_first: &[usize],
    cuts_second: &[usize],
) -> Vec<T> {
    let mut child = Vec::with_capacity(first.len().max(second.len()));
    for k in 0..cuts_first.len() - 1 {
        if k % 2 == 0 {
            child.extend_from_slice(&first[cuts_first[k]..cuts_first[k + 1]]);
        } else {
            child.extend_from_slice(&second[cuts_second[k]..cuts_second[k + 1]]);
        }
    }

    child
}

/// Exchanges one random segment of each parent; the segments may differ in length and
/// position (GLEAM segment recombination).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentExchange {
    /// Children longer than this are truncated.
    pub max_len: usize,
}

impl<T: Clone> Crossover<Vec<T>> for SegmentExchange {
    fn crossover<R: Rng + ?Sized>(&mut self, a: &Vec<T>, b: &Vec<T>, rng: &mut R) -> [Vec<T>; 2] {
        if a.is_empty() || b.is_empty() {
            return [a.clone(), b.clone()];
        }

        let (ia, ja) = segment(rng, a.len(), None);
        let (ib, jb) = segment(rng, b.len(), None);
        let build = |x: &[T], (i, j): (usize, usize), y: &[T], (k, l): (usize, usize)| {
            let mut child = Vec::with_capacity(x.len() - (j - i) + (l - k));
            child.extend_from_slice(&x[..i]);
            child.extend_from_slice(&y[k..l]);
            child.extend_from_slice(&x[j..]);
            child.truncate(self.max_len);
            child
        };

        [
            build(a, (ia, ja), b, (ib, jb)),
            build(b, (ib, jb), a, (ia, ja)),
        ]
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use proptest::prelude::*;

    use super::*;
    use crate::SplitMix64;

    /// A model with a real, an integer and a parameterless action.
    fn model() -> ActionModel {
        ActionModel::new(vec![
            ActionKind::new(vec![
                ParamSpec::real(-1.0, 1.0),
                ParamSpec::integer(0.0, 5.0),
            ]),
            ActionKind::new(vec![ParamSpec::integer(-3.0, 3.0)]),
            ActionKind::new(vec![]),
        ])
    }

    proptest! {
        /// Gene-level mutations keep every action valid; chain mutations respect length limits.
        #[test]
        fn chain_operators_keep_validity(seed in any::<u64>()) {
            let mut rng = SplitMix64::new(seed);
            let m = model();
            let mut init = RandomChain { genes: m.clone(), min_len: 1, max_len: 8 };
            let mut chain = init.initialize(&mut rng);
            let mut ops = mix(
                mix(ParamShift { model: m.clone() }, 1.0, ParamNewValue { model: m.clone() }, 1.0),
                1.0,
                mix(
                    mix(InsertSegment { genes: m.clone(), max_segment: 3, max_len: 12 }, 1.0, DeleteSegment { min_len: 1, max_segment: Some(3) }, 1.0),
                    1.0,
                    mix(SegmentShift::default(), 1.0, mix(DuplicateGene { max_len: 12 }, 1.0, ActionNewParams { model: m.clone() }, 1.0), 1.0),
                    1.0,
                ),
                1.0,
            );
            for _ in 0..50 {
                ops.mutate(&mut chain, &mut rng);
                prop_assert!((1..=12).contains(&chain.len()));
                prop_assert!(chain.iter().all(|a| m.is_valid(a)));
            }
        }

        /// Crossovers only rearrange the parents' genes.
        #[test]
        fn crossovers_reuse_parent_genes(seed in any::<u64>(), la in 0u32..10, lb in 0u32..10) {
            let mut rng = SplitMix64::new(seed);
            let a: Vec<u32> = (0..la).collect();
            let b: Vec<u32> = (100..100 + lb).collect();
            for [c1, c2] in [
                CutAndSplice { max_len: 100 }.crossover(&a, &b, &mut rng),
                NPoint { points: 3 }.crossover(&a, &b, &mut rng),
                SegmentExchange { max_len: 100 }.crossover(&a, &b, &mut rng),
            ] {
                let mut all: Vec<u32> = c1.iter().chain(&c2).copied().collect();
                all.sort_unstable();
                let mut parents: Vec<u32> = a.iter().chain(&b).copied().collect();
                parents.sort_unstable();
                prop_assert_eq!(all, parents);
            }
        }
    }

    use crate::operators::mix;

    /// Segment shift keeps the multiset of genes.
    #[test]
    fn segment_shift_permutes() {
        let mut rng = SplitMix64::new(4);
        let mut chain: Vec<u32> = (0..20).collect();
        for _ in 0..100 {
            SegmentShift {
                max_segment: Some(5),
            }
            .mutate(&mut chain, &mut rng);
            SegmentInversion::default().mutate(&mut chain, &mut rng);
        }

        let mut sorted = chain.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..20).collect::<Vec<_>>());
    }
}
