//! What is optimised: [`Objective`] for one cost, [`MultiObjective`] for a cost vector with a
//! constraint violation.

/// A cost to minimise for each genome.
///
/// Implemented for every `FnMut(&G) -> f64`, so a closure is an objective. Evaluation may
/// have side effects (a simulator, a counter); the algorithms call it exactly once per
/// evaluated individual and count those calls against the budget.
///
/// ```
/// use evolutionary_planning::Objective;
///
/// let mut sphere = |x: &Vec<f64>| x.iter().map(|v| v * v).sum::<f64>();
/// assert_eq!(sphere.evaluate(&vec![1.0, 2.0]), 5.0);
/// ```
pub trait Objective<G: ?Sized> {
    /// Returns the cost of `genome`; smaller is better. NaN is treated as `+∞`.
    fn evaluate(&mut self, genome: &G) -> f64;
}

impl<G: ?Sized, F: FnMut(&G) -> f64> Objective<G> for F {
    fn evaluate(&mut self, genome: &G) -> f64 {
        self(genome)
    }
}

/// Several costs to minimise at once, plus a constraint violation.
///
/// The violation is `0` for a feasible genome and grows with how far it is from feasibility
/// (Jakob's rule: never flatten infeasibility). Constrained domination prefers feasible over
/// infeasible genomes and, among infeasible ones, the smaller violation (Deb 2000).
///
/// ```
/// use evolutionary_planning::{MultiObjective, multi_objective_fn};
///
/// // Schaffer's problem: minimise x^2 and (x - 2)^2.
/// let mut schaffer = multi_objective_fn(2, |x: &f64, out: &mut [f64]| {
///     out[0] = x * x;
///     out[1] = (x - 2.0) * (x - 2.0);
///     0.0
/// });
/// let mut out = [0.0; 2];
/// assert_eq!(schaffer.evaluate(&1.0, &mut out), 0.0);
/// assert_eq!(out, [1.0, 1.0]);
/// ```
pub trait MultiObjective<G: ?Sized> {
    /// The number of objectives, `m >= 1`; constant for the lifetime of the problem.
    fn objectives(&self) -> usize;

    /// Writes the `m` costs of `genome` into `out` (of length [`MultiObjective::objectives`])
    /// and returns its constraint violation (`0` if feasible, NaN treated as `+∞`).
    fn evaluate(&mut self, genome: &G, out: &mut [f64]) -> f64;
}

impl<G: ?Sized, M: MultiObjective<G> + ?Sized> MultiObjective<G> for &mut M {
    fn objectives(&self) -> usize {
        (**self).objectives()
    }

    fn evaluate(&mut self, genome: &G, out: &mut [f64]) -> f64 {
        (**self).evaluate(genome, out)
    }
}

/// Turns a closure `FnMut(&G, &mut [f64]) -> f64` with `objectives` costs into a
/// [`MultiObjective`].
pub fn multi_objective_fn<G: ?Sized, F: FnMut(&G, &mut [f64]) -> f64>(
    objectives: usize,
    f: F,
) -> MultiObjectiveFn<F> {
    MultiObjectiveFn { objectives, f }
}

/// A closure used as a [`MultiObjective`]; created by [`multi_objective_fn`].
#[derive(Debug, Clone, Copy)]
pub struct MultiObjectiveFn<F> {
    /// The number of objectives the closure writes.
    objectives: usize,
    /// The closure: writes the costs and returns the violation.
    f: F,
}

impl<G: ?Sized, F: FnMut(&G, &mut [f64]) -> f64> MultiObjective<G> for MultiObjectiveFn<F> {
    fn objectives(&self) -> usize {
        self.objectives
    }

    fn evaluate(&mut self, genome: &G, out: &mut [f64]) -> f64 {
        (self.f)(genome, out)
    }
}
