//! The bookkeeping shared by the multi-objective algorithms: the evaluator (budget, NaN
//! sanitising, counters, events) and [`MultiOutcome`].

use alloc::vec::Vec;

use crate::pareto::{Evaluated, non_dominated_sort};
use crate::{Budget, Event, MultiObjective, Observer, Stats, Stop, sanitize};

/// The result of a multi-objective run.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiOutcome<G> {
    /// The first non-dominated front (under constrained domination) of the final population
    /// or archive: feasible members only, unless no member is feasible.
    pub front: Vec<Evaluated<G>>,
    /// The final population.
    pub population: Vec<Evaluated<G>>,
    /// The counters at the end of the run. [`Stats::best_cost`] stays `+∞`: there is no single
    /// best cost, so a target cost never stops a multi-objective run.
    pub stats: Stats,
    /// Why the run ended.
    pub stop: Stop,
}

/// Counts evaluations against a budget for a [`MultiObjective`].
pub(crate) struct MultiEvaluator<'a, M: ?Sized, B: ?Sized, Obs: ?Sized> {
    /// The problem.
    problem: &'a mut M,
    /// Asked before every evaluation.
    budget: &'a mut B,
    /// Receives `Generation` events.
    observer: &'a mut Obs,
    /// The counters so far.
    pub(crate) stats: Stats,
    /// The number of objectives.
    pub(crate) objectives: usize,
}

impl<'a, M: ?Sized, B, Obs> MultiEvaluator<'a, M, B, Obs>
where
    B: Budget<Stats> + ?Sized,
    Obs: Observer + ?Sized,
{
    /// Starts counting from zero.
    pub(crate) fn new<G>(problem: &'a mut M, budget: &'a mut B, observer: &'a mut Obs) -> Self
    where
        M: MultiObjective<G>,
    {
        let objectives = problem.objectives();

        Self {
            problem,
            budget,
            observer,
            stats: Stats::default(),
            objectives,
        }
    }

    /// Evaluates `genome` unless the budget is exhausted.
    pub(crate) fn evaluate<G>(&mut self, genome: G) -> Option<Evaluated<G>>
    where
        M: MultiObjective<G>,
    {
        if self.budget.exhausted(&self.stats) {
            return None;
        }

        let mut objectives = alloc::vec![0.0; self.objectives];
        let violation = sanitize(self.problem.evaluate(&genome, &mut objectives)).max(0.0);
        for value in &mut objectives {
            *value = sanitize(*value);
        }

        self.stats.evaluations += 1;

        Some(Evaluated {
            genome,
            objectives,
            violation,
        })
    }

    /// Marks a generation as complete.
    pub(crate) fn end_generation(&mut self) {
        self.stats.generations += 1;
        if Obs::ENABLED {
            self.observer.observe(Event::Generation {
                stats: self.stats,
                population_best: f64::INFINITY,
            });
        }
    }

    /// Ends the run: computes the first front of `population`.
    pub(crate) fn outcome<G: Clone>(
        &self,
        population: Vec<Evaluated<G>>,
        stop: Stop,
    ) -> MultiOutcome<G> {
        MultiOutcome {
            front: first_front(&population),
            population,
            stats: self.stats,
            stop,
        }
    }
}

/// The first front under constrained domination; feasible members only if any is feasible
/// (constrained domination already guarantees that).
pub(crate) fn first_front<G: Clone>(population: &[Evaluated<G>]) -> Vec<Evaluated<G>> {
    let points: Vec<&[f64]> = population.iter().map(|e| e.objectives.as_slice()).collect();
    let violations: Vec<f64> = population.iter().map(|e| e.violation).collect();
    let fronts = non_dominated_sort(&points, &violations);

    fronts
        .first()
        .map(|front| front.iter().map(|&i| population[i].clone()).collect())
        .unwrap_or_default()
}
