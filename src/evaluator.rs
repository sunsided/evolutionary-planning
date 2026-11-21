//! The bookkeeping every single-objective algorithm shares: budget checks before each
//! evaluation, NaN sanitising, the best-so-far genome, counters and events.

use alloc::vec::Vec;

use crate::{Budget, Event, Individual, Objective, Observer, Stats, Stop, sanitize};

/// The result of a single-objective run.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome<G> {
    /// The best genome evaluated during the whole run (not only in the final population).
    /// `None` only if the budget allowed no evaluation at all.
    pub best: Option<Individual<G>>,
    /// The final population, sorted ascending by cost. Empty for single-point algorithms
    /// that keep no population other than `best`.
    pub population: Vec<Individual<G>>,
    /// The counters at the end of the run.
    pub stats: Stats,
    /// Why the run ended.
    pub stop: Stop,
}

impl<G> Outcome<G> {
    /// The best cost found, `+∞` if nothing was evaluated.
    #[must_use]
    pub fn best_cost(&self) -> f64 {
        self.best.as_ref().map_or(f64::INFINITY, |best| best.cost)
    }

    /// The best genome found, if any.
    #[must_use]
    pub fn best_genome(&self) -> Option<&G> {
        self.best.as_ref().map(|best| &best.genome)
    }
}

/// Wraps an objective with a budget and an observer, counting evaluations and keeping the
/// best genome seen.
pub(crate) struct Evaluator<'a, G, O: ?Sized, B: ?Sized, Obs: ?Sized> {
    /// The objective to minimise.
    objective: &'a mut O,
    /// Asked before every evaluation.
    budget: &'a mut B,
    /// Receives `Improved` and `Generation` events.
    observer: &'a mut Obs,
    /// The counters so far.
    pub(crate) stats: Stats,
    /// The best genome evaluated so far.
    pub(crate) best: Option<Individual<G>>,
}

impl<'a, G: Clone, O, B, Obs> Evaluator<'a, G, O, B, Obs>
where
    O: Objective<G> + ?Sized,
    B: Budget<Stats> + ?Sized,
    Obs: Observer + ?Sized,
{
    /// Starts counting from zero.
    pub(crate) fn new(objective: &'a mut O, budget: &'a mut B, observer: &'a mut Obs) -> Self {
        Self {
            objective,
            budget,
            observer,
            stats: Stats::default(),
            best: None,
        }
    }

    /// Returns `true` if the budget forbids another evaluation.
    pub(crate) fn exhausted(&mut self) -> bool {
        self.budget.exhausted(&self.stats)
    }

    /// Evaluates `genome` unless the budget is exhausted; returns its sanitised cost.
    pub(crate) fn evaluate(&mut self, genome: &G) -> Option<f64> {
        if self.exhausted() {
            return None;
        }

        let cost = sanitize(self.objective.evaluate(genome));
        self.stats.evaluations += 1;
        self.record(genome, cost);

        Some(cost)
    }

    /// Evaluates `genome` and pairs it with its cost, unless the budget is exhausted.
    pub(crate) fn individual(&mut self, genome: G) -> Option<Individual<G>> {
        let cost = self.evaluate(&genome)?;

        Some(Individual { genome, cost })
    }

    /// Updates the best-so-far genome with a freshly evaluated one and reports improvements.
    fn record(&mut self, genome: &G, cost: f64) {
        if cost < self.stats.best_cost || self.best.is_none() {
            self.stats.best_cost = cost;
            self.best = Some(Individual {
                genome: genome.clone(),
                cost,
            });
            if Obs::ENABLED {
                self.observer.observe(Event::Improved {
                    evaluations: self.stats.evaluations,
                    cost,
                });
            }
        }
    }

    /// Marks a generation as complete and reports it.
    pub(crate) fn end_generation(&mut self, population_best: f64) {
        self.stats.generations += 1;
        if Obs::ENABLED {
            self.observer.observe(Event::Generation {
                stats: self.stats,
                population_best,
            });
        }
    }

    /// Reports another event, such as a restart.
    pub(crate) fn emit(&mut self, event: Event) {
        if Obs::ENABLED {
            self.observer.observe(event);
        }
    }

    /// Ends the run, returning the outcome; the best genome moves into it.
    pub(crate) fn outcome(&mut self, mut population: Vec<Individual<G>>, stop: Stop) -> Outcome<G> {
        crate::cost::sort_population(&mut population);

        Outcome {
            best: self.best.take(),
            population,
            stats: self.stats,
            stop,
        }
    }
}
