//! Runtime reporting: an [`Observer`] receives [`Event`]s while an algorithm runs.
//!
//! Events are informational only; they never change a run.

use crate::Stats;

/// Something worth reporting while an algorithm runs.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Event {
    /// A generation (or iteration) completed.
    Generation {
        /// The counters after the generation.
        stats: Stats,
        /// The cost of the best member of the current population, which can be worse than
        /// [`Stats::best_cost`] for non-elitist algorithms.
        population_best: f64,
    },
    /// A new best-so-far genome was evaluated.
    Improved {
        /// Evaluations so far, including the improving one.
        evaluations: u64,
        /// The new best cost.
        cost: f64,
    },
    /// A restart began (IPOP-CMA-ES, RMHC with restarts).
    Restart {
        /// The restart number, `1` for the first restart.
        restart: u64,
        /// The population size of the restarted run.
        population: usize,
    },
}

/// Receives [`Event`]s while an algorithm runs.
///
/// Implemented for `()` (ignores everything; [`Observer::ENABLED`] is `false`, so the
/// algorithms skip all event bookkeeping), for `&mut O`, and for closures through
/// [`observer_fn`].
pub trait Observer {
    /// Whether the algorithms should produce events at all; `false` only for `()`.
    const ENABLED: bool = true;

    /// Receives one event.
    fn observe(&mut self, event: Event) {
        let _ = event;
    }
}

impl Observer for () {
    const ENABLED: bool = false;
}

impl<O: Observer + ?Sized> Observer for &mut O {
    const ENABLED: bool = O::ENABLED;

    fn observe(&mut self, event: Event) {
        (**self).observe(event);
    }
}

/// Turns a closure `FnMut(Event)` into an [`Observer`].
///
/// ```
/// use evolutionary_planning::{Event, Observer, observer_fn};
///
/// let mut improvements = 0;
/// {
///     let mut observer = observer_fn(|event| {
///         if let Event::Improved { .. } = event {
///             improvements += 1;
///         }
///     });
///     observer.observe(Event::Improved { evaluations: 1, cost: 0.5 });
/// }
/// assert_eq!(improvements, 1);
/// ```
pub fn observer_fn<F: FnMut(Event)>(f: F) -> ObserverFn<F> {
    ObserverFn(f)
}

/// A closure used as an [`Observer`]; created by [`observer_fn`].
#[derive(Debug, Clone, Copy)]
pub struct ObserverFn<F>(F);

impl<F: FnMut(Event)> Observer for ObserverFn<F> {
    fn observe(&mut self, event: Event) {
        (self.0)(event);
    }
}
