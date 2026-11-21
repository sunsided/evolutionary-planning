//! Rolling-horizon evolutionary planning (RHEA; Perez et al. 2013) over a caller-supplied
//! forward model, with random-mutation hill climbing (RMHC) as the population-of-one case
//! used by Olesen et al. (2020) to plan in the latent space of a learned world model.
//!
//! At every decision the planner evolves action sequences of length `horizon`, each scored
//! by rolling it out through the model from the current state; it returns the first action
//! of the best sequence. With *shift buffering* the best sequence is shifted by one step,
//! padded with a random action and becomes the starting point of the next decision, so the
//! search is not repeated from scratch. Each decision spends exactly
//! [`Config::evaluations`] sequence evaluations (at least one per initial sequence).
//!
//! The model is the caller's: a simulator, an analytic model or a learned one. This crate
//! minimises, so [`ForwardModel::step`] reports a per-step **cost** (use `-reward` for a
//! reward-based model).
//!
//! ```
//! use evolutionary_planning::rhea::{Config, ForwardModel, RollingHorizon, Step};
//! use evolutionary_planning::operators::initializer_fn;
//! use evolutionary_planning::{SplitMix64, below};
//! use rand_core::Rng;
//!
//! /// A corridor: move left (-1) or right (+1); every step before reaching cell 5 costs 1.
//! struct Corridor;
//!
//! impl ForwardModel for Corridor {
//!     type State = i32;
//!     type Action = i32;
//!
//!     fn step<R: Rng + ?Sized>(&mut self, state: &i32, action: &i32, _rng: &mut R) -> Step<i32> {
//!         let next = state + action;
//!         Step { state: next, cost: 1.0, done: next == 5 }
//!     }
//! }
//!
//! let actions = initializer_fn(|rng| if below(rng, 2) == 0 { -1 } else { 1 });
//! let mut planner = RollingHorizon::rmhc(Config { horizon: 8, ..Config::default() }, actions);
//! let mut rng = SplitMix64::new(1);
//! let mut state = 0;
//! for _ in 0..5 {
//!     let decision = planner.plan(&mut Corridor, &state, &mut rng);
//!     state += decision.action;
//! }
//! assert_eq!(state, 5);
//! ```

use alloc::vec::Vec;

use rand_core::Rng;

use crate::operators::selection::{Selection, Tournament};
use crate::operators::sequence::UniformCrossover;
use crate::operators::{Crossover, Initializer, Mutation};
use crate::rng::{below, chance};
use crate::{ranking, sanitize};

/// The result of one model step.
#[derive(Debug, Clone, PartialEq)]
pub struct Step<S> {
    /// The successor state.
    pub state: S,
    /// The cost of the step (minimised; `-reward` for reward models).
    pub cost: f64,
    /// Whether the episode ended: the rest of the sequence is not simulated.
    pub done: bool,
}

/// A forward model: what happens when an action is taken in a state.
///
/// Stochastic models draw from `rng`; [`Config::rollouts`] averages several samples.
pub trait ForwardModel {
    /// The world state.
    type State: Clone;
    /// An action.
    type Action: Clone;

    /// Simulates one step from `state` under `action`.
    fn step<R: Rng + ?Sized>(
        &mut self,
        state: &Self::State,
        action: &Self::Action,
        rng: &mut R,
    ) -> Step<Self::State>;
}

/// Parameters of a [`RollingHorizon`] planner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// The length of every action sequence, at least `1`.
    pub horizon: usize,
    /// The number of sequences evolved; `1` is RMHC.
    pub population_size: usize,
    /// Sequence evaluations (each `rollouts` model rollouts) per decision. Exact, except that
    /// every sequence of the initial population is evaluated, so the effective budget is at
    /// least `population_size`.
    pub evaluations: u64,
    /// The discount `γ ∈ (0, 1]` of step costs; `1` is the undiscounted sum of Olesen et al.
    pub discount: f64,
    /// Rollouts averaged per evaluation, for stochastic models.
    pub rollouts: usize,
    /// Keep the best sequence, shifted by one step, as the seed of the next decision.
    pub shift_buffer: bool,
    /// RMHC on stochastic models: re-evaluate the incumbent each iteration so one lucky
    /// rollout cannot hold the lead forever.
    pub reevaluate: bool,
    /// The per-action mutation probability of the default point mutation; `None` means
    /// `1 / horizon`. At least one action always changes.
    pub mutation_rate: Option<f64>,
}

impl Default for Config {
    /// Horizon 10, RMHC, 200 evaluations per decision, undiscounted, one rollout, shift
    /// buffering on.
    fn default() -> Self {
        Self {
            horizon: 10,
            population_size: 1,
            evaluations: 200,
            discount: 1.0,
            rollouts: 1,
            shift_buffer: true,
            reevaluate: false,
            mutation_rate: None,
        }
    }
}

/// One decision of the planner.
#[derive(Debug, Clone, PartialEq)]
pub struct Decision<A> {
    /// The action to execute now: the first of the best sequence.
    pub action: A,
    /// The whole best sequence.
    pub plan: Vec<A>,
    /// Its estimated (discounted) cost.
    pub cost: f64,
    /// Sequence evaluations spent on this decision.
    pub evaluations: u64,
}

/// Replaces each action with probability `rate` (default `1/n`) by a fresh random action;
/// at least one changes. The default mutation of the planner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointMutation<S> {
    /// The action sampler.
    pub actions: S,
    /// The per-action probability; `None` means `1/n`.
    pub rate: Option<f64>,
}

impl<A, S: Initializer<A>> Mutation<Vec<A>> for PointMutation<S> {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut Vec<A>, rng: &mut R) {
        if genome.is_empty() {
            return;
        }

        #[expect(clippy::cast_precision_loss, reason = "horizons are far below 2^52")]
        let rate = self.rate.unwrap_or(1.0 / genome.len() as f64);
        let mut changed = false;
        for action in genome.iter_mut() {
            if chance(rng, rate) {
                *action = self.actions.initialize(rng);
                changed = true;
            }
        }

        if !changed {
            let at = below(rng, genome.len());
            genome[at] = self.actions.initialize(rng);
        }
    }
}

/// The rolling-horizon planner.
#[derive(Debug, Clone, PartialEq)]
pub struct RollingHorizon<A, S, M> {
    /// The parameters.
    pub config: Config,
    /// Samples random actions (initial sequences and shift-buffer padding).
    pub actions: S,
    /// Mutates a sequence.
    pub mutation: M,
    /// The sequences carried over from the last decision (shifted), if any.
    buffer: Vec<Vec<A>>,
}

impl<A: Clone, S: Initializer<A> + Clone> RollingHorizon<A, S, PointMutation<S>> {
    /// A planner with the default point mutation.
    #[must_use]
    pub fn new(config: Config, actions: S) -> Self {
        let mutation = PointMutation {
            actions: actions.clone(),
            rate: config.mutation_rate,
        };

        Self {
            config,
            actions,
            mutation,
            buffer: Vec::new(),
        }
    }

    /// RMHC: a planner with a population of one (whatever `config` says).
    #[must_use]
    pub fn rmhc(config: Config, actions: S) -> Self {
        Self::new(
            Config {
                population_size: 1,
                ..config
            },
            actions,
        )
    }
}

impl<A: Clone, S: Initializer<A>, M: Mutation<Vec<A>>> RollingHorizon<A, S, M> {
    /// A planner with a custom sequence mutation (for example Gaussian perturbation of
    /// continuous actions).
    #[must_use]
    pub fn with_mutation(config: Config, actions: S, mutation: M) -> Self {
        Self {
            config,
            actions,
            mutation,
            buffer: Vec::new(),
        }
    }

    /// Forgets the shift buffer, for example at the start of a new episode.
    pub fn reset(&mut self) {
        self.buffer.clear();
    }

    /// Plans from `state` and returns the first action of the best sequence found.
    ///
    /// # Panics
    ///
    /// Panics if the horizon, population size or rollout count is `0`.
    pub fn plan<F, R>(&mut self, model: &mut F, state: &F::State, rng: &mut R) -> Decision<A>
    where
        F: ForwardModel<Action = A>,
        R: Rng + ?Sized,
    {
        let c = self.config;
        assert!(
            c.horizon > 0 && c.population_size > 0 && c.rollouts > 0,
            "horizon, population and rollouts must be positive"
        );

        let mut population = self.seed_population(rng);
        let mut evaluations = 0;
        let mut costs: Vec<f64> = Vec::with_capacity(population.len());
        for sequence in &population {
            costs.push(self.evaluate(model, state, sequence, rng));
            evaluations += 1;
        }

        while evaluations < c.evaluations.max(1) {
            let remaining = c.evaluations.max(1) - evaluations;
            if c.population_size == 1 {
                evaluations += self.hill_climb_step(
                    model,
                    state,
                    &mut population[0],
                    &mut costs[0],
                    remaining,
                    rng,
                );
            } else {
                evaluations +=
                    self.generation(model, state, &mut population, &mut costs, remaining, rng);
            }
        }

        self.decide(&population, &costs, evaluations, rng)
    }

    /// The shifted buffer, topped up with random sequences.
    fn seed_population<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Vec<Vec<A>> {
        let c = self.config;
        let mut population: Vec<Vec<A>> = core::mem::take(&mut self.buffer);
        population.truncate(c.population_size);
        for sequence in &mut population {
            sequence.resize_with(c.horizon, || self.actions.initialize(rng));
        }

        while population.len() < c.population_size {
            population.push(
                (0..c.horizon)
                    .map(|_| self.actions.initialize(rng))
                    .collect(),
            );
        }

        population
    }

    /// One RMHC iteration within `remaining` evaluations (the incumbent is re-evaluated only
    /// if two remain); returns the evaluations spent.
    fn hill_climb_step<F, R>(
        &mut self,
        model: &mut F,
        state: &F::State,
        current: &mut Vec<A>,
        cost: &mut f64,
        remaining: u64,
        rng: &mut R,
    ) -> u64
    where
        F: ForwardModel<Action = A>,
        R: Rng + ?Sized,
    {
        let mut spent = 0;
        if self.config.reevaluate && remaining >= 2 {
            *cost = self.evaluate(model, state, current, rng);
            spent += 1;
        }

        let mut candidate = current.clone();
        self.mutation.mutate(&mut candidate, rng);
        let candidate_cost = self.evaluate(model, state, &candidate, rng);
        if candidate_cost <= *cost {
            *current = candidate;
            *cost = candidate_cost;
        }

        spent + 1
    }

    /// One generation of the population variant (elitism one, binary tournament, uniform
    /// crossover, mutation) with at most `remaining` evaluations; slots left when they run
    /// out keep the best old sequences. Returns the evaluations spent.
    fn generation<F, R>(
        &mut self,
        model: &mut F,
        state: &F::State,
        population: &mut Vec<Vec<A>>,
        costs: &mut Vec<f64>,
        remaining: u64,
        rng: &mut R,
    ) -> u64
    where
        F: ForwardModel<Action = A>,
        R: Rng + ?Sized,
    {
        let order = ranking(costs);
        let mut next = alloc::vec![population[order[0]].clone()];
        let mut next_costs = alloc::vec![costs[order[0]]];
        let mut parents = Vec::with_capacity(2);
        let mut spent = 0;
        while next.len() < population.len() && spent < remaining {
            parents.clear();
            Tournament { size: 2 }.select(costs, 2, rng, &mut parents);
            let [mut child, _] = UniformCrossover::default().crossover(
                &population[parents[0]],
                &population[parents[1]],
                rng,
            );
            self.mutation.mutate(&mut child, rng);
            next_costs.push(self.evaluate(model, state, &child, rng));
            next.push(child);
            spent += 1;
        }

        for &k in order.iter().skip(1) {
            if next.len() == population.len() {
                break;
            }
            next.push(population[k].clone());
            next_costs.push(costs[k]);
        }

        *population = next;
        *costs = next_costs;
        spent
    }

    /// Picks the best sequence, fills the shift buffer and builds the decision.
    fn decide<R: Rng + ?Sized>(
        &mut self,
        population: &[Vec<A>],
        costs: &[f64],
        evaluations: u64,
        rng: &mut R,
    ) -> Decision<A> {
        let order = ranking(costs);
        let best = order[0];
        let plan = population[best].clone();
        if self.config.shift_buffer {
            self.buffer = order
                .iter()
                .map(|&k| {
                    let mut shifted: Vec<A> = population[k][1..].to_vec();
                    shifted.push(self.actions.initialize(rng));
                    shifted
                })
                .collect();
        }

        Decision {
            action: plan[0].clone(),
            cost: costs[best],
            plan,
            evaluations,
        }
    }

    /// The mean discounted cost of `rollouts` rollouts of `sequence` from `state`.
    fn evaluate<F, R>(&self, model: &mut F, state: &F::State, sequence: &[A], rng: &mut R) -> f64
    where
        F: ForwardModel<Action = A>,
        R: Rng + ?Sized,
    {
        let mut total = 0.0;
        for _ in 0..self.config.rollouts {
            total += rollout(model, state, sequence, self.config.discount, rng);
        }

        #[expect(clippy::cast_precision_loss, reason = "rollout counts are tiny")]
        let mean = total / self.config.rollouts as f64;

        sanitize(mean)
    }
}

/// The discounted cost of executing `sequence` from `state` until it ends or the episode does.
pub fn rollout<F, R>(
    model: &mut F,
    state: &F::State,
    sequence: &[F::Action],
    discount: f64,
    rng: &mut R,
) -> f64
where
    F: ForwardModel,
    R: Rng + ?Sized,
{
    let mut current = state.clone();
    let mut total = 0.0;
    let mut weight = 1.0;
    for action in sequence {
        let step = model.step(&current, action, rng);
        total += weight * step.cost;
        if step.done {
            break;
        }

        weight *= discount;
        current = step.state;
    }

    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::{initializer_fn, mutation_fn};
    use crate::{SplitMix64, standard_normal, unit_f64};

    /// A corridor of cells; each step costs the remaining distance to the goal (an auxiliary
    /// criterion in Jakob's sense: it rewards progress, not only arrival); with `slip` the
    /// move fails.
    struct Corridor {
        /// The goal cell.
        goal: i32,
        /// The probability that a move does nothing.
        slip: f64,
    }

    impl ForwardModel for Corridor {
        type State = i32;
        type Action = i32;

        fn step<R: Rng + ?Sized>(&mut self, state: &i32, action: &i32, rng: &mut R) -> Step<i32> {
            let next = if unit_f64(rng) < self.slip {
                *state
            } else {
                state + action
            };
            Step {
                state: next,
                cost: f64::from((self.goal - next).abs()),
                done: next == self.goal,
            }
        }
    }

    /// A corridor planner with function-pointer action sampling.
    type CorridorPlanner<M> =
        RollingHorizon<i32, crate::operators::InitializerFn<fn(&mut dyn Rng) -> i32>, M>;

    /// Runs a closed-loop episode; returns the steps to the goal.
    fn episode<M: Mutation<Vec<i32>>>(
        planner: &mut CorridorPlanner<M>,
        slip: f64,
        seed: u64,
    ) -> usize {
        let mut model = Corridor { goal: 6, slip };
        let mut rng = SplitMix64::new(seed);
        let mut state = 0;
        for t in 0..100 {
            if state == 6 {
                return t;
            }
            let decision = planner.plan(&mut model, &state, &mut rng);
            state = model.step(&state, &decision.action, &mut rng).state;
        }

        100
    }

    /// The action sampler of the corridor: -1, 0 or +1.
    fn moves(rng: &mut dyn Rng) -> i32 {
        i32::try_from(below(rng, 3)).expect("small") - 1
    }

    /// RMHC and the population variant walk straight to the goal on the deterministic
    /// corridor.
    #[test]
    fn deterministic_corridor() {
        for population_size in [1, 8] {
            let config = Config {
                horizon: 10,
                population_size,
                evaluations: 300,
                ..Config::default()
            };
            let mut planner =
                RollingHorizon::new(config, initializer_fn(moves as fn(&mut dyn Rng) -> i32));
            assert_eq!(
                episode(&mut planner, 0.0, 3),
                6,
                "population {population_size}"
            );
        }
    }

    /// Each decision spends exactly the configured evaluations, with re-evaluation and with a
    /// population.
    #[test]
    fn exact_evaluations_per_decision() {
        for (population_size, reevaluate) in [(1, false), (1, true), (7, false)] {
            let config = Config {
                horizon: 6,
                population_size,
                reevaluate,
                evaluations: 101,
                ..Config::default()
            };
            let mut planner =
                RollingHorizon::new(config, initializer_fn(moves as fn(&mut dyn Rng) -> i32));
            let mut model = Corridor { goal: 6, slip: 0.0 };
            let decision = planner.plan(&mut model, &0, &mut SplitMix64::new(1));
            assert_eq!(decision.evaluations, 101, "{population_size} {reevaluate}");
        }
    }

    /// With slipping moves and averaged, re-evaluated rollouts the goal is still reached
    /// quickly.
    #[test]
    fn stochastic_corridor() {
        let config = Config {
            horizon: 10,
            rollouts: 4,
            reevaluate: true,
            ..Config::default()
        };
        let mut planner =
            RollingHorizon::new(config, initializer_fn(moves as fn(&mut dyn Rng) -> i32));
        let steps = episode(&mut planner, 0.2, 5);

        assert!(steps <= 14, "{steps} steps");
    }

    /// The shift buffer carries the plan: a deterministic plan found once is kept, so later
    /// decisions follow it even with a single evaluation each.
    #[test]
    fn shift_buffer_keeps_the_plan() {
        let config = Config {
            horizon: 8,
            evaluations: 400,
            ..Config::default()
        };
        let mut planner =
            RollingHorizon::new(config, initializer_fn(moves as fn(&mut dyn Rng) -> i32));
        let mut model = Corridor { goal: 6, slip: 0.0 };
        let mut rng = SplitMix64::new(9);
        let first = planner.plan(&mut model, &0, &mut rng);
        planner.config.evaluations = 1;
        let second = planner.plan(&mut model, &first.action, &mut rng);

        assert_eq!(second.plan[..first.plan.len() - 1], first.plan[1..]);
    }

    /// Continuous actions with a Gaussian sequence mutation steer a point mass to the origin.
    #[test]
    fn continuous_point_mass() {
        /// Position and velocity; the action is an acceleration in [-1, 1].
        struct PointMass;

        impl ForwardModel for PointMass {
            type State = (f64, f64);
            type Action = f64;

            fn step<R: Rng + ?Sized>(
                &mut self,
                s: &(f64, f64),
                a: &f64,
                _rng: &mut R,
            ) -> Step<(f64, f64)> {
                let a = a.clamp(-1.0, 1.0);
                let v = s.1 + 0.1 * a;
                let x = s.0 + 0.1 * v;
                Step {
                    state: (x, v),
                    cost: x * x + 0.1 * v * v + 0.01 * a * a,
                    done: false,
                }
            }
        }

        let mutation = mutation_fn(|plan: &mut Vec<f64>, rng: &mut dyn Rng| {
            for a in plan.iter_mut() {
                *a = (*a + 0.3 * standard_normal(rng)).clamp(-1.0, 1.0);
            }
        });
        let config = Config {
            horizon: 20,
            evaluations: 300,
            ..Config::default()
        };
        let mut planner = RollingHorizon::with_mutation(
            config,
            initializer_fn(|rng| 2.0 * unit_f64(rng) - 1.0),
            mutation,
        );
        let mut rng = SplitMix64::new(2);
        let mut state = (1.0, 0.0);
        for _ in 0..150 {
            let decision = planner.plan(&mut PointMass, &state, &mut rng);
            state = PointMass.step(&state, &decision.action, &mut rng).state;
        }

        assert!(state.0.abs() < 0.1 && state.1.abs() < 0.2, "{state:?}");
    }
}
