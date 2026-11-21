//! Genetic search over SAS+ plans (feature `sas-plus`): plans are GLEAM action chains of
//! operator ids, decoded by simulating the [`sas_plus::Task`], scored by EvoPlan's fitness
//! cascade, and repaired phenotypically or genotypically.
//!
//! Decoding uses only [`Task::initial_state`], [`Task::is_applicable`], [`Task::apply`] and
//! [`Task::is_goal`], exactly the steps of [`Task::validate_plan`], so axioms and conditional
//! effects behave identically and every plan reported as valid passes `validate_plan` with
//! the same cost (tested).
//!
//! The cost of a chain (minimised):
//!
//! - **valid** (the decoded plan reaches the goal): its plan cost;
//! - **invalid**: `penalty + unsatisfied goal facts + (1 - executed fraction)`, where the
//!   penalty exceeds every valid plan's cost. Invalid chains are ranked by how close they get
//!   instead of all scoring the same (Jakob's rule against flat penalties; EvoPlan ranks
//!   invalid plans by the fraction that executes).
//!
//! ```
//! use evolutionary_planning::sas::{Config, solve};
//! use evolutionary_planning::{Limits, SplitMix64};
//! use sas_plus::{Fact, Operator, State, Task, Variable, Effect};
//!
//! // A counter 0 -> 1 -> 2 with a goal of 2.
//! let step = |from: u32| Operator {
//!     name: format!("inc{from}"),
//!     pre: vec![Fact::new(0, from)],
//!     eff: vec![Effect::new(Fact::new(0, from + 1))],
//!     cost: 1,
//! };
//! let task = Task {
//!     variables: vec![Variable::new("x", 3)],
//!     operators: vec![step(0), step(1)],
//!     initial: State(vec![0]),
//!     goal: vec![Fact::new(0, 2)],
//!     axioms: vec![],
//! };
//! let result = solve(&task, &Config::default(), &mut Limits::evaluations(2_000), &mut SplitMix64::new(1));
//! assert_eq!(result.plan, Some(vec![0, 1]));
//! assert_eq!(task.validate_plan(&[0, 1]), result.cost);
//! ```

use alloc::vec::Vec;

use rand_core::Rng;
use sas_plus::{OpId, State, Task};

use crate::ga::{Config as GaConfig, Ga, Operators, Scheme};
use crate::operators::chain::{
    CutAndSplice, DeleteGene, GeneShift, InsertGene, ReplaceGene, SegmentInversion,
};
use crate::operators::selection::Tournament;
use crate::operators::{Initializer, Mutation, Repair, mix};
use crate::rng::below;
use crate::{Budget, Objective, Outcome, Stats};

/// How a chain is interpreted when an operator is not applicable.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Decoding {
    /// Phenotypic repair: inapplicable operators are skipped. The default; it lets order
    /// changes spread over several offspring (Jakob 2021, section 4.1.4).
    #[default]
    Skip,
    /// Execution stops at the first inapplicable operator (EvoPlan's validated prefix).
    Strict,
}

/// The result of simulating a chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decoded {
    /// The executed operators, up to and including the one that reached the goal.
    pub plan: Vec<OpId>,
    /// The cost of `plan`.
    pub cost: u64,
    /// Whether `plan` reaches the goal.
    pub reaches_goal: bool,
    /// How many genes were looked at before decoding stopped (the goal, the end of the chain,
    /// or the first inapplicable gene under [`Decoding::Strict`]).
    pub inspected: usize,
    /// The number of goal facts that do not hold in the final state.
    pub unsatisfied_goals: usize,
}

/// A SAS+ task as an [`Objective`] over operator chains.
#[derive(Debug, Clone, Copy)]
pub struct PlanObjective<'t> {
    /// The task.
    task: &'t Task,
    /// The decoding rule.
    decoding: Decoding,
    /// The cost added to every invalid chain; above any valid plan's cost.
    penalty: f64,
}

impl<'t> PlanObjective<'t> {
    /// Scores chains of at most `max_len` operators of `task`.
    #[must_use]
    pub fn new(task: &'t Task, decoding: Decoding, max_len: usize) -> Self {
        let max_cost = task
            .operators
            .iter()
            .map(|o| u64::from(o.cost))
            .max()
            .unwrap_or(0);
        #[expect(
            clippy::cast_precision_loss,
            reason = "plan cost bounds far below 2^52"
        )]
        let penalty = (max_cost.saturating_mul(max_len as u64) + 1) as f64;

        Self {
            task,
            decoding,
            penalty,
        }
    }

    /// The penalty added to invalid chains.
    #[must_use]
    pub fn penalty(&self) -> f64 {
        self.penalty
    }

    /// Simulates `chain` from the initial state.
    #[must_use]
    pub fn decode(&self, chain: &[OpId]) -> Decoded {
        let task = self.task;
        let mut state = task.initial_state();
        let mut plan = Vec::new();
        let mut cost = 0;
        let mut inspected = 0;
        let mut reaches_goal = task.is_goal(&state);
        for &op in chain {
            if reaches_goal {
                break;
            }

            inspected += 1;
            if op >= task.operators.len() || !task.is_applicable(op, &state) {
                if self.decoding == Decoding::Strict {
                    break;
                }
                continue;
            }

            state = task.apply(op, &state);
            cost += u64::from(task.operators[op].cost);
            plan.push(op);
            reaches_goal = task.is_goal(&state);
        }

        Decoded {
            unsatisfied_goals: unsatisfied(task, &state),
            plan,
            cost,
            reaches_goal,
            inspected,
        }
    }

    /// The cost of a decoded chain of `len` genes: plan cost if valid, otherwise the
    /// penalty plus the distance terms.
    #[must_use]
    pub fn score(&self, decoded: &Decoded, len: usize) -> f64 {
        if decoded.reaches_goal {
            #[expect(clippy::cast_precision_loss, reason = "plan costs far below 2^52")]
            return decoded.cost as f64;
        }

        #[expect(clippy::cast_precision_loss, reason = "counts far below 2^52")]
        let executed = if len == 0 {
            0.0
        } else {
            decoded.plan.len() as f64 / len as f64
        };
        #[expect(clippy::cast_precision_loss, reason = "counts far below 2^52")]
        let unsatisfied = decoded.unsatisfied_goals as f64;

        self.penalty + unsatisfied + (1.0 - executed)
    }
}

impl Objective<Vec<OpId>> for PlanObjective<'_> {
    fn evaluate(&mut self, genome: &Vec<OpId>) -> f64 {
        self.score(&self.decode(genome), genome.len())
    }
}

/// The goal facts that do not hold in `state`.
fn unsatisfied(task: &Task, state: &State) -> usize {
    task.goal
        .iter()
        .filter(|f| state.0[f.var] != f.value)
        .count()
}

/// Uniform random operator ids of a task: the gene sampler of the chain operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RandomOperator {
    /// The number of operators of the task, at least `1`.
    pub count: usize,
}

impl Initializer<OpId> for RandomOperator {
    fn initialize<R: Rng + ?Sized>(&mut self, rng: &mut R) -> OpId {
        below(rng, self.count)
    }
}

/// Random walks of applicable operators from the initial state, of uniform length in
/// `1..=max_len` (shorter if a dead end or the goal is reached): an initial population of
/// executable chains.
#[derive(Debug, Clone, Copy)]
pub struct ApplicableWalk<'t> {
    /// The task.
    pub task: &'t Task,
    /// The longest walk.
    pub max_len: usize,
}

impl Initializer<Vec<OpId>> for ApplicableWalk<'_> {
    fn initialize<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Vec<OpId> {
        let task = self.task;
        let len = 1 + below(rng, self.max_len.max(1));
        let mut state = task.initial_state();
        let mut walk = Vec::with_capacity(len);
        while walk.len() < len && !task.is_goal(&state) {
            let applicable: Vec<OpId> = task.applicable_operators(&state).collect();
            if applicable.is_empty() {
                break;
            }

            let op = applicable[below(rng, applicable.len())];
            state = task.apply(op, &state);
            walk.push(op);
        }

        walk
    }
}

/// Whether and how chains are repaired genotypically.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RepairMode {
    /// No genotypic repair: chains keep their inapplicable genes, and only the decoding
    /// skips them (phenotypic repair; the default, following Jakob's advice that genotypic
    /// repair freezes order changes that need several steps).
    #[default]
    Off,
    /// Write the executed plan back: inapplicable genes are dropped and the chain is cut
    /// after the goal.
    Drop,
    /// Like `Drop`, but each inapplicable gene is replaced by a uniformly chosen applicable
    /// operator.
    Replace,
}

/// Genotypic repair of operator chains, see [`RepairMode`].
#[derive(Debug, Clone, Copy)]
pub struct PlanRepair<'t> {
    /// The task.
    pub task: &'t Task,
    /// What the repair does.
    pub mode: RepairMode,
}

impl Repair<Vec<OpId>> for PlanRepair<'_> {
    fn repair<R: Rng + ?Sized>(&mut self, genome: &mut Vec<OpId>, rng: &mut R) {
        if self.mode == RepairMode::Off {
            return;
        }

        let task = self.task;
        let mut state = task.initial_state();
        let mut repaired = Vec::with_capacity(genome.len());
        for &op in genome.iter() {
            if task.is_goal(&state) {
                break;
            }

            let op = if op < task.operators.len() && task.is_applicable(op, &state) {
                op
            } else if self.mode == RepairMode::Replace {
                let applicable: Vec<OpId> = task.applicable_operators(&state).collect();
                if applicable.is_empty() {
                    continue;
                }
                applicable[below(rng, applicable.len())]
            } else {
                continue;
            };

            state = task.apply(op, &state);
            repaired.push(op);
        }

        *genome = repaired;
    }
}

/// Parameters of [`solve`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// The GA population size.
    pub population_size: usize,
    /// The longest chain.
    pub max_len: usize,
    /// The decoding rule.
    pub decoding: Decoding,
    /// The genotypic repair applied to every child.
    pub repair: RepairMode,
}

impl Default for Config {
    /// 60 chains of at most 40 operators, phenotypic skipping, no genotypic repair.
    fn default() -> Self {
        Self {
            population_size: 60,
            max_len: 40,
            decoding: Decoding::Skip,
            repair: RepairMode::Off,
        }
    }
}

/// The result of [`solve`].
#[derive(Debug, Clone, PartialEq)]
pub struct PlanOutcome {
    /// The best valid plan found, decoded (only executed operators); `None` if no chain
    /// reached the goal.
    pub plan: Option<Vec<OpId>>,
    /// Its cost, equal to `task.validate_plan(plan)`.
    pub cost: Option<u64>,
    /// The raw GA outcome over chains.
    pub outcome: Outcome<Vec<OpId>>,
}

/// Searches for a cheap plan with a steady-state GA over GLEAM chains: applicable random
/// walks as the initial population, cut-and-splice crossover and a mix of gene insertion,
/// deletion, replacement, shift and segment inversion.
///
/// # Panics
///
/// Panics if the task has no operators.
pub fn solve<B, R>(task: &Task, config: &Config, budget: &mut B, rng: &mut R) -> PlanOutcome
where
    B: Budget<Stats> + ?Sized,
    R: Rng + ?Sized,
{
    assert!(
        !task.operators.is_empty(),
        "a task without operators has no plans to evolve"
    );

    let mutation = chain_mutation(task.operators.len(), config.max_len);
    let mut ga = Ga::new(
        GaConfig {
            population_size: config.population_size,
            elitism: 2,
            crossover_rate: 0.6,
            mutation_rate: 1.0,
            scheme: Scheme::SteadyState {
                only_if_better: false,
            },
        },
        Operators {
            init: ApplicableWalk {
                task,
                max_len: config.max_len,
            },
            selection: Tournament { size: 3 },
            crossover: CutAndSplice {
                max_len: config.max_len,
            },
            mutation,
            repair: PlanRepair {
                task,
                mode: config.repair,
            },
        },
    );
    let mut objective = PlanObjective::new(task, config.decoding, config.max_len);
    let outcome = ga.run(&mut objective, budget, &mut (), rng);
    let decoded = outcome.best_genome().map(|chain| objective.decode(chain));
    let (plan, cost) = match decoded {
        Some(d) if d.reaches_goal => (Some(d.plan), Some(d.cost)),
        _ => (None, None),
    };

    PlanOutcome {
        plan,
        cost,
        outcome,
    }
}

/// GLEAM's operator mix for plan chains: gene insertion and deletion, replacement, gene shift
/// and segment inversion, weighted towards insertion and replacement.
fn chain_mutation(operators: usize, max_len: usize) -> impl Mutation<Vec<OpId>> {
    let genes = RandomOperator { count: operators };

    mix(
        mix(
            InsertGene { genes, max_len },
            2.0,
            DeleteGene { min_len: 0 },
            1.0,
        ),
        3.0,
        mix(
            ReplaceGene { genes },
            2.0,
            mix(
                GeneShift,
                1.0,
                SegmentInversion {
                    max_segment: Some(4),
                },
                1.0,
            ),
            2.0,
        ),
        4.0,
    )
}

#[cfg(test)]
mod tests {
    use sas_plus::random::{RandomTaskParams, random_task};

    use super::*;
    use crate::{Limits, SplitMix64};

    /// The outcome counts of [`solve`] over 40 random tasks with conditional effects and
    /// axioms: `(solvable, found, optimal)`. Every reported plan must pass `validate_plan` with
    /// the reported cost and never beat the optimum from `perfect_heuristic`.
    fn validate_over_random_tasks(config: &Config) -> (usize, usize, usize) {
        let (mut solvable, mut found, mut optimal) = (0, 0, 0);
        for seed in 0..40 {
            let params = RandomTaskParams {
                num_vars: 4,
                max_domain: 3,
                num_ops: 12,
                max_cost: 4,
                max_conditions: 1,
                num_derived: usize::from(seed % 2 == 1),
            };
            let task = random_task(seed, params);
            let Some(best) = task.perfect_heuristic()(&task.initial_state()) else {
                continue;
            };

            solvable += 1;
            let result = solve(
                &task,
                config,
                &mut Limits::evaluations(6_000),
                &mut SplitMix64::new(seed),
            );
            if let (Some(plan), Some(cost)) = (&result.plan, result.cost) {
                assert_eq!(
                    task.validate_plan(plan),
                    Some(cost),
                    "{config:?}, seed {seed}"
                );
                assert!(
                    cost >= best,
                    "{config:?}, seed {seed}: {cost} < optimum {best}"
                );
                found += 1;
                optimal += usize::from(cost == best);
            }
        }

        (solvable, found, optimal)
    }

    /// Every decoding and repair combination reports only valid plans; the default finds a
    /// plan for 90 % of the solvable tasks and an optimal one for half.
    #[test]
    fn plans_validate_against_sas_plus() {
        for decoding in [Decoding::Skip, Decoding::Strict] {
            for repair in [RepairMode::Off, RepairMode::Drop, RepairMode::Replace] {
                let config = Config {
                    decoding,
                    repair,
                    ..Config::default()
                };
                let (solvable, found, optimal) = validate_over_random_tasks(&config);
                assert!(solvable >= 10, "only {solvable} solvable tasks");
                if config == Config::default() {
                    assert!(found * 10 >= solvable * 9, "found {found} of {solvable}");
                    assert!(optimal * 2 >= solvable, "optimal {optimal} of {solvable}");
                }
            }
        }
    }

    /// The cascade ranks invalid chains by unsatisfied goals and executed fraction, below any
    /// valid plan; strict decoding stops at the first inapplicable gene.
    #[test]
    fn fitness_cascade() {
        let task = random_task(3, RandomTaskParams::default());
        let objective = PlanObjective::new(&task, Decoding::Strict, 10);
        let decoded = objective.decode(&[usize::MAX, 0, 1]);

        assert_eq!(decoded.inspected, 1);
        assert_eq!(decoded.plan, Vec::<OpId>::new());
        let score = objective.score(&decoded, 3);
        assert!(score >= objective.penalty());
    }

    /// Genotypic repair produces executable chains.
    #[test]
    fn repair_makes_chains_executable() {
        let task = random_task(5, RandomTaskParams::default());
        let mut rng = SplitMix64::new(2);
        let mut repair = PlanRepair {
            task: &task,
            mode: RepairMode::Replace,
        };
        let objective = PlanObjective::new(&task, Decoding::Strict, 20);
        for _ in 0..50 {
            let mut chain: Vec<OpId> = (0..10)
                .map(|_| below(&mut rng, task.operators.len()))
                .collect();
            repair.repair(&mut chain, &mut rng);
            let decoded = objective.decode(&chain);
            assert_eq!(decoded.plan, chain[..decoded.plan.len()]);
            assert_eq!(decoded.plan.len(), chain.len());
        }
    }
}
