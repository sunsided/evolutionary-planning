# evolutionary-planning

[![crates.io](https://img.shields.io/crates/v/evolutionary-planning.svg)](https://crates.io/crates/evolutionary-planning)
[![docs.rs](https://img.shields.io/docsrs/evolutionary-planning)](https://docs.rs/evolutionary-planning)
[![CI](https://github.com/sunsided/evolutionary-planning/actions/workflows/ci.yml/badge.svg)](https://github.com/sunsided/evolutionary-planning/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sunsided/evolutionary-planning/graph/badge.svg)](https://codecov.io/gh/sunsided/evolutionary-planning)
[![MSRV](https://img.shields.io/crates/msrv/evolutionary-planning)](https://crates.io/crates/evolutionary-planning)
[![License](https://img.shields.io/crates/l/evolutionary-planning)](https://github.com/sunsided/evolutionary-planning#license)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)

Evolutionary planning: genetic algorithms, evolution strategies and multi-objective evolutionary search over plans, paths and trajectories. The crate covers the classic toolbox (GA, ES, CMA-ES, differential evolution, NSGA-II, NSGA-III, MOEA/D, MAP-Elites) and connects it to planning: rolling-horizon evolution over a forward model, STL robustness as objective and shield, trajectory re-planning, GA path planning on graphs and grids, and a GA over SAS+ plans with plan repair.

`no_std` + `alloc`, `forbid(unsafe_code)`. Every objective is minimised, all randomness comes
from a caller-provided `rand_core::Rng` (the crate ships `SplitMix64`), and every run takes a
budget that is asked before each evaluation, so seeded runs are reproducible and limits are
exact. CMA-ES follows Hansen's tutorial exactly and is tested against its Table 1. See [`docs/design.md`](https://github.com/sunsided/evolutionary-planning/blob/main/docs/design.md) for the architecture, the design rules and the
paper table.

Part of the [planning and search crates](https://sunsided.github.io/planning-and-search/); see also this crate's [overview page](https://sunsided.github.io/planning-and-search/crates/evolutionary-planning/).

## Who this is for

- Robotics and control engineers who optimise trajectories or motion parameters online, where the objective is a black box or a simulation and gradients are not available.
- Game AI and simulation developers who plan with rolling-horizon evolution over their own forward model, or who need a robustness score for temporal-logic (STL) requirements.
- People with several competing objectives (time, energy, risk) who want a Pareto front from NSGA-II, NSGA-III or MOEA/D instead of a hand-tuned weighted sum.
- Researchers who want reproducible, budget-exact runs of the reference algorithms on embedded or `no_std` targets.
- For optimal shortest paths with a good heuristic, use [`heuristic-search`](https://github.com/sunsided/heuristic-search); for exact Pareto-optimal paths on graphs, [`multiobjective-search`](https://github.com/sunsided/multiobjective-search); for sampling-based motion planning (RRT, PRM), [`sampling-planning`](https://github.com/sunsided/sampling-planning); and for coordinating many agents on a graph, [`mapf-search`](https://github.com/sunsided/mapf-search).

## Installation

```sh
cargo add evolutionary-planning
```

Enable the `sas-plus` feature for the GA over SAS+ plans.

## Scope

Implemented:

- **Core**: operator traits (initialisation, selection, crossover, mutation, repair) with
  closure adapters and GLEAM-style operator mixes; representations for real vectors (SBX,
  BLX-α, polynomial, Gaussian, iso+line, Latin hypercube), bit and integer strings,
  permutations (PMX, OX, CX, ERX, swap, insertion, inversion, scramble) and GLEAM action
  chains (variable-length, typed, parameterised genes with gene- and chain-level operators);
  budgets, observers, benchmark problems.
- **Single objective**: generational and steady-state GA with elitism; GLEAM's ring
  diffusion model; random-mutation hill climbing; (μ/ρ,λ)- and (μ/ρ+λ)-ES with
  self-adaptation and the (1+1)-ES with the 1/5th rule; **CMA-ES exactly per Hansen's
  tutorial** (active weights, lazy eigendecomposition, all Appendix B.3 termination criteria,
  ask/tell) with IPOP restarts; differential evolution with Latin hypercube initialisation and
  a warm-start library; RADES.
- **Memetic**: hill climbing and Rosenbrock local search, Lamarckian or Baldwinian, with
  HyGLEAM's adaptive frequency and intensity; HyGLEAM itself (diffusion model plus adaptive
  local search).
- **Multi objective**: constrained domination, fast non-dominated sorting, crowding distance,
  bounded Pareto archives, exact hypervolume, Das-Dennis reference points, NSGA-II, NSGA-III
  and MOEA/D (weighted sum, Tchebycheff, PBI).
- **Quality diversity**: MAP-Elites with a grid archive.
- **Planning bridges**: rolling-horizon evolutionary planning (RHEA, RMHC) over a caller's
  forward model with shift buffering; finite-trace STL robustness as objective, constraint
  and shield, plus threshold fitting from demonstrations; online trajectory optimisation with
  warm-started CMA-ES; GA path planning on graphs and grids; GA over SAS+ plans with plan
  repair, validated against `sas_plus::Task::validate_plan` (feature `sas-plus`); a
  multi-robot intersection benchmark (`cargo run --release --example intersection`); an
  evolutionary programmer that evolves path-planning GAs; tree-based genetic programming over
  a caller's operator language (for example reactive navigation policies).

Not implemented: parallel evaluation (use the ask/tell CMA-ES interface), the neural and LLM
components of the source papers.

## Example

```rust
use evolutionary_planning::cmaes::{CmaEs, Config};
use evolutionary_planning::problems::single::rosenbrock;
use evolutionary_planning::{Limits, SplitMix64};

let mut cma = CmaEs::new(vec![0.0; 4], 0.5, Config::default());
let mut objective = |x: &Vec<f64>| rosenbrock(x);
let outcome = cma.minimize(&mut objective, &mut Limits::evaluations(20_000).with_target(1e-10), &mut (), &mut SplitMix64::new(1));
assert!(outcome.best_cost() <= 1e-10);
```

## Features

- `sas-plus`: the `sas` module, a GA over SAS+ action sequences with plan repair.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/sunsided/evolutionary-planning/blob/main/LICENSE-APACHE))
- MIT license ([LICENSE-MIT](https://github.com/sunsided/evolutionary-planning/blob/main/LICENSE-MIT))

at your option.

The examples in [`examples/`](https://github.com/sunsided/evolutionary-planning/tree/main/examples) are public domain
([CC0-1.0](https://github.com/sunsided/evolutionary-planning/blob/main/LICENSES/CC0-1.0.txt)): copy them into anything.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in this work by you, as defined in the Apache-2.0 license, shall be dual licensed as above,
without any additional terms or conditions.
