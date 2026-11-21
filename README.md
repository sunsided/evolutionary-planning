# evolutionary-planning

Evolutionary planning: genetic algorithms, evolution strategies and multi-objective evolutionary search over plans, paths and trajectories.

`no_std` + `alloc`, `forbid(unsafe_code)`. Every objective is minimised, all randomness comes
from a caller-provided `rand_core::Rng` (the crate ships `SplitMix64`), and every run takes a
budget that is asked before each evaluation, so seeded runs are reproducible and limits are
exact. See [`docs/design.md`](docs/design.md) for the architecture, the design rules and the
paper table.

## Scope and status

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

## Status

Pre-release. Version 0.0.0 reserves the name; the API will change.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
