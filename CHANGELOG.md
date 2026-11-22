# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1](https://github.com/sunsided/evolutionary-planning/compare/v0.1.0...v0.1.1) - 2026-11-22

### Changed

- Link the crate to its page on the planning and search site

## [0.1.0](https://github.com/sunsided/evolutionary-planning/releases/tag/v0.1.0) - 2026-11-21

### Added

- Add range property tests for the random helpers and real operators
- Add the strict lint table, clippy.toml and rustfmt.toml
- Add dormant release automation with release-plz
- Add CONTRIBUTING.md
- Add git hooks and Copilot review instructions
- Add README badges, installation and license sections
- Add fuzz targets and Pareto property tests
- Add REUSE licensing metadata
- Add cargo-deny, Dependabot and typos
- Add the Codecov configuration
- Add the Taskfile and CI workflows
- Add tree-based genetic programming over an operator language
- Add the intersection example and more benchmarks; cap function length
- Add integration tests and correct the design document
- Add benchmarks, speed up hot paths and update the README
- Add an evolutionary programmer that evolves path-planning GAs
- Add the multi-robot intersection benchmark after Parque (2024)
- Add memetic hybrids with HyGLEAM adaptation and GLEAM's diffusion model
- Add online trajectory optimisation with warm-started CMA-ES
- Add finite-trace STL robustness, an STL shield and threshold fitting
- Add genetic path planning on graphs and grids
- Add rolling-horizon evolutionary planning with RMHC
- Add MAP-Elites and iso+line variation
- Add the SAS+ plan GA with plan repair (feature sas-plus)
- Add Pareto tools, NSGA-II, NSGA-III and MOEA/D
- Add differential evolution, a warm-start library and RADES
- Add CMA-ES exactly per Hansen's tutorial, with IPOP restarts
- Add the core, the operator library, GA, RMHC and evolution strategies
- Add the design document and the dependencies

### Changed

- Treat NaN objectives as worst in non_dominated_sort
- Set the crate version to 0.1.0 for the first release
- Describe the released scope and audience in the README
- Pin taiki-e/install-action to v2.90.7
- Render mermaid on docs.rs and doctest the README
- Depend on search-core and sas-plus from crates.io
- Require search-core 0.3
- Require search-core 0.2
- Require search-core 0.1
- Accept search-core 0.0 and 0.1 during the 0.1.0 release
- Tighten RHEA budgets, trajectory carry-over and HyGLEAM adaptation
- Make genotypic SAS+ repair optional and phenotypic by default
- Price time in the intersection objective
- Document the module map, deviations and open questions
- Reserve the crate name with a 0.0.0 placeholder

### Fixed

- Fix full elitism, add the missed OnePlusOne doctest, document the crossing optimum

### Removed

- Drop pre-release wording from the docs and CI comments
