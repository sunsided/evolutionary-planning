//! Benchmark problems with known optima, for tests, benches and examples.
//!
//! - [`single`]: sphere, ellipsoid, Rosenbrock, Rastrigin, OneMax, LeadingOnes and tours.
//! - [`multi`]: Schaffer, ZDT1–3, DTLZ1–2, LOTZ and a bi-objective knapsack.
//! - [`intersection`]: the multi-robot intersection benchmark of Parque (2024).

pub mod intersection;
pub mod multi;
pub mod single;
