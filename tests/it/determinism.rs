//! Every algorithm returns the same result for the same seed (and the run is not trivial).

use evolutionary_planning::cmaes::{Config as CmaConfig, Ipop};
use evolutionary_planning::diffusion::{Config as DiffusionConfig, Diffusion};
use evolutionary_planning::ga::{Config as GaConfig, Ga, Operators};
use evolutionary_planning::graph_path::{self, Grid};
use evolutionary_planning::map_elites::{Config as QdConfig, Grid as QdGrid, MapElites};
use evolutionary_planning::memetic::{Memetic, Rosenbrock};
use evolutionary_planning::moead::{Config as MoeadConfig, Moead};
use evolutionary_planning::nsga3::{Config as Nsga3Config, Nsga3};
use evolutionary_planning::operators::real::{Bounds, Clamp, Gaussian, Polynomial, Sbx};
use evolutionary_planning::operators::selection::LinearRanking;
use evolutionary_planning::operators::{NoCrossover, NoRepair, Variation, initializer_fn};
use evolutionary_planning::pareto::das_dennis;
use evolutionary_planning::problems::multi::{Dtlz, DtlzKind};
use evolutionary_planning::problems::single::{rastrigin, rosenbrock, sphere};
use evolutionary_planning::programmer::{Instance, MetaConfig, Programmer};
use evolutionary_planning::rhea::{Config as RheaConfig, ForwardModel, RollingHorizon, Step};
use evolutionary_planning::trajectory::{Config as TrajectoryConfig, OnlineOptimizer};
use evolutionary_planning::{Limits, SplitMix64, below};
use rand_core::Rng;

/// Real-vector variation operators on `[-2, 2]^n`.
fn real_variation(n: usize) -> Variation<Bounds, Sbx, Polynomial, NoRepair> {
    let bounds = Bounds::uniform(n, -2.0, 2.0);
    Variation {
        init: bounds.clone(),
        crossover: Sbx {
            bounds: bounds.clone(),
            eta: 15.0,
        },
        mutation: Polynomial {
            bounds,
            eta: 20.0,
            rate: None,
        },
        repair: NoRepair,
    }
}

/// Real-vector GA operators on `[-2, 2]^n`.
fn real_operators(n: usize) -> Operators<Bounds, LinearRanking, Sbx, Polynomial, NoRepair> {
    let v = real_variation(n);
    Operators {
        init: v.init,
        selection: LinearRanking::default(),
        crossover: v.crossover,
        mutation: v.mutation,
        repair: v.repair,
    }
}

/// NSGA-III and MOEA/D.
#[test]
fn many_objective() {
    let run_nsga3 = || {
        let references = das_dennis(3, 4);
        let mut nsga3 = Nsga3::new(
            Nsga3Config::for_references(&references),
            references,
            real_variation(5),
        );
        let mut problem = Dtlz {
            kind: DtlzKind::Dtlz2,
            objectives: 3,
        };
        nsga3.run(
            &mut problem,
            &mut Limits::evaluations(1_500),
            &mut (),
            &mut SplitMix64::new(1),
        )
    };
    let run_moead = || {
        let mut moead = Moead::new(MoeadConfig::default(), das_dennis(3, 6), real_variation(5));
        let mut problem = Dtlz {
            kind: DtlzKind::Dtlz2,
            objectives: 3,
        };
        moead.run(
            &mut problem,
            &mut Limits::evaluations(1_500),
            &mut (),
            &mut SplitMix64::new(1),
        )
    };

    assert_eq!(run_nsga3(), run_nsga3());
    assert_eq!(run_moead(), run_moead());
    assert_eq!(run_moead().stats.evaluations, 1_500);
}

/// MAP-Elites.
#[test]
fn map_elites() {
    let run = || {
        let bounds = Bounds::uniform(3, -1.0, 1.0);
        let mut qd = MapElites::new(
            QdConfig::default(),
            Variation {
                init: bounds.clone(),
                crossover: NoCrossover,
                mutation: Gaussian {
                    sigma: 0.1,
                    rate: None,
                },
                repair: Clamp(bounds),
            },
        );
        let mut problem = |x: &Vec<f64>, d: &mut [f64]| {
            d.copy_from_slice(&x[..2]);
            x[2] * x[2]
        };
        qd.run(
            &mut problem,
            QdGrid::new(vec![-1.0; 2], vec![1.0; 2], vec![5, 5]),
            &mut Limits::evaluations(800),
            &mut (),
            &mut SplitMix64::new(2),
        )
    };

    assert_eq!(run(), run());
}

/// A corridor for RHEA.
struct Corridor;

impl ForwardModel for Corridor {
    type State = i32;
    type Action = i32;

    fn step<R: Rng + ?Sized>(&mut self, state: &i32, action: &i32, _rng: &mut R) -> Step<i32> {
        let next = state + action;
        Step {
            state: next,
            cost: f64::from((8 - next).abs()),
            done: next == 8,
        }
    }
}

/// RHEA decisions, with RMHC and with a population.
#[test]
fn rhea() {
    for population_size in [1, 6] {
        let run = || {
            let actions = initializer_fn(|rng| i32::try_from(below(rng, 3)).expect("small") - 1);
            let mut planner = RollingHorizon::new(
                RheaConfig {
                    horizon: 10,
                    population_size,
                    ..RheaConfig::default()
                },
                actions,
            );
            let mut rng = SplitMix64::new(3);
            (0..4)
                .map(|s| planner.plan(&mut Corridor, &s, &mut rng))
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }
}

/// The diffusion model, plain and with HyGLEAM's adaptive local search.
#[test]
fn diffusion_and_memetic() {
    let plain = || {
        let mut objective = |x: &Vec<f64>| rastrigin(x);
        Diffusion::new(DiffusionConfig::default(), real_operators(3)).run(
            &mut objective,
            &mut Limits::evaluations(2_000),
            &mut (),
            &mut SplitMix64::new(4),
        )
    };
    let hybrid = || {
        let mut objective = |x: &Vec<f64>| rosenbrock(x);
        let mut memetic = Memetic::adaptive(Rosenbrock::default());
        Diffusion::new(DiffusionConfig::default(), real_operators(3)).run_memetic(
            &mut memetic,
            &mut objective,
            &mut Limits::evaluations(2_000),
            &mut (),
            &mut SplitMix64::new(4),
        )
    };
    let memetic_ga = || {
        let mut objective = |x: &Vec<f64>| rosenbrock(x);
        let mut memetic = Memetic::adaptive(Rosenbrock::default());
        Ga::new(
            GaConfig {
                population_size: 10,
                ..GaConfig::default()
            },
            real_operators(3),
        )
        .run_memetic(
            &mut memetic,
            &mut objective,
            &mut Limits::evaluations(2_000),
            &mut (),
            &mut SplitMix64::new(4),
        )
    };

    assert_eq!(plain(), plain());
    assert_eq!(hybrid(), hybrid());
    assert_eq!(memetic_ga(), memetic_ga());
}

/// IPOP-CMA-ES across restarts.
#[test]
fn ipop() {
    let run = || {
        let mut ipop = Ipop::new(CmaConfig::default(), Bounds::uniform(3, -5.0, 5.0));
        let mut objective = |x: &Vec<f64>| rastrigin(x);
        ipop.run(
            &mut objective,
            &mut Limits::evaluations(5_000),
            &mut (),
            &mut SplitMix64::new(5),
        )
    };

    assert_eq!(run(), run());
}

/// Graph paths and the evolutionary programmer.
#[test]
fn graph_paths_and_programmer() {
    let mut grid = Grid::open(10, 8, true);
    for y in 0..6 {
        grid.block(5, y);
    }
    let (start, goal) = (grid.node(0, 0), grid.node(9, 0));
    let path = || {
        graph_path::solve(
            &grid,
            start,
            goal,
            &graph_path::Config::default(),
            &mut Limits::evaluations(2_000),
            &mut SplitMix64::new(6),
        )
    };
    let program = || {
        let mut programmer = Programmer::new(vec![Instance::new(&grid, start, goal)], 200);
        programmer.evolve(
            &MetaConfig {
                population_size: 6,
                evaluations: 24,
            },
            &mut SplitMix64::new(6),
        )
    };

    assert_eq!(path(), path());
    assert_eq!(program(), program());
}

/// Online trajectory re-planning over several periods.
#[test]
fn trajectory() {
    let run = || {
        let mut planner = OnlineOptimizer::new(TrajectoryConfig {
            steps: 8,
            evaluations: 100,
            ..TrajectoryConfig::default()
        });
        let mut rng = SplitMix64::new(7);
        (0..5)
            .map(|k| planner.replan(&mut |u: &Vec<f64>| sphere(u) + f64::from(k), None, &mut rng))
            .collect::<Vec<_>>()
    };

    assert_eq!(run(), run());
}

/// The SAS+ plan GA.
#[cfg(feature = "sas-plus")]
#[test]
fn sas_plans() {
    use evolutionary_planning::sas::{Config, solve};
    use sas_plus::random::{RandomTaskParams, random_task};

    let task = random_task(3, RandomTaskParams::default());
    let run = || {
        solve(
            &task,
            &Config::default(),
            &mut Limits::evaluations(1_000),
            &mut SplitMix64::new(8),
        )
    };

    assert_eq!(run(), run());
}
