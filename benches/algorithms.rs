//! Benchmarks of the hot paths: a CMA-ES generation and a full CMA-ES run, GA, DE and NSGA-II
//! runs at fixed budgets, NSGA-III and MOEA/D on DTLZ2, non-dominated sorting and
//! hypervolume, MAP-Elites insertion, RHEA decisions, STL robustness, an online trajectory
//! replan, graph path planning, the SAS+ plan GA (feature `sas-plus`) and the intersection
//! objective.
//!
//! Budgets are fixed, so each sample measures the same amount of work; results are seeded.

use std::hint::black_box;
use std::time::Duration;

use criterion::Criterion;
use evolutionary_planning::cmaes::{CmaEs, Config as CmaConfig};
use evolutionary_planning::de::{Config as DeConfig, De};
use evolutionary_planning::ga::{Config as GaConfig, Ga, Operators};
use evolutionary_planning::map_elites::{Config as QdConfig, Grid as QdGrid, MapElites};
use evolutionary_planning::moead::{Config as MoeadConfig, Moead};
use evolutionary_planning::nsga2::{Config as Nsga2Config, Nsga2};
use evolutionary_planning::nsga3::{Config as Nsga3Config, Nsga3};
use evolutionary_planning::operators::discrete::{BitFlip, RandomBits};
use evolutionary_planning::operators::real::{Bounds, Clamp, Gaussian, Polynomial, Sbx};
use evolutionary_planning::operators::selection::Tournament;
use evolutionary_planning::operators::sequence::UniformCrossover;
use evolutionary_planning::operators::{NoCrossover, NoRepair, Variation, initializer_fn};
use evolutionary_planning::pareto::{das_dennis, hypervolume, non_dominated_sort};
use evolutionary_planning::problems::intersection::Intersection;
use evolutionary_planning::problems::multi::{Dtlz, DtlzKind, Zdt};
use evolutionary_planning::problems::single::{one_max, rosenbrock, sphere};
use evolutionary_planning::rhea::{Config as RheaConfig, ForwardModel, RollingHorizon, Step};
use evolutionary_planning::stl::Formula;
use evolutionary_planning::trajectory::{Config as TrajectoryConfig, OnlineOptimizer};
use evolutionary_planning::{Limits, Objective, SplitMix64, below, graph_path, unit_f64};
use rand_core::Rng;

/// Runs every benchmark with criterion's command-line flags.
fn main() {
    let mut criterion = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(2))
        .configure_from_args();

    bench_cmaes(&mut criterion);
    bench_population_methods(&mut criterion);
    bench_de(&mut criterion);
    bench_map_elites(&mut criterion);
    bench_pareto(&mut criterion);
    bench_many_objective(&mut criterion);
    bench_pareto_tools(&mut criterion);
    bench_planning(&mut criterion);
    bench_stl(&mut criterion);
    bench_trajectory(&mut criterion);
    bench_graph(&mut criterion);
    bench_sas(&mut criterion);
    bench_intersection(&mut criterion);
    criterion.final_summary();
}

/// One CMA-ES generation in 20 and 50 dimensions, and a full run on the 10D sphere.
fn bench_cmaes(c: &mut Criterion) {
    for n in [20, 50] {
        c.bench_function(&format!("cmaes/generation/{n}d"), |b| {
            let mut cma = CmaEs::new(vec![1.0; n], 0.5, CmaConfig::default());
            let mut rng = SplitMix64::new(1);
            b.iter(|| {
                let costs: Vec<f64> = cma.ask(&mut rng).iter().map(|x| sphere(x)).collect();
                black_box(cma.tell(&costs));
            });
        });
    }

    c.bench_function("cmaes/sphere-10d-to-1e-10", |b| {
        b.iter(|| {
            let mut cma = CmaEs::new(vec![1.0; 10], 0.5, CmaConfig::default());
            let mut objective = |x: &Vec<f64>| sphere(x);
            black_box(cma.minimize(
                &mut objective,
                &mut Limits::evaluations(5_000).with_target(1e-10),
                &mut (),
                &mut SplitMix64::new(1),
            ))
        });
    });
}

/// The GA on OneMax with 5000 evaluations.
fn bench_population_methods(c: &mut Criterion) {
    c.bench_function("ga/one-max-64/5000", |b| {
        b.iter(|| {
            let mut ga = Ga::new(
                GaConfig::default(),
                Operators {
                    init: RandomBits { len: 64 },
                    selection: Tournament { size: 2 },
                    crossover: UniformCrossover::default(),
                    mutation: BitFlip::default(),
                    repair: NoRepair,
                },
            );
            let mut objective = |x: &Vec<bool>| one_max(x);
            black_box(ga.run(
                &mut objective,
                &mut Limits::evaluations(5_000),
                &mut (),
                &mut SplitMix64::new(1),
            ))
        });
    });
}

/// DE on Rosenbrock with 5000 evaluations.
fn bench_de(c: &mut Criterion) {
    c.bench_function("de/rosenbrock-5d/5000", |b| {
        b.iter(|| {
            let mut de = De::new(DeConfig::default(), Bounds::uniform(5, -2.0, 2.0));
            let mut objective = |x: &Vec<f64>| rosenbrock(x);
            black_box(de.run(
                &mut objective,
                &mut Limits::evaluations(5_000),
                &mut (),
                &mut SplitMix64::new(1),
            ))
        });
    });
}

/// MAP-Elites with 5000 evaluations.
fn bench_map_elites(c: &mut Criterion) {
    c.bench_function("map-elites/2d-grid/5000", |b| {
        let bounds = Bounds::uniform(4, -1.0, 1.0);
        b.iter(|| {
            let mut qd = MapElites::new(
                QdConfig::default(),
                Variation {
                    init: bounds.clone(),
                    crossover: NoCrossover,
                    mutation: Gaussian {
                        sigma: 0.1,
                        rate: None,
                    },
                    repair: Clamp(bounds.clone()),
                },
            );
            let mut problem = |x: &Vec<f64>, d: &mut [f64]| {
                d.copy_from_slice(&x[..2]);
                sphere(&x[2..])
            };
            let grid = QdGrid::new(vec![-1.0; 2], vec![1.0; 2], vec![20, 20]);
            black_box(qd.run(
                &mut problem,
                grid,
                &mut Limits::evaluations(5_000),
                &mut (),
                &mut SplitMix64::new(1),
            ))
        });
    });
}

/// NSGA-II on ZDT1 with 5000 evaluations.
fn bench_pareto(c: &mut Criterion) {
    c.bench_function("nsga2/zdt1-30d/5000", |b| {
        let bounds = Bounds::uniform(30, 0.0, 1.0);
        b.iter(|| {
            let mut nsga2 = Nsga2::new(
                Nsga2Config::default(),
                Variation {
                    init: bounds.clone(),
                    crossover: Sbx {
                        bounds: bounds.clone(),
                        eta: 15.0,
                    },
                    mutation: Polynomial {
                        bounds: bounds.clone(),
                        eta: 20.0,
                        rate: None,
                    },
                    repair: NoRepair,
                },
            );
            black_box(nsga2.run(
                &mut Zdt::Zdt1,
                &mut Limits::evaluations(5_000),
                &mut (),
                &mut SplitMix64::new(1),
            ))
        });
    });
}

/// NSGA-III and MOEA/D on three-objective DTLZ2 with 5000 evaluations.
fn bench_many_objective(c: &mut Criterion) {
    let bounds = Bounds::uniform(7, 0.0, 1.0);
    let variation = || Variation {
        init: bounds.clone(),
        crossover: Sbx {
            bounds: bounds.clone(),
            eta: 30.0,
        },
        mutation: Polynomial {
            bounds: bounds.clone(),
            eta: 20.0,
            rate: None,
        },
        repair: NoRepair,
    };
    c.bench_function("nsga3/dtlz2-3obj/5000", |b| {
        b.iter(|| {
            let references = das_dennis(3, 12);
            let mut nsga3 = Nsga3::new(
                Nsga3Config::for_references(&references),
                references,
                variation(),
            );
            let mut problem = Dtlz {
                kind: DtlzKind::Dtlz2,
                objectives: 3,
            };
            black_box(nsga3.run(
                &mut problem,
                &mut Limits::evaluations(5_000),
                &mut (),
                &mut SplitMix64::new(1),
            ))
        });
    });
    c.bench_function("moead/dtlz2-3obj/5000", |b| {
        b.iter(|| {
            let mut moead = Moead::new(MoeadConfig::default(), das_dennis(3, 12), variation());
            let mut problem = Dtlz {
                kind: DtlzKind::Dtlz2,
                objectives: 3,
            };
            black_box(moead.run(
                &mut problem,
                &mut Limits::evaluations(5_000),
                &mut (),
                &mut SplitMix64::new(1),
            ))
        });
    });
}

/// Non-dominated sorting of 400 points and 3D hypervolume of 60 points.
fn bench_pareto_tools(c: &mut Criterion) {
    let mut rng = SplitMix64::new(3);
    let points: Vec<Vec<f64>> = (0..400)
        .map(|_| vec![unit_f64(&mut rng), unit_f64(&mut rng), unit_f64(&mut rng)])
        .collect();
    let violations = vec![0.0; points.len()];
    c.bench_function("pareto/non-dominated-sort/400x3", |b| {
        b.iter(|| black_box(non_dominated_sort(&points, &violations)));
    });

    let sphere_front: Vec<Vec<f64>> = (0..60)
        .map(|_| {
            let v = [unit_f64(&mut rng), unit_f64(&mut rng), unit_f64(&mut rng)];
            let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            v.iter().map(|x| x / r).collect()
        })
        .collect();
    c.bench_function("pareto/hypervolume/60x3", |b| {
        b.iter(|| black_box(hypervolume(&sphere_front, &[1.1, 1.1, 1.1])));
    });
}

/// A corridor model for RHEA: actions -1, 0, +1; cost is the distance to cell 20.
struct Corridor;

impl ForwardModel for Corridor {
    type State = i32;
    type Action = i32;

    fn step<R: Rng + ?Sized>(&mut self, state: &i32, action: &i32, _rng: &mut R) -> Step<i32> {
        let next = state + action;
        Step {
            state: next,
            cost: f64::from((20 - next).abs()),
            done: next == 20,
        }
    }
}

/// One RMHC decision over a horizon of 20.
fn bench_planning(c: &mut Criterion) {
    c.bench_function("rhea/rmhc-decision/horizon-20", |b| {
        let actions = initializer_fn(|rng| i32::try_from(below(rng, 3)).expect("small") - 1);
        let mut planner = RollingHorizon::rmhc(
            RheaConfig {
                horizon: 20,
                ..RheaConfig::default()
            },
            actions,
        );
        let mut rng = SplitMix64::new(1);
        b.iter(|| black_box(planner.plan(&mut Corridor, &0, &mut rng)));
    });
}

/// STL robustness over a 1000-step trace.
fn bench_stl(c: &mut Criterion) {
    let mut rng = SplitMix64::new(5);
    let trace: Vec<[f64; 2]> = (0..1_000)
        .map(|_| [unit_f64(&mut rng), unit_f64(&mut rng)])
        .collect();
    let phi = Formula::always(
        0,
        50,
        Formula::or(vec![
            Formula::le(0, 0.9),
            Formula::eventually(0, 10, Formula::ge(1, 0.5)),
        ]),
    );
    c.bench_function("stl/robustness/1000-steps", |b| {
        b.iter(|| black_box(phi.signal(&trace, trace.len())));
    });
}

/// The path GA against Dijkstra on a 30 x 30 grid.
fn bench_graph(c: &mut Criterion) {
    let mut grid = graph_path::Grid::open(30, 30, true);
    for y in 0..25 {
        grid.block(10, y);
        grid.block(20, 29 - y);
    }
    let (start, goal) = (grid.node(0, 0), grid.node(29, 29));
    c.bench_function("graph-path/ga-30x30/5000", |b| {
        b.iter(|| {
            black_box(graph_path::solve(
                &grid,
                start,
                goal,
                &graph_path::Config::default(),
                &mut Limits::evaluations(5_000),
                &mut SplitMix64::new(1),
            ))
        });
    });
    c.bench_function("graph-path/dijkstra-30x30", |b| {
        b.iter(|| black_box(graph_path::dijkstra(&grid, start, goal)));
    });
}

/// One online trajectory replan of 20 steps with two controls.
fn bench_trajectory(c: &mut Criterion) {
    c.bench_function("trajectory/replan/20x2-300", |b| {
        let mut planner = OnlineOptimizer::new(TrajectoryConfig {
            steps: 20,
            controls: 2,
            ..TrajectoryConfig::default()
        });
        let mut rng = SplitMix64::new(1);
        b.iter(|| {
            let rollout = |u: &Vec<f64>| {
                let (mut x, mut v, mut cost) = ([1.0, -1.0], [0.0, 0.0], 0.0);
                for a in u.chunks(2) {
                    for k in 0..2 {
                        v[k] += 0.1 * a[k];
                        x[k] += 0.1 * v[k];
                        cost += x[k] * x[k] + 0.01 * a[k] * a[k];
                    }
                }
                cost
            };
            black_box(planner.replan(&mut { rollout }, None, &mut rng))
        });
    });
}

/// The SAS+ plan GA on a random task (feature `sas-plus`).
fn bench_sas(c: &mut Criterion) {
    #[cfg(feature = "sas-plus")]
    {
        use sas_plus::random::{RandomTaskParams, random_task};

        let task = random_task(
            3,
            RandomTaskParams {
                num_vars: 6,
                num_ops: 20,
                ..RandomTaskParams::default()
            },
        );
        c.bench_function("sas/solve/random-6x20/3000", |b| {
            b.iter(|| {
                black_box(evolutionary_planning::sas::solve(
                    &task,
                    &evolutionary_planning::sas::Config::default(),
                    &mut Limits::evaluations(3_000),
                    &mut SplitMix64::new(1),
                ))
            });
        });
    }
}

/// One evaluation of the four-robot intersection objective.
fn bench_intersection(c: &mut Criterion) {
    let mut scene = Intersection::crossing(4);
    let x = scene.bounds().center();
    c.bench_function("intersection/objective/4-robots", |b| {
        b.iter(|| black_box(scene.evaluate(&x)));
    });
}
