//! Property tests for range invariants: the random helpers stay in their documented ranges,
//! and the bounded real-vector operators keep children inside arbitrary boxes.

use evolutionary_planning::operators::real::{Blx, Bounds, Polynomial, Sbx};
use evolutionary_planning::operators::{Crossover, Initializer, Mutation};
use evolutionary_planning::rng::two_sorted_indices;
use evolutionary_planning::{SplitMix64, below, uniform, unit_f64};
use proptest::prelude::*;

proptest! {
    /// `below`, `two_sorted_indices`, `unit_f64` and `uniform` return values in their ranges.
    #[test]
    fn random_helpers_stay_in_range(
        seed in any::<u64>(),
        n in 2usize..1_000,
        low in -1e6f64..1e6,
        width in 0.0f64..1e6,
    ) {
        let mut rng = SplitMix64::new(seed);
        let high = low + width;
        for _ in 0..32 {
            prop_assert!(below(&mut rng, n) < n);

            let (a, b) = two_sorted_indices(&mut rng, n);
            prop_assert!(a < b && b < n, "{a} {b} {n}");

            let u = unit_f64(&mut rng);
            prop_assert!((0.0..1.0).contains(&u), "{u}");

            let x = uniform(&mut rng, low, high);
            let in_range = if high > low { (low..high).contains(&x) } else { x.total_cmp(&low).is_eq() };
            prop_assert!(in_range, "{x} not in [{low}, {high})");
        }
    }

    /// SBX, BLX-α and polynomial mutation keep children inside any box, however narrow.
    #[test]
    fn bounded_operators_stay_inside_any_box(
        seed in any::<u64>(),
        dimension in 1usize..8,
        low in -100.0f64..100.0,
        width in 1e-6f64..100.0,
        eta in 1.0f64..100.0,
        alpha in 0.0f64..1.0,
    ) {
        let bounds = Bounds::uniform(dimension, low, low + width);
        let mut rng = SplitMix64::new(seed);
        let mut init = bounds.clone();
        let mut sbx = Sbx { bounds: bounds.clone(), eta };
        let mut blx = Blx { bounds: bounds.clone(), alpha };
        let mut poly = Polynomial { bounds: bounds.clone(), eta, rate: Some(1.0) };
        for _ in 0..16 {
            let a = init.initialize(&mut rng);
            let b = init.initialize(&mut rng);
            for mut child in sbx.crossover(&a, &b, &mut rng).into_iter().chain(blx.crossover(&a, &b, &mut rng)) {
                prop_assert!(bounds.contains(&child), "crossover child {child:?}");

                poly.mutate(&mut child, &mut rng);
                prop_assert!(bounds.contains(&child), "mutated child {child:?}");
            }
        }
    }
}
