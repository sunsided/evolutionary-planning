//! Fuzzes [`hypervolume`]: on small integer points it must equal the number of dominated
//! unit cells, and on arbitrary finite points it must stay within the bounding box.

#![no_main]
#![forbid(unsafe_code)]

use evolutionary_planning::pareto::hypervolume;
use libfuzzer_sys::fuzz_target;

/// Side length of the integer grid; the reference point is `SIDE` in every objective.
const SIDE: u8 = 6;

/// Most points per input; exact hypervolume is exponential in the dimension.
const MAX_POINTS: usize = 16;

/// The number of unit cells of `[0, SIDE)^d` that some point is no greater than in every
/// objective, which is the exact hypervolume of integer points with reference `SIDE`.
fn grid_count(points: &[Vec<f64>], dimensions: usize) -> u32 {
    let side = usize::from(SIDE);
    let cells = side.pow(u32::try_from(dimensions).expect("at most four dimensions"));
    let mut count = 0;
    for index in 0..cells {
        let corner = cell_corner(index, dimensions);
        if points
            .iter()
            .any(|p| p.iter().zip(&corner).all(|(x, c)| x <= c))
        {
            count += 1;
        }
    }

    count
}

/// The lower corner of the cell with the given row-major `index`.
fn cell_corner(mut index: usize, dimensions: usize) -> Vec<f64> {
    let side = usize::from(SIDE);
    let mut corner = Vec::with_capacity(dimensions);
    for _ in 0..dimensions {
        let coordinate = u8::try_from(index % side).expect("coordinates are below SIDE");
        corner.push(f64::from(coordinate));
        index /= side;
    }

    corner
}

/// Checks the exact volume of integer points in two to four dimensions.
fn check_grid(selector: u8, raw: &[[u8; 4]]) {
    let dimensions = 2 + usize::from(selector % 3);
    let points: Vec<Vec<f64>> = raw
        .iter()
        .take(MAX_POINTS)
        .map(|p| {
            p[..dimensions]
                .iter()
                .map(|&x| f64::from(x % SIDE))
                .collect()
        })
        .collect();
    let reference = vec![f64::from(SIDE); dimensions];
    let expected = f64::from(grid_count(&points, dimensions));
    let volume = hypervolume(&points, &reference);

    assert!(
        (volume - expected).abs() < 1e-9,
        "volume {volume}, grid count {expected} for {points:?}"
    );
}

/// Checks that finite three-dimensional points give a volume in `[0, box]`, where `box`
/// is the volume between the per-objective minima and the reference.
fn check_bounds(reference: [f64; 3], raw: &[[f64; 3]]) {
    if !reference
        .iter()
        .chain(raw.iter().flatten())
        .all(|x| x.is_finite())
    {
        return;
    }

    let points = &raw[..raw.len().min(MAX_POINTS)];
    let volume = hypervolume(points, &reference);
    let bound: f64 = (0..3)
        .map(|k| {
            let low = points.iter().map(|p| p[k]).fold(reference[k], f64::min);
            reference[k] - low
        })
        .product();

    assert!(!volume.is_nan(), "NaN volume for {points:?}");
    assert!(volume >= 0.0, "negative volume {volume} for {points:?}");
    // An overflowing extent times a zero one makes the box NaN; nothing to compare then.
    assert!(
        bound.is_nan() || volume <= bound * (1.0 + 1e-9),
        "volume {volume} above the box {bound} for {points:?}"
    );
}

/// One fuzz input: the dimension selector and integer points for [`check_grid`], then the
/// reference and finite points for [`check_bounds`].
type Input = (u8, Vec<[u8; 4]>, [f64; 3], Vec<[f64; 3]>);

fuzz_target!(|input: Input| {
    let (selector, grid, reference, points) = input;

    check_grid(selector, &grid);
    check_bounds(reference, &points);
});
