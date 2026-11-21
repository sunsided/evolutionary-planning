//! Property tests for the Pareto utilities: the exact hypervolume against a grid count, and
//! the unbounded archive against the brute-force non-dominated set.

use evolutionary_planning::pareto::{ParetoArchive, hypervolume, non_dominated};
use proptest::prelude::*;

/// Side length of the integer grid the hypervolume properties live on.
const SIDE: u8 = 5;

/// Points with integer coordinates in `0..SIDE`, in two to four dimensions.
fn grid_points() -> impl Strategy<Value = Vec<Vec<f64>>> {
    (2usize..=4).prop_flat_map(|dimensions| {
        proptest::collection::vec(
            proptest::collection::vec((0..SIDE).prop_map(f64::from), dimensions),
            0..12,
        )
    })
}

/// The number of unit cells of `[0, SIDE)^d` dominated by some point: a cell is covered
/// when a point is no greater than its lower corner in every objective.
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

proptest! {
    /// On integer points the exact hypervolume equals the number of dominated unit cells.
    #[test]
    fn hypervolume_matches_grid_count(points in grid_points()) {
        let dimensions = points.first().map_or(2, Vec::len);
        let reference = vec![f64::from(SIDE); dimensions];
        let expected = f64::from(grid_count(&points, dimensions));

        prop_assert!((hypervolume(&points, &reference) - expected).abs() < 1e-9);
    }

    /// An unbounded archive of feasible points holds exactly the distinct non-dominated
    /// points among everything inserted.
    #[test]
    fn unbounded_archive_is_the_non_dominated_set(
        raw in proptest::collection::vec((0u8..6, 0u8..6), 0..30),
    ) {
        let points: Vec<Vec<f64>> = raw.iter().map(|&(a, b)| vec![f64::from(a), f64::from(b)]).collect();
        let mut archive = ParetoArchive::new(None);
        for (index, point) in points.iter().enumerate() {
            archive.insert(index, point.clone(), 0.0);
        }

        let mut expected: Vec<Vec<f64>> = non_dominated(&points).into_iter().map(|i| points[i].clone()).collect();
        expected.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        expected.dedup();
        let mut archived: Vec<Vec<f64>> = archive.members().iter().map(|m| m.objectives.clone()).collect();
        archived.sort_by(|a, b| a.partial_cmp(b).expect("finite"));

        prop_assert_eq!(archived, expected);
    }
}
