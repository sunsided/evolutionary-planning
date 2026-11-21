//! The symmetric eigendecomposition behind CMA-ES: the cyclic Jacobi method.
//!
//! Jacobi is slower than tridiagonalisation plus QL for large `n`, but it is simple, accurate
//! to rounding for symmetric matrices (eigenvalues to high relative accuracy), needs no
//! `std`, and CMA-ES decomposes lazily (see [`super::Parameters::eigen_interval`]), so its
//! `O(n³)` cost per sweep is amortised over many generations.

use alloc::vec::Vec;

/// The result of decomposing a symmetric matrix `C = B diag(values) Bᵀ`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Eigen {
    /// The eigenvalues, unsorted, in the order of the columns of `vectors`.
    pub(crate) values: Vec<f64>,
    /// The orthonormal eigenvectors as the columns of a row-major `n × n` matrix.
    pub(crate) vectors: Vec<f64>,
}

/// Decomposes the symmetric row-major `n × n` matrix `matrix` (only its upper triangle is
/// trusted; the lower one is mirrored).
pub(crate) fn decompose(matrix: &[f64], n: usize) -> Eigen {
    let mut a = symmetrised(matrix, n);
    let mut v = identity(n);
    for _sweep in 0..100 {
        if off_diagonal_norm(&a, n) <= f64::EPSILON * f64::EPSILON * diagonal_norm(&a, n) {
            break;
        }

        for p in 0..n {
            for q in p + 1..n {
                rotate(&mut a, &mut v, n, p, q);
            }
        }
    }

    Eigen {
        values: (0..n).map(|i| a[i * n + i]).collect(),
        vectors: v,
    }
}

/// Copies the upper triangle of `matrix` into a full symmetric matrix.
fn symmetrised(matrix: &[f64], n: usize) -> Vec<f64> {
    let mut a = matrix.to_vec();
    for i in 0..n {
        for j in 0..i {
            a[i * n + j] = a[j * n + i];
        }
    }

    a
}

/// The `n × n` identity, row-major.
pub(crate) fn identity(n: usize) -> Vec<f64> {
    let mut m = alloc::vec![0.0; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }

    m
}

/// The squared Frobenius norm of the strict upper triangle.
fn off_diagonal_norm(a: &[f64], n: usize) -> f64 {
    let mut sum = 0.0;
    for p in 0..n {
        for q in p + 1..n {
            sum += a[p * n + q] * a[p * n + q];
        }
    }

    sum
}

/// The squared norm of the diagonal.
fn diagonal_norm(a: &[f64], n: usize) -> f64 {
    (0..n).map(|i| a[i * n + i] * a[i * n + i]).sum()
}

/// Applies the Jacobi rotation that zeroes `a[p][q]` to `a` (both sides) and to `v` (right).
#[expect(
    clippy::many_single_char_names,
    reason = "the textbook names of the Jacobi rotation"
)]
fn rotate(a: &mut [f64], v: &mut [f64], n: usize, p: usize, q: usize) {
    let apq = a[p * n + q];
    if apq == 0.0 {
        return;
    }

    let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
    let t = libm::copysign(1.0, theta) / (theta.abs() + libm::sqrt(theta * theta + 1.0));
    let c = 1.0 / libm::sqrt(t * t + 1.0);
    let s = t * c;
    for k in 0..n {
        let (akp, akq) = (a[k * n + p], a[k * n + q]);
        a[k * n + p] = c * akp - s * akq;
        a[k * n + q] = s * akp + c * akq;
    }

    for k in 0..n {
        let (apk, aqk) = (a[p * n + k], a[q * n + k]);
        a[p * n + k] = c * apk - s * aqk;
        a[q * n + k] = s * apk + c * aqk;
    }

    a[p * n + q] = 0.0;
    a[q * n + p] = 0.0;
    for k in 0..n {
        let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
        v[k * n + p] = c * vkp - s * vkq;
        v[k * n + q] = s * vkp + c * vkq;
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        /// `C b_i = λ_i b_i` and `Bᵀ B = I` for random symmetric matrices.
        #[test]
        fn decomposes_symmetric_matrices(entries in proptest::collection::vec(-10.0f64..10.0, 36)) {
            let n = 6;
            let a = symmetrised(&entries, n);
            let eigen = decompose(&a, n);
            for i in 0..n {
                for row in 0..n {
                    let cb: f64 = (0..n).map(|k| a[row * n + k] * eigen.vectors[k * n + i]).sum();
                    prop_assert!((cb - eigen.values[i] * eigen.vectors[row * n + i]).abs() < 1e-9);
                }
                for j in 0..n {
                    let dot: f64 = (0..n).map(|k| eigen.vectors[k * n + i] * eigen.vectors[k * n + j]).sum();
                    let expected = if i == j { 1.0 } else { 0.0 };
                    prop_assert!((dot - expected).abs() < 1e-12);
                }
            }
        }
    }

    /// A diagonal matrix is its own decomposition.
    #[test]
    fn diagonal_is_trivial() {
        let eigen = decompose(&[3.0, 0.0, 0.0, 5.0], 2);

        assert_eq!(eigen.values, [3.0, 5.0]);
        assert_eq!(eigen.vectors, identity(2));
    }
}
