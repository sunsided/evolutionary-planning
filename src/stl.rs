//! Finite-trace signal temporal logic (STL) with quantitative semantics, for evolutionary
//! search under temporal-logic constraints (after EvoPlan, Nukapotula et al. 2026).
//!
//! A trace is a sequence of time steps, each a slice of named signals (by index). The
//! robustness `ρ(φ, trace, t)` of a formula is positive when the trace satisfies it at `t`,
//! negative when it violates it, and its magnitude says by how much (Donzé and Maler 2010):
//!
//! | Formula | Robustness at `t` |
//! |---|---|
//! | `x_i ≥ c` | `x_i(t) - c` |
//! | `x_i ≤ c` | `c - x_i(t)` |
//! | `¬φ` | `-ρ(φ, t)` |
//! | `φ ∧ ψ`, `φ ∨ ψ` | `min`, `max` |
//! | `G[a,b] φ` | `min` of `ρ(φ, t')` over `t' ∈ [t+a, t+b]` |
//! | `F[a,b] φ` | `max` over the same window |
//! | `φ U[a,b] ψ` | `max` over `t'` of `min(ρ(ψ, t'), min over t'' ∈ [t, t') of ρ(φ, t''))` |
//!
//! Windows are clipped to the trace (finite-trace semantics): `G` over an empty window is
//! `+∞` (vacuously true), `F` and `U` are `-∞`; a predicate past the end of the trace is
//! `-∞` (no evidence). Robustness is computed bottom-up as a signal over all time steps, so
//! a formula costs `O(nodes · length · window)`.
//!
//! Uses: an objective (`-ρ`), a graded constraint violation `max(0, -ρ)` (Jakob's rule
//! against flat penalties), a [`shield`] that picks the candidate action whose rollout is most
//! robust, and [`fit()`] to learn thresholds from demonstrations.
//!
//! ```
//! use evolutionary_planning::stl::Formula;
//!
//! // "Always within the first 3 steps, speed (signal 0) stays at most 2, and eventually
//! // clearance (signal 1) reaches at least 1."
//! let phi = Formula::and(vec![
//!     Formula::always(0, 3, Formula::le(0, 2.0)),
//!     Formula::eventually(0, 5, Formula::ge(1, 1.0)),
//! ]);
//! let trace = [[1.0, 0.0], [1.5, 0.5], [1.8, 1.4], [0.5, 2.0]];
//! assert!((phi.robustness(&trace) - 0.2).abs() < 1e-12);
//! assert_eq!(phi.violation(&trace), 0.0);
//! ```

mod fit;

use alloc::boxed::Box;
use alloc::vec::Vec;

pub use fit::{FitConfig, Fitted, fit, score};

/// An STL formula over signals indexed `0..`.
#[derive(Debug, Clone, PartialEq)]
pub enum Formula {
    /// Always true (`ρ = +∞`).
    True,
    /// `x_signal ≥ threshold`.
    AtLeast {
        /// The signal index.
        signal: usize,
        /// The threshold `c`.
        threshold: f64,
    },
    /// `x_signal ≤ threshold`.
    AtMost {
        /// The signal index.
        signal: usize,
        /// The threshold `c`.
        threshold: f64,
    },
    /// Negation.
    Not(Box<Formula>),
    /// Conjunction; empty is `True`.
    And(Vec<Formula>),
    /// Disjunction; empty is false (`-∞`).
    Or(Vec<Formula>),
    /// `G[from, to] φ`: `φ` holds at every step of the window.
    Always {
        /// The window start, relative to the current step.
        from: usize,
        /// The window end (inclusive), `>= from`.
        to: usize,
        /// The formula that must hold throughout.
        formula: Box<Formula>,
    },
    /// `F[from, to] φ`: `φ` holds at some step of the window.
    Eventually {
        /// The window start, relative to the current step.
        from: usize,
        /// The window end (inclusive), `>= from`.
        to: usize,
        /// The formula that must hold once.
        formula: Box<Formula>,
    },
    /// `φ U[from, to] ψ`: `ψ` holds at some step of the window and `φ` at every step before.
    Until {
        /// The window start, relative to the current step.
        from: usize,
        /// The window end (inclusive), `>= from`.
        to: usize,
        /// The formula that must hold until `right` does.
        left: Box<Formula>,
        /// The formula that must eventually hold.
        right: Box<Formula>,
    },
}

impl Formula {
    /// `x_signal ≥ threshold`.
    #[must_use]
    pub fn ge(signal: usize, threshold: f64) -> Self {
        Self::AtLeast { signal, threshold }
    }

    /// `x_signal ≤ threshold`.
    #[must_use]
    pub fn le(signal: usize, threshold: f64) -> Self {
        Self::AtMost { signal, threshold }
    }

    /// `¬φ`.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "a constructor named after the operator, not ops::Not"
    )]
    pub fn not(formula: Self) -> Self {
        Self::Not(Box::new(formula))
    }

    /// `φ_1 ∧ … ∧ φ_k`.
    #[must_use]
    pub fn and(formulas: Vec<Self>) -> Self {
        Self::And(formulas)
    }

    /// `φ_1 ∨ … ∨ φ_k`.
    #[must_use]
    pub fn or(formulas: Vec<Self>) -> Self {
        Self::Or(formulas)
    }

    /// `G[from, to] φ`.
    #[must_use]
    pub fn always(from: usize, to: usize, formula: Self) -> Self {
        Self::Always {
            from,
            to,
            formula: Box::new(formula),
        }
    }

    /// `F[from, to] φ`.
    #[must_use]
    pub fn eventually(from: usize, to: usize, formula: Self) -> Self {
        Self::Eventually {
            from,
            to,
            formula: Box::new(formula),
        }
    }

    /// `φ U[from, to] ψ`.
    #[must_use]
    pub fn until(from: usize, to: usize, left: Self, right: Self) -> Self {
        Self::Until {
            from,
            to,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// The robustness at the first step of `trace`.
    #[must_use]
    pub fn robustness<S: AsRef<[f64]>>(&self, trace: &[S]) -> f64 {
        if trace.is_empty() {
            return self.signal(trace, 1)[0];
        }

        self.signal(trace, trace.len())[0]
    }

    /// The constraint violation `max(0, -ρ)` at the first step: `0` when satisfied.
    #[must_use]
    pub fn violation<S: AsRef<[f64]>>(&self, trace: &[S]) -> f64 {
        (-self.robustness(trace)).max(0.0)
    }

    /// The robustness at every step `0..len` (steps past the trace see an empty future).
    #[must_use]
    pub fn signal<S: AsRef<[f64]>>(&self, trace: &[S], len: usize) -> Vec<f64> {
        let at = |signal: usize, t: usize| {
            trace
                .get(t)
                .and_then(|step| step.as_ref().get(signal))
                .copied()
        };
        match self {
            Self::True => alloc::vec![f64::INFINITY; len],
            Self::AtLeast { signal, threshold } => (0..len)
                .map(|t| at(*signal, t).map_or(f64::NEG_INFINITY, |x| x - threshold))
                .collect(),
            Self::AtMost { signal, threshold } => (0..len)
                .map(|t| at(*signal, t).map_or(f64::NEG_INFINITY, |x| threshold - x))
                .collect(),
            Self::Not(inner) => inner.signal(trace, len).into_iter().map(|v| -v).collect(),
            Self::And(parts) => combine(parts, trace, len, f64::INFINITY, f64::min),
            Self::Or(parts) => combine(parts, trace, len, f64::NEG_INFINITY, f64::max),
            Self::Always { from, to, formula } => window(
                &formula.signal(trace, trace.len()),
                len,
                *from,
                *to,
                f64::INFINITY,
                f64::min,
            ),
            Self::Eventually { from, to, formula } => window(
                &formula.signal(trace, trace.len()),
                len,
                *from,
                *to,
                f64::NEG_INFINITY,
                f64::max,
            ),
            Self::Until {
                from,
                to,
                left,
                right,
            } => until(
                &left.signal(trace, trace.len()),
                &right.signal(trace, trace.len()),
                len,
                *from,
                *to,
            ),
        }
    }

    /// The number of nodes, the complexity term of [`fit()`].
    #[must_use]
    pub fn complexity(&self) -> usize {
        1 + match self {
            Self::True | Self::AtLeast { .. } | Self::AtMost { .. } => 0,
            Self::Not(inner)
            | Self::Always { formula: inner, .. }
            | Self::Eventually { formula: inner, .. } => inner.complexity(),
            Self::And(parts) | Self::Or(parts) => parts.iter().map(Self::complexity).sum(),
            Self::Until { left, right, .. } => left.complexity() + right.complexity(),
        }
    }

    /// The predicate thresholds in depth-first order: the parameters [`fit()`] tunes.
    #[must_use]
    pub fn thresholds(&self) -> Vec<f64> {
        let mut out = Vec::new();
        self.clone().visit_thresholds(&mut |c| out.push(*c));

        out
    }

    /// A copy with the thresholds replaced, in the order of [`Formula::thresholds`]; missing
    /// values leave thresholds unchanged.
    #[must_use]
    pub fn with_thresholds(&self, values: &[f64]) -> Self {
        let mut copy = self.clone();
        let mut next = values.iter();
        copy.visit_thresholds(&mut |c| {
            if let Some(&v) = next.next() {
                *c = v;
            }
        });

        copy
    }

    /// Calls `f` on every threshold in depth-first order.
    fn visit_thresholds(&mut self, f: &mut impl FnMut(&mut f64)) {
        match self {
            Self::True => {}
            Self::AtLeast { threshold, .. } | Self::AtMost { threshold, .. } => f(threshold),
            Self::Not(inner)
            | Self::Always { formula: inner, .. }
            | Self::Eventually { formula: inner, .. } => {
                inner.visit_thresholds(f);
            }
            Self::And(parts) | Self::Or(parts) => {
                for part in parts.iter_mut() {
                    part.visit_thresholds(f);
                }
            }
            Self::Until { left, right, .. } => {
                left.visit_thresholds(f);
                right.visit_thresholds(f);
            }
        }
    }
}

/// Combines the signals of `parts` point-wise with `op`, starting from `identity`.
fn combine<S: AsRef<[f64]>>(
    parts: &[Formula],
    trace: &[S],
    len: usize,
    identity: f64,
    op: fn(f64, f64) -> f64,
) -> Vec<f64> {
    let mut out = alloc::vec![identity; len];
    for part in parts {
        for (o, v) in out.iter_mut().zip(part.signal(trace, len)) {
            *o = op(*o, v);
        }
    }

    out
}

/// Folds `inner` over the clipped window `[t + from, t + to]` for each `t < len`.
fn window(
    inner: &[f64],
    len: usize,
    from: usize,
    to: usize,
    identity: f64,
    op: fn(f64, f64) -> f64,
) -> Vec<f64> {
    (0..len)
        .map(|t| {
            let start = t.saturating_add(from);
            let end = t.saturating_add(to).min(inner.len().saturating_sub(1));
            if start >= inner.len() || start > end {
                return identity;
            }

            inner[start..=end].iter().copied().fold(identity, op)
        })
        .collect()
}

/// The bounded until: `max` over `t' ∈ [t + from, t + to]` of
/// `min(right(t'), min over t'' ∈ [t, t') of left(t''))`.
fn until(left: &[f64], right: &[f64], len: usize, from: usize, to: usize) -> Vec<f64> {
    (0..len)
        .map(|t| {
            let end = t.saturating_add(to).min(right.len().saturating_sub(1));
            let mut best = f64::NEG_INFINITY;
            let mut prefix = f64::INFINITY;
            for t2 in t..=end.max(t) {
                if t2 >= right.len() {
                    break;
                }
                if t2 >= t.saturating_add(from) {
                    best = best.max(right[t2].min(prefix));
                }
                prefix = prefix.min(left[t2]);
            }

            best
        })
        .collect()
}

/// Picks the candidate whose rollout trace is most robust for `formula`: EvoPlan's shield,
/// which vetoes a policy's action when its short-horizon rollout violates the constraint and
/// substitutes the most robust candidate. Returns `(index, robustness)` of the best candidate
/// (the first among equals), or `None` if there are no candidates.
///
/// ```
/// use evolutionary_planning::stl::{Formula, shield};
///
/// // Keep speed (signal 0) at most 1 for the next two steps.
/// let phi = Formula::always(0, 2, Formula::le(0, 1.0));
/// let speeds = [0.5, 1.5, 0.9];
/// let best = shield(&speeds, |&v: &f64| vec![[v], [v], [v]], &phi);
/// assert_eq!(best.map(|(i, _)| i), Some(0));
/// ```
pub fn shield<A, S: AsRef<[f64]>>(
    candidates: &[A],
    mut rollout: impl FnMut(&A) -> Vec<S>,
    formula: &Formula,
) -> Option<(usize, f64)> {
    let mut best: Option<(usize, f64)> = None;
    for (index, candidate) in candidates.iter().enumerate() {
        let rho = formula.robustness(&rollout(candidate));
        if best.is_none_or(|(_, b)| rho > b) {
            best = Some((index, rho));
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    /// Hand-checked robustness values for every operator, including clipped windows.
    #[test]
    fn operator_semantics() {
        let trace = [[0.0], [2.0], [5.0], [1.0]];
        let at_least_1 = Formula::ge(0, 1.0);

        assert_eq!(at_least_1.robustness(&trace), -1.0);
        assert_eq!(Formula::not(at_least_1.clone()).robustness(&trace), 1.0);
        assert_eq!(
            Formula::eventually(0, 3, at_least_1.clone()).robustness(&trace),
            4.0
        );
        assert_eq!(
            Formula::always(1, 3, at_least_1.clone()).robustness(&trace),
            0.0
        );
        assert_eq!(
            Formula::always(1, 10, at_least_1.clone()).robustness(&trace),
            0.0
        );
        assert_eq!(
            Formula::always(9, 10, at_least_1.clone()).robustness(&trace),
            f64::INFINITY
        );
        assert_eq!(
            Formula::eventually(9, 10, at_least_1.clone()).robustness(&trace),
            f64::NEG_INFINITY
        );
        assert_eq!(Formula::and(vec![]).robustness(&trace), f64::INFINITY);
        assert_eq!(Formula::or(vec![]).robustness(&trace), f64::NEG_INFINITY);

        // "x <= 3 until x >= 5": x(2) = 5 is reached with x <= 3 before: min(0, min(3, 1)) = 0.
        let until = Formula::until(0, 3, Formula::le(0, 3.0), Formula::ge(0, 5.0));
        assert_eq!(until.robustness(&trace), 0.0);
        // Requiring x <= 1 before fails at step 1 (x = 2): robustness -1.
        let strict = Formula::until(0, 3, Formula::le(0, 1.0), Formula::ge(0, 5.0));
        assert_eq!(strict.robustness(&trace), -1.0);
    }

    /// The sign of robustness agrees with Boolean satisfaction checked by brute force.
    #[test]
    fn sign_matches_boolean_semantics() {
        let mut rng = crate::SplitMix64::new(4);
        let phi = Formula::and(vec![
            Formula::always(
                0,
                4,
                Formula::or(vec![Formula::le(0, 0.5), Formula::ge(1, 0.2)]),
            ),
            Formula::eventually(1, 3, Formula::ge(0, 0.7)),
        ]);
        for _ in 0..500 {
            let trace: Vec<[f64; 2]> = (0..6)
                .map(|_| [crate::unit_f64(&mut rng), crate::unit_f64(&mut rng)])
                .collect();
            let holds = (0..=4).all(|t| trace[t][0] <= 0.5 || trace[t][1] >= 0.2)
                && (1..=3).any(|t| trace[t][0] >= 0.7);
            let rho = phi.robustness(&trace);
            assert_eq!(rho >= 0.0, holds, "{rho} {trace:?}");
        }
    }

    /// Threshold extraction and replacement round-trip in depth-first order.
    #[test]
    fn thresholds_round_trip() {
        let phi = Formula::until(
            0,
            2,
            Formula::le(0, 1.0),
            Formula::and(vec![Formula::ge(1, 2.0), Formula::le(2, 3.0)]),
        );

        assert_eq!(phi.thresholds(), [1.0, 2.0, 3.0]);
        assert_eq!(
            phi.with_thresholds(&[7.0, 8.0, 9.0]).thresholds(),
            [7.0, 8.0, 9.0]
        );
        assert_eq!(phi.complexity(), 5);
    }
}
