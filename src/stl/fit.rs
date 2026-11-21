//! Evolutionary fitting of STL thresholds to demonstrations, after EvoPlan's constraint
//! mining: maximise
//! `E_{τ ∈ D+} ρ(τ, φ) - λ- E_{τ ∈ D-} max(0, ρ(τ, φ)) - λc · complexity(φ)`
//! over the thresholds of a template formula, with CMA-ES. `D+` are acceptable traces,
//! `D-` perturbed ones that should violate the constraint.
//!
//! EvoPlan searches over formula structure with LLM-proposed program edits; here the
//! structure is the caller's template and only its thresholds evolve (see the paper table in
//! `docs/design.md`).

use alloc::vec::Vec;

use rand_core::Rng;

use super::Formula;
use crate::cmaes::{CmaEs, Config};
use crate::{Limits, Objective};

/// Parameters of [`fit`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitConfig {
    /// The weight `λ-` of satisfied negatives. The positive margin grows with slope `1` as a
    /// threshold loosens, and each admitted negative adds slope `-λ- / |D-|`; so with
    /// `λ- > |D-|` admitting even one negative never pays, while smaller weights trade a few
    /// admitted negatives for margin on the positives.
    pub negative_weight: f64,
    /// The weight `λc` of the complexity term (constant for a fixed template; it matters when
    /// several templates are compared by their [`Fitted::score`]).
    pub complexity_weight: f64,
    /// The initial CMA-ES step size, in threshold units.
    pub sigma: f64,
    /// The evaluation budget of CMA-ES.
    pub evaluations: u64,
}

impl Default for FitConfig {
    /// `λ- = 2`, `λc = 0.01`, `σ = 1`, 3000 evaluations.
    fn default() -> Self {
        Self {
            negative_weight: 2.0,
            complexity_weight: 0.01,
            sigma: 1.0,
            evaluations: 3_000,
        }
    }
}

/// A fitted formula and its score.
#[derive(Debug, Clone, PartialEq)]
pub struct Fitted {
    /// The template with the fitted thresholds.
    pub formula: Formula,
    /// Its score (to maximise).
    pub score: f64,
}

/// The EvoPlan mining score of `formula` on the demonstrations. Robustness values are clamped
/// to `±1e6` so that vacuous windows (`±∞`) do not dominate the means.
#[must_use]
pub fn score<S: AsRef<[f64]>>(
    formula: &Formula,
    positives: &[Vec<S>],
    negatives: &[Vec<S>],
    config: &FitConfig,
) -> f64 {
    /// The clamp of infinite robustness values.
    const BOUND: f64 = 1e6;

    let mean = |values: &mut dyn Iterator<Item = f64>, count: usize| {
        #[expect(clippy::cast_precision_loss, reason = "demonstration counts are small")]
        let count = count.max(1) as f64;
        values.sum::<f64>() / count
    };
    let rho = |trace: &Vec<S>| formula.robustness(trace).clamp(-BOUND, BOUND);
    let satisfied = mean(&mut positives.iter().map(rho), positives.len());
    let admitted = mean(
        &mut negatives.iter().map(|t| rho(t).max(0.0)),
        negatives.len(),
    );
    #[expect(clippy::cast_precision_loss, reason = "formula sizes are small")]
    let complexity = formula.complexity() as f64;

    satisfied - config.negative_weight * admitted - config.complexity_weight * complexity
}

/// Fits the thresholds of `template` to positive and negative demonstrations by maximising
/// [`score`] with CMA-ES from the template's own thresholds.
///
/// ```
/// use evolutionary_planning::stl::{FitConfig, Formula, fit};
/// use evolutionary_planning::SplitMix64;
///
/// // Demonstrations keep speed (signal 0) at most 5; perturbed ones speed up to 10.
/// let positives: Vec<Vec<[f64; 1]>> = (0..5).map(|k| vec![[3.0 + 0.4 * k as f64]; 4]).collect();
/// let negatives: Vec<Vec<[f64; 1]>> = (0..5).map(|k| vec![[10.0 + k as f64]; 4]).collect();
/// let template = Formula::always(0, 3, Formula::le(0, 0.0));
/// // Five negatives: a weight above five forbids admitting any of them.
/// let config = FitConfig { negative_weight: 10.0, ..FitConfig::default() };
/// let fitted = fit(&template, &positives, &negatives, &config, &mut SplitMix64::new(1));
/// let threshold = fitted.formula.thresholds()[0];
/// assert!((4.6..=10.0 + 1e-6).contains(&threshold), "{threshold}");
/// assert!(negatives.iter().all(|t| fitted.formula.robustness(t) <= 1e-6));
/// ```
pub fn fit<S: AsRef<[f64]>, R: Rng + ?Sized>(
    template: &Formula,
    positives: &[Vec<S>],
    negatives: &[Vec<S>],
    config: &FitConfig,
    rng: &mut R,
) -> Fitted {
    let start = template.thresholds();
    if start.is_empty() {
        return Fitted {
            formula: template.clone(),
            score: score(template, positives, negatives, config),
        };
    }

    let mut objective = |thresholds: &Vec<f64>| {
        -score(
            &template.with_thresholds(thresholds),
            positives,
            negatives,
            config,
        )
    };
    let mut cma = CmaEs::new(start.clone(), config.sigma, Config::default());
    let outcome = cma.minimize(
        &mut objective,
        &mut Limits::evaluations(config.evaluations),
        &mut (),
        rng,
    );
    let thresholds = outcome.best.map_or(start, |b| b.genome);
    let formula = template.with_thresholds(&thresholds);
    let fitted_score = -objective.evaluate(&thresholds);

    Fitted {
        formula,
        score: fitted_score,
    }
}
