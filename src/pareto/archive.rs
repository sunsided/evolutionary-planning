//! A Pareto archive: the non-dominated set of everything offered to it, optionally bounded
//! with crowding-distance truncation.

use alloc::vec::Vec;

use super::{Evaluated, constrained_dominates, crowding_distance};

/// The non-dominated members of all genomes inserted so far, under constrained domination.
///
/// Inserting a point that is dominated by, or equal in objectives to, an archived point is
/// rejected; otherwise it enters and every archived point it dominates leaves. With a
/// capacity, an overfull archive drops the member of smallest crowding distance (ties: the
/// oldest), which keeps the extremes and spreads the rest.
///
/// ```
/// use evolutionary_planning::pareto::ParetoArchive;
///
/// let mut archive = ParetoArchive::new(None);
/// assert!(archive.insert("a", vec![1.0, 3.0], 0.0));
/// assert!(archive.insert("b", vec![3.0, 1.0], 0.0));
/// assert!(!archive.insert("c", vec![3.0, 3.0], 0.0));
/// assert!(archive.insert("d", vec![0.5, 0.5], 0.0));
/// assert_eq!(archive.members().len(), 1);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ParetoArchive<G> {
    /// The archived members, mutually non-dominated, in insertion order.
    members: Vec<Evaluated<G>>,
    /// The largest number of members kept; `None` is unbounded.
    capacity: Option<usize>,
}

impl<G> ParetoArchive<G> {
    /// An empty archive with an optional capacity.
    #[must_use]
    pub fn new(capacity: Option<usize>) -> Self {
        Self {
            members: Vec::new(),
            capacity,
        }
    }

    /// The archived members.
    #[must_use]
    pub fn members(&self) -> &[Evaluated<G>] {
        &self.members
    }

    /// Consumes the archive, returning its members.
    #[must_use]
    pub fn into_members(self) -> Vec<Evaluated<G>> {
        self.members
    }

    /// Offers a genome; returns `true` if it entered the archive (and stayed after a
    /// capacity truncation).
    pub fn insert(&mut self, genome: G, objectives: Vec<f64>, violation: f64) -> bool {
        let rejected = self.members.iter().any(|m| {
            constrained_dominates(&m.objectives, m.violation, &objectives, violation)
                || (m.objectives == objectives && m.violation <= violation)
        });
        if rejected {
            return false;
        }

        self.members
            .retain(|m| !constrained_dominates(&objectives, violation, &m.objectives, m.violation));
        self.members.push(Evaluated {
            genome,
            objectives,
            violation,
        });

        self.truncate()
    }

    /// Drops members of smallest crowding distance until within capacity; returns whether
    /// the newest member survived.
    fn truncate(&mut self) -> bool {
        let Some(capacity) = self.capacity else {
            return true;
        };

        let mut newest_survives = true;
        while self.members.len() > capacity.max(1) {
            let points: Vec<&[f64]> = self
                .members
                .iter()
                .map(|m| m.objectives.as_slice())
                .collect();
            let all: Vec<usize> = (0..points.len()).collect();
            let crowding = crowding_distance(&points, &all);
            let victim = (0..crowding.len())
                .min_by(|&a, &b| crowding[a].total_cmp(&crowding[b]).then(a.cmp(&b)))
                .unwrap_or(0);
            newest_survives &= victim != self.members.len() - 1;
            self.members.remove(victim);
        }

        newest_survives
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pareto::dominates;

    /// The archive stays mutually non-dominated and bounded.
    #[test]
    fn stays_non_dominated_and_bounded() {
        let mut archive = ParetoArchive::new(Some(5));
        let mut rng = crate::SplitMix64::new(3);
        for i in 0..500 {
            let a = crate::unit_f64(&mut rng);
            let b = crate::unit_f64(&mut rng);
            archive.insert(i, alloc::vec![a, b + (1.0 - a) * (1.0 - a)], 0.0);
        }

        let members = archive.members();
        assert!(members.len() <= 5);
        for x in members {
            assert!(
                !members
                    .iter()
                    .any(|y| dominates(&y.objectives, &x.objectives))
            );
        }
    }
}
