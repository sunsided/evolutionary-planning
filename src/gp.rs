//! Tree-based genetic programming over a small operator language, for evolving planning
//! programs such as reactive navigation policies (the idea of Lou, Ding and Wu 2022 taken to
//! program trees; see also Koza 1992).
//!
//! A program is a tree in prefix order, `Vec<Symbol>`; a [`Grammar`] gives each symbol its
//! arity (`0` for terminals such as actions, more for functions such as conditionals). The
//! caller interprets programs (see [`children`]); this module supplies ramped half-and-half
//! initialisation, subtree crossover, subtree mutation and point mutation, all respecting a
//! depth and length limit.
//!
//! ```
//! use evolutionary_planning::gp::{Grammar, RampedHalfAndHalf, children, is_valid};
//! use evolutionary_planning::operators::Initializer;
//! use evolutionary_planning::SplitMix64;
//!
//! // Symbols: 0 = add(a, b), 1 = neg(a), 2 = x, 3 = one.
//! let grammar = Grammar::new(vec![2, 1, 0, 0]);
//! let program = RampedHalfAndHalf { grammar: grammar.clone(), min_depth: 1, max_depth: 4 }.initialize(&mut SplitMix64::new(1));
//! assert!(is_valid(&grammar, &program));
//!
//! fn eval(grammar: &Grammar, p: &[u16], at: usize, x: f64) -> f64 {
//!     let kids = children(grammar, p, at);
//!     match p[at] {
//!         0 => eval(grammar, p, kids[0], x) + eval(grammar, p, kids[1], x),
//!         1 => -eval(grammar, p, kids[0], x),
//!         2 => x,
//!         _ => 1.0,
//!     }
//! }
//! assert_eq!(eval(&grammar, &[0, 2, 1, 3], 0, 5.0), 4.0);
//! ```

use alloc::vec::Vec;

use rand_core::Rng;

use crate::operators::{Crossover, Initializer, Mutation};
use crate::rng::{below, chance};

/// A symbol of the language: an index into the grammar.
pub type Symbol = u16;

/// The arities of the symbols `0..n`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grammar {
    /// The arity of each symbol.
    arities: Vec<u8>,
    /// The terminals (arity `0`); never empty.
    terminals: Vec<Symbol>,
    /// The functions (arity above `0`).
    functions: Vec<Symbol>,
}

impl Grammar {
    /// A grammar from the arity of every symbol.
    ///
    /// # Panics
    ///
    /// Panics if there is no terminal or more than `u16::MAX` symbols.
    #[must_use]
    pub fn new(arities: Vec<u8>) -> Self {
        let symbols = |terminal: bool| -> Vec<Symbol> {
            (0..arities.len())
                .filter(|&s| (arities[s] == 0) == terminal)
                .map(|s| Symbol::try_from(s).expect("at most u16::MAX symbols"))
                .collect()
        };
        let (terminals, functions) = (symbols(true), symbols(false));
        assert!(!terminals.is_empty(), "a grammar needs a terminal");

        Self {
            arities,
            terminals,
            functions,
        }
    }

    /// The arity of `symbol`.
    #[must_use]
    pub fn arity(&self, symbol: Symbol) -> usize {
        usize::from(self.arities[usize::from(symbol)])
    }

    /// A uniform terminal.
    fn terminal<R: Rng + ?Sized>(&self, rng: &mut R) -> Symbol {
        self.terminals[below(rng, self.terminals.len())]
    }

    /// A uniform function, or a terminal if there is none.
    fn function<R: Rng + ?Sized>(&self, rng: &mut R) -> Symbol {
        if self.functions.is_empty() {
            return self.terminal(rng);
        }

        self.functions[below(rng, self.functions.len())]
    }
}

/// The index one past the subtree rooted at `start`.
///
/// # Panics
///
/// Panics if the program is not a valid prefix tree from `start`.
#[must_use]
pub fn subtree_end(grammar: &Grammar, program: &[Symbol], start: usize) -> usize {
    let mut open = 1usize;
    for (i, &symbol) in program.iter().enumerate().skip(start) {
        open = open + grammar.arity(symbol) - 1;
        if open == 0 {
            return i + 1;
        }
    }

    panic!("truncated program");
}

/// The start indices of the children of the node at `at`, in order.
#[must_use]
pub fn children(grammar: &Grammar, program: &[Symbol], at: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(grammar.arity(program[at]));
    let mut next = at + 1;
    for _ in 0..grammar.arity(program[at]) {
        out.push(next);
        next = subtree_end(grammar, program, next);
    }

    out
}

/// Whether `program` is exactly one complete prefix tree.
#[must_use]
pub fn is_valid(grammar: &Grammar, program: &[Symbol]) -> bool {
    let mut open = 1isize;
    for (i, &symbol) in program.iter().enumerate() {
        if usize::from(symbol) >= grammar.arities.len() {
            return false;
        }

        open += isize::from(grammar.arities[usize::from(symbol)]) - 1;
        if open == 0 {
            return i + 1 == program.len();
        }
    }

    false
}

/// The depth of the tree (a single terminal has depth `0`).
#[must_use]
pub fn depth(grammar: &Grammar, program: &[Symbol]) -> usize {
    let mut stack: Vec<usize> = Vec::new();
    let mut deepest = 0;
    for &symbol in program {
        let level = stack.len();
        deepest = deepest.max(level);
        if let Some(top) = stack.last_mut() {
            *top -= 1;
        }

        let arity = grammar.arity(symbol);
        if arity > 0 {
            stack.push(arity);
        }

        while stack.last() == Some(&0) {
            stack.pop();
        }
    }

    deepest
}

/// Appends a random tree of at most `depth` levels: the *full* method always picks functions
/// above the last level, the *grow* method any symbol.
fn generate<R: Rng + ?Sized>(
    grammar: &Grammar,
    depth: usize,
    full: bool,
    rng: &mut R,
    out: &mut Vec<Symbol>,
) {
    let terminal_chance = if full || depth == 0 {
        if depth == 0 { 1.0 } else { 0.0 }
    } else {
        #[expect(clippy::cast_precision_loss, reason = "grammars are tiny")]
        let share = grammar.terminals.len() as f64 / grammar.arities.len() as f64;
        share
    };
    let symbol = if chance(rng, terminal_chance) || grammar.functions.is_empty() {
        grammar.terminal(rng)
    } else {
        grammar.function(rng)
    };
    out.push(symbol);
    for _ in 0..grammar.arity(symbol) {
        generate(grammar, depth.saturating_sub(1), full, rng, out);
    }
}

/// Ramped half-and-half initialisation (Koza): a uniform depth in `min_depth..=max_depth`,
/// and the full or grow method with equal probability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RampedHalfAndHalf {
    /// The language.
    pub grammar: Grammar,
    /// The smallest depth.
    pub min_depth: usize,
    /// The largest depth, `>= min_depth`.
    pub max_depth: usize,
}

impl Initializer<Vec<Symbol>> for RampedHalfAndHalf {
    fn initialize<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Vec<Symbol> {
        let depth = self.min_depth + below(rng, self.max_depth.saturating_sub(self.min_depth) + 1);
        let mut program = Vec::new();
        generate(
            &self.grammar,
            depth,
            rng.next_u32() & 1 == 0,
            rng,
            &mut program,
        );

        program
    }
}

/// Picks a crossover or mutation point: a function node with probability `0.9` if there is
/// one (Koza's bias against swapping leaves), otherwise a uniform node.
fn pick_point<R: Rng + ?Sized>(grammar: &Grammar, program: &[Symbol], rng: &mut R) -> usize {
    let internal: Vec<usize> = (0..program.len())
        .filter(|&i| grammar.arity(program[i]) > 0)
        .collect();
    if !internal.is_empty() && chance(rng, 0.9) {
        return internal[below(rng, internal.len())];
    }

    below(rng, program.len())
}

/// Replaces `program[start..end]` by `replacement`.
fn splice(program: &[Symbol], start: usize, end: usize, replacement: &[Symbol]) -> Vec<Symbol> {
    let mut out = Vec::with_capacity(program.len() - (end - start) + replacement.len());
    out.extend_from_slice(&program[..start]);
    out.extend_from_slice(replacement);
    out.extend_from_slice(&program[end..]);

    out
}

/// Subtree crossover: swap a random subtree of each parent; a child over the limits is
/// replaced by a copy of its parent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubtreeCrossover {
    /// The language.
    pub grammar: Grammar,
    /// The largest depth of a child.
    pub max_depth: usize,
    /// The largest number of nodes of a child.
    pub max_len: usize,
}

impl Crossover<Vec<Symbol>> for SubtreeCrossover {
    fn crossover<R: Rng + ?Sized>(
        &mut self,
        a: &Vec<Symbol>,
        b: &Vec<Symbol>,
        rng: &mut R,
    ) -> [Vec<Symbol>; 2] {
        let grammar = &self.grammar;
        let (start_a, start_b) = (pick_point(grammar, a, rng), pick_point(grammar, b, rng));
        let (end_a, end_b) = (
            subtree_end(grammar, a, start_a),
            subtree_end(grammar, b, start_b),
        );
        let within = |child: &Vec<Symbol>| {
            child.len() <= self.max_len && depth(grammar, child) <= self.max_depth
        };
        let c1 = splice(a, start_a, end_a, &b[start_b..end_b]);
        let c2 = splice(b, start_b, end_b, &a[start_a..end_a]);

        [
            if within(&c1) { c1 } else { a.clone() },
            if within(&c2) { c2 } else { b.clone() },
        ]
    }
}

/// Subtree mutation: replace a random subtree by a grown tree of at most `subtree_depth`
/// levels, unless that breaks the limits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubtreeMutation {
    /// The language.
    pub grammar: Grammar,
    /// The depth of the new subtree.
    pub subtree_depth: usize,
    /// The largest depth of the result.
    pub max_depth: usize,
    /// The largest number of nodes of the result.
    pub max_len: usize,
}

impl Mutation<Vec<Symbol>> for SubtreeMutation {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut Vec<Symbol>, rng: &mut R) {
        let g = &self.grammar;
        let at = pick_point(g, genome, rng);
        let end = subtree_end(g, genome, at);
        let mut fresh = Vec::new();
        generate(g, self.subtree_depth, false, rng, &mut fresh);
        let candidate = splice(genome, at, end, &fresh);
        if candidate.len() <= self.max_len && depth(g, &candidate) <= self.max_depth {
            *genome = candidate;
        }
    }
}

/// Point mutation: replace one node's symbol by another of the same arity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PointMutation {
    /// The language.
    pub grammar: Grammar,
}

impl Mutation<Vec<Symbol>> for PointMutation {
    fn mutate<R: Rng + ?Sized>(&mut self, genome: &mut Vec<Symbol>, rng: &mut R) {
        if genome.is_empty() {
            return;
        }

        let at = below(rng, genome.len());
        let arity = self.grammar.arity(genome[at]);
        let same: Vec<Symbol> = (0..self.grammar.arities.len())
            .filter(|&s| usize::from(self.grammar.arities[s]) == arity)
            .map(|s| Symbol::try_from(s).expect("checked at construction"))
            .collect();
        genome[at] = same[below(rng, same.len())];
    }
}

#[cfg(test)]
mod tests;
