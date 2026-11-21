---
applyTo: "**/*.rs"
---

# Rust code

- Every module (`//!`), type, field, variant, function, constant and macro has a doc
  comment, private and `pub(crate)` items included. Docs state intent, invariants, units and
  error/panic conditions (`# Errors`, `# Panics`, `# Safety`). They do not restate the name.
- No `mod.rs`: modules with children are `foo.rs` plus `foo/`. Integration tests live under
  `tests/it/`.
- No separator comments (`// -----`, `// =====`). If a file needs sections, split it into
  submodules.
- Leave a blank line after the closing `}` of a block statement when another statement
  follows, and between logical phases of a function. No lumped code.
- One level of abstraction per function. Long loops (simulations, game ticks, solvers) are
  short drivers that call documented step functions. If a block needs a "what" comment,
  extract it into a function named after that comment.
- `clippy::pedantic` clean. Suppression only with a scoped
  `#[expect(lint, reason = "...")]`. Never `#[allow]`, and never crate-level suppression.
- No `unsafe` unless the crate opts out of `#![forbid(unsafe_code)]` with a documented
  reason. Each unsafe block has one operation and a `// SAFETY:` comment.
- The crate is `#![no_std]` plus `alloc`; floating-point maths goes through `libm`. The `sas`
  module sits behind the `sas-plus` feature. Flag any `std` use, any new `[dependencies]`
  entry, and any feature-gated item without its `#[cfg]`.
- Every objective is minimised, and NaN costs count as `+∞` before they reach sorting or
  comparison. All randomness comes from the caller's `rand_core::Rng`, and every run asks
  its `Budget` before each evaluation, so seeded runs are reproducible and limits are exact.
  Flag hidden randomness (global state, `HashMap` iteration order) and evaluations that
  bypass the budget.
- Algorithms follow their source papers; deviations are recorded in `docs/design.md`.
- Public API changes go with a matching update to the README "Scope and status" list and
  `docs/design.md`.
- Mermaid diagrams in doc comments must match the code they describe. Update them when the
  structure changes.
