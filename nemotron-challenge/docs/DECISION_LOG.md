# Decision Log

> Every strategic or technical decision that shapes the project gets recorded here.
> Decisions are immutable records — if you change course, add a NEW decision that
> supersedes the old one, don't edit the original.

---

## How to Log a Decision

```markdown
### D[number]: [Short title]
**Date**: YYYY-MM-DD
**Context**: What situation prompted this decision?
**Options considered**:
1. Option A — [pros/cons]
2. Option B — [pros/cons]
**Decision**: What we chose
**Rationale**: Why
**Assumptions this depends on**: [link to ASSUMPTIONS.md IDs]
**Revisit if**: [conditions that would reopen this decision]
```

---

### D001: Adopt three-angle unconventional strategy
**Date**: 2026-03-22
**Context**: Needed to choose overall competitive approach for the Nemotron
Reasoning Challenge.
**Options considered**:
1. Standard playbook only (prompt eng → synthetic data → fine-tune → ensemble) —
   Safe, well-documented, but identical to what most competitors will do
2. Fully unconventional (symbolic solver + judge optimization + cognitive traces) —
   Higher ceiling but risky if assumptions are wrong
3. Hybrid: standard playbook as baseline, invest heavily in unconventional edges —
   Captures both safety and upside
**Decision**: Option 3 — Hybrid approach
**Rationale**: The standard playbook provides a floor. The unconventional edges
provide differentiation. Phase 0 intelligence gathering gates the unconventional
investments, so if assumptions are wrong we've lost time but not compute.
**Assumptions this depends on**: A1 (transformation rules), B2 (verbosity penalty),
B3 (correctness dominant)
**Revisit if**: Phase 0 reveals benchmark is fundamentally different from expected

### D002: Phase 0 gates all optimization work
**Date**: 2026-03-22
**Context**: Multiple optimization tracks available; limited time and compute.
**Options considered**:
1. Start optimizing immediately based on best-guess assumptions
2. Invest first week purely in understanding benchmark + judge, then optimize
**Decision**: Option 2 — Intelligence-first approach
**Rationale**: All three unconventional edges depend on specific assumptions about
benchmark structure and judge behavior. Premature optimization risks investing
heavily in the wrong direction. The AIMO-2 and ARC Prize winners all started with
deep benchmark analysis before technique selection.
**Assumptions this depends on**: None — this is our most robust decision
**Revisit if**: Competition deadline is much sooner than expected (< 4 weeks)

### D003: Symbolic solver as primary track, neural as fallback
**Date**: 2026-03-22
**Context**: Need to prioritize between symbolic and neural approaches.
**Options considered**:
1. Neural-first: Improve Nano's reasoning directly
2. Symbolic-first: Solve programmatically, then wrap in traces
3. Equal priority: Run both tracks at equal investment
**Decision**: Option 2 — Symbolic-first
**Rationale**: If correctness weight is indeed +0.80, then being right is worth
more than reasoning beautifully. Symbolic solvers produce verified correct answers.
Neural reasoning is stochastic. However, this decision is heavily conditional on
A1 (puzzles being transformation rules) and will be revisited after Phase 0.
**Assumptions this depends on**: A1, A3, B3
**Revisit if**: A1 busted (not transformation rules), A3 busted (rule space not
enumerable), or B3 busted (correctness not dominant in scoring)

---

*(Add new decisions below as the project progresses)*
