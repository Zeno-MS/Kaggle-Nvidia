# Living Strategy Document

> **Last updated**: 2026-03-22
> **Current phase**: Phase 0 — Intelligence Gathering
> **Confidence level**: LOW — benchmark unseen, judge behavior unverified
> **Next milestone**: Complete benchmark profiling and first judge characterization

---

## Strategic Overview

This strategy has three layers that execute in priority order:

```
┌─────────────────────────────────────────────┐
│  Layer 1: KNOW THE GAME                     │
│  Understand benchmark structure + judge      │
│  behavior before optimizing anything         │
├─────────────────────────────────────────────┤
│  Layer 2: UNCONVENTIONAL EDGES              │
│  Symbolic solving, judge alignment,          │
│  cognitive trace design                      │
├─────────────────────────────────────────────┤
│  Layer 3: STANDARD PLAYBOOK (baseline)      │
│  Prompt engineering, synthetic data,         │
│  fine-tuning, self-consistency               │
└─────────────────────────────────────────────┘
```

Layer 1 gates everything. Without understanding the benchmark and judge, all
optimization is premature. Layers 2 and 3 run in parallel once Layer 1 reaches
sufficient confidence.

---

## Phase 0: Intelligence Gathering (CURRENT)

**Goal**: Reach HIGH confidence on benchmark structure and MEDIUM confidence on
judge behavior before committing compute to any optimization track.

### 0.1 — Benchmark Reconnaissance

- [ ] Download and inspect the competition CSV
- [ ] Profile puzzle types: string transforms? grid transforms? arithmetic? symbolic?
- [ ] Identify input/output representation format
- [ ] Cluster puzzles by apparent rule family
- [ ] Estimate difficulty distribution
- [ ] Determine: is this ARC-like, math-like, or something else entirely?
- [ ] Check: are puzzles deterministic (one correct answer) or open-ended?
- [ ] Document findings in `KNOWLEDGE_BASE.md` § Benchmark

**Key question**: What is the *structure* of the latent rule space?
If it's finite and enumerable → symbolic solving is dominant.
If it's open-ended → neural reasoning becomes primary.

### 0.2 — Judge Characterization

- [ ] Read the Kaggle evaluation page parameters (temperature, max_tokens, top_p)
- [ ] Read any Kaggle discussion threads about metric behavior
- [ ] Inspect the metric model code/config if accessible
- [ ] Run baseline Nemotron Nano on training data; record scores
- [ ] Generate deliberately varied traces for the same correct answer:
  - Minimal (answer only)
  - Short (3-step reasoning)
  - Medium (7-step reasoning with verification)
  - Long (15+ step verbose reasoning)
  - Measure: which scores highest?
- [ ] Test format sensitivity: markdown vs plain text, numbered steps vs prose
- [ ] Test: does an incorrect answer with beautiful reasoning outscore a
  correct answer with poor reasoning? (Tests correctness weight)
- [ ] Document findings in `KNOWLEDGE_BASE.md` § Judge

**Key question**: Is the metric *primarily* scoring answer correctness, or does
reasoning trace quality dominate? This determines the entire strategy allocation.

### 0.3 — Baseline Establishment

- [ ] Run unmodified Nemotron 3 Nano (reasoning ON) on full training set
- [ ] Record per-puzzle scores and aggregate
- [ ] Run with reasoning OFF — measure delta
- [ ] Identify the top-20 easiest and top-20 hardest puzzles
- [ ] Error-categorize failures: wrong answer? right idea wrong execution?
  format issue? complete misunderstanding?
- [ ] Establish the baseline number to beat

---

## Phase 1: Parallel Track Execution

*Begins when Phase 0 reaches sufficient confidence. Tracks run in parallel.*

### Track A — Symbolic Solver (High Priority)

**Thesis**: If puzzles have discoverable transformation rules, a symbolic solver
produces provably correct answers — the highest-value output given correctness
weight of +0.80.

**Implementation path** (adapt based on Phase 0 findings):

```
A.1  Define a DSL vocabulary matching discovered puzzle types
A.2  Implement brute-force enumeration over short DSL programs
A.3  Add LLM-guided program synthesis (generate Python, test, refine)
A.4  Build verification: does candidate program reproduce all training examples?
A.5  Portfolio: run multiple solvers in parallel, take first verified solution
```

**Success metric**: Coverage — what % of puzzles can the symbolic solver verify?
**Pivot trigger**: If coverage < 20% after reasonable effort, deprioritize and
reallocate compute to Track C.

### Track B — Judge-Aligned Trace Generation (High Priority)

**Thesis**: Given a correct answer (from Track A or Track C), the trace wrapper
can add significant score by satisfying judge preferences.

**Implementation path**:

```
B.1  Build the trace template (see Trace Architecture section below)
B.2  Implement execution-trace narration: symbolic solution → NL trace
B.3  Test trace variants against judge; identify scoring sweet spots
B.4  Build best-of-N selection: generate K trace variants, score with judge
B.5  If judge accessible as reward signal, implement GRPO trace optimization
```

**Success metric**: Score delta between raw correct answer and trace-wrapped answer.
**Pivot trigger**: If trace quality has < 0.05 score impact, simplify to minimal
trace and reallocate effort to Track A/C.

### Track C — Neural Reasoning (Standard Playbook, Lower Priority)

**Thesis**: For puzzles the symbolic solver can't crack, structured neural
reasoning with self-consistency provides the fallback.

**Implementation path**:

```
C.1  Prompt engineering: few-shot with trace template structure
C.2  Self-consistency: sample N reasoning paths, majority vote on answer
C.3  Synthetic data generation from teacher models (Nemotron Super, DeepSeek-R1)
C.4  LoRA fine-tuning on synthetic + competition data
C.5  Test-time training on per-puzzle examples (if compute allows)
```

**Success metric**: Accuracy on puzzles where symbolic solver fails.
**Pivot trigger**: If Phase 0 reveals the benchmark is entirely open-ended
reasoning (not transformation rules), promote this to top priority.

---

## Phase 2: Integration and Optimization

**Goal**: Compose all tracks into the unified pipeline and optimize end-to-end.

```
Phase 2.1  Wire orchestrator: symbolic → trace → judge → fallback
Phase 2.2  Profile inference time budget per puzzle
Phase 2.3  Implement adaptive compute allocation (more samples for harder puzzles)
Phase 2.4  Build offline evaluation harness with surrogate judge
Phase 2.5  Systematic hyperparameter search (temperature, top_p, sample count, 
           reasoning budget, trace template variants)
```

---

## Phase 3: Submission and Iteration

**Goal**: Manage limited submission slots wisely.

```
Phase 3.1  Establish submission cadence (test major changes only)
Phase 3.2  Monitor leaderboard vs offline surrogate correlation
Phase 3.3  Diagnose per-puzzle regressions after each submission
Phase 3.4  Update strategy based on public leaderboard dynamics
Phase 3.5  Reserve final submissions for best-performing configuration
```

---

## Trace Architecture (Template v1)

This is the target structure for every reasoning trace, designed to maximize
Nemotron's HelpSteer scores. Adapt based on Phase 0 judge characterization.

```
[COMPREHENSION] — 1-2 sentences
Restate the problem in your own words. Demonstrate you understand what
transformation is being asked for.
→ Maps to: Helpfulness (+0.65)

[STRATEGY] — 1 sentence
State the approach: "I'll compare input-output pairs to identify the rule."
→ Maps to: Coherence (+0.45)

[DERIVATION] — 3-7 steps, one logical move per step
Step 1: Observe [specific pattern] in Example 1...
Step 2: Verify this pattern holds in Example 2...
Step 3: Therefore, the rule appears to be [rule]...
→ Maps to: Correctness (+0.80), Complexity (+0.55)

[VERIFICATION] — 1-2 sentences
"Applying this rule to Example N: [input] → [expected output] ✓"
→ Maps to: Correctness (+0.80)

[ANSWER] — 1-2 sentences
Apply discovered rule to test input. State final answer explicitly.
→ Maps to: Helpfulness (+0.65)
```

**Anti-patterns to avoid** (given -0.40 verbosity weight):
- No preamble ("Sure, I'd be happy to help...")
- No restating already-established facts
- No hedging language ("It might be...", "Perhaps...")
- No meta-commentary about the reasoning process itself
- No unnecessary examples beyond what's needed for verification

---

## Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| Benchmark is nothing like ARC/transformation puzzles | HIGH | MEDIUM | Phase 0 gates all work; pivot plan in place |
| Judge doesn't actually penalize verbosity | MEDIUM | MEDIUM | Phase 0 judge characterization will test this directly |
| Symbolic solver coverage too low to matter | HIGH | MEDIUM | Track C (neural) always runs as fallback |
| Fine-tuning hurts judge alignment (model drift) | MEDIUM | LOW | Test fine-tuned vs base on judge before committing |
| Compute insufficient for TTT or RL | MEDIUM | HIGH | All tracks have compute-light variants |
| Competition rules prohibit external model calls | LOW | LOW | All tracks work with local Nano inference only |

---

## Pivot Conditions

These are pre-committed decision rules. If a condition triggers, execute the
corresponding pivot without deliberation.

| Condition | Pivot |
|-----------|-------|
| Benchmark is NOT transformation-rule puzzles | Abandon Track A; promote Track C to primary; adapt trace template for open-ended reasoning |
| Judge verbosity penalty is NOT confirmed | Remove conciseness constraint; allow longer traces; re-optimize template |
| Symbolic solver achieves > 60% coverage | Go all-in on Track A + B; Track C only for residual puzzles |
| Symbolic solver achieves < 10% coverage | Deprioritize Track A; focus on Track B + C |
| Leaderboard leader uses a simple approach | Study their method; consider whether we're over-engineering |
| Fine-tuned model scores WORSE than base on judge | Revert to base Nano; invest all compute in inference-time optimization |

---

## Resource Allocation (Current)

| Resource | Allocation |
|----------|-----------|
| Your time | 80% Phase 0 intelligence gathering, 20% pipeline skeleton |
| Colab Pro | Benchmark profiling, baseline runs, judge probing |
| Kaggle submissions | SAVE — do not burn on early experiments |

*This allocation will shift as phases progress. Update when entering Phase 1.*

---

## Changelog

| Date | Change | Reason |
|------|--------|--------|
| 2026-03-22 | Initial strategy created | Project kickoff |
