# Living Strategy Document

> **Last updated**: 2026-03-26
> **Current phase**: Phase 0 — Data Access + Baseline (BLOCKED on competition join)
> **Confidence level**: MEDIUM — metric mechanics confirmed, benchmark unseen
> **Next milestone**: Accept competition rules → download data → profile → baseline
> **Deadline**: 2026-06-15 (81 days)

---

## What We Now Know (Phase 0 Intel — 2026-03-26)

**The competition is a LoRA fine-tuning challenge.**

You submit a rank-32 LoRA adapter. The metric code loads it onto Nemotron-3-Nano-30B
via vLLM, generates answers to test prompts with `enable_thinking=True`, extracts
the `\boxed{}` answer, and compares to ground truth. Score = fraction correct.

This means:
- **No judge model** — pure accuracy
- **No trace scoring** — only `\boxed{}` content matters
- **No inference-time tricks** — single forward pass, temperature=1.0
- **One lever**: make the LoRA adapter improve Nano's reasoning accuracy

### Busted Hypotheses
- ~~HelpSteer judge scoring~~ → pure accuracy
- ~~Trace optimization (Track B)~~ → traces not evaluated
- ~~Symbolic solver as submission strategy~~ → must submit LoRA
- ~~Conciseness optimization~~ → no verbosity penalty
- ~~ARC-like grid transforms~~ → text-based reasoning problems

---

## Revised Strategic Overview

```
┌─────────────────────────────────────────────┐
│  Layer 1: KNOW THE BENCHMARK                │
│  Profile problem types, difficulty, baseline │
│  accuracy. Identify where Nano fails.        │
├─────────────────────────────────────────────┤
│  Layer 2: DATA QUALITY                      │
│  Synthetic data generation, data curation,   │
│  curriculum design for LoRA training         │
├─────────────────────────────────────────────┤
│  Layer 3: TRAINING OPTIMIZATION             │
│  LoRA hyperparams, learning rate, epochs,    │
│  loss functions, evaluation harness          │
└─────────────────────────────────────────────┘
```

Layer 1 still gates everything. Layers 2 and 3 iterate together.

---

## Phase 0: Intelligence Gathering (CURRENT — BLOCKED)

**Blocker**: Must accept competition rules on Kaggle to download data.
Account: `stoicknight`. URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge

### 0.1 — Data Access
- [ ] Accept competition rules on Kaggle
- [ ] Download train.csv and test.csv
- [ ] Inspect schema: confirm `prompt` + `answer` columns
- [ ] Count problems, check answer types (numeric, string, mixed)

### 0.2 — Benchmark Profiling
- [ ] Categorize every problem by type (math, logic, games, graph, etc.)
- [ ] Distribution of problem categories
- [ ] Answer format distribution (numeric vs string vs complex)
- [ ] Prompt length distribution
- [ ] Identify problem families that share structure

### 0.3 — Baseline Establishment
- [ ] Run unmodified Nano (no LoRA) on train set — record per-problem accuracy
- [ ] Run with a trivial/identity LoRA — confirm submission pipeline works
- [ ] Categorize failures: wrong reasoning? format issues? knowledge gaps?
- [ ] Rank problem categories by difficulty (accuracy per category)
- [ ] Identify the "free points" (easy categories) and "hard points" (low accuracy)

### 0.4 — Competitive Intel
- [ ] Check public leaderboard scores (current top: ~0.6 range per knowledge base)
- [ ] Read Kaggle discussion forums for hints
- [ ] Review public notebooks for approaches
- [ ] Estimate: what accuracy would place top-10? top-50?

---

## Phase 1: Data Engine

**Goal**: Build the highest-quality training dataset for LoRA fine-tuning.

### Track A — Synthetic Data Generation (HIGH PRIORITY)

**Thesis**: The LoRA needs to see many more reasoning problems than the train set
provides. Generate synthetic problems in the same distribution as the benchmark.

```
A.1  Categorize train problems into families
A.2  For each family, use teacher models to generate similar problems:
     - Nemotron-3-Super-120B (best open teacher)
     - DeepSeek-R1 (strong reasoning)
     - Qwen-2.5-72B (diverse generation)
A.3  Verify synthetic answers programmatically where possible:
     - Math: evaluate with sympy/sage
     - Logic: formal verification
     - Games: simulate (Tower of Hanoi, etc.)
A.4  Filter: keep only verified-correct synthetic data
A.5  Scale: 10x-100x the competition train set
```

**Key insight**: The symbolic solver code already written can be repurposed as a
**verification engine** for synthetic data. Generate candidate problems + solutions
with LLMs, verify with symbolic execution, keep only correct ones.

### Track B — Data Curation (MEDIUM PRIORITY)

```
B.1  Clean train data: check for ambiguous prompts, multiple valid answers
B.2  Augment: rephrase problems while preserving answers
B.3  Difficulty-aware sampling: oversample hard problem types
B.4  Format training data with proper chat template + \boxed{} answers
B.5  Include chain-of-thought reasoning in training targets
```

### Track C — External Data (MEDIUM PRIORITY)

Leverage existing reasoning datasets:
- `nvidia/Nemotron-RL-ReasoningGym-v1` (15K samples, 104 environments)
- `nvidia/Puzzle-KD-Nemotron-Post-Training-Dataset-v2` (851K samples)
- MATH dataset, GSM8K, AIME problems
- ARC-style puzzle datasets (if benchmark includes these)
- Competition-specific: filter external data to match benchmark distribution

---

## Phase 2: LoRA Training

**Goal**: Train the best rank-32 LoRA adapter.

### 2.1 — Training Infrastructure
```
- Set up training on Colab Pro / local GPU
- Use PEFT library (same as demo notebook)
- Target modules: in_proj, out_proj, up_proj, down_proj
- Match demo: rank=32, alpha=16, dropout=0.05
```

### 2.2 — Training Strategy
```
- Baseline: SFT on competition train data with CoT + \boxed{} format
- Add synthetic data progressively (measure accuracy gain per batch)
- Learning rate sweep: 1e-5 to 5e-4
- Epochs: monitor for overfitting (small test set means high overfit risk)
- Loss: standard causal LM loss on full response (reasoning + answer)
```

### 2.3 — Evaluation Harness
```
- Replicate metric locally: vLLM + LoRA loading + extract_final_answer()
- Use competition's exact verify() function
- Hold out validation split from train data
- Track accuracy per problem category
- Must test at temperature=1.0 (matches eval) — run multiple seeds
```

### 2.4 — Advanced Training (Phase 2+)
```
- RLVR: Reinforcement Learning from Verifiable Rewards
  - Use symbolic verifiers as reward signal
  - GRPO or PPO on reasoning traces
- Curriculum learning: easy → hard problem ordering
- DPO: generate correct + incorrect reasoning pairs, train preference
- Test-time training: if compute allows, fine-tune per-problem-family
```

---

## Phase 3: Submission and Iteration

```
3.1  First submission: baseline LoRA (SFT on train data only)
3.2  Monitor leaderboard position vs offline eval correlation
3.3  Iterate: add synthetic data, retrain, submit
3.4  Track accuracy by problem category across submissions
3.5  Reserve final submissions for best configuration
3.6  Ensemble exploration: can we benefit from multiple LoRA training runs?
```

---

## Competitive Edges (Revised)

### Edge 1: Verified Synthetic Data (Symbolic Verification)
Most competitors will generate synthetic data with LLMs and hope it's correct.
We can **verify** math/logic/game solutions programmatically, ensuring our training
data has zero label noise. Higher quality training data → better LoRA.

### Edge 2: Problem-Category-Aware Training
By profiling which categories Nano struggles with, we can over-represent those
categories in training data. Targeted improvement on weak areas.

### Edge 3: Evaluation at Temperature=1.0
Many competitors will evaluate their LoRA at temperature=0 (greedy) during
development. The competition evaluates at temperature=1.0. Training and evaluating
with the correct temperature matters.

### Edge 4: Reasoning Trace Quality in Training Data
While traces aren't scored, they affect the model's ability to reach correct answers.
Training with high-quality step-by-step reasoning (from teacher models) should
improve the LoRA's reasoning path quality → higher accuracy.

---

## Risk Register (Revised)

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| Benchmark is much harder than ReasoningGym | HIGH | MEDIUM | Phase 0 profiling; adjust expectations early |
| Overfitting to small train set | HIGH | HIGH | Hold out validation; use synthetic data; regularization |
| Synthetic data doesn't match benchmark distribution | MEDIUM | MEDIUM | Profile benchmark first; filter synthetic data to match |
| Compute insufficient for large-scale RL training | MEDIUM | HIGH | Start with SFT; use RLVR only if SFT plateaus |
| LoRA rank 32 is too constrained | MEDIUM | LOW | Optimize which layers to target; alpha tuning |
| Temperature=1.0 causes high variance in scoring | MEDIUM | HIGH | Evaluate with multiple seeds; train for robustness |

---

## Pivot Conditions (Revised)

| Condition | Pivot |
|-----------|-------|
| Baseline Nano accuracy > 70% | Focus on hard problems only; targeted synthetic data |
| Baseline Nano accuracy < 30% | The model needs fundamental capability; large-scale SFT + RL |
| Benchmark is mostly math | Prioritize MATH/GSM8K-style synthetic data + sympy verification |
| Benchmark is mostly logic/games | Prioritize programmatic problem generators + simulators |
| SFT on train data alone reaches top-100 | Double down on data quality; skip RL complexity |
| Top leaderboard scores > 0.9 | Competition is about marginal gains; focus on hard tail |
| Top leaderboard scores < 0.5 | Competition is fundamentally hard; novel approaches needed |

---

## Resource Allocation (Current)

| Resource | Allocation |
|----------|-----------|
| Time | 100% Phase 0 — cannot proceed without data |
| Compute | None committed yet — waiting on benchmark profile |
| Kaggle submissions | SAVE until baseline is established |

---

## Changelog

| Date | Change | Reason |
|------|--------|--------|
| 2026-03-22 | Initial strategy created | Project kickoff |
| 2026-03-26 | **MAJOR REWRITE** | Metric source code analysis revealed pure accuracy scoring, LoRA submission format. HelpSteer judge hypothesis busted. Tracks A (symbolic solver) and B (trace optimization) repurposed. Strategy now centered on LoRA fine-tuning with verified synthetic data. |
