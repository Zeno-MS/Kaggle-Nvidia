# Assumptions Tracker

> Every assumption that could change our strategy lives here.
> Each one has a confidence level, a test plan, and a status.
> Review this document at the start of every work session.

---

## How to Use

- **UNTESTED**: We believe this but haven't verified it
- **TESTING**: Experiment designed or in progress
- **CONFIRMED**: Verified with evidence (cite the evidence)
- **BUSTED**: Disproven — trigger strategy update
- **REVISED**: Original assumption was wrong but a corrected version is confirmed

---

## Category: Benchmark Structure

### A1. Puzzles are logical transformation rules
**Status**: REVISED
**Confidence**: HIGH
**Source**: Metric code reveals `prompt` + `answer` CSV format. Nemotron-RL-ReasoningGym-v1
dataset (NVIDIA's open reasoning benchmark) covers algebra, arithmetic, logic, graph
theory, games, geometry, computation, statistics, and string tasks. NOT grid transforms.
**Evidence**: Metric `extract_final_answer()` looks for `\boxed{}` or numeric answers.
`verify()` does numeric tolerance or string comparison. This is math/reasoning, not ARC.
**Revised assumption**: Puzzles are diverse text-based reasoning problems (math, logic,
games, etc.) with verifiable answers. Not transformation-rule puzzles in the ARC sense.
**Impact**: Symbolic solver as standalone track is not viable. But symbolic/programmatic
solving of specific problem types (algebra, arithmetic) can generate **training data**
for the LoRA fine-tune.

### A2. Puzzles have deterministic correct answers
**Status**: UNTESTED
**Confidence**: HIGH
**Source**: Competition uses accuracy-style metric; CSV-based dataset
**Test**: Check if each puzzle has exactly one correct output
**Impact if busted**: Moderate — would need to handle multiple valid answers
in scoring

### A3. The rule space is finite and structured
**Status**: UNTESTED
**Confidence**: MEDIUM
**Source**: Inference from "novel benchmark from NVIDIA Research" — research
benchmarks typically have controlled generation processes
**Test**: Cluster puzzles; check if small set of primitive operations can explain
most puzzles
**Impact if busted**: Symbolic enumeration becomes intractable; shift to
LLM-guided program synthesis only

### A4. Puzzles come with training examples (few-shot format)
**Status**: CONFIRMED (examples embedded in prompt text)
**Confidence**: CONFIRMED
**Source**: train.csv inspection (2026-03-26). Each prompt contains 3-11 input→output
examples embedded in the prompt text, followed by a "Now, determine the output for:" query.
**Evidence**: All 9,500 prompts follow pattern: description → examples → query.
The CSV is flat (one prompt per row) but the prompt itself contains the few-shot examples.
**Impact**: MAJOR — these are genuine rule-induction puzzles. The model must discover the
hidden transformation rule from examples and apply it. Symbolic solvers CAN solve these
by analyzing the examples programmatically → perfect training data at scale.

### A5. The dataset is ~3MB CSV as reported
**Status**: UNTESTED
**Confidence**: HIGH
**Source**: Implementation plan references "~3MB CSV licensed CC BY 4.0"
**Test**: Download and check
**Impact if busted**: LOW — size doesn't change strategy much

---

## Category: Judge Model Behavior

### B1. The judge is Nemotron-3-Nano-30B-A3B-BF16
**Status**: CONFIRMED (but role is different than assumed)
**Confidence**: CONFIRMED
**Source**: Metric source code: `MODEL_PATH = kagglehub.model_download('metric/nemotron-3-nano-30b-a3b-bf16/transformers/default')`
**Evidence**: The model is NOT a judge — it IS the inference model. Your LoRA adapter
is loaded onto this base model via vLLM. The model generates answers which are then
checked against ground truth. There is no separate "judge" model.
**Impact**: The entire "judge optimization" track is moot. There is no judge to optimize for.

### B2. The judge penalizes verbosity
**Status**: BUSTED
**Confidence**: CONFIRMED BUSTED
**Source**: Metric source code — pure accuracy scoring via `verify()`.
**Evidence**: `accuracy = num_correct / len(solution)`. No trace evaluation at all.
Only the extracted `\boxed{}` content is compared to ground truth.
**Impact**: No verbosity penalty exists. The model should reason freely (thinking
is enabled via chat template). Longer reasoning traces may actually help accuracy
by enabling better chain-of-thought. The only thing that matters is getting the
right answer inside `\boxed{}`.

### B3. Correctness is the ONLY scoring factor
**Status**: CONFIRMED (stronger than assumed)
**Confidence**: CONFIRMED
**Source**: Metric source code — `score()` returns pure accuracy.
**Evidence**: No trace evaluation. No partial credit. Binary correct/incorrect per problem.
**Impact**: Strategy is 100% about maximizing the number of correctly answered problems.
Reasoning quality only matters insofar as it leads to correct final answers.

### B4. The judge evaluates both reasoning trace AND final answer
**Status**: BUSTED
**Confidence**: CONFIRMED BUSTED
**Source**: Metric source code — `extract_final_answer()` only extracts `\boxed{}` content.
**Evidence**: The model generates with `enable_thinking=True` (reasoning traces produced),
but only the final `\boxed{}` answer is extracted and verified. Traces are discarded.
**Impact**: Reasoning traces help the MODEL think better but are not directly scored.
The LoRA should be trained to produce good reasoning that leads to correct answers,
but trace formatting is irrelevant to scoring.

### B5. The judge has self-preference bias for Nemotron-style text
**Status**: BUSTED (not applicable)
**Confidence**: CONFIRMED BUSTED
**Source**: No judge exists. Pure accuracy metric.
**Evidence**: There is no LLM-as-judge evaluation. The metric is `verify()` which
does string/numeric comparison against ground truth answers.
**Impact**: None — this entire line of investigation was based on a false premise.

### B6. Evaluation parameters (temperature, top-p, max_tokens) are configurable
**Status**: CONFIRMED (but we don't control them)
**Confidence**: CONFIRMED
**Source**: Metric `score()` function accepts these as kwargs with defaults.
Kaggle evaluation page may override defaults.
**Evidence**: Defaults are `temperature=1.0, top_p=1.0, max_tokens=3584, max_model_len=4096`.
These are set by the competition, not by participants.
**Impact**: We must train the LoRA to perform well at temperature=1.0 (stochastic sampling).
This means the model needs robust reasoning — can't rely on greedy decoding consistency.
Training with temperature sampling may help.

---

## Category: Competition Mechanics

### C1. Offline training on external compute is allowed
**Status**: UNTESTED
**Confidence**: HIGH
**Source**: Standard Kaggle rules; implementation plan states "model weights
uploaded as datasets"
**Test**: Read competition rules page
**Impact if busted**: HIGH — constrains all training to Kaggle G4 VMs

### C2. The competition runs ~3 months
**Status**: UNTESTED
**Confidence**: HIGH
**Source**: Implementation plan reference; typical Kaggle featured competition
**Test**: Check competition deadline on Kaggle page
**Impact if busted**: LOW — adjust timeline but not strategy

### C3. Submissions are limited (typical Kaggle: 2-5 per day)
**Status**: UNTESTED
**Confidence**: HIGH
**Source**: Standard Kaggle practice
**Test**: Check competition rules
**Impact if busted**: LOW — just adjust submission cadence

### C4. We can use the Nemotron 3 Super model as a teacher
**Status**: UNTESTED
**Confidence**: HIGH
**Source**: Super is open-weights; NIM API available; competition rules
presumably allow open models for training
**Test**: Confirm in competition rules
**Impact if busted**: MEDIUM — lose best teacher model; fall back to other
open models (DeepSeek-R1, Qwen)

---

## Category: Technical

### D1. Nemotron 3 Nano fits on Kaggle G4 VM in FP8
**Status**: UNTESTED
**Confidence**: HIGH
**Source**: Model designed for 24GB VRAM consumer GPUs; G4 VMs have RTX-class
GPUs; FP8 quantization available
**Test**: Load model on Kaggle; measure VRAM usage
**Impact if busted**: HIGH — would need different quantization or smaller model

### D2. Symbolic solvers (Python, Prolog) can run in Kaggle environment
**Status**: UNTESTED
**Confidence**: HIGH
**Source**: Kaggle kernels support Python; Prolog via pip installable SWI-Prolog
bindings
**Test**: Try installing and running in Kaggle kernel
**Impact if busted**: MEDIUM — would need Python-only symbolic approaches

### D3. Multiple inference calls per puzzle fit within time budget
**Status**: UNTESTED
**Confidence**: MEDIUM
**Source**: AIMO-2 used 4-16 parallel generations per problem on L4 GPUs
**Test**: Profile Nemotron Nano inference time per puzzle; calculate budget
**Impact if busted**: HIGH — limits self-consistency, best-of-N, and
multi-solver portfolio approaches

---

## Assumption Dependency Map

```
A1 (transformation rules) ──→ Track A viability
    ├── A3 (finite rule space) ──→ Enumeration approach
    └── A4 (training examples) ──→ Verification possible

B1 (judge identity) ──→ All judge optimization
    ├── B2 (verbosity penalty) ──→ Trace template design
    ├── B3 (correctness dominant) ──→ Pipeline priority order
    └── B5 (self-preference) ──→ Style matching strategy

D1 (fits on G4) ──→ All inference-time approaches
    └── D3 (time budget) ──→ Multi-sample viability
```

---

## Review Log

| Date | Assumptions Reviewed | Changes |
|------|---------------------|---------|
| 2026-03-22 | All | Initial creation |
| 2026-03-26 | A1,A4,B1-B6 | Major revision from metric source code. A1 REVISED (not grid transforms), A4 BUSTED (no few-shot), B1 CONFIRMED (model not judge), B2 BUSTED (no verbosity penalty), B3 CONFIRMED (accuracy only), B4 BUSTED (traces not scored), B5 BUSTED (no judge), B6 CONFIRMED (params fixed by competition). Strategy pivot required. |
