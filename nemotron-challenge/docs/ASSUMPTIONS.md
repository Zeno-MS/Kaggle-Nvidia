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
**Status**: UNTESTED
**Confidence**: MEDIUM
**Source**: Competition description says "logical reasoning puzzles requiring
identification and application of underlying transformation rules"
**Test**: Download CSV, inspect puzzle structure
**Impact if busted**: Abandon symbolic solver track (Track A); restructure
entire pipeline around neural reasoning
**Notes**: The phrase "transformation rules" strongly suggests ARC-like structure,
but could be much broader than grid transforms.

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
**Status**: UNTESTED
**Confidence**: HIGH
**Source**: Competition references "transformation rules" implying example-based
rule discovery; ARC-like structure
**Test**: Inspect CSV structure
**Impact if busted**: Major — fundamentally changes the task from rule induction
to something else

### A5. The dataset is ~3MB CSV as reported
**Status**: UNTESTED
**Confidence**: HIGH
**Source**: Implementation plan references "~3MB CSV licensed CC BY 4.0"
**Test**: Download and check
**Impact if busted**: LOW — size doesn't change strategy much

---

## Category: Judge Model Behavior

### B1. The judge is Nemotron-3-Nano-30B-A3B-BF16
**Status**: UNTESTED (but high confidence)
**Confidence**: HIGH
**Source**: Kaggle metric model identifier: `metric/nemotron-3-nano-30b-a3b-bf16`
**Test**: Confirm on Kaggle evaluation page
**Impact if busted**: Moderate — would need to re-profile judge preferences

### B2. The judge penalizes verbosity
**Status**: UNTESTED
**Confidence**: MEDIUM
**Source**: HelpSteer2 reward model weights show -0.40 on verbosity dimension;
Nemotron 3 Nano trained with HelpSteer-derived rewards
**Test**: Submit identical correct answers with varying trace lengths;
measure score differences
**Impact if busted**: Remove conciseness constraint; allow longer traces;
re-optimize trace template for different preference

### B3. Correctness is the dominant scoring factor
**Status**: UNTESTED
**Confidence**: MEDIUM
**Source**: HelpSteer2 weights show +0.80 on correctness
**Test**: Submit (a) correct answer with poor trace vs (b) wrong answer with
excellent trace; compare scores
**Impact if busted**: HIGH — if trace quality > correctness, entire pipeline
priority shifts from "solve correctly" to "reason beautifully"

### B4. The judge evaluates both reasoning trace AND final answer
**Status**: UNTESTED
**Confidence**: HIGH
**Source**: Nemotron Nano architecture produces reasoning trace + final response;
metric described as scoring "reasoning quality and answer correctness"
**Test**: Read Kaggle evaluation page; submit with and without traces
**Impact if busted**: If answer-only, simplify pipeline dramatically

### B5. The judge has self-preference bias for Nemotron-style text
**Status**: UNTESTED
**Confidence**: MEDIUM
**Source**: Self-preference bias is documented across LLM judges (arXiv 2410.21819);
models prefer text with lower perplexity relative to their own distribution
**Test**: Compare scores for traces in Nemotron's native reasoning style vs
generic CoT style vs academic style
**Impact if busted**: LOW — still useful to match style for other reasons

### B6. Evaluation parameters (temperature, top-p, max_tokens) are configurable
**Status**: UNTESTED
**Confidence**: HIGH
**Source**: Kaggle discussion thread mentions "parameters on the Evaluation page
override default code values"
**Test**: Check Kaggle evaluation page
**Impact if busted**: LOW — just means we can't tune these

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
