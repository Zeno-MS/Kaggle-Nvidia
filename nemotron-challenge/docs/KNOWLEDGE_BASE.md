# Knowledge Base

> This document accumulates **verified facts** about the competition.
> Everything here should be backed by evidence. Speculation goes in ASSUMPTIONS.md.
> Update this continuously as you learn.

---

## § Benchmark

### What We Know (Pre-Phase 0)

**Source: Competition Page**
- Dataset is CSV-based, ~3MB, licensed CC BY 4.0
- Described as "logical reasoning puzzles requiring identification and application
  of underlying transformation rules"
- Novel benchmark developed by NVIDIA Research (not a public benchmark)

**Source: Google Cloud Blog (GTC 2026)**
- "Improve Nemotron 3 Nano's reasoning accuracy on a new benchmark"
- Techniques permitted: "prompting, synthetic data generation, data curation,
  and fine-tuning"
- Also mentioned: reinforcement learning, lightweight fine-tuning

**Source: Luma Event Page**
- "Develop techniques that improve reasoning accuracy, evaluated using Nemotron
  models and a new reasoning benchmark developed by the NVIDIA research team"
- "You'll be free to explore your own approaches: prompting, data filtering,
  synthetic data generation, reinforcement learning, and lightweight fine-tuning"

### What We Don't Know Yet
- Exact puzzle format (grid? text? symbolic?)
- Number of puzzles in train/test
- Whether puzzles include training examples per task (few-shot) or just problems
- The specific transformation families represented
- Difficulty distribution
- Whether there's a public test set or only private

### Phase 0 Findings (2026-03-26)

**Puzzle format**: Text-based reasoning prompts in CSV (`prompt` column + `answer` column)
**Number of puzzles**: train.csv=3MB (~large set), test.csv=1.4KB (~small hidden test)
**Input representation**: Natural language prompts (reasoning-gym style)
**Output representation**: Answers extracted from `\boxed{}` — numeric or string
**Training examples per puzzle**: NO — each row is a standalone prompt/answer pair (not ARC-style few-shot)
**Puzzle families identified** (inferred from Nemotron-RL-ReasoningGym-v1):
- Algebra (linear equations, cryptarithmetic)
- Arithmetic (chains, leg counting)
- Logic (self-reference, needle-in-haystack, blicket)
- Graph theory (shortest path, friends graph)
- Games (Tower of Hanoi, Sokoban, Knight Swap)
- Geometry (orthocenters, coordinates)
- Computation (base conversion, bitwise ops, matrix)
- Statistics/probability (dice, combinatorics)
- Text/string (palindromes, synthesis rules)

**Train size**: 9,500 problems (3MB CSV)
**Test size**: 3 problems (public test — real test is hidden/private)
**Problem structure**: ALL problems follow "Alice's Wonderland" template:
  description → few-shot examples → "Now, determine/convert/decrypt:" → query

**6 categories, perfectly balanced (~16.5% each)**:

| Category | Count | Symbolic Accuracy | Notes |
|----------|-------|-------------------|-------|
| bit_manipulation | 1,602 | 7.1% | Complex multi-op transforms, not simple XOR/rotate |
| gravitational_constant | 1,597 | 97.9% | d = 0.5*g*t^2, infer g from examples |
| unit_conversion | 1,594 | 99.4% | y = a*x (pure proportional), linear regression |
| cipher/encryption | 1,576 | 38.4% | Unique substitution cipher per problem, limited by unseen chars |
| numeral_system | 1,576 | 100% | Decimal → Roman numeral (all problems) |
| symbol_transform | 1,555 | TBD | Symbol/character equation transforms |

**Symbolic solver total**: ~57% of train solvable programmatically
**Answer format**: 57.7% numeric, 25.7% string/mixed, 16.6% uppercase (Roman)
**Prompt length**: min 177, max 510, median 281, mean 302 chars
**Answer length**: min 1, max 39, median 5, mean 8 chars
**Each cipher is unique**: 1,576 distinct substitution mappings for 1,576 problems

**Difficulty distribution** (by symbolic solvability):
- Easy (>95% solvable): numeral_system, unit_conversion, gravitational_constant
- Medium (~38%): cipher (limited by incomplete alphabet mapping)
- Hard (<10%): bit_manipulation, symbol_transform

**Baseline Nano accuracy**: [TBD — needs GPU inference run]
**Key error patterns**: [TBD — needs baseline run]

---

## § Judge Model

### What We Know (Pre-Phase 0)

**Identity**: `metric/nemotron-3-nano-30b-a3b-bf16` (from Kaggle model tab)

**Architecture**: Nemotron 3 Nano 30B-A3B
- 31.6B total parameters, ~3.2B active per token
- Hybrid Mamba-2 + Transformer + MoE architecture
- Context length up to 262K (some configs up to 1M)
- Produces reasoning trace followed by final response
- Reasoning can be toggled ON/OFF via system prompt

**Post-training stack** (from tech report):
- SFT on chat, agentic, and reasoning traces
- Multi-environment RLVR (Reinforcement Learning from Verifiable Rewards)
- RLHF using HelpSteer-derived reward model
- The reward model scores on 5 HelpSteer dimensions

**Recommended inference parameters** (from model card):
- temperature=1.0, top_p=0.95
- These may be overridden by Kaggle evaluation page parameters

**HelpSteer2 Reward Weights** (from published documentation):
- Correctness: +0.80
- Helpfulness: +0.65
- Complexity: +0.55
- Coherence: +0.45
- Verbosity: -0.40

*Note: These are HelpSteer2 weights. The actual competition metric may use
different weights or a different scoring mechanism. VERIFY IN PHASE 0.*

### What We Don't Know Yet
- Exact scoring mechanism (is it HelpSteer-based? Something else?)
- Whether the judge evaluates trace + answer together or separately
- Whether the judge outputs a continuous score or binary correct/incorrect
- The specific prompt template the judge uses
- Whether we can access the judge locally or only through submissions

### Phase 0 Findings (2026-03-26) — FROM METRIC SOURCE CODE

**Scoring mechanism**: Pure accuracy. `score()` returns `num_correct / len(solution)`.
Participant submits a LoRA adapter (zip with `adapter_config.json`).
Metric code loads adapter onto Nemotron-3-Nano-30B-A3B-BF16 via vLLM,
generates predictions, extracts answers from `\boxed{}`, compares to ground truth.

**Score type**: Continuous float [0.0, 1.0] — fraction of correct answers

**Evaluation parameters** (from metric `score()` defaults):
- `temperature=1.0`
- `top_p=1.0`
- `max_tokens=3584`
- `max_model_len=4096`
- `max_num_seqs=128`
- `gpu_memory_utilization=0.85`

**Answer verification** (`verify()` function):
- Numeric: `math.isclose(stored, predicted, rel_tol=1e-2, abs_tol=1e-5)`
- String: case-insensitive exact match
- Extraction: last `\boxed{}` content, fallback to "Final answer is:" patterns,
  fallback to last number in text

**Trace impact**: ZERO. Only the extracted answer matters. Reasoning traces are
generated (thinking is enabled) but not scored. Only `\boxed{}` content is evaluated.

**Length/format/style preference**: IRRELEVANT. Pure accuracy metric.

**Prompt modification**: Metric appends to every prompt:
`"\nPlease put your final answer inside \boxed{}. For example: \boxed{your answer}"`

**HelpSteer hypothesis**: BUSTED. The competition does NOT use HelpSteer reward
scoring. The weights (+0.80 correctness, -0.40 verbosity, etc.) are from Nano's
training, NOT from this competition's evaluation.

**vLLM inference details**:
- `enable_lora=True`, `max_lora_rank=32`
- `enable_prefix_caching=True`, `enable_chunked_prefill=True`
- `tensor_parallel_size=1`
- Chat template with `enable_thinking=True`
- Single generation per prompt (no sampling/voting)

**LoRA submission constraints** (from demo notebook):
- Max rank: 32
- `lora_alpha=16`
- Target modules: `r".*\.(in_proj|out_proj|up_proj|down_proj)$"`
- `lora_dropout=0.05`
- `bias="none"`
- `task_type=TaskType.CAUSAL_LM`
- Output: `model.save_pretrained()` → zip → submit

---

## § Competition Mechanics

### What We Know
- Platform: Kaggle (Featured Competition)
- Host: NVIDIA AI + partners
- Compute: Google Cloud G4 VMs (RTX-class Blackwell GPUs — RTX PRO 6000)
- Duration: ~3 months (exact end date TBD)
- Launched: GTC 2026 (March 2026)
- Public scores early in competition: ~0.6 range

### What We Don't Know Yet
- Exact deadline
- Daily submission limit
- Whether there's a private leaderboard shake-up
- Prize structure details
- Code sharing requirements
- Model size or inference time limits

### Phase 0 Findings (2026-03-26)

**Deadline**: 2026-06-15 23:59:00 UTC (81 days from now)
**Prize**: $106,388 USD
**Teams**: 1,063 registered (as of 2026-03-26)
**Entered**: NO — must accept rules to download data
**Submission format**: LoRA adapter zip (adapter_config.json + weights)
**Submission limit**: [TBD — check after joining]
**Inference time limit**: [TBD — check after joining]
**GPU available per submission**: G4 VM (RTX Blackwell-class, likely single GPU)
**Private leaderboard**: [TBD — check after joining]
**Code requirements**: Code competition — notebook must produce submission.zip

---

## § Nemotron 3 Technical Details

### Nano 30B-A3B
- 30B total params, 3B active per token (MoE)
- Hybrid: 52 layers total — 23 MoE + 23 Mamba-2 + 6 GQA attention
- Pre-training: 25T tokens
- Post-training: SFT + RLVR + RLHF
- Context: up to 262K (some configs 1M)
- Inference: supports FP8 quantization (~99% of BF16 accuracy)
- Throughput: ~3.3× higher than similarly sized open models

### Nano 4B (Compressed variant)
- Compressed from Nemotron-Nano-9B-v2 using Nemotron Elastic
- Reasoning trace + final response architecture
- Reasoning ON/OFF via system prompt

### Super 120B-A12B
- 120B total params, 12B active (LatentMoE)
- Multi-Token Prediction (MTP) for faster inference
- Potential teacher model for distillation
- Available via NIM API and Vertex AI

### Key Frameworks
- NeMo-Skills: End-to-end pipeline for LLM skill improvement
- NeMo Data Designer: Synthetic data generation orchestration
- NeMo Gym: RL training environments
- NeMo Evaluator SDK: Evaluation and benchmarking

---

## § ARC Prize Lessons (Transferable Intelligence)

### ARC Prize 2024 Winner (Li et al.)
- Combined inductive (program synthesis) + transductive (neural) approaches
- Found them "strongly complementary" — solved different puzzle subsets
- Induction ~40% alone, Transduction ~40% alone, Ensemble ~47.5%
- Test-Time Training was key enabler

### ARC Prize 2025 Top Approaches
- Jeremy Berman: LLM-guided evolutionary program synthesis, 79.6% on ARC-AGI-1
  - Used natural language transformation descriptions evolved by LLM
  - ~500 LLM calls per task
- Eric Pang: DreamCoder-inspired library learning, 77.1%
  - Only ~10 LLM calls per task (50× more efficient)
  - STITCH compression 1000-10000× faster than DreamCoder
- NVARC (NVIDIA): Synthetic data + TTT + 4B fine-tuned model
  - Cost: $0.20 per task

### Key Transferable Insight
The pattern across all top solutions: **symbolic/programmatic solving verified
against examples + neural generation for the parts that can't be symbolically
solved + aggressive test-time compute allocation**.

---

## § Tools and Resources Catalog

| Tool | Purpose | Location |
|------|---------|----------|
| NeMo-Skills | Training/eval pipelines | github.com/NVIDIA-NeMo/Skills |
| NeMo Data Designer | Synthetic data gen | nvidia-nemo.github.io/DataDesigner |
| NeMo Gym | RL training | Part of NeMo toolkit |
| Hodel ARC-DSL | DSL for grid transforms | github.com/michaelhodel/arc-dsl |
| Popper | Inductive logic programming | github.com/logic-and-learning-lab/Popper |
| ILPAR | ILP for ARC puzzles | Based on Popper + ARC object DSL |
| vLLM | Fast LLM inference | github.com/vllm-project/vllm |
| SGLang | Structured LLM inference | github.com/sgl-project/sglang |

---

## Update Log

| Date | Section | Update |
|------|---------|--------|
| 2026-03-22 | All | Initial knowledge base from research phase |
| 2026-03-26 | Judge, Competition, Benchmark | Phase 0 intel from metric source code + Kaggle API. HelpSteer hypothesis BUSTED. Pure accuracy metric confirmed. LoRA submission format confirmed. Deadline + prize confirmed. |
| 2026-03-26 | Benchmark | Data downloaded. 9,500 train / 3 public test. 6 balanced categories: bit_manipulation, gravitational_constant, unit_conversion, cipher, numeral_system, symbol_transform. All are "Alice's Wonderland" rule-induction puzzles with few-shot examples. Symbolic solver achieves 57% on train. |
