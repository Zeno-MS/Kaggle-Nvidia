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

### Phase 0 Findings
*(Fill in as you investigate)*

**Puzzle format**: [TBD]
**Number of puzzles**: train=[TBD], test=[TBD]
**Input representation**: [TBD]
**Output representation**: [TBD]
**Training examples per puzzle**: [TBD — yes/no, how many]
**Puzzle families identified**:
- Family 1: [TBD]
- Family 2: [TBD]
- ...

**Difficulty distribution**: [TBD]
**Baseline Nano accuracy**: [TBD]
**Key error patterns**: [TBD]

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

### Phase 0 Findings
*(Fill in as you investigate)*

**Scoring mechanism**: [TBD]
**Score type**: continuous / binary / categorical [TBD]
**Evaluation parameters**: temperature=[TBD], max_tokens=[TBD], top_p=[TBD]
**Trace impact**: [TBD — how much does trace quality affect score vs correctness?]
**Length preference**: [TBD — results of length experiment]
**Format preference**: [TBD — results of format experiment]
**Style preference**: [TBD — results of style experiment]

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

### Phase 0 Findings
*(Fill in as you investigate)*

**Deadline**: [TBD]
**Submission limit**: [TBD per day]
**Inference time limit**: [TBD per submission]
**GPU available per submission**: [TBD]
**Private leaderboard**: yes/no [TBD]
**Code requirements**: [TBD]

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
