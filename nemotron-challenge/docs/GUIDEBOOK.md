# The Nemotron Reasoning Challenge — A Complete Guidebook

> Written: 2026-03-29 | Deadline: 2026-06-15 (78 days) | Prize: $106,388 | Teams: 1,063

---

## Part 1: What Is This Competition?

NVIDIA created a model called **Nemotron-3-Nano-30B** — a 30-billion-parameter language model where only 3 billion parameters activate per token (a technique called Mixture of Experts, or MoE). It's a hybrid architecture: part Transformer (the standard attention mechanism behind GPT), part Mamba-2 (a newer state-space model that handles long sequences efficiently). Think of it as a reasonably smart base model that can follow instructions and think through problems.

The competition asks: **Can you make this model smarter at solving reasoning puzzles — using only a small, bolt-on modification?**

That modification is called a **LoRA adapter** (Low-Rank Adaptation). Instead of retraining all 30 billion parameters (which would require massive compute), LoRA injects small trainable matrices into specific layers of the frozen model. Our adapter has rank 32, which means roughly 5-10 million trainable parameters — a tiny fraction of the full model. It's like teaching a fluent English speaker to also handle cryptography by adding a phrasebook, not by rewriting their brain.

### The Puzzles

There are 9,500 training problems across 6 categories. Every problem follows the same pattern: here are some examples of an input-output transformation — figure out the rule, apply it to a new input.

| Category | What It Is | Example | Difficulty |
|----------|-----------|---------|-----------|
| **numeral_system** | Decimal to Roman numerals | 42 → XLII | Trivial |
| **unit_conversion** | Linear scaling (y = a·x) | If 3→7.5, what's 4→? | Easy |
| **gravitational_constant** | Physics: d = ½gt² | Given distance/time pairs, find d at new t | Easy |
| **cipher** | Substitution cipher (unique per problem) | a→m, b→k, ... decode "hello" | Medium |
| **bit_manipulation** | Multi-step binary transforms (XOR, shift, rotate, etc.) | 01111100 → ??? | Hard |
| **symbol_transform** | Symbolic equation transforms | 79+05=1584, what is 48#26? | Hard |

The categories are perfectly balanced (~1,575 each). This is important — it means every category is worth roughly the same fraction of the final score.

### How Scoring Works

This is the single most important thing to understand. We reverse-engineered the exact scoring code:

1. Your LoRA adapter gets loaded onto the base Nemotron model
2. The model receives each test problem with the instruction: *"Please put your final answer inside \boxed{}"*
3. The model generates a response (with `enable_thinking=True`, meaning it can use internal reasoning)
4. The scorer extracts the last `\boxed{...}` content from the response
5. It compares that to the ground truth:
   - **Numbers**: Match within 1% relative tolerance
   - **Strings**: Case-insensitive exact match
6. Score = (number correct) / (total problems). That's it.

**What doesn't matter**: how good the reasoning trace is, how long or short the response is, how "helpful" or "clear" the explanation is. Only the content inside `\boxed{}` is scored. Binary — right or wrong, no partial credit.

**Critical parameter**: inference runs at `temperature=1.0` (stochastic sampling, not greedy). This means the model's output has randomness. A model that's 90% confident in the right answer will sometimes output the wrong one. This is unusual for competitions and is a significant factor.

---

## Part 2: Our Approach

### The Core Thesis

The model already knows how to reason. What it needs is **pattern recognition for these specific puzzle types**. By showing it thousands of worked examples — "here's a cipher problem, here's how to build the mapping, here's the answer" — we teach it to recognize and apply these patterns reliably.

This is Supervised Fine-Tuning (SFT): show the model correct input-output pairs with reasoning, and train it to reproduce that behavior.

### The Strategic Layers

```
Layer 1: KNOW THE BENCHMARK
├── Which categories does Nano already handle well?
├── Where does it fail? Format errors? Wrong reasoning? Knowledge gaps?
└── This determines where to focus training data.

Layer 2: DATA QUALITY
├── Symbolic solvers generate verified training data for easy categories
├── GPT-4.1 generates reasoning traces for cipher problems
├── Synthetic data augmentation for hard categories
└── Quality > quantity: one verified example beats ten noisy ones

Layer 3: TRAINING OPTIMIZATION
├── LoRA hyperparameters (learning rate, epochs, which layers)
├── Evaluation at temperature=1.0 (matching competition conditions)
├── Category-aware oversampling (more data where the model is weakest)
└── Iterate: train → evaluate → adjust → retrain
```

### Why LoRA (and Why It's Enough)

LoRA works by decomposing weight updates into two small matrices. Instead of updating a 4096×4096 weight matrix (16M parameters), you update two matrices of size 4096×32 and 32×4096 (262K parameters). The rank-32 constraint limits how "expressive" these updates can be, but for teaching pattern recognition on 6 well-defined puzzle types, it's more than sufficient.

We target four specific module types in the model: `in_proj`, `out_proj`, `up_proj`, `down_proj` — these are the projection layers in the attention and feed-forward blocks. This is where the model makes its "decisions" about what information to attend to and how to transform it.

### The Data Pipeline

We built a pipeline that converts raw competition problems into training examples:

1. **Load** 9,500 problems from CSV
2. **Categorize** each by type (keyword matching + answer format heuristics)
3. **Generate Chain-of-Thought (CoT)** reasoning for each:
   - Numeral/gravity/unit: programmatic solvers write perfect step-by-step reasoning
   - Cipher: GPT-4.1 generates detailed mapping + decoding traces
   - Bit manipulation/symbol: template-based reasoning (teaches format, not rule)
4. **Format** as `<think>...reasoning...</think>\n\n\boxed{answer}`
5. **Split** 85/15 into train (8,075) and validation (1,425)

The solver accuracy tells us where our training data is strong vs. weak:

| Category | Solver Accuracy | Implication |
|----------|----------------|-------------|
| numeral_system | 100% | Perfect training data |
| unit_conversion | 99.4% | Perfect training data |
| gravitational_constant | 97.9% | Near-perfect training data |
| cipher | 38.4% (+ GPT-4.1 CoTs) | Decent but gaps remain |
| bit_manipulation | 7.1% | Weak — model must generalize |
| symbol_transform | TBD | Weak — model must generalize |

### The Competitive Edges

**Edge 1: Verified synthetic data.** Most competitors will use teacher LLMs to generate training data and hope it's correct. We can *verify* math/logic/physics answers programmatically. Zero label noise for 60% of the training set. Clean data → better LoRA.

**Edge 2: Category-aware training.** After baseline profiling, we know exactly which categories the model struggles with. We oversample those categories in training. Targeted improvement where it matters most.

**Edge 3: Temperature=1.0 alignment.** Most competitors will evaluate during development at temperature=0 (greedy/deterministic). The competition evaluates at temperature=1.0 (stochastic). We train and evaluate at 1.0 from the start. This is a blind spot for others.

**Edge 4: Reasoning trace quality.** While traces aren't scored, they affect how the model reaches its answer. Better reasoning traces in training data → more reliable reasoning → more correct `\boxed{}` answers.

---

## Part 3: The Infrastructure

### Two-Notebook Architecture

We can't train on Kaggle (no persistent GPU access for the model size). We can't easily submit from Colab (different platform). Solution: split the workflow.

**Notebook 1: `train_lora.ipynb`** — Google Colab A100 (80GB VRAM)
- Downloads the 60GB model via `kagglehub` (no manual upload needed)
- Runs baseline evaluation on 500 stratified problems
- Trains the LoRA adapter (SFT, 3 epochs)
- Evaluates the trained adapter vs. baseline
- Exports `adapter.zip` to Google Drive

**Notebook 2: `submit.ipynb`** — Kaggle Notebooks
- Takes the adapter.zip (uploaded as a Kaggle Dataset)
- Packages it as `submission.zip`
- Submits for scoring
- Also has an "identity LoRA" cell for pipeline validation (generates a do-nothing adapter)

### Compute Budget

Colab Education Pro gives ~100 compute units/month, roughly 6-7 hours of A100 time. Every training run and baseline eval costs real hours. The entire project is designed around this constraint:

- Validate the submission pipeline first (identity LoRA — costs nothing)
- Run baseline on 500 problems, not 9,500 (stratified sample covers all categories)
- Train efficiently: bf16 by default, QLoRA 4-bit as fallback
- Iterate deliberately: each run should test a specific hypothesis

### The Pipeline Code

```
pipeline/
├── config.py      — All tunable parameters (LoRA rank, learning rate, batch size, etc.)
├── utils.py       — Data loading, answer extraction, verification (replicates metric exactly)
├── formatter.py   — Converts problems → SFT training examples with CoT reasoning
├── evaluator.py   — vLLM inference harness for Colab (replicates competition metric)
└── __init__.py    — Package exports
```

Supporting modules:
```
analysis/
├── benchmark_profiler.py  — Profile data: category distribution, answer formats, prompt lengths
└── error_analyzer.py      — Categorize prediction errors to guide training focus

scripts/
└── generate_cipher_cot.py — GPT-4.1 cipher CoT generation (408 lines, Azure OpenAI)

tests/
├── test_utils.py          — 42 tests: answer extraction, verification, categorization
├── test_config.py         — 7 tests: config defaults, YAML round-trip
└── test_evaluator.py      — 9 tests: summarization, sampling, reporting
```

### What Was Tried Before (and Why It Failed)

The original approach (now in `_quarantine/`) assumed:
- There's a judge model that scores reasoning traces
- HelpSteer rewards apply (correctness +0.80, verbosity -0.40)
- You submit inference-time solutions, not a fine-tuned model

All wrong. On March 26, we read the actual metric source code and discovered: no judge, no trace scoring, pure accuracy, LoRA submission. The entire strategy pivoted. The quarantined code (symbolic solver, trace generator, judge optimizer) is kept for reference but is not used.

---

## Part 4: The Roadmap

### Phase 0: Intelligence Gathering — COMPLETE
- Downloaded competition data
- Reverse-engineered the metric source code
- Profiled all 6 problem categories
- Built symbolic solvers for verifiable categories
- Documented everything in KNOWLEDGE_BASE.md and ASSUMPTIONS.md

### Phase 1: Data Engine — ~80% COMPLETE
- [x] Base JSONL training data (8,075 examples with CoT)
- [x] GPT-4.1 cipher CoTs generated and merged
- [ ] **Synthetic data for weak categories** (bit_manipulation, symbol_transform)
  - Use teacher models (Nemotron-Super-120B, DeepSeek-R1, Qwen-2.5-72B)
  - Verify answers programmatically where possible
  - Target: 10x the competition data for weak categories

### Phase 2: LoRA Training — READY TO START

**Step 1: Validate pipeline** (no GPU cost)
- Open `submit.ipynb` on Kaggle
- Run the identity LoRA cell (generates a do-nothing adapter)
- Submit → confirm the pipeline accepts our format
- This catches packaging errors before we spend A100 hours

**Step 2: Baseline profiling** (~30 min A100 time)
- Open `train_lora.ipynb` on Colab
- Run cells 1-5 only (setup through baseline eval)
- Get per-category accuracy for unmodified Nano on 500 problems
- This tells us exactly where to focus

**Step 3: First training run** (~2-3 hours A100 time)
- Run cells 6-7 (training + evaluation)
- SFT on current 8,075 examples, 3 epochs
- Compare: baseline vs. trained, per category
- Log everything to experiment tracker

**Step 4: Iterate**
- If easy categories regress → adjust data mix
- If hard categories don't improve → need better training data
- If overall gain is small → try different hyperparameters (LR sweep: 1e-5 to 5e-4)

### Phase 2+: Advanced Training (if SFT plateaus)

| Technique | What It Is | When to Use |
|-----------|-----------|-------------|
| **RLVR** | Reinforcement Learning from Verifiable Rewards — use symbolic solvers as reward signal | SFT accuracy plateaus, compute allows |
| **Curriculum learning** | Train on easy problems first, then hard | Model struggles to learn hard categories |
| **DPO** | Direct Preference Optimization — train on correct vs. incorrect reasoning pairs | Need to improve reasoning quality |
| **Category-specific LoRAs** | Train separate adapters per category, merge | Categories interfere with each other |

### Phase 3: Submission & Iteration — FUTURE

- Submit best checkpoint after each training run
- Track offline eval vs. leaderboard correlation (do our local scores predict Kaggle scores?)
- Identify which categories drive leaderboard movement
- Reserve final submission slots for best configuration
- Explore: can we benefit from multiple training runs or ensemble approaches?

### Timeline (78 days remaining)

```
Week 1 (Mar 29 - Apr 4):   Pipeline validation + baseline profiling
Week 2 (Apr 5 - Apr 11):   First training run + analysis
Week 3-4 (Apr 12-25):      Synthetic data generation for weak categories
Week 5-6 (Apr 26 - May 9): Training iterations + hyperparameter sweeps
Week 7-8 (May 10-23):      Advanced training (RLVR/DPO if needed)
Week 9-10 (May 24 - Jun 8): Final optimization + submission strategy
Week 11 (Jun 9-15):        Final submissions (deadline Jun 15)
```

---

## Part 5: Key Decisions and Why

### Why SFT before RL?
SFT (Supervised Fine-Tuning) is the simplest approach: show the model correct examples, train it to reproduce them. RL (Reinforcement Learning) is more powerful but harder to tune and more compute-hungry. Start simple, escalate only if needed. If SFT on good data reaches top-100, skip RL entirely.

### Why target these four module types?
`in_proj` and `out_proj` are the input/output projections of the attention mechanism — they control what the model "pays attention to." `up_proj` and `down_proj` are the feed-forward projections — they control how the model transforms information. Together, they cover the model's core reasoning pathway. The competition demo notebook uses exactly these targets, and changing them risks incompatibility.

### Why bf16 instead of QLoRA?
bf16 (brain floating point 16-bit) uses the A100's native precision — no information loss, fastest training. QLoRA quantizes the base model to 4-bit and trains adapters in bf16 on top — uses less VRAM but introduces quantization noise. We have 80GB VRAM on A100 High-RAM, which is enough for bf16. QLoRA is the fallback if memory is tight.

### Why 500 problems for baseline, not 9,500?
Inference on 9,500 problems would take hours of A100 time. 500 problems (83 per category, stratified) gives statistically meaningful per-category accuracy with ~30 minutes of compute. We can always run the full set later if needed.

### Why did we use GPT-4.1 for cipher CoTs instead of generating them ourselves?
Cipher problems require inferring character mappings from examples, then applying them — including guessing unmapped characters from English language patterns. Our symbolic solver only achieves 38% accuracy (it can't guess unmapped characters). GPT-4.1 can reason about likely English words and infer missing mappings. Its CoTs are higher quality training signal for the LoRA.

### Why temperature=1.0 matters so much
At temperature=0 (greedy), the model always picks the highest-probability token. At temperature=1.0, token selection is stochastic — proportional to probability. A model that assigns 70% probability to the right answer will get it right ~70% of the time at temp=1.0, but 100% of the time at temp=0. This means:
- Marginal accuracy improvements matter more (every percentage point of confidence counts)
- The model needs to be *confidently* correct, not just *barely* correct
- Training at temp=0 and evaluating at temp=1.0 will overestimate performance during development

---

## Part 6: Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Overfitting to small train set (9,500 is tiny for LLM fine-tuning) | HIGH | Validation split, synthetic data augmentation, regularization (dropout=0.05) |
| Compute budget too tight for iteration | HIGH | Validate pipeline first, use stratified sampling, prioritize hypotheses |
| Synthetic data doesn't match test distribution | MEDIUM | Profile benchmark deeply before generating; filter to match |
| Hard categories (bit/symbol) resist improvement | MEDIUM | Try RLVR with symbolic verifiers; accept ceiling and maximize easy/medium categories |
| Temperature=1.0 variance makes scores noisy | MEDIUM | Evaluate with multiple seeds locally; train for confidence, not just accuracy |
| LoRA rank 32 is too constrained for 6 diverse categories | LOW | Optimize layer targeting, tune alpha; rank 32 has worked well in similar competitions |

### Pivot Conditions

| If We See... | Then... |
|--------------|---------|
| Baseline Nano > 70% accuracy | Focus exclusively on hard categories; the easy ones are free |
| Baseline Nano < 30% accuracy | Model needs fundamental capability; scale up SFT data massively |
| SFT alone reaches top-100 | Data quality is the lever — skip RL, invest in better synthetic data |
| Top leaderboard > 0.9 | Competition is about marginal gains; optimize the hard tail |
| Top leaderboard < 0.5 | Competition is fundamentally hard; novel approaches needed |
| Easy categories regress after training | Data mix is wrong; reduce easy examples, increase hard |

---

## Part 7: Quick Reference

### File Locations
| What | Where |
|------|-------|
| Project root | `/Volumes/WD_BLACK/Desk2/Projects/Kaggle/Nvidia/nemotron-challenge/` |
| Training notebook | `notebooks/train_lora.ipynb` |
| Submission notebook | `notebooks/submit.ipynb` |
| Raw data | `data/train.csv` (9,500 problems) |
| Training data | `data/train_formatted.jsonl` (8,075 examples) |
| Validation data | `data/val_formatted.jsonl` (1,425 examples) |
| Configuration | `configs/default.yaml` |
| Strategy | `docs/STRATEGY.md` |
| Verified facts | `docs/KNOWLEDGE_BASE.md` |
| Kaggle API token | `/Volumes/WD_BLACK/Desk2/Projects/Keys/kaggle.json` |
| Quarantined (old approach) | `_quarantine/` |

### Fixed Competition Parameters (Do Not Change)
```
Model:          nemotron-3-nano-30b-a3b-bf16
LoRA rank:      32 (maximum)
LoRA alpha:     16
LoRA targets:   .*\.(in_proj|out_proj|up_proj|down_proj)$
Temperature:    1.0
top_p:          1.0
max_tokens:     3584
max_model_len:  4096
Answer format:  \boxed{answer}
Numeric tol:    rel_tol=1e-2, abs_tol=1e-5
String match:   case-insensitive exact
```

### Tunable Hyperparameters
```
Learning rate:  2e-4 (sweep: 1e-5 to 5e-4)
Epochs:         3 (monitor for overfitting)
Batch size:     4 (x 4 gradient accumulation = effective 16)
Warmup:         10% of steps
Weight decay:   0.01
Dropout:        0.05
Seq length:     4096
Val split:      15%
```

### Running Tests
```bash
cd /Volumes/WD_BLACK/Desk2/Projects/Kaggle/Nvidia/nemotron-challenge
python -m pytest tests/ -v    # 62/62 should pass
```

### Immediate Next Action
Open `submit.ipynb` on Kaggle. Run Cell 2 (identity LoRA) + Cell 3 (zip). Submit. Validate pipeline works. Zero GPU cost. Do this before anything else.
