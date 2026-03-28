# Nemotron Reasoning Challenge

NVIDIA Nemotron Model Reasoning Challenge on Kaggle.
$106,388 prize pool. Deadline: 2026-06-15.

## The Challenge

Submit a **LoRA adapter** (rank <= 32) that improves Nemotron-3-Nano-30B
accuracy on Alice's Wonderland rule-induction puzzles. Pure accuracy scoring
-- fraction of problems answered correctly.

## The Data

9,500 train / 34 test problems across 6 balanced categories:

| Category | Count | Description |
|----------|-------|-------------|
| bit_manipulation | 1,602 | Multi-op transforms on 8-bit binary |
| gravitational_constant | 1,597 | Infer g from distance/time data |
| unit_conversion | 1,594 | Proportional conversion (y = a*x) |
| cipher | 1,576 | Unique substitution ciphers |
| numeral_system | 1,576 | Decimal to Roman numeral |
| symbol_transform | 1,555 | Symbol/character equation transforms |

All problems: examples of input->output, discover the rule, apply it.

## Strategy

1. **Profile** -- baseline Nano accuracy per category (needs GPU)
2. **Train** -- SFT LoRA on train data with CoT + \boxed{} format
3. **Augment** -- synthetic data for weak categories (bit_manipulation, cipher, symbol_transform)
4. **Iterate** -- eval at temp=1.0, oversample weak areas, retrain

## Key Constraints

- LoRA: rank=32, alpha=16, target `in_proj|out_proj|up_proj|down_proj`
- Eval: temperature=1.0, top_p=1.0, max_tokens=3584, enable_thinking=True
- Answer extracted from last `\boxed{}` in model output
- Numeric: rel_tol=1e-2. Strings: case-insensitive exact match.

## Project Structure

```
nemotron-challenge/
├── data/                  # Competition data (gitignored)
├── pipeline/              # Config, data loading, answer verification
├── analysis/              # Benchmark profiler, error analyzer
├── docs/                  # Strategy, knowledge base, assumptions
├── experiments/           # Experiment tracker
├── configs/               # YAML config files
└── _quarantine/           # Pre-pivot code (kept for reference)
```

## Quick Start

```python
from pipeline.utils import load_problems, verify, extract_final_answer
from pipeline.config import Config
from analysis.benchmark_profiler import profile

# Profile the data
print(profile("data/train.csv"))

# Load and inspect
problems = load_problems("data/train.csv")
print(f"{len(problems)} problems, categories: {set(p.category for p in problems)}")
```
