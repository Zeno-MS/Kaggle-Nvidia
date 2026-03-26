# New Chat Handoff — Nemotron Reasoning Challenge

## Objective
Win (or place highly in) the NVIDIA Nemotron Model Reasoning Challenge on Kaggle.
$106,388 prize pool. Deadline: 2026-06-15. 1,063 teams registered.

## Current Status
- **Phase 0: Intelligence Gathering** — PARTIALLY COMPLETE
- Metric source code analyzed (2026-03-26): pure accuracy, LoRA submission, no judge
- Strategy rewritten to reflect reality: LoRA fine-tuning challenge, not trace optimization
- KNOWLEDGE_BASE.md, ASSUMPTIONS.md, STRATEGY.md all updated
- **BLOCKED**: Must accept competition rules on Kaggle to download data

## Key Facts (Verified from Metric Source Code)
- Submit: LoRA adapter (rank ≤ 32) → loaded onto Nemotron-3-Nano-30B via vLLM
- Metric: accuracy = num_correct / total (binary per problem)
- Answers: extracted from `\boxed{}` in model output
- Inference: temperature=1.0, top_p=1.0, max_tokens=3584, enable_thinking=True
- Data: train.csv (prompt + answer columns), test.csv (prompts only)

## Blocker
**Accept competition rules**: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
Account: `stoicknight`. Then download data via:
```
KAGGLE_CONFIG_DIR=/Volumes/WD_BLACK/Desk2/Projects/Keys kaggle competitions download nvidia-nemotron-model-reasoning-challenge -p data/
```

## Next Actions (After Unblocking)
1. Download + inspect train.csv — profile problem types and answer formats
2. Run BenchmarkProfiler (needs update for new CSV format)
3. Establish baseline: unmodified Nano accuracy on train set
4. Submit identity LoRA to verify submission pipeline
5. Begin synthetic data generation for weak categories

## Key Sources
- Metric notebook: `/tmp/nemotron-metric/nvidia-nemotron-metric.ipynb`
- Demo notebook: `/tmp/nemotron-demo/nvidia-nemotron-submission-demo.ipynb`
- Strategy: `docs/STRATEGY.md`
- Knowledge base: `docs/KNOWLEDGE_BASE.md`
- Assumptions: `docs/ASSUMPTIONS.md`
- Config: `configs/default.yaml`

## Notes
- 2026-03-22 | Project scaffold created, initial strategy
- 2026-03-26 | MAJOR PIVOT: metric analysis revealed pure accuracy scoring, LoRA-only submission. HelpSteer judge hypothesis busted. All docs updated.

_Generated: 2026-03-26_
