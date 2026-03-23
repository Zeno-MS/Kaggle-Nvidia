# Nemotron Reasoning Challenge — Project Core

## What This Is

This is the living brain of our NVIDIA Nemotron Model Reasoning Challenge entry.
It is designed to **adapt** as we learn more about the benchmark, the judge, and
what works. Every assumption is tracked. Every decision is logged. Every experiment
feeds back into the strategy.

## Our Thesis (One Sentence)

> The competition rewards **correctness packaged in concise, coherent traces** — not
> raw reasoning power — so the winning system solves symbolically where possible,
> wraps solutions in judge-optimized traces, and falls back to structured neural
> reasoning only when symbolic methods fail.

## The Three Unconventional Edges

| Edge | Core Idea | Why It's Contrarian |
|------|-----------|---------------------|
| **Judge Reverse-Engineering** | Nemotron's reward model penalizes verbosity (-0.40 weight) and heavily rewards correctness (+0.80). Optimize for the judge, not for "good reasoning." | Most competitors will assume longer = better, as with every other LLM judge. |
| **Symbolic-First Solving** | Treat puzzles as program synthesis problems. Provably correct answers score highest on the dominant correctness axis. | Most competitors will throw neural reasoning at everything. |
| **Cognitive Trace Design** | Structure traces using a specific template that maps to Nemotron's HelpSteer scoring dimensions. | Most competitors will use generic CoT prompting. |

## Project Structure

```
nemotron-challenge/
│
├── docs/                         # Living documents (the strategy brain)
│   ├── STRATEGY.md               # ★ Master strategy — the living plan
│   ├── ASSUMPTIONS.md            # Every assumption, tracked and testable
│   ├── DECISION_LOG.md           # Key decisions with rationale
│   ├── KNOWLEDGE_BASE.md         # What we know about benchmark + judge
│   └── COMPETITIVE_INTEL.md      # Leaderboard trends, forum insights
│
├── pipeline/                     # The four-stage solving pipeline
│   ├── __init__.py
│   ├── config.py                 # Central configuration
│   ├── orchestrator.py           # ★ Main pipeline: symbolic → trace → judge → fallback
│   ├── symbolic_solver.py        # Stage 1: Program synthesis / symbolic solving
│   ├── trace_generator.py        # Stage 2: Neural trace generation around answers
│   ├── judge_optimizer.py        # Stage 3: Judge-alignment post-processing
│   ├── neural_fallback.py        # Stage 4: Direct neural reasoning fallback
│   └── utils.py                  # Shared utilities
│
├── analysis/                     # Intelligence-gathering tools
│   ├── benchmark_profiler.py     # Analyze puzzle structure and difficulty
│   ├── judge_profiler.py         # Profile judge model preferences
│   └── error_analyzer.py         # Categorize and track failure modes
│
├── experiments/                  # Experiment logs (auto-generated)
│   └── EXPERIMENT_TRACKER.md     # Master experiment log
│
├── configs/                      # Configuration files
│   └── default.yaml              # Default pipeline configuration
│
└── notebooks/                    # Exploration notebooks (Colab-compatible)
    └── .gitkeep
```

## How to Use This System

1. **Before any work session**: Read `docs/STRATEGY.md` — it's the current plan of record
2. **When you learn something new**: Update `docs/KNOWLEDGE_BASE.md` and check `docs/ASSUMPTIONS.md`
3. **Before making a choice**: Check `docs/DECISION_LOG.md` for prior context
4. **After every experiment**: Log it in `experiments/EXPERIMENT_TRACKER.md`
5. **When strategy needs to change**: Update `docs/STRATEGY.md` and record why in the decision log

## Compute Envelope

| Resource | Capability | Best Use |
|----------|-----------|----------|
| Kaggle G4 VM | RTX-class GPU, time-limited submissions | Final inference, submission testing |
| NVIDIA Competition Compute | TBD — check competition page | Heavy inference, possible TTT |
| Google Colab Pro | A100/T4, ~12hr sessions | Prototyping, synthetic data gen, fine-tuning |

## Current Phase

**Phase 0: Intelligence Gathering** — See `docs/STRATEGY.md` for details.
