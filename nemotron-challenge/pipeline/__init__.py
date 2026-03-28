"""
Nemotron Challenge Pipeline (Post-Pivot)

LoRA fine-tuning pipeline for the NVIDIA Nemotron Reasoning Challenge.
Submit a rank-32 LoRA adapter that improves Nemotron-3-Nano-30B accuracy
on Alice's Wonderland rule-induction puzzles.

Components:
  config.py    — Evaluation, LoRA, and training configuration
  utils.py     — Data loading, timing, experiment logging
  formatter.py — SFT training data generator with category-aware CoT traces
  evaluator.py — vLLM inference harness for Colab A100 (baseline + LoRA eval)
"""

from pipeline.config import Config
from pipeline.formatter import ProblemFormatter
from pipeline.evaluator import Evaluator, stratified_sample

__version__ = "0.3.0"
__all__ = ["Config", "ProblemFormatter", "Evaluator", "stratified_sample"]
