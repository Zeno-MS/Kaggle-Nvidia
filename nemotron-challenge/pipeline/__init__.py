"""
Nemotron Challenge Pipeline

Four-stage architecture:
  Stage 1 (Symbolic):  Attempt programmatic solution with verification
  Stage 2 (Trace):     Generate judge-optimized reasoning trace around answer
  Stage 3 (Judge):     Post-process and select best trace variant
  Stage 4 (Fallback):  Direct neural reasoning when symbolic fails
"""

from pipeline.config import Config

__version__ = "0.1.0"
__all__ = ["Config"]
