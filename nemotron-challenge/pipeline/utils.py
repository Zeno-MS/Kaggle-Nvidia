"""
Shared utilities for the Nemotron Challenge pipeline.
"""

import json
import csv
import logging
import time
from pathlib import Path
from typing import List
from functools import wraps

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_puzzles_from_csv(csv_path: str) -> list:
    """
    Load puzzles from the competition CSV file.
    
    IMPORTANT: This function's internals MUST be adapted after Phase 0
    when you know the actual CSV format. The current implementation is
    a best-guess skeleton.
    
    Returns a list of Puzzle objects (imported from orchestrator).
    """
    from pipeline.orchestrator import Puzzle
    
    puzzles = []
    path = Path(csv_path)
    
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # [PHASE 0] Adapt these column names based on actual CSV structure
            puzzle = Puzzle(
                puzzle_id=row.get('id', row.get('puzzle_id', str(len(puzzles)))),
                instruction=row.get('instruction', row.get('question', row.get('prompt', ''))),
                test_input=row.get('test_input', row.get('input', None)),
                expected_output=row.get('expected_output', row.get('output', row.get('answer', None))),
            )
            
            # Try to parse training examples if they exist
            examples_raw = row.get('examples', row.get('training_examples', None))
            if examples_raw:
                try:
                    puzzle.training_examples = json.loads(examples_raw)
                except (json.JSONDecodeError, TypeError):
                    puzzle.training_examples = []
            
            puzzles.append(puzzle)
    
    logger.info(f"Loaded {len(puzzles)} puzzles from {csv_path}")
    return puzzles


def save_solutions_to_csv(solutions: list, output_path: str):
    """
    Save solutions in the format expected by Kaggle submission.
    
    [PHASE 0] Adapt based on actual submission format requirements.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        # [PHASE 0] Adapt header based on actual submission format
        writer.writerow(['id', 'answer'])
        
        for solution in solutions:
            writer.writerow([
                solution.puzzle_id,
                solution.format_for_submission()
            ])
    
    logger.info(f"Saved {len(solutions)} solutions to {output_path}")


# ═══════════════════════════════════════════════════════════════════
# TIMING AND PROFILING
# ═══════════════════════════════════════════════════════════════════

def timed(func):
    """Decorator that logs execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper


class TimeBudget:
    """
    Context manager for tracking time budget consumption.
    
    Usage:
        budget = TimeBudget(total_seconds=60)
        with budget.track("symbolic_solving"):
            ...
        print(budget.remaining)  # seconds left
        print(budget.breakdown)  # where time was spent
    """
    
    def __init__(self, total_seconds: float):
        self.total = total_seconds
        self.start_time = time.time()
        self.breakdown = {}
        self._current_label = None
        self._current_start = None
    
    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time
    
    @property
    def remaining(self) -> float:
        return max(0, self.total - self.elapsed)
    
    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0
    
    def track(self, label: str):
        """Return a context manager that tracks time for a labeled section."""
        return _TimeBudgetSection(self, label)


class _TimeBudgetSection:
    def __init__(self, budget: TimeBudget, label: str):
        self.budget = budget
        self.label = label
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        elapsed = time.time() - self.start
        self.budget.breakdown[self.label] = (
            self.budget.breakdown.get(self.label, 0) + elapsed
        )


# ═══════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════

def setup_logging(level: str = "INFO", log_file: str = None):
    """Configure logging for the pipeline."""
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=fmt,
        handlers=handlers
    )


# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT TRACKING HELPERS
# ═══════════════════════════════════════════════════════════════════

def log_experiment(
    experiment_id: str,
    hypothesis: str,
    results: dict,
    tracker_path: str = "experiments/EXPERIMENT_TRACKER.md"
):
    """
    Append an experiment record to the tracker markdown file.
    
    Usage:
        log_experiment(
            experiment_id="EXP-001",
            hypothesis="Longer traces score lower",
            results={"short_score": 0.72, "long_score": 0.65}
        )
    """
    from datetime import datetime
    
    entry = f"""
### {experiment_id}: Auto-logged experiment
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Hypothesis**: {hypothesis}
**Results**: {json.dumps(results, indent=2)}
**Auto-logged**: Yes

---
"""
    
    path = Path(tracker_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'a') as f:
        f.write(entry)
    
    logger.info(f"Logged experiment {experiment_id}")
