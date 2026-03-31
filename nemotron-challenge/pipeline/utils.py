"""
Utilities for the Nemotron Challenge pipeline.

Data loading, answer verification, timing, and experiment logging.
"""

import csv
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Metric appends this to every prompt during evaluation — we match it exactly.
# Shared constant: used by formatter.py and evaluator.py.
BOXED_INSTRUCTION = "\nPlease put your final answer inside \\boxed{}. For example: \\boxed{your answer}"


# ═══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Problem:
    """A single competition problem."""
    id: str
    prompt: str
    answer: Optional[str] = None  # None for test set
    category: Optional[str] = None  # Assigned by categorize()
    examples: list = field(default_factory=list)  # Parsed (input, output) pairs


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_problems(csv_path: str) -> list[Problem]:
    """
    Load problems from competition CSV.
    Format: id, prompt, [answer]
    """
    problems = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = Problem(
                id=row['id'],
                prompt=row['prompt'],
                answer=row.get('answer'),
            )
            p.examples = parse_examples(p.prompt)
            p.category = categorize(p)
            problems.append(p)
    logger.info(f"Loaded {len(problems)} problems from {csv_path}")
    return problems


def parse_examples(prompt: str) -> list[tuple[str, str]]:
    """
    Extract input→output example pairs embedded in the prompt text.

    All problems follow the pattern:
        description → example lines → query

    Example lines look like:
        01010001 -> 11011101
        vsa botjah -> the silver queen
        61"88 = 27
    """
    examples = []
    # Match "X -> Y" or "X → Y" patterns (bit_manipulation, cipher)
    for match in re.finditer(r'^(.+?)\s*(?:->|→)\s*(.+)$', prompt, re.MULTILINE):
        inp, out = match.group(1).strip(), match.group(2).strip()
        if len(inp) > 100 or len(out) > 100:
            logger.debug("Skipping arrow example: input or output exceeds 100 chars")
            continue
        # Skip description lines like "Here are some examples of input → output:"
        if any(w in inp.lower() for w in ['example', 'here are', 'input']):
            logger.debug(f"Skipping description line: {inp[:60]}")
            continue
        examples.append((inp, out))

    # "X becomes Y" pattern (unit_conversion)
    if not examples:
        for match in re.finditer(r'^(.+?)\s+becomes\s+(.+)$', prompt, re.MULTILINE):
            inp, out = match.group(1).strip(), match.group(2).strip()
            if len(inp) > 80 or len(out) > 80:
                logger.debug("Skipping 'becomes' example: input or output exceeds 80 chars")
                continue
            examples.append((inp, out))

    # "For t = Xs, distance = Y m" pattern (gravitational_constant)
    if not examples:
        for match in re.finditer(
            r'For t\s*=\s*([\d.]+)\s*s,\s*distance\s*=\s*([\d.]+)\s*m',
            prompt
        ):
            examples.append((match.group(1), match.group(2)))

    # "X = Y" format (symbol_transform equations)
    if not examples:
        for match in re.finditer(r'^([^\n=]+?)\s*=\s*([^\n]+)$', prompt, re.MULTILINE):
            inp, out = match.group(1).strip(), match.group(2).strip()
            if len(inp) > 60 or len(out) > 60:
                logger.debug("Skipping equals example: input or output exceeds 60 chars")
                continue
            if any(w in inp.lower() for w in ['the', 'for', 'here', 'where']):
                logger.debug(f"Skipping prose line in equals format: {inp[:60]}")
                continue
            examples.append((inp, out))

    return examples


def categorize(problem: Problem) -> str:
    """Assign a category based on prompt content."""
    p = problem.prompt.lower()
    if 'bit manipulation' in p:
        return 'bit_manipulation'
    elif 'gravitational constant' in p:
        return 'gravitational_constant'
    elif 'encryption' in p or 'cipher' in p or 'secret encryption' in p:
        return 'cipher'
    elif 'numeral system' in p or 'roman numeral' in p:
        return 'numeral_system'
    elif 'unit' in p and ('conversion' in p or 'convert' in p):
        return 'unit_conversion'
    elif 'transformation rules' in p and ('=' in p or 'equation' in p):
        return 'symbol_transform'

    # Fallback: infer from answer format if keyword matching failed
    if problem.answer:
        a = problem.answer
        if all(c in '01' for c in a) and len(a) == 8:
            logger.debug(f"Categorized {problem.id} as bit_manipulation via answer format fallback")
            return 'bit_manipulation'
        if all(c in 'IVXLCDM' for c in a):
            logger.debug(f"Categorized {problem.id} as numeral_system via answer format fallback")
            return 'numeral_system'
        try:
            float(a)
            logger.debug(f"Categorized {problem.id} as numeric_unknown via answer format fallback")
            return 'numeric_unknown'
        except ValueError:
            pass
    logger.debug(f"Could not categorize {problem.id} — assigned 'unknown'")
    return 'unknown'


# ═══════════════════════════════════════════════════════════════════
# ANSWER VERIFICATION (replicates competition metric)
# ═══════════════════════════════════════════════════════════════════

def extract_final_answer(text: Optional[str]) -> Optional[str]:
    """
    Extract answer from model output. Replicates metric's extraction logic:
    1. Last \\boxed{} content
    2. "Final answer is:" pattern (matches to end of line, per metric source)
    3. Last number in text
    """
    if text is None:
        return None

    # Strategy 1: Last \boxed{}
    boxed = re.findall(r'\\boxed\{([^}]*)\}', text)
    if boxed:
        return boxed[-1].strip()

    # Strategy 2: "Final answer is:" — match to end of line (per competition metric)
    for pattern in [
        r'The final answer is:\s*([^\n]+)',
        r'Final answer is:\s*([^\n]+)',
        r'Final answer\s*[:]\s*([^\n]+)',
        r'final answer\s*[:]\s*([^\n]+)',
    ]:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    # Strategy 3: Last number
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]

    return None


def verify(expected: str, predicted: str) -> bool:
    """
    Check if predicted answer matches expected.
    Replicates competition metric's verify() function.
    """
    if predicted is None:
        return False

    expected = str(expected).strip()
    predicted = str(predicted).strip()

    # Numeric comparison — tolerances replicate competition metric exactly
    # (see Config.eval.numeric_rel_tol / numeric_abs_tol for documentation)
    try:
        exp_num = float(expected)
        pred_num = float(predicted)
        return math.isclose(exp_num, pred_num, rel_tol=1e-2, abs_tol=1e-5)
    except (ValueError, TypeError):
        pass

    # String comparison: case-insensitive
    return expected.lower() == predicted.lower()


# ═══════════════════════════════════════════════════════════════════
# TIMING
# ═══════════════════════════════════════════════════════════════════

def timed(func):
    """Decorator that logs execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.debug(f"{func.__name__} took {time.time() - start:.2f}s")
        return result
    return wrapper


class TimeBudget:
    """Track time budget consumption across labeled sections."""

    def __init__(self, total_seconds: float):
        self.total = total_seconds
        self.start_time = time.time()
        self.breakdown = {}

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
# LOGGING
# ═══════════════════════════════════════════════════════════════════

def setup_logging(level: str = "INFO", log_file: str = None):
    """Configure logging for the pipeline."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def log_experiment(
    experiment_id: str,
    hypothesis: str,
    results: dict,
    tracker_path: str = "experiments/EXPERIMENT_TRACKER.md",
):
    """Append an experiment record to the tracker."""

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
