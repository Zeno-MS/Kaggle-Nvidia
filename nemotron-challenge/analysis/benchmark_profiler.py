"""
Benchmark Profiler — Post-Pivot

Profiles the Alice's Wonderland competition data. All 9,500 problems are
rule-induction puzzles: given input→output examples, discover the hidden
transformation rule and apply it.

6 balanced categories (~16.5% each):
  bit_manipulation, gravitational_constant, unit_conversion,
  cipher, numeral_system, symbol_transform

Usage:
    from analysis.benchmark_profiler import profile
    report = profile("data/train.csv")
    print(report)
"""

import logging
from collections import Counter
from pipeline.utils import load_problems, Problem

logger = logging.getLogger(__name__)


def profile(csv_path: str) -> str:
    """Generate a full profiling report on the competition data."""
    problems = load_problems(csv_path)

    sections = [
        _basic_stats(problems),
        _category_distribution(problems),
        _answer_format_distribution(problems),
        _prompt_length_stats(problems),
        _examples_per_problem(problems),
        _sample_problems(problems),
    ]
    return "\n\n---\n\n".join(sections)


def category_accuracy(problems: list[Problem], predictions: dict[str, str]) -> str:
    """
    Given a dict of {problem_id: predicted_answer}, compute per-category accuracy.
    """
    from pipeline.utils import verify

    by_cat = {}
    for p in problems:
        if p.id not in predictions:
            continue
        cat = p.category or "unknown"
        if cat not in by_cat:
            by_cat[cat] = {"correct": 0, "total": 0}
        by_cat[cat]["total"] += 1
        if verify(p.answer, predictions[p.id]):
            by_cat[cat]["correct"] += 1

    lines = ["## Per-Category Accuracy\n"]
    lines.append(f"{'Category':<25} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    lines.append("-" * 55)
    total_correct, total_all = 0, 0
    for cat in sorted(by_cat):
        c, t = by_cat[cat]["correct"], by_cat[cat]["total"]
        total_correct += c
        total_all += t
        lines.append(f"{cat:<25} {c:>8} {t:>8} {100*c/t:>9.1f}%")
    lines.append("-" * 55)
    total_acc = 100 * total_correct / total_all if total_all > 0 else 0.0
    lines.append(f"{'TOTAL':<25} {total_correct:>8} {total_all:>8} {total_acc:>9.1f}%")
    return "\n".join(lines)


def _basic_stats(problems: list[Problem]) -> str:
    n = len(problems)
    with_answer = sum(1 for p in problems if p.answer)
    return (
        f"## Basic Statistics\n"
        f"- Total problems: {n}\n"
        f"- With answers (train): {with_answer}\n"
        f"- Without answers (test): {n - with_answer}"
    )


def _category_distribution(problems: list[Problem]) -> str:
    cats = Counter(p.category for p in problems)
    lines = ["## Category Distribution\n"]
    for cat, count in cats.most_common():
        pct = 100 * count / len(problems) if problems else 0.0
        # Each █ represents ~2 percentage points for a compact visual bar
        bar = "█" * int(pct / 2)
        lines.append(f"  {cat:<25} {count:>5} ({pct:>5.1f}%) {bar}")
    return "\n".join(lines)


def _answer_format_distribution(problems: list[Problem]) -> str:
    formats = Counter()
    for p in problems:
        if not p.answer:
            continue
        a = p.answer
        if all(c in '01' for c in a) and len(a) == 8:
            formats['8-bit binary'] += 1
        elif all(c in 'IVXLCDM' for c in a):
            formats['roman numeral'] += 1
        elif a.replace('.', '', 1).replace('-', '', 1).isdigit():
            formats['numeric'] += 1
        elif len(a.split()) >= 2 and a.replace(' ', '').isalpha():
            formats['word sequence'] += 1
        else:
            formats['symbol/other'] += 1

    lines = ["## Answer Format Distribution\n"]
    for fmt, count in formats.most_common():
        lines.append(f"  {fmt:<20} {count:>5}")
    return "\n".join(lines)


def _prompt_length_stats(problems: list[Problem]) -> str:
    lens = sorted(len(p.prompt) for p in problems)
    n = len(lens)
    return (
        f"## Prompt Length (chars)\n"
        f"- Min: {lens[0]}, Max: {lens[-1]}\n"
        f"- Median: {lens[n//2]}, Mean: {sum(lens)/n:.0f}"
    )


def _examples_per_problem(problems: list[Problem]) -> str:
    counts = Counter(len(p.examples) for p in problems)
    lines = ["## Examples Per Problem\n"]
    for n_ex, count in sorted(counts.items()):
        lines.append(f"  {n_ex} examples: {count} problems")
    return "\n".join(lines)


def _sample_problems(problems: list[Problem], n: int = 3) -> str:
    lines = ["## Sample Problems\n"]
    for p in problems[:n]:
        lines.append(f"### {p.id} [{p.category}]")
        lines.append(f"Prompt (first 200 chars): {p.prompt[:200]}...")
        lines.append(f"Answer: {p.answer}")
        lines.append(f"Parsed examples: {len(p.examples)}")
        if p.examples:
            lines.append(f"  First: {p.examples[0][0]} → {p.examples[0][1]}")
        lines.append("")
    return "\n".join(lines)
