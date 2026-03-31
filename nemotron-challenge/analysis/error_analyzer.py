"""
Error Analyzer — Post-Pivot

Analyzes prediction errors for the LoRA fine-tuning pipeline.
Scoring is binary accuracy (correct/incorrect per problem).
No trace scoring, no partial credit.

Usage:
    from analysis.error_analyzer import ErrorAnalyzer
    analyzer = ErrorAnalyzer(problems)
    analyzer.evaluate(predictions)  # {problem_id: predicted_answer}
    print(analyzer.report())
"""

import logging
from collections import defaultdict
from pipeline.utils import Problem, verify

logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """Categorizes prediction errors to guide training data improvements."""

    def __init__(self, problems: list[Problem]):
        self.problems = {}
        for p in problems:
            if not p.answer:
                continue
            if p.id in self.problems:
                logger.warning(f"Duplicate problem ID {p.id} — later entry overwrites earlier")
            self.problems[p.id] = p
        self.results = {}  # problem_id -> (correct: bool, predicted: str)

    def evaluate(self, predictions: dict[str, str]):
        """Evaluate predictions against ground truth."""
        for pid, predicted in predictions.items():
            if pid not in self.problems:
                continue
            expected = self.problems[pid].answer
            correct = verify(expected, predicted)
            self.results[pid] = (correct, predicted)

    def report(self) -> str:
        """Generate error analysis report."""
        if not self.results:
            return "No results to analyze."

        total = len(self.results)
        correct = sum(1 for c, _ in self.results.values() if c)
        accuracy = correct / total

        lines = [
            "# Error Analysis Report\n",
            f"## Summary",
            f"- Total: {total}",
            f"- Correct: {correct} ({100*accuracy:.1f}%)",
            f"- Wrong: {total - correct} ({100*(1-accuracy):.1f}%)\n",
        ]

        # Per-category breakdown
        by_cat = defaultdict(lambda: {"correct": 0, "total": 0})
        for pid, (is_correct, _) in self.results.items():
            cat = self.problems[pid].category or "unknown"
            by_cat[cat]["total"] += 1
            if is_correct:
                by_cat[cat]["correct"] += 1

        lines.append("## Per-Category Accuracy\n")
        lines.append(f"{'Category':<25} {'Acc':>8} {'Correct':>8} {'Total':>8}")
        lines.append("-" * 55)
        for cat in sorted(by_cat, key=lambda c: by_cat[c]["correct"]/max(by_cat[c]["total"],1)):
            c, t = by_cat[cat]["correct"], by_cat[cat]["total"]
            lines.append(f"{cat:<25} {100*c/t:>7.1f}% {c:>8} {t:>8}")

        # Error examples
        wrong = [
            (pid, pred) for pid, (is_correct, pred) in self.results.items()
            if not is_correct
        ]
        if wrong:
            lines.append(f"\n## Sample Errors (first 10)\n")
            for pid, pred in wrong[:10]:
                p = self.problems[pid]
                lines.append(
                    f"  [{p.category}] {pid}: "
                    f"expected={p.answer!r}, got={pred!r}"
                )

        # Recommendations
        lines.append("\n## Recommendations\n")
        weak_cats = [
            cat for cat in by_cat
            if by_cat[cat]["correct"] / max(by_cat[cat]["total"], 1) < 0.5
        ]
        if weak_cats:
            lines.append(f"- Oversample training data for weak categories: {weak_cats}")
        strong_cats = [
            cat for cat in by_cat
            if by_cat[cat]["correct"] / max(by_cat[cat]["total"], 1) > 0.9
        ]
        if strong_cats:
            lines.append(f"- Strong categories (low priority): {strong_cats}")

        return "\n".join(lines)

    def wrong_by_category(self) -> dict[str, list[tuple[str, str, str]]]:
        """Return wrong predictions grouped by category: {cat: [(id, expected, predicted)]}"""
        result = defaultdict(list)
        for pid, (is_correct, pred) in self.results.items():
            if not is_correct:
                p = self.problems[pid]
                result[p.category].append((pid, p.answer, pred))
        return dict(result)
