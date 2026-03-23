"""
Error Analyzer

Categorizes failures from baseline and experimental runs to identify
where the pipeline is losing points and what to fix.

Error categories:
  - WRONG_ANSWER: Reasoning is plausible but answer is incorrect
  - FORMAT_ERROR: Answer is correct but not in expected format
  - PARTIAL_RULE: Found part of the rule but missed a component
  - WRONG_RULE: Identified a completely wrong transformation
  - NO_ANSWER: Model failed to produce any extractable answer
  - TIMEOUT: Ran out of time before completing
  - TRACE_QUALITY: Answer is correct but trace scored poorly

Usage:
    from analysis.error_analyzer import ErrorAnalyzer
    analyzer = ErrorAnalyzer()
    analyzer.add_result(puzzle_id, expected, predicted, score, trace)
    report = analyzer.report()
"""

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ErrorRecord:
    puzzle_id: str
    expected_output: str
    predicted_output: str
    score: float
    trace: str
    method: str              # "symbolic", "neural", "none"
    error_category: str      # Assigned by analysis
    notes: str = ""


class ErrorAnalyzer:
    """
    Categorizes and summarizes pipeline errors to guide optimization.
    """
    
    def __init__(self):
        self.records = []
        self.category_counts = Counter()
    
    def add_result(
        self,
        puzzle_id: str,
        expected: str,
        predicted: str,
        score: float,
        trace: str = "",
        method: str = "unknown"
    ):
        """Add a single result for analysis."""
        category = self._categorize(expected, predicted, score, trace)
        
        record = ErrorRecord(
            puzzle_id=puzzle_id,
            expected_output=expected,
            predicted_output=predicted,
            score=score,
            trace=trace,
            method=method,
            error_category=category
        )
        
        self.records.append(record)
        self.category_counts[category] += 1
    
    def _categorize(self, expected: str, predicted: str, score: float, trace: str) -> str:
        """Assign an error category based on the result."""
        # Normalize for comparison
        exp_norm = str(expected).strip().lower()
        pred_norm = str(predicted).strip().lower()
        
        if not predicted or not predicted.strip():
            return "NO_ANSWER"
        
        if exp_norm == pred_norm:
            if score < 0.8:  # Correct answer but low score
                return "TRACE_QUALITY"
            return "CORRECT"
        
        # Check for format issues (answer is essentially correct but formatted differently)
        if self._is_format_mismatch(expected, predicted):
            return "FORMAT_ERROR"
        
        # Check for partial correctness
        if self._is_partial_match(expected, predicted):
            return "PARTIAL_RULE"
        
        return "WRONG_ANSWER"
    
    def _is_format_mismatch(self, expected: str, predicted: str) -> bool:
        """Check if the answer is correct but just formatted differently."""
        # Strip all whitespace and punctuation
        import re
        clean_exp = re.sub(r'\s+', '', str(expected))
        clean_pred = re.sub(r'\s+', '', str(predicted))
        
        if clean_exp == clean_pred:
            return True
        
        # Check for quote/bracket differences
        for char in ['"', "'", '[', ']', '{', '}', '(', ')']:
            clean_exp = clean_exp.replace(char, '')
            clean_pred = clean_pred.replace(char, '')
        
        return clean_exp == clean_pred
    
    def _is_partial_match(self, expected: str, predicted: str) -> bool:
        """Check if the prediction is partially correct."""
        # Simple heuristic: >50% character overlap
        exp_set = set(str(expected).lower())
        pred_set = set(str(predicted).lower())
        
        if not exp_set:
            return False
        
        overlap = len(exp_set & pred_set) / len(exp_set)
        return overlap > 0.5
    
    def report(self) -> str:
        """Generate a comprehensive error analysis report."""
        total = len(self.records)
        if total == 0:
            return "No results to analyze."
        
        correct = self.category_counts.get("CORRECT", 0)
        accuracy = correct / total
        
        lines = [
            "# Error Analysis Report\n",
            f"## Summary",
            f"- Total puzzles: {total}",
            f"- Correct: {correct} ({100*accuracy:.1f}%)",
            f"- Errors: {total - correct} ({100*(1-accuracy):.1f}%)\n",
            "## Error Distribution\n",
        ]
        
        for category, count in sorted(self.category_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / total
            bar = "█" * int(pct / 2)
            lines.append(f"  {category:20s} {count:4d} ({pct:5.1f}%) {bar}")
        
        # Method breakdown
        method_correct = defaultdict(int)
        method_total = defaultdict(int)
        for r in self.records:
            method_total[r.method] += 1
            if r.error_category == "CORRECT":
                method_correct[r.method] += 1
        
        lines.append("\n## Accuracy by Method\n")
        for method in sorted(method_total.keys()):
            acc = method_correct[method] / max(method_total[method], 1)
            lines.append(f"  {method}: {method_correct[method]}/{method_total[method]} ({100*acc:.1f}%)")
        
        # Actionable recommendations
        lines.append("\n## Actionable Recommendations\n")
        
        if self.category_counts.get("FORMAT_ERROR", 0) > total * 0.05:
            lines.append("- **FORMAT_ERROR is significant**: Fix output formatting in submission pipeline")
        
        if self.category_counts.get("NO_ANSWER", 0) > total * 0.05:
            lines.append("- **NO_ANSWER is significant**: Improve answer extraction or increase time budget")
        
        if self.category_counts.get("TRACE_QUALITY", 0) > total * 0.1:
            lines.append("- **TRACE_QUALITY losses**: Correct answers scoring poorly — prioritize judge optimization (Track B)")
        
        if self.category_counts.get("WRONG_ANSWER", 0) > total * 0.3:
            lines.append("- **WRONG_ANSWER dominant**: Need better solving — invest in Track A or C")
        
        if self.category_counts.get("PARTIAL_RULE", 0) > total * 0.1:
            lines.append("- **PARTIAL_RULE significant**: Model finds partial patterns — consider compositional rule search")
        
        return "\n".join(lines)
    
    def worst_puzzles(self, n: int = 10) -> list:
        """Return the N lowest-scoring puzzles for manual investigation."""
        scored = [(r.score, r) for r in self.records if r.error_category != "CORRECT"]
        scored.sort(key=lambda x: x[0])
        return [record for _, record in scored[:n]]
    
    def category_examples(self, category: str, n: int = 3) -> list:
        """Return N example records for a given error category."""
        return [r for r in self.records if r.error_category == category][:n]
