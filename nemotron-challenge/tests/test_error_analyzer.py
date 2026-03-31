"""Tests for analysis.error_analyzer — error categorization and reporting."""

import pytest

from pipeline.utils import Problem
from analysis.error_analyzer import ErrorAnalyzer


def _make_problems():
    """Create a small problem set for testing."""
    return [
        Problem(id="1", prompt="bit manipulation", answer="10101010", category="bit_manipulation"),
        Problem(id="2", prompt="gravity", answer="9.81", category="gravitational_constant"),
        Problem(id="3", prompt="cipher", answer="hello", category="cipher"),
        Problem(id="4", prompt="numeral", answer="XLII", category="numeral_system"),
        Problem(id="5", prompt="unit", answer="3.28", category="unit_conversion"),
    ]


# ═══════════════════════════════════════════════════════════════════
# ErrorAnalyzer.__init__
# ═══════════════════════════════════════════════════════════════════

class TestErrorAnalyzerInit:
    def test_filters_problems_without_answers(self):
        problems = [
            Problem(id="1", prompt="p", answer="42"),
            Problem(id="2", prompt="p", answer=None),
        ]
        analyzer = ErrorAnalyzer(problems)
        assert len(analyzer.problems) == 1
        assert "1" in analyzer.problems

    def test_empty_problems(self):
        analyzer = ErrorAnalyzer([])
        assert len(analyzer.problems) == 0

    def test_duplicate_ids_last_wins(self):
        problems = [
            Problem(id="dup", prompt="first", answer="a"),
            Problem(id="dup", prompt="second", answer="b"),
        ]
        analyzer = ErrorAnalyzer(problems)
        assert analyzer.problems["dup"].answer == "b"


# ═══════════════════════════════════════════════════════════════════
# evaluate
# ═══════════════════════════════════════════════════════════════════

class TestEvaluate:
    def test_all_correct(self):
        problems = _make_problems()
        analyzer = ErrorAnalyzer(problems)
        predictions = {p.id: p.answer for p in problems}
        analyzer.evaluate(predictions)
        assert all(correct for correct, _ in analyzer.results.values())

    def test_all_wrong(self):
        problems = _make_problems()
        analyzer = ErrorAnalyzer(problems)
        predictions = {p.id: "WRONG" for p in problems}
        analyzer.evaluate(predictions)
        assert all(not correct for correct, _ in analyzer.results.values())

    def test_mixed_results(self):
        problems = _make_problems()
        analyzer = ErrorAnalyzer(problems)
        predictions = {"1": "10101010", "2": "WRONG", "3": "hello"}
        analyzer.evaluate(predictions)
        assert analyzer.results["1"][0] is True
        assert analyzer.results["2"][0] is False
        assert analyzer.results["3"][0] is True

    def test_unknown_ids_skipped(self):
        problems = _make_problems()
        analyzer = ErrorAnalyzer(problems)
        predictions = {"nonexistent": "value"}
        analyzer.evaluate(predictions)
        assert len(analyzer.results) == 0

    def test_numeric_tolerance(self):
        problems = [Problem(id="n1", prompt="p", answer="9.81", category="gravity")]
        analyzer = ErrorAnalyzer(problems)
        analyzer.evaluate({"n1": "9.82"})  # within 1% rel_tol
        assert analyzer.results["n1"][0] is True


# ═══════════════════════════════════════════════════════════════════
# report
# ═══════════════════════════════════════════════════════════════════

class TestReport:
    def test_report_empty(self):
        analyzer = ErrorAnalyzer([])
        assert "No results" in analyzer.report()

    def test_report_summary(self):
        problems = _make_problems()
        analyzer = ErrorAnalyzer(problems)
        analyzer.evaluate({p.id: p.answer for p in problems})
        report = analyzer.report()
        assert "Total: 5" in report
        assert "Correct: 5" in report
        assert "100.0%" in report

    def test_report_shows_errors(self):
        problems = _make_problems()
        analyzer = ErrorAnalyzer(problems)
        analyzer.evaluate({"1": "WRONG", "2": "9.81"})
        report = analyzer.report()
        assert "Sample Errors" in report
        assert "WRONG" in report

    def test_report_recommendations_weak_categories(self):
        problems = [
            Problem(id="1", prompt="p", answer="a", category="hard_cat"),
            Problem(id="2", prompt="p", answer="b", category="hard_cat"),
            Problem(id="3", prompt="p", answer="c", category="easy_cat"),
        ]
        analyzer = ErrorAnalyzer(problems)
        analyzer.evaluate({"1": "WRONG", "2": "WRONG", "3": "c"})
        report = analyzer.report()
        assert "Oversample" in report
        assert "hard_cat" in report

    def test_report_per_category_breakdown(self):
        problems = _make_problems()
        analyzer = ErrorAnalyzer(problems)
        predictions = {"1": "WRONG", "2": "9.81", "3": "hello", "4": "WRONG", "5": "3.28"}
        analyzer.evaluate(predictions)
        report = analyzer.report()
        assert "Per-Category" in report
        assert "bit_manipulation" in report
        assert "gravitational_constant" in report


# ═══════════════════════════════════════════════════════════════════
# wrong_by_category
# ═══════════════════════════════════════════════════════════════════

class TestWrongByCategory:
    def test_groups_errors(self):
        problems = _make_problems()
        analyzer = ErrorAnalyzer(problems)
        analyzer.evaluate({"1": "WRONG", "3": "WRONG"})
        wrong = analyzer.wrong_by_category()
        assert "bit_manipulation" in wrong
        assert "cipher" in wrong
        assert len(wrong["bit_manipulation"]) == 1
        assert wrong["bit_manipulation"][0] == ("1", "10101010", "WRONG")

    def test_no_errors(self):
        problems = _make_problems()
        analyzer = ErrorAnalyzer(problems)
        analyzer.evaluate({p.id: p.answer for p in problems})
        wrong = analyzer.wrong_by_category()
        assert len(wrong) == 0

    def test_empty_results(self):
        analyzer = ErrorAnalyzer([])
        wrong = analyzer.wrong_by_category()
        assert len(wrong) == 0
