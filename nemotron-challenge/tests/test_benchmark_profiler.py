"""Tests for analysis.benchmark_profiler — data profiling and category accuracy."""

import csv
import tempfile
from pathlib import Path

import pytest

from pipeline.utils import Problem
from analysis.benchmark_profiler import (
    profile,
    category_accuracy,
    _basic_stats,
    _category_distribution,
    _answer_format_distribution,
    _prompt_length_stats,
    _examples_per_problem,
    _sample_problems,
)


def _make_problems():
    """Create a small representative problem set for testing."""
    return [
        Problem(id="1", prompt="Apply bit manipulation rules to 01010101",
                answer="10101010", category="bit_manipulation", examples=[("00", "11")]),
        Problem(id="2", prompt="Find the gravitational constant from the data",
                answer="9.81", category="gravitational_constant", examples=[("1.0", "4.9")]),
        Problem(id="3", prompt="Use this unit conversion to convert 5",
                answer="16.4", category="unit_conversion", examples=[("1", "3.28")]),
        Problem(id="4", prompt="Use this secret encryption cipher to decode abc",
                answer="xyz", category="cipher", examples=[("a", "x")]),
        Problem(id="5", prompt="Convert using this numeral system: 42",
                answer="XLII", category="numeral_system", examples=[("10", "X")]),
        Problem(id="6", prompt="Apply the transformation rules to the equation x = y",
                answer="42", category="symbol_transform", examples=[("a=1", "b=2")]),
    ]


# ═══════════════════════════════════════════════════════════════════
# _basic_stats
# ═══════════════════════════════════════════════════════════════════

class TestBasicStats:
    def test_counts_with_and_without_answers(self):
        problems = _make_problems()
        report = _basic_stats(problems)
        assert "Total problems: 6" in report
        assert "With answers (train): 6" in report

    def test_problems_without_answers(self):
        problems = [Problem(id="t1", prompt="test", answer=None)]
        report = _basic_stats(problems)
        assert "Without answers (test): 1" in report


# ═══════════════════════════════════════════════════════════════════
# _category_distribution
# ═══════════════════════════════════════════════════════════════════

class TestCategoryDistribution:
    def test_all_categories_shown(self):
        report = _category_distribution(_make_problems())
        assert "bit_manipulation" in report
        assert "cipher" in report
        assert "numeral_system" in report

    def test_empty_problems(self):
        report = _category_distribution([])
        assert "Category Distribution" in report


# ═══════════════════════════════════════════════════════════════════
# _answer_format_distribution
# ═══════════════════════════════════════════════════════════════════

class TestAnswerFormatDistribution:
    def test_identifies_binary(self):
        problems = [Problem(id="1", prompt="p", answer="01010101")]
        report = _answer_format_distribution(problems)
        assert "8-bit binary" in report

    def test_identifies_roman(self):
        problems = [Problem(id="1", prompt="p", answer="XLII")]
        report = _answer_format_distribution(problems)
        assert "roman numeral" in report

    def test_identifies_numeric(self):
        problems = [Problem(id="1", prompt="p", answer="3.14")]
        report = _answer_format_distribution(problems)
        assert "numeric" in report

    def test_identifies_word_sequence(self):
        problems = [Problem(id="1", prompt="p", answer="hello world")]
        report = _answer_format_distribution(problems)
        assert "word sequence" in report

    def test_skips_no_answer(self):
        problems = [Problem(id="1", prompt="p", answer=None)]
        report = _answer_format_distribution(problems)
        # Should not crash, just have header
        assert "Answer Format" in report


# ═══════════════════════════════════════════════════════════════════
# _prompt_length_stats
# ═══════════════════════════════════════════════════════════════════

class TestPromptLengthStats:
    def test_computes_stats(self):
        problems = [
            Problem(id="1", prompt="short"),
            Problem(id="2", prompt="a" * 100),
        ]
        report = _prompt_length_stats(problems)
        assert "Min: 5" in report
        assert "Max: 100" in report


# ═══════════════════════════════════════════════════════════════════
# _examples_per_problem
# ═══════════════════════════════════════════════════════════════════

class TestExamplesPerProblem:
    def test_counts_examples(self):
        problems = [
            Problem(id="1", prompt="p", examples=[("a", "b"), ("c", "d")]),
            Problem(id="2", prompt="p", examples=[]),
        ]
        report = _examples_per_problem(problems)
        assert "2 examples: 1 problems" in report
        assert "0 examples: 1 problems" in report


# ═══════════════════════════════════════════════════════════════════
# _sample_problems
# ═══════════════════════════════════════════════════════════════════

class TestSampleProblems:
    def test_shows_samples(self):
        problems = _make_problems()
        report = _sample_problems(problems, n=2)
        assert problems[0].id in report
        assert problems[1].id in report

    def test_fewer_than_n(self):
        problems = [Problem(id="only1", prompt="test", answer="42")]
        report = _sample_problems(problems, n=5)
        assert "only1" in report


# ═══════════════════════════════════════════════════════════════════
# category_accuracy
# ═══════════════════════════════════════════════════════════════════

class TestCategoryAccuracy:
    def test_perfect_accuracy(self):
        problems = _make_problems()
        predictions = {p.id: p.answer for p in problems}
        report = category_accuracy(problems, predictions)
        assert "100.0%" in report

    def test_zero_accuracy(self):
        problems = _make_problems()
        predictions = {p.id: "WRONG" for p in problems}
        report = category_accuracy(problems, predictions)
        assert "0.0%" in report

    def test_missing_predictions_skipped(self):
        problems = _make_problems()
        predictions = {"1": "10101010"}  # only one prediction
        report = category_accuracy(problems, predictions)
        assert "bit_manipulation" in report
        assert "TOTAL" in report

    def test_empty_predictions(self):
        problems = _make_problems()
        report = category_accuracy(problems, {})
        # No predictions matched — should not crash
        assert "Per-Category Accuracy" in report


# ═══════════════════════════════════════════════════════════════════
# profile (integration)
# ═══════════════════════════════════════════════════════════════════

class TestProfile:
    def test_full_profile_from_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=["id", "prompt", "answer"])
            writer.writeheader()
            writer.writerow({"id": "p1", "prompt": "Find gravitational constant from data", "answer": "9.8"})
            writer.writerow({"id": "p2", "prompt": "Apply bit manipulation to 01010101", "answer": "10101010"})
            f.flush()
            path = f.name

        report = profile(path)
        assert "Basic Statistics" in report
        assert "Category Distribution" in report
        assert "Prompt Length" in report
        Path(path).unlink()
