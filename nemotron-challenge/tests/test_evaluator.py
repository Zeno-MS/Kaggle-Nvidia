"""Tests for pipeline.evaluator — result summarization and sampling (no GPU needed)."""

from pipeline.evaluator import EvalResult, EvalSummary, Evaluator, stratified_sample
from pipeline.utils import Problem


class TestSummarize:
    """Test the _summarize method (no GPU or model needed)."""

    def _make_evaluator(self):
        """Create an Evaluator without loading a model."""
        ev = Evaluator.__new__(Evaluator)
        ev.lora_path = None
        return ev

    def test_perfect_accuracy(self):
        results = [
            EvalResult("p1", "cat_a", "42", "42", "output", True),
            EvalResult("p2", "cat_a", "10", "10", "output", True),
        ]
        summary = self._make_evaluator()._summarize(results, elapsed=1.0)
        assert summary.accuracy == 1.0
        assert summary.correct == 2
        assert summary.total == 2

    def test_zero_accuracy(self):
        results = [
            EvalResult("p1", "cat_a", "42", "99", "output", False),
        ]
        summary = self._make_evaluator()._summarize(results, elapsed=1.0)
        assert summary.accuracy == 0.0

    def test_per_category_breakdown(self):
        results = [
            EvalResult("p1", "cipher", "a", "a", "out", True),
            EvalResult("p2", "cipher", "b", "x", "out", False),
            EvalResult("p3", "numeral", "X", "X", "out", True),
        ]
        summary = self._make_evaluator()._summarize(results, elapsed=2.0)
        assert summary.per_category["cipher"]["correct"] == 1
        assert summary.per_category["cipher"]["total"] == 2
        assert summary.per_category["cipher"]["accuracy"] == 0.5
        assert summary.per_category["numeral"]["accuracy"] == 1.0

    def test_empty_results(self):
        summary = self._make_evaluator()._summarize([], elapsed=0.0)
        assert summary.total == 0
        assert summary.accuracy == 0.0


class TestStratifiedSample:
    """Test stratified sampling (no GPU needed)."""

    def test_samples_from_each_category(self):
        problems = [
            Problem(id=f"a{i}", prompt="p", category="cat_a") for i in range(20)
        ] + [
            Problem(id=f"b{i}", prompt="p", category="cat_b") for i in range(20)
        ]
        sample = stratified_sample(problems, n_per_category=5)
        cats = [p.category for p in sample]
        assert cats.count("cat_a") == 5
        assert cats.count("cat_b") == 5

    def test_fewer_than_requested(self):
        """If a category has fewer problems than requested, take all of them."""
        problems = [
            Problem(id=f"a{i}", prompt="p", category="cat_a") for i in range(3)
        ]
        sample = stratified_sample(problems, n_per_category=10)
        assert len(sample) == 3

    def test_empty_input(self):
        sample = stratified_sample([], n_per_category=5)
        assert len(sample) == 0


class TestReport:
    """Test the report formatter."""

    def test_report_produces_output(self):
        summary = EvalSummary(
            results=[],
            total=100,
            correct=75,
            accuracy=0.75,
            per_category={"cipher": {"correct": 30, "total": 50, "accuracy": 0.6}},
            elapsed_sec=10.0,
        )
        report = Evaluator.report(summary)
        assert "75.0%" in report
        assert "cipher" in report
        assert "60.0%" in report

    def test_report_with_lora_tag(self):
        summary = EvalSummary(
            results=[], total=10, correct=5, accuracy=0.5,
            per_category={}, lora_path="/some/path/adapter_v1", elapsed_sec=1.0,
        )
        report = Evaluator.report(summary)
        assert "adapter_v1" in report
