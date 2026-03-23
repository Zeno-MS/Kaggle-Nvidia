"""
Benchmark Profiler

Phase 0 intelligence-gathering tool. Analyzes the competition dataset to answer
the critical questions that gate our entire strategy:

  1. What types of puzzles are in the benchmark?
  2. What is the difficulty distribution?
  3. What is the input/output representation?
  4. Can we identify families/clusters of related puzzles?
  5. Is the rule space finite and enumerable?

Run this FIRST before any optimization work.

Usage:
    from analysis.benchmark_profiler import BenchmarkProfiler
    profiler = BenchmarkProfiler("path/to/competition/data.csv")
    report = profiler.full_report()
    print(report)
"""

import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


class BenchmarkProfiler:
    """
    Comprehensive benchmark analysis tool.
    """
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.puzzles = []
        self._loaded = False
    
    def load(self):
        """Load puzzles from the competition data."""
        from pipeline.utils import load_puzzles_from_csv
        self.puzzles = load_puzzles_from_csv(str(self.data_path))
        self._loaded = True
        logger.info(f"Loaded {len(self.puzzles)} puzzles for profiling")
    
    def full_report(self) -> str:
        """Generate a comprehensive profiling report."""
        if not self._loaded:
            self.load()
        
        sections = [
            "# Benchmark Profile Report\n",
            self._basic_stats(),
            self._format_analysis(),
            self._length_analysis(),
            self._difficulty_estimation(),
            self._recommendations(),
        ]
        
        return "\n\n---\n\n".join(sections)
    
    def _basic_stats(self) -> str:
        """Basic dataset statistics."""
        n = len(self.puzzles)
        
        has_examples = sum(1 for p in self.puzzles if p.training_examples)
        has_test_input = sum(1 for p in self.puzzles if p.test_input)
        has_expected = sum(1 for p in self.puzzles if p.expected_output)
        
        return (
            f"## Basic Statistics\n"
            f"- Total puzzles: {n}\n"
            f"- With training examples: {has_examples} ({100*has_examples/max(n,1):.0f}%)\n"
            f"- With test input: {has_test_input} ({100*has_test_input/max(n,1):.0f}%)\n"
            f"- With expected output (train only): {has_expected} ({100*has_expected/max(n,1):.0f}%)\n"
            f"- Avg training examples per puzzle: {self._avg_examples():.1f}"
        )
    
    def _format_analysis(self) -> str:
        """Analyze the format of inputs and outputs."""
        input_types = Counter()
        output_types = Counter()
        
        for p in self.puzzles[:50]:  # Sample first 50
            if p.test_input:
                input_types[self._classify_format(str(p.test_input))] += 1
            if p.expected_output:
                output_types[self._classify_format(str(p.expected_output))] += 1
        
        report = "## Format Analysis (sampled from first 50 puzzles)\n"
        report += f"\nInput types: {dict(input_types)}"
        report += f"\nOutput types: {dict(output_types)}"
        
        # Show a few examples
        report += "\n\n### Sample Puzzles\n"
        for p in self.puzzles[:3]:
            report += f"\n**Puzzle {p.puzzle_id}**\n"
            report += f"  Instruction: {str(p.instruction)[:200]}...\n"
            if p.training_examples:
                report += f"  Training examples ({len(p.training_examples)}):\n"
                for i, (inp, out) in enumerate(p.training_examples[:2]):
                    report += f"    {i+1}. Input: {str(inp)[:100]}... → Output: {str(out)[:100]}...\n"
            if p.test_input:
                report += f"  Test input: {str(p.test_input)[:100]}...\n"
        
        return report
    
    def _length_analysis(self) -> str:
        """Analyze lengths of inputs, outputs, and instructions."""
        instruction_lens = [len(str(p.instruction)) for p in self.puzzles]
        
        input_lens = [
            len(str(p.test_input)) for p in self.puzzles if p.test_input
        ]
        
        output_lens = [
            len(str(p.expected_output)) for p in self.puzzles if p.expected_output
        ]
        
        def stats(values, name):
            if not values:
                return f"{name}: No data"
            values.sort()
            return (
                f"{name}:\n"
                f"  Min: {values[0]}, Max: {values[-1]}, "
                f"Median: {values[len(values)//2]}, "
                f"Mean: {sum(values)/len(values):.0f}"
            )
        
        return (
            f"## Length Analysis\n\n"
            f"{stats(instruction_lens, 'Instruction length (chars)')}\n\n"
            f"{stats(input_lens, 'Test input length (chars)')}\n\n"
            f"{stats(output_lens, 'Expected output length (chars)')}"
        )
    
    def _difficulty_estimation(self) -> str:
        """
        Estimate puzzle difficulty distribution.
        
        Proxies for difficulty:
        - Number of training examples (fewer = harder)
        - Length/complexity of transformation
        - Whether simple pattern matching suffices
        """
        report = "## Difficulty Estimation\n\n"
        report += "*(Run baseline model to get actual difficulty estimates)*\n\n"
        
        # Example count distribution
        example_counts = Counter(
            len(p.training_examples) for p in self.puzzles
        )
        report += f"Training examples distribution: {dict(sorted(example_counts.items()))}\n"
        
        return report
    
    def _recommendations(self) -> str:
        """
        Generate strategic recommendations based on profiling.
        These feed directly into STRATEGY.md and ASSUMPTIONS.md.
        """
        return (
            "## Recommendations\n\n"
            "*(Auto-generated after profiling — fill in based on findings)*\n\n"
            "1. **Puzzle format**: [What format are the puzzles? Update ASSUMPTIONS A1]\n"
            "2. **Symbolic solver viability**: [Can puzzles be solved programmatically?]\n"
            "3. **Trace template**: [What trace structure fits these puzzles?]\n"
            "4. **Priority adjustment**: [Should we shift priority between tracks?]\n"
            "5. **Key vulnerability**: [Where will most competitors struggle?]"
        )
    
    # ── Helpers ───────────────────────────────────────────────────
    
    def _avg_examples(self) -> float:
        counts = [len(p.training_examples) for p in self.puzzles]
        return sum(counts) / max(len(counts), 1)
    
    def _classify_format(self, text: str) -> str:
        """Classify the format of a puzzle input/output."""
        text = text.strip()
        
        if text.startswith('[') or text.startswith('{'):
            return "json/structured"
        elif text.startswith('|') or '\t' in text:
            return "grid/table"
        elif all(c in '0123456789 \n' for c in text):
            return "numeric"
        elif len(text.split('\n')) > 3:
            return "multiline"
        else:
            return "text/string"
