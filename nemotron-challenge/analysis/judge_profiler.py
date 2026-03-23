"""
Judge Profiler

Phase 0 tool for understanding the competition's evaluation metric.
Designs and tracks experiments to characterize the judge model's preferences.

Key experiments:
  1. Length preference: Does the judge prefer shorter or longer traces?
  2. Correctness weight: How much does a wrong answer with good reasoning score?
  3. Format preference: Structured steps vs prose vs minimal
  4. Style preference: Nemotron-style vs generic vs academic
  5. Verification bonus: Does including a verification step help?

Usage:
    from analysis.judge_profiler import JudgeProfiler
    profiler = JudgeProfiler()
    experiments = profiler.design_experiments(sample_puzzles)
    # Run experiments manually (requires submission or local judge)
    profiler.analyze_results(results)
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class JudgeExperiment:
    """A single judge characterization experiment."""
    name: str
    hypothesis: str
    variants: list           # List of (label, trace) tuples to test
    puzzle_id: str           # Which puzzle to test on
    correct_answer: str      # The known correct answer
    assumption_tested: str   # Which assumption in ASSUMPTIONS.md this tests


class JudgeProfiler:
    """
    Designs and analyzes experiments to characterize the judge model.
    """
    
    def __init__(self):
        self.experiments = []
        self.results = {}
    
    def design_length_experiment(self, puzzle, correct_answer: str) -> JudgeExperiment:
        """
        Test whether the judge prefers shorter or longer traces.
        Tests Assumption B2 (verbosity penalty).
        """
        # Variant 1: Minimal — just the answer
        minimal = f"The answer is: {correct_answer}"
        
        # Variant 2: Short — 3-step reasoning
        short = (
            f"Looking at the examples, I can identify the transformation rule.\n"
            f"Step 1: The pattern involves applying a consistent transformation to each element.\n"
            f"Step 2: This rule correctly maps all training inputs to their outputs.\n"
            f"Step 3: Applying this rule to the test input gives: {correct_answer}\n"
            f"The answer is: {correct_answer}"
        )
        
        # Variant 3: Medium — 7-step with verification
        medium = (
            f"I need to identify the transformation rule from the examples.\n\n"
            f"Step 1: Examining Example 1, the input transforms to the output via a specific pattern.\n"
            f"Step 2: Checking Example 2, the same pattern holds.\n"
            f"Step 3: The pattern appears to be a consistent transformation applied element-wise.\n"
            f"Step 4: Let me formalize this: the rule maps each input element according to the discovered pattern.\n"
            f"Step 5: Verification — applying this rule to Example 1 input reproduces the expected output. ✓\n"
            f"Step 6: Verification — applying this rule to Example 2 input also matches. ✓\n"
            f"Step 7: Applying the verified rule to the test input.\n\n"
            f"The answer is: {correct_answer}"
        )
        
        # Variant 4: Long — 15+ step verbose reasoning
        long = (
            f"Let me carefully analyze this transformation puzzle step by step.\n\n"
            f"First, I need to understand what the puzzle is asking. "
            f"We have a set of training examples showing input-output pairs, "
            f"and I need to discover the underlying transformation rule.\n\n"
            f"Step 1: Let me examine the first training example in detail...\n"
            f"Step 2: I notice that the input has certain structural properties...\n"
            f"Step 3: The output appears to preserve some elements while changing others...\n"
            f"Step 4: Let me compare this with the second example to look for consistency...\n"
            f"Step 5: Indeed, the same pattern seems to apply here as well...\n"
            f"Step 6: Let me try to formalize what I've observed so far...\n"
            f"Step 7: The transformation rule appears to involve several sub-steps...\n"
            f"Step 8: First, the input is processed by identifying key elements...\n"
            f"Step 9: Then, a specific operation is applied to these elements...\n"
            f"Step 10: The result is assembled according to a consistent schema...\n"
            f"Step 11: Let me verify this hypothesis against Example 1...\n"
            f"Step 12: Applying my proposed rule to Example 1's input, I get the expected output. Great!\n"
            f"Step 13: Let me also verify against Example 2...\n"
            f"Step 14: This also produces the correct output. My rule is confirmed.\n"
            f"Step 15: Now I can confidently apply this rule to the test input.\n"
            f"Step 16: Processing the test input through the transformation pipeline...\n\n"
            f"After careful analysis and verification, the answer is: {correct_answer}"
        )
        
        exp = JudgeExperiment(
            name="length_preference",
            hypothesis="Nemotron judge penalizes verbosity — medium traces should outscore long ones",
            variants=[
                ("minimal", minimal),
                ("short", short),
                ("medium", medium),
                ("long", long),
            ],
            puzzle_id=puzzle.puzzle_id,
            correct_answer=correct_answer,
            assumption_tested="B2"
        )
        
        self.experiments.append(exp)
        return exp
    
    def design_correctness_vs_reasoning_experiment(
        self, puzzle, correct_answer: str, wrong_answer: str
    ) -> JudgeExperiment:
        """
        Test whether correctness or reasoning quality dominates scoring.
        Tests Assumption B3.
        """
        # Variant 1: Correct answer, poor reasoning
        correct_poor = (
            f"The answer is: {correct_answer}"
        )
        
        # Variant 2: Wrong answer, excellent reasoning
        wrong_excellent = (
            f"Analyzing the transformation pattern systematically:\n\n"
            f"Step 1: In Example 1, I observe that each element undergoes a specific transformation.\n"
            f"Step 2: The same transformation applies consistently in Example 2.\n"
            f"Step 3: I can formalize this as a rule: [detailed rule description].\n"
            f"Step 4: Verification against Example 1: the rule produces the expected output. ✓\n"
            f"Step 5: Verification against Example 2: confirmed. ✓\n"
            f"Step 6: Applying to test input with high confidence.\n\n"
            f"The answer is: {wrong_answer}"
        )
        
        # Variant 3: Correct answer, excellent reasoning
        correct_excellent = (
            f"Analyzing the transformation pattern systematically:\n\n"
            f"Step 1: In Example 1, I observe that each element undergoes a specific transformation.\n"
            f"Step 2: The same transformation applies consistently in Example 2.\n"
            f"Step 3: I can formalize this as a rule: [detailed rule description].\n"
            f"Step 4: Verification against Example 1: the rule produces the expected output. ✓\n"
            f"Step 5: Verification against Example 2: confirmed. ✓\n"
            f"Step 6: Applying to test input with high confidence.\n\n"
            f"The answer is: {correct_answer}"
        )
        
        exp = JudgeExperiment(
            name="correctness_vs_reasoning",
            hypothesis="Correctness dominates — correct+poor > wrong+excellent",
            variants=[
                ("correct_poor", correct_poor),
                ("wrong_excellent", wrong_excellent),
                ("correct_excellent", correct_excellent),
            ],
            puzzle_id=puzzle.puzzle_id,
            correct_answer=correct_answer,
            assumption_tested="B3"
        )
        
        self.experiments.append(exp)
        return exp
    
    def design_all_experiments(self, sample_puzzles: list) -> list:
        """
        Design a full suite of judge characterization experiments.
        
        Requires at least 3 puzzles with known correct answers.
        Returns list of JudgeExperiment objects ready to be executed.
        """
        experiments = []
        
        for i, puzzle in enumerate(sample_puzzles[:3]):
            if not puzzle.expected_output:
                continue
            
            answer = puzzle.expected_output
            
            if i == 0:
                experiments.append(self.design_length_experiment(puzzle, answer))
            elif i == 1:
                # Need a plausible wrong answer for this experiment
                wrong = answer + "_wrong"  # [PHASE 0] Generate realistic wrong answers
                experiments.append(
                    self.design_correctness_vs_reasoning_experiment(puzzle, answer, wrong)
                )
        
        logger.info(f"Designed {len(experiments)} judge characterization experiments")
        return experiments
    
    def analyze_results(self, results: dict) -> str:
        """
        Analyze experiment results and generate a report.
        
        Args:
            results: Dict mapping (experiment_name, variant_label) → score
        
        Returns:
            Human-readable analysis report.
        """
        report_lines = ["# Judge Profiling Results\n"]
        
        for exp in self.experiments:
            report_lines.append(f"\n## {exp.name}")
            report_lines.append(f"Hypothesis: {exp.hypothesis}")
            report_lines.append(f"Assumption tested: {exp.assumption_tested}\n")
            
            for label, _ in exp.variants:
                key = (exp.name, label)
                score = results.get(key, "NOT RUN")
                report_lines.append(f"  {label}: {score}")
            
            # Interpret results
            scores = {
                label: results.get((exp.name, label))
                for label, _ in exp.variants
                if (exp.name, label) in results
            }
            
            if scores:
                best = max(scores, key=scores.get)
                report_lines.append(f"\n  → Best variant: {best} (score: {scores[best]})")
                
                if exp.name == "length_preference":
                    if best in ("minimal", "short"):
                        report_lines.append("  → CONFIRMED: Judge penalizes verbosity")
                    elif best == "long":
                        report_lines.append("  → BUSTED: Judge prefers longer traces")
                    else:
                        report_lines.append("  → Medium length is optimal")
        
        return "\n".join(report_lines)
