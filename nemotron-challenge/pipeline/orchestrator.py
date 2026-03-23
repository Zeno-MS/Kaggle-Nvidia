"""
Pipeline Orchestrator

The main entry point for solving puzzles. Coordinates all four stages:
  1. Symbolic solver attempts the puzzle
  2. If solved: Generate judge-optimized trace around the verified answer
  3. Judge-alignment post-processing on the trace
  4. If not solved: Fall back to neural reasoning with structured prompts

The orchestrator also handles:
  - Adaptive compute allocation (more samples for harder puzzles)
  - Time budget management
  - Per-puzzle logging for experiment tracking
  - Graceful degradation if any stage fails
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional

from pipeline.config import Config

logger = logging.getLogger(__name__)


@dataclass
class Puzzle:
    """
    Represents a single puzzle from the benchmark.
    
    ADAPT THIS after Phase 0 when you know the actual data format.
    The fields below are placeholders based on our best assumptions.
    """
    puzzle_id: str
    instruction: str                        # The puzzle prompt / question
    training_examples: list = None          # List of (input, output) pairs if available
    test_input: Optional[str] = None        # The input to transform
    expected_output: Optional[str] = None   # Ground truth (only for train set)
    metadata: dict = None                   # Any additional puzzle metadata
    
    def __post_init__(self):
        if self.training_examples is None:
            self.training_examples = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Solution:
    """
    A complete solution for a puzzle, ready for submission.
    """
    puzzle_id: str
    answer: str                             # The final answer
    reasoning_trace: str                    # The full reasoning trace
    method: str                             # "symbolic", "neural", "hybrid"
    confidence: float = 0.0                 # 0-1 confidence estimate
    
    # Diagnostic metadata (for experiment tracking)
    symbolic_attempted: bool = False
    symbolic_succeeded: bool = False
    symbolic_time_sec: float = 0.0
    trace_generation_time_sec: float = 0.0
    neural_time_sec: float = 0.0
    total_time_sec: float = 0.0
    num_candidates_generated: int = 0
    
    def format_for_submission(self) -> str:
        """
        Format the solution for Kaggle submission.
        
        ADAPT THIS after Phase 0 when you know the expected submission format.
        Currently assumes the metric expects a reasoning trace followed by
        a final answer, matching Nemotron's native output format.
        """
        # TODO: Adapt based on actual submission format requirements
        return f"{self.reasoning_trace}\n\nFinal Answer: {self.answer}"


class Orchestrator:
    """
    Main pipeline orchestrator. Coordinates all stages to solve puzzles.
    
    Usage:
        cfg = Config()
        orch = Orchestrator(cfg)
        solution = orch.solve(puzzle)
    """
    
    def __init__(self, config: Config):
        self.config = config
        self._symbolic_solver = None
        self._trace_generator = None
        self._judge_optimizer = None
        self._neural_fallback = None
        self._initialized = False
    
    def initialize(self):
        """
        Lazy initialization of all pipeline components.
        Call this once before solving; separates construction from execution
        so you can modify config after creating the orchestrator.
        """
        if self._initialized:
            return
        
        # Import here to allow graceful degradation if a component isn't ready
        if self.config.symbolic.enabled:
            from pipeline.symbolic_solver import SymbolicSolver
            self._symbolic_solver = SymbolicSolver(self.config)
            logger.info("Symbolic solver initialized")
        
        if self.config.trace.enabled:
            from pipeline.trace_generator import TraceGenerator
            self._trace_generator = TraceGenerator(self.config)
            logger.info("Trace generator initialized")
        
        from pipeline.judge_optimizer import JudgeOptimizer
        self._judge_optimizer = JudgeOptimizer(self.config)
        logger.info("Judge optimizer initialized")
        
        if self.config.neural.enabled:
            from pipeline.neural_fallback import NeuralFallback
            self._neural_fallback = NeuralFallback(self.config)
            logger.info("Neural fallback initialized")
        
        self._initialized = True
        logger.info(f"Pipeline initialized:\n{self.config.summary()}")
    
    def solve(self, puzzle: Puzzle) -> Solution:
        """
        Solve a single puzzle through the full pipeline.
        
        Flow:
          1. Try symbolic solving (if enabled)
          2. If symbolic succeeds: wrap answer in judge-optimized trace
          3. If symbolic fails: fall back to neural reasoning
          4. Apply judge-alignment post-processing
          5. Return best solution
        """
        self.initialize()
        
        start_time = time.time()
        solution = Solution(puzzle_id=puzzle.puzzle_id, answer="", reasoning_trace="", method="none")
        
        time_budget = self._get_time_budget(puzzle)
        
        # ── Stage 1: Symbolic Solving ──────────────────────────────────
        symbolic_answer = None
        symbolic_trace = None
        
        if self._symbolic_solver and self.config.orchestrator.symbolic_first:
            logger.info(f"[{puzzle.puzzle_id}] Stage 1: Attempting symbolic solve...")
            solution.symbolic_attempted = True
            
            sym_start = time.time()
            try:
                symbolic_result = self._symbolic_solver.solve(puzzle, timeout=self.config.symbolic.timeout_per_puzzle_sec)
                solution.symbolic_time_sec = time.time() - sym_start
                
                if symbolic_result is not None:
                    symbolic_answer = symbolic_result.answer
                    symbolic_trace = symbolic_result.execution_trace
                    solution.symbolic_succeeded = True
                    solution.confidence = symbolic_result.confidence
                    logger.info(f"[{puzzle.puzzle_id}] Symbolic solve SUCCEEDED (confidence={symbolic_result.confidence:.2f})")
                else:
                    logger.info(f"[{puzzle.puzzle_id}] Symbolic solve failed — falling back to neural")
            except Exception as e:
                solution.symbolic_time_sec = time.time() - sym_start
                logger.warning(f"[{puzzle.puzzle_id}] Symbolic solver error: {e}")
        
        # ── Stage 2 & 3: Trace Generation + Judge Optimization ────────
        if symbolic_answer is not None:
            # We have a verified answer — wrap it in a judge-optimized trace
            solution.method = "symbolic"
            solution.answer = symbolic_answer
            
            if self._trace_generator:
                trace_start = time.time()
                remaining_time = time_budget - (time.time() - start_time)
                
                trace_result = self._trace_generator.generate(
                    puzzle=puzzle,
                    answer=symbolic_answer,
                    execution_trace=symbolic_trace,
                    time_budget=max(remaining_time * 0.5, 5.0)
                )
                solution.reasoning_trace = trace_result.best_trace
                solution.num_candidates_generated = trace_result.num_candidates
                solution.trace_generation_time_sec = time.time() - trace_start
            else:
                # No trace generator — use raw symbolic trace
                solution.reasoning_trace = symbolic_trace or f"The answer is {symbolic_answer}"
        
        # ── Stage 4: Neural Fallback ──────────────────────────────────
        elif self._neural_fallback:
            logger.info(f"[{puzzle.puzzle_id}] Stage 4: Neural fallback...")
            solution.method = "neural"
            
            neural_start = time.time()
            remaining_time = time_budget - (time.time() - start_time)
            
            neural_result = self._neural_fallback.solve(
                puzzle=puzzle,
                time_budget=max(remaining_time, 10.0)
            )
            
            solution.answer = neural_result.answer
            solution.reasoning_trace = neural_result.reasoning_trace
            solution.confidence = neural_result.confidence
            solution.num_candidates_generated = neural_result.num_candidates
            solution.neural_time_sec = time.time() - neural_start
        
        else:
            # Nothing worked — return empty solution
            logger.error(f"[{puzzle.puzzle_id}] All solvers disabled or failed")
            solution.method = "none"
        
        # ── Judge-Alignment Post-Processing ───────────────────────────
        if solution.reasoning_trace:
            solution.reasoning_trace = self._judge_optimizer.optimize(
                trace=solution.reasoning_trace,
                answer=solution.answer,
                puzzle=puzzle
            )
        
        solution.total_time_sec = time.time() - start_time
        
        self._log_solution(solution)
        return solution
    
    def solve_batch(self, puzzles: list) -> list:
        """
        Solve a batch of puzzles, managing global time budget.
        
        Returns list of Solutions in the same order as input puzzles.
        """
        self.initialize()
        
        solutions = []
        global_start = time.time()
        
        for i, puzzle in enumerate(puzzles):
            elapsed = time.time() - global_start
            remaining = self.config.orchestrator.total_time_budget_sec - elapsed
            
            if remaining <= 0:
                logger.warning(f"Time budget exhausted at puzzle {i}/{len(puzzles)}")
                # Return empty solutions for remaining puzzles
                for p in puzzles[i:]:
                    solutions.append(Solution(
                        puzzle_id=p.puzzle_id, answer="", reasoning_trace="",
                        method="timeout"
                    ))
                break
            
            logger.info(f"Solving puzzle {i+1}/{len(puzzles)} ({remaining:.0f}s remaining)")
            solution = self.solve(puzzle)
            solutions.append(solution)
        
        self._log_batch_summary(solutions)
        return solutions
    
    def _get_time_budget(self, puzzle: Puzzle) -> float:
        """
        Determine per-puzzle time budget. Can be adaptive based on
        estimated difficulty.
        """
        base_budget = self.config.orchestrator.time_per_puzzle_sec
        
        if not self.config.orchestrator.adaptive_sampling:
            return base_budget
        
        # TODO: Implement difficulty estimation based on puzzle features
        # For now, return base budget
        return base_budget
    
    def _log_solution(self, solution: Solution):
        """Log solution details for experiment tracking."""
        if self.config.orchestrator.log_per_puzzle:
            logger.info(
                f"[{solution.puzzle_id}] "
                f"method={solution.method} "
                f"confidence={solution.confidence:.2f} "
                f"sym_ok={solution.symbolic_succeeded} "
                f"time={solution.total_time_sec:.1f}s "
                f"candidates={solution.num_candidates_generated}"
            )
    
    def _log_batch_summary(self, solutions: list):
        """Log aggregate statistics for a batch of solutions."""
        total = len(solutions)
        symbolic_ok = sum(1 for s in solutions if s.symbolic_succeeded)
        neural = sum(1 for s in solutions if s.method == "neural")
        failed = sum(1 for s in solutions if s.method in ("none", "timeout"))
        avg_time = sum(s.total_time_sec for s in solutions) / max(total, 1)
        
        logger.info(
            f"=== Batch Summary ===\n"
            f"Total puzzles: {total}\n"
            f"Symbolic solved: {symbolic_ok} ({100*symbolic_ok/max(total,1):.1f}%)\n"
            f"Neural fallback: {neural} ({100*neural/max(total,1):.1f}%)\n"
            f"Failed/timeout: {failed} ({100*failed/max(total,1):.1f}%)\n"
            f"Avg time per puzzle: {avg_time:.1f}s"
        )
