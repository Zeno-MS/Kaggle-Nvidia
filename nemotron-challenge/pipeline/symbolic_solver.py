"""
Stage 1: Symbolic Solver

Attempts to solve puzzles programmatically by discovering transformation rules
and verifying them against training examples. This is our highest-value stage
because verified correct answers score maximum points on the correctness axis
(+0.80 weight in judge scoring).

Solver Portfolio:
  1. DSL Enumeration — Brute-force search over short programs in a puzzle-specific DSL
  2. LLM-Guided Synthesis — Use Nemotron to generate candidate Python programs
  3. ILP (Optional) — Inductive logic programming via Popper

ALL solvers share the same verification protocol: a candidate solution is only
accepted if it correctly transforms ALL training examples.

IMPORTANT: This module's internals will be heavily adapted after Phase 0 reveals
the actual puzzle format. The current implementation is a structural skeleton
with extension points.
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional, Callable
from abc import ABC, abstractmethod

from pipeline.config import Config

logger = logging.getLogger(__name__)


@dataclass
class SymbolicResult:
    """Result from the symbolic solver."""
    answer: str                     # The predicted output for the test input
    execution_trace: str            # Human-readable trace of how the answer was derived
    confidence: float               # 1.0 if verified against all examples, lower otherwise
    solver_name: str                # Which solver found this solution
    program: Optional[str] = None   # The program/rule discovered (for debugging)


class BaseSolver(ABC):
    """Abstract base class for individual symbolic solvers."""
    
    @abstractmethod
    def attempt(self, puzzle, timeout: float) -> Optional[SymbolicResult]:
        """
        Attempt to solve the puzzle symbolically.
        Returns SymbolicResult if successful, None if not.
        """
        pass


class DSLEnumerator(BaseSolver):
    """
    Brute-force enumeration over short programs in a domain-specific language.
    
    PHASE 0 TODO:
    - Define DSL primitives based on actual puzzle types
    - Implement the DSL interpreter
    - Calibrate max_depth based on puzzle complexity
    
    The DSL should cover primitive operations discovered during benchmark analysis.
    For grid-like puzzles: rotate, flip, fill, filter, shift, recolor, etc.
    For string puzzles: reverse, substitute, insert, delete, repeat, etc.
    For symbolic puzzles: increment, swap, map, fold, etc.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.primitives = []  # [PHASE 0] Define after seeing puzzles
        self.max_depth = config.symbolic.max_program_depth
        self.max_candidates = config.symbolic.max_candidates
    
    def attempt(self, puzzle, timeout: float) -> Optional[SymbolicResult]:
        """
        Enumerate DSL programs up to max_depth, testing each against
        training examples. Return the first program that passes all examples.
        """
        start = time.time()
        
        if not self.primitives:
            logger.debug("DSL primitives not yet defined — skipping enumeration")
            return None
        
        candidates_tested = 0
        
        for program in self._enumerate_programs():
            if time.time() - start > timeout:
                logger.debug(f"DSL enumeration timeout after {candidates_tested} candidates")
                return None
            
            if candidates_tested >= self.max_candidates:
                logger.debug(f"DSL enumeration hit max candidates ({self.max_candidates})")
                return None
            
            candidates_tested += 1
            
            if self._verify_program(program, puzzle):
                answer = self._execute_program(program, puzzle.test_input)
                return SymbolicResult(
                    answer=answer,
                    execution_trace=self._format_execution_trace(program, puzzle),
                    confidence=1.0,  # Verified against all examples
                    solver_name="dsl_enumeration",
                    program=str(program)
                )
        
        return None
    
    def _enumerate_programs(self):
        """
        Generator that yields DSL programs in order of increasing complexity.
        
        [PHASE 0] Implement based on actual DSL primitives.
        """
        # Placeholder — implement after defining primitives
        return iter([])
    
    def _verify_program(self, program, puzzle) -> bool:
        """
        Test a candidate program against ALL training examples.
        Returns True only if the program correctly transforms every example.
        """
        for example_input, example_output in puzzle.training_examples:
            try:
                result = self._execute_program(program, example_input)
                if result != example_output:
                    return False
            except Exception:
                return False
        return True
    
    def _execute_program(self, program, input_data) -> str:
        """Execute a DSL program on an input. [PHASE 0] Implement."""
        raise NotImplementedError("Implement after defining DSL")
    
    def _format_execution_trace(self, program, puzzle) -> str:
        """Format a human-readable trace of the program's execution."""
        return f"Applied rule: {program}"


class LLMProgramSynthesizer(BaseSolver):
    """
    Use an LLM to generate candidate Python programs that implement the
    transformation, then verify them against training examples.
    
    This is the most flexible solver — it can handle arbitrary transformation
    types as long as the LLM can express them in Python.
    
    Inspired by Jeremy Berman's ARC-AGI approach (79.6% accuracy) and
    Eric Pang's DreamCoder-inspired library learning.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.max_attempts = config.symbolic.synthesis_max_attempts
        self.temperature = config.symbolic.synthesis_temperature
        self._model = None  # Lazy loaded
    
    def attempt(self, puzzle, timeout: float) -> Optional[SymbolicResult]:
        """
        Generate candidate Python programs via LLM, test each against
        training examples, return the first verified solution.
        """
        start = time.time()
        
        for attempt_num in range(self.max_attempts):
            if time.time() - start > timeout:
                logger.debug(f"LLM synthesis timeout after {attempt_num} attempts")
                return None
            
            # Generate candidate program
            prompt = self._build_synthesis_prompt(puzzle, attempt_num)
            candidate_code = self._generate_program(prompt)
            
            if candidate_code is None:
                continue
            
            # Verify against training examples
            verified, test_output = self._verify_and_execute(candidate_code, puzzle)
            
            if verified and test_output is not None:
                return SymbolicResult(
                    answer=test_output,
                    execution_trace=self._format_execution_trace(candidate_code, puzzle),
                    confidence=1.0,
                    solver_name="llm_program_synthesis",
                    program=candidate_code
                )
        
        return None
    
    def _build_synthesis_prompt(self, puzzle, attempt_num: int) -> str:
        """
        Build the prompt for program synthesis.
        
        Key design choices (from ARC Prize research):
        - Include multiple representations of the examples
        - Ask for a Python function, not just a description
        - On retry attempts, include previous failures as negative examples
        """
        examples_str = ""
        for i, (inp, out) in enumerate(puzzle.training_examples):
            examples_str += f"Example {i+1}:\n  Input:  {inp}\n  Output: {out}\n\n"
        
        prompt = f"""You are solving a transformation puzzle. Given input-output examples,
write a Python function `transform(input_data)` that implements the transformation rule.

{examples_str}

Requirements:
- The function must work for ALL examples above
- It must be a pure function (no side effects)
- Return the transformed output in the same format as the examples
- Be concise — implement the simplest rule that explains all examples

```python
def transform(input_data):
"""
        
        if attempt_num > 0:
            prompt += f"\n\n# Note: Previous attempt(s) failed. Try a DIFFERENT approach."
        
        return prompt
    
    def _generate_program(self, prompt: str) -> Optional[str]:
        """
        Call the LLM to generate a candidate program.
        
        [PHASE 0] Wire up to actual Nemotron Nano inference.
        """
        # TODO: Implement actual LLM inference
        # For now, return None (placeholder)
        logger.debug("LLM program synthesis not yet wired to model inference")
        return None
    
    def _verify_and_execute(self, code: str, puzzle) -> tuple:
        """
        Safely execute candidate code against training examples,
        then execute on test input if all examples pass.
        
        Returns (verified: bool, test_output: str or None)
        """
        try:
            # Create isolated execution environment
            exec_globals = {}
            exec(code, exec_globals)
            
            transform_fn = exec_globals.get('transform')
            if transform_fn is None:
                return False, None
            
            # Verify against all training examples
            for inp, expected_out in puzzle.training_examples:
                result = transform_fn(inp)
                if str(result) != str(expected_out):
                    return False, None
            
            # All examples pass — execute on test input
            if puzzle.test_input is not None:
                test_output = str(transform_fn(puzzle.test_input))
                return True, test_output
            
            return True, None
            
        except Exception as e:
            logger.debug(f"Program execution failed: {e}")
            return False, None
    
    def _format_execution_trace(self, code: str, puzzle) -> str:
        """Format the program synthesis result as a readable trace."""
        return (
            f"I discovered a transformation rule by analyzing the examples.\n"
            f"The rule can be expressed as the following program:\n\n"
            f"```python\n{code}\n```\n\n"
            f"I verified this rule produces correct outputs for all training examples."
        )


class SymbolicSolver:
    """
    Portfolio solver that runs multiple symbolic approaches and returns
    the first verified solution.
    
    Design principle: Run solvers in order of speed (fastest first).
    Each solver has a timeout; if it fails, the next one gets the
    remaining time budget.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.solvers = []
        
        # Build solver portfolio based on config
        if config.symbolic.use_dsl_enumeration:
            self.solvers.append(("dsl", DSLEnumerator(config)))
        
        if config.symbolic.use_llm_program_synthesis:
            self.solvers.append(("llm_synthesis", LLMProgramSynthesizer(config)))
        
        # ILP solver would be added here if Prolog is available
        # if config.symbolic.use_ilp:
        #     self.solvers.append(("ilp", ILPSolver(config)))
        
        logger.info(f"Symbolic solver portfolio: {[name for name, _ in self.solvers]}")
    
    def solve(self, puzzle, timeout: float) -> Optional[SymbolicResult]:
        """
        Try each solver in the portfolio until one succeeds or all fail.
        Remaining time budget cascades to the next solver.
        """
        start = time.time()
        
        for solver_name, solver in self.solvers:
            elapsed = time.time() - start
            remaining = timeout - elapsed
            
            if remaining <= 0:
                logger.debug("Symbolic solver portfolio time exhausted")
                return None
            
            logger.debug(f"Trying {solver_name} ({remaining:.1f}s remaining)")
            
            try:
                result = solver.attempt(puzzle, timeout=remaining)
                if result is not None:
                    return result
            except Exception as e:
                logger.warning(f"Solver {solver_name} crashed: {e}")
                continue
        
        return None
