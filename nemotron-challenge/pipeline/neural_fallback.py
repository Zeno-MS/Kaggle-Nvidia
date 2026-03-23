"""
Stage 4: Neural Fallback

When the symbolic solver cannot find a verified solution, this stage uses
Nemotron 3 Nano's native reasoning capabilities with structured prompts
and self-consistency voting.

This is the "standard playbook" but enhanced with:
  1. Cognitively structured prompt templates (from Trace Architecture)
  2. Self-consistency voting across multiple reasoning paths
  3. Few-shot examples sourced from symbolically-solved puzzles (when available)
  4. Reasoning budget control optimized for the time/accuracy tradeoff

Design principle: Quality over quantity. Better to generate fewer, higher-quality
reasoning paths than many low-quality ones — especially given the verbosity penalty.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter

from pipeline.config import Config

logger = logging.getLogger(__name__)


@dataclass
class NeuralResult:
    """Result from neural reasoning."""
    answer: str
    reasoning_trace: str
    confidence: float               # Based on voting agreement
    num_candidates: int
    vote_distribution: dict = field(default_factory=dict)  # answer → count
    all_traces: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_REASONING_ON = """You are a precise logical reasoning assistant. 
When solving transformation puzzles:
- Analyze the input-output examples systematically
- Identify the transformation rule step by step
- Verify your rule against each example
- Apply the rule to produce the answer
- State your final answer clearly

Be concise. Every sentence should advance the solution."""


def build_puzzle_prompt(puzzle, few_shot_examples: list = None) -> str:
    """
    Build the user prompt for a puzzle, following the cognitive trace template.
    
    Structure:
      1. Task instruction (brief)
      2. Few-shot examples if available
      3. The actual puzzle with training examples
      4. Test input requiring answer
    
    [PHASE 0] Adapt based on actual puzzle format.
    """
    parts = []
    
    # Task instruction
    parts.append(
        "Identify the transformation rule from the examples below, "
        "then apply it to the test input."
    )
    
    # Few-shot solved examples (from symbolic solver successes)
    if few_shot_examples:
        parts.append("\nHere are some solved examples for reference:")
        for i, ex in enumerate(few_shot_examples[:3]):
            parts.append(f"\n--- Solved Example {i+1} ---")
            parts.append(f"Training pairs: {ex.get('training_pairs', 'N/A')}")
            parts.append(f"Rule discovered: {ex.get('rule', 'N/A')}")
            parts.append(f"Test answer: {ex.get('answer', 'N/A')}")
    
    # The actual puzzle
    parts.append("\n--- Your Puzzle ---")
    parts.append(f"Task: {puzzle.instruction}")
    
    if puzzle.training_examples:
        parts.append("\nTraining examples:")
        for i, (inp, out) in enumerate(puzzle.training_examples):
            parts.append(f"  Example {i+1}: Input: {inp} → Output: {out}")
    
    if puzzle.test_input is not None:
        parts.append(f"\nTest input: {puzzle.test_input}")
        parts.append("\nWhat is the output? Show your reasoning concisely, then state your answer.")
    
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════
# ANSWER EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def extract_answer(model_output: str) -> Optional[str]:
    """
    Extract the final answer from a model's reasoning output.
    
    Tries multiple extraction strategies in order:
      1. Look for explicit "answer is:" or "output is:" patterns
      2. Look for the last line after "therefore" or "thus"
      3. Take the last non-empty line as the answer
    
    [PHASE 0] Adapt based on actual puzzle output format.
    """
    import re
    
    # Strategy 1: Explicit answer markers
    patterns = [
        r'(?i)(?:the |my |final )?answer\s*(?:is|:)\s*(.+?)(?:\.|$)',
        r'(?i)(?:the |final )?output\s*(?:is|:)\s*(.+?)(?:\.|$)',
        r'(?i)result\s*(?:is|:)\s*(.+?)(?:\.|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, model_output)
        if match:
            return match.group(1).strip()
    
    # Strategy 2: After conclusion markers
    for marker in ['therefore', 'thus', 'hence', 'so the']:
        idx = model_output.lower().rfind(marker)
        if idx >= 0:
            after_marker = model_output[idx:].split('\n')[0]
            # Try to extract the value after the marker
            parts = after_marker.split(':')
            if len(parts) > 1:
                return parts[-1].strip().rstrip('.')
    
    # Strategy 3: Last non-empty line
    lines = [l.strip() for l in model_output.strip().split('\n') if l.strip()]
    if lines:
        return lines[-1]
    
    return None


# ═══════════════════════════════════════════════════════════════════
# NEURAL FALLBACK SOLVER
# ═══════════════════════════════════════════════════════════════════

class NeuralFallback:
    """
    Structured neural reasoning with self-consistency voting.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.neural_config = config.neural
        self._model = None
        self._solved_examples_cache = []  # Few-shot examples from symbolic solver
    
    def register_solved_example(self, puzzle, answer: str, rule: str):
        """
        Register a symbolically-solved puzzle for use as a few-shot example
        in future neural reasoning. This creates a virtuous cycle:
        symbolic successes improve neural performance on harder puzzles.
        """
        self._solved_examples_cache.append({
            "training_pairs": str(puzzle.training_examples[:2]),
            "rule": rule,
            "answer": answer
        })
        logger.debug(f"Registered solved example (total: {len(self._solved_examples_cache)})")
    
    def solve(self, puzzle, time_budget: float = 60.0) -> NeuralResult:
        """
        Solve a puzzle using neural reasoning with self-consistency.
        
        Flow:
          1. Build structured prompt
          2. Generate N reasoning paths
          3. Extract answer from each path
          4. Vote on the best answer
          5. Return the winning answer with its best trace
        """
        # Build prompt
        few_shot = self._select_few_shot(puzzle)
        prompt = build_puzzle_prompt(puzzle, few_shot)
        system = SYSTEM_PROMPT_REASONING_ON if self.neural_config.reasoning_on else ""
        
        # Generate multiple reasoning paths
        num_samples = self.neural_config.num_samples
        traces = self._generate_traces(system, prompt, num_samples, time_budget)
        
        if not traces:
            logger.warning("Neural fallback generated no traces")
            return NeuralResult(
                answer="", reasoning_trace="", confidence=0.0,
                num_candidates=0
            )
        
        # Extract answers and vote
        answers_with_traces = []
        for trace in traces:
            answer = extract_answer(trace)
            if answer:
                answers_with_traces.append((answer, trace))
        
        if not answers_with_traces:
            # No extractable answers — return first trace with empty answer
            return NeuralResult(
                answer="", reasoning_trace=traces[0], confidence=0.0,
                num_candidates=len(traces)
            )
        
        # Vote
        answer_counts = Counter(a for a, _ in answers_with_traces)
        best_answer = answer_counts.most_common(1)[0][0]
        agreement = answer_counts[best_answer] / len(answers_with_traces)
        
        # Select best trace for the winning answer
        best_trace = self._select_best_trace(
            [(a, t) for a, t in answers_with_traces if a == best_answer]
        )
        
        return NeuralResult(
            answer=best_answer,
            reasoning_trace=best_trace,
            confidence=agreement,
            num_candidates=len(traces),
            vote_distribution=dict(answer_counts),
            all_traces=traces
        )
    
    def _generate_traces(
        self,
        system_prompt: str,
        user_prompt: str,
        num_samples: int,
        time_budget: float
    ) -> list:
        """
        Generate multiple reasoning traces from the model.
        
        [PHASE 1] Wire to actual Nemotron Nano inference.
        This is the critical integration point with the competition infrastructure.
        
        Implementation options:
          - vLLM batch inference (fastest on GPU)
          - Transformers pipeline (simplest)
          - TensorRT-LLM (most optimized for Nemotron)
          - SGLang (good for structured generation)
        """
        # TODO: Implement actual model inference
        # Placeholder: return empty list
        logger.warning("Neural inference not yet wired — returning empty traces")
        return []
    
    def _select_few_shot(self, puzzle) -> list:
        """
        Select the most relevant few-shot examples from solved puzzles.
        
        Currently: random selection from cache.
        [PHASE 2] Implement similarity-based selection (e.g., embedding distance).
        """
        if not self._solved_examples_cache:
            return []
        
        n = min(self.neural_config.num_few_shot, len(self._solved_examples_cache))
        return self._solved_examples_cache[:n]
    
    def _select_best_trace(self, answer_trace_pairs: list) -> str:
        """
        Among traces that produced the winning answer, select the best one.
        
        Heuristic selection (no judge access):
          - Prefer traces with verification steps
          - Prefer traces closer to target length
          - Prefer traces with clearer structure
        
        [PHASE 2] Replace with judge-based selection when surrogate is available.
        """
        if len(answer_trace_pairs) == 1:
            return answer_trace_pairs[0][1]
        
        # Score each trace with simple heuristics
        scored = []
        for answer, trace in answer_trace_pairs:
            score = 0.0
            
            # Has verification? (+1)
            if any(w in trace.lower() for w in ['verify', 'check', 'confirm', '✓']):
                score += 1.0
            
            # Has clear steps? (+1)
            if any(w in trace.lower() for w in ['step 1', 'first,', '1.']):
                score += 1.0
            
            # Length in sweet spot (100-500 tokens)? (+1)
            est_tokens = len(trace) // 4
            if 100 <= est_tokens <= 500:
                score += 1.0
            
            # Penalize very long traces
            if est_tokens > 800:
                score -= 1.0
            
            scored.append((score, trace))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]
