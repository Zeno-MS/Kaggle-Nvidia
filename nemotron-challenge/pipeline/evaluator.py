"""
Evaluator — Colab GPU Inference Harness

Runs Nemotron-3-Nano-30B on competition problems and measures accuracy.
Designed to run on Colab A100. Replicates the competition metric exactly.

Usage (in Colab):
    from pipeline.evaluator import Evaluator
    from pipeline.utils import load_problems

    # Baseline (no LoRA)
    ev = Evaluator(model_path="/path/to/nemotron-nano")
    results = ev.run(problems[:500])
    ev.report(results)

    # Post-training (with LoRA)
    ev = Evaluator(model_path="/path/to/nemotron-nano", lora_path="/path/to/adapter")
    results = ev.run(problems)
    ev.report(results)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pipeline.utils import Problem, BOXED_INSTRUCTION, extract_final_answer, verify

logger = logging.getLogger(__name__)

# Competition eval parameters — [VERIFIED] from metric source code.
# These mirror Config.eval defaults but are kept as module constants so the
# evaluator works without instantiating Config (e.g. in Colab notebooks).
EVAL_TEMPERATURE = 1.0
EVAL_TOP_P = 1.0
EVAL_MAX_TOKENS = 3584
EVAL_MAX_MODEL_LEN = 4096
EVAL_GPU_MEMORY_UTILIZATION = 0.85


@dataclass
class EvalResult:
    problem_id: str
    category: str
    expected: str
    predicted: Optional[str]
    raw_output: str
    correct: bool
    latency_sec: float = 0.0


@dataclass
class EvalSummary:
    results: list[EvalResult] = field(default_factory=list)
    total: int = 0
    correct: int = 0
    accuracy: float = 0.0
    per_category: dict = field(default_factory=dict)
    lora_path: Optional[str] = None
    elapsed_sec: float = 0.0


class Evaluator:
    """
    vLLM-based evaluator. Replicates competition metric exactly.

    Requires GPU. Import and run in Colab.
    """

    def __init__(
        self,
        model_path: str,
        lora_path: Optional[str] = None,
        tensor_parallel_size: int = 1,
    ):
        self.model_path = model_path
        self.lora_path = lora_path
        self.tensor_parallel_size = tensor_parallel_size
        self._llm = None

    def _load_model(self):
        """Lazy-load vLLM model."""
        if self._llm is not None:
            return

        from vllm import LLM

        kwargs = dict(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=EVAL_GPU_MEMORY_UTILIZATION,
            max_model_len=EVAL_MAX_MODEL_LEN,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
        )

        if self.lora_path:
            kwargs["enable_lora"] = True
            kwargs["max_lora_rank"] = 32
            logger.info(f"Loading model with LoRA: {self.lora_path}")
        else:
            logger.info(f"Loading base model (no LoRA): {self.model_path}")

        try:
            self._llm = LLM(**kwargs)
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise
        logger.info("Model loaded.")

    def _build_prompt(self, problem: Problem) -> str:
        """Build the exact prompt the competition metric uses."""
        return problem.prompt + BOXED_INSTRUCTION

    def run(
        self,
        problems: list[Problem],
        batch_size: int = 128,
        sample_seed: int = 42,
    ) -> EvalSummary:
        """
        Run inference on a list of problems and return evaluation results.

        Args:
            problems: List of Problem objects (must have .answer for eval)
            batch_size: vLLM batch size
            sample_seed: Random seed for temp=1.0 sampling

        Returns:
            EvalSummary with per-problem results and aggregate stats
        """
        self._load_model()

        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        sampling_params = SamplingParams(
            temperature=EVAL_TEMPERATURE,
            top_p=EVAL_TOP_P,
            max_tokens=EVAL_MAX_TOKENS,
            seed=sample_seed,
        )

        lora_request = None
        if self.lora_path:
            lora_request = LoRARequest("adapter", 1, self.lora_path)

        # Build prompts
        prompts = []
        tokenizer = self._llm.get_tokenizer()
        for p in problems:
            raw_prompt = self._build_prompt(p)
            # Apply chat template with enable_thinking=True
            messages = [{"role": "user", "content": raw_prompt}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompts.append(formatted)

        logger.info(f"Running inference on {len(prompts)} problems...")
        start = time.time()

        outputs = self._llm.generate(
            prompts,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )

        elapsed = time.time() - start
        logger.info(f"Inference done in {elapsed:.1f}s ({elapsed/len(problems):.2f}s/problem)")

        # Score results
        results = []
        for problem, output in zip(problems, outputs):
            raw_output = output.outputs[0].text
            predicted = extract_final_answer(raw_output)
            correct = verify(problem.answer, predicted) if problem.answer else False

            results.append(EvalResult(
                problem_id=problem.id,
                category=problem.category or "unknown",
                expected=problem.answer or "",
                predicted=predicted,
                raw_output=raw_output,
                correct=correct,
            ))

        return self._summarize(results, elapsed)

    def _summarize(self, results: list[EvalResult], elapsed: float) -> EvalSummary:
        from collections import defaultdict

        total = len(results)
        correct = sum(1 for r in results if r.correct)

        by_cat = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in results:
            by_cat[r.category]["total"] += 1
            if r.correct:
                by_cat[r.category]["correct"] += 1

        per_category = {
            cat: {
                "correct": v["correct"],
                "total": v["total"],
                "accuracy": v["correct"] / max(v["total"], 1),
            }
            for cat, v in by_cat.items()
        }

        return EvalSummary(
            results=results,
            total=total,
            correct=correct,
            accuracy=correct / max(total, 1),
            per_category=per_category,
            lora_path=self.lora_path,
            elapsed_sec=elapsed,
        )

    @staticmethod
    def report(summary: EvalSummary) -> str:
        """Print a formatted accuracy report."""
        lora_tag = f" + LoRA({Path(summary.lora_path).name})" if summary.lora_path else " (baseline)"
        lines = [
            f"\n{'='*55}",
            f"  Evaluation Report{lora_tag}",
            f"{'='*55}",
            f"  Overall: {summary.correct}/{summary.total}  ({summary.accuracy*100:.1f}%)",
            f"  Time:    {summary.elapsed_sec:.1f}s  ({summary.elapsed_sec/max(summary.total,1):.2f}s/problem)",
            f"\n  Per-Category:",
            f"  {'Category':<25} {'Acc':>7}  {'Correct':>8}  {'Total':>6}",
            f"  {'-'*52}",
        ]
        for cat in sorted(summary.per_category):
            r = summary.per_category[cat]
            lines.append(
                f"  {cat:<25} {r['accuracy']*100:>6.1f}%  {r['correct']:>8}  {r['total']:>6}"
            )
        lines.append(f"{'='*55}\n")
        report = "\n".join(lines)
        print(report)
        return report

    def save_results(self, summary: EvalSummary, path: str):
        """Save full results to JSONL for later analysis."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for r in summary.results:
                f.write(json.dumps({
                    "id": r.problem_id,
                    "category": r.category,
                    "expected": r.expected,
                    "predicted": r.predicted,
                    "correct": r.correct,
                    "raw_output": r.raw_output[:500],  # truncate for storage
                }) + "\n")
        logger.info(f"Saved {len(summary.results)} results to {path}")


def stratified_sample(problems: list[Problem], n_per_category: int = 83) -> list[Problem]:
    """
    Sample n_per_category problems from each category for baseline estimation.
    Default 83/category × 6 categories ≈ 500 problems — fast baseline run.
    """
    import random
    from collections import defaultdict

    by_cat = defaultdict(list)
    for p in problems:
        by_cat[p.category or "unknown"].append(p)

    sample = []
    for cat, probs in by_cat.items():
        random.shuffle(probs)
        sample.extend(probs[:n_per_category])

    logger.info(
        f"Stratified sample: {len(sample)} problems "
        f"({n_per_category}/category × {len(by_cat)} categories)"
    )
    return sample
