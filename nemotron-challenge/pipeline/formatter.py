"""
SFT Training Data Formatter

Converts competition problems into training examples for LoRA SFT.
Each example: prompt + CoT reasoning trace + \\boxed{answer}.

CoT strategy by category:
  numeral_system        — programmatic Roman numeral breakdown (perfect)
  gravitational_constant — solve for g, apply d=0.5*g*t² (perfect)
  unit_conversion       — compute ratio from examples, apply (perfect)
  cipher                — build char map from examples, apply (~90%)
  bit_manipulation      — structured template (teaches format, not rule)
  symbol_transform      — structured template (teaches format, not rule)

Output: list of dicts with keys: id, prompt, response, category, solved_answer, ground_truth
"""

import re
import json
import logging
import random
from pathlib import Path
from typing import Optional

from pipeline.utils import Problem, BOXED_INSTRUCTION

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# ROMAN NUMERAL SOLVER
# ═══════════════════════════════════════════════════════════════════

_ROMAN_TABLE = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100,  "C"), (90,  "XC"), (50,  "L"), (40,  "XL"),
    (10,   "X"), (9,   "IX"), (5,   "V"), (4,   "IV"), (1, "I"),
]


def int_to_roman(n: int) -> str:
    result = []
    for value, numeral in _ROMAN_TABLE:
        while n >= value:
            result.append(numeral)
            n -= value
    return "".join(result)


def _cot_numeral(problem: Problem) -> tuple[str, str]:
    """Returns (cot_trace, solved_answer)."""
    match = re.search(r'(?:number|write)\s+(\d+)', problem.prompt, re.IGNORECASE)
    if not match:
        # Fallback: last number in prompt
        nums = re.findall(r'\b(\d+)\b', problem.prompt)
        if not nums:
            logger.debug(f"[{problem.id}] numeral: no number found in prompt, falling back to template")
            return _cot_template(problem)
        n_str = nums[-1]
    else:
        n_str = match.group(1)

    try:
        n = int(n_str)
        roman = int_to_roman(n)
    except (ValueError, TypeError):
        logger.debug(f"[{problem.id}] numeral: could not parse '{n_str}' as int, falling back to template")
        return _cot_template(problem)

    # Build step-by-step breakdown
    steps = []
    remaining = n
    parts = []
    for value, numeral in _ROMAN_TABLE:
        count = remaining // value
        if count:
            parts.append(f"{count}×{numeral}({value})")
            steps.append(f"  {remaining} ÷ {value} = {count} remainder {remaining % value}  → {''.join([numeral]*count)}")
            remaining %= value

    cot = (
        f"I need to convert {n} to the Wonderland numeral system (Roman numerals).\n\n"
        f"Breaking down {n}:\n"
        + "\n".join(steps) +
        f"\n\nCombining: {' + '.join(p.split('(')[0] for p in parts)} = {roman}"
    )
    return cot, roman


# ═══════════════════════════════════════════════════════════════════
# GRAVITY SOLVER
# ═══════════════════════════════════════════════════════════════════

def _cot_gravity(problem: Problem) -> tuple[str, str]:
    """Returns (cot_trace, solved_answer). Formula: d = 0.5 * g * t²"""
    if not problem.examples:
        logger.debug(f"[{problem.id}] gravity: no examples, falling back to template")
        return _cot_template(problem)

    # Fit g from examples: g = 2d/t²
    g_estimates = []
    for t_str, d_str in problem.examples:
        try:
            t, d = float(t_str), float(d_str)
            g_estimates.append(2 * d / (t ** 2))
        except (ValueError, ZeroDivisionError):
            logger.debug(f"[{problem.id}] gravity: skipping bad example t={t_str}, d={d_str}")
            continue

    if not g_estimates:
        logger.debug(f"[{problem.id}] gravity: no valid g estimates, falling back to template")
        return _cot_template(problem)

    g = sum(g_estimates) / len(g_estimates)

    # Extract query t from prompt
    t_match = re.search(r't\s*=\s*([\d.]+)\s*s', problem.prompt.split("Now,")[-1])
    if not t_match:
        # Try last float before end
        floats = re.findall(r'[\d.]+', problem.prompt.split("Now,")[-1])
        if not floats:
            logger.debug(f"[{problem.id}] gravity: no query t found, falling back to template")
            return _cot_template(problem)
        t_query = float(floats[0])
    else:
        t_query = float(t_match.group(1))

    d_answer = 0.5 * g * t_query ** 2
    solved = f"{d_answer:.2f}"

    # Show g derivation from first two examples
    ex = problem.examples
    cot_lines = [
        f"Using the formula d = 0.5 × g × t², I'll find the hidden gravitational constant g.",
        f"",
        f"From the examples:",
    ]
    for t_str, d_str in ex[:3]:
        t_ex, d_ex = float(t_str), float(d_str)
        g_ex = 2 * d_ex / (t_ex ** 2)
        cot_lines.append(f"  t={t_ex}s, d={d_ex}m → g = 2×{d_ex}/{t_ex}² = {g_ex:.4f} m/s²")

    cot_lines += [
        f"",
        f"Average g ≈ {g:.4f} m/s²",
        f"",
        f"For t = {t_query}s:",
        f"  d = 0.5 × {g:.4f} × {t_query}² = 0.5 × {g:.4f} × {t_query**2:.4f} = {d_answer:.2f} m",
    ]
    return "\n".join(cot_lines), solved


# ═══════════════════════════════════════════════════════════════════
# UNIT CONVERSION SOLVER
# ═══════════════════════════════════════════════════════════════════

def _cot_unit(problem: Problem) -> tuple[str, str]:
    """Returns (cot_trace, solved_answer). Formula: output = ratio × input"""
    if not problem.examples:
        logger.debug(f"[{problem.id}] unit: no examples, falling back to template")
        return _cot_template(problem)

    ratios = []
    for inp_str, out_str in problem.examples:
        try:
            inp_val = float(inp_str.split()[0])
            out_val = float(out_str.strip())
            ratios.append(out_val / inp_val)
        except (ValueError, ZeroDivisionError, IndexError):
            logger.debug(f"[{problem.id}] unit: skipping bad example in={inp_str}, out={out_str}")
            continue

    if not ratios:
        logger.debug(f"[{problem.id}] unit: no valid ratios, falling back to template")
        return _cot_template(problem)

    ratio = sum(ratios) / len(ratios)

    # Extract query value from "Now, convert the following measurement: 25.09 m"
    query_part = problem.prompt.split("Now,")[-1]
    val_match = re.search(r'([\d.]+)', query_part)
    if not val_match:
        logger.debug(f"[{problem.id}] unit: no query value found, falling back to template")
        return _cot_template(problem)

    x_query = float(val_match.group(1))
    answer = ratio * x_query
    solved = f"{answer:.2f}"

    cot_lines = [
        "Finding the conversion ratio from examples:",
        "",
    ]
    for inp_str, out_str in problem.examples[:3]:
        inp_val = float(inp_str.split()[0])
        out_val = float(out_str.strip())
        r = out_val / inp_val
        cot_lines.append(f"  {out_val} / {inp_val} = {r:.6f}")

    cot_lines += [
        f"",
        f"Average ratio ≈ {ratio:.6f}",
        f"",
        f"Applying to {x_query}:",
        f"  {x_query} × {ratio:.6f} = {answer:.2f}",
    ]
    return "\n".join(cot_lines), solved


# ═══════════════════════════════════════════════════════════════════
# CIPHER SOLVER
# ═══════════════════════════════════════════════════════════════════

def _build_cipher_map(examples: list[tuple[str, str]]) -> dict[str, str]:
    """Build char→char mapping from (ciphertext, plaintext) word pairs."""
    mapping = {}
    for cipher_text, plain_text in examples:
        cipher_words = cipher_text.split()
        plain_words = plain_text.split()
        for cw, pw in zip(cipher_words, plain_words):
            if len(cw) == len(pw):
                for cc, pc in zip(cw, pw):
                    if cc not in mapping:
                        mapping[cc] = pc
    return mapping


def _apply_cipher(mapping: dict, text: str) -> str:
    """Apply a char→char cipher mapping to text. Unmapped alpha chars become '?'."""
    return "".join(mapping.get(c, "?" if c.isalpha() else c) for c in text)


def _cot_cipher(problem: Problem) -> tuple[str, str]:
    """Returns (cot_trace, solved_answer)."""
    if not problem.examples:
        logger.debug(f"[{problem.id}] cipher: no examples, falling back to template")
        return _cot_template(problem)

    mapping = _build_cipher_map(problem.examples)

    # Extract query: look for "Now, decrypt the following text: <query>"
    query = ""
    decrypt_match = re.search(
        r'(?:decrypt|decipher|decode|translate)[^:]*:\s*([a-z][a-z ]+)',
        problem.prompt, re.IGNORECASE
    )
    if decrypt_match:
        query = decrypt_match.group(1).strip()

    if not query:
        logger.debug(f"[{problem.id}] cipher: no query text found, falling back to template")
        return _cot_template(problem)

    decoded = _apply_cipher(mapping, query)

    # Show mapping derivation from first example
    ex_cipher, ex_plain = problem.examples[0]
    sample_mappings = []
    for cw, pw in zip(ex_cipher.split()[:2], ex_plain.split()[:2]):
        if len(cw) == len(pw):
            for cc, pc in zip(cw[:4], pw[:4]):
                if f"{cc}→{pc}" not in sample_mappings:
                    sample_mappings.append(f"{cc}→{pc}")

    cot_lines = [
        f"Building the substitution cipher from the examples.",
        f"",
        f"From example 1: \"{ex_cipher}\" → \"{ex_plain}\"",
        f"  Character mappings: {', '.join(sample_mappings[:6])}{'...' if len(mapping) > 6 else ''}",
        f"",
        f"Total mappings discovered: {len(mapping)} characters",
        f"",
        f"Decoding query \"{query}\":",
        f"  {' '.join(mapping.get(c, '?') if c != ' ' else ' ' for c in query).strip()}",
        f"  = \"{decoded}\"",
    ]
    return "\n".join(cot_lines), decoded


# ═══════════════════════════════════════════════════════════════════
# TEMPLATE (bit_manipulation, symbol_transform)
# ═══════════════════════════════════════════════════════════════════

def _cot_template(problem: Problem) -> tuple[str, str]:
    """
    Structured template for categories where we can't derive the rule programmatically.
    The correct answer comes from ground truth — the CoT teaches the model the format
    and the approach (examine examples, identify rule, apply to query).
    """
    ex_lines = []
    for i, (inp, out) in enumerate(problem.examples[:4], 1):
        ex_lines.append(f"  Example {i}: {inp} → {out}")

    # Extract what looks like the query input
    query_match = re.search(
        r'(?:determine the (?:result|output) for|convert(?:ing)?)[:\s]+([^\n.]+)',
        problem.prompt, re.IGNORECASE
    )
    query = query_match.group(1).strip() if query_match else "[query]"

    answer = problem.answer or "?"

    cot_lines = [
        f"I need to identify the transformation rule from the examples and apply it to the query.",
        f"",
        f"Examining the input→output examples:",
        *ex_lines,
        f"",
        f"Analyzing the pattern across all {len(problem.examples)} examples, I can see a consistent",
        f"transformation rule applied to each input.",
        f"",
        f"Applying the rule to \"{query}\":",
        f"  The transformation yields: {answer}",
    ]
    return "\n".join(cot_lines), answer


# ═══════════════════════════════════════════════════════════════════
# FORMATTER
# ═══════════════════════════════════════════════════════════════════

class ProblemFormatter:
    """
    Converts Problem objects into SFT training examples.

    Each output example has:
      id            — problem id
      prompt        — competition prompt + boxed instruction
      response      — <think>CoT</think>\\n\\n\\boxed{answer}
      category      — problem category
      solved_answer — the answer derived by our solver (may differ from ground truth for templates)
      ground_truth  — the actual correct answer from train.csv
    """

    def format_one(self, problem: Problem) -> dict:
        """Format a single problem into an SFT training example."""
        cot, solved = self._generate_cot(problem)

        prompt = problem.prompt + BOXED_INSTRUCTION
        response = f"<think>\n{cot}\n</think>\n\n\\boxed{{{problem.answer}}}"

        return {
            "id": problem.id,
            "prompt": prompt,
            "response": response,
            "category": problem.category,
            "solved_answer": solved,
            "ground_truth": problem.answer,
        }

    def format_all(
        self,
        problems: list[Problem],
        val_split: float = 0.15,
        seed: int = 42,
    ) -> tuple[list[dict], list[dict]]:
        """
        Format all problems and split into train/val sets.

        Returns (train_examples, val_examples).
        """
        examples = [self.format_one(p) for p in problems if p.answer]
        logger.info(f"Formatted {len(examples)} problems")

        random.seed(seed)
        random.shuffle(examples)
        split = int(len(examples) * (1 - val_split))
        return examples[:split], examples[split:]

    def _generate_cot(self, problem: Problem) -> tuple[str, str]:
        """Dispatch to category-specific CoT generator."""
        cat = problem.category
        if cat == "numeral_system":
            return _cot_numeral(problem)
        elif cat == "gravitational_constant":
            return _cot_gravity(problem)
        elif cat == "unit_conversion":
            return _cot_unit(problem)
        elif cat == "cipher":
            return _cot_cipher(problem)
        else:
            return _cot_template(problem)

    def solver_accuracy(self, problems: list[Problem]) -> dict[str, dict]:
        """
        Evaluate how well the programmatic solvers match ground truth.
        Only meaningful for programmatic categories (numeral, gravity, unit, cipher).
        """
        from pipeline.utils import verify
        from collections import defaultdict

        stats = defaultdict(lambda: {"correct": 0, "total": 0})
        for p in problems:
            if not p.answer:
                continue
            _, solved = self._generate_cot(p)
            cat = p.category or "unknown"
            stats[cat]["total"] += 1
            if verify(p.answer, solved):
                stats[cat]["correct"] += 1

        return {
            cat: {
                "correct": v["correct"],
                "total": v["total"],
                "accuracy": v["correct"] / max(v["total"], 1),
            }
            for cat, v in stats.items()
        }


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def save_jsonl(examples: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    logger.info(f"Saved {len(examples)} examples to {path}")


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from pipeline.utils import load_problems, setup_logging
    setup_logging()

    data_path = "data/train.csv"
    problems = load_problems(data_path)

    fmt = ProblemFormatter()

    print("Checking solver accuracy...")
    acc = fmt.solver_accuracy(problems)
    for cat in sorted(acc):
        r = acc[cat]
        print(f"  {cat:<25} {r['correct']:>5}/{r['total']:<5} ({r['accuracy']*100:.1f}%)")

    print("\nFormatting training data...")
    train_ex, val_ex = fmt.format_all(problems)
    save_jsonl(train_ex, "data/train_formatted.jsonl")
    save_jsonl(val_ex, "data/val_formatted.jsonl")
    print(f"Train: {len(train_ex)}, Val: {len(val_ex)}")

    # Print a sample from each category
    print("\nSample formatted examples:")
    seen = set()
    for ex in train_ex:
        cat = ex["category"]
        if cat not in seen:
            seen.add(cat)
            print(f"\n--- {cat} ---")
            print(f"Prompt (truncated): {ex['prompt'][:120]}...")
            print(f"Response:\n{ex['response'][:400]}...")
        if len(seen) == 6:
            break


if __name__ == "__main__":
    main()
