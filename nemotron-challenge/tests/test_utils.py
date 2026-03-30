"""Tests for pipeline.utils — answer extraction, verification, categorization."""

import csv
import math
import tempfile
from pathlib import Path

import pytest

from pipeline.utils import (
    Problem,
    categorize,
    extract_final_answer,
    load_problems,
    parse_examples,
    verify,
)


# ═══════════════════════════════════════════════════════════════════
# extract_final_answer
# ═══════════════════════════════════════════════════════════════════

class TestExtractFinalAnswer:
    """Replicates the competition metric's extraction logic."""

    def test_boxed_simple(self):
        assert extract_final_answer(r"The answer is \boxed{42}") == "42"

    def test_boxed_last_wins(self):
        text = r"First \boxed{10} then \boxed{42}"
        assert extract_final_answer(text) == "42"

    def test_boxed_string_answer(self):
        assert extract_final_answer(r"\boxed{XLVII}") == "XLVII"

    def test_boxed_with_whitespace(self):
        assert extract_final_answer(r"\boxed{ 3.14 }") == "3.14"

    def test_boxed_empty(self):
        # Empty boxed — falls through to other strategies
        result = extract_final_answer(r"text \boxed{}")
        # Empty string from boxed, should still return it or fallback
        assert result is not None

    def test_final_answer_pattern(self):
        text = "After analysis, the final answer is: 99.5"
        assert extract_final_answer(text) == "99.5"

    def test_last_number_fallback(self):
        text = "The calculation gives 7 then 14 then 21"
        assert extract_final_answer(text) == "21"

    def test_negative_number(self):
        text = "The result is -5.5"
        assert extract_final_answer(text) == "-5.5"

    def test_no_answer_returns_none(self):
        assert extract_final_answer("no numbers here at all") is None

    def test_none_input(self):
        assert extract_final_answer(None) is None

    def test_empty_string(self):
        assert extract_final_answer("") is None

    def test_boxed_takes_priority_over_number(self):
        text = r"I got 100 but actually \boxed{42}"
        assert extract_final_answer(text) == "42"

    def test_binary_answer(self):
        assert extract_final_answer(r"\boxed{01010101}") == "01010101"

    def test_multiline_with_thinking(self):
        text = """Let me think step by step.
        First, I note that 5 + 3 = 8.
        Then 8 * 2 = 16.
        \\boxed{16}"""
        assert extract_final_answer(text) == "16"


# ═══════════════════════════════════════════════════════════════════
# verify
# ═══════════════════════════════════════════════════════════════════

class TestVerify:
    """Replicates the competition metric's verify() function."""

    def test_exact_match(self):
        assert verify("42", "42") is True

    def test_case_insensitive_strings(self):
        assert verify("XLVII", "xlvii") is True

    def test_numeric_close(self):
        assert verify("3.14", "3.14159") is True  # within 1% rel_tol

    def test_numeric_not_close(self):
        assert verify("100", "50") is False

    def test_numeric_near_zero(self):
        # abs_tol=1e-5 for values near zero
        assert verify("0.0", "0.000001") is True
        assert verify("0.0", "1.0") is False

    def test_predicted_none(self):
        assert verify("42", None) is False

    def test_whitespace_handling(self):
        assert verify("  42  ", " 42") is True

    def test_string_mismatch(self):
        assert verify("abc", "xyz") is False

    def test_binary_string(self):
        assert verify("01010101", "01010101") is True
        assert verify("01010101", "10101010") is False

    def test_roman_numeral(self):
        assert verify("XLVII", "XLVII") is True
        assert verify("XLVII", "XLVI") is False

    def test_integer_vs_float(self):
        assert verify("42", "42.0") is True

    def test_negative_numbers(self):
        assert verify("-5.5", "-5.5") is True
        assert verify("-5.5", "5.5") is False


# ═══════════════════════════════════════════════════════════════════
# categorize
# ═══════════════════════════════════════════════════════════════════

class TestCategorize:
    """Tests for problem category inference."""

    def test_bit_manipulation(self):
        p = Problem(id="1", prompt="Apply bit manipulation rules to 01010101")
        assert categorize(p) == "bit_manipulation"

    def test_gravitational_constant(self):
        p = Problem(id="2", prompt="Find the gravitational constant from the data")
        assert categorize(p) == "gravitational_constant"

    def test_cipher(self):
        p = Problem(id="3", prompt="Use this secret encryption cipher to decode")
        assert categorize(p) == "cipher"

    def test_numeral_system(self):
        p = Problem(id="4", prompt="Convert using this numeral system")
        assert categorize(p) == "numeral_system"

    def test_unit_conversion(self):
        p = Problem(id="5", prompt="Use the unit conversion table to convert")
        assert categorize(p) == "unit_conversion"

    def test_symbol_transform(self):
        p = Problem(id="6", prompt="Apply the transformation rules to the equation x = y")
        assert categorize(p) == "symbol_transform"

    def test_fallback_binary_answer(self):
        p = Problem(id="7", prompt="Unknown type", answer="01100110")
        assert categorize(p) == "bit_manipulation"

    def test_fallback_roman_answer(self):
        p = Problem(id="8", prompt="Unknown type", answer="XLVII")
        assert categorize(p) == "numeral_system"

    def test_fallback_numeric_answer(self):
        p = Problem(id="9", prompt="Unknown type", answer="3.14")
        assert categorize(p) == "numeric_unknown"

    def test_fallback_unknown(self):
        p = Problem(id="10", prompt="Unknown type", answer="hello world")
        assert categorize(p) == "unknown"


# ═══════════════════════════════════════════════════════════════════
# parse_examples
# ═══════════════════════════════════════════════════════════════════

class TestParseExamples:
    """Tests for extracting input→output pairs from prompts."""

    def test_arrow_format(self):
        prompt = "Examples:\n01010001 -> 11011101\n10101010 -> 01010101\nQuery: 11110000"
        examples = parse_examples(prompt)
        assert len(examples) == 2
        assert examples[0] == ("01010001", "11011101")

    def test_unicode_arrow(self):
        prompt = "abc → xyz\ndef → ghi"
        examples = parse_examples(prompt)
        assert len(examples) == 2

    def test_becomes_format(self):
        prompt = "5 miles becomes 8.04672 kilometers\n10 miles becomes 16.0934 kilometers"
        examples = parse_examples(prompt)
        assert len(examples) == 2
        assert examples[0] == ("5 miles", "8.04672 kilometers")

    def test_gravity_format(self):
        prompt = "For t = 2.0 s, distance = 19.6 m\nFor t = 3.0 s, distance = 44.1 m"
        examples = parse_examples(prompt)
        assert len(examples) == 2

    def test_equals_format(self):
        prompt = '61"88 = 27\n42"13 = 5'
        examples = parse_examples(prompt)
        assert len(examples) == 2

    def test_skips_description_lines(self):
        prompt = "Here are some examples of input → output:\nabc → xyz"
        examples = parse_examples(prompt)
        # Should skip the "Here are some examples" line
        assert all("example" not in ex[0].lower() for ex in examples)

    def test_empty_prompt(self):
        assert parse_examples("") == []


# ═══════════════════════════════════════════════════════════════════
# load_problems
# ═══════════════════════════════════════════════════════════════════

class TestLoadProblems:
    """Tests for CSV loading."""

    def test_load_with_answers(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=["id", "prompt", "answer"])
            writer.writeheader()
            writer.writerow({"id": "p1", "prompt": "Find gravitational constant from data", "answer": "9.8"})
            writer.writerow({"id": "p2", "prompt": "Apply bit manipulation to 01010101", "answer": "10101010"})
            f.flush()
            path = f.name

        problems = load_problems(path)
        assert len(problems) == 2
        assert problems[0].id == "p1"
        assert problems[0].answer == "9.8"
        assert problems[0].category is not None
        Path(path).unlink()

    def test_load_without_answers(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=["id", "prompt"])
            writer.writeheader()
            writer.writerow({"id": "t1", "prompt": "Some test problem"})
            f.flush()
            path = f.name

        problems = load_problems(path)
        assert len(problems) == 1
        assert problems[0].answer is None
        Path(path).unlink()

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_problems("/nonexistent/path.csv")
