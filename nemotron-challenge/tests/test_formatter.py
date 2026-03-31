"""Tests for pipeline.formatter — CoT generation, Roman numeral solver, formatting."""

import json
import tempfile
from pathlib import Path

import pytest

from pipeline.utils import Problem, BOXED_INSTRUCTION
from pipeline.formatter import (
    ProblemFormatter,
    int_to_roman,
    save_jsonl,
    _build_cipher_map,
    _apply_cipher,
    _cot_numeral,
    _cot_gravity,
    _cot_unit,
    _cot_cipher,
    _cot_template,
)


# ═══════════════════════════════════════════════════════════════════
# int_to_roman
# ═══════════════════════════════════════════════════════════════════

class TestIntToRoman:
    def test_basic_values(self):
        assert int_to_roman(1) == "I"
        assert int_to_roman(5) == "V"
        assert int_to_roman(10) == "X"
        assert int_to_roman(50) == "L"
        assert int_to_roman(100) == "C"
        assert int_to_roman(500) == "D"
        assert int_to_roman(1000) == "M"

    def test_subtractive_notation(self):
        assert int_to_roman(4) == "IV"
        assert int_to_roman(9) == "IX"
        assert int_to_roman(40) == "XL"
        assert int_to_roman(90) == "XC"
        assert int_to_roman(400) == "CD"
        assert int_to_roman(900) == "CM"

    def test_compound_values(self):
        assert int_to_roman(42) == "XLII"
        assert int_to_roman(1994) == "MCMXCIV"
        assert int_to_roman(3999) == "MMMCMXCIX"
        assert int_to_roman(2024) == "MMXXIV"

    def test_zero(self):
        assert int_to_roman(0) == ""

    def test_repeated_numerals(self):
        assert int_to_roman(3) == "III"
        assert int_to_roman(30) == "XXX"
        assert int_to_roman(300) == "CCC"
        assert int_to_roman(3000) == "MMM"


# ═══════════════════════════════════════════════════════════════════
# _build_cipher_map / _apply_cipher
# ═══════════════════════════════════════════════════════════════════

class TestCipherHelpers:
    def test_build_map_simple(self):
        examples = [("abc", "xyz")]
        mapping = _build_cipher_map(examples)
        assert mapping["a"] == "x"
        assert mapping["b"] == "y"
        assert mapping["c"] == "z"

    def test_build_map_multiple_words(self):
        examples = [("ab cd", "xy zw")]
        mapping = _build_cipher_map(examples)
        assert mapping["a"] == "x"
        assert mapping["d"] == "w"

    def test_build_map_length_mismatch_skipped(self):
        # Words of different lengths should be skipped
        examples = [("abc", "xy")]
        mapping = _build_cipher_map(examples)
        assert len(mapping) == 0

    def test_build_map_first_mapping_wins(self):
        # First seen mapping is kept
        examples = [("ab", "xy"), ("ac", "xz")]
        mapping = _build_cipher_map(examples)
        assert mapping["a"] == "x"  # both agree

    def test_apply_cipher_full_mapping(self):
        mapping = {"a": "x", "b": "y", "c": "z"}
        assert _apply_cipher(mapping, "abc") == "xyz"

    def test_apply_cipher_unmapped_becomes_question(self):
        mapping = {"a": "x"}
        assert _apply_cipher(mapping, "abc") == "x??"

    def test_apply_cipher_preserves_non_alpha(self):
        mapping = {"a": "x"}
        assert _apply_cipher(mapping, "a 1 a") == "x 1 x"

    def test_apply_cipher_empty(self):
        assert _apply_cipher({}, "") == ""


# ═══════════════════════════════════════════════════════════════════
# _cot_numeral
# ═══════════════════════════════════════════════════════════════════

class TestCotNumeral:
    def test_basic_conversion(self):
        p = Problem(id="n1", prompt="Convert the number 42 to the numeral system", category="numeral_system", answer="XLII")
        cot, solved = _cot_numeral(p)
        assert solved == "XLII"
        assert "42" in cot
        assert "XLII" in cot

    def test_large_number(self):
        p = Problem(id="n2", prompt="Write 1994 in the numeral system", category="numeral_system", answer="MCMXCIV")
        cot, solved = _cot_numeral(p)
        assert solved == "MCMXCIV"

    def test_fallback_to_last_number(self):
        # When "number X" pattern doesn't match, uses last number in prompt
        p = Problem(id="n3", prompt="In this system, represent 7", category="numeral_system", answer="VII")
        cot, solved = _cot_numeral(p)
        assert solved == "VII"

    def test_no_number_falls_to_template(self):
        p = Problem(id="n4", prompt="Convert this value", category="numeral_system", answer="X",
                    examples=[("5", "V")])
        cot, solved = _cot_numeral(p)
        # Falls through to template — answer comes from ground truth
        assert solved == "X"


# ═══════════════════════════════════════════════════════════════════
# _cot_gravity
# ═══════════════════════════════════════════════════════════════════

class TestCotGravity:
    def test_basic_gravity(self):
        p = Problem(
            id="g1",
            prompt="Observe the data. Now, determine the distance for t = 3.0 s",
            category="gravitational_constant",
            answer="44.10",
            examples=[("1.0", "4.9"), ("2.0", "19.6")],
        )
        cot, solved = _cot_gravity(p)
        assert abs(float(solved) - 44.10) < 0.5
        assert "g" in cot.lower() or "gravitational" in cot.lower() or "0.5" in cot

    def test_no_examples_falls_to_template(self):
        p = Problem(id="g2", prompt="Now, t = 5.0 s", category="gravitational_constant",
                    answer="122.5", examples=[])
        cot, solved = _cot_gravity(p)
        assert solved == "122.5"  # template uses ground truth

    def test_bad_examples_falls_to_template(self):
        p = Problem(id="g3", prompt="Now, t = 2.0 s", category="gravitational_constant",
                    answer="19.6", examples=[("abc", "def")])
        cot, solved = _cot_gravity(p)
        assert solved == "19.6"  # template uses ground truth


# ═══════════════════════════════════════════════════════════════════
# _cot_unit
# ═══════════════════════════════════════════════════════════════════

class TestCotUnit:
    def test_basic_conversion(self):
        p = Problem(
            id="u1",
            prompt="Convert units. Now, convert the following measurement: 10.0 m",
            category="unit_conversion",
            answer="32.81",
            examples=[("1.0 m", "3.281"), ("2.0 m", "6.562")],
        )
        cot, solved = _cot_unit(p)
        assert abs(float(solved) - 32.81) < 0.1
        assert "ratio" in cot.lower()

    def test_no_examples_falls_to_template(self):
        p = Problem(id="u2", prompt="Now, convert: 5.0", category="unit_conversion",
                    answer="16.4", examples=[])
        cot, solved = _cot_unit(p)
        assert solved == "16.4"  # template


# ═══════════════════════════════════════════════════════════════════
# _cot_cipher
# ═══════════════════════════════════════════════════════════════════

class TestCotCipher:
    def test_full_mapping(self):
        p = Problem(
            id="c1",
            prompt='Decrypt the following text: bcd',
            category="cipher",
            answer="xyz",
            examples=[("abc", "wxy"), ("bcd", "xyz")],
        )
        cot, solved = _cot_cipher(p)
        assert "bcd" in cot or "xyz" in cot or "mapping" in cot.lower()

    def test_no_examples_falls_to_template(self):
        p = Problem(id="c2", prompt="Decrypt: hello", category="cipher",
                    answer="world", examples=[])
        cot, solved = _cot_cipher(p)
        assert solved == "world"  # template

    def test_no_query_falls_to_template(self):
        p = Problem(id="c3", prompt="No decrypt keyword here",
                    category="cipher", answer="test",
                    examples=[("abc", "xyz")])
        cot, solved = _cot_cipher(p)
        assert solved == "test"  # template


# ═══════════════════════════════════════════════════════════════════
# _cot_template
# ═══════════════════════════════════════════════════════════════════

class TestCotTemplate:
    def test_uses_ground_truth(self):
        p = Problem(id="t1", prompt="Apply transformation rules to determine the result for: 0101",
                    category="bit_manipulation", answer="1010",
                    examples=[("0000", "1111"), ("0011", "1100")])
        cot, solved = _cot_template(p)
        assert solved == "1010"
        assert "1010" in cot

    def test_shows_examples(self):
        p = Problem(id="t2", prompt="Determine the result for: query",
                    category="symbol_transform", answer="ans",
                    examples=[("a", "b"), ("c", "d")])
        cot, solved = _cot_template(p)
        assert "Example 1" in cot
        assert "a" in cot and "b" in cot

    def test_no_answer_uses_question_mark(self):
        p = Problem(id="t3", prompt="Determine the result for: query",
                    category="bit_manipulation", answer=None, examples=[])
        cot, solved = _cot_template(p)
        assert solved == "?"

    def test_no_examples(self):
        p = Problem(id="t4", prompt="Unknown thing", category="unknown",
                    answer="42", examples=[])
        cot, solved = _cot_template(p)
        assert solved == "42"
        assert "Example" not in cot  # no examples to show


# ═══════════════════════════════════════════════════════════════════
# ProblemFormatter
# ═══════════════════════════════════════════════════════════════════

class TestProblemFormatter:
    def _make_problem(self, category, answer="42"):
        return Problem(
            id=f"test_{category}",
            prompt=f"Test prompt for {category}",
            category=category,
            answer=answer,
            examples=[("a", "b")],
        )

    def test_format_one_structure(self):
        fmt = ProblemFormatter()
        p = self._make_problem("numeral_system", answer="XLII")
        p.prompt = "Convert the number 42 to the numeral system"
        result = fmt.format_one(p)

        assert result["id"] == "test_numeral_system"
        assert result["category"] == "numeral_system"
        assert result["ground_truth"] == "XLII"
        assert BOXED_INSTRUCTION in result["prompt"]
        assert "\\boxed{XLII}" in result["response"]
        assert "<think>" in result["response"]
        assert "</think>" in result["response"]

    def test_format_one_dispatches_by_category(self):
        fmt = ProblemFormatter()
        for cat in ["numeral_system", "gravitational_constant", "unit_conversion",
                     "cipher", "bit_manipulation", "symbol_transform"]:
            p = self._make_problem(cat)
            result = fmt.format_one(p)
            assert result["category"] == cat
            assert "<think>" in result["response"]

    def test_format_all_splits_train_val(self):
        fmt = ProblemFormatter()
        problems = [self._make_problem("bit_manipulation", answer=str(i)) for i in range(100)]
        train, val = fmt.format_all(problems, val_split=0.2, seed=42)
        assert len(train) == 80
        assert len(val) == 20

    def test_format_all_skips_no_answer(self):
        fmt = ProblemFormatter()
        problems = [
            self._make_problem("bit_manipulation", answer="1010"),
            Problem(id="no_ans", prompt="test", category="cipher", answer=None, examples=[]),
        ]
        train, val = fmt.format_all(problems, val_split=0.0)
        assert len(train) == 1  # only the one with answer

    def test_solver_accuracy(self):
        fmt = ProblemFormatter()
        p = Problem(
            id="num1",
            prompt="Convert the number 10 to the numeral system",
            category="numeral_system",
            answer="X",
            examples=[],
        )
        acc = fmt.solver_accuracy([p])
        assert "numeral_system" in acc
        assert acc["numeral_system"]["accuracy"] == 1.0


# ═══════════════════════════════════════════════════════════════════
# save_jsonl
# ═══════════════════════════════════════════════════════════════════

class TestSaveJsonl:
    def test_round_trip(self):
        examples = [
            {"id": "p1", "prompt": "test", "response": "\\boxed{42}"},
            {"id": "p2", "prompt": "test2", "response": "\\boxed{7}"},
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        save_jsonl(examples, path)

        loaded = []
        with open(path) as f:
            for line in f:
                loaded.append(json.loads(line))

        assert len(loaded) == 2
        assert loaded[0]["id"] == "p1"
        assert loaded[1]["response"] == "\\boxed{7}"
        Path(path).unlink()

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "sub" / "dir" / "out.jsonl")
            save_jsonl([{"a": 1}], path)
            assert Path(path).exists()
