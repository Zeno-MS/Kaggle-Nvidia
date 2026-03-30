"""Tests for pipeline.config — configuration loading and YAML round-trip."""

import tempfile
from pathlib import Path

import pytest
import yaml

from pipeline.config import Config, EvalConfig, LoRAConfig, TrainingConfig


class TestConfigDefaults:
    """Verify default values match competition-verified parameters."""

    def test_eval_defaults_match_competition(self):
        cfg = Config()
        assert cfg.eval.temperature == 1.0
        assert cfg.eval.top_p == 1.0
        assert cfg.eval.max_tokens == 3584
        assert cfg.eval.max_model_len == 4096
        assert cfg.eval.enable_thinking is True

    def test_lora_defaults_match_demo(self):
        cfg = Config()
        assert cfg.lora.rank == 32
        assert cfg.lora.alpha == 16
        assert cfg.lora.dropout == 0.05
        assert cfg.lora.bias == "none"
        assert cfg.lora.task_type == "CAUSAL_LM"

    def test_lora_target_modules_regex(self):
        """Target modules regex should match the competition demo."""
        import re
        cfg = Config()
        pattern = cfg.lora.target_modules
        # Should match these layer names
        assert re.match(pattern, "layer.0.in_proj")
        assert re.match(pattern, "blocks.5.out_proj")
        assert re.match(pattern, "model.layers.10.up_proj")
        assert re.match(pattern, "decoder.12.down_proj")
        # Should NOT match these
        assert not re.match(pattern, "layer.0.q_proj")
        assert not re.match(pattern, "embedding.weight")


class TestConfigYaml:
    """Test YAML round-trip."""

    def test_save_and_load(self):
        cfg = Config()
        cfg.training.learning_rate = 1e-5
        cfg.lora.rank = 16

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name

        cfg.save_yaml(path)
        cfg2 = Config()
        cfg2.load_yaml(path)

        assert cfg2.training.learning_rate == 1e-5
        assert cfg2.lora.rank == 16
        # Unchanged values should remain default
        assert cfg2.eval.temperature == 1.0
        Path(path).unlink()

    def test_load_partial_override(self):
        """YAML with only some fields should leave others as defaults."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"training": {"learning_rate": 5e-5}}, f)
            path = f.name

        cfg = Config()
        cfg.load_yaml(path)
        assert cfg.training.learning_rate == 5e-5
        assert cfg.training.num_epochs == 3  # unchanged
        assert cfg.lora.rank == 32  # unchanged
        Path(path).unlink()

    def test_load_invalid_key_ignored(self):
        """Unknown keys in YAML should be silently ignored."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"training": {"nonexistent_param": 999}}, f)
            path = f.name

        cfg = Config()
        cfg.load_yaml(path)  # should not raise
        assert not hasattr(cfg.training, "nonexistent_param")
        Path(path).unlink()


class TestConfigSummary:
    def test_summary_contains_key_info(self):
        cfg = Config()
        s = cfg.summary()
        assert "rank=32" in s
        assert "alpha=16" in s
        assert "temp=1.0" in s
