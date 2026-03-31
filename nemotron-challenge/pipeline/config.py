"""
Configuration for the Nemotron Challenge LoRA training pipeline.

All evaluation and LoRA parameters marked [VERIFIED] are confirmed from
the competition metric source code and demo notebook (2026-03-26).

Usage:
    from pipeline.config import Config
    cfg = Config()
    cfg.load_yaml("configs/default.yaml")
"""

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Competition evaluation parameters. [VERIFIED] from metric source code."""
    model_id: str = "nemotron-3-nano-30b-a3b-bf16"
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 3584
    max_model_len: int = 4096
    max_num_seqs: int = 128
    gpu_memory_utilization: float = 0.85
    tensor_parallel_size: int = 1
    enable_thinking: bool = True
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    # Answer verification
    numeric_rel_tol: float = 1e-2
    numeric_abs_tol: float = 1e-5
    string_comparison: str = "case_insensitive"


@dataclass
class LoRAConfig:
    """LoRA adapter constraints. [VERIFIED] from demo notebook."""
    rank: int = 32
    alpha: int = 16
    target_modules: str = r".*\.(in_proj|out_proj|up_proj|down_proj)$"
    dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """LoRA SFT training parameters. [TUNABLE] — sweep these."""
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 4096
    bf16: bool = True
    validation_split: float = 0.15
    include_reasoning_in_target: bool = True
    answer_format: str = "\\boxed{}"


@dataclass
class DataConfig:
    """Dataset paths and synthetic data settings."""
    train_path: str = "data/train.csv"
    test_path: str = "data/test.csv"
    # Synthetic data generation
    teacher_models: list = field(default_factory=lambda: [
        "nemotron-3-super-120b-a12b",
        "deepseek-r1",
        "qwen-2.5-72b",
    ])
    target_multiplier: int = 10
    verification_enabled: bool = True
    categories_to_oversample: list = field(default_factory=lambda: [
        "bit_manipulation",
        "symbol_transform",
        "cipher",
    ])


@dataclass
class LocalEvalConfig:
    """Local evaluation harness settings."""
    num_eval_seeds: int = 3
    replicate_metric: bool = True
    track_per_category: bool = True


@dataclass
class Config:
    """Master configuration for the LoRA training pipeline."""
    eval: EvalConfig = field(default_factory=EvalConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    local_eval: LocalEvalConfig = field(default_factory=LocalEvalConfig)

    def load_yaml(self, path: str) -> None:
        """Override defaults from a YAML config file."""
        try:
            with open(path, 'r') as f:
                overrides = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {path}")
            raise
        if overrides:
            self._apply_overrides(overrides)

    def _apply_overrides(self, overrides: dict) -> None:
        """Recursively apply overrides from a nested dict."""
        for key, value in overrides.items():
            if isinstance(value, dict):
                sub_config = getattr(self, key, None)
                if sub_config is not None:
                    for sub_key, sub_value in value.items():
                        if hasattr(sub_config, sub_key):
                            setattr(sub_config, sub_key, sub_value)
            elif hasattr(self, key):
                setattr(self, key, value)

    def save_yaml(self, path: str) -> None:
        """Save current configuration to YAML for reproducibility."""
        config_dict = {}
        for fi in dataclasses.fields(self):
            sub = getattr(self, fi.name)
            if dataclasses.is_dataclass(sub):
                config_dict[fi.name] = dataclasses.asdict(sub)
            else:
                config_dict[fi.name] = sub
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def summary(self) -> str:
        return (
            "=== Nemotron LoRA Training Config ===\n"
            f"Model: {self.eval.model_id}\n"
            f"LoRA rank={self.lora.rank}, alpha={self.lora.alpha}\n"
            f"Training: lr={self.training.learning_rate}, epochs={self.training.num_epochs}, "
            f"batch={self.training.batch_size}x{self.training.gradient_accumulation_steps}\n"
            f"Eval: temp={self.eval.temperature}, top_p={self.eval.top_p}, "
            f"max_tokens={self.eval.max_tokens}\n"
            f"Oversample: {self.data.categories_to_oversample}"
        )
