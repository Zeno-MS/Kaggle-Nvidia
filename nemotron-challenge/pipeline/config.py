"""
Central configuration for the Nemotron Challenge pipeline.

This module defines all tunable parameters in one place. Values marked [PHASE 0]
are placeholders that MUST be updated based on Phase 0 findings. Values marked
[TUNABLE] are hyperparameters to optimize during Phase 2.

Usage:
    from pipeline.config import Config
    cfg = Config()
    cfg.load_yaml("configs/default.yaml")  # Override defaults from file
"""

from dataclasses import dataclass, field
from typing import Optional
import yaml
from pathlib import Path


@dataclass
class BenchmarkConfig:
    """What we know about the benchmark. Update after Phase 0."""
    puzzle_format: str = "unknown"         # [PHASE 0] "grid", "text", "symbolic", etc.
    has_training_examples: bool = True      # [PHASE 0] Do puzzles come with few-shot examples?
    num_examples_per_puzzle: int = 3        # [PHASE 0] How many training examples per puzzle?
    deterministic_answers: bool = True      # [PHASE 0] Exactly one correct answer per puzzle?
    input_representation: str = "unknown"   # [PHASE 0] How inputs are encoded
    output_representation: str = "unknown"  # [PHASE 0] How outputs are encoded


@dataclass
class JudgeConfig:
    """Judge model preferences. Update after Phase 0 characterization."""
    model_id: str = "nemotron-3-nano-30b-a3b-bf16"
    
    # HelpSteer2 weights (our STARTING hypothesis — verify in Phase 0)
    weight_correctness: float = 0.80
    weight_helpfulness: float = 0.65
    weight_complexity: float = 0.55
    weight_coherence: float = 0.45
    weight_verbosity: float = -0.40       # Negative = penalizes verbosity
    
    # Evaluation parameters (check Kaggle evaluation page)
    eval_temperature: float = 1.0          # [PHASE 0] From Kaggle eval page
    eval_max_tokens: int = 4096            # [PHASE 0] From Kaggle eval page
    eval_top_p: float = 0.95              # [PHASE 0] From Kaggle eval page
    
    # Judge characterization results (fill in after experiments)
    prefers_concise: Optional[bool] = None        # [PHASE 0] Result of length experiment
    trace_quality_impact: Optional[float] = None  # [PHASE 0] Score delta from trace quality
    correctness_is_dominant: Optional[bool] = None # [PHASE 0] Is correctness > trace quality?


@dataclass  
class SymbolicSolverConfig:
    """Track A: Symbolic solver parameters."""
    enabled: bool = True
    timeout_per_puzzle_sec: float = 30.0   # [TUNABLE] Max time for symbolic solving
    max_program_depth: int = 5             # [TUNABLE] Max DSL program depth
    max_candidates: int = 100              # [TUNABLE] Max candidate programs to evaluate
    
    # Solver portfolio (enable/disable individual solvers)
    use_dsl_enumeration: bool = True
    use_llm_program_synthesis: bool = True
    use_ilp: bool = False                  # Requires Prolog — enable if available
    
    # LLM-guided synthesis parameters
    synthesis_model: str = "nemotron-nano"  # Model for generating candidate programs
    synthesis_temperature: float = 0.8      # [TUNABLE]
    synthesis_max_attempts: int = 10        # [TUNABLE] Max LLM calls per puzzle


@dataclass
class TraceConfig:
    """Track B: Trace generation parameters."""
    enabled: bool = True
    
    # Trace template version (see STRATEGY.md § Trace Architecture)
    template_version: str = "v1"
    
    # Trace generation parameters
    generation_temperature: float = 0.7    # [TUNABLE] Lower = more deterministic traces
    max_trace_tokens: int = 512            # [TUNABLE] Keep traces concise (verbosity penalty)
    num_trace_candidates: int = 3          # [TUNABLE] Generate K traces, select best
    
    # Trace structure constraints (from cognitive trace design)
    max_derivation_steps: int = 7          # [TUNABLE] Based on judge preference data
    min_derivation_steps: int = 3          # [TUNABLE]
    include_verification: bool = True       # Include verification step in trace
    include_strategy_statement: bool = True  # Include strategy identification step
    
    # Style matching
    match_nemotron_style: bool = True      # Try to match judge's own output style
    use_think_tags: bool = True            # Use Nemotron's <think> tag format


@dataclass
class NeuralFallbackConfig:
    """Track C: Neural reasoning fallback parameters."""
    enabled: bool = True
    
    # Self-consistency parameters
    num_samples: int = 8                   # [TUNABLE] Samples per puzzle for voting
    voting_method: str = "majority"        # "majority", "weighted", "ranked"
    sampling_temperature: float = 0.8      # [TUNABLE] Diversity for self-consistency
    
    # Reasoning mode
    reasoning_on: bool = True              # Enable Nemotron reasoning traces
    reasoning_budget: int = 2048           # [TUNABLE] Max thinking tokens
    
    # Few-shot configuration
    num_few_shot: int = 3                  # [TUNABLE] Number of in-context examples
    few_shot_source: str = "symbolic"      # "symbolic" (from solved puzzles) or "training"
    
    # Model configuration
    model_id: str = "nemotron-3-nano-30b-a3b-bf16"
    quantization: str = "fp8"              # "bf16", "fp8", "int4"


@dataclass
class OrchestratorConfig:
    """Pipeline orchestrator parameters."""
    # Stage execution
    symbolic_first: bool = True            # Try symbolic before neural?
    skip_trace_if_symbolic_fails: bool = False  # If symbolic fails, trace wrap neural output?
    
    # Adaptive compute allocation
    adaptive_sampling: bool = True         # Allocate more compute to harder puzzles?
    easy_puzzle_samples: int = 2           # [TUNABLE] Samples for easy puzzles
    hard_puzzle_samples: int = 16          # [TUNABLE] Samples for hard puzzles
    difficulty_threshold: float = 0.5      # [TUNABLE] Confidence threshold for easy/hard
    
    # Time management
    total_time_budget_sec: float = 18000   # [PHASE 0] Total submission time budget (5hrs default)
    time_per_puzzle_sec: float = 60.0      # [PHASE 0] Derived from total / num_puzzles
    
    # Logging
    verbose: bool = True
    log_per_puzzle: bool = True


@dataclass
class Config:
    """Master configuration. Single source of truth for all parameters."""
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    symbolic: SymbolicSolverConfig = field(default_factory=SymbolicSolverConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    neural: NeuralFallbackConfig = field(default_factory=NeuralFallbackConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    
    def load_yaml(self, path: str):
        """Override defaults from a YAML config file."""
        with open(path, 'r') as f:
            overrides = yaml.safe_load(f)
        if overrides:
            self._apply_overrides(overrides)
    
    def _apply_overrides(self, overrides: dict, prefix: str = ""):
        """Recursively apply overrides from a flat or nested dict."""
        for key, value in overrides.items():
            if isinstance(value, dict):
                sub_config = getattr(self, key, None)
                if sub_config is not None:
                    for sub_key, sub_value in value.items():
                        if hasattr(sub_config, sub_key):
                            setattr(sub_config, sub_key, sub_value)
            elif hasattr(self, key):
                setattr(self, key, value)
    
    def save_yaml(self, path: str):
        """Save current configuration to YAML for reproducibility."""
        import dataclasses
        config_dict = {}
        for field_info in dataclasses.fields(self):
            sub_config = getattr(self, field_info.name)
            if dataclasses.is_dataclass(sub_config):
                config_dict[field_info.name] = dataclasses.asdict(sub_config)
            else:
                config_dict[field_info.name] = sub_config
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def summary(self) -> str:
        """Print a human-readable summary of key configuration values."""
        lines = [
            "=== Pipeline Configuration Summary ===",
            f"Benchmark format: {self.benchmark.puzzle_format}",
            f"Judge: {self.judge.model_id}",
            f"Symbolic solver: {'ON' if self.symbolic.enabled else 'OFF'}",
            f"  Timeout per puzzle: {self.symbolic.timeout_per_puzzle_sec}s",
            f"Trace generation: {'ON' if self.trace.enabled else 'OFF'}",
            f"  Max trace tokens: {self.trace.max_trace_tokens}",
            f"  Template version: {self.trace.template_version}",
            f"Neural fallback: {'ON' if self.neural.enabled else 'OFF'}",
            f"  Samples per puzzle: {self.neural.num_samples}",
            f"  Voting method: {self.neural.voting_method}",
            f"Pipeline order: {'Symbolic → Trace → Judge → Fallback' if self.orchestrator.symbolic_first else 'Neural → Trace → Judge'}",
            f"Time budget: {self.orchestrator.total_time_budget_sec}s total",
        ]
        return "\n".join(lines)
