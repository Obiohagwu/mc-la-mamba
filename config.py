"""
Experiment configuration for MC-Mamba: Memory Caching for Mamba SSMs.

Extends the ssm-mamba config with MCConfig for segment caching and GRM gating.

Codec: DAC 44.1kHz, 9 codebooks @ 86Hz frame rate
Models: ~128-140M params
  1. Baseline mamba1 (control)
  2. MC-Mamba (memory-cached Mamba with GRM gating)
  3. Ablations: segment size sweep, partial MC, uniform gating
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path


@dataclass
class CodecConfig:
    """DAC codec configuration."""
    model_type: str = "44khz"
    sample_rate: int = 44100
    n_codebooks: int = 9
    frame_rate: int = 86
    codebook_size: int = 1024
    pad_token: int = 1024
    bos_token: int = 1025
    eos_token: int = 1026
    vocab_size: int = 1027


@dataclass
class MCConfig:
    """Memory Caching configuration for MC-Mamba and MC-Linear-Attention."""
    # Segment size S: cache boundary activations every S timesteps
    segment_size: int = 256
    # Scaling factor for cached retrieval residual
    retrieval_scale: float = 1.0
    # Maximum number of cached segment entries (oldest evicted first)
    max_cache_entries: int = 64


@dataclass
class LinearAttentionConfig:
    """Linear Attention configuration."""
    # Number of attention heads (d_k = d_v = d_model // n_heads)
    n_heads: int = 12
    # Use DeltaNet variant: S_t = S_{t-1} + beta_t * (v_t - S^T k_t) outer k_t^T
    use_deltanet: bool = False
    # Feature map type (currently only elu_plus_1 supported)
    feature_map: str = "elu_plus_1"


@dataclass
class ModelConfig:
    """Model hyperparameters."""
    arch: Literal[
        "transformer",
        "mamba1",
        "mamba2",
        "hybrid_1_7",
        "hybrid_1_3",
        "mc_mamba",
        "linear_attention",
        "mc_linear_attention",
    ] = "mc_mamba"

    # Dimensions
    d_model: int = 768
    n_layers: int = 20
    n_heads: int = 12
    d_ff: int = 3072

    # Mamba-specific
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    dt_rank: str = "auto"

    # RVQ token handling
    n_codebooks: int = 9
    codebook_size: int = 1024
    vocab_size: int = 1027
    use_delay_pattern: bool = True

    # Regularization
    dropout: float = 0.2
    layer_norm_eps: float = 1e-5

    # Sequence length
    max_seq_len: int = 2048

    # MC config (used when arch == "mc_mamba" or "mc_linear_attention")
    mc: MCConfig = field(default_factory=MCConfig)

    # Linear attention config (used when arch == "linear_attention" or "mc_linear_attention")
    la: LinearAttentionConfig = field(default_factory=LinearAttentionConfig)

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def computed_dt_rank(self) -> int:
        if self.dt_rank == "auto":
            import math
            return math.ceil(self.d_model / 16)
        return int(self.dt_rank)


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # Data
    data_dir: str = "./data/dac_tokens"
    dataset_name: str = "fma_large"
    max_duration_sec: float = 24.0
    num_workers: int = 4

    # Training
    batch_size: int = 16
    grad_accum_steps: int = 4
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100_000
    eval_every: int = 2000
    save_every: int = 5000
    log_every: int = 100

    # Optimizer
    optimizer: str = "adamw"
    betas: tuple = (0.9, 0.95)
    max_grad_norm: float = 1.0

    # LR schedule
    lr_schedule: str = "cosine"
    min_lr: float = 1e-5

    # Device
    device: str = "cuda"
    mixed_precision: bool = True
    compile_model: bool = False

    # Checkpointing
    output_dir: str = "./runs"
    run_name: Optional[str] = None

    # Seed
    seed: int = 42

    # EMA
    ema_decay: float = 0.999
    use_ema: bool = True

    # Codebook loss weighting: weight for each codebook level
    # Codebook 0 (coarse) gets highest weight, finer codebooks less
    codebook_loss_weights: Optional[tuple] = None  # None = uniform


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    fad_model: str = "vggish"
    n_gen_samples: int = 256
    gen_duration_sec: float = 10.0

    temperature: float = 0.9
    top_k: int = 250
    top_p: float = 0.0

    coherence_window_sec: float = 4.0
    coherence_hop_sec: float = 2.0


@dataclass
class ExperimentConfig:
    """Full experiment config."""
    codec: CodecConfig = field(default_factory=CodecConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
