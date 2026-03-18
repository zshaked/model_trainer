"""
Shared configuration for all model variants.
Sizes calibrated for free-tier GPUs (T4 16GB, P100 16GB).
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Architecture config — same for all models to ensure fair comparison."""
    vocab_size: int = 50257          # GPT-2 tokenizer
    max_seq_len: int = 512
    dim: int = 512
    num_layers: int = 8
    num_heads: int = 8               # transformer only
    ff_mult: float = 2.667           # ff_dim = dim * ff_mult (SwiGLU effective 4x)
    dropout: float = 0.0
    tie_weights: bool = True


@dataclass
class TrainConfig:
    """Training config — identical across all experiments."""
    # Data
    dataset: str = "openwebtext"     # or "roneneldan/TinyStories"
    tokenizer: str = "gpt2"
    num_workers: int = 4

    # Batch
    batch_size: int = 32             # per-device
    grad_accum_steps: int = 4        # effective batch = 32 * 4 = 128
    max_seq_len: int = 512

    # Optimization
    lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0

    # Schedule
    warmup_tokens: int = 10_000_000       # 10M tokens warmup
    total_tokens: int = 1_000_000_000     # 1B tokens total (adjust to time budget)

    # Logging
    log_every_tokens: int = 1_000_000     # log loss every 1M tokens
    eval_every_tokens: int = 10_000_000   # eval every 10M tokens
    eval_tokens: int = 500_000            # tokens used for each eval
    save_every_tokens: int = 50_000_000   # checkpoint every 50M tokens

    # Paths
    output_dir: str = "runs"
    seed: int = 42


# Preset sizes for different GPU budgets
SMALL = ModelConfig(dim=384, num_layers=6, num_heads=6)     # ~15M params
MEDIUM = ModelConfig(dim=512, num_layers=8, num_heads=8)    # ~45M params
LARGE = ModelConfig(dim=768, num_layers=12, num_heads=12)   # ~125M params


def get_config(size: str = "medium") -> ModelConfig:
    return {"small": SMALL, "medium": MEDIUM, "large": LARGE}[size]
