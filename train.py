"""
Training script for MC-Mamba: Memory Caching for Mamba SSMs.

Usage:
    python train.py --arch mc_mamba                              # MC-Mamba (novel)
    python train.py --arch mc_mamba --segment_size 128           # Ablation: smaller segments
    python train.py --arch mamba1                                # Baseline (control)
    python train.py --arch transformer                           # Transformer baseline
    python train.py --arch hybrid_1_7                            # Hybrid baseline
    python train.py --arch hybrid_1_3                            # Hybrid baseline
    python train.py --arch mc_linear_attention --preset speech_960h_pilot
    python train.py --arch mc_linear_attention --preset speech_scaled

Features:
  - EMA (Exponential Moving Average) model for evaluation
  - Codebook loss weighting (coarse codebooks weighted higher)
  - Weight tying (first codebook embedding ↔ first output head)
  - Scaled initialization for deep networks
  - MC-specific logging: GRM attention entropy, cache utilization
"""

import os
import sys
import time
import math
import json
import copy
import argparse
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import ModelConfig, TrainConfig, CodecConfig, ExperimentConfig, MCConfig, LinearAttentionConfig
from src.models.factory import build_model
from src.data.tokenizer import PreTokenizedDataset, collate_fn


SPEECH_PRESETS = {
    "speech_960h_pilot": {
        "train": {
            "dataset_name": "librispeech_960h_encodec24",
            "data_dir": "./data/speech/librispeech_960h_encodec24",
            "output_dir": "./runs/speech",
            "batch_size": 12,
            "grad_accum_steps": 4,
            "lr": 3e-4,
            "warmup_steps": 1000,
            "max_steps": 30_000,
            "eval_every": 1000,
            "save_every": 2000,
            "log_every": 50,
            "max_duration_sec": 20.0,
        },
        "model": {
            "max_seq_len": 2048,
            "dropout": 0.1,
        },
        # Fallback defaults if codec_meta.json is absent.
        "codec": {
            "sample_rate": 24000,
            "n_codebooks": 8,
            "codebook_size": 1024,
            "pad_token": 1024,
            "bos_token": 1025,
            "eos_token": 1026,
            "vocab_size": 1027,
        },
    },
    "speech_scaled": {
        "train": {
            "dataset_name": "speech_scaled",
            "data_dir": "./data/speech/scaled_encodec24",
            "output_dir": "./runs/speech",
            "batch_size": 16,
            "grad_accum_steps": 8,
            "lr": 2e-4,
            "warmup_steps": 4000,
            "max_steps": 200_000,
            "eval_every": 2000,
            "save_every": 5000,
            "log_every": 100,
            "max_duration_sec": 20.0,
        },
        "model": {
            "max_seq_len": 2048,
            "dropout": 0.1,
        },
        "codec": {
            "sample_rate": 24000,
            "n_codebooks": 8,
            "codebook_size": 1024,
            "pad_token": 1024,
            "bos_token": 1025,
            "eos_token": 1026,
            "vocab_size": 1027,
        },
    },
}


def apply_codec_settings(config: ExperimentConfig, codec_settings: dict):
    """Apply codec settings to both codec and model token geometry."""
    codec_keys = {
        "sample_rate",
        "n_codebooks",
        "codebook_size",
        "pad_token",
        "bos_token",
        "eos_token",
        "vocab_size",
    }
    for key in codec_keys:
        if key in codec_settings:
            setattr(config.codec, key, codec_settings[key])

    # Keep model tokenization geometry aligned with codec settings.
    config.model.n_codebooks = config.codec.n_codebooks
    config.model.codebook_size = config.codec.codebook_size
    config.model.vocab_size = config.codec.vocab_size


def apply_preset(config: ExperimentConfig, preset_name: str):
    """Apply a named experiment preset to model/train config."""
    preset = SPEECH_PRESETS[preset_name]
    for key, value in preset.get("train", {}).items():
        setattr(config.train, key, value)
    for key, value in preset.get("model", {}).items():
        setattr(config.model, key, value)
    if "codec" in preset:
        apply_codec_settings(config, preset["codec"])


def maybe_apply_codec_metadata(config: ExperimentConfig, codec_meta_path: Path) -> bool:
    """Load codec metadata from disk and apply it if present."""
    if not codec_meta_path.exists():
        return False

    with open(codec_meta_path) as f:
        meta = json.load(f)
    apply_codec_settings(config, meta)
    print(f"Loaded codec metadata from {codec_meta_path}: {meta}")
    return True


def get_lr(step: int, config: TrainConfig) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < config.warmup_steps:
        return config.lr * step / config.warmup_steps
    if step >= config.max_steps:
        return config.min_lr

    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.lr - config.min_lr)


def count_parameters(model: nn.Module) -> dict:
    """Count total and per-component parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embed = sum(p.numel() for p in model.embed.parameters())
    output = sum(p.numel() for p in model.output.parameters())
    backbone = total - embed - output

    result = {
        "total": total,
        "trainable": trainable,
        "embed": embed,
        "backbone": backbone,
        "output_head": output,
    }

    # MC-specific: count GRM parameters
    if hasattr(model, 'blocks') and len(model.blocks) > 0:
        first_block = model.blocks[0]
        if hasattr(first_block, 'grm'):
            grm_params = sum(
                sum(p.numel() for p in block.grm.parameters())
                for block in model.blocks
            )
            result["mc_grm"] = grm_params
            result["mc_overhead_pct"] = 100.0 * grm_params / total

    return result


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {name: p.data.clone() for name, p in model.named_parameters()}

    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module):
        """Swap model params with EMA params (for eval)."""
        self.backup = {name: p.data.clone() for name, p in model.named_parameters()}
        for name, p in model.named_parameters():
            p.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original model params after eval."""
        for name, p in model.named_parameters():
            p.data.copy_(self.backup[name])
        self.backup = None


class Trainer:
    def __init__(self, config: ExperimentConfig, arch: str):
        self.config = config
        self.arch = arch

        config.model.arch = arch

        # Setup device
        self.device = torch.device(config.train.device if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if config.train.mixed_precision and self.device.type == "cuda" else torch.float32
        self.ctx = torch.amp.autocast(device_type=self.device.type, dtype=self.dtype) if self.dtype != torch.float32 else nullcontext()
        self.scaler = torch.amp.GradScaler(enabled=(self.dtype == torch.float16))

        # Seed
        torch.manual_seed(config.train.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.train.seed)

        # Build model
        self.model = build_model(config.model).to(self.device)
        self.param_counts = count_parameters(self.model)

        # Compile if requested
        if config.train.compile_model and hasattr(torch, "compile"):
            print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        # EMA
        self.ema = EMA(self.model, decay=config.train.ema_decay) if config.train.use_ema else None

        # Codebook loss weights
        if config.train.codebook_loss_weights is not None:
            self.codebook_weights = torch.tensor(
                config.train.codebook_loss_weights, dtype=torch.float32, device=self.device
            )
        else:
            self.codebook_weights = None

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.train.lr,
            betas=config.train.betas,
            weight_decay=config.train.weight_decay,
        )

        # Output directory
        run_name = config.train.run_name or f"{arch}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = Path(config.train.output_dir) / run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.log_file = open(self.output_dir / "train.log", "w")
        self.metrics_log = []

        # Save config (copy dicts to avoid mutating original config)
        config_dict = {
            "arch": arch,
            "params": self.param_counts,
            "codec": dict(vars(config.codec)),
            "model": dict(vars(config.model)),
            "train": dict(vars(config.train)),
        }
        # Serialize nested configs properly
        if hasattr(config.model, 'mc'):
            config_dict["model"]["mc"] = dict(vars(config.model.mc))
        if hasattr(config.model, 'la'):
            config_dict["model"]["la"] = dict(vars(config.model.la))

        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

    def _compute_loss(self, codes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute loss, handling codebook_weights for any model architecture."""
        pad_token = self.config.codec.pad_token

        if self.codebook_weights is not None:
            # Weighted codebook loss — works for any model
            logits = self.model(codes, mask)  # (B, K, T, V)
            B, K, T, V = logits.shape
            total_loss = 0.0
            for k in range(K):
                k_logits = logits[:, k, :-1, :].contiguous().view(-1, V)
                k_targets = codes[:, k, 1:].contiguous().view(-1)
                k_loss = F.cross_entropy(k_logits, k_targets, ignore_index=pad_token, label_smoothing=0.1)
                total_loss = total_loss + self.codebook_weights[k] * k_loss
            return total_loss / self.codebook_weights.sum()
        else:
            return self.model.compute_loss(codes, mask, pad_token=pad_token)

    def log(self, msg: str):
        print(msg)
        self.log_file.write(msg + "\n")
        self.log_file.flush()

    def build_dataloader(self, split: str = "train") -> DataLoader:
        """Build dataloader for pre-tokenized data."""
        data_dir = Path(self.config.train.data_dir) / split
        dataset = PreTokenizedDataset(
            data_dir=str(data_dir),
            max_seq_len=self.config.model.max_seq_len,
            n_codebooks=self.config.codec.n_codebooks,
            pad_token=self.config.codec.pad_token,
            use_delay_pattern=self.config.model.use_delay_pattern,
        )
        self.log(f"Loaded {split} dataset: {len(dataset)} samples from {data_dir}")

        return DataLoader(
            dataset,
            batch_size=self.config.train.batch_size,
            shuffle=(split == "train"),
            num_workers=self.config.train.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=(split == "train"),
        )

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, max_batches: int = 50) -> dict:
        """Run evaluation on validation set."""
        # Use EMA params for evaluation if available
        if self.ema is not None:
            self.ema.apply(self.model)

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        n_batches = 0

        for batch in val_loader:
            if n_batches >= max_batches:
                break

            codes = batch["codes"].to(self.device)
            mask = batch["mask"].to(self.device)

            with self.ctx:
                loss = self._compute_loss(codes, mask)

            total_loss += loss.item() * codes.shape[0]
            total_tokens += codes.shape[0]
            n_batches += 1

        self.model.train()

        # Restore original params after EMA eval
        if self.ema is not None:
            self.ema.restore(self.model)

        avg_loss = total_loss / max(total_tokens, 1)
        return {
            "val_loss": avg_loss,
            "val_perplexity": math.exp(min(avg_loss, 20)),
        }

    def train(self):
        """Main training loop."""
        tc = self.config.train

        self.log(f"Architecture: {self.arch}")
        self.log(
            f"Codec: sr={self.config.codec.sample_rate}, "
            f"n_codebooks={self.config.codec.n_codebooks}, "
            f"codebook_size={self.config.codec.codebook_size}, "
            f"vocab={self.config.codec.vocab_size}"
        )
        self.log(f"Parameters: {self.param_counts}")
        self.log(f"Device: {self.device}, dtype: {self.dtype}")
        self.log(f"Effective batch size: {tc.batch_size * tc.grad_accum_steps}")
        if self.arch in ("mc_mamba", "mc_linear_attention"):
            mc = self.config.model.mc
            self.log(f"MC Config: segment_size={mc.segment_size}, "
                     f"retrieval_scale={mc.retrieval_scale}, "
                     f"max_cache_entries={mc.max_cache_entries}")
        if self.arch in ("linear_attention", "mc_linear_attention"):
            la = self.config.model.la
            self.log(f"LA Config: n_heads={la.n_heads}, "
                     f"use_deltanet={la.use_deltanet}, "
                     f"feature_map={la.feature_map}")
        if self.ema is not None:
            self.log(f"EMA decay: {tc.ema_decay}")
        self.log("-" * 60)

        # Build dataloaders
        train_loader = self.build_dataloader("train")
        val_loader = self.build_dataloader("val")

        # Training state
        step = 0
        best_val_loss = float("inf")
        train_iter = iter(train_loader)

        self.model.train()
        t0 = time.time()

        while step < tc.max_steps:
            self.optimizer.zero_grad()
            accum_loss = 0.0

            for micro_step in range(tc.grad_accum_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                codes = batch["codes"].to(self.device)
                mask = batch["mask"].to(self.device)

                with self.ctx:
                    loss = self._compute_loss(codes, mask)
                    loss = loss / tc.grad_accum_steps

                self.scaler.scale(loss).backward()
                accum_loss += loss.item()

            # Gradient clipping
            if tc.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), tc.max_grad_norm
                )
            else:
                grad_norm = 0.0

            # Update LR
            lr = get_lr(step, tc)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # EMA update
            if self.ema is not None:
                self.ema.update(self.model)

            step += 1

            # Logging
            if step % tc.log_every == 0:
                dt = time.time() - t0
                tokens_per_sec = (
                    tc.batch_size * tc.grad_accum_steps
                    * self.config.model.n_codebooks
                    * self.config.model.max_seq_len
                    * tc.log_every / dt
                )
                metrics = {
                    "step": step,
                    "train_loss": accum_loss,
                    "lr": lr,
                    "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "tokens_per_sec": tokens_per_sec,
                    "elapsed_sec": time.time() - t0,
                }

                # MC-specific logging
                if hasattr(self.model, 'get_mc_stats'):
                    mc_stats = self.model.get_mc_stats()
                    metrics.update(mc_stats)

                self.metrics_log.append(metrics)

                log_msg = (
                    f"step {step:6d} | loss {accum_loss:.4f} | "
                    f"lr {lr:.2e} | grad_norm {metrics['grad_norm']:.2f} | "
                    f"tok/s {tokens_per_sec:.0f}"
                )

                # Add MC stats to log line
                if "avg_grm_entropy" in metrics:
                    log_msg += f" | grm_ent {metrics['avg_grm_entropy']:.3f}"
                if "avg_cache_entries" in metrics:
                    log_msg += f" | cache {metrics['avg_cache_entries']:.0f}"

                self.log(log_msg)
                t0 = time.time()

            # Evaluation
            if step % tc.eval_every == 0:
                eval_metrics = self.evaluate(val_loader)
                ema_tag = " (EMA)" if self.ema is not None else ""
                self.log(
                    f"  EVAL step {step}{ema_tag}: val_loss={eval_metrics['val_loss']:.4f} "
                    f"ppl={eval_metrics['val_perplexity']:.2f}"
                )

                if eval_metrics["val_loss"] < best_val_loss:
                    best_val_loss = eval_metrics["val_loss"]
                    self.save_checkpoint(step, is_best=True)

            # Save checkpoint
            if step % tc.save_every == 0:
                self.save_checkpoint(step)

        # Final save
        self.save_checkpoint(step, is_best=False)
        self.log(f"Training complete. Best val loss: {best_val_loss:.4f}")

        # Save metrics
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(self.metrics_log, f, indent=2)

        self.log_file.close()

    def save_checkpoint(self, step: int, is_best: bool = False):
        """Save model checkpoint."""
        ckpt = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": vars(self.config.model),
        }

        # Save EMA state if available
        if self.ema is not None:
            ckpt["ema_shadow"] = self.ema.shadow

        path = self.output_dir / f"checkpoint_{step}.pt"
        torch.save(ckpt, path)

        if is_best:
            best_path = self.output_dir / "best.pt"
            torch.save(ckpt, best_path)
            self.log(f"  Saved best checkpoint at step {step}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train MC-Mamba / baselines over RVQ tokens")
    parser.add_argument(
        "--preset",
        type=str,
        choices=sorted(SPEECH_PRESETS.keys()),
        default=None,
        help="Optional speech preset (dataset + training hyperparameters)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=["transformer", "mamba1", "mamba2", "hybrid_1_7", "hybrid_1_3", "mc_mamba",
                 "linear_attention", "mc_linear_attention"],
        help="Model architecture to train",
    )
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--codec_meta",
        type=str,
        default=None,
        help="Path to codec metadata JSON (default: <data_dir>/codec_meta.json)",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    # MC-specific arguments
    parser.add_argument("--segment_size", type=int, default=256,
                        help="MC segment size S (cache boundary every S timesteps)")
    parser.add_argument("--retrieval_scale", type=float, default=1.0,
                        help="Scale factor for GRM retrieval residual")
    parser.add_argument("--max_cache_entries", type=int, default=64,
                        help="Maximum number of cached segments per layer")
    parser.add_argument("--no_ema", action="store_true", help="Disable EMA")
    parser.add_argument("--codebook_weights", type=str, default=None,
                        help="Comma-separated codebook loss weights (e.g. '2,1.5,1,1,1,0.8,0.8,0.5,0.5')")

    # Linear attention arguments
    parser.add_argument("--use_deltanet", action="store_true",
                        help="Use DeltaNet variant for linear attention (error-correction update)")
    parser.add_argument("--la_n_heads", type=int, default=None,
                        help="Number of heads for linear attention (default: same as n_heads)")

    return parser.parse_args()


def main():
    args = parse_args()

    config = ExperimentConfig()

    # Apply preset first, then let explicit CLI args override.
    if args.preset:
        apply_preset(config, args.preset)

    config.model.arch = args.arch
    config.train.seed = args.seed
    config.train.device = args.device

    if args.data_dir:
        config.train.data_dir = args.data_dir
    if args.output_dir:
        config.train.output_dir = args.output_dir
    if args.run_name:
        config.train.run_name = args.run_name

    # Align codec/model settings from preprocessed dataset metadata when available.
    codec_meta_path = Path(args.codec_meta) if args.codec_meta else Path(config.train.data_dir) / "codec_meta.json"
    maybe_apply_codec_metadata(config, codec_meta_path)

    # Override optional args
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.max_steps:
        config.train.max_steps = args.max_steps
    if args.lr:
        config.train.lr = args.lr
    if args.d_model:
        config.model.d_model = args.d_model
        config.model.d_ff = args.d_model * 4
    if args.n_layers:
        config.model.n_layers = args.n_layers

    # MC config
    config.model.mc.segment_size = args.segment_size
    config.model.mc.retrieval_scale = args.retrieval_scale
    config.model.mc.max_cache_entries = args.max_cache_entries

    # Linear attention config
    if args.use_deltanet:
        config.model.la.use_deltanet = True
    if args.la_n_heads:
        config.model.la.n_heads = args.la_n_heads

    # EMA
    if args.no_ema:
        config.train.use_ema = False

    # Codebook weights
    if args.codebook_weights:
        weights = tuple(float(w) for w in args.codebook_weights.split(","))
        assert len(weights) == config.model.n_codebooks, \
            f"Expected {config.model.n_codebooks} weights, got {len(weights)}"
        config.train.codebook_loss_weights = weights

    trainer = Trainer(config, args.arch)
    trainer.train()


if __name__ == "__main__":
    main()
