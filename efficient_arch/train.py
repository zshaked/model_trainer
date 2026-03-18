"""
Shared training loop — identical for all model variants.

The only variable is the model architecture. Everything else
(data, optimizer, schedule, logging, evaluation) is the same.

Usage:
    python train.py --model transformer --size medium
    python train.py --model ssm --size medium
    python train.py --model predictive_ssm --size medium --pred-loss-weight 0.05
"""
import os
import sys
import json
import math
import time
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from configs.base import ModelConfig, TrainConfig, get_config
from models import build_model
from data.pipeline import get_dataloader, get_eval_batches


def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU found. Training will be very slow.")
    return device


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_lr(step, warmup_steps, total_steps, max_lr, min_lr):
    """Linear warmup + cosine decay."""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, eval_batches, device):
    """Evaluate on pre-loaded batches. Returns mean loss."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in eval_batches:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    model.train()
    return total_loss / max(total_tokens, 1)


def train(args):
    # Setup
    device = setup_device()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Config
    model_config = get_config(args.size)
    train_config = TrainConfig(
        dataset=args.dataset,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        total_tokens=args.total_tokens,
        seed=args.seed,
    )

    # Output directory
    run_name = f"{args.model}_{args.size}_seed{args.seed}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build model
    model = build_model(args.model, model_config).to(device)
    total_params, trainable_params = count_params(model)
    print(f"Model: {args.model} ({args.size})")
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Save config
    config_dict = {
        "model": args.model, "size": args.size,
        "total_params": total_params,
        "model_config": vars(model_config),
        "train_config": vars(train_config),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=train_config.weight_decay,
    )

    # Calculate steps from tokens
    tokens_per_step = (train_config.batch_size * train_config.grad_accum_steps
                       * model_config.max_seq_len)
    total_steps = train_config.total_tokens // tokens_per_step
    warmup_steps = train_config.warmup_tokens // tokens_per_step

    print(f"Tokens per step: {tokens_per_step:,}")
    print(f"Total steps: {total_steps:,}")
    print(f"Warmup steps: {warmup_steps:,}")

    # Data
    print(f"Dataset: {train_config.dataset}")
    train_loader = get_dataloader(train_config, split="train")
    train_iter = iter(train_loader)

    # Pre-load eval data
    print("Loading eval data...")
    eval_batches = get_eval_batches(train_config, num_tokens=train_config.eval_tokens)
    print(f"Eval batches: {len(eval_batches)}")

    # Learning curve log
    curve_path = run_dir / "learning_curve.jsonl"
    curve_file = open(curve_path, "w")

    def log_point(tokens_seen, train_loss, val_loss=None, extra=None):
        point = {
            "tokens": tokens_seen,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "wall_time": time.time() - start_time,
        }
        if extra:
            point.update(extra)
        curve_file.write(json.dumps(point) + "\n")
        curve_file.flush()

    # Training loop
    model.train()
    step = 0
    tokens_seen = 0
    running_loss = 0.0
    start_time = time.time()

    print(f"\nTraining {args.model} for {train_config.total_tokens:,} tokens...")
    print("=" * 70)

    while tokens_seen < train_config.total_tokens:
        # LR schedule
        lr = get_lr(step, warmup_steps, total_steps, train_config.lr, train_config.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation
        optimizer.zero_grad()
        accum_loss = 0.0

        for micro_step in range(train_config.grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            # Predictive coding auxiliary loss
            if args.model == "predictive_ssm" and args.pred_loss_weight > 0:
                pred_loss = model.get_prediction_loss()
                loss = loss + args.pred_loss_weight * pred_loss

            loss_scaled = loss / train_config.grad_accum_steps
            loss_scaled.backward()
            accum_loss += loss.item() / train_config.grad_accum_steps

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_config.max_grad_norm
        )
        optimizer.step()

        # Bookkeeping
        step += 1
        tokens_seen += tokens_per_step
        running_loss += accum_loss

        # Log training loss
        if tokens_seen % train_config.log_every_tokens < tokens_per_step:
            avg_loss = running_loss / max(
                train_config.log_every_tokens // tokens_per_step, 1
            )
            elapsed = time.time() - start_time
            tok_per_sec = tokens_seen / elapsed
            extra = {}
            if args.model == "predictive_ssm" and hasattr(model, 'get_error_stats'):
                extra["pred_errors"] = model.get_error_stats()

            log_point(tokens_seen, avg_loss, extra=extra)

            print(
                f"step={step:>6d} | tokens={tokens_seen:>12,} | "
                f"loss={avg_loss:.4f} | lr={lr:.2e} | "
                f"tok/s={tok_per_sec:,.0f} | "
                f"elapsed={elapsed:.0f}s"
            )
            running_loss = 0.0

        # Evaluate
        if tokens_seen % train_config.eval_every_tokens < tokens_per_step:
            val_loss = evaluate(model, eval_batches, device)
            val_ppl = math.exp(val_loss)
            elapsed = time.time() - start_time
            print(
                f"  >>> EVAL | tokens={tokens_seen:>12,} | "
                f"val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f} | "
                f"elapsed={elapsed:.0f}s"
            )
            log_point(tokens_seen, accum_loss, val_loss=val_loss)

        # Save checkpoint
        if tokens_seen % train_config.save_every_tokens < tokens_per_step:
            ckpt_path = run_dir / f"ckpt_{tokens_seen}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "tokens_seen": tokens_seen,
                "config": config_dict,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # Final evaluation
    val_loss = evaluate(model, eval_batches, device)
    val_ppl = math.exp(val_loss)
    total_time = time.time() - start_time
    print("=" * 70)
    print(f"Training complete!")
    print(f"  Model: {args.model} ({args.size})")
    print(f"  Params: {total_params:,}")
    print(f"  Tokens: {tokens_seen:,}")
    print(f"  Final val_loss: {val_loss:.4f}")
    print(f"  Final val_ppl: {val_ppl:.2f}")
    print(f"  Wall time: {total_time:.0f}s")
    print(f"  Learning curve: {curve_path}")

    log_point(tokens_seen, accum_loss, val_loss=val_loss)
    curve_file.close()

    # Save final model
    torch.save(model.state_dict(), run_dir / "model_final.pt")

    return val_loss


def main():
    parser = argparse.ArgumentParser(description="Train and compare architectures")
    parser.add_argument("--model", type=str, required=True,
                        choices=["transformer", "ssm", "predictive_ssm"])
    parser.add_argument("--size", type=str, default="medium",
                        choices=["small", "medium", "large"])
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories",
                        help="HuggingFace dataset name")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--total-tokens", type=int, default=500_000_000,
                        help="Total tokens to train on")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--pred-loss-weight", type=float, default=0.05,
                        help="Weight for predictive coding auxiliary loss")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
