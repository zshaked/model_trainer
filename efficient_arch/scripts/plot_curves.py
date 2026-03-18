"""
Plot learning curves — the key comparison.

Usage:
    python scripts/plot_curves.py runs/transformer_medium_seed42 runs/ssm_medium_seed42 runs/predictive_ssm_medium_seed42
    python scripts/plot_curves.py runs/*_medium_seed42 --metric val_loss
"""
import json
import sys
import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_curve(run_dir):
    """Load learning curve from a run directory."""
    run_dir = Path(run_dir)
    curve_path = run_dir / "learning_curve.jsonl"

    # Load config for model name
    config_path = run_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    points = []
    with open(curve_path) as f:
        for line in f:
            if line.strip():
                points.append(json.loads(line))

    return {
        "name": f"{config['model']} ({config['size']}, {config['total_params']:,} params)",
        "model": config["model"],
        "points": points,
    }


def print_comparison(curves, metric="val_loss"):
    """Print text-based comparison table."""
    print(f"\n{'='*80}")
    print(f"Learning Efficiency Comparison — {metric}")
    print(f"{'='*80}")

    # Collect all eval points
    for curve in curves:
        eval_points = [p for p in curve["points"] if p.get(metric) is not None]
        if not eval_points:
            print(f"\n{curve['name']}: No {metric} data")
            continue

        print(f"\n{curve['name']}:")
        print(f"  {'Tokens':>15} | {metric:>12} | {'Wall Time':>10}")
        print(f"  {'-'*15}-+-{'-'*12}-+-{'-'*10}")
        for p in eval_points:
            print(f"  {p['tokens']:>15,} | {p[metric]:>12.4f} | {p['wall_time']:>9.0f}s")

    # Find tokens-to-target for each model
    print(f"\n{'='*80}")
    print(f"Tokens to reach target loss (LOWER = MORE EFFICIENT)")
    print(f"{'='*80}")

    targets = [3.0, 2.5, 2.0, 1.5, 1.0]  # loss targets
    header = f"  {'Model':<30}"
    for t in targets:
        header += f" | loss<{t}"
    print(header)
    print(f"  {'-'*30}" + "".join(f"-+-{'-'*8}" for _ in targets))

    for curve in curves:
        eval_points = [p for p in curve["points"] if p.get(metric) is not None]
        row = f"  {curve['name'][:30]:<30}"
        for target in targets:
            reached = [p for p in eval_points if p[metric] < target]
            if reached:
                tokens = reached[0]["tokens"]
                if tokens >= 1_000_000:
                    row += f" | {tokens/1e6:>6.1f}M"
                else:
                    row += f" | {tokens/1e3:>6.0f}K"
            else:
                row += f" |     N/A"
        print(row)


def plot_comparison(curves, metric="val_loss", output_path=None):
    """Plot learning curves with matplotlib."""
    if not HAS_MPL:
        print("matplotlib not installed. Install with: pip install matplotlib")
        print("Falling back to text output.\n")
        print_comparison(curves, metric)
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"transformer": "#e74c3c", "ssm": "#3498db", "predictive_ssm": "#2ecc71"}

    # Plot 1: Loss vs tokens (the key chart)
    ax = axes[0]
    for curve in curves:
        eval_points = [p for p in curve["points"] if p.get(metric) is not None]
        if not eval_points:
            continue
        tokens = [p["tokens"] / 1e6 for p in eval_points]
        losses = [p[metric] for p in eval_points]
        color = colors.get(curve["model"], "#95a5a6")
        ax.plot(tokens, losses, marker="o", markersize=3, label=curve["name"],
                color=color, linewidth=2)

    ax.set_xlabel("Tokens Seen (millions)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("Learning Efficiency: Loss vs Tokens")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Loss vs wall time (controls for speed differences)
    ax = axes[1]
    for curve in curves:
        eval_points = [p for p in curve["points"] if p.get(metric) is not None]
        if not eval_points:
            continue
        times = [p["wall_time"] / 60 for p in eval_points]
        losses = [p[metric] for p in eval_points]
        color = colors.get(curve["model"], "#95a5a6")
        ax.plot(times, losses, marker="o", markersize=3, label=curve["name"],
                color=color, linewidth=2)

    ax.set_xlabel("Wall Time (minutes)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("Compute Efficiency: Loss vs Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.savefig("learning_curves.png", dpi=150, bbox_inches="tight")
        print("Saved plot to learning_curves.png")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare learning curves")
    parser.add_argument("run_dirs", nargs="+", help="Paths to run directories")
    parser.add_argument("--metric", default="val_loss", choices=["val_loss", "train_loss"])
    parser.add_argument("--output", type=str, default=None, help="Output plot path")
    args = parser.parse_args()

    curves = []
    for d in args.run_dirs:
        try:
            curves.append(load_curve(d))
        except Exception as e:
            print(f"Warning: couldn't load {d}: {e}")

    if not curves:
        print("No valid runs found.")
        sys.exit(1)

    print_comparison(curves, args.metric)
    plot_comparison(curves, args.metric, args.output)


if __name__ == "__main__":
    main()
