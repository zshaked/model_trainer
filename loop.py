#!/usr/bin/env python3
"""
loop.py — Autonomous experiment loop.

Minimal MVP: modify train.py → run → evaluate → keep or revert.

This script manages the experiment cycle:
1. Run train.py and capture output
2. Parse the RESULT line for val_bpb
3. Compare to previous best
4. If improved: git commit (advance the branch)
5. If not: git reset (discard the change)
6. Log to results.tsv

Usage:
    # Run a single experiment (after manually editing train.py):
    python loop.py --once

    # Run the loop autonomously (for agent use):
    python loop.py --run

    # Show experiment history:
    python loop.py --history
"""

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

RESULTS_FILE = "results.tsv"
TRAIN_SCRIPT = "train.py"
RUN_LOG = "run.log"


def git(*args):
    """Run a git command and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        ["git"] + list(args),
        capture_output=True, text=True, cwd=os.path.dirname(__file__) or "."
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def run_experiment(timeout=300):
    """
    Run train.py and capture output.
    Returns (val_bpb, steps, time, params, log) or (None, ...) on failure.
    """
    print(f"{'='*60}")
    print(f"Running experiment...")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            [sys.executable, TRAIN_SCRIPT],
            capture_output=True, text=True,
            timeout=timeout,
            cwd=os.path.dirname(__file__) or "."
        )
        log = result.stdout + "\n" + result.stderr
    except subprocess.TimeoutExpired:
        print("TIMEOUT: Experiment exceeded time limit")
        return None, None, None, None, "TIMEOUT"
    except Exception as e:
        print(f"ERROR: {e}")
        return None, None, None, None, str(e)

    # Save log
    log_path = os.path.join(os.path.dirname(__file__) or ".", RUN_LOG)
    with open(log_path, "w") as f:
        f.write(log)

    # Parse result line
    match = re.search(
        r"=== RESULT val_bpb=([\d.]+) steps=(\d+) time=([\d.]+)s params=(\d+) ===",
        log
    )
    if match:
        val_bpb = float(match.group(1))
        steps = int(match.group(2))
        train_time = float(match.group(3))
        params = int(match.group(4))
        print(f"Result: val_bpb={val_bpb:.6f} steps={steps} time={train_time:.1f}s")
        return val_bpb, steps, train_time, params, log
    else:
        print("CRASH: Could not parse RESULT line from output")
        # Print last 20 lines for debugging
        lines = log.strip().split("\n")
        print("--- Last 20 lines of output ---")
        for line in lines[-20:]:
            print(f"  {line}")
        print("--- End ---")
        return None, None, None, None, log


def get_best_bpb():
    """Read the best val_bpb from results.tsv."""
    results_path = os.path.join(os.path.dirname(__file__) or ".", RESULTS_FILE)
    if not os.path.exists(results_path):
        return float("inf")

    best = float("inf")
    with open(results_path, "r") as f:
        for line in f:
            if line.startswith("timestamp"):
                continue  # header
            parts = line.strip().split("\t")
            if len(parts) >= 3 and parts[2] != "CRASH":
                try:
                    bpb = float(parts[2])
                    if bpb < best:
                        best = bpb
                except ValueError:
                    pass
    return best


def log_result(tag, val_bpb, steps, train_time, params, kept):
    """Append a line to results.tsv."""
    results_path = os.path.join(os.path.dirname(__file__) or ".", RESULTS_FILE)
    write_header = not os.path.exists(results_path)

    with open(results_path, "a") as f:
        if write_header:
            f.write("timestamp\ttag\tval_bpb\tsteps\ttime_s\tparams\tkept\n")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        bpb_str = f"{val_bpb:.6f}" if val_bpb is not None else "CRASH"
        steps_str = str(steps) if steps is not None else ""
        time_str = f"{train_time:.1f}" if train_time is not None else ""
        params_str = str(params) if params is not None else ""
        kept_str = "yes" if kept else "no"
        f.write(f"{ts}\t{tag}\t{bpb_str}\t{steps_str}\t{time_str}\t{params_str}\t{kept_str}\n")


def get_current_tag():
    """Get a tag for the current experiment from git diff."""
    rc, diff, _ = git("diff", "--stat", "HEAD", "--", TRAIN_SCRIPT)
    if rc != 0 or not diff:
        return "no-change"
    # Try to summarize from diff
    rc, full_diff, _ = git("diff", "HEAD", "--", TRAIN_SCRIPT)
    lines_changed = len([l for l in full_diff.split("\n") if l.startswith("+") or l.startswith("-")])
    return f"edit-{lines_changed}lines"


def run_once(tag=None):
    """
    Run a single experiment cycle:
    1. Run train.py
    2. Compare to best
    3. Keep or revert
    4. Log result
    """
    if tag is None:
        tag = get_current_tag()

    best_bpb = get_best_bpb()
    print(f"Current best val_bpb: {best_bpb:.6f}" if best_bpb < float("inf") else "No previous results")

    val_bpb, steps, train_time, params, log = run_experiment()

    if val_bpb is None:
        # Crash — revert
        print("Experiment crashed. Reverting changes to train.py...")
        git("checkout", "--", TRAIN_SCRIPT)
        log_result(tag, None, None, None, None, False)
        return None, False

    improved = val_bpb < best_bpb
    if improved:
        delta = best_bpb - val_bpb
        print(f"IMPROVED by {delta:.6f} bpb! Keeping change.")
        git("add", TRAIN_SCRIPT)
        git("add", RESULTS_FILE)
        rc, _, err = git("commit", "-m", f"experiment: {tag} — val_bpb={val_bpb:.6f} (improved by {delta:.6f})")
        if rc != 0:
            print(f"Git commit warning: {err}")
    else:
        delta = val_bpb - best_bpb
        print(f"No improvement (+{delta:.6f} bpb). Reverting changes to train.py...")
        git("checkout", "--", TRAIN_SCRIPT)

    log_result(tag, val_bpb, steps, train_time, params, improved)
    return val_bpb, improved


def show_history():
    """Print experiment history."""
    results_path = os.path.join(os.path.dirname(__file__) or ".", RESULTS_FILE)
    if not os.path.exists(results_path):
        print("No experiments yet.")
        return

    with open(results_path, "r") as f:
        print(f.read())

    best = get_best_bpb()
    if best < float("inf"):
        print(f"\nBest val_bpb: {best:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Autonomous experiment loop")
    parser.add_argument("--once", action="store_true",
                        help="Run a single experiment")
    parser.add_argument("--run", action="store_true",
                        help="Run the experiment loop (for agent use)")
    parser.add_argument("--history", action="store_true",
                        help="Show experiment history")
    parser.add_argument("--tag", type=str, default=None,
                        help="Tag for the experiment")
    parser.add_argument("--max-experiments", type=int, default=None,
                        help="Maximum number of experiments (default: unlimited)")
    args = parser.parse_args()

    if args.history:
        show_history()
        return

    if args.once:
        val_bpb, kept = run_once(tag=args.tag)
        sys.exit(0 if val_bpb is not None else 1)

    if args.run:
        print("Starting autonomous experiment loop")
        print("Press Ctrl+C to stop\n")
        experiment_num = 0
        while True:
            if args.max_experiments and experiment_num >= args.max_experiments:
                print(f"\nReached max experiments ({args.max_experiments}). Stopping.")
                break
            experiment_num += 1
            print(f"\n{'#'*60}")
            print(f"# Experiment {experiment_num}")
            print(f"{'#'*60}\n")
            try:
                val_bpb, kept = run_once(tag=args.tag or f"exp-{experiment_num}")
            except KeyboardInterrupt:
                print("\nStopped by user.")
                break
            except Exception as e:
                print(f"Loop error: {e}")
                continue
        show_history()
        return

    # Default: show usage
    parser.print_help()


if __name__ == "__main__":
    main()
