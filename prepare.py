"""
prepare.py — Data preparation, constants, and evaluation utilities.

THIS FILE IS READ-ONLY for the experiment loop.
The agent modifies train.py only. Constants here ensure all experiments
produce directly comparable metrics.
"""

import os
import struct
import urllib.request
import numpy as np

# ============================================================================
# FIXED CONSTANTS — Do not modify. These ensure fair comparison across runs.
# ============================================================================

MAX_SEQ_LEN = 256          # Context length (short for CPU efficiency)
TIME_BUDGET = 120           # Seconds of wall-clock training time per experiment
EVAL_TOKENS = 50_000        # Number of tokens for validation evaluation
VOCAB_SIZE = 256            # Character-level: one byte = one token
BATCH_SIZE = 32             # Fixed batch size
DATA_DIR = os.path.join(os.path.expanduser("~"), ".cache", "model_trainer")

# ============================================================================
# Data Download and Preparation
# ============================================================================

DATASET_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATASET_FILE = os.path.join(DATA_DIR, "input.txt")
TRAIN_BIN = os.path.join(DATA_DIR, "train.bin")
VAL_BIN = os.path.join(DATA_DIR, "val.bin")


def download_data():
    """Download tiny shakespeare dataset if not cached."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATASET_FILE):
        print(f"Downloading dataset to {DATASET_FILE}...")
        urllib.request.urlretrieve(DATASET_URL, DATASET_FILE)
        print("Download complete.")
    return DATASET_FILE


def prepare_data(val_fraction=0.1):
    """
    Tokenize (character-level) and split into train/val binary files.
    Each token is stored as a uint8.
    Returns (train_tokens, val_tokens) as numpy arrays.
    """
    if os.path.exists(TRAIN_BIN) and os.path.exists(VAL_BIN):
        train_data = np.fromfile(TRAIN_BIN, dtype=np.uint8)
        val_data = np.fromfile(VAL_BIN, dtype=np.uint8)
        return train_data, val_data

    filepath = download_data()
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Character-level tokenization: each byte is a token
    data = np.array(list(text.encode("utf-8")), dtype=np.uint8)

    # Split
    split_idx = int(len(data) * (1 - val_fraction))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # Save
    os.makedirs(DATA_DIR, exist_ok=True)
    train_data.tofile(TRAIN_BIN)
    val_data.tofile(VAL_BIN)

    print(f"Prepared {len(train_data)} train tokens, {len(val_data)} val tokens")
    return train_data, val_data


# ============================================================================
# Data Loading Utilities
# ============================================================================

def get_batch(data, batch_size=BATCH_SIZE, seq_len=MAX_SEQ_LEN):
    """
    Sample a random batch of sequences from data.
    Returns (inputs, targets) as numpy arrays of shape (batch_size, seq_len).
    inputs[i] = data[start:start+seq_len]
    targets[i] = data[start+1:start+seq_len+1]
    """
    max_start = len(data) - seq_len - 1
    starts = np.random.randint(0, max_start, size=batch_size)
    inputs = np.stack([data[s:s + seq_len] for s in starts])
    targets = np.stack([data[s + 1:s + seq_len + 1] for s in starts])
    return inputs, targets


# ============================================================================
# Evaluation
# ============================================================================

def compute_bpb(avg_loss_nats):
    """
    Convert average cross-entropy loss (in nats per token) to bits per byte.
    For character-level models where 1 token = 1 byte, this is simply:
        bpb = loss_nats / ln(2)
    """
    return avg_loss_nats / np.log(2)


def evaluate_model(model_forward_fn, val_data, device="cpu"):
    """
    Evaluate a model on validation data.

    Args:
        model_forward_fn: callable(inputs) -> avg_loss_nats
            Takes a numpy array of shape (batch_size, seq_len) of uint8 token ids,
            returns the average cross-entropy loss in nats across all positions.
        val_data: numpy array of uint8 tokens
        device: unused (kept for API compatibility)

    Returns:
        val_bpb: float — validation bits per byte (lower is better)
    """
    import torch
    total_loss = 0.0
    total_tokens = 0
    tokens_remaining = EVAL_TOKENS

    while tokens_remaining > 0:
        current_batch = min(BATCH_SIZE, max(1, tokens_remaining // MAX_SEQ_LEN))
        inputs, targets = get_batch(val_data, batch_size=current_batch, seq_len=MAX_SEQ_LEN)

        with torch.no_grad():
            loss = model_forward_fn(inputs, targets)

        n_tokens = current_batch * MAX_SEQ_LEN
        total_loss += loss * n_tokens
        total_tokens += n_tokens
        tokens_remaining -= n_tokens

    avg_loss = total_loss / total_tokens
    return compute_bpb(avg_loss)


# ============================================================================
# Main — run standalone to prepare data
# ============================================================================

if __name__ == "__main__":
    train_data, val_data = prepare_data()
    print(f"Train: {len(train_data):,} tokens")
    print(f"Val:   {len(val_data):,} tokens")
    print(f"Vocab size: {VOCAB_SIZE} (character-level)")
    print(f"Sequence length: {MAX_SEQ_LEN}")
    print(f"Time budget: {TIME_BUDGET}s per experiment")

    # Print some stats
    unique = len(set(train_data.tolist()))
    print(f"Unique byte values in training data: {unique}")
