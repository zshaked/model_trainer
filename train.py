"""
train.py — Phase 1: Pole-Parameterized Sequence Model

THIS IS THE FILE THE AGENT MODIFIES.

Architecture: A recurrent sequence model where each unit is parameterized by
a learnable pole in the complex plane (sigma + i*omega).
- sigma (real part): decay rate, constrained < 0 for stability
- omega (imaginary part): oscillation frequency

The model does character-level language modeling on tiny shakespeare.
It outputs val_bpb (validation bits per byte) as the single evaluation metric.
"""

import time
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Compatibility: nn.RMSNorm was added in PyTorch 2.4
if hasattr(nn, 'RMSNorm'):
    RMSNorm = nn.RMSNorm
else:
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-8):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
        def forward(self, x):
            rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
            return x / rms * self.weight

# Setup deterministic seed if provided
SEED = int(os.environ.get("SEED", 42))
torch.manual_seed(SEED)
np.random.seed(SEED)

# Import fixed constants from prepare.py
from prepare import (
    MAX_SEQ_LEN, TIME_BUDGET, VOCAB_SIZE, BATCH_SIZE,
    prepare_data, get_batch, evaluate_model, compute_bpb,
)

# ============================================================================
# Hyperparameters (the agent may tune these)
# ============================================================================

HIDDEN_DIM = 128        # Hidden state dimension
NUM_LAYERS = 2          # Number of pole-parameterized layers
NUM_HEADS = 4           # Number of pole heads for multi-head structure
LEARNING_RATE = 3e-3    # Learning rate
WEIGHT_DECAY = 0.01     # Weight decay
WARMUP_STEPS = 20       # Linear warmup steps
LOG_INTERVAL = 10       # Print loss every N steps

# ============================================================================
# Phase 1: Pole-Parameterized Unit with Multi-Head Structure
# ============================================================================

class PoleUnit(nn.Module):
    """
    Multi-head pole-parameterized recurrent unit with frequency-band attention.

    Splits hidden_dim into NUM_HEADS heads, each processing independently.
    Each head has its own poles (sigma_j + i*omega_j) initialized at different
    timescale bands:
    - Head 0: fast decay (large negative sigma)
    - Head 3: slow decay (sigma near 0)

    The recurrence is:
        h[t] = exp(sigma + i*omega) * h[t-1] + W_in * x[t]

    sigma is constrained < 0 via: sigma = -softplus(raw_sigma)
    This guarantees stability by construction.

    Each head processes its portion of the hidden state independently,
    then the outputs are concatenated.

    After FFT convolution, applies lightweight frequency-band attention:
    splits the output into 4 frequency bands based on omega values,
    applies dot-product attention across bands, then reshapes back.
    """

    def __init__(self, input_dim, hidden_dim, num_heads=NUM_HEADS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        # Learnable pole parameters — stored as (num_heads, head_dim) for
        # per-head initialization, but flattened to (hidden_dim,) for vectorized FFT
        self.raw_sigma = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.5)
        self.omega = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.1)

        # Learned time step (dt) per dimension
        self.log_dt = nn.Parameter(torch.zeros(num_heads, self.head_dim))

        # Input projection
        self.W_in = nn.Linear(input_dim, hidden_dim, bias=False)

        # Output projection
        self.W_out = nn.Linear(hidden_dim, input_dim, bias=False)

        # Mixing gate
        self.gate = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def get_poles(self):
        """Return (sigma, omega) with sigma constrained < 0.
        Returns flattened to (hidden_dim,) for vectorized computation.
        """
        sigma = -F.softplus(self.raw_sigma).reshape(-1)  # (hidden_dim,)
        omega = self.omega.reshape(-1)  # (hidden_dim,)
        return sigma, omega

    def forward(self, x):
        """
        Vectorized forward — no per-head loop. All heads processed in one FFT.
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            output: (batch, seq_len, input_dim)
        """
        B, T, D = x.shape
        sigma, omega = self.get_poles()  # (hidden_dim,)

        # Compute discrete-time pole using bilinear (Tustin) transform
        # z = (1 + s*dt/2) / (1 - s*dt/2) where s = sigma + i*omega
        dt = torch.exp(self.log_dt.reshape(-1))  # (hidden_dim,)
        s_dt_half = (sigma + 1j * omega) * dt / 2.0
        z = (1.0 + s_dt_half) / (1.0 - s_dt_half)  # (hidden_dim,) complex

        # Extract magnitude and angle for the kernel
        z_mag = torch.abs(z)  # (hidden_dim,)
        z_angle = torch.angle(z)  # (hidden_dim,)

        # Project input to hidden space
        x_proj = self.W_in(x)  # (B, T, hidden_dim)

        # Build causal convolution kernel via FFT (parallel, O(T log T))
        # kernel[t] = z^t = |z|^t * cos(angle*t)
        t_idx = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(1)  # (T, 1)
        log_mag = torch.log(z_mag.clamp(min=1e-8))  # (hidden_dim,)
        kernel_mag = torch.exp(t_idx * log_mag.unsqueeze(0))  # (T, hidden_dim)
        kernel_phase = t_idx * z_angle.unsqueeze(0)  # (T, hidden_dim)
        kernel_real = kernel_mag * torch.cos(kernel_phase)  # (T, hidden_dim)

        # Causal convolution via FFT
        fft_len = 2 * T
        x_f = torch.fft.rfft(x_proj.transpose(1, 2), n=fft_len, dim=-1)  # (B, H, fft_len//2+1)
        k_f = torch.fft.rfft(kernel_real.T, n=fft_len, dim=-1)  # (H, fft_len//2+1)
        h_seq = torch.fft.irfft(x_f * k_f.unsqueeze(0), n=fft_len, dim=-1)[..., :T]  # (B, H, T)
        h_seq = h_seq.transpose(1, 2)  # (B, T, H)

        # Gate: mix hidden state with input
        gate_input = torch.cat([x, h_seq], dim=-1)
        g = torch.sigmoid(self.gate(gate_input))
        h_gated = g * h_seq

        # Project back to input dim
        output = self.W_out(h_gated)
        return output


class TransformerUnit(nn.Module):
    """
    Causal multi-head self-attention — drop-in replacement for PoleUnit.
    Used for matched baseline comparison against the pole-parameterized model.
    """

    def __init__(self, input_dim, hidden_dim, num_heads=NUM_HEADS):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert input_dim % num_heads == 0, f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads})"
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(input_dim, 3 * input_dim, bias=False)
        self.out_proj = nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H = self.num_heads
        qkv = self.qkv(x).reshape(B, T, 3, H, self.head_dim)
        q, k, v = qkv.unbind(2)  # each (B, T, H, head_dim)
        q = q.transpose(1, 2)  # (B, H, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        causal_mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
        attn = attn.masked_fill(~causal_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)  # (B, T, D)
        return self.out_proj(out)


class LSTMUnit(nn.Module):
    """
    LSTM-based sequence unit — drop-in replacement for PoleUnit.
    Used for matched baseline comparison against the pole-parameterized model.
    Uses a single LSTM layer with the same input/output dimensions.
    """

    def __init__(self, input_dim, hidden_dim, num_heads=NUM_HEADS):
        super().__init__()
        self.hidden_dim = hidden_dim
        # LSTM: 4 gates, each needs (input→hidden) and (hidden→hidden) weights
        # To match param count better, use hidden_dim // 2 as LSTM hidden size
        self.lstm_hidden = hidden_dim // 2
        self.lstm = nn.LSTM(input_dim, self.lstm_hidden, batch_first=True)
        self.out_proj = nn.Linear(self.lstm_hidden, input_dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        # LSTM processes sequence; no explicit hidden state carried across batches
        lstm_out, _ = self.lstm(x)  # (B, T, lstm_hidden)
        return self.out_proj(lstm_out)  # (B, T, input_dim)


# MODEL_TYPE: switch between "pole", "transformer", "lstm" for baseline comparison
MODEL_TYPE = "pole"


class PoleLayer(nn.Module):
    """
    A single layer: sequence unit + feedforward + residual.
    Uses PoleUnit or TransformerUnit depending on MODEL_TYPE.
    """

    def __init__(self, dim, hidden_dim):
        super().__init__()
        if MODEL_TYPE == "transformer":
            self.pole_unit = TransformerUnit(dim, hidden_dim)
        elif MODEL_TYPE == "lstm":
            self.pole_unit = LSTMUnit(dim, hidden_dim)
        else:
            self.pole_unit = PoleUnit(dim, hidden_dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        # Sequence unit with residual
        x = x + self.pole_unit(self.norm1(x))
        # Feedforward with residual
        x = x + self.ff(self.norm2(x))
        return x


class PoleModel(nn.Module):
    """
    Full sequence model: embedding -> N pole layers -> output projection.
    Character-level language model.
    """

    def __init__(self, vocab_size=VOCAB_SIZE, dim=HIDDEN_DIM,
                 hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            PoleLayer(dim, hidden_dim) for _ in range(num_layers)
        ])
        self.norm_out = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.embedding.weight

        # Initialize poles with spread of timescales per head
        self._init_poles()

    def _init_poles(self):
        """Initialize poles so heads span different timescale bands.
        No-op for non-pole MODEL_TYPEs.
        """
        if MODEL_TYPE != "pole":
            return
        for i, layer in enumerate(self.layers):
            n_layers = len(self.layers)
            layer_decay_scale = 0.7 + 0.3 * (i / max(n_layers - 1, 1))
            with torch.no_grad():
                for h in range(NUM_HEADS):
                    head_scale = h / max(NUM_HEADS - 1, 1)
                    target_sigma = -3.0 + 2.7 * head_scale
                    target_sigma *= layer_decay_scale
                    layer.pole_unit.raw_sigma[h].fill_(-target_sigma)
                    target_omega_scale = 0.3 + 0.4 * head_scale
                    layer.pole_unit.omega[h].uniform_(-target_omega_scale, target_omega_scale)

    def forward(self, idx):
        """
        Args:
            idx: (batch, seq_len) long tensor of token ids
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        x = self.embedding(idx)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_out(x)
        logits = self.head(x)
        return logits

    def compute_loss(self, inputs, targets):
        """
        Args:
            inputs: numpy array (batch, seq_len) uint8
            targets: numpy array (batch, seq_len) uint8
        Returns:
            loss: scalar float (nats per token)
        """
        device = next(self.parameters()).device
        x = torch.from_numpy(inputs.astype(np.int64)).to(device)
        y = torch.from_numpy(targets.astype(np.int64)).to(device)
        logits = self.forward(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        return loss

    def get_pole_stats(self):
        """Return pole statistics for logging. Returns None for non-pole models."""
        if MODEL_TYPE != "pole":
            return None
        all_sigma = []
        all_omega = []
        for layer in self.layers:
            sigma, omega = layer.pole_unit.get_poles()
            sigma_flat = sigma.reshape(-1)
            omega_flat = omega.reshape(-1)
            all_sigma.append(sigma_flat.detach().cpu())
            all_omega.append(omega_flat.detach().cpu())
        return {
            "sigma_mean": [s.mean().item() for s in all_sigma],
            "sigma_std": [s.std().item() for s in all_sigma],
            "omega_mean": [o.abs().mean().item() for o in all_omega],
            "omega_std": [o.std().item() for o in all_omega],
        }


# ============================================================================
# Training Loop
# ============================================================================

def train():
    device = "cpu"
    print(f"Device: {device}")
    print(f"Time budget: {TIME_BUDGET}s")

    # Prepare data
    train_data, val_data = prepare_data()
    print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")

    # Build model
    model = PoleModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # LR scheduler with warmup + cosine decay
    import math
    EST_TOTAL_STEPS = 700  # Estimated total steps in TIME_BUDGET (FFT model: ~670 steps/120s)
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(WARMUP_STEPS, 1)
        progress = (step - WARMUP_STEPS) / max(EST_TOTAL_STEPS - WARMUP_STEPS, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop — runs for TIME_BUDGET seconds
    model.train()
    step = 0
    best_train_loss = float("inf")
    start_time = time.time()

    print("Training...")
    while True:
        elapsed = time.time() - start_time
        if elapsed >= TIME_BUDGET:
            break

        inputs, targets = get_batch(train_data)
        loss = model.compute_loss(inputs, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        if loss_val < best_train_loss:
            best_train_loss = loss_val

        if step % LOG_INTERVAL == 0:
            bpb = compute_bpb(loss_val)
            elapsed = time.time() - start_time
            print(f"step={step} loss={loss_val:.4f} bpb={bpb:.4f} time={elapsed:.1f}s")

        step += 1

    total_time = time.time() - start_time
    print(f"\nTraining complete: {step} steps in {total_time:.1f}s")

    # Evaluate
    print("Evaluating...")
    model.eval()

    def model_forward(inputs, targets):
        return model.compute_loss(inputs, targets).item()

    val_bpb = evaluate_model(model_forward, val_data, device=device)

    # Print pole statistics (only for pole model)
    pole_stats = model.get_pole_stats()
    if pole_stats is not None:
        print(f"\nPole statistics:")
        for i in range(NUM_LAYERS):
            print(f"  Layer {i}: sigma={pole_stats['sigma_mean'][i]:.3f}±{pole_stats['sigma_std'][i]:.3f}  "
                  f"omega={pole_stats['omega_mean'][i]:.3f}±{pole_stats['omega_std'][i]:.3f}")
    else:
        print(f"\nModel type: {MODEL_TYPE} (no pole statistics)")

    # === RESULT LINE — parsed by the experiment loop ===
    print(f"\n=== RESULT val_bpb={val_bpb:.6f} steps={step} time={total_time:.1f}s params={n_params} ===")

    return val_bpb


if __name__ == "__main__":
    train()
