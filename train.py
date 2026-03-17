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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    Multi-head pole-parameterized recurrent unit.

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
    """

    def __init__(self, input_dim, hidden_dim, num_heads=NUM_HEADS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        # Learnable pole parameters per head
        # Shape: (num_heads, head_dim)
        # raw_sigma: unconstrained, mapped to sigma < 0 via -softplus
        # omega: oscillation frequency, unconstrained
        self.raw_sigma = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.5)
        self.omega = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.1)

        # Learned time step (dt) per head dimension
        self.log_dt = nn.Parameter(torch.zeros(num_heads, self.head_dim))

        # Input projection
        self.W_in = nn.Linear(input_dim, hidden_dim, bias=False)

        # Output projection
        self.W_out = nn.Linear(hidden_dim, input_dim, bias=False)

        # Mixing gate
        self.gate = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Frequency-band attention parameters
        # Split hidden_dim into num_bands groups based on omega values
        self.num_bands = 4
        self.band_size = hidden_dim // self.num_bands
        assert hidden_dim % self.num_bands == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_bands ({self.num_bands})"

        # Small Q/K/V projections for cross-band attention
        band_proj_dim = 16
        self.band_q = nn.Linear(self.band_size, band_proj_dim, bias=False)
        self.band_k = nn.Linear(self.band_size, band_proj_dim, bias=False)
        self.band_v = nn.Linear(self.band_size, band_proj_dim, bias=False)
        self.band_out = nn.Linear(band_proj_dim, self.band_size, bias=False)

    def get_poles(self):
        """Return (sigma, omega) with sigma constrained < 0.
        Returns:
            sigma: (num_heads, head_dim)
            omega: (num_heads, head_dim)
        """
        sigma = -F.softplus(self.raw_sigma)  # Always negative
        omega = self.omega
        return sigma, omega

    def apply_frequency_band_attention(self, h_seq):
        """
        Apply lightweight cross-band attention based on frequency bands.

        Args:
            h_seq: (B, T, hidden_dim) output from FFT convolution

        Returns:
            h_seq_attn: (B, T, hidden_dim) after band attention
        """
        B, T, D = h_seq.shape
        # Reshape into bands: (B, T, num_bands, band_size)
        h_bands = h_seq.reshape(B, T, self.num_bands, self.band_size)
        # Reshape for attention: (B*T, num_bands, band_size)
        h_bands_flat = h_bands.reshape(B * T, self.num_bands, self.band_size)

        # Compute Q, K, V for each band
        # Q, K, V: (B*T, num_bands, band_proj_dim)
        q = self.band_q(h_bands_flat)  # (B*T, num_bands, band_proj_dim)
        k = self.band_k(h_bands_flat)  # (B*T, num_bands, band_proj_dim)
        v = self.band_v(h_bands_flat)  # (B*T, num_bands, band_proj_dim)

        # Dot-product attention across bands
        # scores: (B*T, num_bands, num_bands)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)  # (B*T, num_bands, num_bands)

        # Apply attention to values
        # attn_out: (B*T, num_bands, band_proj_dim)
        attn_out = torch.matmul(attn_weights, v)

        # Project back to band_size
        # h_attn: (B*T, num_bands, band_size)
        h_attn = self.band_out(attn_out)

        # Reshape back: (B*T, num_bands, band_size) -> (B, T, hidden_dim)
        h_seq_attn = h_attn.reshape(B, T, D)

        return h_seq_attn

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            output: (batch, seq_len, input_dim)
        """
        B, T, D = x.shape
        sigma, omega = self.get_poles()  # (num_heads, head_dim)

        # Project input to hidden space
        x_proj = self.W_in(x)  # (B, T, hidden_dim)
        # Reshape to (B, T, num_heads, head_dim)
        x_proj = x_proj.reshape(B, T, self.num_heads, self.head_dim)

        # Process each head independently
        head_outputs = []

        for h in range(self.num_heads):
            x_head = x_proj[:, :, h, :]  # (B, T, head_dim)
            sigma_h = sigma[h]  # (head_dim,)
            omega_h = omega[h]  # (head_dim,)
            log_dt_h = self.log_dt[h]  # (head_dim,)

            # Compute discrete-time pole using bilinear (Tustin) transform
            # z = (1 + s*dt/2) / (1 - s*dt/2) where s = sigma + i*omega
            dt = torch.exp(log_dt_h)  # (head_dim,) — learned positive time step

            # Bilinear transform: s*dt/2
            s_dt_half = (sigma_h + 1j * omega_h) * dt / 2.0

            # numerator = 1 + s*dt/2, denominator = 1 - s*dt/2
            numerator = 1.0 + s_dt_half  # (head_dim,) complex
            denominator = 1.0 - s_dt_half  # (head_dim,) complex

            # z = numerator / denominator
            z = numerator / denominator  # (head_dim,) complex

            # Extract magnitude and angle for the kernel
            z_mag = torch.abs(z)  # (head_dim,)
            z_angle = torch.angle(z)  # (head_dim,)

            # Build causal convolution kernel via FFT (parallel, O(T log T))
            # kernel[t] = z^t = |z|^t * e^(i*angle*t)
            t_idx = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(1)  # (T, 1)
            # kernel_real[t] = |z|^t * cos(angle*t)
            log_mag = torch.log(z_mag.clamp(min=1e-8))  # (head_dim,)
            kernel_mag = torch.exp(t_idx * log_mag.unsqueeze(0))  # (T, head_dim)
            kernel_angle = t_idx * z_angle.unsqueeze(0)  # (T, head_dim)
            kernel_real = kernel_mag * torch.cos(kernel_angle)  # (T, head_dim)

            # Causal convolution via FFT: h_real[t] = sum_{k=0}^{t} kernel[k] * x[t-k]
            # Pad to avoid circular convolution
            fft_len = 2 * T
            # x_head: (B, T, head_dim) -> (B, head_dim, T) for conv
            x_f = torch.fft.rfft(x_head.transpose(1, 2), n=fft_len, dim=-1)  # (B, head_dim, fft_len//2+1)
            k_f = torch.fft.rfft(kernel_real.T, n=fft_len, dim=-1)  # (head_dim, fft_len//2+1)
            h_seq = torch.fft.irfft(x_f * k_f.unsqueeze(0), n=fft_len, dim=-1)[..., :T]  # (B, head_dim, T)
            h_seq = h_seq.transpose(1, 2)  # (B, T, head_dim)

            head_outputs.append(h_seq)

        # Concatenate all heads
        h_seq = torch.cat(head_outputs, dim=-1)  # (B, T, hidden_dim)

        # Apply frequency-band attention
        h_seq = self.apply_frequency_band_attention(h_seq)  # (B, T, hidden_dim)

        # Gate: mix hidden state with input
        gate_input = torch.cat([x, h_seq], dim=-1)
        g = torch.sigmoid(self.gate(gate_input))
        h_gated = g * h_seq

        # Project back to input dim
        output = self.W_out(h_gated)
        return output


class PoleLayer(nn.Module):
    """
    A single layer: pole-parameterized recurrence + feedforward + residual.
    """

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.pole_unit = PoleUnit(dim, hidden_dim)
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        # Pole recurrence with residual
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
        self.norm_out = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.embedding.weight

        # Initialize poles with spread of timescales per head
        self._init_poles()

    def _init_poles(self):
        """Initialize poles so heads span different timescale bands.
        Within each layer:
        - Head 0: very fast decay (large negative sigma)
        - Head 1: fast decay
        - Head 2: slow decay
        - Head 3: very slow decay (sigma near 0)
        """
        for i, layer in enumerate(self.layers):
            n_layers = len(self.layers)
            # Layer-based variation: early layers faster than later layers
            layer_decay_scale = 0.7 + 0.3 * (i / max(n_layers - 1, 1))

            with torch.no_grad():
                # Initialize each head with different sigma ranges
                for h in range(NUM_HEADS):
                    # Head 0 (h=0): fast, Head 3 (h=3): slow
                    head_scale = h / max(NUM_HEADS - 1, 1)
                    # Sigma ranges from -3.0 (fast) to -0.3 (slow), scaled by layer
                    target_sigma = -3.0 + 2.7 * head_scale
                    target_sigma *= layer_decay_scale

                    layer.pole_unit.raw_sigma[h].fill_(-target_sigma)

                    # Spread of oscillation frequencies
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
        """Return pole statistics for logging."""
        all_sigma = []
        all_omega = []
        for layer in self.layers:
            sigma, omega = layer.pole_unit.get_poles()
            # Flatten heads: (num_heads, head_dim) -> (hidden_dim,)
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
    EST_TOTAL_STEPS = 130  # Estimated total steps in TIME_BUDGET
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

    # Print pole statistics
    pole_stats = model.get_pole_stats()
    print(f"\nPole statistics:")
    for i in range(NUM_LAYERS):
        print(f"  Layer {i}: sigma={pole_stats['sigma_mean'][i]:.3f}±{pole_stats['sigma_std'][i]:.3f}  "
              f"omega={pole_stats['omega_mean'][i]:.3f}±{pole_stats['omega_std'][i]:.3f}")

    # === RESULT LINE — parsed by the experiment loop ===
    print(f"\n=== RESULT val_bpb={val_bpb:.6f} steps={step} time={total_time:.1f}s params={n_params} ===")

    return val_bpb


if __name__ == "__main__":
    train()
