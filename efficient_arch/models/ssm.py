"""
SSM model — pole-parameterized state space model.
Same parameter count as transformer, but replaces attention with
FFT-based causal convolution using learnable complex poles.

This is the foundation model that predictive coding builds on.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PoleUnit(nn.Module):
    """
    Multi-head pole-parameterized sequence mixer.
    Each head has learnable poles (sigma + i*omega) at different timescales.
    Uses FFT for O(T log T) parallel convolution.
    """

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        # Pole parameters
        self.raw_sigma = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.5)
        self.omega = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.1)
        self.log_dt = nn.Parameter(torch.zeros(num_heads, self.head_dim))

        # Projections
        self.W_in = nn.Linear(dim, dim, bias=False)
        self.W_out = nn.Linear(dim, dim, bias=False)

        # Short causal conv for local patterns
        self.short_conv = nn.Conv1d(
            dim, dim, kernel_size=4, padding=3, groups=dim, bias=False
        )

        # Gate: uses input + hidden + shifted hidden
        self.gate = nn.Linear(dim * 3, dim)

        self._init_poles()

    def _init_poles(self):
        """Spread heads across timescales."""
        with torch.no_grad():
            for h in range(self.num_heads):
                head_scale = h / max(self.num_heads - 1, 1)
                target_sigma = -5.0 + 4.9 * head_scale
                self.raw_sigma[h].fill_(-target_sigma)
                self.omega[h].uniform_(-(0.5 + head_scale), 0.5 + head_scale)
                self.log_dt[h].fill_(-1.0 + 2.0 * head_scale)

    def forward(self, x):
        B, T, D = x.shape

        # Poles
        sigma = -F.softplus(self.raw_sigma).reshape(-1)
        omega = self.omega.reshape(-1)

        # Bilinear discretization
        dt = torch.exp(self.log_dt.reshape(-1))
        s_dt_half = (sigma + 1j * omega) * dt / 2.0
        z = (1.0 + s_dt_half) / (1.0 - s_dt_half)
        z_mag = torch.abs(z)
        z_angle = torch.angle(z)

        # Input projection + short conv
        x_proj = self.W_in(x)
        x_proj = F.silu(
            self.short_conv(x_proj.transpose(1, 2))[:, :, :T]
        ).transpose(1, 2)

        # Build kernel: kernel[t] = |z|^t * cos(angle*t)
        t_idx = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(1)
        log_mag = torch.log(z_mag.clamp(min=1e-8))
        kernel_mag = torch.exp(t_idx * log_mag.unsqueeze(0))
        kernel_phase = t_idx * z_angle.unsqueeze(0)
        kernel_real = kernel_mag * torch.cos(kernel_phase)

        # FFT convolution
        fft_len = 2 * T
        x_f = torch.fft.rfft(x_proj.transpose(1, 2), n=fft_len, dim=-1)
        k_f = torch.fft.rfft(kernel_real.T, n=fft_len, dim=-1)
        h_seq = torch.fft.irfft(x_f * k_f.unsqueeze(0), n=fft_len, dim=-1)[..., :T]
        h_seq = h_seq.transpose(1, 2)

        # Gate with hyperpolarization
        h_shifted = F.pad(h_seq[:, :-1, :], (0, 0, 1, 0))
        gate_input = torch.cat([x, h_seq, h_shifted], dim=-1)
        g = torch.sigmoid(self.gate(gate_input))
        h_gated = g * h_seq

        return self.W_out(h_gated)


class SwiGLUFF(nn.Module):
    def __init__(self, dim, mult=2.667):
        super().__init__()
        ff_dim = int(dim * mult)
        self.w_gate = nn.Linear(dim, ff_dim, bias=False)
        self.w_up = nn.Linear(dim, ff_dim, bias=False)
        self.w_down = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class SSMBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.dim)
        self.ssm = PoleUnit(config.dim, num_heads=config.num_heads)
        self.norm2 = nn.RMSNorm(config.dim)
        self.ff = SwiGLUFF(config.dim, config.ff_mult)

    def forward(self, x):
        x = x + self.ssm(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class SSM(nn.Module):
    """
    Pole-parameterized SSM. Same interface as Transformer.
    forward(idx) -> logits
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.dim)
        self.blocks = nn.ModuleList([
            SSMBlock(config) for _ in range(config.num_layers)
        ])
        self.norm_out = nn.RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        if config.tie_weights:
            self.head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.tok_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.pos_emb.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm_out(x)
        return self.head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def name():
        return "ssm"
