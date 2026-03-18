"""
Transformer baseline — standard GPT-2 style.
This is the control group. No tricks, just a clean implementation.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.dim % config.num_heads == 0
        self.num_heads = config.num_heads
        self.head_dim = config.dim // config.num_heads

        self.qkv = nn.Linear(config.dim, 3 * config.dim, bias=False)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, H, head_dim)
        q = q.transpose(1, 2)  # (B, H, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use PyTorch's scaled_dot_product_attention (Flash Attention when available)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class SwiGLUFF(nn.Module):
    """SwiGLU feedforward — same across all model variants."""

    def __init__(self, dim, mult=2.667):
        super().__init__()
        ff_dim = int(dim * mult)
        self.w_gate = nn.Linear(dim, ff_dim, bias=False)
        self.w_up = nn.Linear(dim, ff_dim, bias=False)
        self.w_down = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.dim)
        self.attn = CausalSelfAttention(config)
        self.norm2 = nn.RMSNorm(config.dim)
        self.ff = SwiGLUFF(config.dim, config.ff_mult)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class Transformer(nn.Module):
    """
    GPT-2 style transformer. The control group.
    Interface: forward(idx) -> logits
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
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
        return "transformer"
