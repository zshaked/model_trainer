"""
Predictive Coding SSM — the experimental model.

Core idea from neuroscience (Rao & Ballard 1999, Friston 2005):
Each layer predicts what the NEXT layer will receive. Only the
prediction error propagates forward. This means:

1. If a layer already "knows" what's coming, it contributes nothing
   → compute is focused on surprising/informative tokens
2. Higher layers see increasingly abstract error signals
   → natural hierarchy of representations
3. The prediction targets create a self-supervised signal at every layer
   → gradients don't have to flow through the entire network

The hypothesis: this learns more per token than standard architectures
because it extracts more signal from each example.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ssm import PoleUnit, SwiGLUFF


class PredictiveCodingBlock(nn.Module):
    """
    SSM block with predictive coding.

    Each block:
    1. Receives input x and (optionally) a prediction from the layer above
    2. If prediction exists, computes error = x - prediction
    3. Uses error-gated input (amplify surprising parts, suppress predicted parts)
    4. Runs SSM + FFN as usual
    5. Generates a prediction of what the next layer will receive
    """

    def __init__(self, config, is_first=False, is_last=False):
        super().__init__()
        self.is_first = is_first
        self.is_last = is_last
        dim = config.dim

        # Standard SSM block components
        self.norm1 = nn.RMSNorm(dim)
        self.ssm = PoleUnit(dim, num_heads=config.num_heads)
        self.norm2 = nn.RMSNorm(dim)
        self.ff = SwiGLUFF(dim, config.ff_mult)

        # Predictive coding components
        if not is_first:
            # Error gate: modulates input based on prediction error
            # Surprising tokens (high error) get amplified
            self.error_norm = nn.RMSNorm(dim)
            self.error_gate = nn.Linear(dim * 2, dim, bias=False)

        if not is_last:
            # Predictor: predicts what the next layer's input will look like
            # This is a lightweight projection (not a full SSM)
            self.predictor = nn.Linear(dim, dim, bias=False)

    def forward(self, x, prediction_from_above=None):
        """
        Args:
            x: (B, T, D) input to this layer
            prediction_from_above: (B, T, D) what the previous layer
                                   predicted this layer's input would be

        Returns:
            output: (B, T, D) layer output (with residual)
            prediction: (B, T, D) this layer's prediction for the next layer
            error_magnitude: scalar, mean |error| for logging
        """
        error_mag = torch.tensor(0.0, device=x.device)

        # Step 1: Compute prediction error and gate input
        if prediction_from_above is not None and not self.is_first:
            error = x - prediction_from_above
            error_mag = error.abs().mean().detach()

            # Error-gated input: blend original with error signal
            # When prediction is good (small error), mostly pass through x
            # When prediction is bad (large error), amplify the error signal
            h_err = self.error_norm(error)
            gate_input = torch.cat([x, h_err], dim=-1)
            modulation = torch.sigmoid(self.error_gate(gate_input))
            x = x * modulation + h_err * (1 - modulation)

        # Step 2: Standard SSM + FFN (identical to base SSM block)
        x = x + self.ssm(self.norm1(x))
        x = x + self.ff(self.norm2(x))

        # Step 3: Generate prediction for next layer
        prediction = None
        if not self.is_last:
            prediction = self.predictor(x)

        return x, prediction, error_mag


class PredictiveSSM(nn.Module):
    """
    SSM with hierarchical predictive coding.

    Key difference from standard SSM: layers communicate predictions
    top-down, and only prediction errors flow bottom-up. This creates
    an implicit curriculum where the model focuses on what it doesn't
    yet know how to predict.

    forward(idx) -> logits (same interface as Transformer and SSM)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.dim)

        self.blocks = nn.ModuleList([
            PredictiveCodingBlock(
                config,
                is_first=(i == 0),
                is_last=(i == config.num_layers - 1),
            )
            for i in range(config.num_layers)
        ])

        self.norm_out = nn.RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        if config.tie_weights:
            self.head.weight = self.tok_emb.weight

        # Store error magnitudes for logging
        self._error_mags = []

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

        # Initialize predictors with small weights — they'll learn to predict
        # Starting near-zero means errors ≈ input initially (no prediction yet)
        for block in self.blocks:
            if hasattr(block, 'predictor'):
                nn.init.trunc_normal_(block.predictor.weight, std=0.002)
            if hasattr(block, 'error_gate'):
                # Initialize error gate so output ≈ x (pass-through) at init
                # First half of weights (for x) = positive, second half (for error) = zero
                nn.init.zeros_(block.error_gate.weight)
                dim = self.config.dim
                # Set the x portion to produce ~0.9 (sigmoid(2.2) ≈ 0.9)
                with torch.no_grad():
                    block.error_gate.weight[:, :dim].fill_(0.02)
                    block.error_gate.weight[:, dim:].fill_(0.0)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        # Forward pass with top-down predictions
        # Each layer passes its prediction to the next layer
        prediction = None
        self._error_mags = []

        for block in self.blocks:
            x, prediction, error_mag = block(x, prediction)
            self._error_mags.append(error_mag)

        x = self.norm_out(x)
        return self.head(x)

    def get_prediction_loss(self):
        """
        Auxiliary loss: mean prediction error across layers.
        This can be added to the main loss to encourage good predictions.
        Weight it small (0.01-0.1) so it doesn't dominate.
        Returns 0 if no errors computed (first forward pass).
        """
        if not self._error_mags:
            return torch.tensor(0.0)
        return torch.stack(self._error_mags).mean()

    def get_error_stats(self):
        """Return per-layer error magnitudes for logging."""
        return [e.item() for e in self._error_mags]

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def name():
        return "predictive_ssm"
