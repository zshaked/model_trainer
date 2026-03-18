from .transformer import Transformer
from .ssm import SSM
from .predictive import PredictiveSSM


MODEL_REGISTRY = {
    "transformer": Transformer,
    "ssm": SSM,
    "predictive_ssm": PredictiveSSM,
}


def build_model(name: str, config):
    """Build a model by name. All share the same interface: forward(idx) -> logits."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](config)
