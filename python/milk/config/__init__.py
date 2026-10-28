"""What a pipeline is: the config model, ``{var}`` substitution and confresolve. Pure data."""

from .model import PipelineConfig
from .resolve import confresolve

__all__ = ["PipelineConfig", "confresolve"]
