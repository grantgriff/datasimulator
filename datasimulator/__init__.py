"""
Data Simulation SDK for Post-Training Data Generation

Generate high-quality synthetic training data for SFT, DPO, PPO, GRPO, and RL training.
"""

from .sdk import DataSimulator
from .core.data_models import (
    SFTMessages,
    SFTCompletion,
    DPOPreference,
    DPOMessages,
    PPOPrompt,
    GRPOPrompt,
    RLVerifiable,
)

__version__ = "0.1.0"
__all__ = [
    "DataSimulator",
    "SFTMessages",
    "SFTCompletion",
    "DPOPreference",
    "DPOMessages",
    "PPOPrompt",
    "GRPOPrompt",
    "RLVerifiable",
]
