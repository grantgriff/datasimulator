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
from .sources.document_loader import DocumentLoader, load_document

__version__ = "0.2.0"
__all__ = [
    "DataSimulator",
    "DocumentLoader",
    "load_document",
    "SFTMessages",
    "SFTCompletion",
    "DPOPreference",
    "DPOMessages",
    "PPOPrompt",
    "GRPOPrompt",
    "RLVerifiable",
]
