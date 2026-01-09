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
from .quality.quality_scorer import QualityScorer, QualityFilter
from .quality.diversity_checker import DiversityChecker
from .quality.human_review import HumanReviewer
from .quality.validators import DataValidator, ContentFilter
from .refinement.iterative_refiner import IterativeRefiner, AdaptiveRefiner
from .analytics.visualizations import DatasetAnalytics

__version__ = "0.3.0"
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
    "QualityScorer",
    "QualityFilter",
    "DiversityChecker",
    "HumanReviewer",
    "DataValidator",
    "ContentFilter",
    "IterativeRefiner",
    "AdaptiveRefiner",
    "DatasetAnalytics",
]
