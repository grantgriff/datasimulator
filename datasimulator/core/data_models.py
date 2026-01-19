"""
Pydantic data models for all training data formats.

Supports:
- SFT (Supervised Fine-Tuning): Messages and Completion formats
- DPO (Direct Preference Optimization): Preference pairs
- PPO (Proximal Policy Optimization): Prompt-only
- GRPO (Group Relative Policy Optimization): Prompt-only with multi-completion
- RL with Verifiable Rewards: Ground truth verification
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal, Union, Dict, Any
from datetime import datetime


# ============================================================================
# Message Components
# ============================================================================

class Message(BaseModel):
    """Single message in a conversation."""
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)

    model_config = {"frozen": True}


# ============================================================================
# SFT (Supervised Fine-Tuning) Formats
# ============================================================================

class SFTMessages(BaseModel):
    """SFT format with messages (conversation-style)."""
    messages: List[Message] = Field(min_length=2)

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: List[Message]) -> List[Message]:
        """Ensure messages alternate properly and end with assistant."""
        if not v:
            raise ValueError("Messages cannot be empty")

        # Last message should be from assistant
        if v[-1].role != "assistant":
            raise ValueError("Last message must be from assistant")

        return v


class SFTCompletion(BaseModel):
    """SFT format with simple prompt/completion pairs."""
    prompt: str = Field(min_length=1)
    completion: str = Field(min_length=1)


# ============================================================================
# DPO (Direct Preference Optimization) Formats
# ============================================================================

class DPOPreference(BaseModel):
    """DPO format with string-based chosen/rejected responses."""
    prompt: str = Field(min_length=1)
    chosen: str = Field(min_length=1, description="Better/preferred response")
    rejected: str = Field(min_length=1, description="Worse/rejected response")

    @field_validator("rejected")
    @classmethod
    def validate_different_responses(cls, v: str, info) -> str:
        """Ensure chosen and rejected are different."""
        if "chosen" in info.data and v == info.data["chosen"]:
            raise ValueError("Chosen and rejected responses must be different")
        return v


class DPOMessages(BaseModel):
    """DPO format with message-based chosen/rejected responses."""
    prompt: List[Message] = Field(min_length=1)
    chosen: List[Message] = Field(min_length=1)
    rejected: List[Message] = Field(min_length=1)

    @field_validator("chosen", "rejected")
    @classmethod
    def validate_assistant_message(cls, v: List[Message]) -> List[Message]:
        """Ensure response messages are from assistant."""
        if not v:
            raise ValueError("Response messages cannot be empty")

        # All messages should be from assistant for responses
        for msg in v:
            if msg.role != "assistant":
                raise ValueError("Response messages must be from assistant")

        return v


# ============================================================================
# PPO (Proximal Policy Optimization) Format
# ============================================================================

class PPOPrompt(BaseModel):
    """PPO format - prompt-only (rewards come from reward model)."""
    prompt: str = Field(min_length=1)


# ============================================================================
# GRPO (Group Relative Policy Optimization) Format
# ============================================================================

class GRPOPrompt(BaseModel):
    """
    GRPO format - prompt-only with multiple completions generated.

    Uses relative ranking within completion groups rather than absolute rewards.
    Particularly good for verifiable tasks.
    """
    prompt: str = Field(min_length=1)
    num_completions: Optional[int] = Field(
        default=4,
        ge=2,
        description="Number of completions to generate per prompt"
    )


# ============================================================================
# RL with Verifiable Rewards
# ============================================================================

class RLVerifiable(BaseModel):
    """
    RL format with verifiable ground truth.

    Useful for tasks with correct answers (e.g., math, accounting calculations).
    """
    prompt: str = Field(min_length=1)
    ground_truth: str = Field(min_length=1)
    verification_type: Literal[
        "numeric_match",
        "exact_match",
        "semantic_match",
        "contains",
        "regex"
    ] = "exact_match"
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context for verification"
    )


# ============================================================================
# Quality Metadata
# ============================================================================

class QualityMetrics(BaseModel):
    """Quality and performance metrics for generated samples."""
    quality_score: float = Field(ge=1.0, le=10.0)
    diversity_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    token_count: int = Field(ge=0)
    generation_cost: float = Field(ge=0.0)
    model_used: str
    generation_time: float = Field(ge=0.0, description="Time in seconds")
    regeneration_count: int = Field(default=0, ge=0)
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================================
# Dataset Container
# ============================================================================

class DatasetSample(BaseModel):
    """Container for a single dataset sample with metadata."""
    data: Union[
        SFTMessages,
        SFTCompletion,
        DPOPreference,
        DPOMessages,
        PPOPrompt,
        GRPOPrompt,
        RLVerifiable
    ]
    metrics: QualityMetrics
    source_hash: Optional[str] = Field(
        default=None,
        description="Hash of source material used"
    )


class Dataset(BaseModel):
    """Complete dataset with metadata."""
    samples: List[DatasetSample]
    data_type: Literal["sft", "dpo", "ppo", "grpo", "rl_verifiable"]
    total_samples: int
    average_quality: float
    total_cost: float
    generation_config: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.now)
    version: str = "0.1.0"

    @property
    def summary(self) -> Dict[str, Any]:
        """Get dataset summary statistics."""
        return {
            "total_samples": self.total_samples,
            "average_quality": self.average_quality,
            "total_cost": f"${self.total_cost:.2f}",
            "data_type": self.data_type,
            "created_at": self.created_at.isoformat(),
        }


# ============================================================================
# Configuration Models
# ============================================================================

class ModelConfig(BaseModel):
    """Configuration for different model roles."""
    generator: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Model for main data generation"
    )
    verifier: str = Field(
        default="gpt-4o-mini",
        description="Model for quality verification"
    )
    diversity: str = Field(
        default="qwen2.5:7b",
        description="Model for diversity checks (can be local)"
    )


class GenerationConfig(BaseModel):
    """Configuration for data generation."""
    num_samples: int = Field(ge=1)
    batch_size: int = Field(default=10, ge=1, le=50)
    quality_threshold: float = Field(default=6.0, ge=1.0, le=10.0)
    diversity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    max_cost: float = Field(default=20.0, ge=0.0)
    enable_human_review: bool = Field(default=False)
    max_retries_per_sample: int = Field(default=3, ge=1)
    domain_context: Optional[str] = Field(default=None)


# ============================================================================
# Type Aliases for convenience
# ============================================================================

TrainingDataFormat = Union[
    SFTMessages,
    SFTCompletion,
    DPOPreference,
    DPOMessages,
    PPOPrompt,
    GRPOPrompt,
    RLVerifiable
]
