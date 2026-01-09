"""
Quality validators for training data.

Validates data format, content quality, and correctness.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Callable
from ..core.data_models import (
    SFTMessages,
    SFTCompletion,
    DPOPreference,
    TrainingDataFormat
)

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates training data samples.

    Checks format correctness, content quality, and data integrity.
    """

    def __init__(self, data_type: str = "sft"):
        """
        Initialize validator.

        Args:
            data_type: Type of data to validate (sft, dpo, etc.)
        """
        self.data_type = data_type

    def validate_sample(self, sample: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate a single sample.

        Args:
            sample: Sample to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Format validation
        if not self._validate_format(sample):
            errors.append("Invalid data format")

        # Content validation
        content_errors = self._validate_content(sample)
        errors.extend(content_errors)

        # Length validation
        length_errors = self._validate_lengths(sample)
        errors.extend(length_errors)

        # Specific validations based on data type
        if self.data_type == "sft":
            sft_errors = self._validate_sft(sample)
            errors.extend(sft_errors)
        elif self.data_type == "dpo":
            dpo_errors = self._validate_dpo(sample)
            errors.extend(dpo_errors)

        is_valid = len(errors) == 0

        if not is_valid:
            logger.debug(f"Validation failed: {', '.join(errors)}")

        return is_valid, errors

    def _validate_format(self, sample: Dict[str, Any]) -> bool:
        """Validate basic data format."""
        if not isinstance(sample, dict):
            return False

        # Check for required keys
        if "data" not in sample and "messages" not in sample and "prompt" not in sample:
            return False

        return True

    def _validate_content(self, sample: Dict[str, Any]) -> List[str]:
        """Validate content quality."""
        errors = []

        data = sample.get("data", sample)

        # Check for empty content
        if "messages" in data:
            for msg in data["messages"]:
                content = msg.get("content", "").strip()
                if not content:
                    errors.append("Empty message content found")

        elif "prompt" in data or "completion" in data:
            prompt = data.get("prompt", "").strip()
            completion = data.get("completion", "").strip()

            if not prompt:
                errors.append("Empty prompt")
            if not completion:
                errors.append("Empty completion")

        # Check for placeholder text
        placeholder_patterns = [
            r"\[.*?\]",  # [placeholder]
            r"\{.*?\}",  # {placeholder}
            r"TODO",
            r"FIXME",
            r"XXX",
            r"\.\.\.+",  # Multiple dots
        ]

        text_content = str(data)
        for pattern in placeholder_patterns:
            if re.search(pattern, text_content):
                errors.append(f"Placeholder text found: {pattern}")
                break

        return errors

    def _validate_lengths(self, sample: Dict[str, Any]) -> List[str]:
        """Validate content lengths."""
        errors = []

        data = sample.get("data", sample)

        # Minimum length requirements
        min_prompt_length = 10
        min_completion_length = 20

        if "messages" in data:
            for msg in data["messages"]:
                content = msg.get("content", "")
                role = msg.get("role")

                if role == "user" and len(content) < min_prompt_length:
                    errors.append(f"User message too short ({len(content)} chars)")

                if role == "assistant" and len(content) < min_completion_length:
                    errors.append(f"Assistant message too short ({len(content)} chars)")

        elif "prompt" in data:
            prompt = data.get("prompt", "")
            if len(prompt) < min_prompt_length:
                errors.append(f"Prompt too short ({len(prompt)} chars)")

        if "completion" in data:
            completion = data.get("completion", "")
            if len(completion) < min_completion_length:
                errors.append(f"Completion too short ({len(completion)} chars)")

        # Maximum length check (avoid extremely long samples)
        max_total_length = 10000

        total_length = len(str(data))
        if total_length > max_total_length:
            errors.append(f"Sample too long ({total_length} chars)")

        return errors

    def _validate_sft(self, sample: Dict[str, Any]) -> List[str]:
        """Validate SFT-specific requirements."""
        errors = []

        data = sample.get("data", sample)

        if "messages" in data:
            messages = data["messages"]

            # Check message alternation
            prev_role = None
            for msg in messages:
                role = msg.get("role")

                # Should have system, user, assistant in some order
                if role not in ["system", "user", "assistant"]:
                    errors.append(f"Invalid role: {role}")

                # Assistant shouldn't follow assistant directly
                if role == "assistant" and prev_role == "assistant":
                    errors.append("Consecutive assistant messages")

                prev_role = role

            # Last message should be assistant
            if messages and messages[-1].get("role") != "assistant":
                errors.append("Last message should be from assistant")

        return errors

    def _validate_dpo(self, sample: Dict[str, Any]) -> List[str]:
        """Validate DPO-specific requirements."""
        errors = []

        data = sample.get("data", sample)

        # Must have prompt, chosen, rejected
        if "prompt" not in data:
            errors.append("Missing prompt")
        if "chosen" not in data:
            errors.append("Missing chosen response")
        if "rejected" not in data:
            errors.append("Missing rejected response")

        # Chosen and rejected should be different
        if "chosen" in data and "rejected" in data:
            if data["chosen"] == data["rejected"]:
                errors.append("Chosen and rejected responses are identical")

            # Rejected should ideally be lower quality
            # (could add more sophisticated checks here)

        return errors

    def batch_validate(
        self,
        samples: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Validate a batch of samples.

        Args:
            samples: List of samples to validate

        Returns:
            Tuple of (valid_samples, invalid_samples_with_errors)
        """
        valid = []
        invalid = []

        for sample in samples:
            is_valid, errors = self.validate_sample(sample)

            if is_valid:
                valid.append(sample)
            else:
                sample["validation_errors"] = errors
                invalid.append(sample)

        logger.info(
            f"Validation: {len(valid)}/{len(samples)} passed "
            f"({len(valid)/len(samples)*100:.1f}%)"
        )

        return valid, invalid


class ContentFilter:
    """
    Filter samples based on content rules.

    Removes samples with inappropriate content, PII, etc.
    """

    def __init__(self):
        """Initialize content filter."""
        # Patterns for filtering
        self.forbidden_patterns = [
            # PII patterns
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{16}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email

            # Offensive language (basic list - expand as needed)
            # Add your patterns here
        ]

    def filter_sample(self, sample: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Filter a single sample.

        Args:
            sample: Sample to filter

        Returns:
            Tuple of (should_keep, rejection_reason)
        """
        data = sample.get("data", sample)
        text_content = str(data)

        # Check forbidden patterns
        for pattern in self.forbidden_patterns:
            if re.search(pattern, text_content, re.IGNORECASE):
                return False, f"Contains forbidden pattern: {pattern}"

        # Check for excessively repetitive content
        if self._is_repetitive(text_content):
            return False, "Content is excessively repetitive"

        return True, None

    def _is_repetitive(self, text: str, threshold: float = 0.5) -> bool:
        """Check if text is excessively repetitive."""
        words = text.lower().split()

        if len(words) < 20:
            return False

        # Count unique words
        unique_ratio = len(set(words)) / len(words)

        return unique_ratio < threshold

    def batch_filter(
        self,
        samples: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Filter a batch of samples.

        Args:
            samples: List of samples to filter

        Returns:
            Tuple of (kept_samples, filtered_samples_with_reasons)
        """
        kept = []
        filtered = []

        for sample in samples:
            should_keep, reason = self.filter_sample(sample)

            if should_keep:
                kept.append(sample)
            else:
                sample["filter_reason"] = reason
                filtered.append(sample)

        if filtered:
            logger.info(f"Filtered {len(filtered)} samples: {reason}")

        return kept, filtered
