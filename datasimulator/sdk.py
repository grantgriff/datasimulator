"""
Main SDK interface for DataSimulator.

Simple API for generating high-quality post-training datasets.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Literal, Dict, Any, List
from datetime import datetime

from .core.data_models import (
    ModelConfig,
    GenerationConfig,
    Dataset,
    DatasetSample
)
from .core.models.llm_client import ModelRouter
from .core.generators.sft_generator import SFTGenerator
from .core.generators.dpo_generator import DPOGenerator
from .core.generators.ppo_generator import PPOGenerator
from .core.generators.grpo_generator import GRPOGenerator
from .core.generators.rl_verifiable_generator import RLVerifiableGenerator
from .utils.cost_tracker import CostTracker
from .sources.document_loader import DocumentLoader, load_document
from .sources.base_loader import LoaderException

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeneratedDataset:
    """
    Container for generated dataset with export capabilities.

    Provides methods to save, analyze, and manipulate generated data.
    """

    def __init__(
        self,
        samples: List[DatasetSample],
        data_type: str,
        generation_config: Dict[str, Any],
        cost_tracker: CostTracker
    ):
        self.samples = samples
        self.data_type = data_type
        self.generation_config = generation_config
        self.cost_tracker = cost_tracker

        # Calculate statistics
        self._calculate_stats()

    def _calculate_stats(self):
        """Calculate dataset statistics."""
        if not self.samples:
            self.total_samples = 0
            self.average_quality = 0.0
            self.total_cost = self.cost_tracker.total_cost
            return

        self.total_samples = len(self.samples)
        quality_scores = [s.metrics.quality_score for s in self.samples]
        self.average_quality = sum(quality_scores) / len(quality_scores)
        self.total_cost = self.cost_tracker.total_cost

    def save(self, output_path: str, format: Literal["jsonl", "json"] = "jsonl"):
        """
        Save dataset to file.

        Args:
            output_path: Path to save file
            format: Output format ('jsonl' or 'json')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            self._save_jsonl(output_path)
        elif format == "json":
            self._save_json(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Dataset saved to {output_path}")
        print(f"\n✅ Dataset saved to: {output_path}")

    def _save_jsonl(self, output_path: Path):
        """Save as JSONL (one JSON object per line)."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in self.samples:
                # Extract just the data portion for training
                data_dict = sample.data.model_dump()
                f.write(json.dumps(data_dict, ensure_ascii=False) + '\n')

    def _save_json(self, output_path: Path):
        """Save as single JSON file with metadata."""
        dataset_dict = {
            "metadata": {
                "data_type": self.data_type,
                "total_samples": self.total_samples,
                "average_quality": self.average_quality,
                "total_cost": self.total_cost,
                "created_at": datetime.now().isoformat(),
                "generation_config": self.generation_config
            },
            "samples": [
                {
                    "data": sample.data.model_dump(),
                    "metrics": sample.metrics.model_dump()
                }
                for sample in self.samples
            ]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_dict, f, indent=2, ensure_ascii=False)

    def show_analytics(self):
        """Display dataset analytics."""
        print("\n" + "=" * 60)
        print("📊 DATASET ANALYTICS")
        print("=" * 60)
        print(f"Data Type:         {self.data_type.upper()}")
        print(f"Total Samples:     {self.total_samples}")
        print(f"Average Quality:   {self.average_quality:.2f}/10")

        if self.samples:
            quality_scores = [s.metrics.quality_score for s in self.samples]
            print(f"Quality Range:     {min(quality_scores):.1f} - {max(quality_scores):.1f}")

            total_tokens = sum(s.metrics.token_count for s in self.samples)
            avg_tokens = total_tokens / len(self.samples)
            print(f"Avg Tokens/Sample: {avg_tokens:.0f}")

        print(f"\nTotal Cost:        ${self.total_cost:.2f}")

        # Show cost breakdown
        cost_summary = self.cost_tracker.get_summary()
        print("\nCost Breakdown:")
        for operation, cost in cost_summary['cost_by_operation'].items():
            if cost > 0:
                percentage = (cost / self.total_cost) * 100
                print(f"  {operation.capitalize():12s}: ${cost:6.2f} ({percentage:5.1f}%)")

        print("=" * 60 + "\n")

    def filter_by_quality(self, min_score: float) -> 'GeneratedDataset':
        """
        Filter samples by minimum quality score.

        Args:
            min_score: Minimum quality score (1-10)

        Returns:
            New GeneratedDataset with filtered samples
        """
        filtered_samples = [
            s for s in self.samples
            if s.metrics.quality_score >= min_score
        ]

        logger.info(
            f"Filtered {len(self.samples)} samples to {len(filtered_samples)} "
            f"(quality >= {min_score})"
        )

        return GeneratedDataset(
            samples=filtered_samples,
            data_type=self.data_type,
            generation_config=self.generation_config,
            cost_tracker=self.cost_tracker
        )

    def sample_examples(self, n: int = 3):
        """
        Display sample examples from the dataset.

        Args:
            n: Number of examples to show
        """
        print(f"\n📝 Sample Examples (showing {min(n, len(self.samples))}):")
        print("=" * 60)

        for i, sample in enumerate(self.samples[:n], 1):
            print(f"\nExample {i}:")
            print(f"Quality: {sample.metrics.quality_score:.1f}/10")
            print(f"Data: {json.dumps(sample.data.model_dump(), indent=2)}")
            print("-" * 60)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DatasetSample:
        return self.samples[idx]


class DataSimulator:
    """
    Main SDK interface for generating post-training datasets.

    Simple API for creating high-quality SFT, DPO, PPO, GRPO, and RL data.

    Example:
        ```python
        sdk = DataSimulator(
            source="accounting_textbook.pdf",
            data_type="sft",
            models={
                "generator": "claude-3-5-sonnet-20241022",
                "verifier": "gpt-4o-mini",
            }
        )

        dataset = sdk.generate(num_samples=1000)
        dataset.save("output.jsonl")
        ```
    """

    def __init__(
        self,
        source: Optional[str] = None,
        data_type: Literal["sft", "dpo", "ppo", "grpo", "rl_verifiable"] = "sft",
        models: Optional[Dict[str, str]] = None,
        quality_threshold: float = 6.0,
        diversity_threshold: float = 0.85,
        max_cost: float = 20.0,
        batch_size: int = 20,
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        """
        Initialize DataSimulator.

        Args:
            source: Path to source document or URL (optional)
            data_type: Type of training data to generate
            models: Dictionary mapping roles to model names
                - generator: Main generation model
                - verifier: Quality verification model
                - diversity: Diversity checking model
            quality_threshold: Minimum quality score (1-10)
            diversity_threshold: Maximum similarity for diversity (0-1)
            max_cost: Maximum cost before prompting user (USD)
            batch_size: Number of samples per API call
            anthropic_api_key: Anthropic API key (or use ANTHROPIC_API_KEY env)
            openai_api_key: OpenAI API key (or use OPENAI_API_KEY env)
        """
        self.source = source
        self.data_type = data_type

        # Load source content if provided
        self.source_content = self._load_source() if source else None

        # Setup models
        model_config = models or {}
        generator_model = model_config.get("generator", "claude-3-5-sonnet-20241022")
        verifier_model = model_config.get("verifier", "gpt-4o-mini")
        diversity_model = model_config.get("diversity", "qwen2.5:7b")

        self.model_router = ModelRouter(
            generator_model=generator_model,
            verifier_model=verifier_model,
            diversity_model=diversity_model,
            anthropic_api_key=anthropic_api_key,
            openai_api_key=openai_api_key
        )

        # Setup cost tracking
        self.cost_tracker = CostTracker(max_cost=max_cost)

        # Store configuration
        self.quality_threshold = quality_threshold
        self.diversity_threshold = diversity_threshold
        self.batch_size = batch_size

        logger.info(f"DataSimulator initialized for {data_type.upper()} generation")
        logger.info(f"Generator: {generator_model}")
        logger.info(f"Verifier: {verifier_model}")

    def _load_source(self) -> str:
        """
        Load source content from file or URL.

        Supports:
        - Plain text files (.txt, .md)
        - PDF files (.pdf)
        - Word documents (.docx)
        - Images (.jpg, .png, etc.) via OCR
        - Web pages (http://, https://)
        - Google Docs (URLs or IDs)
        """
        if not self.source:
            return ""

        try:
            logger.info(f"Loading source: {self.source}")

            # Use unified document loader
            loader = DocumentLoader(self.source)
            content = loader.load()

            # Get metadata
            metadata = loader.get_metadata()
            logger.info(
                f"Loaded {len(content)} characters from {self.source} "
                f"(type: {metadata.get('method', 'unknown')})"
            )

            return content

        except LoaderException as e:
            logger.error(f"Error loading source: {e}")
            # Don't fail - just return empty string and continue
            logger.warning("Continuing without source content")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error loading source: {e}")
            logger.warning("Continuing without source content")
            return ""

    def generate(
        self,
        num_samples: int,
        domain_context: Optional[str] = None,
        enable_human_review: bool = False,
        show_progress: bool = True
    ) -> GeneratedDataset:
        """
        Generate training dataset.

        Args:
            num_samples: Number of samples to generate
            domain_context: Optional domain-specific context
            enable_human_review: Enable manual review of samples
            show_progress: Show generation progress

        Returns:
            GeneratedDataset object with samples and analytics
        """
        # Create generation config
        config = GenerationConfig(
            num_samples=num_samples,
            batch_size=self.batch_size,
            quality_threshold=self.quality_threshold,
            diversity_threshold=self.diversity_threshold,
            max_cost=self.cost_tracker.max_cost,
            enable_human_review=enable_human_review,
            domain_context=domain_context
        )

        # Create generator based on data type
        if self.data_type == "sft":
            generator = SFTGenerator(
                format_type="messages",  # Default to messages format
                model_router=self.model_router,
                cost_tracker=self.cost_tracker,
                config=config,
                source_content=self.source_content
            )
        elif self.data_type == "dpo":
            generator = DPOGenerator(
                format_type="preference",  # Default to preference format
                preference_strategy="quality",  # Default to quality-based preferences
                model_router=self.model_router,
                cost_tracker=self.cost_tracker,
                config=config,
                source_content=self.source_content
            )
        elif self.data_type == "ppo":
            generator = PPOGenerator(
                prompt_style="open_ended",  # Default to open-ended prompts
                model_router=self.model_router,
                cost_tracker=self.cost_tracker,
                config=config,
                source_content=self.source_content
            )
        elif self.data_type == "grpo":
            generator = GRPOGenerator(
                num_completions=4,  # Default to 4 completions per prompt
                task_type="verifiable",  # Default to verifiable tasks
                model_router=self.model_router,
                cost_tracker=self.cost_tracker,
                config=config,
                source_content=self.source_content
            )
        elif self.data_type == "rl_verifiable":
            generator = RLVerifiableGenerator(
                verification_type="exact_match",  # Default verification type
                model_router=self.model_router,
                cost_tracker=self.cost_tracker,
                config=config,
                source_content=self.source_content
            )
        else:
            raise ValueError(
                f"Unknown data type: {self.data_type}. "
                f"Supported types: sft, dpo, ppo, grpo, rl_verifiable"
            )

        # Generate samples (using asyncio)
        import asyncio
        samples = asyncio.run(generator.generate(num_samples, show_progress))

        # Create dataset
        dataset = GeneratedDataset(
            samples=samples,
            data_type=self.data_type,
            generation_config=config.model_dump(),
            cost_tracker=self.cost_tracker
        )

        return dataset
