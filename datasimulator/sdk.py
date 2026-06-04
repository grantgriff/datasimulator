"""
Main SDK interface for DataSimulator.

Simple API for generating high-quality post-training datasets.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Literal, Dict, Any, List, Union
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
from .core.generators.verifiable_qa_generator import VerifiableQAGenerator
from .core.generators.ranked_generator import RankedGenerator
from .core.generators.full_generator import FullGenerator
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
        """
        Save as JSONL (one JSON object per line) plus a sidecar metadata
        file (one metadata JSON per line, in the same order).

        Training file:  <name>.jsonl              — clean {messages: ...} etc.
        Sidecar:        <name>.metadata.jsonl     — per-sample quality score,
                                                    topic, cost, token count,
                                                    model used, idx for joining
                                                    back to the training file.
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in self.samples:
                # Extract just the data portion for training
                data_dict = sample.data.model_dump()
                f.write(json.dumps(data_dict, ensure_ascii=False) + '\n')

        sidecar_path = output_path.with_suffix(".metadata.jsonl")
        with open(sidecar_path, 'w', encoding='utf-8') as f:
            for idx, sample in enumerate(self.samples):
                m = sample.metrics
                f.write(json.dumps({
                    "idx": idx,
                    "quality_score": m.quality_score,
                    "topic": m.topic,
                    "subtopic": m.subtopic,
                    "token_count": m.token_count,
                    "generation_cost": m.generation_cost,
                    "model_used": m.model_used,
                    "regeneration_count": m.regeneration_count,
                    "generation_time": m.generation_time,
                    "timestamp": m.timestamp.isoformat(),
                }, ensure_ascii=False) + '\n')
        logger.info(f"Metadata sidecar saved to {sidecar_path}")

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
            # Defaults to Gemini Flash for generator/verifier/diversity, so
            # only GOOGLE_API_KEY is required. Override `models` if you want
            # to mix providers.
        )

        dataset = sdk.generate(num_samples=1000)
        dataset.save("output.jsonl")
        ```
    """

    def __init__(
        self,
        source: Optional[Union[str, List[str]]] = None,
        data_type: Literal["sft", "dpo", "verifiable_qa", "ranked", "full"] = "sft",
        models: Optional[Dict[str, str]] = None,
        quality_threshold: float = 6.0,
        diversity_threshold: float = 0.85,
        max_cost: float = 20.0,
        batch_size: int = 10,
        parallel_batches: int = 4,
        interactive: bool = False,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 20,
        enable_planning: bool = True,
        ranked_config: Optional[Dict[str, Any]] = None,
        google_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        do_api_key: Optional[str] = None,
        cloudflare_api_key: Optional[str] = None,
        cloudflare_account_id: Optional[str] = None,
        progress_callback: Optional[Any] = None,
    ):
        """
        Initialize DataSimulator.

        Args:
            source: Path to source document(s) or URL(s) - can be single string or list of strings
            data_type: Type of training data to generate
            models: Dictionary mapping roles to model names
                - generator: Main generation model
                - verifier: Quality verification model
                - diversity: Diversity checking model
            quality_threshold: Minimum quality score (1-10)
            diversity_threshold: Maximum similarity for diversity (0-1)
            max_cost: Maximum cost before prompting user (USD)
            batch_size: Number of samples per API call (default 10).
                Each batch is one LLM call returning a JSON array of N
                records. Higher = fewer round trips but risks hitting
                model output ceilings (most models cap actual output
                around 16K tokens). For dense content (1KB+ per record),
                stay at 10 or below. See CLAUDE.md.
            parallel_batches: Number of batches to generate simultaneously (default: 4)
            interactive: Whether to prompt the user when the cost cap is hit
                (default False — safe for programmatic / CLI integrations).
                Set True for interactive Python sessions.
            checkpoint_dir: Directory to save checkpoints (optional)
            checkpoint_interval: Save checkpoint every N samples (default: 20)
            enable_planning: Use Gemini to analyze sources and create generation plan
            ranked_config: Configuration for data_type="ranked" and "full":
                - num_responses (int, default 4): responses per prompt
                - quality_spread ("wide"|"narrow", default "wide"): target gap
                  between best/worst response scores
            google_api_key: Google API key for Gemini planning (or use GOOGLE_API_KEY env)
            anthropic_api_key: Anthropic API key (or use ANTHROPIC_API_KEY env)
            openai_api_key: OpenAI API key (or use OPENAI_API_KEY env)
            openrouter_api_key: OpenRouter API key (or use OPENROUTER_API_KEY env).
                Activates the `openrouter/<provider>/<model>` model prefix.
            do_api_key: DigitalOcean Serverless Inference key (or use
                DO_INFERENCE_KEY env). Activates the `do/<model>` prefix.
            cloudflare_api_key: Cloudflare API token (or use CLOUDFLARE_API_TOKEN
                env). Activates the `cf/<model>` prefix for Workers AI.
            cloudflare_account_id: Cloudflare account ID (or use
                CLOUDFLARE_ACCOUNT_ID env). Required when using `cf/...` —
                CF's endpoint URL embeds the account ID.
            progress_callback: Optional callable invoked with a dict on each
                lifecycle event (generation_started, batch_completed,
                checkpoint_saved, cost_limit_reached, generation_completed).
                Can be sync or async. Exceptions are logged and swallowed —
                a buggy callback will not break generation. See INTEGRATION.md
                for the event payload shape.
        """
        self.source = source
        self.data_type = data_type
        self.batch_size = batch_size
        self.parallel_batches = parallel_batches
        self.quality_threshold = quality_threshold
        self.diversity_threshold = diversity_threshold
        self.ranked_config = ranked_config or {}
        self.progress_callback = progress_callback

        # Store source files list for planner
        self.source_files = [source] if isinstance(source, str) else (source if source else [])

        # Store checkpoint configuration
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.checkpoint_interval = checkpoint_interval
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Checkpointing enabled: {self.checkpoint_dir} (every {checkpoint_interval} samples)")

        # Load source content if provided (both combined and per-file)
        self.source_content = None
        self.source_content_by_file = {}  # NEW: Per-file content mapping
        if source:
            self.source_content, self.source_content_by_file = self._load_source()

        # Setup models
        model_config = models or {}
        # Default to Gemini via OpenRouter: Flash for generation/verification/
        # diversity (cheap + fast), Pro for the planner (better long-context
        # reasoning over the source material). Override via the `models=` dict.
        generator_model = model_config.get("generator", "openrouter/google/gemini-flash-latest")
        verifier_model = model_config.get("verifier", "openrouter/google/gemini-flash-latest")
        diversity_model = model_config.get("diversity", "openrouter/google/gemini-flash-latest")

        self.model_router = ModelRouter(
            generator_model=generator_model,
            verifier_model=verifier_model,
            diversity_model=diversity_model,
            anthropic_api_key=anthropic_api_key,
            openai_api_key=openai_api_key,
            google_api_key=google_api_key,
            openrouter_api_key=openrouter_api_key,
            do_api_key=do_api_key,
            cloudflare_api_key=cloudflare_api_key,
            cloudflare_account_id=cloudflare_account_id,
        )

        # Setup cost tracking with interactive mode
        self.cost_tracker = CostTracker(max_cost=max_cost, interactive=interactive)
        if not interactive:
            logger.info(f"Non-interactive mode: will not prompt when cost limit reached")

        # Setup planning (optional Gemini integration)
        self.enable_planning = enable_planning
        self.planner = None
        if enable_planning:
            try:
                from .planning import GeminiPlanner
                planner_model = model_config.get("planner", "openrouter/google/gemini-pro-latest")
                self.planner = GeminiPlanner(
                    model=planner_model,
                    anthropic_api_key=anthropic_api_key,
                    openai_api_key=openai_api_key,
                    google_api_key=google_api_key,
                    openrouter_api_key=openrouter_api_key,
                    do_api_key=do_api_key,
                    cloudflare_api_key=cloudflare_api_key,
                    cloudflare_account_id=cloudflare_account_id,
                )
                logger.info(f"Gemini planning enabled (model={planner_model})")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini planner: {e}")
                logger.warning("Continuing without planning layer")
                self.enable_planning = False

        # Store configuration
        self.quality_threshold = quality_threshold
        self.diversity_threshold = diversity_threshold
        self.batch_size = batch_size

        logger.info(f"DataSimulator initialized for {data_type.upper()} generation")
        logger.info(f"Generator: {generator_model}")
        logger.info(f"Verifier: {verifier_model}")

    def _validate_topic_emphasis(
        self,
        topic_emphasis: Optional[Dict[str, float]]
    ) -> Optional[Dict[str, float]]:
        """
        Validate and normalize topic_emphasis input.

        Rejects bad shapes; warns when planning is disabled (in which case
        the emphasis cannot be applied and is dropped).
        """
        if topic_emphasis is None:
            return None

        if not isinstance(topic_emphasis, dict) or not topic_emphasis:
            raise ValueError(
                "topic_emphasis must be a non-empty dict of {topic: weight}"
            )

        for topic, weight in topic_emphasis.items():
            if not isinstance(topic, str) or not topic.strip():
                raise ValueError(f"topic_emphasis keys must be non-empty strings (got {topic!r})")
            if not isinstance(weight, (int, float)) or weight <= 0 or weight > 1:
                raise ValueError(
                    f"topic_emphasis weight for {topic!r} must be in (0, 1], got {weight}"
                )

        total = sum(topic_emphasis.values())
        # Allow tiny floating-point slack above 1.0
        if total > 1.0 + 1e-6:
            raise ValueError(
                f"topic_emphasis weights must sum to <= 1.0, got {total:.4f}"
            )

        if not self.enable_planning:
            logger.warning(
                "topic_emphasis was provided but enable_planning=False; "
                "emphasis will be ignored. Pass enable_planning=True to apply."
            )
            return None

        return topic_emphasis

    @staticmethod
    def _looks_like_raw_text(s: str) -> bool:
        """
        Decide whether a `source` string is raw text content vs. a path/URL.

        Heuristic: treat as raw text if it contains a newline OR is longer
        than a sensible filename/URL. This lets callers (e.g. Posty's CLI)
        pass already-loaded content directly without writing to a temp file.
        """
        if "\n" in s:
            return True
        # URLs and file paths are virtually never > 500 chars
        if len(s) > 500:
            return True
        return False

    def _load_source(self) -> tuple[str, Dict[str, str]]:
        """
        Load source content from file(s), URL(s), or raw text.

        Returns:
            Tuple of (combined_content, content_by_file)
            - combined_content: All sources combined into one string
            - content_by_file: Dict mapping file path / synthetic key to its content

        Supports:
        - Single source: string path / URL / raw text
        - Multiple sources: list of paths / URLs / raw text blobs
        - Plain text files (.txt, .md)
        - PDF files (.pdf)
        - Word documents (.docx)
        - Images (.jpg, .png, etc.) via OCR
        - Web pages (http://, https://)
        - Google Docs (URLs or IDs)
        - Raw text strings (any string containing newlines or >500 chars)
        """
        if not self.source:
            return "", {}

        # Handle multiple sources
        sources = [self.source] if isinstance(self.source, str) else self.source

        logger.info(f"Loading {len(sources)} source(s)...")
        combined_content = []
        content_by_file = {}  # NEW: Store content per file
        successful_loads = 0

        for i, source in enumerate(sources, 1):
            try:
                # Raw text path — caller passed already-loaded content
                if self._looks_like_raw_text(source):
                    content = source
                    file_key = f"inline_text_{i}"
                    logger.info(
                        f"  [{i}/{len(sources)}] Inline text ({len(content)} chars)"
                    )
                    content_by_file[file_key] = content
                    if len(sources) > 1:
                        combined_content.append(f"\n\n=== Source {i}: {file_key} ===\n\n{content}")
                    else:
                        combined_content.append(content)
                    successful_loads += 1
                    continue

                logger.info(f"  [{i}/{len(sources)}] Loading: {source}")

                # Use unified document loader
                loader = DocumentLoader(source)
                content = loader.load()

                # Get metadata
                metadata = loader.get_metadata()
                logger.info(
                    f"    ✓ Loaded {len(content)} characters "
                    f"(type: {metadata.get('method', 'unknown')})"
                )

                # Store per-file content (use filename as key)
                file_key = Path(source).name if "/" in source or "\\" in source else source
                content_by_file[file_key] = content

                # Add source separator for multiple files
                if len(sources) > 1:
                    combined_content.append(f"\n\n=== Source {i}: {source} ===\n\n{content}")
                else:
                    combined_content.append(content)

                successful_loads += 1

            except LoaderException as e:
                logger.error(f"    ✗ Error loading {source}: {e}")
            except Exception as e:
                logger.error(f"    ✗ Unexpected error loading {source}: {e}")

        if successful_loads == 0:
            logger.warning("Failed to load any sources, continuing without source content")
            return "", {}

        full_content = "\n\n".join(combined_content)
        logger.info(
            f"✓ Successfully loaded {successful_loads}/{len(sources)} source(s) "
            f"({len(full_content)} total characters)"
        )

        return full_content, content_by_file

    def generate(
        self,
        num_samples: int,
        domain_context: Optional[str] = None,
        enable_human_review: bool = False,
        show_progress: bool = True,
        topic_emphasis: Optional[Dict[str, float]] = None
    ) -> GeneratedDataset:
        """
        Generate training dataset.

        Args:
            num_samples: Number of samples to generate
            domain_context: Optional domain-specific context
            enable_human_review: Enable manual review of samples
            show_progress: Show generation progress
            topic_emphasis: Optional dict mapping topic strings to weights (0-1).
                When provided alongside enable_planning=True, biases the Gemini
                planner toward allocating more batches to the weighted topics.
                Values must sum to <= 1.0 (remainder distributed across other
                topics naturally extracted from sources). Ignored with a warning
                when enable_planning=False.

        Returns:
            GeneratedDataset object with samples and analytics
        """
        # Validate topic_emphasis
        topic_emphasis = self._validate_topic_emphasis(topic_emphasis)

        # Create generation plan if planning is enabled
        generation_plan = None
        if self.enable_planning and self.planner and self.source_content:
            logger.info("Creating generation plan with Gemini...")
            import asyncio
            generation_plan = asyncio.run(
                self.planner.create_generation_plan(
                    source_content=self.source_content,
                    total_samples=num_samples,
                    data_type=self.data_type,
                    source_files=self.source_files,
                    batch_size=self.batch_size,
                    topic_emphasis=topic_emphasis
                )
            )
            logger.info(f"✓ Plan created: {generation_plan.get('num_batches', 0)} batches")

        # Create generation config
        config = GenerationConfig(
            num_samples=num_samples,
            batch_size=self.batch_size,
            parallel_batches=self.parallel_batches,
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
                source_content=self.source_content,
                source_content_by_file=self.source_content_by_file
            )
        elif self.data_type == "dpo":
            generator = DPOGenerator(
                format_type="preference",  # Default to preference format
                preference_strategy="quality",  # Default to quality-based preferences
                model_router=self.model_router,
                cost_tracker=self.cost_tracker,
                config=config,
                source_content=self.source_content,
                source_content_by_file=self.source_content_by_file
            )
        elif self.data_type == "verifiable_qa":
            generator = VerifiableQAGenerator(
                verification_type="exact_match",  # Default verification type
                model_router=self.model_router,
                cost_tracker=self.cost_tracker,
                config=config,
                source_content=self.source_content,
                source_content_by_file=self.source_content_by_file
            )
        elif self.data_type == "ranked":
            generator = RankedGenerator(
                num_responses=self.ranked_config.get("num_responses", 4),
                quality_spread=self.ranked_config.get("quality_spread", "wide"),
                model_router=self.model_router,
                cost_tracker=self.cost_tracker,
                config=config,
                source_content=self.source_content,
                source_content_by_file=self.source_content_by_file
            )
        elif self.data_type == "full":
            generator = FullGenerator(
                num_responses=self.ranked_config.get("num_responses", 4),
                quality_spread=self.ranked_config.get("quality_spread", "wide"),
                model_router=self.model_router,
                cost_tracker=self.cost_tracker,
                config=config,
                source_content=self.source_content,
                source_content_by_file=self.source_content_by_file
            )
        else:
            raise ValueError(
                f"Unknown data type: {self.data_type}. "
                f"Supported types: sft, dpo, verifiable_qa, ranked, full"
            )

        # Wire the progress callback through to the generator (all generators
        # inherit it from BaseGenerator, so we set it directly rather than
        # threading it through five constructors).
        generator.progress_callback = self.progress_callback

        # Generate samples (using asyncio) with checkpointing and optional plan
        import asyncio
        samples = asyncio.run(
            generator.generate(
                num_samples,
                show_progress,
                checkpoint_dir=self.checkpoint_dir,
                checkpoint_interval=self.checkpoint_interval,
                generation_plan=generation_plan  # Pass plan to generator
            )
        )

        # Create dataset
        dataset = GeneratedDataset(
            samples=samples,
            data_type=self.data_type,
            generation_config=config.model_dump(),
            cost_tracker=self.cost_tracker
        )

        return dataset
