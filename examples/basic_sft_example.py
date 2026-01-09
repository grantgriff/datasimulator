"""
Basic example: Generate SFT training data

This example demonstrates the simplest usage of DataSimulator
to generate supervised fine-tuning data.
"""

import os
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasimulator import DataSimulator


def main():
    """Generate a small SFT dataset."""

    print("=" * 60)
    print("DataSimulator - Basic SFT Example")
    print("=" * 60)

    # Initialize SDK
    sdk = DataSimulator(
        # No source document - will use domain context
        data_type="sft",

        # Model configuration
        models={
            "generator": "claude-3-5-sonnet-20241022",  # Main generation
            "verifier": "gpt-4o-mini",  # Quality verification
        },

        # Quality settings
        quality_threshold=6.0,  # Minimum quality score (1-10)

        # Cost controls
        max_cost=5.0,  # Stop at $5 for this example
        batch_size=5,  # Generate 5 samples per batch
    )

    # Generate dataset
    print("\nGenerating 10 accounting examples...\n")

    dataset = sdk.generate(
        num_samples=10,
        domain_context="Generate practical accounting questions and answers about basic concepts like debits, credits, journal entries, and financial statements.",
        show_progress=True
    )

    # Show analytics
    dataset.show_analytics()

    # Display some examples
    dataset.sample_examples(n=2)

    # Save to file
    output_path = "outputs/basic_sft_example.jsonl"
    os.makedirs("outputs", exist_ok=True)
    dataset.save(output_path)

    print(f"\n✅ Done! Generated {len(dataset)} samples")
    print(f"💾 Saved to: {output_path}")


if __name__ == "__main__":
    main()
