"""
Example: Advanced quality control and diversity checking

This example demonstrates Phase 3 features:
- Advanced quality scoring with dimension breakdown
- Diversity checking with semantic similarity
- Quality filtering and validation
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasimulator import (
    DataSimulator,
    QualityScorer,
    DiversityChecker,
    DataValidator,
    ContentFilter
)


def main():
    """Demonstrate quality control features."""

    print("=" * 60)
    print("DataSimulator - Quality Control Example")
    print("=" * 60)

    # Create sample source content
    os.makedirs("examples/sample_data", exist_ok=True)

    source_path = "examples/sample_data/quality_demo.txt"
    with open(source_path, "w") as f:
        f.write("""
        Financial Statement Analysis

        The income statement shows a company's profitability over time.
        Key metrics include revenue, expenses, and net income.

        The balance sheet provides a snapshot of assets, liabilities, and equity.

        Cash flow statements track the movement of cash in and out of the business.
        """)

    print(f"\n📄 Created source: {source_path}")

    # Initialize SDK with quality controls
    print("\n🔧 Initializing SDK with quality controls...")

    sdk = DataSimulator(
        source=source_path,
        data_type="sft",
        models={
            "generator": "claude-3-5-sonnet-20241022",
            "verifier": "gpt-4o-mini",
        },
        quality_threshold=7.0,  # Higher threshold for demo
        max_cost=2.0,
        batch_size=5,
    )

    # Generate samples
    print("\n🚀 Generating samples with quality scoring...")
    print("   Quality threshold: 7.0/10")
    print("   Diversity checking enabled")
    print()

    dataset = sdk.generate(
        num_samples=10,
        domain_context="Generate questions about financial statement analysis"
    )

    # Show enhanced analytics
    print("\n" + "="*60)
    print("📊 QUALITY & DIVERSITY ANALYSIS")
    print("="*60)

    dataset.show_analytics()

    # Demonstrate standalone quality checking
    print("\n" + "="*60)
    print("🔍 STANDALONE QUALITY CHECKS")
    print("="*60)

    if dataset.samples:
        sample = dataset.samples[0]

        print("\n1. Data Validation:")
        validator = DataValidator(data_type="sft")
        is_valid, errors = validator.validate_sample(sample.model_dump())

        if is_valid:
            print("   ✓ Sample passes validation")
        else:
            print(f"   ✗ Validation errors: {errors}")

        print("\n2. Content Filtering:")
        content_filter = ContentFilter()
        should_keep, reason = content_filter.filter_sample(sample.model_dump())

        if should_keep:
            print("   ✓ Sample passes content filter")
        else:
            print(f"   ✗ Filtered: {reason}")

        print("\n3. Diversity Analysis:")
        diversity_checker = DiversityChecker()

        # Check diversity of first 3 samples
        if len(dataset.samples) >= 3:
            texts = []
            for s in dataset.samples[:3]:
                # Extract text for comparison
                data = s.data.model_dump()
                if "messages" in data:
                    texts.append(data["messages"][-1]["content"])
                else:
                    texts.append(str(data))

            # Check pairwise similarity
            print("   Pairwise similarities:")
            for i in range(len(texts)):
                for j in range(i+1, len(texts)):
                    sim = diversity_checker.compute_similarity(texts[i], texts[j])
                    print(f"   Sample {i} ↔ Sample {j}: {sim:.3f}")

    # Save high-quality samples
    print("\n" + "="*60)
    print("💾 SAVING RESULTS")
    print("="*60)

    # Filter to only highest quality (8+)
    high_quality = dataset.filter_by_quality(min_score=8.0)

    if len(high_quality) > 0:
        output_path = "outputs/high_quality_samples.jsonl"
        os.makedirs("outputs", exist_ok=True)
        high_quality.save(output_path)
        print(f"\n✓ Saved {len(high_quality)} high-quality samples to {output_path}")
    else:
        print("\n⚠️  No samples met 8.0+ quality threshold")

    # Save all samples
    output_path = "outputs/quality_demo.jsonl"
    dataset.save(output_path)
    print(f"✓ Saved all {len(dataset)} samples to {output_path}")

    print("\n" + "="*60)
    print("✅ Quality control demo complete!")
    print("="*60)


if __name__ == "__main__":
    main()
