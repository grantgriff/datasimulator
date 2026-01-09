"""
Example: RL with Verifiable Rewards Data Generation

This example demonstrates generating training data with ground truth answers
for reinforcement learning with automatic verification.

Perfect for:
- Mathematical calculations
- Factual questions
- Accounting problems
- Coding challenges
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasimulator import DataSimulator


def main():
    """Generate RL verifiable training data."""

    print("=" * 60)
    print("DataSimulator - RL Verifiable Example")
    print("=" * 60)

    # Create sample source with calculations
    os.makedirs("examples/sample_data", exist_ok=True)

    source_path = "examples/sample_data/rl_demo.txt"
    with open(source_path, "w") as f:
        f.write("""
        Financial Calculations

        Common Formulas:

        1. Simple Interest = Principal × Rate × Time

        2. Compound Interest = Principal × (1 + Rate)^Time - Principal

        3. Return on Investment (ROI) = (Gain - Cost) / Cost × 100%

        4. Net Present Value (NPV) = Σ (Cash Flow / (1 + r)^t)

        5. Break-Even Point = Fixed Costs / (Price - Variable Cost)

        6. Depreciation (Straight-Line) = (Cost - Salvage Value) / Useful Life

        Example Problems:
        - If you invest $10,000 at 5% for 3 years, what is the simple interest?
        - A company has $100,000 in fixed costs, sells products for $50, and variable cost is $30. What is the break-even point in units?
        """)

    print(f"\n📄 Created source: {source_path}")

    # Initialize SDK for RL Verifiable
    print("\n🔧 Initializing RL Verifiable generator...")

    sdk = DataSimulator(
        source=source_path,
        data_type="rl_verifiable",  # ← RL Verifiable format!
        models={
            "generator": "claude-3-5-sonnet-20241022",
            "verifier": "gpt-4o-mini",
        },
        quality_threshold=7.0,  # Higher threshold for verifiable data
        max_cost=2.0,
        batch_size=5,
    )

    # Generate verifiable problems
    print("\n🚀 Generating RL verifiable problems...")
    print("   Format: Each sample has:")
    print("   • Prompt (problem/question)")
    print("   • Ground Truth (correct answer)")
    print("   • Verification Type (how to verify)")
    print()

    dataset = sdk.generate(
        num_samples=10,
        domain_context="Generate financial calculation problems with exact numerical answers"
    )

    # Show samples
    print("\n" + "="*60)
    print("📝 SAMPLE VERIFIABLE PROBLEMS")
    print("="*60)

    if dataset.samples:
        import json

        for i, sample in enumerate(dataset.samples[:3], 1):
            data = sample.data.model_dump()

            print(f"\n🔢 Problem {i}:")
            print(f"   Prompt: {data.get('prompt', 'N/A')}")
            print(f"   Ground Truth: {data.get('ground_truth', 'N/A')}")
            print(f"   Verification: {data.get('verification_type', 'N/A')}")

            if 'metadata' in data and data['metadata']:
                print(f"   Metadata: {data['metadata']}")

    # Show analytics
    dataset.show_analytics()

    # Save
    output_path = "outputs/rl_verifiable_finance.jsonl"
    os.makedirs("outputs", exist_ok=True)
    dataset.save(output_path)

    print(f"\n✓ Generated {len(dataset)} RL verifiable samples")
    print(f"💾 Saved to: {output_path}")

    print("\n" + "="*60)
    print("💡 RL Verifiable Training Tips:")
    print("="*60)
    print("  • Automatic verification of model outputs")
    print("  • Perfect for math, calculations, factual questions")
    print("  • Reward = 1 if answer matches ground truth, 0 otherwise")
    print("  • Verification types available:")
    print("    - numeric_match: For numbers (handles formatting)")
    print("    - exact_match: For exact strings")
    print("    - semantic_match: For meaning-based matching")
    print("    - contains: For multi-part answers")
    print("    - regex: For pattern matching")


if __name__ == "__main__":
    main()
