"""
Example: DPO (Direct Preference Optimization) Data Generation

This example demonstrates generating preference pairs for DPO training.
Each sample has a prompt with both a "chosen" (better) and "rejected" (worse) response.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasimulator import DataSimulator


def main():
    """Generate DPO preference pairs."""

    print("=" * 60)
    print("DataSimulator - DPO Example")
    print("=" * 60)

    # Create sample source
    os.makedirs("examples/sample_data", exist_ok=True)

    source_path = "examples/sample_data/dpo_demo.txt"
    with open(source_path, "w") as f:
        f.write("""
        Customer Service Best Practices

        Professional Response Guidelines:
        1. Always greet the customer warmly
        2. Listen carefully to their concerns
        3. Provide clear, accurate information
        4. Offer specific solutions
        5. End with a polite closing

        Common Issues:
        - Product questions
        - Return policies
        - Shipping inquiries
        - Technical support
        """)

    print(f"\n📄 Created source: {source_path}")

    # Initialize SDK for DPO
    print("\n🔧 Initializing DPO generator...")

    sdk = DataSimulator(
        source=source_path,
        data_type="dpo",  # ← DPO format!
        models={
            "generator": "claude-3-5-sonnet-20241022",
            "verifier": "gpt-4o-mini",
        },
        quality_threshold=6.5,
        max_cost=2.0,
        batch_size=5,
    )

    # Generate preference pairs
    print("\n🚀 Generating DPO preference pairs...")
    print("   Format: Each sample has:")
    print("   • Prompt (customer question)")
    print("   • Chosen (better response)")
    print("   • Rejected (worse response)")
    print()

    dataset = sdk.generate(
        num_samples=10,
        domain_context="Generate customer service scenarios with professional vs unprofessional responses"
    )

    # Show sample
    print("\n" + "="*60)
    print("📝 SAMPLE PREFERENCE PAIR")
    print("="*60)

    if dataset.samples:
        import json
        sample = dataset.samples[0].data.model_dump()

        print("\n🔹 Prompt:")
        print(f"   {sample.get('prompt', 'N/A')}")

        print("\n✅ Chosen (Better Response):")
        print(f"   {sample.get('chosen', 'N/A')[:200]}...")

        print("\n❌ Rejected (Worse Response):")
        print(f"   {sample.get('rejected', 'N/A')[:200]}...")

    # Show analytics
    dataset.show_analytics()

    # Save
    output_path = "outputs/dpo_customer_service.jsonl"
    os.makedirs("outputs", exist_ok=True)
    dataset.save(output_path)

    print(f"\n✓ Generated {len(dataset)} DPO preference pairs")
    print(f"💾 Saved to: {output_path}")

    print("\n" + "="*60)
    print("💡 DPO Training Tips:")
    print("="*60)
    print("  • Use these pairs to train preference models")
    print("  • The 'chosen' response should be clearly better")
    print("  • The 'rejected' response should be plausible but worse")
    print("  • Vary the types of improvements (quality, style, tone)")
    print("  • DPO works well for aligning model behavior to preferences")


if __name__ == "__main__":
    main()
