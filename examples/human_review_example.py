"""
Example: Human-in-the-loop review

This example demonstrates interactive human review of generated samples.
Users can approve, reject, or skip samples in real-time.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasimulator import DataSimulator, HumanReviewer


def main():
    """Demonstrate human review interface."""

    print("=" * 60)
    print("DataSimulator - Human Review Example")
    print("=" * 60)

    # Create sample source
    os.makedirs("examples/sample_data", exist_ok=True)

    source_path = "examples/sample_data/review_demo.txt"
    with open(source_path, "w") as f:
        f.write("""
        Accounting Principles

        Double-entry bookkeeping requires every transaction to affect
        at least two accounts, keeping the accounting equation in balance.

        Assets = Liabilities + Equity

        Debits increase asset accounts and decrease liability accounts.
        Credits increase liability accounts and decrease asset accounts.
        """)

    print(f"\n📄 Created source: {source_path}")

    # Generate some samples first
    print("\n🚀 Generating samples for review...")

    sdk = DataSimulator(
        source=source_path,
        data_type="sft",
        models={
            "generator": "claude-3-5-sonnet-20241022",
            "verifier": "gpt-4o-mini",
        },
        quality_threshold=6.0,
        max_cost=1.5,
        batch_size=5,
    )

    dataset = sdk.generate(
        num_samples=5,  # Small batch for interactive review
        domain_context="Generate questions about accounting principles"
    )

    print(f"\n✓ Generated {len(dataset)} samples")

    # Human review
    print("\n" + "="*60)
    print("👤 HUMAN REVIEW SESSION")
    print("="*60)

    print("""
This is an interactive review session. You will see each generated sample
and can decide to:
  - (a)pprove: Keep the sample
  - (r)eject: Remove the sample (will be regenerated if needed)
  - (s)kip: Skip this sample (don't include in final dataset)
  - (q)uit: End review session

Let's start!
""")

    # Initialize reviewer
    reviewer = HumanReviewer(interactive=True)

    # Review the batch
    approved, rejected_indices = reviewer.review_batch(
        [s.model_dump() for s in dataset.samples],
        show_quality_scores=True
    )

    # Show review statistics
    stats = reviewer.get_stats()

    print("\n" + "="*60)
    print("📈 REVIEW STATISTICS")
    print("="*60)

    print(f"Total Reviewed:  {stats['total_reviewed']}")
    print(f"Approved:        {stats['approved']}")
    print(f"Rejected:        {stats['rejected']}")
    print(f"Skipped:         {stats['skipped']}")

    approval_rate = (stats['approved'] / stats['total_reviewed'] * 100
                    if stats['total_reviewed'] > 0 else 0)
    print(f"Approval Rate:   {approval_rate:.1f}%")

    # Save approved samples
    if approved:
        output_path = "outputs/human_reviewed.jsonl"
        os.makedirs("outputs", exist_ok=True)

        # Convert approved samples back to dataset format
        import json
        with open(output_path, 'w') as f:
            for sample in approved:
                if "data" in sample:
                    json.dump(sample["data"], f)
                    f.write('\n')

        print(f"\n✓ Saved {len(approved)} approved samples to {output_path}")

    print("\n" + "="*60)
    print("✅ Human review demo complete!")
    print("="*60)

    print("""
💡 Tips for Human Review:
  • Review in smaller batches (5-10 samples at a time)
  • Use quality scores as a guide but trust your judgment
  • Reject samples that don't meet your specific needs
  • Skip samples if you're unsure (better to be selective)
  • Provide rejection reasons to track common issues
""")


if __name__ == "__main__":
    main()
