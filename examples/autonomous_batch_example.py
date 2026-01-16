"""
Example: Autonomous generation from multiple files with checkpointing

This example demonstrates:
1. Loading multiple source documents at once
2. Non-interactive mode (no prompts when hitting cost limits)
3. Automatic checkpointing for progress persistence
4. Large-scale autonomous generation

Perfect for: Loading 20+ accounting docs and generating 10K samples unattended
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasimulator import DataSimulator


def main():
    """Generate large dataset from multiple sources autonomously."""

    print("=" * 60)
    print("DataSimulator - Autonomous Batch Generation Example")
    print("=" * 60)

    # Example: Load multiple accounting documents
    # In your case, you'd list all 20+ of your actual PDF files here
    accounting_docs = [
        "accounting_doc1.pdf",
        "accounting_doc2.pdf",
        "accounting_doc3.pdf",
        # ... add all your documents
    ]

    # For demo purposes, create some sample files
    os.makedirs("examples/sample_data", exist_ok=True)
    demo_files = []

    for i in range(1, 4):
        demo_file = f"examples/sample_data/accounting_doc{i}.txt"
        with open(demo_file, "w") as f:
            f.write(f"""
            Accounting Document {i}

            This document covers accounting principles and practices related to
            accounts receivable, revenue recognition, and financial reporting.

            Key Topics:
            - Double-entry bookkeeping
            - Journal entries
            - General ledger
            - Financial statements
            """)
        demo_files.append(demo_file)

    print(f"\n📚 Loading {len(demo_files)} source documents:")
    for i, doc in enumerate(demo_files, 1):
        print(f"  {i}. {doc}")

    # Initialize SDK with autonomous settings
    print("\n⚙️  Initializing SDK...")
    sdk = DataSimulator(
        # Multiple sources - pass as list!
        source=demo_files,

        data_type="sft",

        models={
            "generator": "claude-3-5-sonnet-20241022",
            "verifier": "gpt-4o-mini",
        },

        # Quality settings
        quality_threshold=6.0,
        batch_size=20,

        # AUTONOMOUS SETTINGS
        max_cost=200.0,        # Set high upfront (won't prompt every $20)
        interactive=False,     # Don't prompt user - fully autonomous

        # CHECKPOINTING
        checkpoint_dir="checkpoints",       # Save progress here
        checkpoint_interval=100,            # Save every 100 samples
    )

    print("\n✓ SDK initialized in AUTONOMOUS mode")
    print("  • Multiple sources will be combined automatically")
    print("  • No prompts when hitting cost limits (up to $200)")
    print("  • Checkpoints saved every 100 samples")
    print("  • Can resume from checkpoint if interrupted\n")

    # Generate large dataset
    print("🚀 Starting autonomous generation...")
    print("=" * 60)
    print("You can safely walk away - this will run unattended!")
    print("=" * 60)

    dataset = sdk.generate(
        num_samples=1000,  # In your case: 10000
        domain_context="""
        Generate diverse accounting questions and answers covering:
        - Accounts receivable and accounts payable
        - Journal entries and general ledger
        - Financial statements (Balance Sheet, Income Statement, Cash Flow)
        - Revenue recognition and matching principles
        - Bad debt estimation and write-offs
        """,
        show_progress=True
    )

    # Show results
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)

    dataset.show_analytics()

    # Save final output
    output_path = "outputs/autonomous_accounting_dataset.jsonl"
    os.makedirs("outputs", exist_ok=True)
    dataset.save(output_path)

    print(f"\n✅ Done! Generated {len(dataset)} samples")
    print(f"💾 Final dataset: {output_path}")
    print(f"💾 Checkpoints: checkpoints/")
    print(f"💰 Total cost: ${dataset.total_cost:.2f}")

    print("\n" + "=" * 60)
    print("💡 TIPS FOR AUTONOMOUS GENERATION")
    print("=" * 60)
    print("1. Set max_cost high enough (estimate: $0.02-0.05 per sample)")
    print("2. Use interactive=False to prevent blocking on prompts")
    print("3. Enable checkpointing in case of crashes/interruptions")
    print("4. Load all sources upfront as a list")
    print("5. Use quality_threshold=5.0 to reduce regenerations")
    print("6. Consider disabling verification for faster generation:")
    print("   (just remove 'verifier' from models dict)")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
