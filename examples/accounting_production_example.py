"""
Production Example: Generate SFT dataset from accounting decks

This is a complete production-ready script for generating 2000-3000 SFT samples
from multiple accounting documents autonomously.

SETUP INSTRUCTIONS:
1. Place your accounting PDF/DOCX files in: examples/accounting_docs/
2. Set environment variables:
   export ANTHROPIC_API_KEY="your_key"
   export OPENAI_API_KEY="your_key"
   export GOOGLE_API_KEY="your_key"  # For Gemini planning
3. Run: python examples/accounting_production_example.py

OUTPUT:
- Final dataset: outputs/accounting_sft_dataset.jsonl
- Checkpoints: checkpoints/ (saves every 20 samples)
- Analytics displayed at completion
"""

import os
import sys
from pathlib import Path
from glob import glob

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasimulator import DataSimulator


def main():
    """Generate accounting SFT dataset autonomously."""

    print("=" * 70)
    print("ACCOUNTING SFT DATASET GENERATION")
    print("=" * 70)

    # STEP 1: Find all accounting documents
    docs_dir = Path("examples/accounting_docs")

    # Support PDF, DOCX, TXT files
    accounting_files = []
    for ext in ["*.pdf", "*.docx", "*.txt"]:
        accounting_files.extend(glob(str(docs_dir / ext)))

    if not accounting_files:
        print(f"\n❌ ERROR: No files found in {docs_dir}/")
        print(f"\nPlease place your accounting documents in: {docs_dir}/")
        print("Supported formats: PDF, DOCX, TXT")
        return

    print(f"\n📚 Found {len(accounting_files)} accounting documents:")
    for i, doc in enumerate(accounting_files, 1):
        file_size = os.path.getsize(doc) / 1024  # KB
        print(f"  {i:2d}. {Path(doc).name:40s} ({file_size:8.1f} KB)")

    # STEP 2: Verify API keys
    print("\n🔑 Checking API keys...")
    required_keys = {
        "ANTHROPIC_API_KEY": "Claude (generator)",
        "OPENAI_API_KEY": "GPT-4o-mini (verifier)",
        "GOOGLE_API_KEY": "Gemini (planning)"
    }

    missing_keys = []
    for key, purpose in required_keys.items():
        if os.getenv(key):
            print(f"  ✓ {key:20s} - {purpose}")
        else:
            print(f"  ✗ {key:20s} - MISSING")
            missing_keys.append(key)

    if missing_keys:
        print(f"\n❌ ERROR: Missing API keys: {', '.join(missing_keys)}")
        print("\nSet them with:")
        for key in missing_keys:
            print(f'  export {key}="your_key_here"')
        return

    # STEP 3: Configure generation parameters
    TARGET_SAMPLES = 40  # Adjust this for testing (use 2500 for production)
    MAX_BUDGET = 40.0      # Your budget

    print(f"\n⚙️  Generation Configuration:")
    print(f"  Target Samples:     {TARGET_SAMPLES:,}")
    print(f"  Max Budget:         ${MAX_BUDGET:.2f}")
    print(f"  Estimated Cost:     ${TARGET_SAMPLES * 0.011:.2f}")
    print(f"  Quality Threshold:  6.0/10")
    print(f"  Checkpointing:      Every 20 samples")
    print(f"  Planning:           Gemini-powered (topic extraction)")

    # STEP 4: Initialize SDK
    print("\n🚀 Initializing DataSimulator...")

    sdk = DataSimulator(
        # Load all accounting documents
        source=accounting_files,

        # SFT format
        data_type="sft",

        # Model configuration
        models={
            "generator": "claude-sonnet-4-5-20250929",  # Main generator
            "verifier": "gpt-4o-mini-2024-07-18",       # Quality scoring
        },

        # Quality settings
        quality_threshold=6.0,        # Accept samples scored 6.0+
        batch_size=20,                # Generate 20 samples per batch

        # Autonomous settings (no prompts)
        max_cost=MAX_BUDGET,          # Stop at $40
        interactive=False,            # Don't prompt user

        # Checkpointing (crash recovery)
        checkpoint_dir="checkpoints",
        checkpoint_interval=20,       # Save every 20 samples

        # Gemini planning layer
        enable_planning=True,         # Analyze docs, extract topics
    )

    print("✓ SDK initialized successfully")

    # STEP 5: Generate dataset
    print("\n" + "=" * 70)
    print("STARTING AUTONOMOUS GENERATION")
    print("=" * 70)
    print("\n💡 You can safely walk away - this runs unattended!")
    print("💡 Checkpoints saved every 20 samples in: checkpoints/")
    print("💡 Press Ctrl+C to stop (progress is saved)\n")

    try:
        dataset = sdk.generate(
            num_samples=TARGET_SAMPLES,
            domain_context="""
            Generate diverse accounting questions and answers covering:

            TOPICS:
            - Accounts receivable and accounts payable
            - Journal entries and general ledger transactions
            - Financial statements (Balance Sheet, Income Statement, Cash Flow)
            - Revenue recognition and matching principles
            - Bad debt estimation and write-offs
            - Asset valuation and depreciation
            - Inventory accounting (FIFO, LIFO, weighted average)
            - Accrual vs. cash basis accounting
            - Internal controls and audit procedures

            QUESTION TYPES:
            - Conceptual questions (explain principles)
            - Calculation questions (compute values)
            - Journal entry questions (record transactions)
            - Analysis questions (interpret financial data)
            - Problem-solving questions (apply principles)

            DIFFICULTY:
            - Mix of basic, intermediate, and advanced
            - Include both theoretical and practical scenarios
            """,
            show_progress=True
        )

    except KeyboardInterrupt:
        print("\n\n⚠️  Generation stopped by user (Ctrl+C)")
        print("💾 Progress saved in checkpoints/")
        print("💡 Re-run this script to resume from checkpoint")
        return

    # STEP 6: Display results
    print("\n" + "=" * 70)
    print("✅ GENERATION COMPLETE")
    print("=" * 70)

    dataset.show_analytics()

    # STEP 7: Save final output
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "accounting_sft_dataset.jsonl"
    dataset.save(str(output_path))

    print(f"\n📊 SUMMARY:")
    print(f"  Total Samples:      {len(dataset):,}")
    print(f"  Average Quality:    {dataset.average_quality:.2f}/10")
    print(f"  Total Cost:         ${dataset.total_cost:.2f}")
    if len(dataset) > 0:
        print(f"  Cost per Sample:    ${dataset.total_cost / len(dataset):.4f}")
    else:
        print(f"  Cost per Sample:    N/A (no samples generated)")

    print(f"\n💾 OUTPUT FILES:")
    print(f"  Dataset (JSONL):    {output_path}")
    print(f"  Checkpoints:        checkpoints/")

    print(f"\n✅ SUCCESS! Your training dataset is ready.")
    print(f"\n💡 NEXT STEPS:")
    print(f"  1. Review samples: dataset.sample_examples(5)")
    print(f"  2. Filter by quality: dataset.filter_by_quality(7.0)")
    print(f"  3. Use for training: Load {output_path} into your training pipeline")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
