"""
Example: All Training Data Formats

This example demonstrates generating data for all supported formats:
- SFT (Supervised Fine-Tuning)
- DPO (Direct Preference Optimization)
- PPO (Proximal Policy Optimization)
- GRPO (Group Relative Policy Optimization)
- RL Verifiable (RL with ground truth)
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasimulator import DataSimulator


def create_sample_source():
    """Create sample source content for all generators."""
    os.makedirs("examples/sample_data", exist_ok=True)

    source_path = "examples/sample_data/all_formats.txt"
    with open(source_path, "w") as f:
        f.write("""
        Accounting Fundamentals

        Double-Entry Bookkeeping:
        Every transaction affects at least two accounts.
        Assets = Liabilities + Equity must always balance.

        Common Transactions:
        1. Recording Sales Revenue
           Debit: Accounts Receivable or Cash
           Credit: Sales Revenue

        2. Recording Expenses
           Debit: Expense Account
           Credit: Cash or Accounts Payable

        3. Bad Debt Write-Off
           Debit: Allowance for Doubtful Accounts
           Credit: Accounts Receivable

        Calculations:
        - Bad Debt Expense = Accounts Receivable × Estimated Uncollectible %
        - Depreciation (Straight-Line) = (Cost - Salvage Value) / Useful Life
        - Net Income = Revenue - Expenses
        """)

    return source_path


def generate_sft_data(source_path: str):
    """Generate SFT (Supervised Fine-Tuning) data."""
    print("\n" + "="*60)
    print("1️⃣  SFT (Supervised Fine-Tuning)")
    print("="*60)
    print("Format: Instruction-response pairs")
    print()

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
        num_samples=5,
        domain_context="Generate accounting Q&A pairs"
    )

    output_path = "outputs/sft_data.jsonl"
    os.makedirs("outputs", exist_ok=True)
    dataset.save(output_path)

    print(f"✓ Generated {len(dataset)} SFT samples")
    print(f"💾 Saved to: {output_path}")


def generate_dpo_data(source_path: str):
    """Generate DPO (Direct Preference Optimization) data."""
    print("\n" + "="*60)
    print("2️⃣  DPO (Direct Preference Optimization)")
    print("="*60)
    print("Format: Preference pairs (chosen vs rejected)")
    print()

    sdk = DataSimulator(
        source=source_path,
        data_type="dpo",  # ← DPO format
        models={
            "generator": "claude-3-5-sonnet-20241022",
            "verifier": "gpt-4o-mini",
        },
        quality_threshold=6.0,
        max_cost=1.5,
        batch_size=5,
    )

    dataset = sdk.generate(
        num_samples=5,
        domain_context="Generate accounting questions with better/worse responses"
    )

    output_path = "outputs/dpo_data.jsonl"
    dataset.save(output_path)

    print(f"✓ Generated {len(dataset)} DPO preference pairs")
    print(f"💾 Saved to: {output_path}")


def generate_ppo_data(source_path: str):
    """Generate PPO (Proximal Policy Optimization) data."""
    print("\n" + "="*60)
    print("3️⃣  PPO (Proximal Policy Optimization)")
    print("="*60)
    print("Format: Prompts only (rewards from reward model)")
    print()

    sdk = DataSimulator(
        source=source_path,
        data_type="ppo",  # ← PPO format
        models={
            "generator": "claude-3-5-sonnet-20241022",
            "verifier": "gpt-4o-mini",
        },
        quality_threshold=6.0,
        max_cost=1.5,
        batch_size=5,
    )

    dataset = sdk.generate(
        num_samples=5,
        domain_context="Generate open-ended accounting questions"
    )

    output_path = "outputs/ppo_data.jsonl"
    dataset.save(output_path)

    print(f"✓ Generated {len(dataset)} PPO prompts")
    print(f"💾 Saved to: {output_path}")


def generate_grpo_data(source_path: str):
    """Generate GRPO (Group Relative Policy Optimization) data."""
    print("\n" + "="*60)
    print("4️⃣  GRPO (Group Relative Policy Optimization)")
    print("="*60)
    print("Format: Prompts with multi-completion generation")
    print()

    sdk = DataSimulator(
        source=source_path,
        data_type="grpo",  # ← GRPO format
        models={
            "generator": "claude-3-5-sonnet-20241022",
            "verifier": "gpt-4o-mini",
        },
        quality_threshold=6.0,
        max_cost=1.5,
        batch_size=5,
    )

    dataset = sdk.generate(
        num_samples=5,
        domain_context="Generate verifiable accounting problems"
    )

    output_path = "outputs/grpo_data.jsonl"
    dataset.save(output_path)

    print(f"✓ Generated {len(dataset)} GRPO prompts")
    print(f"💾 Saved to: {output_path}")


def generate_rl_verifiable_data(source_path: str):
    """Generate RL with Verifiable Rewards data."""
    print("\n" + "="*60)
    print("5️⃣  RL Verifiable (RL with Ground Truth)")
    print("="*60)
    print("Format: Prompts with verifiable ground truth answers")
    print()

    sdk = DataSimulator(
        source=source_path,
        data_type="rl_verifiable",  # ← RL Verifiable format
        models={
            "generator": "claude-3-5-sonnet-20241022",
            "verifier": "gpt-4o-mini",
        },
        quality_threshold=6.0,
        max_cost=1.5,
        batch_size=5,
    )

    dataset = sdk.generate(
        num_samples=5,
        domain_context="Generate accounting calculations with correct answers"
    )

    output_path = "outputs/rl_verifiable_data.jsonl"
    dataset.save(output_path)

    print(f"✓ Generated {len(dataset)} RL verifiable samples")
    print(f"💾 Saved to: {output_path}")


def main():
    """Generate all training data formats."""

    print("=" * 60)
    print("DataSimulator - All Training Formats Example")
    print("=" * 60)
    print("\nGenerating data for ALL post-training formats:")
    print("  1. SFT (Supervised Fine-Tuning)")
    print("  2. DPO (Direct Preference Optimization)")
    print("  3. PPO (Proximal Policy Optimization)")
    print("  4. GRPO (Group Relative Policy Optimization)")
    print("  5. RL Verifiable (RL with Ground Truth)")

    # Create sample source
    source_path = create_sample_source()
    print(f"\n📄 Created source: {source_path}")

    # Generate each format
    try:
        generate_sft_data(source_path)
        generate_dpo_data(source_path)
        generate_ppo_data(source_path)
        generate_grpo_data(source_path)
        generate_rl_verifiable_data(source_path)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nNote: Make sure you have set your API keys:")
        print("  export ANTHROPIC_API_KEY=your_key")
        print("  export OPENAI_API_KEY=your_key")
        return

    # Summary
    print("\n" + "="*60)
    print("✅ ALL FORMATS GENERATED SUCCESSFULLY!")
    print("="*60)

    print("\nGenerated files:")
    print("  📁 outputs/sft_data.jsonl           - Instruction-response pairs")
    print("  📁 outputs/dpo_data.jsonl           - Preference pairs")
    print("  📁 outputs/ppo_data.jsonl           - Prompts for reward model")
    print("  📁 outputs/grpo_data.jsonl          - Prompts for multi-completion")
    print("  📁 outputs/rl_verifiable_data.jsonl - Prompts with ground truth")

    print("\n💡 Use Case Guide:")
    print("  • SFT: Initial model training with demonstrations")
    print("  • DPO: Fine-tuning with preference feedback")
    print("  • PPO: RL training with external reward model")
    print("  • GRPO: RL with relative ranking (great for verifiable tasks)")
    print("  • RL Verifiable: RL with automatic verification (math, code, facts)")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
