"""
Example: Generate training data from a source document

This example shows how to use a source text file to guide data generation.
"""

import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasimulator import DataSimulator


# Sample source content (you would load this from a file)
SAMPLE_SOURCE = """
Accounting Fundamentals

The Accounting Equation:
Assets = Liabilities + Equity

This fundamental equation must always balance. Every transaction affects at least
two accounts to maintain this balance.

Common Account Types:
1. Assets - Resources owned by the company (cash, inventory, equipment)
2. Liabilities - Debts owed to others (accounts payable, loans)
3. Equity - Owner's interest in the company (capital, retained earnings)
4. Revenue - Income from business operations
5. Expenses - Costs of doing business

Debit and Credit Rules:
- Assets: Debit increases, Credit decreases
- Liabilities: Credit increases, Debit decreases
- Equity: Credit increases, Debit decreases
- Revenue: Credit increases, Debit decreases
- Expenses: Debit increases, Credit decreases
"""


def main():
    """Generate dataset from source material."""

    print("=" * 60)
    print("DataSimulator - Source Document Example")
    print("=" * 60)

    # Create a temporary source file
    os.makedirs("examples/temp", exist_ok=True)
    source_path = "examples/temp/accounting_basics.txt"

    with open(source_path, "w") as f:
        f.write(SAMPLE_SOURCE)

    print(f"\n📄 Created source document: {source_path}")

    # Initialize SDK with source
    sdk = DataSimulator(
        source=source_path,  # Use source document
        data_type="sft",
        models={
            "generator": "claude-3-5-sonnet-20241022",
            "verifier": "gpt-4o-mini",
        },
        quality_threshold=6.5,  # Slightly higher threshold
        max_cost=5.0,
        batch_size=5,
    )

    # Generate dataset
    print("\n🚀 Generating examples based on source material...\n")

    dataset = sdk.generate(
        num_samples=10,
        domain_context="Focus on practical applications of the concepts in the source material"
    )

    # Analytics
    dataset.show_analytics()

    # Save
    output_path = "outputs/source_based_sft.jsonl"
    os.makedirs("outputs", exist_ok=True)
    dataset.save(output_path)

    # Also save as JSON with metadata
    dataset.save("outputs/source_based_sft.json", format="json")

    print(f"\n✅ Done! Generated {len(dataset)} samples")
    print(f"💾 Saved to: {output_path}")


if __name__ == "__main__":
    main()
