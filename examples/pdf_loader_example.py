"""
Example: Generate training data from a PDF document

This example demonstrates loading content from PDF files
and generating SFT training data.
"""

import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasimulator import DataSimulator, load_document


def main():
    """Generate dataset from PDF document."""

    print("=" * 60)
    print("DataSimulator - PDF Loader Example")
    print("=" * 60)

    # Example 1: Direct loading with DataSimulator
    print("\n📄 Example 1: Load PDF directly in DataSimulator")
    print("-" * 60)

    # Note: Replace with your actual PDF path
    pdf_path = "examples/sample_data/accounting_guide.pdf"

    if not Path(pdf_path).exists():
        print(f"⚠️  PDF not found at {pdf_path}")
        print("Creating a sample text file instead...")

        # Create sample directory
        os.makedirs("examples/sample_data", exist_ok=True)

        # Create sample content
        sample_content = """
        Accounting Fundamentals - Bad Debt Expense

        Bad debt expense represents accounts receivable that are unlikely to be collected.

        Recording Bad Debt:
        1. Direct Write-Off Method
           Debit: Bad Debt Expense
           Credit: Accounts Receivable

        2. Allowance Method (GAAP Preferred)
           Debit: Bad Debt Expense
           Credit: Allowance for Doubtful Accounts

        The allowance method follows the matching principle by estimating
        uncollectible accounts in the same period as the sale.
        """

        # Save as text file for demo
        pdf_path = "examples/sample_data/accounting_guide.txt"
        with open(pdf_path, "w") as f:
            f.write(sample_content)

        print(f"✓ Created sample file: {pdf_path}")

    # Initialize SDK with PDF source
    sdk = DataSimulator(
        source=pdf_path,  # Can be .pdf, .txt, .docx, etc.
        data_type="sft",
        models={
            "generator": "claude-3-5-sonnet-20241022",
            "verifier": "gpt-4o-mini",
        },
        quality_threshold=6.0,
        max_cost=3.0,  # Small budget for example
        batch_size=5,
    )

    # Generate dataset
    print(f"\n🚀 Generating 10 examples from {pdf_path}...\n")

    dataset = sdk.generate(
        num_samples=10,
        show_progress=True
    )

    # Show results
    dataset.show_analytics()
    dataset.sample_examples(n=2)

    # Save
    output_path = "outputs/pdf_based_sft.jsonl"
    os.makedirs("outputs", exist_ok=True)
    dataset.save(output_path)

    print(f"\n✅ Done! Saved to {output_path}")

    # Example 2: Standalone document loading
    print("\n\n📄 Example 2: Standalone document loading")
    print("-" * 60)

    print(f"Loading content from {pdf_path}...")
    content = load_document(pdf_path)

    print(f"✓ Loaded {len(content)} characters")
    print(f"\nFirst 200 characters:\n{content[:200]}...")


if __name__ == "__main__":
    main()
