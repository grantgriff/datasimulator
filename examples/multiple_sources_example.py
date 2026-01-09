"""
Example: Generate training data from multiple sources

This example demonstrates combining content from multiple document types
to create a comprehensive training dataset.
"""

import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasimulator import DataSimulator, load_document


def load_multiple_sources(sources):
    """
    Load and combine content from multiple sources.

    Args:
        sources: List of file paths or URLs

    Returns:
        Combined content string
    """
    print("📚 Loading content from multiple sources:")
    print("-" * 60)

    combined_content = []

    for source in sources:
        try:
            print(f"  Loading: {source}...")
            content = load_document(source)
            combined_content.append(f"\n\n=== Source: {source} ===\n\n{content}")
            print(f"    ✓ Loaded {len(content)} characters")
        except Exception as e:
            print(f"    ❌ Error: {e}")

    full_content = "\n\n".join(combined_content)
    print(f"\n✓ Total content: {len(full_content)} characters\n")

    return full_content


def main():
    """Generate dataset from multiple sources."""

    print("=" * 60)
    print("DataSimulator - Multiple Sources Example")
    print("=" * 60)

    # Create sample files for demonstration
    os.makedirs("examples/sample_data", exist_ok=True)

    # Create sample content files
    sources = []

    # Source 1: Text file about accounting basics
    source1 = "examples/sample_data/accounting_basics.txt"
    with open(source1, "w") as f:
        f.write("""
        Accounting Basics

        The accounting equation: Assets = Liabilities + Equity

        This fundamental equation must always balance.
        """)
    sources.append(source1)

    # Source 2: Text file about financial statements
    source2 = "examples/sample_data/financial_statements.txt"
    with open(source2, "w") as f:
        f.write("""
        Financial Statements

        1. Balance Sheet - Shows financial position at a point in time
        2. Income Statement - Shows profitability over a period
        3. Cash Flow Statement - Shows cash movements
        """)
    sources.append(source2)

    # Source 3: Could add a URL (optional)
    # sources.append("https://example.com/accounting-guide")

    print(f"\n📋 Sources prepared:")
    for i, src in enumerate(sources, 1):
        print(f"  {i}. {src}")

    # Load all content
    combined_content = load_multiple_sources(sources)

    # Save combined content temporarily
    combined_path = "examples/sample_data/combined_content.txt"
    with open(combined_path, "w") as f:
        f.write(combined_content)

    print(f"💾 Saved combined content to: {combined_path}")

    # Generate training data from combined sources
    print("\n🚀 Generating training data from combined sources...")
    print("=" * 60)

    sdk = DataSimulator(
        source=combined_path,
        data_type="sft",
        models={
            "generator": "claude-3-5-sonnet-20241022",
            "verifier": "gpt-4o-mini",
        },
        quality_threshold=6.0,
        max_cost=3.0,
        batch_size=5,
    )

    dataset = sdk.generate(
        num_samples=15,
        domain_context="Generate diverse questions covering all source materials",
        show_progress=True
    )

    # Show results
    dataset.show_analytics()
    dataset.sample_examples(n=2)

    # Save
    output_path = "outputs/multi_source_sft.jsonl"
    os.makedirs("outputs", exist_ok=True)
    dataset.save(output_path)

    print(f"\n✅ Done! Generated dataset from {len(sources)} sources")
    print(f"💾 Saved to {output_path}")

    # Tips
    print("\n💡 Tips for multiple sources:")
    print("  • Combine related documents for comprehensive coverage")
    print("  • Mix formats: PDFs, docs, web pages, images")
    print("  • Use domain_context to guide question generation")
    print("  • Larger source content = more diverse training data")


if __name__ == "__main__":
    main()
