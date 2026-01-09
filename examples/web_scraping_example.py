"""
Example: Generate training data from web pages

This example demonstrates web scraping and generating training data
from online content.
"""

import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasimulator import DataSimulator, load_document


def main():
    """Generate dataset from web content."""

    print("=" * 60)
    print("DataSimulator - Web Scraping Example")
    print("=" * 60)

    # Example 1: Scrape a simple web page
    print("\n🌐 Example 1: Load content from URL")
    print("-" * 60)

    # Use a simple, public documentation page
    url = "https://en.wikipedia.org/wiki/Accounting"

    print(f"Loading content from {url}...")

    try:
        content = load_document(url)
        print(f"✓ Loaded {len(content)} characters")
        print(f"\nFirst 200 characters:\n{content[:200]}...")

        # Example 2: Generate training data from web content
        print("\n\n🚀 Example 2: Generate training data from web page")
        print("-" * 60)

        sdk = DataSimulator(
            source=url,
            data_type="sft",
            models={
                "generator": "claude-3-5-sonnet-20241022",
                "verifier": "gpt-4o-mini",
            },
            quality_threshold=6.0,
            max_cost=3.0,
            batch_size=5,
        )

        # Generate dataset
        dataset = sdk.generate(
            num_samples=10,
            domain_context="Focus on accounting concepts and principles",
            show_progress=True
        )

        # Show results
        dataset.show_analytics()
        dataset.sample_examples(n=2)

        # Save
        output_path = "outputs/web_based_sft.jsonl"
        os.makedirs("outputs", exist_ok=True)
        dataset.save(output_path)

        print(f"\n✅ Done! Saved to {output_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nNote: This example requires internet connection and:")
        print("  pip install requests beautifulsoup4")

    # Example 3: JavaScript-heavy website (commented out - requires Playwright)
    print("\n\n🎭 Example 3: JavaScript-heavy website (requires Playwright)")
    print("-" * 60)
    print("For single-page applications (SPAs), use javascript=True:")
    print("""
    # Install first: pip install playwright && playwright install chromium

    content = load_document(
        "https://example-spa.com",
        javascript=True,  # Use Playwright for JS rendering
        wait_for=".main-content"  # Wait for element
    )
    """)


if __name__ == "__main__":
    main()
