"""
Smoke test for the new data_type="ranked" path.

Runs the full end-to-end flow:
  - GeminiPlanner builds a batch plan from a tiny source doc
  - topic_emphasis biases the planner toward ASC 606
  - RankedGenerator produces 4 records, each with 3 ranked responses
  - Verifier scores each response and they get sorted into ranks

Cost: a few cents on Gemini 2.5 Flash. Runtime: 1-3 minutes.

Run with:
    python scratch_ranked.py
"""

import json
from pathlib import Path

from dotenv import load_dotenv

from datasimulator import DataSimulator

load_dotenv()

src = Path("/tmp/accounting_blurb.txt")
src.write_text(
    "ASC 606 governs revenue recognition. The 5-step model: identify "
    "the contract, identify performance obligations, determine "
    "transaction price, allocate to obligations, and recognize revenue "
    "when control transfers to the customer.\n\n"
    "ASC 842 governs lease accounting. Lessees recognize a right-of-use "
    "asset and lease liability for nearly all leases > 12 months."
)

sdk = DataSimulator(
    source=str(src),
    data_type="ranked",
    enable_planning=True,
    # Generator does the heavy lifting (drafting N candidate responses at
    # varied quality levels) — try Pro for it. Planner and verifier stay on
    # Flash for cost and rate-limit headroom.
    models={
        "generator": "gemini-2.5-pro",
        "verifier": "gemini-2.5-flash",
        "planner": "gemini-2.5-flash",
    },
    ranked_config={"num_responses": 3, "quality_spread": "wide"},
    batch_size=2,           # Small batch keeps us under free-tier 5 RPM
    parallel_batches=1,
    max_cost=1.00,
    interactive=False,
)

ds = sdk.generate(
    num_samples=2,           # 1 planner + 1 generator + 2 verifier = 4 calls
    topic_emphasis={"ASC 606 revenue recognition": 0.5},
)

ds.save("scratch_ranked.jsonl")
ds.show_analytics()

print("\n--- First record ---")
with open("scratch_ranked.jsonl") as f:
    print(json.dumps(json.loads(f.readline()), indent=2))
