"""
Full-run dataset generator. Edit the CONFIG block, drop docs in ./data/,
then run: python run_dataset.py

The script auto-loads everything in data/ (.pdf .docx .txt .md), pipes it
through the planner, and writes the dataset to outputs/.

Cost estimate at the configured defaults (gpt-5.4-mini + gpt-4.1-nano +
gpt-5.4 planner):
  ~$0.008 per ranked sample (3 responses each)
  ~$0.005 per SFT sample
  ~$0.012 per "full" sample (SFT + DPO + GRPO views on one record)
"""

import json
import os
from glob import glob
from pathlib import Path

from dotenv import load_dotenv

from datasimulator import DataSimulator

load_dotenv(override=True)

# ============================================================================
# CONFIG — edit these
# ============================================================================

DATA_DIR = "data"                     # folder containing your reference docs
OUTPUT_PATH = "outputs/dataset.jsonl" # where the generated dataset is written

DATA_TYPE = "ranked"                  # "sft" | "dpo" | "verifiable_qa" | "ranked" | "full"
NUM_SAMPLES = 50                      # how many records to generate
BATCH_SIZE = 10                       # records per generation batch
PARALLEL_BATCHES = 2                  # batches in flight at once
MAX_COST_USD = 5.00                   # hard cap; generation stops if hit

QUALITY_THRESHOLD = 6.0               # min score (1-10); samples below this get regenerated
CHECKPOINT_DIR = None                 # e.g. "outputs/checkpoints" to save partial output every N samples
CHECKPOINT_INTERVAL = 100             # save a checkpoint every N samples (ignored if CHECKPOINT_DIR is None)

# For data_type="ranked" or "full":
NUM_RESPONSES = 3                     # how many candidates per prompt
QUALITY_SPREAD = "wide"               # "wide" (gap >5.0) or "narrow" (gap <2.5)

# Bias the planner toward specific topics. Set to {} to let the planner choose.
# Weights must sum to <= 1.0. Remainder goes to topics the planner extracts.
TOPIC_EMPHASIS = {
    # "ASC 606 revenue recognition": 0.4,
    # "deferred tax assets": 0.3,
}

# Models — defaults work; override here to mix providers.
MODELS = {
    # "generator": "gpt-5.4-mini",
    # "verifier":  "gpt-4.1-nano",
    # "planner":   "gpt-5.4",
}

# ============================================================================


def main() -> None:
    # 1. Find source docs
    data_dir = Path(DATA_DIR)
    if not data_dir.exists():
        raise SystemExit(
            f"❌ {DATA_DIR}/ doesn't exist. Create it and drop your reference docs in."
        )

    sources: list[str] = []
    for ext in ("*.pdf", "*.docx", "*.txt", "*.md"):
        sources.extend(glob(str(data_dir / ext)))

    if not sources:
        raise SystemExit(
            f"❌ No source docs found in {DATA_DIR}/.\n"
            f"   Supported: .pdf .docx .txt .md\n"
            f"   Drop your files in {DATA_DIR}/ and re-run."
        )

    print(f"📚 Loaded {len(sources)} source doc(s) from {DATA_DIR}/:")
    for path in sources:
        size_kb = os.path.getsize(path) / 1024
        print(f"   - {Path(path).name}  ({size_kb:.1f} KB)")

    # 2. Sanity-check the API key
    if not os.environ.get("OPENAI_API_KEY", "").startswith("sk-"):
        raise SystemExit(
            "❌ OPENAI_API_KEY is missing or looks like a placeholder.\n"
            "   Put your real key in .env (it should start with 'sk-')."
        )

    # 3. Cost estimate
    per_sample = {"sft": 0.005, "dpo": 0.008, "verifiable_qa": 0.003,
                  "ranked": 0.008, "full": 0.012}.get(DATA_TYPE, 0.008)
    est_cost = NUM_SAMPLES * per_sample
    print(
        f"\n⚙️  Config:\n"
        f"   data_type:        {DATA_TYPE}\n"
        f"   num_samples:      {NUM_SAMPLES}\n"
        f"   batch_size:       {BATCH_SIZE}\n"
        f"   topic_emphasis:   {TOPIC_EMPHASIS or '(planner decides)'}\n"
        f"   est. cost:        ~${est_cost:.2f}  (hard cap ${MAX_COST_USD:.2f})"
    )

    # 4. Build the SDK and generate
    sdk_kwargs = dict(
        source=sources,
        data_type=DATA_TYPE,
        enable_planning=True,
        batch_size=BATCH_SIZE,
        parallel_batches=PARALLEL_BATCHES,
        max_cost=MAX_COST_USD,
        quality_threshold=QUALITY_THRESHOLD,
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        interactive=False,
    )
    if DATA_TYPE in ("ranked", "full"):
        sdk_kwargs["ranked_config"] = {
            "num_responses": NUM_RESPONSES,
            "quality_spread": QUALITY_SPREAD,
        }
    if MODELS:
        sdk_kwargs["models"] = MODELS

    sdk = DataSimulator(**sdk_kwargs)
    ds = sdk.generate(
        num_samples=NUM_SAMPLES,
        topic_emphasis=TOPIC_EMPHASIS or None,
    )

    # 5. Save and summarize
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    ds.save(OUTPUT_PATH)
    ds.show_analytics()

    # 6. Preview first record
    with open(OUTPUT_PATH) as f:
        first = f.readline()
    print("\n--- First record ---")
    print(json.dumps(json.loads(first), indent=2) if first.strip() else "(no records)")


if __name__ == "__main__":
    main()
