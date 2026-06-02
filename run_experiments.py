"""Run a fleet of smoke / scale-up experiments across providers.

Each experiment writes to outputs/experiments/<name>/ with:
  dataset.jsonl           clean training records
  dataset.metadata.jsonl  per-sample sidecar (score, topic, cost, model)
  summary.json            aggregate metrics for cross-experiment comparison

Usage:
  python run_experiments.py <name>           # run one named experiment
  python run_experiments.py --smokes         # run all *_smoke experiments
  python run_experiments.py --scale          # run all *_scale_* experiments
  python run_experiments.py --all            # smokes + scale-up
  python run_experiments.py --list           # show registered experiments

Behavior on failure (per user spec): retry once with batch_size=5; if that
also fails, skip and continue. Final aggregate summary lands at
outputs/experiments/FINAL_SUMMARY.md and triggers a macOS notification.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from datasimulator import DataSimulator


REPO_ROOT = Path(__file__).parent
EXP_DIR = REPO_ROOT / "outputs" / "experiments"
SMOKE_SOURCE = str(REPO_ROOT / "data" / "accounting_docs" / "handout on allowance.pdf")
# Total budget for the whole batch — sum of per-experiment max_cost caps
# must stay under this. Currently ~ sum below = ~$8, well under $25 cap.
TOTAL_BUDGET_USD = 25.0


def experiment(name: str, **overrides):
    base = dict(
        name=name,
        source=SMOKE_SOURCE,
        data_type="sft",
        num_samples=10,
        batch_size=10,
        parallel_batches=1,
        max_cost=2.0,
        quality_threshold=6.0,
        enable_planning=False,
        models=None,
    )
    base.update(overrides)
    return base


# Model triplets per provider — overridable per-experiment via models=
MODELS_OPENAI = {
    "generator": "gpt-4.1-mini",
    "verifier": "gpt-4.1-nano",
    "diversity": "gpt-4.1-nano",
}
MODELS_OPENROUTER = {
    # Gemini 2.5 Flash through OpenRouter for all three roles —
    # different model family than the OpenAI baseline so we can compare
    # whether Gemini Flash is a more discriminating verifier than nano.
    "generator": "openrouter/google/gemini-2.5-flash",
    "verifier":  "openrouter/google/gemini-2.5-flash",
    "diversity": "openrouter/google/gemini-2.5-flash",
}
MODELS_CLOUDFLARE = {
    # Big OSS generator + Qwen 32B/30B-MoE for evals. gpt-oss-120b is
    # the largest text-gen model CF hosts. qwq-32b (reasoning) handles
    # verification; qwen3-30b-a3b (MoE) handles diversity.
    "generator": "cf/@cf/openai/gpt-oss-120b",
    "verifier":  "cf/@cf/qwen/qwq-32b",
    "diversity": "cf/@cf/qwen/qwen3-30b-a3b-fp8",
}
MODELS_GEMINI_PLANNED = {
    # Gemini 2.5 Pro reads the full source corpus and emits a topic
    # allocation; Flash handles generation, verification, and diversity.
    # Pro is ~4x the cost of Flash but is called only once per run.
    "planner":   "openrouter/google/gemini-2.5-pro",
    "generator": "openrouter/google/gemini-2.5-flash",
    "verifier":  "openrouter/google/gemini-2.5-flash",
    "diversity": "openrouter/google/gemini-2.5-flash",
}

# Full source corpus for planner-driven experiments — globbed from
# data/ recursively, same convention as run_dataset.py.
ALL_SOURCES = sorted(
    str(p) for ext in ("pdf", "docx", "txt", "md")
    for p in (REPO_ROOT / "data").rglob(f"*.{ext}")
)


def smoke(name, models):
    return experiment(name, num_samples=10, max_cost=0.50, models=models)


def scale(name, models, data_type):
    # 50 samples for scale-up runs (per user spec). max_cost gives generous
    # headroom; actual spend will be much lower for these models.
    caps = {"sft": 1.50, "dpo": 2.00, "verifiable_qa": 1.50, "ranked": 3.00}
    return experiment(
        name,
        data_type=data_type,
        num_samples=50,
        batch_size=10,
        parallel_batches=2,
        max_cost=caps.get(data_type, 2.0),
        models=models,
    )


def planned(name, data_type):
    """Planner-driven experiment over the full source corpus.

    Uses Gemini 2.5 Pro as the planner (one call upfront to extract a
    topic allocation from all 35 source docs) and Gemini 2.5 Flash for
    generation, verification, and diversity. 50 samples per data type.
    """
    caps = {"sft": 3.00, "dpo": 4.00, "verifiable_qa": 3.00,
            "ranked": 5.00, "full": 6.00}
    return experiment(
        name,
        source=ALL_SOURCES,
        data_type=data_type,
        num_samples=50,
        batch_size=10,
        parallel_batches=2,
        max_cost=caps.get(data_type, 4.0),
        models=MODELS_GEMINI_PLANNED,
        enable_planning=True,
    )


EXPERIMENTS: dict[str, dict] = {
    # ---- Smoke tests: 10 samples each, ~$0.10-0.20 expected ----
    "openai_smoke":     smoke("openai_smoke",     MODELS_OPENAI),
    "openrouter_smoke": smoke("openrouter_smoke", MODELS_OPENROUTER),
    "cloudflare_smoke": smoke("cloudflare_smoke", MODELS_CLOUDFLARE),

    # ---- Scale-up: 50 samples per (provider, data_type) ----
    # SFT — the bread-and-butter format
    "openai_scale_sft":     scale("openai_scale_sft",     MODELS_OPENAI,     "sft"),
    "openrouter_scale_sft": scale("openrouter_scale_sft", MODELS_OPENROUTER, "sft"),
    "cloudflare_scale_sft": scale("cloudflare_scale_sft", MODELS_CLOUDFLARE, "sft"),
    # DPO — preference pairs (chosen/rejected)
    "openai_scale_dpo":     scale("openai_scale_dpo",     MODELS_OPENAI,     "dpo"),
    "openrouter_scale_dpo": scale("openrouter_scale_dpo", MODELS_OPENROUTER, "dpo"),
    # verifiable_qa — for RL-style training
    "openai_scale_qa":     scale("openai_scale_qa",     MODELS_OPENAI,     "verifiable_qa"),
    "openrouter_scale_qa": scale("openrouter_scale_qa", MODELS_OPENROUTER, "verifiable_qa"),

    # ---- Planner-driven: all 5 data types, Gemini Pro planner +
    #      Gemini Flash gen/verifier/diversity, full corpus
    "planned_sft":     planned("planned_sft",     "sft"),
    "planned_dpo":     planned("planned_dpo",     "dpo"),
    "planned_qa":      planned("planned_qa",      "verifiable_qa"),
    "planned_ranked":  planned("planned_ranked",  "ranked"),
    "planned_full":    planned("planned_full",    "full"),
}


def _score_distribution(metadata_path: Path) -> dict:
    if not metadata_path.exists():
        return {"error": "no sidecar"}
    scores = []
    with metadata_path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
                if "quality_score" in row:
                    scores.append(float(row["quality_score"]))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    if not scores:
        return {"n": 0}
    return {
        "n": len(scores),
        "mean": round(statistics.mean(scores), 3),
        "median": round(statistics.median(scores), 3),
        "min": min(scores),
        "max": max(scores),
        "stdev": round(statistics.pstdev(scores), 3),
        # The 5.0-fallback bug from commit 5274969 manifests as scores
        # stuck at exactly 5.0. >50% means the bug is back or a related
        # silent-failure path got triggered.
        "exact_5p0_count": sum(1 for s in scores if s == 5.0),
        "exact_5p0_fraction": round(sum(1 for s in scores if s == 5.0) / len(scores), 3),
    }


def _run_attempt(cfg: dict, out_dir: Path, batch_size: int) -> tuple[object, str | None]:
    """Run one DataSimulator.generate attempt. Returns (dataset, error_str)."""
    try:
        sdk = DataSimulator(
            source=cfg["source"],
            data_type=cfg["data_type"],
            models=cfg["models"],
            batch_size=batch_size,
            parallel_batches=cfg["parallel_batches"],
            max_cost=cfg["max_cost"],
            quality_threshold=cfg["quality_threshold"],
            enable_planning=cfg["enable_planning"],
            interactive=False,
        )
        ds = sdk.generate(num_samples=cfg["num_samples"])
        ds.save(str(out_dir / "dataset.jsonl"))
        return ds, None
    except Exception as e:
        traceback.print_exc()
        return None, f"{type(e).__name__}: {e}"


def run_one(name: str, *, retry_on_failure: bool = True) -> dict:
    if name not in EXPERIMENTS:
        raise SystemExit(f"Unknown experiment: {name}. Try --list.")

    cfg = EXPERIMENTS[name]
    out_dir = EXP_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "dataset.metadata.jsonl"
    summary_path = out_dir / "summary.json"

    src = cfg["source"]
    src_label = (f"{len(src)} files (corpus)" if isinstance(src, list)
                 else Path(src).name)
    print(f"\n{'='*72}\n▶  {name}\n{'='*72}")
    print(f"  source:       {src_label}")
    print(f"  data_type:    {cfg['data_type']}")
    print(f"  num_samples:  {cfg['num_samples']}  (batch_size={cfg['batch_size']})")
    print(f"  models:       {cfg['models']}")

    started = time.time()
    attempts: list[dict] = []
    ds, err = _run_attempt(cfg, out_dir, cfg["batch_size"])
    attempts.append({"batch_size": cfg["batch_size"], "error": err})

    if err and retry_on_failure and cfg["batch_size"] > 5:
        print(f"\n⚠  First attempt failed: {err}\n   Retrying with batch_size=5…")
        ds, err = _run_attempt(cfg, out_dir, 5)
        attempts.append({"batch_size": 5, "error": err})

    elapsed = time.time() - started

    summary = {
        "name": name,
        "config": {k: v for k, v in cfg.items() if k != "name"},
        "elapsed_sec": round(elapsed, 2),
        "attempts": attempts,
        "final_error": err,
        "output_path": str(out_dir / "dataset.jsonl"),
        "metadata_path": str(metadata_path),
    }
    if ds is not None:
        summary.update({
            "total_samples": ds.total_samples,
            "average_quality_in_memory": round(ds.average_quality, 3),
            "total_cost_usd": round(ds.total_cost, 4),
        })
    summary["score_distribution"] = _score_distribution(metadata_path)

    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n📊 {name} →  {summary_path}")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("config",)}, indent=2))
    return summary


def _aggregate_summary(results: list[dict]) -> str:
    """Render a FINAL_SUMMARY.md from per-experiment summaries."""
    lines = ["# Overnight experiment summary", ""]
    total_cost = sum(r.get("total_cost_usd", 0) for r in results)
    total_samples = sum(r.get("total_samples", 0) or 0 for r in results)
    n_failed = sum(1 for r in results if r.get("final_error"))
    lines += [
        f"- Experiments run: **{len(results)}**",
        f"- Failed: **{n_failed}**",
        f"- Total samples generated: **{total_samples}**",
        f"- Total cost: **${total_cost:.4f}** (budget cap ${TOTAL_BUDGET_USD:.2f})",
        "",
        "## Per-experiment results",
        "",
        "| name | samples | mean score | %@5.0 | cost | elapsed | error |",
        "|------|---------|------------|-------|------|---------|-------|",
    ]
    for r in results:
        sd = r.get("score_distribution") or {}
        lines.append(
            f"| {r['name']} | {sd.get('n', '?')} | "
            f"{sd.get('mean', '—')} | "
            f"{sd.get('exact_5p0_fraction', '—')} | "
            f"${r.get('total_cost_usd', 0):.4f} | "
            f"{r.get('elapsed_sec', 0):.1f}s | "
            f"{(r.get('final_error') or '—')[:40]} |"
        )
    lines.append("")
    lines.append("## Quality-scorer integrity check")
    lines.append("")
    lines.append("The 5.0-fallback bug (commit 5274969) manifests as every score == 5.0.")
    lines.append("Any row with **%@5.0 > 0.3** warrants investigation.")
    suspect = [r for r in results
               if (r.get("score_distribution") or {}).get("exact_5p0_fraction", 0) > 0.3]
    if suspect:
        lines.append("")
        lines.append("**Suspect experiments:**")
        for r in suspect:
            sd = r["score_distribution"]
            lines.append(f"- `{r['name']}` — {sd['exact_5p0_count']}/{sd['n']} "
                         f"samples scored exactly 5.0")
    else:
        lines.append("")
        lines.append("✓ No experiments exceeded the 30% 5.0-fallback threshold.")
    return "\n".join(lines)


def _notify_mac(title: str, body: str):
    """Best-effort macOS notification. Quietly no-op on failure."""
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{body}" with title "{title}"'],
            check=False, timeout=5,
        )
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--smokes", action="store_true")
    ap.add_argument("--scale", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-retry", action="store_true",
                    help="Disable batch_size=5 retry on failure")
    args = ap.parse_args()

    if args.list:
        for k in EXPERIMENTS:
            print(f"  {k}")
        return

    if args.name and not (args.smokes or args.scale or args.all):
        run_one(args.name, retry_on_failure=not args.no_retry)
        return

    # Pick the experiment set
    if args.all:
        names = list(EXPERIMENTS.keys())
    elif args.smokes:
        names = [n for n in EXPERIMENTS if n.endswith("_smoke")]
    elif args.scale:
        names = [n for n in EXPERIMENTS if "_scale_" in n]
    else:
        ap.print_help()
        sys.exit(1)

    # Cost guard: stop before exceeding TOTAL_BUDGET_USD
    spent = 0.0
    results: list[dict] = []
    for name in names:
        if spent >= TOTAL_BUDGET_USD:
            print(f"\n🛑 Budget cap ${TOTAL_BUDGET_USD:.2f} reached "
                  f"(spent ${spent:.2f}). Stopping.")
            break
        r = run_one(name, retry_on_failure=not args.no_retry)
        results.append(r)
        spent += r.get("total_cost_usd", 0) or 0
        print(f"\n💰 Cumulative spend: ${spent:.4f} / ${TOTAL_BUDGET_USD:.2f}")

    # Write aggregate summary and notify
    summary_md = _aggregate_summary(results)
    summary_path = EXP_DIR / "FINAL_SUMMARY.md"
    summary_path.write_text(summary_md)
    summary_json = EXP_DIR / "FINAL_SUMMARY.json"
    summary_json.write_text(json.dumps(results, indent=2))
    print(f"\n📦 Final summary → {summary_path}")
    print(f"📦 Per-experiment JSON → {summary_json}")
    n_failed = sum(1 for r in results if r.get("final_error"))
    _notify_mac(
        "DataSimulator experiments done",
        f"{len(results)} runs, {n_failed} failed, ${spent:.2f} spent",
    )


if __name__ == "__main__":
    main()
