# Posty ⇄ DataSimulator integration handoff

**Audience:** the Claude working on the Posty CLI (a separate project that
wraps this SDK to provide an interactive, user-friendly data generation
experience).

**Goal of this doc:** give you everything you need to integrate Posty with
the `datasimulator` SDK — what the SDK does, how to call it, what to ask
the user, in what order, with what validation, and what gotchas to avoid.

You can read this top-to-bottom. Code examples are copy-pasteable.

---

## 1. Context: who does what

Posty and DataSimulator are two separate projects with one division of
responsibility:

| Layer | Owns |
|---|---|
| **Posty CLI** (what you're building) | All user interaction — prompts, validation, progress UI, file pickers, URL input, "are you sure?" confirms, saving output, error rendering. |
| **DataSimulator SDK** (this repo) | Pure function: `(sources, config) → dataset`. No stdin, no `print` chrome, no human decisions. Headless by default. |

This split is deliberate. The SDK was recently repositioned to be a clean
programmatic backend — `interactive=False` is the default, sources accept
raw text, progress fires as structured events. **Do not add interactive
prompts to the SDK side.** All UX lives in Posty.

---

## 2. What DataSimulator does (one paragraph)

Given some source material (PDFs, Word docs, text, Markdown, URLs, Google
Docs, or raw text strings) and a target sample count, the SDK generates
training-data records in one of five formats (`sft`, `dpo`,
`verifiable_qa`, `ranked`, `full`). It uses a planner LLM to extract
topics and allocate batches across them, generates samples in parallel,
scores each sample for quality, drops samples below threshold, retries
failures, and tracks spend against a hard cost cap. Output is a list of
Pydantic-validated records you can write to JSONL or consume in-memory.

---

## 3. The user flow Posty needs to drive

This is the interactive flow Posty should walk a user through. Each step
maps to one or more SDK parameters.

### Step 1 — Collect source materials

**What to ask the user:**
- "What source material do you want to base the dataset on?"
- Accept any combination of: local file paths, URLs (web pages or Google
  Docs), or pasted text content.
- Let the user add multiple sources in one session. Don't force one at a
  time.

**Validation Posty should do:**
- Local paths: verify the file exists and is readable. Surface a clear
  error if not — the SDK will fail late otherwise.
- URLs: basic URL shape check (`http://` / `https://`). Don't try to
  fetch yourself; let the SDK do it.
- Pasted text: any non-empty string works. The SDK treats strings with
  newlines or >500 chars as raw text content.
- **Warn if total source material is < 1 KB.** With tiny inputs the LLM
  runs out of distinct prompts and quality drops fast. See gotcha #1
  below.

**How it maps to the SDK:**
```python
# Pass everything as a single list, mixing types freely.
source = [
    "/Users/grant/docs/asc606.pdf",
    "https://www.iasplus.com/en/standards/ifrs/ifrs15",
    "Raw pasted text from the user...\nwith newlines\n...",
]
```

### Step 2 — Pick the data format

**What to ask the user:**
- "What kind of training data?"
- Offer these options, ideally with a one-line description of each:

| Option | When to pick it |
|---|---|
| `sft` | Supervised fine-tuning: `{messages: [{role, content}, ...]}` |
| `dpo` | Preference data: `{prompt, chosen, rejected}` |
| `verifiable_qa` | Q&A with ground truth + auto-verification rules |
| `ranked` | One prompt + N candidate responses ranked by quality (GRPO-style) |
| `full` | All of the above in one record — SFT + DPO + GRPO views from a single generation |

**Recommended default:** `full`. One record carries every downstream
training view; no need to regenerate when the user changes their mind.

**How it maps to the SDK:**
```python
data_type = "full"  # one of the five strings above
```

### Step 3 — Sample count and budget

**What to ask the user:**
- "How many records?" (integer)
- "Maximum spend?" (USD, default $10)

**Validation:**
- Sample count must be ≥ 1. Warn if > 5000 (long run, expect ~10+ min).
- Budget must be > 0. Warn if cost-per-sample estimate × samples > budget
  (the SDK will halt mid-run — annoying UX).

**Current default models** (OpenRouter Gemini): `gemini-3.5-flash` for
generator/verifier/diversity, `gemini-2.5-pro` for the planner. Pinned to
concrete versions (not the `-latest` moving aliases) so behaviour can't
shift without a code change. At these Flash defaults, expect very roughly
~$0.0015/sample for SFT.

**Rough cost estimates** (order-of-magnitude; the absolute figures below
were calibrated on an earlier OpenAI default set — gpt-5.4-mini generator,
gpt-4.1-nano verifier, gpt-5.4 planner — and are kept for relative
comparison across data types):
- SFT: ~$0.005/sample
- DPO: ~$0.007/sample
- ranked (3 responses): ~$0.008/sample
- full (4 responses): ~$0.012/sample

**How it maps to the SDK:**
```python
num_samples = 500   # constructor arg to .generate()
max_cost   = 10.0   # constructor arg to DataSimulator()
```

### Step 4 — Topic emphasis (optional but high-leverage)

**What to ask the user:**
- "Are there specific topics you want emphasized?" (skippable)
- If yes, collect topic strings + a weight per topic. Weights must sum to
  ≤ 1.0. Remainder goes to whatever topics the planner naturally extracts.

**Posty UX tip:** make this a two-stage interaction. First ask whether
they want emphasis at all. If yes, let them add topics one at a time with
sliders or numeric input for weights, validating the running total.

**How it maps to the SDK:**
```python
topic_emphasis = {
    "ASC 606 revenue recognition": 0.4,
    "deferred tax assets": 0.3,
    # remaining 0.3 distributed by the planner
}
# Requires enable_planning=True (set it automatically)
```

### Step 5 — Output destination

**What to ask the user:**
- "Where should the dataset go?" (file path, default `outputs/dataset.jsonl`)
- Optionally: "Anything to do with the records after generation?" (e.g.
  pipe into another tool, upload somewhere)

**How it maps to the SDK:**
```python
dataset.save("outputs/dataset.jsonl")
# OR consume in-memory:
records = [s.data.model_dump() for s in dataset.samples]
```

### Step 6 — Confirmation screen

**What to show before kicking off generation:**
- Summary of all collected inputs
- Estimated cost (samples × per-sample-cost from the table above)
- Expected runtime estimate (rough: 1-2 samples/sec wall-clock)
- "Proceed? [y/n]"

---

## 4. Code: minimal Posty integration

This is the smallest snippet that wires Posty to the SDK end-to-end.

```python
from datasimulator import DataSimulator

# Posty has collected these from the user interactively:
sources         = [...]   # list[str]: paths, URLs, or raw text
data_type       = "full"
num_samples     = 500
max_cost        = 10.00
topic_emphasis  = {"...": 0.4} or None
output_path     = "outputs/dataset.jsonl"

sdk = DataSimulator(
    source=sources,
    data_type=data_type,
    max_cost=max_cost,
    enable_planning=True,                # always on — much higher quality
    ranked_config={                       # required for ranked / full
        "num_responses": 4,
        "quality_spread": "wide",         # see gotcha #2
    },
    # interactive defaults to False — leave it that way for CLI use.
)

dataset = sdk.generate(
    num_samples=num_samples,
    topic_emphasis=topic_emphasis,
)

dataset.save(output_path)
# Posty now reports:
print(f"Wrote {len(dataset.samples)} records, "
      f"avg quality {dataset.average_quality:.1f}, "
      f"${dataset.total_cost:.2f} spent")
```

---

## 5. Code: recommended Posty integration with live progress

The SDK fires structured events during generation. Use these to drive your
progress UI (Rich, textual, or whatever Posty uses).

```python
from datasimulator import DataSimulator
from rich.progress import Progress

with Progress() as progress:
    task = progress.add_task("Generating...", total=num_samples)

    def on_event(e: dict) -> None:
        # e is a flat dict with an "event" key.
        if e["event"] == "generation_started":
            # e: num_samples, data_type, quality_threshold, max_cost,
            #    num_planned_batches, domain
            pass
        elif e["event"] == "batch_completed":
            # e: samples_in_batch, samples_passed, batch_cost,
            #    average_quality, samples_generated, samples_target,
            #    total_cost
            progress.update(task, completed=e["samples_generated"])
        elif e["event"] == "checkpoint_saved":
            # e: samples_generated, checkpoint_dir
            pass
        elif e["event"] == "cost_limit_reached":
            # e: total_cost, max_cost, samples_generated, samples_target
            progress.console.log(
                f"[yellow]Cost cap hit at "
                f"{e['samples_generated']}/{e['samples_target']}"
            )
        elif e["event"] == "generation_completed":
            # e: samples_generated, samples_target, failed, total_cost
            pass

    sdk = DataSimulator(
        source=sources,
        data_type=data_type,
        max_cost=max_cost,
        enable_planning=True,
        ranked_config={"num_responses": 4, "quality_spread": "wide"},
        progress_callback=on_event,
    )
    dataset = sdk.generate(num_samples=num_samples,
                           topic_emphasis=topic_emphasis)
```

**Callback rules:**
- Can be sync or async. Both are awaited correctly.
- Exceptions raised inside the callback are logged and swallowed —
  a buggy Posty handler can't kill the generation run.
- Fires at batch granularity, not per record. Don't try to render
  individual sample events; they're not emitted.

---

## 6. Output shapes (what records look like)

You'll need to know this to render previews, filter, or pipe into other
tools. Each `dataset.samples[i].data.model_dump()` returns:

| `data_type` | Record shape |
|---|---|
| `sft` | `{messages: [{role: "user", content: "..."}, {role: "assistant", content: "..."}]}` |
| `dpo` | `{prompt: "...", chosen: "...", rejected: "..."}` |
| `verifiable_qa` | `{prompt: "...", ground_truth: "...", verification_type: "exact_match" \| ...}` |
| `ranked` | `{prompt: "...", ranked_responses: [{rank, text, quality_score}], topic, subtopic}` |
| `full` | `{prompt, gold_answer, chosen, rejected, ranked_responses, topic, subtopic}` |

Each sample also has `sample.metrics`:
```python
sample.metrics.quality_score   # float (1-10)
sample.metrics.token_count     # int
sample.metrics.generation_cost # float (USD)
sample.metrics.model_used      # str
```

---

## 7. Gotchas we've learned the hard way

### Gotcha #1: tiny source material → low yield
The SDK target is `num_samples`, but actual output can be lower if the
source is too thin. With ~225 chars of source content asking for 50
ranked samples, the planner runs out of distinct prompts and the
quality-spread filter drops most candidates. We saw 43/50 in testing.

**What Posty should do:** if `sum(len(s) for s in resolved_sources) <
1000`, warn the user that they should add more material. Suggest a soft
floor of ~5 KB per topic.

### Gotcha #2: `quality_spread="wide"` + `num_responses=3` is too aggressive
For `ranked` and `full` data types, `quality_spread="wide"` requires a
>5.0 score gap between best and worst response. With only 3 responses,
the LLM rarely produces one bad enough to hit that gap. Records get
dropped en masse.

**What Posty should do:** if the user picks `quality_spread="wide"`,
auto-set `num_responses=4` (minimum) or `5` (better). If they pick
`"narrow"`, 3 responses is fine.

### Gotcha #3: planner needs source content
`enable_planning=True` does nothing if `source=None`. It silently skips
planning. Posty should require at least one source when the user picks a
data_type that benefits from planning (which is all of them).

### Gotcha #4: cost cap halts mid-run silently
With `interactive=False` (the default and the right setting), if the cost
cap is hit, the SDK stops cleanly and returns partial results. Posty must
check `len(dataset.samples) < num_samples` after generation and tell the
user: "Hit cap at X/Y records. Want to raise the budget and continue?"
(No, you can't actually continue from where it stopped — see gotcha #5.)

### Gotcha #5: no resumable runs (yet)
If a run dies (process killed, network outage), there's no
`resume_from=...` API. Checkpoints write to JSONL but can't be replayed.
Posty should set `checkpoint_dir` for long runs so the user at least has
the partial output, and warn them resume isn't available yet.

### Gotcha #6: model defaults are OpenAI-only
The default generator/verifier/planner are all OpenAI models. The user
needs `OPENAI_API_KEY` set. If you let users override models (e.g.
Anthropic or Gemini), they also need the matching API key. Posty should
detect missing keys before calling `.generate()` and surface a clear
error.

---

## 8. Files in this repo to read

In rough priority order:

| File | Why |
|---|---|
| `INTEGRATION.md` | **Read this first.** The authoritative contract — every input, every event, every output shape. |
| `README.md` | High-level overview, especially the **Roadmap** section near the bottom. |
| `run_dataset.py` | Working end-to-end example. The CONFIG block at the top maps 1:1 to SDK parameters. |
| `examples/basic_sft_example.py` | Simplest possible call. |
| `examples/all_generators_example.py` | One file showing all 5 data types in action. |
| `examples/autonomous_batch_example.py` | Long-running batch example with checkpointing. |
| `datasimulator/sdk.py` | The actual `DataSimulator` class. Look at `__init__` and `generate()` signatures for the source of truth on parameters. |
| `tests/test_progress_callback.py` | Working examples of every progress event being consumed. |

---

## 9. Testing your Posty integration

You don't need real LLM calls to verify Posty is wiring things correctly.
Patch `ModelRouter` to a mock and verify Posty:

1. Passes a `list[str]` (not a single string) for multi-source.
2. Sets `interactive=False` (or omits it — that's now the default).
3. Auto-sets `enable_planning=True` when the user supplied any source.
4. Auto-sets `num_responses>=4` when `quality_spread="wide"`.
5. Hooks `progress_callback` and renders at least the `batch_completed`
   event.
6. After `.generate()` returns, surfaces partial-result UX if
   `len(samples) < num_samples`.

```python
from unittest.mock import patch, MagicMock

def test_posty_wires_sdk_correctly():
    with patch("datasimulator.sdk.ModelRouter", lambda *a, **kw: MagicMock()):
        # Run Posty's main flow with stubbed user inputs
        result = posty.run(
            sources=["raw text\nwith newline"],
            data_type="ranked",
            num_samples=10,
            max_cost=1.0,
            quality_spread="wide",
        )
    # Assert Posty called the SDK with num_responses>=4
    # etc.
```

---

## 10. Open questions to confirm with Grant

These are things we explicitly deferred. Confirm with him before
implementing:

1. **Should Posty allow resuming a killed run?** Today the SDK has no
   resume API. If this matters, escalate — the SDK roadmap has it but
   it's not built.
2. **Should Posty stream output records as they're generated?** Today
   `dataset.samples` materializes at the end. Live record preview would
   need either per-sample SDK events (not built) or polling the
   checkpoint dir.
3. **Should Posty handle multi-provider keys (OpenAI + Anthropic + Gemini)
   or just OpenAI for v1?** OpenAI-only is simpler and covers the
   defaults.
4. **Confirmation flow:** before kicking off a paid run, should Posty
   show a "this will cost ~$X, proceed?" prompt every time, or only above
   a configurable threshold?

---

## 11. TL;DR for the impatient

1. Posty collects sources, data_type, num_samples, max_cost, optional
   topic_emphasis from the user.
2. Posty calls `DataSimulator(...)` with those, plus
   `enable_planning=True` and a `progress_callback`.
3. Posty calls `.generate(num_samples=..., topic_emphasis=...)`.
4. Posty renders progress from the callback events.
5. Posty saves output and reports final stats to the user.

The SDK is headless by design. Posty is the UX. They meet in the middle
through the parameters in `INTEGRATION.md`.

Ship it.
