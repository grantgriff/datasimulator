# Integration guide: calling DataSimulator from another tool

This document is the contract for tools (like the Posty CLI) that drive
`DataSimulator` programmatically. It lists every input you might surface
to your users, what shape it should be in, and what the SDK returns.

## TL;DR

```python
from datasimulator import DataSimulator

sdk = DataSimulator(
    source=["/path/to/doc.pdf", "https://example.com/spec"],
    data_type="ranked",
    enable_planning=True,
    ranked_config={"num_responses": 4, "quality_spread": "wide"},
    max_cost=10.00,
    # Defaults to interactive=False — safe for headless / CLI use.
)
dataset = sdk.generate(
    num_samples=500,
    topic_emphasis={"ASC 606 revenue recognition": 0.4},
)
dataset.save("out.jsonl")          # write to disk (JSONL, one record per line)
for sample in dataset.samples:     # OR consume in-memory
    record = sample.data.model_dump()
    score  = sample.metrics.quality_score
```

The SDK never prompts the user, never reads stdin, and never writes to disk
unless you call `.save()`. `interactive=False` is the default — leave it
that way for any CLI/server use.

## Inputs Posty should expose to its users

### Required

| User-facing field | SDK parameter | Type | Notes |
|---|---|---|---|
| Source materials | `source=` (constructor) | `str`, `List[str]`, or raw text | File paths, URLs, or already-loaded text. See "Source formats" below. |
| Dataset size | `num_samples=` (`.generate`) | `int` | Total records to generate. |
| Output type | `data_type=` (constructor) | `"sft"` \| `"dpo"` \| `"verifiable_qa"` \| `"ranked"` \| `"full"` | See "Output formats" below. |

### Strongly recommended (give the user a knob for each)

| User-facing field | SDK parameter | Type | Default | Notes |
|---|---|---|---|---|
| Topic emphasis | `topic_emphasis=` (`.generate`) | `Dict[str, float]` | `None` | Weights must sum to ≤1.0. Requires `enable_planning=True`. |
| Smart planning | `enable_planning=` (constructor) | `bool` | `False` | Highly recommended on — uses a planner LLM to allocate batches across topics. |
| Cost cap | `max_cost=` (constructor) | `float` (USD) | `20.0` | Generation halts cleanly at this number. |
| Quality threshold | `quality_threshold=` (constructor) | `float` (1-10) | `6.0` | Records scoring below this are rejected and regenerated. |
| Ranked config | `ranked_config=` (constructor) | `Dict[str, Any]` | `None` | Required when `data_type` is `"ranked"` or `"full"`. Shape: `{"num_responses": int, "quality_spread": "wide" \| "narrow"}` |

### Optional (sensible defaults already in place)

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `models` | `Dict[str, str]` | Gemini Flash via OpenRouter (Pro for planner) | Override any role: `generator`, `verifier`, `diversity`, `planner`. See "Default models" below. |
| `batch_size` | `int` | `20` | Records per generation call. Lower = finer cost control. |
| `parallel_batches` | `int` | `4` | Concurrent batches. Lower if you hit rate limits. |
| `checkpoint_dir` | `str` | `None` | Auto-save partial output every N records. Set this for long runs. |
| `checkpoint_interval` | `int` | `20` | Checkpoint frequency. |
| `domain_context` | `str` | `None` | Free-text hint passed to generators. |
| `diversity_threshold` | `float` | `0.85` | Similarity cap. |
| `openai_api_key` | `str` | reads `OPENAI_API_KEY` | Override the env var. |
| `anthropic_api_key` | `str` | reads `ANTHROPIC_API_KEY` | Only needed if you use Claude models. |
| `google_api_key` | `str` | reads `GOOGLE_API_KEY` | Only needed if you use Gemini models. |
| `openrouter_api_key` | `str` | reads `OPENROUTER_API_KEY` | Activates the `openrouter/...` model prefix. |
| `cloudflare_api_key` | `str` | reads `CLOUDFLARE_API_TOKEN` | Activates the `cf/...` model prefix for Cloudflare Workers AI. |
| `cloudflare_account_id` | `str` | reads `CLOUDFLARE_ACCOUNT_ID` | Required with `cf/...`. CF's endpoint URL embeds the account ID. |
| `do_api_key` | `str` | reads `DO_INFERENCE_KEY` | Activates the `do/...` model prefix for DigitalOcean Serverless Inference. |
| `progress_callback` | `Callable[[dict], None \| Awaitable[None]]` | `None` | Structured progress events for your UI. See "Progress callbacks" below. |

### Default models

If you don't pass `models=`, the SDK uses Gemini through OpenRouter:

| Role | Default | Why |
|---|---|---|
| `generator` | `openrouter/google/gemini-flash-latest` | Cheap, fast, strong instruction-following for SFT/DPO output. |
| `verifier` | `openrouter/google/gemini-flash-latest` | Same model batches well for the batch quality scorer. |
| `diversity` | `openrouter/google/gemini-flash-latest` | Used for prompt embedding/similarity checks. |
| `planner` | `openrouter/google/gemini-pro-latest` | Reads source material end-to-end; benefits from long context + stronger reasoning. Only invoked when `enable_planning=True`. |

Set `OPENROUTER_API_KEY` and you're done. Override any role via the
`models=` dict if you want to mix providers or pin specific versions
(e.g. `{"generator": "openrouter/openai/gpt-5.4-mini"}`).

### Provider routing

Model strings determine the provider. Prefixes are checked in this order:

| Prefix | Provider | Example |
|---|---|---|
| `openrouter/` | OpenRouter (OpenAI-compatible). Cost is read directly from the response. | `openrouter/anthropic/claude-3.5-sonnet`, `openrouter/openai/gpt-4o`, `openrouter/meta-llama/llama-3.3-70b-instruct` |
| `cf/` | Cloudflare Workers AI (OpenAI-compatible). Edge-hosted OSS models. | `cf/@cf/meta/llama-3.3-70b-instruct-fp8-fast`, `cf/@cf/openai/gpt-oss-120b`, `cf/@cf/google/gemma-3-12b-it` |
| `do/` | DigitalOcean Serverless Inference (OpenAI-compatible). Hosts open-source models. | `do/llama3.3-70b-instruct`, `do/mistral-nemo-instruct-2407` |
| `claude*` | Anthropic direct | `claude-3-5-sonnet-20241022` |
| `gpt*` | OpenAI direct | `gpt-5.4-mini`, `gpt-4.1-nano` |
| `gemini*` | Google direct | `gemini-2.5-flash` |
| anything else | Ollama (local) | `llama3.3`, `qwen2.5` |

The proxy prefixes (`openrouter/`, `cf/`, `do/`) win over `claude*`/`gpt*`/`gemini*` —
so `openrouter/openai/gpt-4o` routes to OpenRouter (one bill), not to OpenAI direct.

### Switching between OpenRouter and Cloudflare

The common case for this SDK — use OpenRouter for frontier-quality
generation and Cloudflare for cheap bulk work (e.g. verification scoring):

```python
sdk = DataSimulator(
    source="textbook.pdf",
    generator_model="openrouter/openai/gpt-5.4-mini",
    verifier_model="cf/@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    diversity_model="cf/@cf/google/gemma-3-12b-it",
    # Keys can also come from env vars
    openrouter_api_key="sk-or-...",
    cloudflare_api_key="cf-token-...",
    cloudflare_account_id="your-acct-id",
)
```

Or flip the same flow to fully OpenRouter or fully CF without changing
anything else — just swap the model strings.

**Mix providers in one run:**

```python
sdk = DataSimulator(
    source="textbook.pdf",
    generator_model="openrouter/openai/gpt-5.4-mini",   # OR for generation
    verifier_model="do/llama3.3-70b-instruct",          # DO for cheap scoring
    diversity_model="openrouter/openai/gpt-4.1-nano",   # OR for embeddings
    openrouter_api_key="sk-or-...",
    do_api_key="...",
)
```

## Source formats

`source=` accepts any of:

- **A file path** — `.pdf`, `.docx`, `.txt`, `.md`, images (OCR)
- **A URL** — `http(s)://...` (scraped), Google Docs URLs/IDs
- **Raw text** — any string with a newline OR longer than 500 chars is treated as already-loaded content
- **A list of any of the above** — they all get concatenated

```python
# All of these work:
source="docs/asc606.pdf"
source="https://example.com/spec"
source="""ASC 606 governs revenue recognition.\nThe 5-step model is..."""  # raw text
source=["docs/asc606.pdf", "docs/asc842.pdf", "https://example.com/spec"]
```

## Output formats

| `data_type` | Shape per record |
|---|---|
| `"sft"` | `{messages: [{role, content}, ...]}` |
| `"dpo"` | `{prompt, chosen, rejected}` |
| `"verifiable_qa"` | `{prompt, ground_truth, verification_type}` |
| `"ranked"` | `{prompt, ranked_responses: [{rank, text, quality_score}], topic, subtopic}` |
| `"full"` | `{prompt, gold_answer, chosen, rejected, ranked_responses, topic, subtopic}` |

`"full"` is the canonical Posty format — one record carries SFT (`gold_answer`),
DPO (`chosen`/`rejected`), and GRPO (`ranked_responses`) views. Generated
once, no extra LLM calls beyond what `"ranked"` would use.

## Consuming results

`sdk.generate(...)` returns a `GeneratedDataset`:

```python
dataset.samples           # List[DatasetSample] — pydantic models
dataset.total_samples     # int
dataset.average_quality   # float (1-10)
dataset.total_cost        # float (USD)
dataset.save(path)        # write to disk as JSONL
dataset.filter_by_quality(min_score=8.0)   # returns new GeneratedDataset
```

Each `DatasetSample` exposes:

```python
sample.data               # Pydantic model — call .model_dump() for plain dict
sample.metrics.quality_score    # float
sample.metrics.token_count
sample.metrics.generation_cost
```

For Posty, the most ergonomic pattern is:

```python
dataset = sdk.generate(...)
records = [s.data.model_dump() for s in dataset.samples]
# `records` is a List[dict] matching the output shape table above
```

### Metadata sidecar

`dataset.save("outputs/dataset.jsonl")` writes **two files**:

| File | Contents |
|---|---|
| `outputs/dataset.jsonl` | Clean training records (no metrics). One JSON object per line. |
| `outputs/dataset.metadata.jsonl` | Per-sample metadata, **same order** as the training file. One JSON object per line: `{idx, quality_score, topic, subtopic, token_count, generation_cost, model_used, regeneration_count, generation_time, timestamp}` |

Join them by line index (`idx`) for post-hoc QA: "what's the avg quality
per topic?", "which batches cost the most?", "did Revenue Recognition
underperform?". The sidecar is always written — there's no opt-out
because the cost is negligible and the data is otherwise lost when the
process exits.

`topic` and `subtopic` are populated when generation used a Gemini plan
(`enable_planning=True`); they're `null` for plan-less runs.

## Progress callbacks

Pass `progress_callback=fn` to receive structured events for your UI
(Rich progress bars, status spinners, websocket pushes, etc.). The
callback can be **sync or async**; exceptions raised inside it are logged
and swallowed so a buggy callback can never break generation.

```python
def on_event(e: dict) -> None:
    if e["event"] == "batch_completed":
        progress_bar.update(e["samples_generated"], total=e["samples_target"])

sdk = DataSimulator(..., progress_callback=on_event)
```

### Event reference

Each event is a flat `dict` whose `"event"` key tells you the type.

| `event` | Payload | When it fires |
|---|---|---|
| `generation_started` | `num_samples`, `data_type`, `quality_threshold`, `max_cost`, `num_planned_batches` (or `None`), `domain` (or `None`) | Once, at the very start. |
| `batch_completed` | `samples_in_batch`, `samples_passed`, `batch_cost`, `average_quality`, `samples_generated`, `samples_target`, `total_cost` | After each batch finishes (post-validation, post-quality-check). This is the event to drive a progress bar from. |
| `checkpoint_saved` | `samples_generated`, `checkpoint_dir` | After each checkpoint write (only if `checkpoint_dir` was set). |
| `cost_limit_reached` | `total_cost`, `max_cost`, `samples_generated`, `samples_target` | When the cost cap halts the run early. |
| `generation_completed` | `samples_generated`, `samples_target`, `failed`, `total_cost` | Once, at the very end (whether the run completed normally or stopped early). |

### Minimal sketch with Rich

```python
from rich.progress import Progress

with Progress() as progress:
    task = progress.add_task("Generating...", total=500)

    def on_event(e):
        if e["event"] == "batch_completed":
            progress.update(task, completed=e["samples_generated"])

    sdk = DataSimulator(..., progress_callback=on_event)
    sdk.generate(num_samples=500)
```

## Error handling

- API errors (auth, quota, network): the SDK logs them and falls back to
  regeneration or default scores. Catastrophic failures raise; wrap in
  `try/except` if you want Posty to handle them gracefully.
- Cost cap reached: in `interactive=False` (the default), generation stops
  cleanly and returns whatever was generated so far. Check
  `len(dataset.samples) < num_samples` to detect this.
- No records generated: `dataset.samples` will be empty. Surface a useful
  error to the user (probably their threshold is too high or sources too
  thin).

## Minimal Posty CLI integration sketch

```python
import click
from datasimulator import DataSimulator

@click.command()
@click.option("--source", multiple=True, required=True, help="File path or URL")
@click.option("--data-type", default="ranked")
@click.option("--num-samples", type=int, required=True)
@click.option("--max-cost", type=float, default=10.0)
@click.option("--emphasis", multiple=True, help="topic=weight, repeatable")
def generate(source, data_type, num_samples, max_cost, emphasis):
    topic_emphasis = dict(
        (t.split("=")[0].strip(), float(t.split("=")[1]))
        for t in emphasis
    ) or None

    sdk = DataSimulator(
        source=list(source),
        data_type=data_type,
        enable_planning=True,
        ranked_config={"num_responses": 4, "quality_spread": "wide"}
            if data_type in ("ranked", "full") else None,
        max_cost=max_cost,
    )
    dataset = sdk.generate(num_samples=num_samples, topic_emphasis=topic_emphasis)
    dataset.save("dataset.jsonl")
    click.echo(f"Wrote {len(dataset.samples)} records, ${dataset.total_cost:.2f} spent")

if __name__ == "__main__":
    generate()
```

## What this SDK guarantees

- ✅ Headless by default. No `input()` calls, no stdin reads, no progress
  prompts. Set `interactive=True` only if you want the cost-cap y/n prompt.
- ✅ Deterministic output shape per `data_type`. Pydantic-validated.
- ✅ All generation is async under the hood; `sdk.generate()` is a sync
  wrapper that drives the asyncio event loop for you.
- ✅ Single API key is enough — defaults route every role to OpenAI.

## What this SDK does NOT do (yet)

- ❌ Resumable generation across processes. Checkpoints write JSONL but
  there's no `resume_from=...` API yet — if a run dies, you re-run from
  scratch.
- ❌ Streaming output. `dataset.samples` is materialized at the end.
  Use `progress_callback` for live updates, not partial results.
- ❌ Per-sample events. `progress_callback` fires at batch granularity,
  not per record.
