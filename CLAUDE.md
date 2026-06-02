# Working agreements for this repo

Durable rules and gotchas to follow when working on `datasimulator`. Read this
before making changes.

## Never use low `max_tokens` defaults

**Rule:** when calling any LLM, set `max_tokens` to a high value that
comfortably covers the largest plausible response. Default to **at least
16000** for any call that returns structured JSON, batch responses, or
multi-record output. Use higher (32000-128000) for planner/aggregator calls
that emit large structured documents.

**Why:** every time a `max_tokens` cap has been hit in this codebase,
responses get truncated mid-JSON, parsers fail silently, fallback paths
return default scores (typically 5.0), and entire batches get dropped
without obvious error. Cost is identical when you set a high cap and the
model only emits a small response — you only pay for actual output tokens.
There is no upside to a tight cap.

**Exceptions where low caps are correct:**
- Single-token / single-number outputs (e.g. "Output a number from 1-10"
  with `max_tokens=10`). Keep these tight — the cap is the validator.

**When in doubt:** higher is safer. If a model's actual output ceiling
is lower than what you asked for (some OpenAI models cap output at
16K-32K regardless of `max_tokens`), the request still succeeds, the
model just emits up to its own limit.

## When batches truncate, two things are usually wrong

1. `max_tokens` is set too low on the call (most common — see rule above).
2. `BATCH_SIZE` is too high for the density of content the model is
   generating (e.g. 20 dense accounting records of ~1.5KB each = 30KB+
   of JSON, which can hit model-side output caps even with high
   `max_tokens`). Recommend reducing `BATCH_SIZE` to 5-10 for dense
   content.

## Other durable rules

- `interactive=False` is the SDK default. Never re-enable it as a default —
  the SDK is a headless backend; CLI tools (e.g. Posty) own all user
  interaction.
- New SDK inputs go in `INTEGRATION.md` and the Posty handoff doc
  (`POSTY_HANDOFF.md`) when they're added. The contract there is what
  downstream tools rely on.
- Progress events go through `progress_callback`. Never add new print-based
  status that doesn't also emit an event.
- Roadmap items in `README.md` (resumable runs, streaming output, chunked
  planner output, per-sample events) are the agreed deferral list. Don't
  silently implement them without checking with Grant.
