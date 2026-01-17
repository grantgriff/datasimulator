# DataSimulator

Generate high-quality post-training datasets for fine-tuning language models.

DataSimulator is an SDK that creates synthetic training data for SFT, DPO, and verifiable Q&A from documents, images, and natural language prompts.

## Features

- **Multiple Training Formats**: SFT, DPO, and verifiable Q&A
- **Multi-File Loading**: Load 10+ documents simultaneously for comprehensive training data
- **Gemini Planning**: AI-powered analysis extracts 5-50 topics and allocates samples intelligently
- **Autonomous Generation**: Non-interactive mode for large-scale unattended generation
- **Batched Quality Checks**: 50 samples per API call reduces verification cost by 50x
- **Checkpointing**: Auto-save every 20 samples for crash recovery
- **Retry Limits**: Max 10 retries prevents infinite regeneration loops
- **Cost Controls**: Automatic tracking with configurable limits
- **Multi-Model Support**: Claude, OpenAI, Ollama, and local models
- **Universal Document Loading**: PDFs, Word docs, images (OCR), web pages, Google Docs
- **Analytics Dashboard**: Real-time quality metrics and cost breakdown

## Installation

```bash
git clone https://github.com/grantgriff/datasimulator.git
cd datasimulator
pip install -r requirements.txt
```

## Setup

```bash
cp .env.example .env
# Add your API keys to .env
```

## Quick Start

```python
from datasimulator import DataSimulator

# Load multiple files with Gemini planning
sdk = DataSimulator(
    source=["doc1.pdf", "doc2.pdf", "doc3.pdf"],
    data_type="sft",
    models={
        "generator": "claude-3-5-sonnet-20241022",
        "verifier": "gpt-4o-mini"
    },
    quality_threshold=6.0,
    max_cost=40.0,
    interactive=False,          # Autonomous mode
    enable_planning=True,       # Gemini topic extraction
    checkpoint_dir="checkpoints"
)

dataset = sdk.generate(num_samples=2500)
dataset.save("output.jsonl")
dataset.show_analytics()
```

## Supported Data Formats

### SFT (Supervised Fine-Tuning)

**Messages Format:**
```json
{
  "messages": [
    {"role": "system", "content": "You are an expert."},
    {"role": "user", "content": "Question here"},
    {"role": "assistant", "content": "Answer here"}
  ]
}
```

**Completion Format:**
```json
{
  "prompt": "Question: ...",
  "completion": "Answer: ..."
}
```

### DPO (Direct Preference Optimization)

```json
{
  "prompt": "Question or instruction",
  "chosen": "High quality response",
  "rejected": "Lower quality response"
}
```

### Verifiable Q&A

```json
{
  "prompt": "Question with verifiable answer",
  "ground_truth": "correct_answer",
  "verification_type": "exact_match"
}
```

## Usage Examples

### Load Multiple Files with Gemini Planning

```python
# Load 10+ documents and let Gemini analyze them
accounting_files = [
    "docs/financial_accounting.pdf",
    "docs/managerial_accounting.pdf",
    "docs/accounts_receivable.pdf",
    # ... up to 10+ files
]

sdk = DataSimulator(
    source=accounting_files,  # Pass list of files
    data_type="sft",
    models={
        "generator": "claude-3-5-sonnet-20241022",
        "verifier": "gpt-4o-mini"
    },
    enable_planning=True,  # Gemini extracts topics
    google_api_key="YOUR_KEY"
)
```

### Autonomous Generation with Checkpointing

```python
sdk = DataSimulator(
    source=["doc1.pdf", "doc2.pdf"],
    data_type="sft",
    max_cost=40.0,         # Set high budget upfront
    interactive=False,     # No prompts - fully autonomous
    checkpoint_dir="checkpoints",
    checkpoint_interval=20  # Save every 20 samples
)

dataset = sdk.generate(num_samples=2500)
```

### Different Models per Task

```python
sdk = DataSimulator(
    source="docs.pdf",
    data_type="sft",
    models={
        "generator": "claude-3-5-sonnet-20241022",
        "verifier": "gpt-4o-mini",
        "diversity": "qwen2.5:7b"
    }
)
```

### Load from Various Sources

```python
# PDF, Word, text
sdk = DataSimulator(source="guide.pdf", data_type="sft")
sdk = DataSimulator(source="manual.docx", data_type="sft")

# Web scraping
sdk = DataSimulator(source="https://example.com/docs", data_type="sft")

# Image OCR
sdk = DataSimulator(source="scanned.jpg", data_type="sft")

# Google Docs
sdk = DataSimulator(
    source="https://docs.google.com/document/d/DOC_ID/edit",
    data_type="sft"
)
```

### Standalone Document Loading

```python
from datasimulator import load_document

text = load_document("document.pdf")
text = load_document("https://example.com")
text = load_document("image.jpg", language='eng')
```

### Generate Without Source

```python
sdk = DataSimulator(
    data_type="sft",
    models={"generator": "claude-3-5-sonnet-20241022"}
)

dataset = sdk.generate(
    num_samples=1000,
    domain_context="Generate Python programming examples"
)
```

### Filter by Quality

```python
dataset = sdk.generate(num_samples=1000)
high_quality = dataset.filter_by_quality(min_score=8.0)
high_quality.save("high_quality.jsonl")
```

## Cost Management

The SDK automatically tracks costs with two modes:

**Interactive Mode (default):** Prompts when limits are reached
```
COST LIMIT REACHED: $20.00 / $20.00
========================================
Cost Breakdown:
  Generation  : $ 16.50 (82.5%)
  Verification: $  3.50 (17.5%)
  Total       : $ 20.00

Continue? This will increase limit by $20.00. (y/n):
```

**Non-Interactive Mode:** For autonomous generation
```python
sdk = DataSimulator(
    source=files,
    max_cost=40.0,      # Set high limit upfront
    interactive=False   # No prompts
)
```

## Analytics

```python
dataset.show_analytics()
```

Output:
```
============================================================
DATASET ANALYTICS
============================================================
Data Type:         SFT
Total Samples:     1000
Average Quality:   7.8/10
Quality Range:     6.2 - 9.5

Total Cost:        $18.50
Cost Breakdown:
  Generation  : $15.20 (82.2%)
  Verification: $ 3.30 (17.8%)
============================================================
```

## Configuration

### Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...  # For Gemini planning
MAX_COST_USD=20.0
DEFAULT_BATCH_SIZE=20
QUALITY_THRESHOLD=6.0
```

### Programmatic Configuration

```python
sdk = DataSimulator(
    source=["doc1.pdf", "doc2.pdf"],  # Single file or list
    data_type="sft",
    models={
        "generator": "claude-3-5-sonnet-20241022",
        "verifier": "gpt-4o-mini"
    },
    quality_threshold=6.0,
    max_cost=40.0,
    batch_size=20,
    interactive=False,           # Autonomous mode
    checkpoint_dir="checkpoints",
    checkpoint_interval=20,       # Save every 20 samples
    enable_planning=True,         # Gemini topic extraction
    anthropic_api_key="...",
    openai_api_key="...",
    google_api_key="..."
)
```

## Examples

See the `examples/` directory:

- `examples/accounting_production_example.py` - Production-ready: Load 10 docs, generate 2000-3000 samples
- `examples/autonomous_batch_example.py` - Multi-file autonomous generation with checkpointing
- `examples/basic_sft_example.py` - Basic SFT generation
- `examples/pdf_loader_example.py` - Generate from PDFs
- `examples/web_scraping_example.py` - Generate from web pages
- `examples/dpo_example.py` - DPO preference pairs
- `examples/verifiable_qa_example.py` - Verifiable Q&A

**Quick Start Guide:** See `QUICKSTART_ACCOUNTING.md` for production use case

Run any example:
```bash
python examples/accounting_production_example.py
```

## Project Structure

```
datasimulator/
├── core/
│   ├── generators/              # Data generation engines
│   │   ├── base_generator.py   # Batched quality checks, retry limits
│   │   ├── sft_generator.py
│   │   ├── dpo_generator.py
│   │   └── verifiable_qa_generator.py
│   ├── models/
│   │   └── llm_client.py        # Multi-provider LLM client
│   └── data_models.py           # Pydantic schemas
├── planning/                    # Gemini planning layer
│   └── gemini_planner.py        # Topic extraction, chunking, allocation
├── sources/                     # Document loaders
│   ├── base_loader.py
│   ├── document_loader.py       # Multi-file support
│   └── loaders/
│       ├── pdf_loader.py
│       ├── word_loader.py
│       ├── image_loader.py
│       ├── web_scraper.py
│       └── google_docs_loader.py
├── quality/                     # Quality assurance
│   ├── quality_scorer.py
│   ├── diversity_checker.py
│   └── validators.py
├── refinement/
│   ├── iterative_refiner.py
│   └── human_review.py
├── analytics/
│   └── visualizations.py
├── utils/
│   └── cost_tracker.py          # Interactive & non-interactive modes
├── sdk.py                       # Main SDK interface
└── requirements.txt
```

## Development Status

Version: 1.0.0

- Phase 1: Core foundation (Complete)
- Phase 2: Document loading (Complete)
- Phase 3: Quality & refinement (Complete)
- Phase 4: All generators (Complete)

## License

MIT License - see LICENSE file for details.

## Contact

Grant - [@grantgriff](https://github.com/grantgriff)

Project: [https://github.com/grantgriff/datasimulator](https://github.com/grantgriff/datasimulator)
