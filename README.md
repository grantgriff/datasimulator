# DataSimulator

Generate high-quality post-training datasets for fine-tuning language models.

DataSimulator is an SDK that creates synthetic training data for SFT, DPO, PPO, GRPO, and RL from documents, images, and natural language prompts.

## Features

- **Multiple Training Formats**: SFT, DPO, PPO, GRPO, and RL with verifiable rewards
- **Autonomous Generation**: Specify target rows and let the system handle the rest
- **Quality Assurance**: Built-in quality scoring with configurable thresholds
- **Cost Controls**: Automatic tracking with configurable limits
- **Batch Processing**: Generate 20+ rows per API call for efficiency
- **Multi-Model Support**: Claude, OpenAI, Ollama, and local models
- **Universal Document Loading**: PDFs, Word docs, images (OCR), web pages, Google Docs
- **Automatic Format Detection**: Smart loader selection based on file type
- **Iterative Refinement**: Automatically regenerate low-quality samples
- **Analytics Dashboard**: Real-time quality metrics and visualizations

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

sdk = DataSimulator(
    source="document.pdf",
    data_type="sft",
    models={
        "generator": "claude-3-5-sonnet-20241022",
        "verifier": "gpt-4o-mini"
    },
    quality_threshold=6.0,
    max_cost=20.0
)

dataset = sdk.generate(num_samples=1000)
dataset.save("output.jsonl")
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

### PPO (Proximal Policy Optimization)

```json
{
  "prompt": "Question or task"
}
```

### GRPO (Group Relative Policy Optimization)

```json
{
  "prompt": "Question or task requiring multiple completions"
}
```

### RL with Verifiable Rewards

```json
{
  "prompt": "Question with verifiable answer",
  "ground_truth": "correct_answer",
  "verification_type": "exact_match"
}
```

## Usage Examples

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

The SDK automatically tracks costs and prompts when limits are reached:

```
COST LIMIT REACHED: $20.00 / $20.00
========================================
Cost Breakdown:
  Generation  : $ 16.50 (82.5%)
  Verification: $  3.50 (17.5%)
  Total       : $ 20.00

Continue? This will increase limit by $20.00. (y/n):
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
MAX_COST_USD=20.0
DEFAULT_BATCH_SIZE=20
QUALITY_THRESHOLD=6.0
```

### Programmatic Configuration

```python
sdk = DataSimulator(
    source="document.pdf",
    data_type="sft",
    models={
        "generator": "claude-3-5-sonnet-20241022",
        "verifier": "gpt-4o-mini"
    },
    quality_threshold=6.0,
    max_cost=20.0,
    batch_size=20,
    anthropic_api_key="...",
    openai_api_key="..."
)
```

## Examples

See the `examples/` directory:

- `examples/basic_sft_example.py` - Basic SFT generation
- `examples/pdf_loader_example.py` - Generate from PDFs
- `examples/web_scraping_example.py` - Generate from web pages
- `examples/all_generators_example.py` - All format types
- `examples/dpo_example.py` - DPO preference pairs
- `examples/rl_verifiable_example.py` - RL with ground truth

Run any example:
```bash
python examples/basic_sft_example.py
```

## Project Structure

```
datasimulator/
├── core/
│   ├── generators/              # Data generation engines
│   │   ├── base_generator.py
│   │   ├── sft_generator.py
│   │   ├── dpo_generator.py
│   │   ├── ppo_generator.py
│   │   ├── grpo_generator.py
│   │   └── rl_verifiable_generator.py
│   ├── models/
│   │   └── llm_client.py        # Multi-provider LLM client
│   └── data_models.py           # Pydantic schemas
├── sources/                     # Document loaders
│   ├── base_loader.py
│   ├── document_loader.py
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
│   └── cost_tracker.py
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
