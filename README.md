# DataSimulator

Generate high-quality post-training datasets for fine-tuning language models.

DataSimulator is an SDK that creates synthetic training data for SFT, DPO, and verifiable Q&A from documents, images, and natural language prompts.

---

## 🚀 Quick Start - Production Example

The easiest way to get started is using the **production example script** that generates datasets from your documents.

### Step 1: Install Dependencies

```bash
git clone https://github.com/grantgriff/datasimulator.git
cd datasimulator
pip install -r requirements.txt
```

### Step 2: Set Up API Keys

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

### Step 3: Add Your Documents

Place your training documents in a folder:

```bash
mkdir examples/my_docs
# Copy your PDFs, Word docs, or text files to examples/my_docs/
```

Supported formats: `.pdf`, `.docx`, `.txt` - **All files in the folder will be used!**

### Step 4: Configure Generation Settings

Open `examples/accounting_production_example.py` and configure these variables:

```python
# Line 81: Number of samples to generate
TARGET_SAMPLES = 1500  # Change to your desired number (40 for testing, 1500-2500 for production)

# Line 82: Budget limit
MAX_BUDGET = 40.0  # Maximum spend in USD (script stops if exceeded)

# Line 100: Data type - choose ONE
data_type="sft",  # Options: "sft", "dpo", "verifiable_qa"

# Lines 104-106: Model selection (IMPORTANT: affects cost and quality)
models={
    "generator": "claude-sonnet-4-5-20250929",  # Options: see Model Selection below
    "verifier": "gpt-4o-mini-2024-07-18",       # Quality scoring (recommend keeping this)
},

# Line 109: Quality threshold
quality_threshold=6.0,  # Accept samples scored 6.0-10.0 (lower = more samples, less quality)

# Line 110: Batch size
batch_size=10,  # Samples per batch (10 is safe for 64k token limit)

# Line 39: Document folder (if you created a different folder)
docs_dir = Path("examples/my_docs")  # Change to your folder path
```

### Step 5: Run Generation

```bash
python examples/accounting_production_example.py
```

The script will:
1. ✅ **Load documents** from your folder
2. ✅ **Check API keys** are configured
3. ✅ **Create generation plan** with Gemini (extracts topics from docs)
4. ✅ **Generate batches** of 10 samples using Claude Sonnet
5. ✅ **Quality check** each batch with GPT-4o-mini
6. ✅ **Save passing samples** (≥6.0/10 quality score)
7. ✅ **Checkpoint every 20 samples** for crash recovery
8. ✅ **Display final analytics** (cost, quality, sample count)

**Output files:**
- `outputs/accounting_sft_dataset.jsonl` - Your training dataset
- `checkpoints/` - Intermediate saves (for resuming if interrupted)

---

## 📊 Process Flow

Here's how DataSimulator works under the hood:

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. DOCUMENT LOADING                                             │
│    • Reads all docs from folder (PDFs, DOCX, TXT)              │
│    • Extracts text content (37,872 chars total)                │
│    • Stores per-file for targeted generation                   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. GEMINI PLANNING (Topic Extraction)                          │
│    • Analyzes all document content                             │
│    • Extracts major topics and subtopics                       │
│    • Creates batch plan: 150 batches × 10 samples = 1500      │
│    • Assigns relevant docs to each batch                       │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. BATCH GENERATION (Claude Sonnet)                            │
│    Batch 1/150: "Financial Statements → Balance Sheet"         │
│    • Generates 10 samples on this specific topic               │
│    • Uses only relevant documents as source                    │
│    • 64k token limit, ~6-8 minutes per batch                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. QUALITY SCORING (GPT-4o-mini)                               │
│    • Scores each sample 1-10 for quality                       │
│    • Checks: accuracy, completeness, relevance, clarity        │
│    • Batch processing (10 samples at once)                     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. FILTERING & REGENERATION                                     │
│    • Keep samples with score ≥ 6.0 (quality threshold)         │
│    • Regenerate failed samples (up to 10 retries)              │
│    • Only stops after 3 consecutive failures                   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. SAVE & CHECKPOINT                                            │
│    • Saves passing samples to outputs/dataset.jsonl            │
│    • Checkpoints every 20 samples (crash recovery)             │
│    • Final analytics displayed                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Configuration Reference

All configurable variables in `examples/accounting_production_example.py`:

| Variable | Line | Options | Default | Description |
|----------|------|---------|---------|-------------|
| **TARGET_SAMPLES** | 81 | Any integer | 1500 | Number of samples to generate |
| **MAX_BUDGET** | 82 | Any float | 40.0 | Max cost in USD before stopping |
| **data_type** | 100 | `"sft"`, `"dpo"`, `"verifiable_qa"` | `"sft"` | Training data format |
| **generator** | 104 | See Model Selection | `"claude-sonnet-4-5-20250929"` | Main generation model |
| **verifier** | 105 | See Model Selection | `"gpt-4o-mini-2024-07-18"` | Quality scoring model |
| **quality_threshold** | 109 | 1.0 - 10.0 | 6.0 | Minimum quality score to accept |
| **batch_size** | 110 | 5 - 20 | 10 | Samples per batch (10 is safe) |
| **checkpoint_interval** | 118 | Any integer | 20 | Save progress every N samples |
| **docs_dir** | 39 | Any path | `"examples/accounting_docs"` | Folder with training documents |

### Model Selection

**Generator Models (Line 104):**

| Model | Cost | Speed | Quality | Recommendation |
|-------|------|-------|---------|----------------|
| `claude-sonnet-4-5-20250929` | $$$ | Slow | Excellent | Production (complex responses) |
| `claude-3-5-haiku-20241022` | $ | Fast | Good | Cost-effective (simpler responses) |
| `gpt-4o-2024-08-06` | $$ | Medium | Very Good | Alternative to Sonnet |
| `gpt-4o-mini-2024-07-18` | $ | Fast | Good | Budget option |

**Cost Comparison (1500 samples):**
- **Sonnet 4.5:** ~$16.50 (best quality)
- **Haiku 3.5:** ~$4.00 (4x cheaper, good quality)
- **GPT-4o-mini:** ~$2.50 (cheapest, decent quality)

**Verifier Models (Line 105):**

Recommendation: **Keep `gpt-4o-mini-2024-07-18`** - it's cheap ($0.150/M tokens) and accurate for quality scoring.

### Dataset Types

**SFT (Supervised Fine-Tuning)** - Line 100: `data_type="sft"`
- Format: System/User/Assistant messages
- Use case: Training conversational models
- Example: Chatbot training data

**DPO (Direct Preference Optimization)** - Line 100: `data_type="dpo"`
- Format: Prompt + Chosen/Rejected responses
- Use case: Aligning models to preferences
- Example: Helpful vs unhelpful responses

**Verifiable Q&A** - Line 100: `data_type="verifiable_qa"`
- Format: Question + Ground truth answer
- Use case: Training models with verifiable correctness
- Example: Math problems, fact-based QA

---

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
