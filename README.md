# 🎯 DataSimulator

**Generate high-quality post-training datasets for fine-tuning small language models (SLMs)**

DataSimulator is an autonomous SDK that creates synthetic training data for SFT, DPO, PPO, GRPO, and RL from documents, images, and natural language prompts. Perfect for building domain-specific language models in accounting, finance, healthcare, and more.

---

## ✨ Features

- **Multiple Training Formats**: Generate data for SFT, DPO, PPO, GRPO, and RL with verifiable rewards
- **Autonomous Generation**: Specify target rows and let the system handle the rest
- **Quality Assurance**: Built-in quality scoring (1-10 scale) with configurable thresholds
- **Cost Controls**: Automatic stop at $20 with user prompts to continue
- **Batch Processing**: Generate 20+ rows per API call for efficiency
- **Multi-Model Support**: Use Claude, GPT-4, or local models (Ollama, Qwen)
- **Universal Document Loading**: 📄 PDFs, Word docs, images (OCR), web pages, Google Docs
- **Automatic Format Detection**: Smart loader selection based on file type or URL
- **Iterative Refinement**: Automatically regenerate low-quality samples
- **Domain Adaptation**: Extract patterns from source material
- **Rich Analytics**: Real-time dashboards and quality metrics

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/grantgriff/datasimulator.git
cd datasimulator

# Install dependencies
pip install -r requirements.txt

# Or install with pip (once published)
pip install datasimulator
```

### Setup API Keys

```bash
# Create .env file
cp .env.example .env

# Add your API keys
echo "ANTHROPIC_API_KEY=your_key_here" >> .env
echo "OPENAI_API_KEY=your_key_here" >> .env
```

### Basic Usage

```python
from datasimulator import DataSimulator

# Initialize with source document (supports PDF, Word, images, URLs, Google Docs)
sdk = DataSimulator(
    source="accounting_textbook.pdf",  # or .docx, .jpg, https://..., etc.
    data_type="sft",
    models={
        "generator": "claude-3-5-sonnet-20241022",
        "verifier": "gpt-4o-mini",
        "diversity": "qwen2.5:7b"  # Local model
    },
    quality_threshold=6.0,  # Minimum quality (1-10 scale)
    max_cost=20.0,  # Stop at $20
    batch_size=20  # Generate 20 rows per API call
)

# Generate dataset
dataset = sdk.generate(
    num_samples=1000,
    domain_context="Focus on accounts receivable and bad debt"
)

# View analytics
dataset.show_analytics()

# Save to file
dataset.save("accounting_sft_data.jsonl")
```

---

## 📊 Supported Data Formats

### SFT (Supervised Fine-Tuning)

**Messages Format:**
```json
{
  "messages": [
    {"role": "system", "content": "You are an accounting expert."},
    {"role": "user", "content": "How do I record a bad debt expense?"},
    {"role": "assistant", "content": "To record bad debt expense..."}
  ]
}
```

**Completion Format:**
```json
{
  "prompt": "Question: How do I calculate depreciation?\nAnswer:",
  "completion": "To calculate straight-line depreciation..."
}
```

### DPO (Direct Preference Optimization)

```json
{
  "prompt": "Explain the matching principle in accounting.",
  "chosen": "The matching principle requires expenses to be recorded in the same period as the revenues they helped generate...",
  "rejected": "The matching principle is about matching things together in accounting..."
}
```

### PPO (Proximal Policy Optimization)

```json
{
  "prompt": "Calculate the allowance for doubtful accounts given..."
}
```

### GRPO (Group Relative Policy Optimization)

```json
{
  "prompt": "What journal entry records an accounts receivable write-off?"
}
```

### RL with Verifiable Rewards

```json
{
  "prompt": "A company has $500,000 in receivables and estimates 3% will be uncollectible. What is the bad debt expense?",
  "ground_truth": "15000",
  "verification_type": "numeric_match"
}
```

---

## 🎨 Advanced Usage

### Use Different Models for Different Tasks

```python
sdk = DataSimulator(
    source="finance_docs.pdf",
    data_type="sft",
    models={
        "generator": "claude-3-5-sonnet-20241022",  # High quality
        "verifier": "gpt-4o-mini",  # Fast verification
        "diversity": "qwen2.5:7b"  # Free local model
    }
)
```

### Filter by Quality

```python
dataset = sdk.generate(num_samples=1000)

# Keep only high-quality samples (8+/10)
high_quality = dataset.filter_by_quality(min_score=8.0)
high_quality.save("high_quality_data.jsonl")
```

### Load from Different Document Types

```python
# PDF document
sdk = DataSimulator(source="guide.pdf", data_type="sft")

# Word document
sdk = DataSimulator(source="manual.docx", data_type="sft")

# Web page
sdk = DataSimulator(source="https://example.com/docs", data_type="sft")

# Image (OCR)
sdk = DataSimulator(source="scanned_doc.jpg", data_type="sft")

# Google Docs
sdk = DataSimulator(
    source="https://docs.google.com/document/d/DOC_ID/edit",
    data_type="sft"
)

# Text file
sdk = DataSimulator(source="notes.txt", data_type="sft")
```

### Standalone Document Loading

```python
from datasimulator import load_document

# Quick content extraction
text = load_document("document.pdf")
text = load_document("https://example.com")
text = load_document("image.jpg", language='eng')  # OCR
```

### Generate Without Source (Use Domain Context)

```python
sdk = DataSimulator(
    data_type="sft",
    models={"generator": "claude-3-5-sonnet-20241022"}
)

dataset = sdk.generate(
    num_samples=1000,
    domain_context="Generate advanced Python programming examples"
)
```

---

## 💰 Cost Management

The SDK automatically tracks costs and prompts when limits are reached:

```
💰 COST LIMIT REACHED: $20.00 / $20.00
========================================
Cost Breakdown:
  Generation  : $ 16.50 (82.5%)
  Verification: $  3.50 (17.5%)

  Total       : $ 20.00

Time elapsed: 145.3s
Cost per minute: $8.26/min
========================================

Continue generation? This will increase limit by $20.00. (y/n):
```

---

## 📈 Analytics Dashboard

View detailed analytics for your generated dataset:

```python
dataset.show_analytics()
```

Output:
```
============================================================
📊 DATASET ANALYTICS
============================================================
Data Type:         SFT
Total Samples:     1000
Average Quality:   7.8/10
Quality Range:     6.2 - 9.5
Avg Tokens/Sample: 245

Total Cost:        $18.50

Cost Breakdown:
  Generation  : $15.20 (82.2%)
  Verification: $ 3.30 (17.8%)
============================================================
```

---

## 🏗️ Project Structure

```
datasimulator/
├── core/
│   ├── generators/          # Data generation engines
│   │   ├── base_generator.py
│   │   ├── sft_generator.py
│   │   ├── dpo_generator.py (Phase 4)
│   │   └── ...
│   ├── models/
│   │   └── llm_client.py    # Multi-provider LLM client
│   └── data_models.py       # Pydantic schemas
│
├── sources/                 # Document loaders ✅
│   ├── base_loader.py       # Base loader class
│   ├── document_loader.py   # Unified loader
│   └── loaders/
│       ├── pdf_loader.py
│       ├── word_loader.py
│       ├── image_loader.py
│       ├── web_scraper.py
│       └── google_docs_loader.py
│
├── utils/
│   └── cost_tracker.py      # Cost tracking & limits
│
├── sdk.py                   # Main SDK interface
└── requirements.txt
```

---

## 🛣️ Roadmap

### ✅ Phase 1: Core Foundation (Complete)
- [x] Project structure
- [x] Pydantic data models
- [x] Multi-provider LLM client (Claude, OpenAI, Ollama)
- [x] Cost tracking with $20 limit
- [x] Base generator with quality scoring
- [x] SFT generator with batch processing
- [x] Main SDK interface

### ✅ Phase 2: Source Loading (Complete)
- [x] PDF loader (PyPDF2, pdfplumber)
- [x] Word document loader (.docx)
- [x] Google Docs integration
- [x] Image OCR (JPEG, PNG, TIFF)
- [x] Web scraping (BeautifulSoup + Playwright)
- [x] Automatic format detection
- [x] Unified DocumentLoader API

### 📋 Phase 3: Quality & Refinement
- [ ] Advanced quality scoring
- [ ] Diversity checking (semantic similarity)
- [ ] Human review interface
- [ ] Iterative refinement system

### 🎯 Phase 4: Additional Generators
- [ ] DPO generator (preference pairs)
- [ ] PPO generator (prompts only)
- [ ] GRPO generator (multi-completion)
- [ ] RL verifiable generator (ground truth)

### 📊 Phase 5: Analytics & Polish
- [ ] Real-time generation dashboard
- [ ] Quality distribution plots
- [ ] Export to multiple formats (Parquet, CSV)
- [ ] CLI interface
- [ ] Documentation & tutorials

---

## 🔧 Configuration

### Environment Variables

```bash
# API Keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Configuration
MAX_COST_USD=20.0
DEFAULT_BATCH_SIZE=20
QUALITY_THRESHOLD=6.0
DIVERSITY_THRESHOLD=0.85
```

### Programmatic Configuration

```python
from datasimulator import DataSimulator

sdk = DataSimulator(
    source="document.pdf",
    data_type="sft",

    # Model configuration
    models={
        "generator": "claude-3-5-sonnet-20241022",
        "verifier": "gpt-4o-mini",
        "diversity": "qwen2.5:7b"
    },

    # Quality settings
    quality_threshold=6.0,      # 1-10 scale
    diversity_threshold=0.85,   # 0-1 similarity

    # Cost controls
    max_cost=20.0,             # USD
    batch_size=20,             # Samples per API call

    # API keys (optional, uses env vars by default)
    anthropic_api_key="sk-ant-...",
    openai_api_key="sk-..."
)
```

---

## 📝 Examples

See the `examples/` directory for complete examples:

**Phase 1 (Core):**
- `examples/basic_sft_example.py` - Basic SFT generation
- `examples/with_source_document.py` - Generate from text files

**Phase 2 (Document Loaders):**
- `examples/pdf_loader_example.py` - Generate from PDF documents
- `examples/web_scraping_example.py` - Generate from web pages
- `examples/multiple_sources_example.py` - Combine multiple sources

Run any example:
```bash
python examples/pdf_loader_example.py
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Inspired by the need for high-quality, domain-specific training data
- Built for fine-tuning small language models (SLMs)
- Supports the open-source AI community

---

## 📬 Contact

Grant - [@grantgriff](https://github.com/grantgriff)

Project Link: [https://github.com/grantgriff/datasimulator](https://github.com/grantgriff/datasimulator)

---

## 🌟 Star History

If you find this project helpful, please consider giving it a star! ⭐
