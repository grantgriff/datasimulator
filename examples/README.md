# DataSimulator Examples

This directory contains examples demonstrating various features of the DataSimulator SDK.

## Phase 1 Examples (Core Functionality)

### `basic_sft_example.py`
Basic SFT data generation without source documents.
```bash
python examples/basic_sft_example.py
```

### `with_source_document.py`
Generate data from a text source file.
```bash
python examples/with_source_document.py
```

## Phase 2 Examples (Document Loaders)

### `pdf_loader_example.py`
Load and generate data from PDF documents.
```bash
python examples/pdf_loader_example.py
```

**Requirements:**
```bash
pip install PyPDF2 pdfplumber
```

### `web_scraping_example.py`
Scrape web pages and generate training data.
```bash
python examples/web_scraping_example.py
```

**Requirements:**
```bash
pip install requests beautifulsoup4
```

For JavaScript-heavy sites:
```bash
pip install playwright
playwright install chromium
```

### `multiple_sources_example.py`
Combine multiple documents into one dataset.
```bash
python examples/multiple_sources_example.py
```

## Other Loaders (Not in Examples Yet)

### Word Documents
```python
from datasimulator import DataSimulator

sdk = DataSimulator(
    source="document.docx",
    data_type="sft"
)
```

**Requirements:**
```bash
pip install python-docx
```

### Images (OCR)
```python
from datasimulator import load_document

# Extract text from image
text = load_document("document.jpg", language='eng')
```

**Requirements:**
```bash
# Install Python packages
pip install pytesseract Pillow

# Install Tesseract OCR
# Ubuntu/Debian: sudo apt-get install tesseract-ocr
# macOS: brew install tesseract
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
```

### Google Docs
```python
from datasimulator import DataSimulator

# Set up Google credentials first
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"

sdk = DataSimulator(
    source="https://docs.google.com/document/d/DOCUMENT_ID/edit",
    data_type="sft"
)
```

**Requirements:**
```bash
pip install google-auth google-api-python-client
```

**Setup:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project
3. Enable Google Docs API
4. Create Service Account credentials
5. Download credentials JSON
6. Set `GOOGLE_APPLICATION_CREDENTIALS` environment variable

## Running Examples

### Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up API keys:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Run any example:
```bash
python examples/basic_sft_example.py
```

## Example Output

All examples save generated datasets to the `outputs/` directory:
- `outputs/basic_sft_example.jsonl`
- `outputs/pdf_based_sft.jsonl`
- `outputs/web_based_sft.jsonl`
- `outputs/multi_source_sft.jsonl`

## Supported File Formats

| Format | Extension | Loader | Requirements |
|--------|-----------|--------|--------------|
| Text | `.txt`, `.md` | TextLoader | None |
| PDF | `.pdf` | PDFLoader | PyPDF2, pdfplumber |
| Word | `.docx` | WordLoader | python-docx |
| Images | `.jpg`, `.png` | ImageLoader | pytesseract, Pillow |
| Web | `http://`, `https://` | WebScraperLoader | requests, beautifulsoup4 |
| Web (JS) | `http://`, `https://` | JavaScriptWebLoader | playwright |
| Google Docs | Google Docs URL | GoogleDocsLoader | google-auth, google-api-python-client |

## Tips

- **Start small**: Use `max_cost=3.0` and `num_samples=10` for testing
- **Quality threshold**: Adjust `quality_threshold` based on your needs (default: 6.0)
- **Batch size**: Increase `batch_size` for faster generation (default: 20)
- **Domain context**: Provide specific context for better-targeted data
- **Multiple sources**: Combine related documents for comprehensive coverage

## Need Help?

Check the main [README.md](../README.md) for full documentation or open an issue on GitHub.
