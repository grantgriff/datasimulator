# Accounting Documents Upload Directory

## Instructions

**Place your 10 accounting PDF/DOCX files here.**

Supported formats:
- PDF files (.pdf)
- Word documents (.docx)
- Text files (.txt)

## Example structure:
```
accounting_docs/
├── financial_accounting_chapter1.pdf
├── financial_accounting_chapter2.pdf
├── managerial_accounting.pdf
├── accounts_receivable_guide.docx
├── revenue_recognition.pdf
├── financial_statements.pdf
├── journal_entries_examples.pdf
├── audit_procedures.pdf
├── tax_accounting.pdf
└── cost_accounting.pdf
```

## Once files are uploaded:

Run the production script:
```bash
python examples/accounting_production_example.py
```

The script will:
1. Auto-detect all files in this directory
2. Load and analyze all documents with Gemini
3. Generate 2000-3000 SFT samples autonomously
4. Save output to: `outputs/accounting_sft_dataset.jsonl`
