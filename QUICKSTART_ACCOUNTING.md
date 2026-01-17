# Quick Start: Generate Accounting SFT Dataset

**Goal:** Load 10 accounting decks → Generate 2000-3000 SFT samples → Download training data

---

## 1️⃣ Upload Your Files

Place your 10 accounting documents here:
```
examples/accounting_docs/
```

Supported: PDF, DOCX, TXT files

---

## 2️⃣ Set API Keys

```bash
export ANTHROPIC_API_KEY="your_anthropic_key"
export OPENAI_API_KEY="your_openai_key"
export GOOGLE_API_KEY="your_google_key"
```

---

## 3️⃣ Run Generation

```bash
python examples/accounting_production_example.py
```

**This will:**
- ✅ Auto-detect all files in `accounting_docs/`
- ✅ Analyze documents with Gemini (extract topics)
- ✅ Generate 2500 SFT samples autonomously
- ✅ Save checkpoints every 20 samples
- ✅ Stop at $40 budget limit

**Estimated time:** 30-60 minutes (depends on API speed)

---

## 4️⃣ Download Your Dataset

**Output location:**
```
outputs/accounting_sft_dataset.jsonl
```

**Format:** One training example per line (JSONL)

**Example sample:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is the journal entry for recording accounts receivable?"
    },
    {
      "role": "assistant",
      "content": "The journal entry is:\nDR Accounts Receivable $X\n  CR Revenue $X\n\nThis records the sale on credit..."
    }
  ]
}
```

---

## 5️⃣ Monitor Progress

**While running:**
- Progress bar shows completion
- Cost tracker displays spend
- Quality scores shown in real-time

**If interrupted (Ctrl+C):**
- Progress saved in `checkpoints/`
- Re-run script to resume

**Analytics at end:**
- Total samples generated
- Average quality score
- Total cost
- Cost per sample

---

## 📊 Expected Results

**With $40 budget:**
- ~2,500-3,000 samples
- Quality: 6.0+ average
- Cost: ~$0.011/sample
- Time: 30-60 minutes

**Output file size:** ~5-10 MB (JSONL)

---

## 🔧 Customization

Edit `accounting_production_example.py` to adjust:

```python
TARGET_SAMPLES = 2500     # Change to 2000-3000
MAX_BUDGET = 40.0         # Increase if needed
quality_threshold=6.0     # Raise to 7.0 for higher quality
checkpoint_interval=20    # Save more/less frequently
```

---

## ❓ Troubleshooting

**No files found:**
- Check files are in `examples/accounting_docs/`
- Verify file extensions: .pdf, .docx, .txt

**Missing API keys:**
- Run `echo $ANTHROPIC_API_KEY` to verify
- Re-export if empty

**Cost limit reached:**
- Increase `MAX_BUDGET` in script
- Check `checkpoints/` for saved progress

**Low quality scores:**
- Review source documents (are they clear?)
- Increase `quality_threshold` to be more selective
- Check domain_context matches your content

---

## 🎯 Next Steps After Generation

1. **Review samples:**
   ```python
   from datasimulator import DataSimulator
   dataset = GeneratedDataset.load("outputs/accounting_sft_dataset.jsonl")
   dataset.sample_examples(10)
   ```

2. **Filter by quality:**
   ```python
   high_quality = dataset.filter_by_quality(7.5)
   high_quality.save("outputs/accounting_sft_high_quality.jsonl")
   ```

3. **Use for training:**
   - Load JSONL into your training framework
   - Compatible with OpenAI fine-tuning format
   - Ready for Anthropic/HuggingFace/Axolotl

---

**Ready to start? Run:**
```bash
python examples/accounting_production_example.py
```
