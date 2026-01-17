# Cost Estimate for Accounting SFT Dataset

## Your Target: 2000-3000 Samples, $40 Budget

---

## Cost Breakdown (per sample)

### With Current Configuration:
- **Generator:** Claude 3.5 Sonnet (~8K tokens output)
  - Cost: ~$0.008/sample
- **Verifier:** GPT-4o-mini (batched quality scoring)
  - Cost: ~$0.0001/sample (50 samples per API call)
- **Planning:** Gemini 1.5 Pro (one-time analysis)
  - Cost: ~$0.05 total (not per sample)

**Total cost per sample:** ~$0.011

---

## Budget Scenarios

### Conservative (2000 samples):
- **Estimated cost:** $22.00
- **Buffer remaining:** $18.00 (45%)
- **Generation time:** 25-35 minutes

### Target (2500 samples):
- **Estimated cost:** $27.50
- **Buffer remaining:** $12.50 (31%)
- **Generation time:** 30-45 minutes

### Aggressive (3000 samples):
- **Estimated cost:** $33.00
- **Buffer remaining:** $7.00 (17%)
- **Generation time:** 40-60 minutes

---

## Cost Optimizations Applied

### 1. Batched Quality Scoring ✅
- **Old:** 1 API call per sample = 2500 calls
- **New:** 1 API call per 50 samples = 50 calls
- **Savings:** 98% reduction in verification calls

### 2. Single Planning Pass ✅
- Gemini analyzes all 10 documents once
- Extracts topics and allocates samples upfront
- **Cost:** ~$0.05 total (negligible)

### 3. Checkpointing ✅
- Saves progress every 20 samples
- Prevents re-generating if interrupted
- **Savings:** Avoid duplicate work on crashes

### 4. Retry Limits ✅
- Max 10 retries per failed sample
- Prevents infinite regeneration loops
- **Savings:** Caps worst-case retry costs

---

## Real-World Example

**If you generate 2500 samples:**

```
Planning (Gemini):          $0.05
Generation (Claude):        $20.00  (2500 × $0.008)
Quality Scoring (GPT):      $0.25   (50 batches × $0.005)
Diversity Checks (Ollama):  $0.00   (local model)
────────────────────────────────────
Total:                      $20.30

Actual total with retries:  $24-28  (10-20% retry rate)
```

**Your $40 budget covers 2500 samples comfortably.**

---

## Cost vs. Quality Tradeoff

### Lower Cost ($15-20 for 2500 samples):
```python
quality_threshold=5.0,    # Accept lower quality
batch_size=50,            # Larger batches
models={
    "generator": "claude-3-5-sonnet-20241022",
    # Remove verifier to skip quality checks
}
```

### Higher Quality ($30-40 for 2500 samples):
```python
quality_threshold=7.5,    # Only accept high quality
batch_size=10,            # Smaller batches, more focused
models={
    "generator": "claude-3-5-sonnet-20241022",
    "verifier": "gpt-4o",  # Use full GPT-4o (more expensive)
}
```

### Recommended (Current Setup):
```python
quality_threshold=6.0,    # Balanced quality
batch_size=20,            # Standard batch size
models={
    "generator": "claude-3-5-sonnet-20241022",
    "verifier": "gpt-4o-mini",  # Cheap verification
}
```

---

## FAQ

**Q: Why does actual cost differ from estimate?**
- Retries for failed samples (typically 10-20%)
- Token count variance (some samples longer than average)
- Planning overhead (minimal, ~$0.05)

**Q: How can I reduce costs?**
1. Remove verifier (skip quality checks)
2. Lower quality_threshold to 5.0
3. Use smaller batches (batch_size=50)
4. Disable planning (enable_planning=False)

**Q: How can I increase quality?**
1. Raise quality_threshold to 7.5
2. Use GPT-4o for verification (higher cost)
3. Reduce batch_size to 10 (more focused)
4. Enable planning for better topic coverage

**Q: What if I hit the $40 limit mid-generation?**
- Generation stops gracefully
- Progress saved in checkpoints/
- Re-run script to continue (increase max_cost)

---

## Monitoring Costs During Generation

The script displays real-time cost tracking:

```
Cost Tracker: $12.45 / $40.00 (31%)
Samples: 1140 / 2500 (46%)
Cost per sample: $0.011
```

**Green flags:**
- Cost per sample stays ~$0.011
- Progress matches cost proportionally

**Red flags:**
- Cost per sample > $0.020 (too many retries)
- Cost exceeds 50% before 50% progress (adjust expectations)

---

## Summary

**Your $40 budget is sufficient for:**
- ✅ 2000-3000 SFT samples
- ✅ High quality (6.0+ threshold)
- ✅ Gemini planning with topic extraction
- ✅ Quality verification on all samples
- ✅ Checkpointing for crash recovery

**Expected outcome:**
- 2500 samples for ~$25-28
- Average quality: 6.5-7.5/10
- Generation time: 30-45 minutes
- Buffer remaining: $12-15
