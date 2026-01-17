# Generator Prompt Template

This is the EXACT prompt that will be sent to Claude for EACH batch of 20 samples when using Gemini planning.

## Example: Batch 15 of 125 (for 2500 total samples)

### Gemini Plan Output for This Batch:
```json
{
  "batch_number": 15,
  "topic": "Accounts Receivable",
  "subtopic": "Bad Debt Write-offs and Allowance Method",
  "guidance": "Generate 20 samples focusing on the allowance method for bad debts. Include questions about estimating uncollectible accounts, recording bad debt expense, writing off specific accounts, and recovery of previously written-off accounts. Cover both the allowance method and direct write-off method, emphasizing when each is appropriate. Vary difficulty from basic definitions to complex journal entry scenarios.",
  "relevant_files": [
    "docs/financial_accounting_chapter5.pdf",
    "docs/accounts_receivable_guide.pdf"
  ],
  "focus_areas": [
    "Allowance for Doubtful Accounts",
    "Bad Debt Expense calculation",
    "Write-off journal entries",
    "Recovery of written-off accounts",
    "Percentage of sales vs aging method"
  ]
}
```

---

## Prompt Sent to Claude Sonnet:

```
You are an expert at generating high-quality training data for fine-tuning language models.

=== GENERATION TASK ===

DATA TYPE: SFT (Supervised Fine-Tuning)
FORMAT: Messages (system/user/assistant)
BATCH SIZE: 20 samples

=== TOPIC CONTEXT ===

MAJOR TOPIC: Accounts Receivable
SUBTOPIC: Bad Debt Write-offs and Allowance Method

FOCUS AREAS:
- Allowance for Doubtful Accounts
- Bad Debt Expense calculation
- Write-off journal entries
- Recovery of written-off accounts
- Percentage of sales vs aging method

GENERATION GUIDANCE:
Generate 20 samples focusing on the allowance method for bad debts. Include questions about estimating uncollectible accounts, recording bad debt expense, writing off specific accounts, and recovery of previously written-off accounts. Cover both the allowance method and direct write-off method, emphasizing when each is appropriate. Vary difficulty from basic definitions to complex journal entry scenarios.

=== SOURCE MATERIAL ===

RELEVANT DOCUMENTS:
- docs/financial_accounting_chapter5.pdf
- docs/accounts_receivable_guide.pdf

SOURCE CONTENT (excerpts from relevant files):
[The system will extract and include relevant sections from these 2 PDFs only,
not all 10 docs - keeping context focused on AR and bad debts]

--- EXCERPT FROM financial_accounting_chapter5.pdf ---

Chapter 5: Accounts Receivable and Bad Debts

The allowance method is the preferred approach under GAAP for handling uncollectible accounts. This method requires companies to estimate bad debts at the end of each period...

[Key concepts about allowance for doubtful accounts]
[Journal entry examples]
[Aging schedule examples]

--- EXCERPT FROM accounts_receivable_guide.pdf ---

Bad Debt Estimation Methods:

1. Percentage of Sales Method:
   - Estimates bad debt expense as % of credit sales
   - DR Bad Debt Expense
   - CR Allowance for Doubtful Accounts

2. Accounts Receivable Aging Method:
   - Estimates ending balance needed in allowance account
   - Analyzes AR by age (current, 30 days, 60 days, 90+ days)
   - Applies increasing percentages to older receivables

[Detailed examples and calculations]

=== GENERATION REQUIREMENTS ===

1. Generate EXACTLY 20 SFT training samples
2. Each sample must include:
   - system: Role description for the AI assistant
   - user: A question or instruction
   - assistant: A detailed, accurate response

3. **CRITICAL**: All samples MUST focus on the topic/subtopic above
   - Topic: Accounts Receivable
   - Subtopic: Bad Debt Write-offs and Allowance Method
   - Do NOT generate samples about other accounting topics

4. Use the FOCUS AREAS as your guide:
   - Allowance for Doubtful Accounts (questions about what it is, how it works)
   - Bad Debt Expense calculation (percentage of sales, aging method)
   - Write-off journal entries (how to record when account becomes uncollectible)
   - Recovery of written-off accounts (when customer pays after write-off)
   - Percentage of sales vs aging method (compare/contrast, when to use each)

5. Vary the difficulty across the 20 samples:
   - Samples 1-7: Basic/foundational (definitions, concepts, simple examples)
   - Samples 8-14: Intermediate (calculations, journal entries, method comparisons)
   - Samples 15-20: Advanced (complex scenarios, multi-step problems, analysis)

6. Vary the question types:
   - Conceptual: "What is the allowance method?"
   - Calculation: "Calculate bad debt expense given..."
   - Journal Entry: "Record the journal entry for..."
   - Analysis: "Compare the two methods and explain when..."
   - Application: "A company has the following AR aging schedule..."

7. Ensure all information is accurate and based on the source material provided

8. Make responses detailed and educational (not just brief answers)

OUTPUT FORMAT (JSON only):
Return a JSON array with EXACTLY 20 objects in this format:

[
  {
    "messages": [
      {
        "role": "system",
        "content": "You are an expert accounting tutor specializing in financial accounting and accounts receivable."
      },
      {
        "role": "user",
        "content": "What is the allowance method for bad debts and why is it preferred under GAAP?"
      },
      {
        "role": "assistant",
        "content": "The allowance method is an accounting approach for handling uncollectible accounts receivable. Under this method, companies estimate the amount of accounts receivable that will likely become uncollectible at the end of each accounting period, rather than waiting until specific accounts are identified as uncollectible.\n\nKey features:\n\n1. **Timing**: Bad debt expense is recognized in the same period as the related credit sales (matching principle)\n\n2. **Journal Entry**:\n   DR Bad Debt Expense\n     CR Allowance for Doubtful Accounts\n\n3. **Balance Sheet Presentation**: Accounts Receivable is shown net of the allowance\n\nWhy it's preferred under GAAP:\n\n- **Matching Principle**: Expenses are matched to the revenues they help generate\n- **Conservatism**: Assets (AR) are not overstated\n- **More accurate financial statements**: Reflects economic reality\n\nThis contrasts with the direct write-off method, which only recognizes bad debt expense when a specific account becomes uncollectible (violates matching principle)."
      }
    ]
  },
  {
    "messages": [
      ... next sample focusing on calculation ...
    ]
  },
  ... 18 more samples ...
]

CRITICAL: Provide ONLY the JSON array. No introduction, no explanation, just the JSON.
```

---

## Key Points About This Prompt:

### 🎯 **High Fidelity Topic Guidance**
- Batch 15 gets ONLY "Accounts Receivable → Bad Debt Write-offs"
- Batch 16 might get "Accounts Receivable → AR Aging Reports"
- Batch 50 might get "Journal Entries → Adjusting Entries"

### 📚 **Relevant Docs Only**
- Only 2 of the 10 uploaded PDFs are relevant to bad debts
- System extracts excerpts from just those 2 files
- Reduces noise, increases relevance

### 🎓 **Explicit Focus Areas**
- 5 specific concepts the generator MUST cover
- Prevents drift to other AR topics
- Ensures comprehensive coverage of the subtopic

### 📊 **Difficulty Progression**
- Samples 1-7: Basic
- Samples 8-14: Intermediate
- Samples 15-20: Advanced
- Natural curriculum within each batch

### 🔄 **Variety Enforcement**
- Lists 5 question types explicitly
- Ensures diverse sample formats
- Prevents repetitive questions

---

## What You Can Edit:

**In the Gemini Planning Prompt** (`gemini_planner.py` line 255-313):
- How topics are identified
- How subtopics are broken down
- What goes into "guidance"
- How focus_areas are determined

**In the Generator Prompt** (will show you next in `sft_generator.py`):
- The instructions format
- How source material is presented
- Requirements/constraints language
- Output format instructions

---

## Example Flow for 2500 Samples:

```
Batch 1/125: Accounts Receivable → Basic AR Concepts (20 samples)
Batch 2/125: Accounts Receivable → Recording Credit Sales (20 samples)
Batch 3/125: Accounts Receivable → AR Aging Reports (20 samples)
Batch 4/125: Accounts Receivable → Collection Procedures (20 samples)
Batch 5/125: Accounts Receivable → Bad Debt Estimation (20 samples)
...
Batch 25/125: Journal Entries → Basic Debit/Credit Rules (20 samples)
Batch 26/125: Journal Entries → Adjusting Entries (20 samples)
...
Batch 65/125: Financial Statements → Balance Sheet Basics (20 samples)
...
Batch 125/125: Advanced Topics → Complex Multi-Step Problems (20 samples)
```

Each batch gets its own focused guidance, relevant docs, and specific goals.
