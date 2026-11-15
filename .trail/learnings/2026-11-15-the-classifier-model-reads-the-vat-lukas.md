---
title: The classifier model reads the VAT-class digit on ALDI receipts as a quantity; prompt wording does not fix it
author: Lukas
date: 2026-11-15
branch: issue-14-receipt-metadata
cost: { minutes: 25, tokens: 230774 }
id: l-8a0e81a8
scope: repo
status: active
src: agent
touches:
  - openspec/changes/extract-receipt-metadata/tasks.md
  - src/receipt_extractor/llm/prompts/classify_prompt.py
  - CLAUDE.md
  - src/receipt_extractor/api/router/classification.py
  - src/receipt_extractor/services/classifier.py
  - tests/receipt_extractor/api/test_classification.py
  - tests/receipt_extractor/services/test_classifier.py
  - src/receipt_extractor/models/receipt.py
  - tests/receipt_extractor/models/test_receipt.py
  - openspec/changes/extract-receipt-metadata/workflow-log.md
---

## Decided
ALDI prints the VAT class as a single digit after the amount (`1,59 € 2`, `PFANDWERT 0,25 € 2`; `MWST 1` = 7 %, `2` = 19 %). It is not a quantity; quantities are printed as `2 x 1,39 €` on their own line. Qwen3.5-9B still sets `quantity: 2` on VAT-class-2 lines. Accepted as a known limit in #14; the fix is a deterministic step before the LLM call (#18): remove a lone digit that follows `<amount> €` at the end of a line, in the LLM input only, `raw_text` unchanged.

## Tried and ruled out
- Five prompt wordings (explicit rule, worked examples, a VAT-2 drink in the JSON example): the misread stayed in every run. Do not spend more time on prompt tuning for this.
- Applying the strip only after detecting an ALDI receipt: rejected, because the store detection adds a failure mode (header missing or misread in OCR) and the pattern is unambiguous anyway. Chains that print letters (A/B) are not affected.

## Watch out
Issue #5 first read this digit as a quantity too. Deposit lines have the same pattern, so coordinate #5 and #18.

In the code: src/receipt_extractor/llm/prompts/classify_prompt.py (CLASSIFY_PROMPT), src/receipt_extractor/services/classifier.py (classify_text); sample D:/Projects/receipt_extractor/data/aldi_quittung.jpeg (data/ is gitignored).
