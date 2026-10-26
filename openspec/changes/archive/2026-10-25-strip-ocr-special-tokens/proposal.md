## Why

`ReceiptExtractor.get_text()` decodes the OCR model's output with `skip_special_tokens=False`, so the model's end-of-turn token `<|user|>` is appended to every extracted text (observed on all four images in `data/`, e.g. `...19:33:55<|user|>`). It leaks into the `POST /extract_text` response, into `raw_text` of `POST /classify_receipt`, and into the prompt sent to the classification LLM. It was found while verifying `add-receipt-classification-pipeline` and deliberately kept out of that change.

## What Changes

- OCR text returned by the extractor no longer contains tokenizer special tokens (chat-role markers, end-of-turn/end-of-text, image/video markers, etc.). Everything else in the text is unchanged.
- The cleaned text is what `POST /extract_text` returns, what `POST /classify_receipt` returns as `raw_text`, and what the classifier receives.
- The existing extractor unit test, which pins `skip_special_tokens` to `False`, is updated; `CLAUDE.md` is adjusted accordingly.

## Capabilities

### New Capabilities
- `receipt-ocr`: The content contract of the text produced by OCR on a receipt image (`POST /extract_text`, and the OCR step behind `POST /classify_receipt`) — specifically that it is plain recognized text without model control tokens.

### Modified Capabilities
<!-- None. `receipt-classification` (from the not-yet-archived change add-receipt-classification-pipeline) already requires that raw_text is the OCR text the classification was based on; that stays true. -->

## Impact

- **Code**: one argument in `services/extractor.py` (`ReceiptExtractor.get_text`).
- **APIs**: response *content* of `/extract_text` and `raw_text` of `/classify_receipt` lose the trailing `<|user|>`; no schema change.
- **Classification input**: the LLM prompt no longer ends with a stray control token. Classification output may change slightly for the same image, so it is re-checked on the sample receipt.
- **Tests/docs**: `tests/receipt_extractor/services/test_extractor.py`, `CLAUDE.md`.
- **Dependencies/config**: none.
