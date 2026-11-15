## Why

To track spending over time, every receipt needs a purchase date, a store and a total. `POST /classify_text` and `POST /classify_receipt` return only `items` today, although these values are printed on the receipt. Quantities (`2 x 1,39 €`) are lost, and prices are binary floats, so stored sums would not be exact. First step toward the spending dashboard. Closes #14.

## What Changes

- New receipt-level fields in both classification responses: `purchased_at` (date, optional time), `store` (free text), `total` (amount paid). Always present; `null` when the receipt does not show them.
- New item field `quantity` (positive integer, default `1`). `price` is specified as the line total.
- `price` and `total` are `Decimal` in the schema and stay JSON numbers in the response.
- Strict parsing: a date or amount that is present but not parseable is invalid model output (HTTP 502), like prices today.
- A mismatch between `sum(price)` and `total` is logged as a warning; the response is unchanged.
- `CLASSIFY_PROMPT` asks for the new fields. The change is minimal: categories and deposit lines (`PFANDWERT`) stay as they are for #5.
- Response shape grows by four fields (additive, not breaking).

## Capabilities

### New Capabilities
<!-- None. -->

### Modified Capabilities
- `receipt-classification`: classification result gains receipt metadata (`purchased_at`, `store`, `total`) and item `quantity`; `price` is the line total and an exact decimal; total mismatch is logged.

## Impact

- **Code**: `models/receipt.py` (fields, parsers), `llm/prompts/classify_prompt.py`, `services/classifier.py` (mismatch warning), `api/router/classification.py` (pass metadata into `ReceiptClassificationResponse`).
- **APIs**: `/classify_text` and `/classify_receipt` responses get `purchased_at`, `store`, `total` and `items[].quantity`.
- **Tests**: `tests/receipt_extractor/models`, `services`, `api`; existing float price assertions become `Decimal`.
- **Docs**: `CLAUDE.md` (schemas, router responses).
- **Dependencies/config**: none. Real-model check needs the GPU and port 8080.
- **Coordination**: #5 (categories, deposits) builds on the prompt from this change.
