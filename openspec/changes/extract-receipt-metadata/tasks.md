Every task: failing test first, then the minimal code, then refactor. Scenario names refer to `specs/receipt-classification/spec.md`.

## 1. Money as Decimal

- [x] 1.1 "Prices are exact decimals" (`tests/receipt_extractor/models/test_receipt.py`): `price` is a `Decimal`; JSON numbers and strings `0.1`/`0.2` parse to exact `Decimal("0.1")`/`Decimal("0.2")`; `model_dump(mode="json")` and the JSON response emit numbers, not strings. Then make `parse_price` return `Decimal` and convert numeric input via `Decimal(str(value))` (design 3)
- [x] 1.2 Update the existing price tests to expect `Decimal` values (required by the modified "Classified item structure" requirement, not a loosening); all existing price scenarios ("Comma decimal…", "Thousands separators…", "Currency symbols…", "Ambiguous price text…", "Non-finite…", "Negative prices…") stay green

## 2. Item quantity

- [x] 2.1 "Missing quantity defaults to 1" and "Multi-unit line keeps the line total as price": missing or `null` → `1`; `2` and `"2"` → `2`; `price` unchanged. Then add `quantity` to `ReceiptItem`
- [x] 2.2 "Invalid quantity is rejected": `0`, `-1`, `1.5`, `"2x"`, `True` raise a validation error; through the classifier they raise `ClassificationError`. Then add the validator
- [x] 2.3 "Items preserve receipt order and structure": serialized items contain `product`, `quantity`, `price`, `category`

## 3. Receipt metadata

- [x] 3.1 "Metadata is returned", "Date with time is kept", "German date formats are accepted": parametrized tests for a `purchased_at` parser (ISO date, ISO with `T`/space and `HH:MM[:SS]`, `DD.MM.YYYY`, `DD.MM.YY`, optional time). Then add the parser and `purchased_at: date | datetime | None` to `ReceiptClassification` (design 4)
- [x] 3.2 "Unparseable date is rejected": `"25/02/2026"`, `"gestern"`, `"2026-02-30"`, `"19:33"`, `20260225`, `True` are rejected
- [x] 3.3 `store`: trimmed; blank → `null`; number/boolean/list/object rejected ("Non-text store is rejected")
- [x] 3.4 `total`: same rules as `price`, `Decimal`, serialized as number; "Unparseable total is rejected" (`"2 x 1,99"`, `"NaN"`, `True`)
- [x] 3.5 "Missing metadata yields null" and "Receipt without recognizable products": omitted, `null` and `""` give `null`; all three keys are always present in the JSON output

## 4. Classifier and API

- [x] 4.1 "Sum of items differs from total" and "Sum of items matches total" (`tests/receipt_extractor/services/test_classifier.py`, `caplog`): warning with `3.28` and `21.89` on mismatch, none on match, none when `total` is `null`; result unchanged. Then add the check to `ReceiptClassifier.classify_text` (design 6)
- [x] 4.2 Classifier: invalid metadata or quantity in model output raises `ClassificationError` (HTTP 502 path), valid metadata is parsed
- [x] 4.3 "Successful text classification" (`tests/receipt_extractor/api/test_classification.py`, mocked classifier): `/classify_text` returns `purchased_at`, `store`, `total`, `items[].quantity` with number amounts and no `warnings` key
- [x] 4.4 "Successful image classification": `/classify_receipt` returns the metadata and `raw_text`. Then pass the metadata into `ReceiptClassificationResponse` in `api/router/classification.py`

## 5. Prompt

- [x] 5.1 Prompt test (`TestPrompting`): the rendered system prompt asks for `purchased_at` (ISO), `store`, `total`, `quantity`; says missing values are `null`, `price` is the line total and the digit after `€` is the VAT class; its JSON example parses into a valid `ReceiptClassification`. Then update `CLASSIFY_PROMPT` minimally (design 7); category list and deposit handling unchanged (#5)

## 6. Docs

- [x] 6.1 Update `CLAUDE.md`: `models/receipt.py` schemas (`quantity`, `price` as line-total `Decimal`, `purchased_at`/`store`/`total`, parsing rules), `/classify_text` and `/classify_receipt` response shapes, the mismatch warning log

## 7. Verification

- [x] 7.1 Run `make check` twice in a row; both pass
- [ ] 7.2 Real-model check (needs GPU, real models, server on port 8080; only one worktree at a time): start the server, `POST /classify_receipt` with the main checkout's `data/aldi_quittung.jpeg` (`data/` is gitignored, not in this worktree). Verifies "Metadata of the sample ALDI receipt is extracted": `purchased_at` `2026-02-25` (time optional), `store` contains `ALDI`, `total` `21.89`, `NITT KÄSEBRÖTCHEN` quantity `2` price `2.78`, `CC ZERO/LIGHT/KOFF.FRE` quantity `1`. Record the actual values and any item/category differences from the previous run here below this task
  Result 2026-11-15 (real models, `data/aldi_quittung.jpeg`, final prompt): HTTP 200, `purchased_at` `2026-02-25T19:33:00`, `store` `ALDI`, `total` `21.89`, 11 items, sum `21.64` (mismatch warning logged; the `PFANDWERT` lines are missing, see #5).
  `NITT KÄSEBRÖTCHEN`: quantity `2`, price `2.78` (ok). `CC ZERO/LIGHT/KOFF.FRE`: quantity `2`, price `1.59` (**fails**, expected quantity `1`: the model reads the VAT class `€ 2` as a quantity).
  Tried 5 prompt wordings (explicit rules, worked examples, a VAT-2 drink in the example); the first run had `NITT` price `1.39`, now fixed. `CC` stays at `2` in every run. Categories are the same in all runs (Gemüse: Kartoffeln, Rahmspinat, Tomatenmark; Süßigkeiten: 2 chocolate items; Brot: Käsebrötchen; rest Anderes). No earlier run is recorded, so there is no baseline to diff.
- [x] 7.3 Stop the server; port 8080 and the GPU are free
