## Context

See proposal.md for motivation. Facts from `data/aldi_quittung.jpeg` (read by hand; `data/` is gitignored and exists only in the main checkout):

- Store header `ALDI`; date `Datum 25.02.26 19:33 Uhr` and `25/02/2026 19:33:55`; `ZU ZAHLEN 21,89 €`.
- The digit after `€` on each line (`1,59 € 2`, `PFANDWERT 0,25 € 2`) is the VAT class (`MWST 1 7,00%`, `2 19,00%`), not a quantity. Issue #5 read it as a quantity.
- `2 x 1,39 €` is printed above `NITT KÄSEBRÖTCHEN 2,78 € 1`. All 12 right-column amounts (deposit included) sum to exactly `21,89`.

## Goals / Non-Goals

**Goals:** receipt-level date, store, total; item quantity; exact money; strict parsing; prompt extended so #5 can build on it.

**Non-Goals:** deposit lines and categories (#5); persistence and dashboard; weight-based quantities; currency field; time zones.

## Decisions

### 1. `price` is the line total
Choice: `price` is the right-column amount; unit price = `price / quantity`.
Why: it is what the receipt prints per line, and `sum(price)` equals `total` without multiplying or rounding.
Rejected: unit price. It needs `quantity × price` for every sum and is not printed for most lines.

### 2. `quantity` is a positive integer
Choice: `int ≥ 1`, default `1`; weighed goods use `1`. Accept JSON integers and digit strings; reject `0`, negatives, fractions, booleans.
Rejected: decimal quantity with a unit (`0,534 kg`). Needs a unit field; no sample receipt needs it.

### 3. Money is `Decimal`, serialized as a JSON number
Choice: `price` and `total` are `Decimal`. Model numbers are converted via `Decimal(str(value))`, so `1.99` does not become `1.9899999…`. JSON output uses a number serializer (`PlainSerializer(float, when_used="json")` or equivalent).
Rejected: Pydantic's default `Decimal` JSON serialization as a string. It breaks the existing contract (`price` is a number).

### 4. `purchased_at` is one field: `date` or naive `datetime`
Choice: `date | datetime | None`, parsed by our own function (like `parse_price`) from ISO and German `DD.MM.YY[YY]` formats with optional time.
Why: one field matches the issue; German formats are what receipts print and are unambiguous.
Rejected: slash formats (`25/02/2026` is unambiguous here, but `MM/DD` exists elsewhere; no guessing). Rejected: separate `date`/`time` fields; time zones (receipts show local time only).

### 5. Present but unparseable → HTTP 502; missing or blank → `null`
Choice: same strictness as prices today.
Rejected: degrade a bad value to `null`. It hides model errors. Trade-off: one bad date fails the whole receipt (see Risks).

### 6. Total mismatch: log a warning, no `warnings` field
Choice: `ReceiptClassifier` logs a warning with both amounts when `total` is set and differs from `sum(price)`.
Why: until #5 keeps deposit lines, most ALDI receipts mismatch, so a response field would be noise. Adding a field later is non-breaking.
Rejected: a `warnings` list in the response.

### 7. Minimal prompt change
Choice: extend the JSON example and add short rules: ISO date, `null` when not printed, `price` is the line total, quantity from `N x` lines, the digit after `€` is the VAT class. Category list and deposit handling unchanged.
Why: #5 rewrites categories and deposits on top of this prompt.

### 8. No ADR
No new component, dependency or layering change; only the schema and prompt grow.

## Risks / Trade-offs

- [The model returns `25/02/2026` or another rejected date format] → whole request is 502. Mitigation: prompt asks for ISO; the real-model check covers the sample receipt. Revisit decision 5 if real receipts hit it.
- [`float` conversion in JSON output loses precision for very large amounts] → irrelevant for receipt amounts; the exact `Decimal` stays in the Python model.
- [The model reads the VAT-class digit after the amount as the quantity, e.g. `CC ZERO/LIGHT/KOFF.FRE` `1,59 € 2` gets `quantity` `2`] → known limit of the 9B model, not fixed by prompt wording (5 variants tried). Follow-up: #18.
- [Prompt change shifts item categories on the sample receipt] → real-model check compares items with the previous run; differences are recorded, not assumed harmless.
