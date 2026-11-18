# receipt-classification Specification

## Purpose
Turns receipt content (raw OCR text or a receipt image) into a validated, structured list of purchased products, each with its price and a shopping category, so downstream consumers can aggregate spending by category.

## Requirements

### Requirement: Classified item structure
Every classification result SHALL be a JSON object with the receipt metadata fields `purchased_at`, `store` and `total` (see "Receipt metadata") and an `items` array. Each item SHALL contain `product` (non-empty string), `quantity` (positive integer, default `1`), `price` and `category` (one of `Süßigkeiten`, `Obst`, `Gemüse`, `Fleisch`, `Brot`, `Anderes`). `price` SHALL be the line total, i.e. the amount the receipt charges for the whole line (quantity × unit price), in the receipt's currency. `price` SHALL be held as an exact decimal value, not a binary float, and SHALL be serialized as a JSON number using `.` as decimal separator. Items SHALL appear in the order they occur on the receipt. Leading and trailing whitespace of `product` SHALL be ignored.

#### Scenario: Items preserve receipt order and structure
- **WHEN** a receipt containing "Apfel 1.99", "Milch 1.29", "Gummibärchen 0.99" (in that order) is classified
- **THEN** the response `items` array contains three objects in that order, each with `product`, `quantity`, `price`, and `category` fields, and `price` values are JSON numbers (e.g. `1.99`), not strings

#### Scenario: Prices are exact decimals
- **WHEN** the model reports prices `0.1` and `0.2` (as JSON numbers or strings)
- **THEN** the parsed prices are the exact decimals `0.1` and `0.2`, their sum is exactly `0.3`, and the response serializes them as the JSON numbers `0.1` and `0.2`

#### Scenario: Missing quantity defaults to 1
- **WHEN** the model returns an item without `quantity` or with `quantity` `null`
- **THEN** the item is returned with `quantity` `1`

#### Scenario: Multi-unit line keeps the line total as price
- **WHEN** the model reports an item with `quantity` `2` (or `"2"`) and `price` `2.78` for a line printed as "2 x 1,39 €" / "2,78 €"
- **THEN** the item is returned with `quantity` `2` and `price` `2.78`

#### Scenario: Invalid quantity is rejected
- **WHEN** the model reports a `quantity` of `0`, `-1`, `1.5`, `"2x"` or a boolean
- **THEN** the system responds as for invalid model output (HTTP 502)

#### Scenario: Comma decimal prices are normalized
- **WHEN** the model reports a price using a comma decimal separator (e.g. `"1,99"`)
- **THEN** the item's `price` in the response is the number `1.99`

#### Scenario: Thousands separators are normalized
- **WHEN** the model reports a price with a thousands separator and a decimal part (e.g. `"1.234,56"` or `"1,234.56"`)
- **THEN** the item's `price` in the response is the number `1234.56`

#### Scenario: Currency symbols and whitespace around a price are ignored
- **WHEN** the model reports a price such as `" 1,99 € "`, `"€1,99"` or `"1,99 EUR"`
- **THEN** the item's `price` in the response is the number `1.99`

#### Scenario: Ambiguous price text is rejected instead of guessed
- **WHEN** the model reports a price that contains more than one number or other text, such as `"2 x 1,99"`, `"19,99 B 2"`, `"12 345"` or `"1e3"`
- **THEN** the system does not return a price derived from it (such as `21.99`) and responds as for invalid model output (HTTP 502)

#### Scenario: Non-finite or non-numeric price is rejected
- **WHEN** the model reports a price of `NaN`, `Infinity` or a boolean
- **THEN** the system responds as for invalid model output (HTTP 502) and never returns `null` for `price`

#### Scenario: Negative prices are allowed
- **WHEN** the model reports a negative price (for example a discount or a deposit refund)
- **THEN** the item is returned with that negative number as `price`

#### Scenario: Blank product name is rejected
- **WHEN** the model returns an item whose `product` is empty or only whitespace
- **THEN** the system responds as for invalid model output (HTTP 502)

#### Scenario: Unknown category falls back to Anderes
- **WHEN** the model assigns a category that is not in the allowed set (e.g. `"Milchprodukte"`)
- **THEN** the item is returned with `category` set to `Anderes` instead of failing the request

#### Scenario: Category matching ignores case, surrounding whitespace and a trailing period
- **WHEN** the model assigns a category such as `"obst"`, `"OBST"`, `" Obst "` or `"Obst."`
- **THEN** the item is returned with the matching category (`Obst`), not `Anderes`

#### Scenario: Receipt without recognizable products
- **WHEN** the input contains no recognizable purchased products and the model returns an empty list and no metadata
- **THEN** the system responds with HTTP 200 and `{"purchased_at": null, "store": null, "total": null, "items": []}`

### Requirement: Classify receipt text
The system SHALL expose `POST /classify_text` accepting a JSON body `{"text": "<receipt text>"}` and returning the classification result object (`{"purchased_at": ..., "store": ..., "total": ..., "items": [...]}`). This endpoint SHALL NOT invoke the OCR model.

#### Scenario: Successful text classification
- **WHEN** a client posts `{"text": "ALDI\nApfel 1.99\nMilch 1.29\nZU ZAHLEN 3,28\nDatum 25.02.26"}` to `/classify_text` and the model returns the matching result
- **THEN** the system responds with HTTP 200 and a body of the form `{"purchased_at": "2026-02-25", "store": "ALDI", "total": 3.28, "items": [{"product": "Apfel", "quantity": 1, "price": 1.99, "category": "Obst"}, ...]}`

#### Scenario: Empty or missing text is rejected
- **WHEN** a client posts `{"text": ""}` or a body without `text` to `/classify_text`
- **THEN** the system responds with HTTP 422 and does not invoke the language model

### Requirement: Classify receipt image
The system SHALL expose `POST /classify_receipt` accepting a multipart upload with an `image` field. It SHALL extract the receipt text via OCR, classify that text with the same behavior as `/classify_text`, and return the same result object plus `raw_text`: `{"purchased_at": ..., "store": ..., "total": ..., "items": [...], "raw_text": "<OCR text>"}`.

#### Scenario: Successful image classification
- **WHEN** a client uploads a valid receipt image to `/classify_receipt`
- **THEN** the system responds with HTTP 200, the metadata fields and `items` contain the classification result, and `raw_text` contains the OCR text the classification was based on

#### Scenario: Unreadable image is rejected
- **WHEN** a client uploads a file that is not a decodable image, or OCR yields no text
- **THEN** the system responds with HTTP 422 with an error detail stating that no text could be extracted, and does not invoke the language model

#### Scenario: Metadata of the sample ALDI receipt is extracted
- **WHEN** `data/aldi_quittung.jpeg` is uploaded to `/classify_receipt` on the real models (manual check, GPU)
- **THEN** `purchased_at` is `2026-02-25` (with or without the time `19:33`/`19:33:55`), `store` contains `ALDI`, `total` is `21.89`, and `NITT KÄSEBRÖTCHEN` has `quantity` `2` and `price` `2.78`

### Requirement: Invalid model output is reported, not passed through
When the language model's output cannot be parsed into the classified item structure, both classification endpoints SHALL respond with HTTP 502 and an error detail indicating that the classification model returned invalid output. The system SHALL NOT return unvalidated model text as a successful result.

#### Scenario: Model returns non-JSON text
- **WHEN** the model responds with prose instead of JSON (e.g. "Hier ist deine Liste: ...")
- **THEN** the endpoint responds with HTTP 502 and the response is not a classification result

#### Scenario: Model output wrapped in a Markdown code fence
- **WHEN** the model responds with valid JSON wrapped in a ```` ```json ```` code fence
- **THEN** the fence is ignored and the endpoint responds with HTTP 200 and the parsed items

### Requirement: Classification is deterministic for identical input
Classification SHALL use deterministic decoding so that classifying identical text with the same model yields identical results.

#### Scenario: Repeated classification of the same text
- **WHEN** the same receipt text is posted to `/classify_text` twice against the same running server
- **THEN** both responses contain identical `items`

### Requirement: Receipt metadata
The classification result SHALL contain `purchased_at`, `store` and `total` as always-present keys. A value the receipt does not show SHALL be `null`, never invented. A missing key, `null` or a blank string from the model SHALL yield `null` without an error. A value that is present but cannot be parsed SHALL be treated as invalid model output (HTTP 502).
- `purchased_at`: the purchase date, serialized as `YYYY-MM-DD`, or as `YYYY-MM-DDTHH:MM:SS` when the receipt shows a time; no time zone. Accepted model formats: ISO `YYYY-MM-DD`, ISO date with time (`T` or space separator, `HH:MM` or `HH:MM:SS`), German `DD.MM.YYYY` and `DD.MM.YY` (year `20YY`), optionally followed by a space and `HH:MM` or `HH:MM:SS`. Any other format or a non-existent date is rejected.
- `store`: free text as printed on the receipt, with surrounding whitespace removed.
- `total`: the amount paid, parsed with the same rules as an item price, held as an exact decimal and serialized as a JSON number.

#### Scenario: Metadata is returned
- **WHEN** the model reports `purchased_at` `"2026-02-25"`, `store` `" ALDI "` and `total` `"21,89 €"`
- **THEN** the result contains `"purchased_at": "2026-02-25"`, `"store": "ALDI"` and `"total": 21.89`

#### Scenario: Date with time is kept
- **WHEN** the model reports `purchased_at` as `"2026-02-25T19:33:55"`, `"2026-02-25 19:33"` or `"25.02.26 19:33"`
- **THEN** `purchased_at` is `"2026-02-25T19:33:55"`, `"2026-02-25T19:33:00"` or `"2026-02-25T19:33:00"` respectively

#### Scenario: German date formats are accepted
- **WHEN** the model reports `purchased_at` as `"25.02.2026"` or `"25.02.26"`
- **THEN** `purchased_at` is `"2026-02-25"`

#### Scenario: Missing metadata yields null
- **WHEN** the model output omits `purchased_at`, `store` and `total`, or sets them to `null` or `""`
- **THEN** the system responds with HTTP 200 and all three fields are `null`

#### Scenario: Unparseable date is rejected
- **WHEN** the model reports `purchased_at` as `"25/02/2026"`, `"gestern"`, `"2026-02-30"`, `"19:33"` (time without date), a number or a boolean
- **THEN** the system responds as for invalid model output (HTTP 502)

#### Scenario: Unparseable total is rejected
- **WHEN** the model reports `total` as `"2 x 1,99"`, `"NaN"` or a boolean
- **THEN** the system responds as for invalid model output (HTTP 502)

#### Scenario: Non-text store is rejected
- **WHEN** the model reports `store` as a number, a boolean, a list or an object
- **THEN** the system responds as for invalid model output (HTTP 502)

### Requirement: Total mismatch is logged, not reported
When `total` is not `null` and differs from the sum of all item `price` values, the system SHALL log a warning that names both amounts and SHALL return the classification result unchanged (HTTP 200, no extra field).

#### Scenario: Sum of items differs from total
- **WHEN** the model reports items with prices `1.99` and `1.29` and `total` `21.89`
- **THEN** the result is returned unchanged with HTTP 200, it has no `warnings` field, and a warning log record contains `3.28` and `21.89`

#### Scenario: Sum of items matches total
- **WHEN** the model reports items with prices `0.1` and `0.2` and `total` `0.3`
- **THEN** no mismatch warning is logged
