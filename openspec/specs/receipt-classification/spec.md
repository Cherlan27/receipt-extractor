# receipt-classification Specification

## Purpose
Turns receipt content (raw OCR text or a receipt image) into a validated, structured list of purchased products, each with its price and a shopping category, so downstream consumers can aggregate spending by category.

## Requirements

### Requirement: Classified item structure
Every classification result SHALL be a JSON object with an `items` array. Each item SHALL contain `product` (non-empty string), `price` (number, in the receipt's currency, using `.` as decimal separator), and `category` (one of `Süßigkeiten`, `Obst`, `Gemüse`, `Fleisch`, `Brot`, `Anderes`). Items SHALL appear in the order they occur on the receipt.

#### Scenario: Items preserve receipt order and structure
- **WHEN** a receipt containing "Apfel 1.99", "Milch 1.29", "Gummibärchen 0.99" (in that order) is classified
- **THEN** the response `items` array contains three objects in that order, each with `product`, `price`, and `category` fields, and `price` values are numbers (e.g. `1.99`), not strings

#### Scenario: Comma decimal prices are normalized
- **WHEN** the model reports a price using a comma decimal separator (e.g. `"1,99"`)
- **THEN** the item's `price` in the response is the number `1.99`

#### Scenario: Unknown category falls back to Anderes
- **WHEN** the model assigns a category that is not in the allowed set (e.g. `"Milchprodukte"`)
- **THEN** the item is returned with `category` set to `Anderes` instead of failing the request

#### Scenario: Receipt without recognizable products
- **WHEN** the input contains no recognizable purchased products and the model returns an empty list
- **THEN** the system responds with HTTP 200 and `{"items": []}`

### Requirement: Classify receipt text
The system SHALL expose `POST /classify_text` accepting a JSON body `{"text": "<receipt text>"}` and returning the classification result object (`{"items": [...]}`). This endpoint SHALL NOT invoke the OCR model.

#### Scenario: Successful text classification
- **WHEN** a client posts `{"text": "Apfel 1.99\nMilch 1.29"}` to `/classify_text`
- **THEN** the system responds with HTTP 200 and a body of the form `{"items": [{"product": "Apfel", "price": 1.99, "category": "Obst"}, ...]}`

#### Scenario: Empty or missing text is rejected
- **WHEN** a client posts `{"text": ""}` or a body without `text` to `/classify_text`
- **THEN** the system responds with HTTP 422 and does not invoke the language model

### Requirement: Classify receipt image
The system SHALL expose `POST /classify_receipt` accepting a multipart upload with an `image` field. It SHALL extract the receipt text via OCR, classify that text with the same behavior as `/classify_text`, and return `{"items": [...], "raw_text": "<OCR text>"}`.

#### Scenario: Successful image classification
- **WHEN** a client uploads a valid receipt image to `/classify_receipt`
- **THEN** the system responds with HTTP 200, `items` contains the classified products, and `raw_text` contains the OCR text the classification was based on

#### Scenario: Unreadable image is rejected
- **WHEN** a client uploads a file that is not a decodable image, or OCR yields no text
- **THEN** the system responds with HTTP 422 with an error detail stating that no text could be extracted, and does not invoke the language model

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
