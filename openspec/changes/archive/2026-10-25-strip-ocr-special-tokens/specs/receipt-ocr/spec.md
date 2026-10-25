## Purpose

Defines the content of the text that OCR produces from a receipt image, so that clients and downstream steps such as classification receive plain recognized text and no model control tokens.

## ADDED Requirements

### Requirement: OCR text is free of model control tokens
The text produced by OCR for a receipt image SHALL NOT contain tokenizer special tokens of the OCR model, such as end-of-turn or end-of-text markers (e.g. `<|user|>`, `<|endoftext|>`), chat-role markers, or image/video markers. The recognized receipt text itself SHALL be otherwise unchanged. This applies to every place the OCR text is exposed or consumed: the response of `POST /extract_text`, the `raw_text` field of `POST /classify_receipt`, and the text handed to receipt classification.

#### Scenario: Extract text returns plain text
- **WHEN** a client uploads a receipt image to `/extract_text`
- **THEN** the response text contains the recognized receipt text and does not contain any `<|...|>` control token, in particular not a trailing `<|user|>`

#### Scenario: Classify receipt exposes the same clean text
- **WHEN** a client uploads a receipt image to `/classify_receipt`
- **THEN** `raw_text` in the response contains no model control token, and the text the classification was based on is identical to `raw_text`

#### Scenario: Recognized content is preserved
- **WHEN** the OCR model recognizes a receipt whose raw output is the receipt text followed by an end-of-turn token
- **THEN** the returned text equals the receipt text, with every recognized line, number and symbol present

#### Scenario: Image without recognizable text still yields empty text
- **WHEN** the uploaded file is not a decodable image, or the OCR output consists only of control tokens
- **THEN** `/extract_text` returns an empty string, and `/classify_receipt` responds with HTTP 422 stating that no text could be extracted
