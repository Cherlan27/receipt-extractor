## 1. Data model

- [x] 1.1 Create `src/receipt_extractor/models/receipt.py` with `Category` (`StrEnum`), `ReceiptItem`, `ReceiptClassification`, `ReceiptClassificationResponse`, `ClassifyTextRequest` per design §3
- [x] 1.2 Add `price` before-validator (comma → dot, strip whitespace/currency symbols) and `category` before-validator (unknown → `Anderes`, WARNING log)
- [x] 1.3 Add `tests/receipt_extractor/models/test_receipt.py` covering price normalization, category fallback, item order preserved, and empty/whitespace `text` rejection

## 2. Generic LlmClient

- [x] 2.1 Load tokenizer and model from `model_path or os.getenv("LLM_MODEL_PATH", "Qwen/Qwen3.5-9B")`; remove the hardcoded model id
- [x] 2.2 Drop `self.device`; move inputs to `self.model.device`
- [x] 2.3 Add `chat(messages, max_new_tokens, do_sample=False) -> str` (chat template with `add_generation_prompt=True`, `enable_thinking=False`; decode only new tokens)
- [x] 2.4 Reimplement `generate(req)` as a wrapper over `chat` returning `{"response": ...}`; remove the `system_prompt` property and the `CLASSIFY_SYSTEM_PROMPT` import
- [x] 2.5 Update `tests/receipt_extractor/llm/test_client.py`: remove French-prompt test, assert no system message is injected, assert env var / `Qwen/Qwen3.5-9B` default, assert `do_sample=False` and `enable_thinking=False`, add a test for `chat`

## 3. Classification prompt and service

- [x] 3.1 Rewrite `llm/prompts/classify_prompt.py` as a LangChain `ChatPromptTemplate` (system + `{receipt_text}` user message) with a `{format_instructions}` placeholder; update the worked example to the `{"items": [...]}` shape, keeping `{{ }}` escaping
- [x] 3.2 Implement `ReceiptClassifier` and `ClassificationError` in `services/classifier.py` per design §5 (prompt `.partial(format_instructions=...)`, LangChain message → role dict conversion, `CLASSIFY_MAX_NEW_TOKENS = 4096`, `OutputParserException` → `ClassificationError` with raw output)
- [x] 3.3 Add `tests/receipt_extractor/services/test_classifier.py` with a mocked `LlmClient`: valid JSON, fenced JSON, prose → `ClassificationError`, empty list, rendered messages contain system prompt + receipt text, `chat` called with `CLASSIFY_MAX_NEW_TOKENS`

## 4. API endpoints

- [x] 4.1 Instantiate `ReceiptClassifier(app.state.llm_client)` as `app.state.classifier` in `lifespan` (and tear it down); add `get_classifier` to `api/deps.py`
- [x] 4.2 Create `api/router/classification.py` with `classification_router`: `POST /classify_text` → `ReceiptClassification`, `POST /classify_receipt` → `ReceiptClassificationResponse`
- [x] 4.3 In `/classify_receipt`, return 422 "No text could be extracted from the image" when OCR text is empty/whitespace, without calling the classifier
- [x] 4.4 Map `ClassificationError` → 502 "Classification model returned invalid output" in both endpoints; log the raw model output
- [x] 4.5 Include `classification_router` in `api/app.py`
- [x] 4.6 Extend `tests/receipt_extractor/api/conftest.py` to patch `ReceiptClassifier` and expose a `mock_classifier` fixture
- [x] 4.7 Add `tests/receipt_extractor/api/test_classification.py`: 200 / 422 (empty text, missing field) / 502 for `/classify_text`; 200 with `raw_text` / 422 on empty OCR / 502 for `/classify_receipt`
- [x] 4.8 Update `tests/receipt_extractor/api/test_generate.py` if it relies on the removed system prompt

## 5. Verification and docs

- [x] 5.1 Run `poetry run pytest` — all tests green
- [x] 5.2 Run `pre-commit run --all-files` (ruff, isort) and fix findings
- [x] 5.3 Manual check with the real models: start the server, `POST /classify_receipt` with `data/aldi_quittung.jpeg`, and `POST /classify_text` with its `raw_text`; confirm identical `items` on a repeated call
- [x] 5.4 Update `CLAUDE.md`: new endpoints and data flow, `ReceiptClassifier`, generic `LlmClient` (no system prompt, `chat()`), `LLM_MODEL_PATH` default `Qwen/Qwen3.5-9B`, LangChain dependency, `classifier.py` no longer a stub, GPU memory note
- [x] 5.5 Remind the user to update their local `.env` `LLM_MODEL_PATH` (currently OpenChat) to the Qwen model
