## Why

The service can turn a receipt image into raw OCR text (`POST /extract_text`), but nothing turns that text into structured, categorized purchase data. Work on `feat/pipeline_classify` has started wiring a classification prompt into `LlmClient`, but it hijacks the generic `/llm/generate` endpoint, hardcodes the model, and returns unvalidated free text. This change finishes the pipeline properly: receipt in, validated list of categorized products out.

## What Changes

- Add a classification capability that takes receipt text and returns a validated list of `{product, price, category}` items, with categories `Süßigkeiten`, `Obst`, `Gemüse`, `Fleisch`, `Brot`, `Anderes`.
- Add `POST /classify_text`: JSON body with receipt text → classified items (no OCR model involved; fast iteration on the prompt).
- Add `POST /classify_receipt`: multipart image upload → OCR (GLM-OCR) → classification → classified items plus the raw OCR text.
- Use LangChain (`langchain-core`) for the classification prompt (`ChatPromptTemplate`) and for parsing/validating the model output (`PydanticOutputParser`) into Pydantic schemas.
- Restore `POST /llm/generate` to a generic chat endpoint with no hardcoded system prompt. **BREAKING** relative to the uncommitted branch state (which injected the classification prompt); compared to `main`, the "Answer in French" system prompt is also removed.
- `LlmClient` honors `LLM_MODEL_PATH` again instead of hardcoding `Qwen/Qwen3.5-9B`; the default when unset becomes `Qwen/Qwen3.5-9B`.
- Fix the classification prompt's escaped `{{ }}` braces so they are rendered correctly through the template engine.
- Update the pytest suite and `CLAUDE.md` to reflect the new endpoints and behavior.

## Capabilities

### New Capabilities
- `receipt-classification`: Classifying receipt text into categorized line items, exposed via `/classify_text` (text input) and `/classify_receipt` (image input chained through OCR), including output validation and error behavior.
- `llm-generation`: The generic local-LLM chat endpoint `/llm/generate` and how the model is selected (`LLM_MODEL_PATH`).

### Modified Capabilities
<!-- No existing specs in openspec/specs/ yet. -->

## Impact

- **Code**: `llm/client.py` (model loading, generic generation API), `llm/prompts/classify_prompt.py` (LangChain template), `services/classifier.py` (currently an empty stub → `ReceiptClassifier`), new router `api/router/classification.py`, `api/app.py` + `api/deps.py` (wiring), `models/` (new receipt schemas).
- **APIs**: two new endpoints; `/llm/generate` loses its implicit system prompt. `/extract_text` is unchanged.
- **Dependencies**: `langchain` stays (already added to `pyproject.toml`), used via `langchain_core`.
- **Configuration**: the local `.env` currently points `LLM_MODEL_PATH` at an OpenChat model; it must be pointed at the Qwen model (or removed to use the default) for classification to behave as tuned.
- **Runtime**: GLM-OCR and a 9B-parameter LLM (fp16, ~18 GB) are both resident on the GPU.
- **Tests**: `tests/receipt_extractor/llm/test_client.py` asserts the old French system prompt and must be updated; new tests for the classifier and routes.
