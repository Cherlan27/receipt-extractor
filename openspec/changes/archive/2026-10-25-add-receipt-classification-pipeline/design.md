## Context

See `proposal.md` for motivation. Relevant current state on `feat/pipeline_classify` (uncommitted):

- `LlmClient` (`llm/client.py`) hardcodes `Qwen/Qwen3.5-9B` (ignoring `self.model_path`), always prepends `CLASSIFY_SYSTEM_PROMPT` as system message, loads with `device_map="auto"` but still moves inputs to `self.device`, and passes `enable_thinking=False` to `apply_chat_template`.
- `llm/prompts/classify_prompt.py` is a plain string whose JSON example uses `{{ }}`, i.e. it was written for a template engine that is not used yet.
- `services/classifier.py` exists but is empty.
- `langchain` 1.3.4 (with `langchain-core` 1.4.1) is installed but unused.
- Both models are instantiated once in the FastAPI `lifespan` and injected via `api/deps.py`; tests replace them with `MagicMock`s in `tests/receipt_extractor/api/conftest.py`.

## Goals / Non-Goals

**Goals:**
- Keep `LlmClient` a task-agnostic wrapper around the local model; all classification knowledge lives in the classifier service and its prompt.
- Make classification testable without loading any model (prompt rendering and output parsing are pure functions over strings).
- Share one classification code path between `/classify_text` and `/classify_receipt`.

**Non-Goals:**
- Running LangChain's own model integrations (`langchain-huggingface`, `ChatHuggingFace`) — the model stays loaded through `transformers` as today.
- Retries, self-correction loops (`OutputFixingParser`), or constrained/grammar-based decoding.
- Non-blocking inference: endpoints keep the existing pattern of calling the model synchronously inside `async def` handlers. Concurrency/queuing is a separate concern.
- Persisting results, multi-currency handling, totals/VAT extraction, or user-configurable categories.

## Decisions

### 1. LangChain is used only for prompt templating and output parsing
`ChatPromptTemplate` renders the system + user messages; `PydanticOutputParser` parses the model's text into Pydantic models. The rendered LangChain messages are converted to `{"role", "content"}` dicts and handed to `LlmClient`, which applies the tokenizer's chat template as today.

- *Why*: matches the user's choice to use LangChain for prompt and parsing, gives us Markdown-fence stripping and a uniform `OutputParserException` for free, and avoids a new heavyweight integration package.
- *Alternative*: `langchain-huggingface` `HuggingFacePipeline` + `ChatHuggingFace` so the whole thing is an LCEL chain. Rejected: extra dependency, its own model-loading path conflicts with our lifespan-managed singleton, and Qwen's `enable_thinking` kwarg is awkward to pass through.
- *Alternative*: plain `str` prompt + `json.loads` + `TypeAdapter`. Simpler, but the user explicitly opted for LangChain.
- Imports come from `langchain_core` (the stable home of these classes); `langchain` remains the declared dependency that pulls it in.

### 2. Prompt template and format instructions
`classify_prompt.py` exposes a `ChatPromptTemplate` with a system message (German instructions, category list, one worked example) and a `{receipt_text}` user message. The JSON example keeps `{{ }}` escaping, which is now correct because the template engine un-escapes it. The prompt additionally includes `{format_instructions}` filled from `PydanticOutputParser.get_format_instructions()` (via `.partial()`), so the schema the parser validates against is the schema the model is shown.

- The expected output becomes an object `{"items": [...]}` instead of a bare list, because `PydanticOutputParser` targets a single `BaseModel`. The worked example in the prompt is updated accordingly.

### 3. Data model (`models/receipt.py`)
- `Category`: `StrEnum` with values `Süßigkeiten`, `Obst`, `Gemüse`, `Fleisch`, `Brot`, `Anderes`.
- `ReceiptItem`: `product: str` (min length 1), `price: float`, `category: Category`.
  - `field_validator("price", mode="before")`: strings have `,` replaced by `.` and surrounding whitespace/currency symbols stripped before float conversion.
  - `field_validator("category", mode="before")`: values not in `Category` are mapped to `Category.ANDERES` and logged at WARNING.
- `ReceiptClassification`: `items: list[ReceiptItem]` — response of `/classify_text`, and the parser target.
- `ReceiptClassificationResponse(ReceiptClassification)`: adds `raw_text: str` — response of `/classify_receipt`.
- `ClassifyTextRequest`: `text: str` with `min_length=1` (whitespace-only is stripped first, so it is rejected too).

- *Why `float` over `Decimal`*: values are for display and per-category aggregation, not accounting; Pydantic serializes `Decimal` as a JSON string by default, which would contradict the spec's "price is a number". Revisit if exact sums matter.
- *Why coerce unknown categories*: a single creative label from the model shouldn't fail an otherwise good receipt; the fallback is visible (`Anderes`) and logged.

### 4. `LlmClient` becomes task-agnostic
- Model path: `model_path or os.getenv("LLM_MODEL_PATH", "Qwen/Qwen3.5-9B")`, used for both tokenizer and model. The hardcoded id is removed.
- Loading stays `torch_dtype=torch.float16, device_map="auto"`; inputs are moved to `self.model.device` (the `self.device` attribute is dropped), which is correct under `device_map="auto"`.
- New core method `chat(messages: list[dict], max_new_tokens: int, do_sample: bool = False) -> str` applies the chat template (`add_generation_prompt=True`, `enable_thinking=False`), generates, and decodes only the new tokens.
- `generate(req: PromptData) -> dict` becomes a thin wrapper: `{"response": self.chat([m.model_dump() for m in req.messages], req.max_new_tokens)}`. The `system_prompt` property is removed.
- `enable_thinking=False` is passed unconditionally: Qwen3.x templates use it to suppress `<think>` blocks; Jinja chat templates of other models ignore unknown variables.
- Greedy decoding (`do_sample=False`) is the default, satisfying the determinism requirement for classification and making `/llm/generate` reproducible as well.

### 5. `ReceiptClassifier` service (`services/classifier.py`)
Constructed with an `LlmClient`; owns the prompt and the parser.

```
classify_text(text) -> ReceiptClassification
  messages = prompt.format_messages(receipt_text=text) → [{"role","content"}]
  raw = llm_client.chat(messages, max_new_tokens=CLASSIFY_MAX_NEW_TOKENS)  # 4096
  return parser.parse(raw)        # raises ClassificationError on OutputParserException
```

- Defines `ClassificationError` (wrapping the raw model output for logging) so the service stays FastAPI-free, as `CLAUDE.md` requires for services.
- LangChain message types map to roles: `system → system`, `human → user`, `ai → assistant`.
- `CLASSIFY_MAX_NEW_TOKENS = 4096` is a module constant: enough for long receipts, bounded to avoid runaway generation.

### 6. Wiring and routing
- `lifespan` creates `ReceiptClassifier(app.state.llm_client)` as `app.state.classifier`; `deps.get_classifier` returns it.
- New router module `api/router/classification.py` (`classification_router`, no prefix) with both endpoints; included in `app.py`.
- `/classify_text`: `classifier.classify_text(body.text)`.
- `/classify_receipt`: `text = extractor.get_text(bytes)`; if `text.strip()` is empty → `HTTPException(422, "No text could be extracted from the image")`; else classify and return `ReceiptClassificationResponse(items=..., raw_text=text)`.
- Both map `ClassificationError` → `HTTPException(502, "Classification model returned invalid output")`; the raw output is logged, not returned.

### 7. Testing approach
- `models/receipt.py`: price normalization, category fallback, empty-text rejection.
- `ReceiptClassifier` with a `MagicMock` `LlmClient` whose `chat` returns canned strings: valid JSON, fenced JSON, prose (→ `ClassificationError`), empty list; assert rendered messages contain the receipt text and a system message.
- `LlmClient`: update `test_client.py` — drop the French system prompt test, assert no system message is injected by `generate`, assert env/default model path, assert `do_sample=False` and `enable_thinking=False`.
- Routes: extend `conftest.py` to also mock the classifier (it's built from the mocked `LlmClient`, so patch `ReceiptClassifier` in `app.py`), then test 200/422/502 paths for both endpoints.

## Risks / Trade-offs

- [GPU memory: GLM-OCR + 9B fp16 LLM may not fit on one consumer GPU] → `device_map="auto"` offloads to CPU when needed (slow but functional); document requirement in `CLAUDE.md`. Quantization is out of scope.
- [Local `.env` points `LLM_MODEL_PATH` at OpenChat, so after this change the server silently loads OpenChat instead of Qwen] → called out in proposal and tasks; update `.env` (not committed) as part of rollout.
- [Small-model JSON reliability: the model may drop fields or add prose] → format instructions + worked example + greedy decoding; failures surface as 502 instead of bad data. Retry/fixing parser can be added later without spec change.
- [OCR text quality (line breaks, discount lines, deposit "Pfand") leads to odd items] → accepted for now; `raw_text` is returned so issues can be diagnosed, and `/classify_text` allows prompt iteration on saved OCR output.
- [Blocking inference in `async` handlers stalls the event loop for the request duration] → existing behavior, explicitly a non-goal here.
- [`/llm/generate` no longer injects any system prompt] → intended; clients that relied on "Answer in French" from `main` must send their own system message (spec scenario covers this).

## Migration Plan

1. Merge code changes; run `poetry install` (no new packages beyond the already-locked `langchain`).
2. Update local `.env`: set `LLM_MODEL_PATH` to the Qwen model (local path or `Qwen/Qwen3.5-9B`), or remove it to use the default.
3. Rollback: revert the merge; no data or schema migrations involved.
