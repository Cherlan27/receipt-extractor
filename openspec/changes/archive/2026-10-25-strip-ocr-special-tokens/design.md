## Context

See `proposal.md` for motivation. `ReceiptExtractor.get_text()` generates up to 8192 tokens with GLM-OCR and decodes only the newly generated ids via `processor.decode(..., skip_special_tokens=False)`. A unit test pins that argument to `False`; the reason for `False` is not documented in the code.

Measured on the real model (all four images in `data/`, run before writing this change):

- The generation config stops on `eos_token_id = [<|endoftext|>, <|user|>]`. GLM-OCR therefore ends its answer with `<|user|>`, and that token is part of the generated ids.
- Each of the four raw outputs contains **exactly one** special token: the trailing `<|user|>`. No chat-role, image, think, tool or box marker occurs inside the text.
- For all four images, `decode(skip_special_tokens=True)` equals the raw output with special tokens removed by regex, i.e. it differs from today's output only by the trailing `<|user|>`.

The tokenizer defines 36 special tokens (`<|endoftext|>`, `[MASK]`, `[gMASK]`, `<|system|>`, `<|user|>`, `<|assistant|>`, `<|begin_of_image|>`, `<think>`, `<tool_call>`, `<|begin_of_box|>`, …). None of them is receipt content.

## Goals / Non-Goals

**Goals:**
- Remove special tokens from the OCR text at its single source, so every consumer (`/extract_text`, `raw_text`, classifier prompt) benefits without further changes.
- Keep all recognized text byte-for-byte as before.

**Non-Goals:**
- Any other OCR post-processing: whitespace normalization, line-break cleanup, deduplication of repeated lines, correcting recognition errors.
- Changing generation (`max_new_tokens`, prompt `"Text Recognition:"`, sampling).
- Sanitizing control tokens inside the classifier or router; the extractor is the one place that produces OCR text.

## Decisions

### 1. Decode with `skip_special_tokens=True`
Flip the argument in `get_text()`; nothing else in the method changes.

- *Why*: it is the tokenizer's own notion of "control token", so it also covers tokens that did not show up in the four samples (e.g. `<|endoftext|>` if generation ends on that one instead). The measurements above show it removes exactly the trailing `<|user|>` and no content.
- *Alternative — regex/`removesuffix("<|user|>")` on the decoded string*: needs a hand-maintained token list, only handles the one observed token, and would still leave other control tokens in. Rejected.
- *Alternative — cut the generated ids at the first id in `generation_config.eos_token_id` and keep `skip_special_tokens=False`*: more code and still keeps any non-terminal special token. Rejected.

### 2. No whitespace trimming
The decoded text is returned as-is apart from the removed token. The measured outputs have no trailing whitespace before the token, and trimming is outside this change's goal.

### 3. Testing approach
- Unit test (mocked processor, as today): assert `decode` is called with `skip_special_tokens=True` and that the decoded value is returned unchanged. A mock cannot prove tokenizer behavior, so this is a regression pin, not the evidence.
- Evidence comes from a manual run against the real model (design context above is the baseline): rerun the four `data/` images, check that no `<|` marker remains and that the new text equals the old text minus the trailing `<|user|>`.

## Risks / Trade-offs

- [A future GLM-OCR revision emits a special token that carries content, e.g. layout/box markup] → `skip_special_tokens=True` would silently drop it. Mitigation: the manual verification is repeatable and documented in tasks; revisit if the model is upgraded.
- [Classification output shifts for the same image because the LLM prompt no longer ends in `<|user|>`] → re-run `/classify_receipt` on `data/aldi_quittung.jpeg` and compare with the previous result (11 items; known categories noted in issue #5). Any difference is reviewed, not assumed harmless.
- [Reverting the unit-test assertion from `False` to `True` could mask a regression if someone flips it back] → the test name and the spec scenario document the intent.

## Migration Plan

Code-only change, no config, data or dependency migration. Rollback is reverting the one-line change.
