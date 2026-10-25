## 1. Implementation

- [x] 1.1 In `services/extractor.py`, change `ReceiptExtractor.get_text()` to decode with `skip_special_tokens=True`
- [x] 1.2 Update `tests/receipt_extractor/services/test_extractor.py` (`test_returns_decoded_text_for_valid_image`): assert `skip_special_tokens is True` instead of `False`, keeping the returned text unchanged
- [x] 1.3 Add a test that a decoded value is returned verbatim and that `get_text` still returns `""` for an undecodable image (guards the scenario "image without recognizable text")

## 2. Verification

- [x] 2.1 Run `poetry run pytest` — all tests green
- [x] 2.2 Run the pre-commit hooks (`uvx pre-commit run --files <changed files>` if `pre-commit` is not installed) and fix findings
- [x] 2.3 Real-model check: run `ReceiptExtractor.get_text()` on all images in `data/`; confirm none of the outputs contains `<|`, and that each equals the old output minus the trailing `<|user|>` (record the result in the PR/notes)
- [x] 2.4 Real-model check through the API: `POST /extract_text` and `POST /classify_receipt` with `data/aldi_quittung.jpeg`; `raw_text` has no control token and equals the `/extract_text` text; compare `items` with the previous run (11 items) and review any differences
- [x] 2.5 Stop the server afterwards (port 8080 free)

## 3. Docs

- [x] 3.1 Update `CLAUDE.md` (`ReceiptExtractor.get_text` description): decoded text has special tokens skipped, so no `<|user|>` end token appears in OCR output
