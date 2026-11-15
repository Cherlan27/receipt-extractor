# Review: extract-receipt-metadata

## Round 1

Base `origin/main` @ 10774e7 · Head 1ff62d3 · 2026-11-15 · Reviewer: claude-opus-5-5

### Summary
The change implements every scenario of the delta spec with tests, and the real-model result is recorded in task 7.2. Layering, lifespan loading and docs follow the project rules. One MINOR prompt risk and one nit; the accepted CC ZERO limit (#18) is handled consistently in spec, design and tasks.

### Verification
| step | result |
|---|---|
| `make check` | ✅ 204 passed; pre-commit all passed; openspec validate 4 passed, 0 failed |
| `trail check` | ✅ no conflicts |
| probe: `RC(purchased_at=s)` → `ReceiptClassificationResponse(**dict(c))` for `T19:33:55`, `T00:00:00`, `25.02.26 00:00` | ✅ stays `datetime`, serializes with time |
| probe: `total=1e-7`, `quantity=2.0` | ✅ `1e-07` number; `2.0` rejected (matches design 2) |

### Findings

#### R1 · MINOR · `src/receipt_extractor/llm/prompts/classify_prompt.py:21`
The prompt says the price is "immer der Betrag unter dem Produktnamen" and the example shows only the ALDI layout (amount on the next line). The old same-line example (`Apfel 1.99`) was removed. When OCR text puts the amount on the same line as the product (as in the spec's `/classify_text` example), the model may take the next line's amount. Not proven: the real-model check covers only the ALDI sample; found by reading the prompt.
- Recommendation: defer — #5 rewrites this prompt; drop "immer" or add a same-line example there and check a second receipt layout.
- Decision: pending
- Status: open

### Nits
- N1 · `openspec/changes/extract-receipt-metadata/tasks.md:40` — the task text still lists "`CC ZERO/LIGHT/KOFF.FRE` quantity `1`" as verified; the note below explains it, but striking that part of the text would match the spec.
