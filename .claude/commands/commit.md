Analyze the staged changes and produce a conventional commit message.

## Steps

1. Run `git diff --cached` to get the full staged diff.
2. Run `git diff --cached --stat` to get a file-level summary.
3. If there are no staged changes, tell the user and stop.

## Conventional Commits standard

Choose the **type** from this list — pick the single best fit:

| Type | When to use |
|------|-------------|
| `feat` | A new feature or capability visible to users/callers |
| `fix` | A bug fix |
| `refactor` | Code change that is neither a fix nor a feature (restructure, rename, simplify) |
| `perf` | A change that improves performance |
| `test` | Adding or correcting tests |
| `docs` | Documentation only |
| `chore` | Build system, dependencies, tooling, config (no production code change) |
| `ci` | CI/CD pipeline changes |
| `style` | Formatting, whitespace — no logic change |

**Scope** (optional): a short noun in parentheses naming the subsystem affected, e.g. `feat(api):`, `fix(extractor):`.
Use a scope when the change is clearly isolated to one component; omit it when the change is cross-cutting.

**Subject line rules**:
- Imperative mood ("add", not "added" or "adds")
- Lowercase first letter
- No period at the end
- ≤ 72 characters total (including `type(scope): `)

**Body** (optional): Add a body only when the *why* or *how* is not obvious from the subject. Separate from the subject with a blank line. Wrap at 72 characters.

**Breaking changes**: append `!` after the type/scope and add a `BREAKING CHANGE:` footer if the public API changes incompatibly.

## Output

Print **only** the final commit message, wrapped in a code block, ready to copy-paste. Do not explain your reasoning unless the staged diff is ambiguous and you need to ask a clarifying question.

Example output:
```
feat(extractor): support multi-page receipt images
```

Or with a body:
```
fix(api): return 400 on unsupported image format

Previously the endpoint would crash with an unhandled PIL exception
when the uploaded file was not a valid image.
```
