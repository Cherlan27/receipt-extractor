# Workflow Retrospectives

Recorded by `/agent-workflow:adjust-workflow <issue>` after a shipped issue.
Each entry is marked *generic* (belongs in the agent-workflow plugin repo) or
*project* (specific to this repository).

## 2026-11-20 · Issue #14 · extract-receipt-metadata

Track Full · code map trail · ~25 min worker time (wall 4 days incl. gates and a 10.5 h stall) · 5 workers · 1 review rounds ·
findings C/M/m 0/0/1 · findings judged wrong 0 · escaped defects 0 · recommendations accepted unchanged yes

Evidence:
- Apply worker crashed 1 min after start (provider exit, code 3221225786, 2026-11-14 23:19). The coordinator answered the `providerExited` escalation with nothing. The human typed `continue` in the worker chat at 09:53 the next day: 10.5 h lost. The 30-min phase budget did not fire because nothing woke the coordinator.
- After resuming, the apply worker ended its turn while the server loaded the models (waiting on a background monitor) and needed a second nudge.
- Tooling (graft/trail hooks) rewrites `.claude/settings.json` and `.claude/settings.local.json` in every worktree, so every worker spec needed a "do not commit these" caveat. The trail PreToolUse hook also blocked one `worker-start` whose spec text named the trail hook script path.
- No change: missing deps in the fresh worktree were caught by the precondition in ~2 min. The real-model scenario the proposal added beyond the issue (CC ZERO quantity 1) failed, but it found a real defect (follow-up #18), and the worker did not fake the tick.
- Code map: first retro, no baseline on `none` yet. No worker reported trail as missing.

| # | Change | Expected effect | File | Scope | Status |
|---|---|---|---|---|---|
| A1 | Coordinator: answer an escalation with `failureKind: providerExited` at once with one `continue` send, log it, and tell the human if it exits again. After every `worker-start`, set a wake-up at `limits.phase_minutes` so the time budget is a mechanism, not a rule | No multi-hour stall after a worker crash | plugin `skills/orchestrate-issue` | generic | proposed for plugin repo |
| A2 | Apply rule: start the server and wait for port 8080 in the same turn with a bounded loop (max 15 min); never end the turn while the server loads | Real-model checks finish without a coordinator nudge | `openspec/config.yaml` | project | applied |
| A3 | Commit the tooling-written `.claude/settings.json` and `.claude/settings.local.json` on main once | `git status` is clean after setup; specs need no caveat | `.claude/settings*.json` | project | applied |

Previous changes: none (first retro).
