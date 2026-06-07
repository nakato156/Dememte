---
name: close-pr
description: Merge a PR the right way for this repo — CI green, Copilot review resolved, squash, cleanup, close the bead. Use when finishing/merging a pull request (e.g. "merge #9", "close this PR").
---

# close-pr — merge a PR correctly

Encapsulates the PR-close checklist for DeMemte so steps don't get skipped. The step that
caused a real miss before: **merging on green CI alone, without reading the Copilot review.**
Never merge before the Copilot review is read, answered, and its threads resolved.

> Note: this skill is ergonomics, not enforcement — it only runs when invoked. The hard
> guarantee is GitHub branch protection on `main` (*require conversation resolution before
> merging*). If a merge ever goes through with unresolved Copilot threads, that rule is missing
> — ask the repo admin (nakato156) to enable it (see "Hard gate" below).

## Inputs
- PR number (from the argument). If absent, resolve it from the current branch:
  `gh pr view --json number,headRefName,url`.

## Workflow

1. **CI must be green.**
   ```bash
   gh pr checks <PR>            # every check must pass (not pending/fail)
   gh pr view <PR> --json mergeable,mergeStateStatus
   ```
   If pending, wait and re-check. If failing, stop and fix — do not merge.

2. **Copilot review — read, answer, resolve. NEVER skip this.**
   ```bash
   gh pr view <PR> --json reviews \
     -q '.reviews[] | select(.author.login=="copilot-pull-request-reviewer") | .body'
   gh api repos/nakato156/Dememte/pulls/<PR>/comments \
     -q '.[] | "ID:\(.id) \(.path):\(.line // .original_line)\n\(.body)\n---"'
   ```
   For each inline comment: judge it. If valid, **fix it** (in this branch if not yet merged,
   or a follow-up PR if already merged) and reply pointing to the fix; if a false positive, reply
   why. Reply via:
   ```bash
   gh api repos/nakato156/Dememte/pulls/<PR>/comments -f body="…" -F in_reply_to=<comment_id>
   ```
   Resolve the threads. Only proceed once every comment is answered.
   (Reminder: `ruff` skips `notebooks/`, so dead imports there won't fail CI — read Copilot for those.)

3. **Merge** (squash is the repo default) and delete the remote branch.
   ```bash
   gh pr merge <PR> --squash --delete-branch
   ```

4. **Sync + cleanup** locally.
   ```bash
   git checkout main && git pull --ff-only origin main
   git branch -D <feature-branch>        # if the local checkout step earlier was blocked, clean .beads/issues.jsonl first
   ```

5. **Close the bead** the PR completed (commit/push the beads export to main).
   ```bash
   bd close <id> --reason "PR #<PR> merged: …" --suggest-next
   bd export > .beads/issues.jsonl && git add .beads/issues.jsonl
   git commit -m "chore(beads): close <id>" && git push origin main
   ```

6. **Verify**: `gh pr view <PR> --json state` is MERGED; `git status` clean; `git branch` has no
   stale feature branch; `bd ready` reflects newly unblocked work.

## Conventions
- Git identity for commits: `r0sewt <u202216562@upc.edu.pe>`. No Claude/AI co-author trailer.
- `out/` is gitignored everywhere — CSV/checkpoints are regenerated, not versioned.

## Hard gate (one-time, repo admin only)
Branch protection is what actually prevents the "merged without resolving Copilot" miss
regardless of who merges. Admin runs:
```bash
gh api -X PUT repos/nakato156/Dememte/branches/main/protection --input - <<'JSON'
{"required_status_checks":null,
 "enforce_admins":false,"required_pull_request_reviews":null,
 "restrictions":null,"required_conversation_resolution":true}
JSON
```
The key field is `required_conversation_resolution` (blocks merge with unresolved Copilot
threads). To *also* require CI, confirm the check's exact context name first (`gh pr checks`
shows it — here `test`) and set `"required_status_checks":{"strict":true,"contexts":["test"]}`;
using a wrong context silently fails to require the check.
