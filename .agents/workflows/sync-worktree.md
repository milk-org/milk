---
description: Sync worktree to latest framework-dev and start a new feature branch
---

# Sync Worktree

Run this workflow at the **start of a new task** in any
worktree to rebase onto the latest `framework-dev` and
create a fresh feature branch.

## Steps

1. **Check for uncommitted changes.** Run `git status`
   in the worktree root. If there are uncommitted
   changes, ask the user whether to:
   - `git stash` them, or
   - abort the sync.

2. **Fetch latest refs from origin:**

// turbo

```bash
git fetch origin
```

3. **Switch to framework-dev and fast-forward:**

```bash
git checkout framework-dev
git pull --ff-only origin framework-dev
```

If the fast-forward fails (local `framework-dev` has
diverged), stop and report the issue to the user.

4. **Ask the user for the new branch name** if they
   haven't provided one. Follow the naming convention
   from the `git-workflow` rule:
   - `feat/`, `fix/`, `perf/`, `refactor/`, `docs/`,
     `chore/` prefixes.

5. **Create and switch to the new feature branch:**

```bash
git checkout -b <branch-name>
```

6. **Prompt the user** whether to rebuild now. If yes,
   run the `/compile-test` workflow. If no, skip.

7. Report that the worktree is synced and ready for
   work on the new branch.
