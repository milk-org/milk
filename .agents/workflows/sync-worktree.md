---
description: Sync worktree to latest framework-dev and start a new feature branch
---

# Sync Worktree

Run this workflow at the **start of a new task** in any
worktree to fast-forward `framework-dev` and create a
fresh feature branch from it.

## Steps

1. **Check for uncommitted changes.** Run `git status`
   in the worktree root. If there are uncommitted
   changes, ask the user whether to:
   - `git stash` them, or
   - abort the sync.

2. **Ask the user for the new branch name** if they
   haven't provided one. Use the prefixes of the
   `git-workflow` rule, *Commit Messages*.

3. **Update `framework-dev` and create the branch** with the commands of WorkingWithGit.md
   [§3](../../docs/developer/WorkingWithGit.md#3-parallel-development-using-git-worktrees)
   (*Reusing Worktrees*). If the fast-forward fails (local `framework-dev` has
   diverged), stop and report the issue to the user.

4. **Prompt the user** whether to rebuild now. If yes,
   run the `/compile-test` workflow. If no, skip.

5. Report that the worktree is synced and ready for
   work on the new branch.
