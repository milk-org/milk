---
trigger: always_on
---

# Git Workflow — Agent Policy

The workflow (feature branches, pull requests, worktrees) is in
[docs/developer/WorkingWithGit.md](../../docs/developer/WorkingWithGit.md). For agents:

- Agents never commit to `dev` or `framework-dev`. They may commit to other branches.
- Every agent change to `dev` or `framework-dev` goes through a pull request.

## Commit Messages

- Conventional prefixes: `feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `chore:`, `test:`;
  branch names use the same words (`feat/new-stream-processor`, `fix/imfunctions-else-if`).
- Subject: at most 72 characters, imperative mood. Body wrapped at 72 characters, saying why, not
  only what. Reference issues when applicable (`Fixes #42`).

## Pull Requests — User Confirmation Required

Never submit a pull request without the user's explicit approval: show them the title, body,
target and source branches and draft status, apply their edits, and create the PR only once they
approve, even for a trivial change.

Every PR body includes:

- **Prompt summary:** what the user asked to be implemented (not the back-and-forth), and why the
  chosen approach was preferred to the alternatives considered.
- **Authorship:** every model that contributed code (ask the user to confirm; add `+ user edits`
  when they edited the code), and who reviewed and signed off on the PR.
