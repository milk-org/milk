---
trigger: always_on
---

# Git Workflow — Branching & Commit Policy

**CRITICAL BRANCHING RULE:**

- You are **STRICTLY FORBIDDEN** from modifying
  or pushing to the `dev` branch for `milk`,
  `cacao`, and `ImageStreamIO`.
- You must **ONLY** push to and merge into
  `framework-dev` or feature branches derived
  from `framework-dev`.

## Direct Commits to `framework-dev`

**Small, trivial changes** may be committed
directly to `framework-dev` without a PR.
Examples of small changes:

- Typo fixes in comments, docs, or strings
- Whitespace / formatting corrections
- Updating a single line in a config file
- Minor doc wording improvements

For these changes:

1. Commit directly to `framework-dev`.
2. Use a clear commit message with a conventional
   prefix (e.g., `docs: fix typo in README`,
   `chore: fix whitespace`).
3. Push to origin.

## Non-Trivial Changes — Ask the User

For **all other changes** (new features, bug
fixes, refactors, performance improvements,
multi-file edits, etc.), **ask the user** how
they want to proceed before committing:

> "Should I commit this directly to
> `framework-dev`, or create a feature branch
> and PR?"

Then follow whichever path the user chooses.

### Option A: Direct Commit to `framework-dev`

If the user says to commit directly:

1. Commit to `framework-dev` with a clear,
   conventional-style message.
2. Push to origin.

### Option B: Feature Branch + PR

If the user says to use a PR:

1. **Create a feature branch** from
   `framework-dev`:
   - Use a descriptive name, e.g.
     `perf/mvm-blas-upgrade`,
     `fix/imfunctions-else-if`,
     `feat/new-stream-processor`.
   - Prefix conventions: `feat/`, `fix/`,
     `perf/`, `refactor/`, `docs/`, `chore/`.

2. **Make commits** on the feature branch.
   - Keep commits atomic and well-described.
   - Use conventional-style prefixes:
     `feat:`, `fix:`, `perf:`, `refactor:`,
     `docs:`, `chore:`, `test:`.
   - Subject line: max 72 characters, imperative
     mood (e.g., "feat: add stream filter").
   - Body: wrap at 72 characters, explain _why_
     not just _what_.
   - Reference issues when applicable
     (e.g., `Fixes #42`).

3. **Push** the feature branch to origin.

4. **Create a PR** targeting `framework-dev`.
   - Include a clear title and description.

5. **Do not merge** — let the maintainer review
   and merge.

## PR Submission — User Confirmation Required

**Never** submit a pull request without explicit
user approval. Before calling any PR-creation tool:

1. **Draft the PR text** and present it to the
   user via `notify_user`, including:
   - **Title** — concise summary of the change.
   - **Body** — full description (motivation,
     what changed, testing done, etc.).
   - **Target branch** (normally `framework-dev`).
   - **Source branch**.
   - **Draft status** (draft or ready for review).

2. **Wait for the user to confirm or edit.**
   The user may revise the title, body, or any
   other field. Apply all requested changes.

3. **Only after explicit approval**, call the
   PR-creation tool with the finalized text.

> [!CAUTION]
> Submitting a PR without user confirmation is
> **strictly forbidden**, even if the changes
> seem trivial.

## PR Body — Required Content

Every PR body **must** include:

### Prompt Summary

A concise summary of the user prompts/requests
that led to the PR. Focus on **what** was asked
to be implemented, not the detailed steps or
back-and-forth discussions. If multiple design
options were considered, briefly explain why the
chosen approach was selected over the alternatives.

### Authorship

- **Model(s) used** — list every model that
  contributed code in this PR (e.g., **Gemini 3.1 Pro**).
  Always ask the user to confirm which model was used.
  If the user contributed by directly editing the code,
  append `+ user edits` next to the AI model.
- **Reviewed and signed off by** — list the user
  who reviewed and approved the PR draft.

## Parallel Development (Git Worktrees)

To work on multiple feature tracks simultaneously
(e.g., CLI, Performance, Documentation), use
**Git Worktrees** rather than switching branches
in a single directory.

1. Keep your main repository (e.g., `~/src/milk`)
   on `framework-dev`.
2. Create dedicated worktrees for different tracks
   next to your main repository:
   ```bash
   git worktree add ../milk-docs \
       -b docs/current-task framework-dev
   git worktree add ../milk-cli \
       -b feat/current-task framework-dev
   ```
3. **Reusing Worktrees:** When starting a new task
   in the same track, do not create a new worktree.
   Go to the existing worktree, update
   `framework-dev`, and checkout a new branch:
   ```bash
   cd ~/src/milk-cli
   git checkout framework-dev
   git pull --ff-only origin framework-dev
   git checkout -b feat/next-task
   ```
4. **Build Isolation:** Always use a dedicated
   `_build` directory inside each worktree so
   CMake caches and object files remain isolated.
