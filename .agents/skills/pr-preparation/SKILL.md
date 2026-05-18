---
name: pr-preparation
description: End-to-end pull request packaging with
  templated title, body, AI authorship, and prompt
  summary sections
---

# PR Preparation

This skill automates the end-to-end workflow of
packaging completed work into a pull request,
following the project's `git-workflow.md` rules.

## When to Use

- After completing any code changes that should
  be merged into `framework-dev`
- Triggered by user saying "create PR", "submit
  PR", "prepare PR", or similar

## Prerequisites

- All changes are committed on a feature branch
- The feature branch has been pushed to origin
- Build passes (`/compile-test`)
- Tests pass (if applicable)

## Step 1 — Gather Context

Collect the following automatically:

```bash
# Current branch name
BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Target branch (always framework-dev)
TARGET="framework-dev"

# Commits on this branch not in target
git log --oneline ${TARGET}..HEAD

# Files changed
git diff --stat ${TARGET}..HEAD

# Diff summary by component
git diff --stat ${TARGET}..HEAD | \
  awk -F/ '{print $1"/"$2}' | \
  sort | uniq -c | sort -rn
```

## Step 2 — Draft the PR Title

Format: `<type>: <concise description>`

Type prefixes (match branch prefix):
- `feat:` — new feature
- `fix:` — bug fix
- `perf:` — performance improvement
- `refactor:` — code restructuring
- `docs:` — documentation only
- `chore:` — maintenance, CI, tooling

Example: `refactor: split CLIcore_script.c into
logical sub-modules`

Keep the title under 72 characters.

## Step 3 — Draft the PR Body

Use this template:

```markdown
## Summary

[1-2 sentence overview of what changed and why]

## Changes

### [Component 1]
- Change description
- Change description

### [Component 2]
- Change description

## Verification

- [ ] Build passes (`/compile-test`)
- [ ] Tests pass (ctest / CLI robustness)
- [Other verification steps performed]

## Prompt Summary

[Concise summary of the user prompts/requests
that led to this PR. Focus on WHAT was asked,
not the detailed back-and-forth. If design
alternatives were considered, briefly explain
why this approach was chosen.]

## Authorship

- **Model**: Gemini 3.1 Pro (ask user to confirm)
- **Reviewed and signed off by**: [User's name]
```

## Step 4 — Present to User

Present the complete PR draft to the user via
`notify_user`, including:
- **Title**
- **Body** (full markdown)
- **Source branch**
- **Target branch** (`framework-dev`)
- **Draft status** (draft or ready for review)

> [!CAUTION]
> Never submit the PR without explicit user
> approval. Wait for confirmation.

## Step 5 — Submit (after approval)

Only after the user approves:

1. Push the branch if not already pushed:
   ```bash
   git push -u origin ${BRANCH}
   ```

2. Create the PR using the approved title and
   body.

3. Report the PR URL to the user.

## Common Adjustments

- **User wants to split into multiple PRs**:
  help them identify commit boundaries and
  create separate branches.
- **User edits the draft**: apply all requested
  changes before submitting.
- **Draft PR**: if user says "draft", set
  `is_draft: true`.
- **Reviewers**: ask if they want to request
  specific reviewers.

## Branch Naming Verification

Before creating the PR, verify the branch name
follows conventions:

| Prefix | For |
|--------|-----|
| `feat/` | New features |
| `fix/` | Bug fixes |
| `perf/` | Performance work |
| `refactor/` | Restructuring |
| `docs/` | Documentation |
| `chore/` | Maintenance |

If the branch name doesn't follow convention,
suggest renaming before PR creation (this
requires creating a new branch and
cherry-picking).
