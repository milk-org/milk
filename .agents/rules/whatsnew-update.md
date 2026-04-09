---
description: Update What's New page when significant
  features are added.
---

# What's New Page Updates

When a PR introduces a **significant** feature,
you MUST add an entry to `docs/whatsnew.md`.

## What Qualifies

Include:

- New user-facing features or commands
- Major refactors that change APIs or behavior
- Performance improvements with measurable impact
- New build modes or infrastructure
- New standalone executables

Exclude:

- Minor bugfixes and typo corrections
- CI-only changes
- Internal code cleanup with no API impact
- Documentation-only PRs (unless a new doc
  page is added)

## Entry Format

Prepend a one-line entry to the current month
under the correct branch heading (`framework-dev`
or `dev`). Create a new `### YYYY-MM` sub-heading
if the month doesn't exist yet.

```markdown
- **YYYY-MM-DD** — Short description `#tag1`
  `#tag2`
  ([PR #NNN](https://github.com/milk-org/milk/pull/NNN))
```

### Rules

1. Keep the description to **one line** (≤ 100 chars
   before the tag+link continuation line).
2. Use one or more tags from: `#cli`,
   `#performance`, `#streams`, `#fps`, `#build`,
   `#docs`, `#api`, `#refactor`.
3. Link to the PR. If no PR exists yet, link to the
   commit SHA instead.
4. Newest entries go **first** within their month.
5. Date is the merge date (or commit date if no PR).
