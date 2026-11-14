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

Follow the entries already in [`docs/whatsnew.md`](../../docs/whatsnew.md): newest first, under the
`## framework-dev` heading and a `### YYYY-MM` sub-heading (create it for a new month), as
`- **YYYY-MM-DD** — <bold summary sentence>. <what changed, with measured numbers when it is about
speed>` followed by tags from the list at the top of the page (`#cli`, `#performance`, ...). Link
the PR or commit when one exists.
