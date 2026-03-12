---
trigger: always_on
---

# Git Workflow — Pull Request Policy

All code changes **must** go through a pull
request. Never commit directly to `framework-dev`
(or `main`).

## Required Steps

1. **Create a feature branch** from `framework-dev`:
   - Use a descriptive name, e.g.
     `perf/mvm-blas-upgrade`,
     `fix/imfunctions-else-if`,
     `feat/new-stream-processor`.
   - Prefix conventions: `feat/`, `fix/`, `perf/`,
     `refactor/`, `docs/`, `chore/`.

2. **Make commits** on the feature branch.
   - Keep commits atomic and well-described.

3. **Push** the feature branch to origin.

4. **Create a PR** targeting `framework-dev`.
   - Include a clear title and description of
     the changes.

5. **Do not merge** — let the maintainer review
   and merge.
