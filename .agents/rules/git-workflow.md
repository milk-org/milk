---
trigger: always_on
---

# Git Workflow — Pull Request Policy

All code changes **must** go through a pull
request. Never commit directly to `framework-dev`
(or `main`).

**CRITICAL BRANCHING RULE:**
- You are **STRICTLY FORBIDDEN** from modifying or pushing to the `dev` branch for `milk`, `cacao`, and `ImageStreamIO`.
- You must **ONLY** push to and merge into `framework-dev` or feature branches derived from `framework-dev`.

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

## Parallel Development (Git Worktrees)

To work on multiple feature tracks simultaneously (e.g., CLI, Performance, Documentation), use **Git Worktrees** rather than switching branches in a single directory.

1. Keep your main repository (e.g., `~/src/milk`) on `framework-dev`.
2. Create dedicated worktrees for different tracks next to your main repository:
   ```bash
   git worktree add ../milk-docs -b docs/current-task framework-dev
   git worktree add ../milk-cli -b feat/current-task framework-dev
   ```
3. **Reusing Worktrees:** When starting a new task in the same track, do not create a new worktree. Go to the existing worktree, update `framework-dev`, and checkout a new branch:
   ```bash
   cd ~/src/milk-cli
   git checkout framework-dev
   git pull --ff-only origin framework-dev
   git checkout -b feat/next-task
   ```
4. **Build Isolation:** Always use a dedicated `_build` directory inside each worktree so CMake caches and object files remain isolated.
