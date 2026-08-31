# Working with git

<!-- prettier-ignore -->
!!! note
    This file: `docs/developer/WorkingWithGit.md`

See also: [Documenting Code](DocumentingCode.md) · [Adding Plugins](plugins.md) ·
[Compile Instructions](../quickstart/compile.md)

---

## 1. Source Code and Modules

Most of milk's code is organized in directories under `src/` (core modules) and `plugins/` (optional
modules). Modules are compiled as shared objects loaded by the main process at runtime.

Some plugin directories (e.g., `plugins/cacao-src`) may be symbolic links to external source trees
or git submodules.

## 2. Standard Development Workflow

All code changes **must** go through a pull request. Never commit directly to `framework-dev` (or
`main`).

**CRITICAL BRANCHING RULE:**

- You are **STRICTLY FORBIDDEN** from modifying or pushing to the `dev` branch for `milk`, `cacao`,
  and `ImageStreamIO`.
- You must **ONLY** push to and merge into `framework-dev` or feature branches derived from
  `framework-dev`.

### Required Steps

1. **Create a feature branch** from `framework-dev`:
   - Use a descriptive name, e.g., `feat/new-stream-processor`, `fix/imfunctions-else-if`,
     `perf/mvm-blas-upgrade`, `docs/update-guide`.

2. **Make commits** on the feature branch. Keep commits atomic and well-described.
3. **Push** the feature branch to origin.
4. **Create a PR** targeting `framework-dev`. Include a clear title and description of the changes.
5. **Do not merge** — let the maintainer review and merge.

## 3. Parallel Development using Git Worktrees

The `milk` project enforces strict branching rules where all work must occur on feature branches off
of `framework-dev`. If you need to work on multiple tracks simultaneously (e.g., optimizing
performance while also updating documentation or fixing a CLI bug), you should use **Git
Worktrees**.

Git worktrees allow you to have multiple isolated working directories that share the exact same
underlying local `.git` repository.

### Best Practices

1. **Keep the Main Clone Clean:** Keep your primary repository (e.g., `~/src/milk`) checked out to
   `framework-dev`.

2. **Create Track-Specific Worktrees:** Create sibling directories for your parallel tracks.

   ```bash
   cd ~/src/milk
   git worktree add ../milk-docs -b docs/my-task framework-dev
   git worktree add ../milk-cli -b feat/my-cli-task framework-dev
   ```

3. **Isolate Builds:** Each worktree should have its own dedicated `_build` directory. This ensures
   CMake caches and compiled object files do not interfere with each other when you context-switch.

4. **Reusing Worktrees:** When you finish a task (e.g., your CLI feature is merged), **do not**
   delete the worktree. Reuse it for your next CLI task to preserve your CMake build cache:

   ```bash
   cd ~/src/milk-cli
   git checkout framework-dev
   git pull --ff-only origin framework-dev
   git checkout -b feat/my-next-cli-task
   ```

<!-- prettier-ignore -->
!!! tip
   A helper script `milk-setup-worktrees` is available to automatically scaffold this directory layout
   and initialize the build directories for you.

## 4. Releasing a New Version

When releasing a new version from `framework-dev` (or the equivalent stable branch):

- Update version number in `CMakeLists.txt`
- Update version information in `README.md`

Merge `framework-dev` into `main`, then tag:

```bash
$ git checkout main
$ git merge framework-dev
$ git tag -a vX.YY.ZZ -m "milk version X.YY.ZZ"
$ git push origin main --tags
```

<!-- prettier-ignore -->
!!! note
    Modules that are shared between packages (e.g., `milk` and `cacao`) can have parallel version number
    histories. Any new version, regardless of which package it is associated with, includes all previous
changes.

---

## 5. Source Code Documentation (doxygen)

For generating HTML source code documentation via Doxygen, refer to the up-to-date guide in
[DocumentingCode.md](DocumentingCode.md).

---

← [Documentation Index](../index.md)
