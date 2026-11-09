---
description: Set up a development environment from scratch
---

# Set Up Development Environment

Use this workflow for first-time setup of a milk
development environment.

## 1. Install Dependencies and Clone

Install the system dependencies and clone the repository with its submodules (`ImageStreamIO`
and others) as [compile.md §1](../../docs/quickstart/compile.md#1-download-and-install-milk)
shows.

## 2. Checkout the Development Branch

```bash
git checkout framework-dev
```

> [!CAUTION]
> Never commit to `dev` or `framework-dev` (`git-workflow` rule).
> Create feature branches from `framework-dev`.

## 3. Configure, Build and Install

Install under `_build/_install`, not the system prefix
([tooling.md §6](../../docs/developer/tooling.md#6-dev-installs-without-sudo)):

```bash
mkdir _build && cd _build
cmake ..
make -j$(nproc)
cmake --install . --prefix _install
```

Optional build flags:
[build_tiers.md](../../docs/quickstart/build_tiers.md#optional-dependencies),
[performance.md §7](../../docs/operations/performance.md#7-quick-checklist) (`-DVEC_REPORT=ON`,
PGO, static LTO) and
[debugging.md §5](../../docs/operations/debugging.md#5-build-time-debugging) (Debug build).

## 4. Set the Environment

Set `MILK_INSTALLDIR` and `PATH` as
[tooling.md §6.1](../../docs/developer/tooling.md#61-install-and-use) shows (`milk-check`
verifies them); to use milk commands from bash, `source milk.bash`
([sessions.md](../../docs/guide/sessions.md#using-milk-from-plain-bash)).

Verify the installation:

```bash
milk-cli -i
```

## 5. Run Tests

Set up and run the test suite as
[tooling.md §4.1](../../docs/developer/tooling.md#41-setup) and
[§4.2](../../docs/developer/tooling.md#42-pytest) show.

## 6. Verify the CLI

```bash
milk-cli -c '@module list'
```

## 7. Set Up Git Worktrees (Optional)

For working on multiple features simultaneously, follow WorkingWithGit.md
[§3](../../docs/developer/WorkingWithGit.md#3-parallel-development-using-git-worktrees); each
worktree needs its own `_build` directory.

## 8. Read the Onboarding Docs

- [`AGENTS.md`](../../AGENTS.md) — if using AI
  coding agents
- [`CONTRIBUTING.md`](../../CONTRIBUTING.md) —
  contribution guidelines
- [`docs/developer/code_assist.md`](../../docs/developer/code_assist.md)
  — rules and workflows index
