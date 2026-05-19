# Contributing to milk

Thank you for your interest in contributing!

See also: [Coding Standards](docs/developer/coding_standards.md) ·
[Programmer's Guide](docs/programmers_guide.md) ·
[Developer Tutorial](docs/developer/tutorial.md) ·
[Working with Git](docs/developer/WorkingWithGit.md) ·
[Code Assist](docs/code_assist.md)

## AI-Assisted Development

This project includes extensive support for AI coding
agents. See [`AGENTS.md`](AGENTS.md) for full onboarding.
Key resources live under `.agents/`:

- **Rules** — always-on guardrails that enforce
  conventions
- **Skills** — deep-dive instruction sets for
  specialized tasks
- **Workflows** — step-by-step task templates
  (invoke with slash commands)

## Getting Started

1. Fork the repository and clone your fork.
2. Create a feature branch from `framework-dev`:

   ```bash
   git checkout framework-dev
   git pull --ff-only origin framework-dev
   git checkout -b feat/my-change
   ```

3. Build and test your changes (see [Installation](docs/install/compile.md)).

## Code Style

Follow the project's [Coding Standards](docs/developer/coding_standards.md):

- Allman brace style, 100-character line limit
- One argument per line in function prototypes
- Kernel-Doc style documentation (`/** @brief ... */`)
- Minimize variable scope using code blocks
- Every `.c` file must include only the headers
  it directly uses
- Compile with `-Wall -Wextra` and fix all warnings

## Writing Compute Units

New functions should follow the V2 template pattern:

1. Copy `src/milk_module_example/examplefunc_fps_cli_poc.c`
2. Follow the 8-section structure documented in the file header
3. See the [Developer Tutorial](docs/developer/tutorial.md) for a walkthrough
4. See the [fpsexec conventions](.agents/rules/fpsexec-conventions.md)
   for CMake setup

## Commit Messages

Use conventional-style prefixes:

```text
feat: add new stream filter function
fix: correct semaphore race in DM combiner
docs: update FPS parameter table
refactor: extract processinfo loop into helper
```

## Pull Request Process

1. Target `framework-dev` (never `dev` or `main`).
2. Ensure your code compiles cleanly with
   `-Wall -Wextra`.
3. Run the test suite:
   `cd _build && ctest --output-on-failure`.
4. Update documentation if you add new functions,
   parameters, or CLI commands.
5. Keep PRs focused — one feature or fix per PR.
6. If AI-assisted, include "Prompt Summary" and
   "AI Authorship" sections in the PR body.

## Architecture Guidelines

- Avoid introducing cross-module dependencies. Review
  `docs/dependency_graph.md` before adding `#include` from
  other modules.
- Use the public API headers (`module_name/header.h`), not
  internal headers.
- See `docs/programmers_guide.md` for the layered architecture.

## Reporting Issues

Use the [issue templates](https://github.com/milk-org/milk/issues/new/choose)
when filing a bug, feature request, build problem, or
performance issue. The templates guide you through the
required information (version, reproduction steps, etc.)
and are integrated with the project's agent workflows
for faster triage.
