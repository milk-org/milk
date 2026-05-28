# Code Assist Tools

See also: [Programmer's Guide](programmers_guide.md) ·
[Coding Standards](developer/coding_standards.md) ·
[Adding Plugins](developer/plugins.md) ·
[Template Source Code](developer/TemplateSourceCode.md)

The `milk` project includes **agent rules** and
**workflows** that guide AI coding assistants
(Gemini, Copilot, etc.) to follow project
conventions automatically. They live under:

```text
.agents/
├── rules/        # Always-on guardrails
├── skills/       # Deep-dive instruction sets
└── workflows/    # On-demand task templates
```

Collaborators benefit even without using an AI
assistant — the rule and workflow files document
the conventions, checklists, and cross-references
that every contributor should know.

## Getting Started

New to adding capabilities? Start with these:

1. [Developer Tutorial](developer/tutorial.md) —
   write your first module end-to-end.
2. [Adding Plugins](developer/plugins.md) — CMake
   setup, dual-mode headers, `_compute` variants.
3. [Template Source Code](developer/TemplateSourceCode.md)
   — which template file to copy for each use case.

## Agent Rules

Rules fire **automatically** when a task touches
relevant code. They enforce conventions without
requiring you to remember every checklist.

| Rule                    | File                                                                                                                                         | What it enforces                                                                     |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Agent docs update       | [`agent-docs-update.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/rules/agent-docs-update.md)                             | Update AGENTS.md and code_assist.md when agent files change.                         |
| Architecture principles | [`architecture-principles.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/rules/architecture-principles.md)                 | Minimize cross-module deps; consult `dependency_graph.md` before adding new ones.    |
| CMake conventions       | [`cmake-conventions.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/rules/cmake-conventions.md)                             | Use `PUBLIC`/`INTERFACE` properties; each module owns its headers.                   |
| Code style              | [`code-style-guide.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/rules/code-style-guide.md)                               | 100-char lines, Kernel-Doc, Linux kernel style, explicit includes.                   |
| Common agent mistakes   | [`common-agent-mistakes.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/rules/common-agent-mistakes.md)                     | Consolidated checklist of frequent AI code-generation pitfalls.                      |
| Compile after edit      | [`compile-after-edit.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/rules/compile-after-edit.md)                           | Always run `/compile-test` after modifying C/CMake.                                  |
| Concurrency practices   | [`concurrency-practices.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/rules/concurrency-practices.md)                     | Semaphore protocol, FPS sync, process coordination.                                  |
| Defensive programming   | [`defensive-programming-practices.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/rules/defensive-programming-practices.md) | Buffer safety, pointer discipline, bounded input validation.                         |
| Documentation site      | [`documentation-site.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/rules/documentation-site.md)                           | MkDocs structure, page creation, tag categories.                                     |
| Documentation standards | [`documentation-standards.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/rules/documentation-standards.md)                 | Markdown formatting, shell prompts, link checking.                                   |
| Error handling          | [`error-handling-practices.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/rules/error-handling-practices.md)               | Use milkDebugTools.h macros for errors.                                              |
| fpsexec conventions     | [`fpsexec-conventions.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/rules/fpsexec-conventions.md)                         | V2 template, 8-section layout, `-h1` requirement.                                    |
| Git workflow            | [`git-workflow.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/rules/git-workflow.md)                                       | PRs from feature branches into `framework-dev`; commit conventions.                  |
| Help consistency        | [`help-consistency.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/rules/help-consistency.md)                               | Cross-check all sibling help sources when editing help content.                      |
| Help message standard   | [`help-message-standard.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/rules/help-message-standard.md)                     | Unified help format, flags (`-h`/`-h1`/`-hm`), ANSI color palette via `milk_help.h`. |
| Local install/test      | [`local-install-test.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/rules/local-install-test.md)                           | Install to `_build/_install` via `--prefix`; never use `sudo` or system paths.       |
| Programmer's Guide      | [`maintain-programmers-guide.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/rules/maintain-programmers-guide.md)           | Update `docs/programmers_guide.md` on architectural changes.                         |
| Module dependencies     | [`module-deps-declaration.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/rules/module-deps-declaration.md)                 | MODULE_DEPS and INIT_MODULE_LIB_DEPS macros.                                         |
| Naming conventions      | [`naming-conventions.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/rules/naming-conventions.md)                           | File, function, variable, macro naming; loop index types.                            |
| Performance practices   | [`performance-practices.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/rules/performance-practices.md)                     | SIMD, BLAS, pointer alignment, type dispatch, memory allocation, CPU pinning.        |
| README updates          | [`readme-update.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/rules/readme-update.md)                                     | Update module README when source files are added/removed.                            |
| Running commands        | [`run-milk-commands.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/rules/run-milk-commands.md)                             | Environment setup, SHM cleanup, tmux session management.                             |
| Script documentation    | [`script-docs.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/rules/script-docs.md)                                         | Update `docs/scripts.md` and add `--help` when scripts change.                       |
| Script naming           | [`script-naming.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/rules/script-naming.md)                                     | `milk-*` for OS executables, `.milk` for CLI scripts.                                |
| Shared memory safety    | [`shared-memory-safety.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/rules/shared-memory-safety.md)                       | SHM cleanup, stale detection, stream creation.                                       |
| Testing practices       | [`testing-practices.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/rules/testing-practices.md)                             | Run tests after changes; add regression tests.                                       |
| TUI browser testing     | [`tui-browser-testing.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/rules/tui-browser-testing.md)                         | milk TUIs cannot be tested using browser testing tools.                              |
| Workspace layout        | [`files-directories.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/rules/files-directories.md)                             | cacao lives at `plugins/cacao-src` → `~/src/cacao`.                                  |
| What's New              | [`whatsnew-update.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/rules/whatsnew-update.md)                                 | Add entry to `docs/whatsnew.md` for significant features.                            |

## Skills

Skills live in `.agents/skills/` and provide deep
context for specialized tasks. Each skill folder
contains a `SKILL.md` with detailed instructions.

| Skill                      | Folder                                                                                                                                 | What it provides                                                                                |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Advanced math patterns     | [`advanced-math-patterns`](https://github.com/milk-org/milk/blob/framework-dev/.agents/skills/advanced-math-patterns/SKILL.md)         | High-performance mathematical and DSP operations, BLAS, FFT, and vectorization.                 |
| API quick reference        | [`api-quick-reference`](https://github.com/milk-org/milk/blob/framework-dev/.agents/skills/api-quick-reference/SKILL.md)               | API cheat sheet for IMGID, processinfo macros, stream variables, datatypes, and parameter sync. |
| Feature planner            | [`feature-planner`](https://github.com/milk-org/milk/blob/framework-dev/.agents/skills/feature-planner/SKILL.md)                       | Structured planning and architectural decomposition for new features.                           |
| Batch Kernel-Doc           | [`batch-kernel-doc`](https://github.com/milk-org/milk/blob/framework-dev/.agents/skills/batch-kernel-doc/SKILL.md)                     | Systematic function documentation with scanning, templates, and batch processing.               |
| CLI test writer            | [`cli-test-writer`](https://github.com/milk-org/milk/blob/framework-dev/.agents/skills/cli-test-writer/SKILL.md)                       | Writing test cases for the CLI robustness suite with coverage analysis.                         |
| CMake patterns             | [`cmake-patterns`](https://github.com/milk-org/milk/blob/framework-dev/.agents/skills/cmake-patterns/SKILL.md)                         | Module CMake setup, standalone builds, `_compute` variants.                                     |
| Debug CLI behavior         | [`debug-cli-behavior`](https://github.com/milk-org/milk/blob/framework-dev/.agents/skills/debug-cli-behavior/SKILL.md)                 | Crash investigation, command registration tracing, display debugging.                           |
| Diagnose build failure     | [`diagnose-build-failure`](https://github.com/milk-org/milk/blob/framework-dev/.agents/skills/diagnose-build-failure/SKILL.md)         | CMake/GCC error triage mapped to milk's build tiers.                                            |
| FPS parameter guide        | [`fps-parameter-guide`](https://github.com/milk-org/milk/blob/framework-dev/.agents/skills/fps-parameter-guide/SKILL.md)               | FPS parameter types, flags, X-macro patterns, common mistakes.                                  |
| ImageStream internals      | [`imagestream-internals`](https://github.com/milk-org/milk/blob/framework-dev/.agents/skills/imagestream-internals/SKILL.md)           | SHM stream layout, semaphore protocol, circular buffers.                                        |
| Module loading internals   | [`module-loading-internals`](https://github.com/milk-org/milk/blob/framework-dev/.agents/skills/module-loading-internals/SKILL.md)     | `dlopen` sequence, `data.moduleindex` race, constructor timing.                                 |
| Milk script writer         | [`milk-script-writer`](https://github.com/milk-org/milk/blob/framework-dev/.agents/skills/milk-script-writer/SKILL.md)                 | Generate correct milk-cli scripts from natural language prompts.                                |
| Optimize compute function  | [`optimize-compute-function`](https://github.com/milk-org/milk/blob/framework-dev/.agents/skills/optimize-compute-function/SKILL.md)   | Systematic performance optimization methodology.                                                |
| Plugin creator             | [`plugin-creator`](https://github.com/milk-org/milk/blob/framework-dev/.agents/skills/plugin-creator/SKILL.md)                         | Scaffolding a new plugin module with CMake and module registration.                             |
| PR preparation             | [`pr-preparation`](https://github.com/milk-org/milk/blob/framework-dev/.agents/skills/pr-preparation/SKILL.md)                         | End-to-end PR packaging with template body and AI authorship.                                   |
| Pseudocode to compute unit | [`pseudocode-to-compute-unit`](https://github.com/milk-org/milk/blob/framework-dev/.agents/skills/pseudocode-to-compute-unit/SKILL.md) | Translating algorithms to V2 compute units.                                                     |
| Refactor C source          | [`refactor-c-source`](https://github.com/milk-org/milk/blob/framework-dev/.agents/skills/refactor-c-source/SKILL.md)                   | Safe file splitting with dependency analysis and CMake updates.                                 |
| Stream modifier guide      | [`stream-modifier-guide`](https://github.com/milk-org/milk/blob/framework-dev/.agents/skills/stream-modifier-guide/SKILL.md)           | IMGID parsing pipeline, `@S:`/`@L:`/`@F:` modifiers, slice syntax.                              |

## Workflows

Workflows are invoked by typing the slash command
as a chat message (e.g., `/compile-test`). They are
step-by-step checklists for common tasks.

| Command                     | File                                                                                                                               | What it does                                                                                  |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `/compile-test`             | [`compile-test.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/workflows/compile-test.md)                         | Incremental build, install, and test from `_build/`.                                          |
| `/create-fpsexec`           | [`create-fpsexec.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/workflows/create-fpsexec.md)                     | Scaffold a new V2 fpsexec standalone executable.                                              |
| `/create-plugin`            | [`create-plugin.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/workflows/create-plugin.md)                       | Scaffold a new plugin module with full boilerplate.                                           |
| `/add-new-module`           | [`add-new-module.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/workflows/add-new-module.md)                     | Scaffold a new module (README, CMake, boilerplate).                                           |
| `/add-function`             | [`add-function.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/workflows/add-function.md)                         | Add a function to an existing module (dispatches to sub-workflows).                           |
| `/add-stream-processor`     | [`add-stream-processor.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/workflows/add-stream-processor.md)         | Scaffold a stream processing loop compute unit.                                               |
| `/add-cli-command`          | [`add-cli-command.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/workflows/add-cli-command.md)                   | Add a CLI command to an existing module.                                                      |
| `/fix-bug`                  | [`fix-bug.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/workflows/fix-bug.md)                                   | Investigate, fix, and verify a bug with regression test.                                      |
| `/migrate-to-v2`            | [`migrate-to-v2.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/workflows/migrate-to-v2.md)                       | Convert V1 fpsexec code to V2 template layout.                                                |
| `/review-pr`                | [`review-pr.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/workflows/review-pr.md)                               | Review a PR for coding standards compliance.                                                  |
| `/setup-dev-environment`    | [`setup-dev-environment.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/workflows/setup-dev-environment.md)       | First-time development environment setup.                                                     |
| `/update-programmers-guide` | [`update-programmers-guide.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/workflows/update-programmers-guide.md) | Scan recent commits and refresh `docs/programmers_guide.md`.                                  |
| `/audit-code-quality`       | [`audit-code-quality.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/workflows/audit-code-quality.md)             | Audit code for readability, simplicity, duplication, file length, and removable dependencies. |
| `/audit-help-consistency`   | [`audit-help-consistency.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/workflows/audit-help-consistency.md)     | Cross-check all help sources for drift or contradictions.                                     |
| `/check-type-consistency`   | [`check-type-consistency.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/workflows/check-type-consistency.md)     | Audit `switch` blocks for incomplete type handling.                                           |
| `/cli-robustness-test`      | [`cli-robustness-test.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/workflows/cli-robustness-test.md)           | Run the CLI robustness test suite.                                                            |
| `/inspect-machine-code`     | [`inspect-machine-code.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/workflows/inspect-machine-code.md)         | Assembly inspection for performance optimization.                                             |
| `/sync-worktree`            | [`sync-worktree.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/workflows/sync-worktree.md)                       | Sync worktree to latest framework-dev.                                                        |
| `/update-docs-site`         | [`update-docs-site.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/workflows/update-docs-site.md)                 | Add or update MkDocs pages.                                                                   |
| `/update-scripts-docs`      | [`update-scripts-docs.md`](https://github.com/milk-org/milk/blob/framework-dev/.agents/workflows/update-scripts-docs.md)           | Sync `docs/scripts.md` after script changes.                                                  |

## Adding New Rules or Workflows

### Rules

Create a new `.md` file in `.agents/rules/` with
YAML frontmatter:

```yaml
---
description: Short description of what this rule does
---
```

Then write the trigger conditions and required
actions in markdown below the frontmatter.

### Workflows

Create a new `.md` file in `.agents/workflows/`
with YAML frontmatter:

```yaml
---
description: Short description of the workflow
---
```

Then write numbered steps below the frontmatter.
Use `// turbo-all` at the top of the steps section
if every command should auto-run without confirmation.

---

← [Documentation Index](index.md)
