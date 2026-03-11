# Code Assist Tools

See also: [Programmer's Guide](programmers_guide.md) ·
[Coding Standards](developer/coding_standards.md) ·
[Adding Plugins](developer/plugins.md) ·
[Template Source Code](developer/TemplateSourceCode.md)

The `milk` project includes **agent rules** and
**workflows** that guide AI coding assistants
(Gemini, Copilot, etc.) to follow project
conventions automatically. They live under:

```
.agents/
├── rules/        # Always-on guardrails
└── workflows/    # On-demand task templates
```

Collaborators benefit even without using an AI
assistant — the rule and workflow files document
the conventions, checklists, and cross-references
that every contributor should know.

## Agent Rules

Rules fire **automatically** when a task touches
relevant code. They enforce conventions without
requiring you to remember every checklist.

| Rule | File | What it enforces |
|------|------|------------------|
| Architecture principles | `architecture-principles.md` | Minimize cross-module deps; consult `dependency_graph.md` before adding new ones. |
| CMake conventions | `cmake-conventions.md` | Use `PUBLIC`/`INTERFACE` properties; each module owns its headers. |
| Code style | `code-style-guide.md` | 80-char lines, Kernel-Doc, Linux kernel style, explicit includes. |
| Compile after edit | `compile-after-edit.md` | Always run `/compile-test` after modifying C/CMake. |
| Documentation standards | `documentation-standards.md` | Markdown formatting, shell prompts, link checking. |
| fpsexec conventions | `fpsexec-conventions.md` | V2 template, 8-section layout, `-h1` requirement. |
| Help consistency | `help-consistency.md` | Cross-check all sibling help sources when editing help content. |
| Programmer's Guide | `maintain-programmers-guide.md` | Update `docs/programmers_guide.md` on architectural changes. |
| Script documentation | `script-docs.md` | Update `docs/scripts.md` and add `--help` when scripts change. |
| README updates | `readme-update.md` | Update module README when source files are added/removed. |
| Workspace layout | `files-directories.md` | cacao lives at `plugins/cacao-src` → `~/src/cacao`. |

## Workflows

Workflows are invoked by typing the slash command
as a chat message (e.g., `/compile-test`). They are
step-by-step checklists for common tasks.

| Command | File | What it does |
|---------|------|--------------|
| `/compile-test` | `compile-test.md` | Incremental build from `_build/`, report errors. |
| `/create-fpsexec` | `create-fpsexec.md` | Scaffold a new V2 fpsexec standalone executable. |
| `/update-programmers-guide` | `update-programmers-guide.md` | Scan recent commits and refresh `docs/programmers_guide.md`. |
| `/audit-help-consistency` | `audit-help-consistency.md` | Cross-check all help sources for drift or contradictions. |
| `/add-new-module` | `add-new-module.md` | Scaffold a new plugin module (README, CMake, boilerplate). |
| `/update-scripts-docs` | `update-scripts-docs.md` | Sync `docs/scripts.md` after script changes. |
| `/check-type-consistency` | `check-type-consistency.md` | Audit `switch` blocks for incomplete type handling. |

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
