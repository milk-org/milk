---
render_macros: true
---

# Code Assist Tools

See also: [Programmer's Guide](../arch/programmers_guide.md) ·
[Coding Standards](coding_standards.md) · [Adding Plugins](plugins.md) ·
[Developer Tutorial](tutorial.md)

The `milk` project includes **agent rules**, **skills** and **workflows** that guide AI coding
assistants (Gemini, Copilot, etc.) to follow project conventions automatically. They live under
[`.agents/`]({{ github_blob_url }}/.agents):

```text
.agents/
├── rules/        # Always-on guardrails (one short file per topic)
├── skills/       # Deep-dive instruction sets, one folder with a SKILL.md each
└── workflows/    # On-demand task templates, invoked as /<file name>
```

Collaborators benefit even without an AI assistant: the files document the conventions, checklists
and cross-references that every contributor should know. The directories are the index — list them to
see what exists, and open the file for its trigger conditions and required actions.

## Getting Started

1. [Developer Tutorial](tutorial.md) — write your first module end-to-end.
2. [Adding Plugins](plugins.md) — module layout, CMake setup.
3. [Programmer's Guide](../arch/programmers_guide.md#3-writing-a-compute-unit) — which template file to
   copy for each use case.

## Adding New Rules or Workflows

Rules and workflows are markdown files with YAML frontmatter:

```yaml
---
description: Short description of what this rule / workflow does
---
```

A rule then describes its trigger conditions and required actions; a workflow lists numbered steps
(use `// turbo-all` at the top of the steps if every command should auto-run). Rules and workflows
point to the human documentation in `docs/` rather than duplicating it.

---

← [Documentation Index](../index.md)
