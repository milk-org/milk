---
description: Update AGENTS.md and code_assist.md when
  agent infrastructure files change.
---

# Agent Documentation Updates

When a task adds, removes, or renames a file in
`.agents/rules/`, `.agents/skills/`, or
`.agents/workflows/`, you MUST update the index
tables so they stay in sync.

## Trigger Conditions

1. A new rule, skill, or workflow file is created.
2. An existing file is renamed or deleted.
3. A file's description changes significantly.

## Required Actions

### 1. Update `AGENTS.md`

- **Section 6** (Agent Rules): update the rule
  table to include or remove the entry.
- **Section 7** (Skills): update the skills table.
- **Section 8** (Workflows): update the workflows
  table.

### 2. Update `docs/code_assist.md`

- Update the **Agent Rules** table.
- Update the **Skills** table.
- Update the **Workflows** table.

### 3. Verify consistency

After updating, verify that the number of entries
in each table matches the number of files:

```bash
ls .agents/rules/*.md | wc -l
ls .agents/skills/*/SKILL.md | wc -l
ls .agents/workflows/*.md | wc -l
```
