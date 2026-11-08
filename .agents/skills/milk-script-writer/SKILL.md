---
name: milk-script-writer
description: Generate correct milk-cli scripts from
  natural language prompts, using the scripting reference in docs/
---

# milk-script-writer

Write `.milk` scripts (bash plus module commands, `@` keywords and `@{…}`) for `milk-cli`,
`milk-script`, or `milk.bash`.

## Read First

[docs/guide/scripting.md](../../../docs/guide/scripting.md): the language, and a table of the other
guide pages (calculator, sessions, slicing, C extensions) with what each covers.

## Output Rules

Generated scripts MUST:

1. Start with `#!/usr/bin/env milk-script` (or `source milk.bash` when embedded in plain bash).
2. Follow the rules and recommendations of:
   - [The Language at a Glance](../../../docs/guide/scripting.md#the-language-at-a-glance): what
     `@` marks;
   - [Arguments](../../../docs/guide/scripting.md#arguments): values, bash variables and image
     names in module-command arguments;
   - [Calculator](../../../docs/guide/calculator.md#calculator-calc-test-pipe): names, quoting,
     and getting values back into bash;
   - [Writing and Actions](../../../docs/guide/scripting.md#writing-and-actions) and
     [Waiting, Events and Cleanup](../../../docs/guide/scripting.md#waiting-events-and-cleanup):
     `@set --restore`, `@defer`, `@sleep`;
   - [Cost Model](../../../docs/guide/scripting.md#cost-model): the forms to prefer in loops;
   - [Migrating Old Scripts](../../../docs/guide/scripting.md#migrating-old-scripts): spellings
     that no longer work.
3. Keep lines ≤ 100 characters.
