---
name: milk-script-writer
description: Generate correct milk-cli scripts from
  natural language prompts, using the scripting reference in docs/
---

# milk-script-writer

Write `.milk` scripts (bash plus module commands, `@` keywords and `@{…}`) for milk-cli, milk-script
or `milk.bash`.

## Read First

- [docs/guide/scripting.md](../../../docs/guide/scripting.md) — the language reference: keywords,
  module commands and their arguments, entities (`s.`, `l.`, `f.`, `p.`, `q.`), the calculator and
  its variable rules, waits and events, exit status, cost model, migration, examples.
- [docs/guide/slicing.md](../../../docs/guide/slicing.md) — image slices in calculator expressions.

## Output Rules

Generated scripts MUST:

1. Start with `#!/usr/bin/env milk-script`.
2. Call module commands as `module.command` (`mem.listim`, `mem.mk2Dim`); `@` marks only the
   keywords (`@calc`, `@get`) and the `@{…}` expansion.
3. Get values into bash with `-v` (`@calc -v`, `@get -v`) or `@{…}`, never `$(…)` in loops; a
   calculator assignment does not set a bash variable.
4. Pass bash values to module commands as `$var`; quote calculator expressions and the string
   literals inside them (`@calc 'strlen("abc")'`).
5. Clean up with `@defer`, use `"@L:name"` for scratch images and `@sleep` for short waits.
6. Pass `milk-script --check` (no removed spelling), with lines ≤ 100 characters.
