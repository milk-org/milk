---
name: milk-script-writer
description: Generate correct milk-cli scripts from
  natural language prompts, using the scripting reference in docs/
---

# milk-script-writer

Write `.milk` scripts (bash plus the milk `@` syntax) for milk-cli, milk-script or `milk.bash`.

## Read First

- [docs/guide/scripting.md](../../../docs/guide/scripting.md) — the reference: calling commands,
  entities (`s.`, `l.`, `f.`, `p.`, `q.`), `@get`/`@set`, the calculator, waits and events, errors,
  cost model, examples. Every form there is tested.
- [docs/guide/slicing.md](../../../docs/guide/slicing.md) — image slices in calculator expressions.
- [docs/whatsnew.md](../../../docs/whatsnew.md) — migration table from the old syntax.

## Output Rules

Generated scripts MUST:

1. Start with `#!/usr/bin/env milk-script`.
2. Call milk commands as `module.command` (`mem.listim`) or `@name` (`@mk2Dim`, `@calc`); a bare
   `listim` is just a shell word.
3. Bind values with `-v` / inline `@…` rather than `$(…)` inside loops (see the cost model).
4. Quote calculator expressions, and string literals inside them (`@calc 'strlen("abc")'`).
5. Clean up with `@defer` and use `@L:` for scratch images.
6. Keep lines ≤ 100 characters and use descriptive variable names.
