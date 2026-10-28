---
name: milk-script-writer
description: Generate correct milk-cli scripts from
  natural language prompts, using the scripting reference in docs/
---

# milk-script-writer

Write `.milk` scripts (bash plus module commands, `@` keywords and `@{…}`) for `milk-cli`,
`milk-script`, or `milk.bash`.

## Read First

- [docs/guide/scripting.md](../../../docs/guide/scripting.md) — top-level language summary:
  keywords, module commands and their arguments (`@args`), entities (`s.`, `l.`, `f.`, `p.`),
  waiting/events (`@waitfor`, `@waitany`, `@grab`, `@replay`, `@on`, `@sleep`, `@defer`), exit
  status, cost summary, migration, and complete script examples.
- [docs/guide/calculator.md](../../../docs/guide/calculator.md) — `@calc` / `@test` / `@pipe`
  expressions, operator & function tables (math, reductions, stencils, spatial, BLAS/LAPACK),
  `milk-fpsexec-milk-calc` / `milk-fpsexec-milk-script` compute units, and SIMD/OpenMP fusion rules.
- [docs/guide/sessions.md](../../../docs/guide/sessions.md) — `@session` lifecycle (`new`, `use`,
  `leave`, `ls`, `end`, `save`/`restore`), `source milk.bash`, multi-shell concurrency, and
  `@snapshot`/`@diff`/`@restore`/`@log`.
- [docs/guide/slicing.md](../../../docs/guide/slicing.md) — image and stream slices (`im[x0:x1, :]`)
  in calculator expressions and stream modifiers.
- [docs/guide/calc_c_extensions.md](../../../docs/guide/calc_c_extensions.md) — extending `@calc`
  and `milk.calc` with compiled C kernels (`-l file.c`) and `milk-cc`.

## Output Rules

Generated scripts MUST:

1. Start with `#!/usr/bin/env milk-script` (or `source milk.bash` when embedded in plain bash).
2. Call module commands as `module.command` (`mem.listim`, `mem.mk2Dim`); `@` marks only the
   keywords (`@calc`, `@get`, `@set`, `@waitfor`, ...) and the `@{…}` expansion.
3. Get values into bash with `-v` (`@calc -v`, `@get -v`) or `NAME=@{…}`, never `$(…)` in loops; a
   calculator assignment (`@calc "x = 5"`) does not set a bash variable.
4. Pass bash values to module commands as `$var`; quote calculator expressions and the string
   literals inside them (`@calc 'strlen("abc")'`).
5. Clean up with `@defer` (and `@set --restore`), use `"@L:name"` for session-local scratch images,
   and use `@sleep` for short or paced waits (`@sleep --hz`).
6. Keep lines ≤ 100 characters and avoid removed legacy spellings (see `Migrating Old Scripts` in
   [docs/guide/scripting.md](../../../docs/guide/scripting.md#migrating-old-scripts)).
