---
description: Guidance for running milk and cacao
  commands in the agent environment
---

# Running milk Commands

When running `milk-cli`, `milk-script`, `milk.bash`, or standalone executables (`milk-fpsexec-*`,
`cacao-fpsexec-*`), consult:

- [`docs/guide/scripting.md`](../../docs/guide/scripting.md) — running commands and `.milk` scripts,
  module arguments, entities (`@get`, `@set`), waiting (`@waitfor`, `@grab`), and exit codes
- [`docs/guide/calculator.md`](../../docs/guide/calculator.md) — `@calc` expressions and
  `milk-fpsexec-milk-calc` / `milk-fpsexec-milk-script` units
- [`docs/guide/sessions.md`](../../docs/guide/sessions.md) — `@session` lifecycle and
  `source milk.bash`

## Environment Setup

Source the install's `milk.bash` (or pass `MILK_INSTALLDIR` to binaries, which locate their own
install automatically):

```bash
source <install-dir>/bin/milk.bash
```

## Common Pitfalls

1. **Do not run `milk-cli` interactively** from the agent — it is a REPL and will block. Use
   `milk-cli -c 'commands'`, `milk-cli -s script.milk`, `milk-script script.milk`, or `source
   milk.bash` in `bash -c`:

   ```bash
   milk-cli -c 'mem.mk2Dim im 64 64; @calc -v m "imean(im)"'
   ```

2. **Stale streams and FPS instances** — remove them as
   [Debugging §4.2](../../docs/operations/debugging.md#42-shared-memory-leaks) shows.

3. **Units** — start, inspect and stop them, and read their output, as
   [FPS §3](../../docs/guide/fps.md#3-running-units) shows (from a script: the `f.` entity,
   [Scripting — Entities](../../docs/guide/scripting.md#entities)).

4. **Build directory** — always build from `_build/`, never in-source.
