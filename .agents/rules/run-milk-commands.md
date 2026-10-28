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

2. **Shared memory isolation** — streams and FPS live in `$MILK_SHM_DIR` (`/dev/shm` by default). To
   clean stale FPS instances or streams, use `milk-fps-rm -f NAME` or `milk-stream-rm NAME`.

3. **Units** — `runstart` starts a unit's loop in the background and returns once it runs. Use
   `@ls 'f.*'` (or `milk-fps-list`) to inspect states, `@get f.NAME.run.out` (or
   `milk-fps-log NAME`) for output, and `@set f.NAME.run=stop` (or `milk-fps-rm -f NAME`) to stop.

4. **Build directory** — always build from `_build/`, never in-source.
