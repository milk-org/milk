---
name: stream-modifier-guide
description: Where stream names, their '@X:' modifiers and
  slice syntax are defined and parsed (IMGID)
---

# Stream Modifier Guide

Use when debugging stream creation or connection, changing the modifier grammar, or working on
slices `im[a:b]`.

## Read First

- [docs/guide/streams.md](../../../docs/guide/streams.md) — stream naming and modifiers.
- [docs/guide/slicing.md](../../../docs/guide/slicing.md) — slice syntax in the calculator.
- [docs/guide/scripting.md](../../../docs/guide/scripting.md) — image arguments of module
  commands, entities (`s.`, `l.`).

## Code

- `src/engine/libfps/io/fps_streamname_parse.c` — the one parser: an optional `@XY:` prefix before
  the name (location `S` shared, `L` local, `F`; `E` must exist; `N` must be new; `T<type>` such as
  `@TF32:`) or an `s.`/`l.` namespace prefix. Returns the bare name without allocating.
- `src/engine/libfps/IMGID.h` — `IMGID`, the image handle (name, template, connection).
- `src/cli/libmilkscript/calc/cli_calc_slice.c` — slices in calculator expressions.
