# libmilkprop (`src/prop/`)

Shared-memory entity property and action library for the `milk` framework.

## Purpose

`libmilkprop` provides a single, CLI-independent implementation for reading, writing, and querying
live `milk` shared-memory entities across namespaces:

- **Streams & Local Images (`s.<name>.<prop>`, `l.<name>.<prop>`)**: metadata, pixel statistics
  (computed with COREMOD_arith's kernels, as in the calculator) and FITS keywords; the names are in
  `milk_stream_props.h`.
- **Function Parameter System (`f.<name>.<param>`)**: parameter values and unit lifecycle properties
  and actions.
- **Processinfo Telemetry (`p.<name>.<prop>`)**: process state, loop counters and controls.

The property names of each namespace are listed for users in
[Scripting — Entities](../../docs/guide/scripting.md#entities).

---

## Source Files

| File | Description |
| :--- | :--- |
| `milk_shm_prop.h` | Public API: get/set/exists of entity properties, `@snapshot`/`@diff`/`@restore`, stream lookup |
| `milk_stream_props.h` | The single table of stream property names (listings, completion, `@help streams`) |
| `milk_shm_prop_internal.h` | Internal per-namespace handler declarations (`stream`, `fps`, `proc`) and the helpers they share |
| `milk_entity_ns.h` | Canonical table of entity namespaces (`s`, `l`, `f`, `p`, and long aliases) |
| `milk_shm_cache.h` / `milk_shm_cache.c` | Inode-validated process-local cache of mapped SHM streams and FPS instances, and the reader semaphore slots of a stream |
| `milk_shm_prop.c` | Top-level `type.name.prop` dispatcher, glob discovery in the shared-memory directory, `--json` formatting, `@snapshot`/`@diff`/`@restore` |
| `milk_shm_prop_stream.c` | Stream/image metadata queries, statistical reductions, FITS keywords, and JSON output |
| `milk_shm_prop_fps.c` | FPS parameter reads/writes, unit state inspection and actions, and JSON serialization |
| `milk_shm_prop_proc.c` | `PROCESSINFO` telemetry queries, control actions, and JSON serialization |

---

## Dependencies & Build Tier

- **Build Tier**: Calculator & JIT tier (`USE_CALC=ON`; not `USE_CLI`).
- **Internal Dependencies**: `milkcomputeunit`, `milkfps`, `milkprocessinfo`, `milkdata`,
  `ImageStreamIO`, `milkcommon`.
