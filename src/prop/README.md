# libmilkprop (`src/prop/`)

Shared-memory entity property and action library for the `milk` framework.

## Purpose

`libmilkprop` provides a single, CLI-independent implementation for reading, writing, and querying
live `milk` shared-memory entities across namespaces:

- **Streams & Local Images (`s.<name>.<prop>`, `l.<name>.<prop>`)**: dimensions (`xsize`, `ysize`,
  `zsize`, `naxis`, `nelement`), datatypes (`type`, `dtype`), frame counters (`cnt0`, `cnt1`),
  semaphores (`sem`), process ownership (`pid`, `ownerPID`), pixel reductions (`mean`, `rms`,
  `min`, `max`, `sum`, `median`, `p01`..`p99`), and FITS keywords (`s.<name>.kw.<KEY>`).
- **Function Parameter System (`f.<name>.<param>`)**: parameter values, parameter wildcards (`*`),
  and unit lifecycle properties/actions (`status`, `run`, `conf`, `ok`, `errors`, `msg`).
- **Processinfo Telemetry (`p.<name>.<prop>`)**: process status, loop counters (`loopcnt`, `hz`,
  `dt`, `dtmax`, `jitter`, `exectime`), trigger diagnostics (`trigmode`, `missedframes`,
  `timeouts`), and control actions (`ctrlval`).

---

## Source Files

| File | Description |
| :--- | :--- |
| `milk_shm_prop.h` | Public API (`milk_shm_prop_get`, `milk_shm_prop_get_num`, `milk_shm_prop_set`, `milk_shm_prop_exists`) |
| `milk_shm_prop_internal.h` | Internal per-namespace handler declarations (`stream`, `fps`, `proc`) |
| `milk_entity_ns.h` | Canonical table of entity namespaces (`s`, `l`, `f`, `p`, and long aliases) |
| `milk_shm_cache.h` / `milk_shm_cache.c` | Inode-validated process-local cache of mapped SHM streams and FPS instances |
| `milk_shm_prop.c` | Top-level `type.name.prop` dispatcher, `/dev/shm` glob discovery, and `--json` formatting |
| `milk_shm_prop_stream.c` | Stream/image metadata queries, statistical reductions, FITS keywords, and JSON output |
| `milk_shm_prop_fps.c` | FPS parameter reads/writes, unit state inspection, and JSON serialization |
| `milk_shm_prop_proc.c` | `PROCESSINFO` telemetry queries, control actions, and JSON serialization |

---

## Dependencies & Build Tier

- **Build Tier**: Core tier (`USE_COREMODS=ON`; does not require `USE_CLI`).
- **Internal Dependencies**: `milkcomputeunit`, `milkfps`, `milkprocessinfo`, `milkdata`,
  `ImageStreamIO`, `milkcommon`.
