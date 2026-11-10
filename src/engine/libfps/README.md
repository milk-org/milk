# Module: libfps

FPS (Function Parameter Structure) core: the shared-memory struct, parameters, and their
persistence. No CLI dependency. The runtime that executes compute units lives in
[`libcomputeunit`](../libcomputeunit/README.md), which depends on this library, never the reverse.

## Source Files

Public headers (`fps.h`, `fps_*.h`, `imgid_slice.h`, `milk_help.h`) and `imgid_slice.c` stay at the
top; implementation sources are grouped into `core/` (lifecycle, connection, unit state), `params/`
(parameter entries, values, formatting, argument binding), `io/` (disk persistence, stream loading,
local store), `cmd/` (directory scanning), `tools/` (standalone CLI utilities), and `scripts/`.

| File                                        | Description                                                                                                       |
| :------------------------------------------ | :---------------------------------------------------------------------------------------------------------------- |
| `core/fps_connect.c`                        | Connect to an existing FPS in shared memory                                                                       |
| `core/fps_connectExternalFPS.c`             | Connect to an external FPS                                                                                        |
| `core/fps_disconnect.c`                     | Disconnect from an FPS                                                                                            |
| `core/fps_globals.c`                        | Global FPS array and state storage                                                                                |
| `core/fps_ID.c`                             | Find FPS ID(s) from name                                                                                          |
| `core/fps_isvalid.c`                        | Check if an FPS structure is valid                                                                                |
| `core/fps_shmdirname.c`                     | Resolve and create the FPS shared-memory directory name                                                           |
| `core/fps_struct_create.c`                  | Create a Function Parameter Structure in shared memory                                                            |
| `core/fps_unit_state.c`                     | Unit directory (`*.fps.d`), identity records, unit lock, and `fps_unit_status()`                                  |
| `params/fps_add_entry.c`                    | Add a parameter entry to an FPS                                                                                   |
| `params/fps_args.c`                         | Argument values (calculator literals) and binding of argument words (`module` commands, unit CLI, `@args`, `@set`)|
| `params/fps_checkparameter.c`               | Validate FPS parameter entries against constraints                                                                |
| `params/fps_FPSremove.c`                    | Remove an FPS and its shared-memory backing file                                                                  |
| `params/fps_GetFileName.c`                  | Resolve filename parameter entry                                                                                  |
| `params/fps_GetParamIndex.c`                | Look up parameter index by keyword                                                                                |
| `params/fps_GetTypeString.c`                | Format FPS parameter type as string                                                                               |
| `params/fps_paramvalue.c`                   | Get and set typed FPS parameter values                                                                            |
| `params/fps_print_info.c`                   | Print summary and detailed content of an FPS                                                                      |
| `params/fps_printlist.c`                    | Print table of loaded FPS instances and parameters                                                                |
| `params/fps_PrintParameterInfo.c`           | Print single FPS parameter status and value                                                                       |
| `params/fps_printparameter_valuestring.c`   | Format FPS parameter value into a string buffer                                                                   |
| `params/fps_SetParamCLIindex.c`             | Bind parameter entry to its CLI argument index                                                                    |
| `params/fps_userinputsetparamvalue.c`       | Parse user string input and update parameter value                                                                |
| `io/fps_load.c`                             | Load an FPS from shared memory                                                                                    |
| `io/fps_loadmemstream_lite.c`               | Lightweight shared-memory stream loader for `libfps`                                                              |
| `io/fps_loadstream.c`                       | Load image stream with `@X:` modifier prefix support                                                              |
| `io/fps_local_store.c`                      | In-process (local) FPS instance table management                                                                  |
| `io/fps_outlog.c`                           | Resolve FPS output log filename                                                                                   |
| `io/fps_save2disk.c`                        | Export FPS parameter tree and scripts to disk                                                                     |
| `io/fps_streamname_parse.c`                 | Parse `@X:` modifier prefixes from stream names; modifier list for completion and help                            |
| `io/fps_WriteParameterToDisk.c`             | Write a single FPS parameter value to disk                                                                        |
| `cmd/fps_scan.c`                            | Scan shared-memory directory and load active FPS structures                                                       |
| `imgid_slice.c`                             | `IMGID` slice syntax parsing                                                                                      |
| `tools/milk-fps-list.c`                     | `milk-fps-list`: list active FPS instances                                                                        |
| `tools/milk-fps-search.c`                   | `milk-fps-search`: search FPS parameters by pattern                                                               |
| `tools/milk-fps-info.c`                     | `milk-fps-info`: inspect FPS parameters, state, and stream connections                                            |
| `tools/milk-fps-set.c`                      | `milk-fps-set`: set or query an FPS parameter from the command line                                               |
| `tools/milk-fps-track.c`                    | `milk-fps-track`: follow FPS parameter updates in real time                                                       |
| `tools/milk-fps-help.c`                     | `milk-fps-help`: display FPS command-line tool reference                                                          |
| `tools/milk-fpsexec-list.c`                 | `milk-fpsexec-list`: list installed `fpsexec` compute-unit binaries                                               |
| `tools/milk-fpsexec-help.c`                 | `milk-fpsexec-help`: display `fpsexec` usage guide                                                                |
| `scripts/milk-latency-audit`                | Audit real-time loop latency across active FPS/processinfo units                                                  |

## Tools

`milk-fps-list`, `milk-fps-search`, `milk-fps-info`, `milk-fps-set`, `milk-fps-track`,
`milk-fps-help`, `milk-fpsexec-list`, `milk-fpsexec-help`, `milk-latency-audit` (`milk-fps-rm`,
`milk-fps-log`, and `milk-fps-{conf,run}{start,stop,step}` live in
[`libcomputeunit`](../libcomputeunit/README.md)).

## Dependencies

- `ImageStreamIO`, `milkprocessinfo`, `milkcommon`
