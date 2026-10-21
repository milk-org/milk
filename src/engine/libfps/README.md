# Module: libfps

FPS (Function Parameter Structure) core: the shared-memory struct, parameters, and their
persistence. No CLI dependency. The runtime that executes compute units lives in
[`libcomputeunit`](../libcomputeunit/README.md), which depends on this library, never the reverse.

## Source Files

| File                               | Description                                           |
| ---------------------------------- | ----------------------------------------------------- |
| `fps_FPSremove.c`                  | remove FPS                                            |
| `fps_GetFileName.c`                | get FPS filename for entry                            |
| `fps_GetParamIndex.c`              | Get index of parameter                                |
| `fps_GetTypeString.c`              | Fps gettypestring module                              |
| `fps_ID.c`                         | find fps ID(s) from name                              |
| `fps_PrintParameterInfo.c`         | print FPS parameter status/values                     |
| `fps_SetParamCLIindex.c`           | set parameter CLI index                               |
| `fps_WriteParameterToDisk.c`       | Write parameter to disk                               |
| `fps_add_entry.c`                  | add parameter entry to FPS                            |
| `fps_checkparameter.c`             | check FPS entries                                     |
| `fps_connect.c`                    | connect to FPS                                        |
| `fps_connectExternalFPS.c`         | connect to external FPS                               |
| `fps_disconnect.c`                 | Disconnect from FPS                                   |
| `fps_globals.c`                    | Fps globals module                                    |
| `fps_isvalid.c`                    | Check if FPS is valid                                 |
| `fps_load.c`                       | Load FPS                                              |
| `fps_loadmemstream_lite.c`         | Lite version of load memory stream for libfps.        |
| `fps_loadstream.c`                 | Load image stream with @X: prefix support             |
| `fps_local_store.c`                | In-process (local) FPS instance management            |
| `fps_outlog.c`                     | Get FPS log filename                                  |
| `fps_paramvalue.c`                 | set and get parameter values                          |
| `fps_print_info.c`                 | Print content of a Function Parameter Structure (FPS) |
| `fps_printlist.c`                  | print list of parameters                              |
| `fps_printparameter_valuestring.c` | print parameter value string                          |
| `fps_save2disk.c`                  | Save FPS content to disk                              |
| `fps_scan.c`                       | scan and load FPSs                                    |
| `fps_shmdirname.c`                 | create FPS shared memory directory name               |
| `fps_streamname_parse.c`           | Parse @X: modifier prefixes from stream names         |
| `fps_struct_create.c`              | create function parameter structure                   |
| `fps_userinputsetparamvalue.c`     | read user input to set parameter value                |
| `imgid_slice.c`                    | IMGID slice parsing                                   |

## Tools

`milk-fps-list`, `milk-fps-search`, `milk-fps-info`, `milk-fps-rm`, `milk-fps-set`,
`milk-fps-track`, `milk-fps-help`, `milk-fpsexec-list`, `milk-fpsexec-help`.

## Dependencies

- `ImageStreamIO`, `milkprocessinfo`, `milkcommon`
