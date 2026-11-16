# Module: libfps

Standalone applications using `libfps` typically support the following commands:

## Source Files

| File | Description |
|------|-------------|
| `fps_CONFstart.c` | FPS conf process start |
| `fps_CONFstop.c` | FPS conf process stop |
| `fps_FPCONFexit.c` | Exit FPS conf process |
| `fps_FPCONFloopstep.c` | FPS conf process loop step |
| `fps_FPCONFsetup.c` | FPS config setup |
| `fps_FPSremove.c` | remove FPS |
| `fps_GetFileName.c` | get FPS filename for entry |
| `fps_GetParamIndex.c` | Get index of parameter |
| `fps_GetTypeString.c` | Fps gettypestring module |
| `fps_ID.c` | find fps ID(s) from name |
| `fps_PrintParameterInfo.c` | print FPS parameter status/values |
| `fps_RUNexit.c` | Exit FPS run process |
| `fps_RUNstart.c` | FPS run process start |
| `fps_RUNstop.c` | FPS run process stop |
| `fps_SetParamCLIindex.c` | set parameter CLI index |
| `fps_WriteParameterToDisk.c` | Write parameter to disk |
| `fps_add_entry.c` | add parameter entry to FPS |
| `fps_checkparameter.c` | check FPS entries |
| `fps_cli_function_registry.c` | Global function pointer definitions |
| `fps_cli_init.c` | Initialize FPS entries from a bindings array |
| `fps_connect.c` | connect to FPS |
| `fps_connectExternalFPS.c` | connect to external FPS |
| `fps_disconnect.c` | Disconnect from FPS |
| `fps_execFPScmd.c` | Execute FPS command |
| `fps_getFPSargs.c` | read FPS args from CLI |
| `fps_globals.c` | Fps globals module |
| `fps_isvalid.c` | Check if FPS is valid |
| `fps_load.c` | Load FPS |
| `fps_loadmemstream_lite.c` | Lite version of load memory stream for libfps. |
| `fps_loadstream.c` | Load image stream with @X: prefix support |
| `fps_local_store.c` | In-process (local) FPS instance management |
| `fps_outlog.c` | Get FPS log filename |
| `fps_paramvalue.c` | set and get parameter values |
| `fps_print_info.c` | Print content of a Function Parameter Structure (FPS) |
| `fps_printlist.c` | print list of parameters |
| `fps_printparameter_valuestring.c` | print parameter value string |
| `fps_process_fpsCMDarray.c` | Find the next task to execute |
| `fps_processcmdline.c` | FPS process command line |
| `fps_processinfo.c` | ProcessInfo integration helpers for FPS |
| `fps_processinfo_entries.c` | Add parameters to FPS for real-time process settings |
| `fps_read_fpsCMD_fifo.c` | fill up task list from fifo submissions |
| `fps_save2disk.c` | Save FPS content to disk |
| `fps_scan.c` | scan and load FPSs |
| `fps_shmdirname.c` | create FPS shared memory directory name |
| `fps_streamname_parse.c` | Parse @X: modifier prefixes from stream names |
| `fps_struct_create.c` | create function parameter structure |
| `fps_tmux.c` | tmux session management |
| `fps_userinputsetparamvalue.c` | read user input to set parameter value |

## Dependencies
- `ImageStreamIO`, `milkprocessinfo`
