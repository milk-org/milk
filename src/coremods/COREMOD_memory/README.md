# Module: COREMOD_memory

Memory management for images

## Source Files

| File                           | Description                                              |
| ------------------------------ | -------------------------------------------------------- |
| `COREMOD_memory.c`             | Module registry (`MILK_MODULE_RUNNABLES`)                |
| `clearall.c`                   | Clear all local images and variables                     |
| `compute_image_memory.c`       | Compute total image memory usage                         |
| `compute_nb_image.c`           | Count active images                                      |
| `compute_nb_variable.c`        | Count active variables                                   |
| `create_image.c`               | Create images and shared-memory streams                  |
| `create_variable.c`            | Create variables                                         |
| `delete_image.c`               | Delete image(s)                                          |
| `delete_sharedmem_image.c`     | Delete shared-memory image and backing files             |
| `delete_variable.c`            | Delete variable(s)                                       |
| `fps_ID.c`                     | Find FPS ID(s) from name                                 |
| `fps_list.c`                   | List Function Parameter Structures                       |
| `im3D_to_stream2D.c`           | Replay 3D image cube to 2D stream                        |
| `image_ID.c`                   | Find image ID(s) from name                               |
| `image_checksize.c`            | Check image dimensions and type                          |
| `image_complex.c`              | Complex number conversions                               |
| `image_copy.c`                 | Copy images                                              |
| `image_copy_shm.c`             | Copy image to shared memory                              |
| `image_keyword.c`              | Read/write FITS-style image keywords                     |
| `image_keyword_add.c`          | Add or update a typed keyword in an image                |
| `image_keyword_list.c`         | List image keywords                                      |
| `image_make2D.c`               | Create 2D test/shape images                              |
| `image_make3D.c`               | Create 3D images                                         |
| `image_mk_amph_from_complex.c` | Complex to amplitude and phase                           |
| `image_mk_complex_from_amph.c` | Amplitude and phase to complex                           |
| `image_mk_complex_from_reim.c` | Real and imaginary to complex                            |
| `image_mk_reim_from_complex.c` | Complex to real and imaginary                            |
| `image_set_counters.c`         | Set image flags and counters                             |
| `list_image.c`                 | List images in memory                                    |
| `list_variable.c`              | List variables in memory                                 |
| `logshmim.c`, `logshmim_save.c`| Log shared-memory telemetry stream cubes to disk         |
| `read_shmim.c`                 | Map shared-memory stream                                 |
| `read_shmim_size.c`            | Read shared-memory stream dimensions                     |
| `read_shmimall.c`              | Map all shared-memory streams                            |
| `saveall.c`                    | Save all images in memory to FITS                        |
| `shmim_purge.c`                | Purge orphan shared-memory streams                       |
| `shmim_setowner.c`             | Set stream owner PID                                     |
| `stream_TCP.c`, `stream_TCP_receive.c` | TCP stream transmit and receive                  |
| `stream_UDP.c`, `stream_UDP_receive.c` | UDP stream transmit and receive                  |
| `stream_ave.c`                 | Average stream of images                                 |
| `stream_copy.c`                | Copy input stream to output stream                       |
| `stream_delay.c`               | Stream delay, fractional resampling, and temporal FIR/window filtering |
| `stream_diff.c`                | Consecutive frame difference stream                      |
| `stream_halfimdiff.c`          | Difference between two halves of a stream image          |
| `stream_merge.c`               | Merge n independently triggered streams                  |
| `stream_monitorlimits.c`       | Monitor stream values for safety limits                  |
| `stream_paste.c`               | Paste two equal-size 2D streams into an output 2D stream |
| `stream_pixmapdecode.c`        | Decode detector pixel map stream                         |
| `stream_poke.c`                | Trigger stream semaphores without modifying pixel data   |
| `stream_sem.c`                 | Stream semaphore operations                              |
| `stream_slice.c`               | Extract slice views from streams                         |
| `stream_updateloop.c`          | Send burst of frames to a stream                         |
| `tlm_timing.c`                 | Fast fixed-point reader/writer for telemetry timing files |
| `trigmerge.c`                  | Merge up to 16 input stream triggers (any-of / all-of) into one trigger stream |
| `variable_ID.c`                | Find variable ID(s) from name                            |

## Standalone Executables

| Executable                       | Source File                  | Description                                                             |
| -------------------------------- | ---------------------------- | ----------------------------------------------------------------------- |
| `milk-fpsexec-mem-streamFITSlog` | `logshmim.c`                 | Log shared-memory stream frames to FITS cubes                           |
| `milk-fpsexec-mem-streamave`     | `stream_ave.c`               | Average stream of images                                                |
| `milk-fpsexec-mem-streampoke`    | `stream_poke.c`              | Update stream counters/semaphores without changing pixels               |
| `milk-fpsexec-mem-streamcopy`    | `stream_copy.c`              | Copy input stream to existing output stream                             |
| `milk-fpsexec-mem-streamdelay`   | `stream_delay.c`             | Delay and temporally filter/resample a stream in time or frame units    |
| `milk-fpsexec-mem-trigmerge`     | `trigmerge.c`                | Merge multiple stream triggers into a single trigger stream             |
| `milk-fpsexec-imkwadd`           | `image_keyword_add.c`        | Add or update a keyword in an image                                     |
| `milk-fpsexec-cnt2push`          | `bin/cnt2push_main.c`        | Update stream `cnt2` flow-control counter                               |
| `milk-stream-list`               | `bin/milk-stream-list.c`     | List active shared-memory streams                                       |
| `milk-stream-create`             | `bin/milk-stream-create.c`   | Create a shared-memory stream                                           |
| `milk-stream-rm`                 | `bin/milk-stream-rm.c`       | Remove shared-memory streams (`-r` regex, `--ignore-missing`)           |
| `milk-stream-link`               | `bin/milk-stream-link.c`     | Create or atomically swap `.im.shm` symlinks (`-s`, `<src>=<link>`, `-b`) |
| `milk-stream-wait`               | `bin/milk-stream-wait.c`     | Wait via `inotify` and semaphores for stream geometry and `+N` frames   |
| `milk-shmimpurge`                | `bin/milk-shmimpurge.c`      | Remove orphan shared-memory streams                                     |
| `milk-logshim-ctrl`              | `bin/milk-logshim-ctrl.c`    | Control shared-memory FITS logger units                                 |
| `milk-streamCTRL-cli`            | `bin/milk-streamCTRL-cli.c`  | Non-interactive CLI stream table dump                                   |

## Dependencies

- `milk_runnable.h` (`libcomputeunit`), which provides the FPS and processinfo macros
