# Module: COREMOD_memory

Memory management for images

## Source Files

| File                           | Description                                              |
| ------------------------------ | -------------------------------------------------------- |
| `clearall.c`                   | Clearall module                                          |
| `compute_image_memory.c`       | Compute image memory module                              |
| `compute_nb_image.c`           | Compute nb image module                                  |
| `compute_nb_variable.c`        | Compute nb variable module                               |
| `create_image.c`               | create images and streams                                |
| `create_variable.c`            | create variables                                         |
| `delete_image.c`               | delete image(s)                                          |
| `delete_sharedmem_image.c`     | delete shared memory image and files                     |
| `delete_variable.c`            | Delete variable module                                   |
| `fps_ID.c`                     | find fps ID(s) from name                                 |
| `fps_create.c`                 | create function parameter structure                      |
| `fps_list.c`                   | list function parameter structure                        |
| `im3D_to_stream2D.c`           | convert 3D image to 2D stream                            |
| `image_ID.c`                   | find image ID(s) from name                               |
| `image_checksize.c`            | check image size                                         |
| `image_complex.c`              | complex number conversion                                |
| `image_copy.c`                 | Image copy module                                        |
| `image_copy_shm.c`             | copy image to shared memory                              |
| `image_keyword.c`              | Image keyword module                                     |
| `image_keyword_addD.c`         | Image keyword addd module                                |
| `image_keyword_addL.c`         | Image keyword addl module                                |
| `image_keyword_addS.c`         | Image keyword adds module                                |
| `image_keyword_list.c`         | Image keyword list module                                |
| `image_make2D.c`               | Image make2d module                                      |
| `image_make3D.c`               | Image make3d module                                      |
| `image_mk_amph_from_complex.c` | complex -> amplitude, phase                              |
| `image_mk_complex_from_amph.c` | amplitude, phase -> complex                              |
| `image_mk_complex_from_reim.c` | real, imaginary -> complex                               |
| `image_mk_reim_from_complex.c` | complex -> re, im                                        |
| `image_set_counters.c`         | SET IMAGE FLAGS / COUNTERS                               |
| `list_image.c`                 | list images                                              |
| `list_variable.c`              | list variables                                           |
| `logshmim.c`                   | Save telemetry stream data                               |
| `read_shmim.c`                 | read shared memory stream                                |
| `read_shmim_size.c`            | read shared memory image size                            |
| `read_shmimall.c`              | read all shared memory stream                            |
| `saveall.c`                    | Saveall module                                           |
| `shmim_purge.c`                | purge shared memory stream                               |
| `shmim_setowner.c`             | set stream owner PID                                     |
| `stream_TCP.c`                 | TCP stream transfer                                      |
| `stream_UDP.c`                 | TCP stream transfer                                      |
| `stream_ave.c`                 | Average stream of images                                 |
| `stream_copy.c`                | copy image stream                                        |
| `stream_delay.c`               | delay input stream to output stream                      |
| `stream_diff.c`                | Stream diff module                                       |
| `stream_halfimdiff.c`          | difference between two halves of stream image            |
| `stream_merge.c`               | Merge n independently triggered streams                  |
| `stream_monitorlimits.c`       | Monitor stream values for safety limits                  |
| `stream_paste.c`               | Paste two equal size 2D streams into an output 2D stream |
| `stream_pixmapdecode.c`        | Stream pixmapdecode module                               |
| `stream_poke.c`                | poke image stream                                        |
| `stream_sem.c`                 | stream semaphores                                        |
| `stream_updateloop.c`          | Send single burst of frames to stream                    |
| `variable_ID.c`                | find variable ID(s) from name                            |

## Standalone Executables

| Executable                   | Source File    | Description              |
| ---------------------------- | -------------- | ------------------------ |
| `milk-fpsexec-mem-streamave` | `stream_ave.c` | Average stream of images |

## Dependencies

- `CLIcore` (includes transitive: `milkfps`, `ImageStreamIO`, `milkdata`)
