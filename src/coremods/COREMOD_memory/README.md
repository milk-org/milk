# Module: COREMOD_memory

Memory management for images

## Source Files

| File | Description |
|------|-------------|
| `clearall.c` | No description available. |
| `compute_image_memory.c` | No description available. |
| `compute_nb_image.c` | No description available. |
| `compute_nb_variable.c` | No description available. |
| `create_image.c` | create images and streams |
| `create_variable.c` | create variables |
| `delete_image.c` | delete image(s) |
| `delete_sharedmem_image.c` | delete shared memory image and files |
| `delete_variable.c` | No description available. |
| `fps_ID.c` | find fps ID(s) from name |
| `fps_create.c` | create function parameter structure |
| `fps_list.c` | list function parameter structure |
| `im3D_to_stream2D.c` | convert 3D image to 2D stream |
| `image_ID.c` | find image ID(s) from name |
| `image_checksize.c` | check image size |
| `image_complex.c` | complex number conversion |
| `image_copy.c` | No description available. |
| `image_copy_shm.c` | copy image to shared memory |
| `image_keyword.c` | No description available. |
| `image_keyword_addD.c` | No description available. |
| `image_keyword_addL.c` | No description available. |
| `image_keyword_addS.c` | No description available. |
| `image_keyword_list.c` | No description available. |
| `image_make2D.c` | No description available. |
| `image_make3D.c` | No description available. |
| `image_mk_amph_from_complex.c` | complex -> amplitude, phase |
| `image_mk_complex_from_amph.c` | amplitude, phase -> complex |
| `image_mk_complex_from_reim.c` | real, imaginary -> complex |
| `image_mk_reim_from_complex.c` | complex -> re, im |
| `image_set_counters.c` | SET IMAGE FLAGS / COUNTERS |
| `list_image.c` | list images |
| `list_variable.c` | list variables |
| `logshmim.c` | Save telemetry stream data |
| `read_shmim.c` | read shared memory stream |
| `read_shmim_size.c` | read shared memory image size |
| `read_shmimall.c` | read all shared memory stream |
| `saveall.c` | No description available. |
| `shmim_purge.c` | purge shared memory stream |
| `shmim_setowner.c` | set stream owner PID |
| `stream_TCP.c` | TCP stream transfer |
| `stream_UDP.c` | TCP stream transfer |
| `stream_ave.c` | Average stream of images |
| `stream_copy.c` | copy image stream |
| `stream_delay.c` | delay input stream to output stream |
| `stream_diff.c` | No description available. |
| `stream_halfimdiff.c` | difference between two halves of stream image |
| `stream_merge.c` | Merge n independently triggered streams |
| `stream_monitorlimits.c` | Monitor stream values for safety limits |
| `stream_paste.c` | Paste two equal size 2D streams into an output 2D stream |
| `stream_pixmapdecode.c` | No description available. |
| `stream_poke.c` | poke image stream |
| `stream_sem.c` | stream semaphores |
| `stream_updateloop.c` | Send single burst of frames to stream |
| `variable_ID.c` | find variable ID(s) from name |

## Standalone Executables

| Executable | Source File | Description |
|------------|-------------|-------------|
| `milk-fpsexec-mem-streamave` | `stream_ave.c` | Average stream of images |

## Dependencies
- Implicit standard: `milkdata`, `ImageStreamIO`, `CLIcore`
