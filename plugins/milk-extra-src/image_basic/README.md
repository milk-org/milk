# Module: image_basic

Frequently used image functions

## Source Files

| File | Description |
|------|-------------|
| `cubecollapse.c` | Collapse a cube along z axis |
| `extrapolate_nearestpixel.c` | Extrapolate nearestpixel module |
| `im3Dto2D.c` | Im3dto2d module |
| `image_add.c` | Image add module |
| `imcontract.c` | Imcontract module |
| `imexpand.c` | Imexpand module |
| `imgetcircasym.c` | Imgetcircasym module |
| `imgetcircsym.c` | Imgetcircsym module |
| `imresize.c` | Resize 2D image |
| `imrotate.c` | Rotate 2D image |
| `imstretch.c` | Imstretch module |
| `imswapaxis2D.c` | Imswapaxis2d module |
| `indexmap.c` | Indexmap module |
| `loadfitsimgcube.c` | Loadfitsimgcube module |
| `measure_transl.c` | Measure transl module |
| `naninf2zero.c` | Naninf2zero module |
| `streamfeed.c` | Streamfeed module |
| `streamrecord.c` | Streamrecord module |
| `tableto2Dim.c` | Tableto2dim module |

## Standalone Executables

| Executable | Source File | Description |
|------------|-------------|-------------|
| `milk-fpsexec-imgbasic-cubecollapse` | `cubecollapse.c` | Collapse a cube along z axis |
| `milk-fpsexec-imgbasic-rotateim` | `imrotate.c` | Rotate 2D image |
| `milk-fpsexec-imgbasic-resizeim` | `imresize.c` | Resize 2D image |

## Dependencies
- Implicit standard: `milkdata`, `ImageStreamIO`, `CLIcore`
