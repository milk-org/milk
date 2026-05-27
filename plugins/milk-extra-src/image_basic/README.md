# Module: image_basic

Frequently used image functions

## Source Files

| File                         | Description                                     |
| ---------------------------- | ----------------------------------------------- |
| `cubecollapse.c`             | Collapse a 3D cube along z axis                 |
| `extrapolate_nearestpixel.c` | Extrapolate missing pixels via nearest-neighbor |
| `im3Dto2D.c`                 | Convert 3D cube to tiled 2D image               |
| `image_add.c`                | Add (co-add) images with optional weighting     |
| `imcontract.c`               | Bin-down (contract) image by integer factor     |
| `imexpand.c`                 | Expand image by integer factor                  |
| `imgetcircasym.c`            | Extract circular asymmetry component            |
| `imgetcircsym.c`             | Extract circular symmetric component            |
| `imresize.c`                 | Resize 2D image (interpolation)                 |
| `imrotate.c`                 | Rotate 2D image                                 |
| `imstretch.c`                | Stretch image by scaling factor                 |
| `imswapaxis2D.c`             | Swap X and Y axes of 2D image                   |
| `indexmap.c`                 | Create index-based remapping of pixels          |
| `loadfitsimgcube.c`          | Load FITS files into image cube                 |
| `measure_transl.c`           | Measure translation (shift) between images      |
| `naninf2zero.c`              | Replace NaN/Inf values with zero                |
| `streamfeed.c`               | Feed FITS data into shared memory stream        |
| `streamrecord.c`             | Record stream frames to FITS files              |
| `tableto2Dim.c`              | Convert table data to 2D image                  |

## Standalone Executables

| Executable                           | Source File      | Description                  |
| ------------------------------------ | ---------------- | ---------------------------- |
| `milk-fpsexec-imgbasic-cubecollapse` | `cubecollapse.c` | Collapse a cube along z axis |
| `milk-fpsexec-imgbasic-rotateim`     | `imrotate.c`     | Rotate 2D image              |
| `milk-fpsexec-imgbasic-resizeim`     | `imresize.c`     | Resize 2D image              |

## Dependencies

- `CLIcore` (includes transitive: `milkfps`, `ImageStreamIO`, `milkdata`)
