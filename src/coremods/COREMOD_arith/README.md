# Module: COREMOD_arith

Arith functions on images

## Source Files

| File | Description |
|------|-------------|
| `execute_arith.c` | image arithmetic parser |
| `image_arith__Cim_Cim__Cim.c` | arith functions |
| `image_arith__im__im.c` | arith functions |
| `image_arith__im_f__im.c` | arith functions |
| `image_arith__im_f_f__im.c` | arith functions |
| `image_arith__im_im__im.c` | arith functions |
| `image_crop.c` | crop functions |
| `image_crop2D.c` | Crop a 2D rectangular region from stream |
| `image_cropmask.c` | Image cropmask module |
| `image_dxdy.c` | spatial derivatives |
| `image_merge3D.c` | Merge images along an axis |
| `image_multicrop2D.c` | Multi-window 2D cropping from stream |
| `image_norm.c` | Compute per-slice norm of an image |
| `image_pixremap.c` | Image pixremap module |
| `image_pixunmap.c` | Image pixunmap module |
| `image_set_1Dpixrange.c` | Set pixels in a 1D index range |
| `image_set_2Dpix.c` | Set a single pixel value in a 2D image |
| `image_set_3Daxes.c` | Set 3D image axes size |
| `image_set_col.c` | Set image column pixels to a value |
| `image_set_row.c` | Set image row pixels to a value |
| `image_setzero.c` | Set all image pixels to zero |
| `image_slicenormalize.c` | Image slicenormalize module |
| `image_stats.c` | simple stats functions |
| `image_total.c` | sum image pixels |
| `image_unfold.c` | Image unfold module |
| `image_vecmult.c` | multiply image by vector |
| `imfunctions.c` | apply math functions to images |
| `mathfuncs.c` | simple math functions |

## Standalone Executables

| Executable | Source File | Description |
|------------|-------------|-------------|
| `milk-fpsexec-arith-crop2D` | `image_crop2D.c` | Crop a 2D rectangular region from stream |
| `milk-fpsexec-arith-multicrop2D` | `image_multicrop2D.c` | Multi-window 2D cropping from stream |
| `milk-fpsexec-arith-immerge` | `image_merge3D.c` | Merge images along an axis |
| `milk-fpsexec-arith-setrow` | `image_set_row.c` | Set image row pixels to a value |
| `milk-fpsexec-arith-setcol` | `image_set_col.c` | Set image column pixels to a value |
| `milk-fpsexec-arith-setpix` | `image_set_2Dpix.c` | Set a single pixel value in a 2D image |
| `milk-fpsexec-arith-imsetzero` | `image_setzero.c` | Set all image pixels to zero |
| `milk-fpsexec-arith-setpix1Drange` | `image_set_1Dpixrange.c` | Set pixels in a 1D index range |
| `milk-fpsexec-arith-set3Daxes` | `image_set_3Daxes.c` | Set 3D image axes size |
| `milk-fpsexec-arith-normslice` | `image_norm.c` | Compute per-slice norm of an image |
| `milk-fpsexec-arith-cropmask` | `image_cropmask.c` | Image cropmask module |
| `milk-fpsexec-arith-unfold` | `image_unfold.c` | Image unfold module |
| `milk-fpsexec-arith-pixremap` | `image_pixremap.c` | Image pixremap module |
| `milk-fpsexec-arith-slicenormalize` | `image_slicenormalize.c` | Image slicenormalize module |
| `milk-fpsexec-arith-pixunmap` | `image_pixunmap.c` | Image pixunmap module |
| `milk-fpsexec-arith-vecmult` | `image_vecmult.c` | multiply image by vector |
| `milk-fpsexec-arith-imtrunc` | `image_arith__im_f_f__im.c` | arith functions |

## Dependencies
- `CLIcore` (includes transitive: `milkfps`, `ImageStreamIO`, `milkdata`)
