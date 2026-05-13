/**
 * @file image_crop3D.h
 * @brief 2D and 3D image crop functions
 */

errno_t
CLIADDCMD_COREMOD_arith__image_crop();

imageID arith_image_crop(
    const char *ID_name,
    const char *ID_out,
    int64_t *start,
    int64_t *end,
    int64_t cropdim);

imageID arith_image_extract2D(
    const char *in_name,
    const char *out_name,
    int64_t size_x,
    int64_t size_y,
    int64_t xstart,
    int64_t ystart);

imageID arith_image_extract3D(
    const char *in_name,
    const char *out_name,
    int64_t size_x,
    int64_t size_y,
    int64_t size_z,
    int64_t xstart,
    int64_t ystart,
    int64_t zstart);
