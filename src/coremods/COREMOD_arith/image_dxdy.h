/**
 * @file image_dxdy.h
 * @brief Image dxdy module
 */

/**
 * @file    image_dxdy.c
 */

#include <libfps/IMGID.h>

imageID arith_image_dx(
    const char *ID_name,
    const char *IDout_name);
imageID arith_image_dx_IMGID(
    IMGID *imgin,
    IMGID *imgout);

imageID arith_image_dy(
    const char *ID_name,
    const char *IDout_name);
imageID arith_image_dy_IMGID(
    IMGID *imgin,
    IMGID *imgout);
