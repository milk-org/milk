/**
 * @file    image_crop.h
 *
 */

errno_t image_crop_addCLIcmd();

imageID arith_image_crop(const char *ID_name,
                         const char *ID_out,
                         long       *start,
                         long       *end,
                         long        cropdim);

imageID arith_image_extract2D(const char *in_name,
                              const char *out_name,
                              long        size_x,
                              long        size_y,
                              long        xstart,
                              long        ystart);

imageID arith_image_extract3D(const char *in_name,
                              const char *out_name,
                              long        size_x,
                              long        size_y,
                              long        size_z,
                              long        xstart,
                              long        ystart,
                              long        zstart);
