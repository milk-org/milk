/**
 * @file    image_merge3D.h
 *
 */

errno_t image_merge3D_addCLIcmd();

imageID arith_image_merge3D(const char *ID_name1,
                            const char *ID_name2,
                            const char *IDout_name);
