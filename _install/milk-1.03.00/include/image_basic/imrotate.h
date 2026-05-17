/**
 * @file    imrotate.h
 * @brief   Rotate 2D image.
 */

#ifndef IMAGE_BASIC_IMROTATE_H
#define IMAGE_BASIC_IMROTATE_H

errno_t CLIADDCMD_image_basic__imrotate();

imageID basic_rotate(
    const char *ID_name,
    const char *IDout_name,
    float       angle
);

imageID basic_rotate90(
    const char *ID_name,
    const char *ID_out_name
);

imageID basic_rotate_int(
    const char *ID_name,
    const char *ID_out_name,
    long        nbstep
);

imageID basic_rotate2(
    const char *ID_name_in,
    const char *ID_name_out,
    float       angle
);

#endif