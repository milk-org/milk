/**
 * @file    cubecollapse.h
 * @brief   Collapse a cube along z axis.
 */

#ifndef IMAGE_BASIC_CUBECOLLAPSE_H
#define IMAGE_BASIC_CUBECOLLAPSE_H

errno_t __attribute__((cold)) CLIADDCMD_image_basic__cubecollapse();

imageID cube_collapse(const char *ID_in_name, const char *ID_out_name);

#endif
