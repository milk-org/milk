/**
 * @file    imresize.h
 * @brief   Resize 2D image.
 */

#ifndef IMAGE_BASIC_IMRESIZE_H
#define IMAGE_BASIC_IMRESIZE_H

errno_t imresize_addCLIcmd();

long basic_resizeim(
    const char *imname_in,
    const char *imname_out,
    long        xsizeout,
    long        ysizeout
);

#endif