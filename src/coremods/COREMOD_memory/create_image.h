/**
 * @file create_image.h
 * @brief Create image module
 */

/**
 * @file    create_image.h
 */

#ifndef _CREATE_IMAGE_H_
#define _CREATE_IMAGE_H_

#include "ImageStreamIO/ImageStreamIO.h"
#include "libfps/IMGID.h"

errno_t create_image_ID(const char *name,
                        long        naxis,
                        uint32_t   *size,
                        uint8_t     datatype,
                        int         shared,
                        int         nbkw,
                        int         CBsize,
                        imageID    *outID);
errno_t create_image_ID_IMGID(IMGID *img);

errno_t create_1Dimage_ID(const char *ID_name, uint32_t xsize, imageID *outID);
errno_t create_1Dimage_ID_IMGID(IMGID *img);

errno_t create_1DCimage_ID(const char *ID_name, uint32_t xsize, imageID *outID);
errno_t create_1DCimage_ID_IMGID(IMGID *img);

errno_t create_2Dimage_ID(const char *ID_name,
                          uint32_t    xsize,
                          uint32_t    ysize,
                          imageID    *outID);
errno_t create_2Dimage_ID_IMGID(IMGID *img);

errno_t create_2Dimage_ID_double(const char *ID_name,
                                 uint32_t    xsize,
                                 uint32_t    ysize,
                                 imageID    *outID);
errno_t create_2Dimage_ID_double_IMGID(IMGID *img);

errno_t create_2DCimage_ID(const char *ID_name,
                           uint32_t    xsize,
                           uint32_t    ysize,
                           imageID    *outID);
errno_t create_2DCimage_ID_IMGID(IMGID *img);

errno_t create_2DCimage_ID_double(const char *ID_name,
                                  uint32_t    xsize,
                                  uint32_t    ysize,
                                  imageID    *outID);
errno_t create_2DCimage_ID_double_IMGID(IMGID *img);

errno_t create_3Dimage_ID(const char *ID_name,
                          uint32_t    xsize,
                          uint32_t    ysize,
                          uint32_t    zsize,
                          imageID    *outID);
errno_t create_3Dimage_ID_IMGID(IMGID *img);

errno_t create_3Dimage_ID_float(const char *ID_name,
                                uint32_t    xsize,
                                uint32_t    ysize,
                                uint32_t    zsize,
                                imageID    *outID);
errno_t create_3Dimage_ID_float_IMGID(IMGID *img);

errno_t create_3Dimage_ID_double(const char *ID_name,
                                 uint32_t    xsize,
                                 uint32_t    ysize,
                                 uint32_t    zsize,
                                 imageID    *outID);
errno_t create_3Dimage_ID_double_IMGID(IMGID *img);

errno_t create_3DCimage_ID(const char *ID_name,
                           uint32_t    xsize,
                           uint32_t    ysize,
                           uint32_t    zsize,
                           imageID    *outID);
errno_t create_3DCimage_ID_IMGID(IMGID *img);
#endif // _CREATE_IMAGE_H_
