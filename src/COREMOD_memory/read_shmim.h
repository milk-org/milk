/**
 * @file    read_shmim.h
 */


#ifndef _READ_SHMIM_H
#define _READ_SHMIM_H


imageID    read_sharedmem_image_size(
    const char *name,
    const char *fname
);

imageID    read_sharedmem_image(
    const char *name
);

#endif
