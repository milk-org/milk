/**
 * @file read_shmim.h
 * @brief Read shmim module
 */

/**
 * @file    read_shmim.h
 */

#ifndef COREMOD_MEMORY_READ_SHMIM_H
#define COREMOD_MEMORY_READ_SHMIM_H

errno_t CLIADDCMD_COREMOD_memory__read_sharedmem_image();


imageID read_sharedmem_image(const char *sname, IMAGE *imagearray, long NB_images);

#endif
