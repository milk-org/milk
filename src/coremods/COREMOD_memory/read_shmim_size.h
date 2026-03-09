/**
 * @file read_shmim_size.h
 * @brief Read shmim size module
 */

#ifndef COREMOD_MEMORY_READ_SHMIM_SIZE_H
#define COREMOD_MEMORY_READ_SHMIM_SIZE_H

errno_t CLIADDCMD_COREMOD_memory__read_sharedmem_image_size();

imageID read_sharedmem_image_size(const char *name, const char *fname);

#endif
