/**
 * @file    read_shmim.h
 */


#ifndef COREMOD_MEMORY_READ_SHMIM_H
#define COREMOD_MEMORY_READ_SHMIM_H

errno_t CLIADDCMD_COREMOD_memory__read_sharedmem_image();

imageID    read_sharedmem_image(
    const char *name
);

#endif
