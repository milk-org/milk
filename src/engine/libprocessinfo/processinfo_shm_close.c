/**
 * @file processinfo_shm_close.c
 * @brief Processinfo shm close module
 */

#include <sys/mman.h>

#include "processinfo_internal.h"


/**
 * @brief Close and unmap a processinfo shared memory file.
 *
 * Releases the mapped region and closes the file
 * descriptor.
 */
int processinfo_shm_close(
    PROCESSINFO *pinfo,
    int fd)
{
    if(munmap(pinfo, sizeof(PROCESSINFO)) == -1)
    {
        PRINT_ERROR("Error un-mmapping the file: %s", strerror(errno));
    }
    close(fd);

    return 0;
}