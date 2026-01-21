#include <sys/file.h>
#include <sys/mman.h> // mmap()
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#include "processinfo_internal.h"
#include "processinfo.h"

PROCESSINFO *processinfo_shm_link(const char *pname, int *fd)
{
    size_t sharedsize = 0; // shared memory size in bytes
    int    SM_fd;          // shared memory file descriptor

    SM_fd = open(pname, O_RDWR);
    if(SM_fd == -1)
    {
        return (PROCESSINFO *) MAP_FAILED;
    }

    struct stat SM_stat;
    if (fstat(SM_fd, &SM_stat) == -1) {
        close(SM_fd);
        return (PROCESSINFO *) MAP_FAILED;
    }
    sharedsize = SM_stat.st_size;

    PROCESSINFO *pinfo = (PROCESSINFO *)
                             mmap(0, sharedsize, PROT_READ | PROT_WRITE, MAP_SHARED, SM_fd, 0);

    if(pinfo == MAP_FAILED)
    {
        close(SM_fd);
        return (PROCESSINFO *) MAP_FAILED;
    }

    *fd = SM_fd;

    return pinfo;
}