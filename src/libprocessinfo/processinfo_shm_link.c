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

    sharedsize = sizeof(PROCESSINFO);

    SM_fd = open(pname, O_RDWR);
    if(SM_fd == -1)
    {
        perror("Error opening file for writing");
        exit(0);
    }

    PROCESSINFO *pinfolist = (PROCESSINFO *)
                             mmap(0, sharedsize, PROT_READ | PROT_WRITE, MAP_SHARED, SM_fd, 0);

    if(pinfolist == MAP_FAILED)
    {
        close(SM_fd);
        perror("Error mmapping the file");
        exit(0);
    }

    *fd = SM_fd;

    return pinfolist;
}