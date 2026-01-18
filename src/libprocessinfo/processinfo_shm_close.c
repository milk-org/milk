#include <sys/mman.h>
#include <unistd.h>
#include <stdio.h>

#include "processinfo_internal.h"
#include "processinfo.h"


int processinfo_shm_close(PROCESSINFO *pinfo, int fd)
{
    if(munmap(pinfo, sizeof(PROCESSINFO)) == -1)
    {
        perror("Error un-mmapping the file");
    }
    close(fd);

    return 0;
}