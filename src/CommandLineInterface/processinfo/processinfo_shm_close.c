#include <sys/mman.h> // mmap()
#include <sys/stat.h>

#include "CLIcore.h"
#include <processtools.h>


int processinfo_shm_close(PROCESSINFO *pinfo, int fd)
{
    struct stat file_stat;
    fstat(fd, &file_stat);
    munmap(pinfo, file_stat.st_size);
    close(fd);
    return EXIT_SUCCESS;
}
