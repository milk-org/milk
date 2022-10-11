#include <sys/file.h>
#include <sys/mman.h> // mmap()

#include "CLIcore.h"
#include <processtools.h>

#include <sys/stat.h>


PROCESSINFO *processinfo_shm_link(const char *pname, int *fd)
{
    struct stat file_stat;

    *fd = open(pname, O_RDWR);
    if(*fd == -1)
    {
        perror("Error opening file");
        exit(0);
    }
    fstat(*fd, &file_stat);
    //printf("[%d] File %s size: %zd\n", __LINE__, pname, file_stat.st_size);

    PROCESSINFO *pinfolist = (PROCESSINFO *)
                             mmap(0, file_stat.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, *fd, 0);
    if(pinfolist == MAP_FAILED)
    {
        close(*fd);
        fprintf(stderr, "Error mmapping the file");
        exit(0);
    }

    return pinfolist;
}
