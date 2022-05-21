
#include <sys/file.h>
#include <sys/mman.h> // mmap()
#include <sys/stat.h>

#include "CLIcore.h"
#include <processtools.h>

#include "processinfo_procdirname.h"

#define FILEMODE 0666

extern PROCESSINFOLIST *pinfolist;




long processinfo_shm_list_create()
{
    char SM_fname[STRINGMAXLEN_FULLFILENAME];
    long pindex = 0;

    char procdname[STRINGMAXLEN_DIRNAME];
    processinfo_procdirname(procdname);

    WRITE_FULLFILENAME(SM_fname, "%s/processinfo.list.shm", procdname);

    /*
    * Check if a file exist using stat() function.
    * return 1 if the file exist otherwise return 0.
    */
    struct stat buffer;
    int         exists = stat(SM_fname, &buffer);

    if (exists == -1)
    {
        printf("CREATING PROCESSINFO LIST\n");

        size_t sharedsize = 0; // shared memory size in bytes
        int    SM_fd;          // shared memory file descriptor

        sharedsize = sizeof(PROCESSINFOLIST);
        umask(0);
        SM_fd = open(SM_fname, O_RDWR | O_CREAT | O_TRUNC, (mode_t) FILEMODE);
        if (SM_fd == -1)
        {
            perror("Error opening file for writing");
            exit(0);
        }

        int result;
        result = lseek(SM_fd, sharedsize - 1, SEEK_SET);
        if (result == -1)
        {
            close(SM_fd);
            fprintf(stderr, "Error calling lseek() to 'stretch' the file");
            exit(0);
        }

        result = write(SM_fd, "", 1);
        if (result != 1)
        {
            close(SM_fd);
            perror("Error writing last byte of the file");
            exit(0);
        }

        pinfolist = (PROCESSINFOLIST *)
            mmap(0, sharedsize, PROT_READ | PROT_WRITE, MAP_SHARED, SM_fd, 0);
        if (pinfolist == MAP_FAILED)
        {
            close(SM_fd);
            perror("Error mmapping the file");
            exit(0);
        }

        for (pindex = 0; pindex < PROCESSINFOLISTSIZE; pindex++)
        {
            pinfolist->active[pindex] = 0;
        }

        pindex = 0;
    }
    else
    {
        int SM_fd;
        //struct stat file_stat;

        pinfolist = (PROCESSINFOLIST *) processinfo_shm_link(SM_fname, &SM_fd);
        while ((pinfolist->active[pindex] != 0) &&
               (pindex < PROCESSINFOLISTSIZE))
        {
            pindex++;
        }

        if (pindex == PROCESSINFOLISTSIZE)
        {
            printf("ERROR: pindex reaches max value\n");
            exit(0);
        }
    }

    printf("pindex = %ld\n", pindex);

    return pindex;
}
