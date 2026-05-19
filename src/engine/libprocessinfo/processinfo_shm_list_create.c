/**
 * @file processinfo_shm_list_create.c
 * @brief Processinfo shm list create module
 */

#include <sys/mman.h> // mmap()
#include <sys/stat.h>
#include <fcntl.h>

#include "processinfo_internal.h"
#include "processinfo_procdirname.h"
#include "processinfo_shm_link.h"

#define FILEMODE 0666


/**
 * @brief Create or attach to the processinfo list SHM and reserve a slot.
 *
 * If the list file does not exist, create it and zero its `active` flags.
 * Otherwise, link to the existing file and find the first inactive slot.
 *
 * @param pindex_out  On success, receives the reserved slot index. Must
 *                    not be NULL.
 *
 * @return RETURN_SUCCESS on success, RETURN_FAILURE on any error
 *         (open/lseek/write/mmap failure, link failure, list full).
 */
errno_t processinfo_shm_list_create(long *pindex_out)
{
    errno_t rv     = RETURN_FAILURE;
    int     SM_fd  = -1;
    long    pindex = 0;

    if (pindex_out == NULL)
    {
        FUNC_RETURN_FAILURE("pindex_out is NULL");
    }

    char SM_fname[STRINGMAXLEN_FULLFILENAME];
    char procdname[STRINGMAXLEN_DIRNAME];
    processinfo_procdirname(procdname);

    WRITE_FULLFILENAME(SM_fname, "%s/processinfo.list.shm", procdname);

    /* Check whether the list file already exists. */
    struct stat buffer;
    int         exists = stat(SM_fname, &buffer);

    if (exists == -1)
    {
        printf("CREATING PROCESSINFO LIST\n");

        size_t sharedsize = sizeof(PROCESSINFOLIST);
        umask(0);
        SM_fd = open(SM_fname,
                     O_RDWR | O_CREAT | O_TRUNC,
                     (mode_t) FILEMODE);
        if (SM_fd == -1)
        {
            PRINT_ERROR("open(%s) failed: %s",
                        SM_fname, strerror(errno));
            goto fail;
        }

        if (lseek(SM_fd, sharedsize - 1, SEEK_SET) == -1)
        {
            PRINT_ERROR("lseek failed: %s", strerror(errno));
            goto fail;
        }

        if (write(SM_fd, "", 1) != 1)
        {
            PRINT_ERROR("write last byte failed: %s",
                        strerror(errno));
            goto fail;
        }

        pinfolist = (PROCESSINFOLIST *)
                    mmap(0,
                         sharedsize,
                         PROT_READ | PROT_WRITE,
                         MAP_SHARED,
                         SM_fd,
                         0);
        if (pinfolist == MAP_FAILED)
        {
            PRINT_ERROR("mmap(%s) failed: %s",
                        SM_fname, strerror(errno));
            pinfolist = NULL;
            goto fail;
        }

        for (pindex = 0; pindex < PROCESSINFOLISTSIZE; pindex++)
        {
            pinfolist->active[pindex] = 0;
        }

        pindex = 0;
    }
    else
    {
        int link_fd;

        pinfolist = (PROCESSINFOLIST *)
                    processinfo_shm_link(SM_fname, &link_fd);
        if (pinfolist == MAP_FAILED)
        {
            FUNC_RETURN_FAILURE(
                "processinfo_shm_link(%s) failed", SM_fname);
        }

        while ((pinfolist->active[pindex] != 0) &&
               (pindex < PROCESSINFOLISTSIZE))
        {
            pindex++;
        }

        if (pindex == PROCESSINFOLISTSIZE)
        {
            FUNC_RETURN_FAILURE(
                "pindex reached max value (%d)",
                PROCESSINFOLISTSIZE);
        }
    }

    *pindex_out = pindex;
    rv = RETURN_SUCCESS;

fail:
    if (SM_fd != -1)
    {
        close(SM_fd);
    }
    return rv;
}
