// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file processinfo_shm_link.c
 * @brief Processinfo shm link module
 */

#include <fcntl.h>
#include <string.h>
#include <sys/mman.h> // mmap()
#include <sys/stat.h>
#include <unistd.h>

#include "milk_config.h"
#include "processinfo_internal.h"
#include "processinfo_shm_link.h"

static PROCESSINFO *processinfo_shm_link_mode(const char *pname, int *fd, int readonly)
{
    int SM_fd = open(pname, readonly ? O_RDONLY : O_RDWR);
    if (SM_fd == -1)
    {
        return (PROCESSINFO *) MAP_FAILED;
    }

    struct stat SM_stat;
    if (fstat(SM_fd, &SM_stat) == -1)
    {
        close(SM_fd);
        return (PROCESSINFO *) MAP_FAILED;
    }
    size_t sharedsize = SM_stat.st_size;

    /* Exact sizes only: guarantees callers may munmap(sizeof(struct)). */
    int compatible = (sharedsize == sizeof(PROCESSINFO) || sharedsize == sizeof(PROCESSINFOLIST));

    PROCESSINFO *pinfo = (PROCESSINFO *) MAP_FAILED;
    if (compatible)
    {
        pinfo = (PROCESSINFO *) mmap(0, sharedsize, readonly ? PROT_READ : PROT_READ | PROT_WRITE,
                                     MAP_SHARED, SM_fd, 0);
        if (pinfo == MAP_FAILED)
        {
            close(SM_fd);
            return (PROCESSINFO *) MAP_FAILED;
        }
        compatible =
            (strncmp(pinfo->version, MILK_PROCESSINFO_VERSION, sizeof(pinfo->version)) == 0);
    }

    if (!compatible)
    {
        /* Reported once per process: TUIs re-scan every refresh. */
        static int reported = 0;
        if (!reported)
        {
            PRINT_ERROR("%s: incompatible processinfo structure (expected version %s)", pname,
                        MILK_PROCESSINFO_VERSION);
            reported = 1;
        }
        if (pinfo != MAP_FAILED)
        {
            munmap(pinfo, sharedsize);
        }
        close(SM_fd);
        return (PROCESSINFO *) MAP_FAILED;
    }

    *fd = SM_fd;

    return pinfo;
}

/**
 * @brief Open and map a processinfo shared memory file read-write.
 *
 * Works for both per-process PROCESSINFO files and the PROCESSINFOLIST
 * file. Returns a pointer to the mapped struct and the file descriptor.
 * Returns MAP_FAILED if the file cannot be opened, or if its size or
 * structure version does not match MILK_PROCESSINFO_VERSION.
 * The mapping is exactly sizeof(PROCESSINFO) or sizeof(PROCESSINFOLIST).
 */
PROCESSINFO *processinfo_shm_link(const char *pname, int *fd)
{
    return processinfo_shm_link_mode(pname, fd, 0);
}

/**
 * @brief Same as processinfo_shm_link(), but opened and mapped read-only.
 */
PROCESSINFO *processinfo_shm_link_ro(const char *pname, int *fd)
{
    return processinfo_shm_link_mode(pname, fd, 1);
}
