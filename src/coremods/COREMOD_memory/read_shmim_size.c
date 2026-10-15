// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    read_shmim_size.c
 * @brief   read shared memory image size
 *
 * Uses FPS V2 framework.
 */

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "image_ID.h"
#include "list_image.h"
#include "read_shmim_size.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char insname[FUNCTION_PARAMETER_STRMAXLEN]  = "stream";
static char outfname[FUNCTION_PARAMETER_STRMAXLEN] = "imsize.txt";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                               \
    X(".in_sname", insname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "input stream") \
    X(".outfname", outfname, FPTYPE_STRING_NOT_STREAM, 1, FPFLAG_DEFAULT_INPUT, "output file name")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

/**
 * @brief Read shared memory image size
 *
 * @param name   stream name
 * @param fname  file name to write image size
 */
imageID read_sharedmem_image_size(const char *name, const char *fname)
{
    int             SM_fd;
    struct stat     file_stat;
    char            SM_fname[STRINGMAXLEN_FULLFILENAME];
    IMAGE_METADATA *map;

    FILE *fp;

    IMGID img = imgid_make_from_name(name);
    resolveIMGID(&img, ERRMODE_NULL, dcimg, dcnimg);

    if (img.ID == -1)
    {
        WRITE_FULLFILENAME(SM_fname, "%s/%s.im.shm", dcshmdir, name);

        SM_fd = open(SM_fname, O_RDWR);
        if (SM_fd == -1)
        {
            printf("Cannot import file"
                   " - continuing\n");
        }
        else
        {
            fstat(SM_fd, &file_stat);

            map = (IMAGE_METADATA *) mmap(0, sizeof(IMAGE_METADATA), PROT_READ | PROT_WRITE,
                                          MAP_SHARED, SM_fd, 0);
            if (map == MAP_FAILED)
            {
                close(SM_fd);
                PRINT_ERROR("mmap failed for %s", SM_fname);
                return -1;
            }

            fp = fopen(fname, "w");
            if (fp == NULL)
            {
                PRINT_ERROR("cannot open file \"%s\"", fname);
                if (munmap(map, sizeof(IMAGE_METADATA)) == -1)
                {
                    printf("unmapping %s\n", SM_fname);
                    PRINT_ERROR("Error un-mmapping the file: %s", strerror(errno));
                }
                close(SM_fd);
                return -1;
            }
            for (int i = 0; i < map[0].naxis; i++)
            {
                fprintf(fp, "%ld ", (long) map[0].size[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);

            if (munmap(map, sizeof(IMAGE_METADATA)) == -1)
            {
                printf("unmapping %s\n", SM_fname);
                PRINT_ERROR("Error un-mmapping the file: %s", strerror(errno));
            }
            close(SM_fd);
        }
    }
    else
    {
        fp = fopen(fname, "w");
        if (fp == NULL)
        {
            PRINT_ERROR("cannot open file \"%s\"", fname);
            return -1;
        }
        for (int i = 0; i < img.im->md[0].naxis; i++)
        {
            fprintf(fp, "%ld ", (long) img.im->md[0].size[i]);
        }
        fprintf(fp, "\n");
        fclose(fp);
    }

    return img.ID;
}


/* ================================================================
 * 5.  BINDINGS AND FARG
 * ============================================================= */

MILK_PARAMS_DEFINE(read_sharedmem_image_size, FPS_PARAMS);


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t read_sharedmem_image_size_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    read_sharedmem_image_size(insname, outfname);

    MILK_PROCINFO_END(self)
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  RUNNABLE
 * ============================================================= */

const MILK_RUNNABLE runnable_COREMOD_memory_readshmimsize = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "readshmimsize",
                 .cmdkey      = "readshmimsize",
                 .description = "read shared memory image size",
                 .description_long =
                     "Read only the size metadata of a shared memory stream without mapping the "
                     "full pixel buffer. Lightweight probe for stream dimensions." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(read_sharedmem_image_size),
    .compute = read_sharedmem_image_size_compute,
};
