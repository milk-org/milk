// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file FITS_to_floatbin_lock.c
 * @brief Write float binary with file locking
 */

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include "milk_runnable.h"
#include "computeunit.h"

#include "COREMOD_memory/COREMOD_memory.h"

// Forward declaration
imageID IMAGE_FORMAT_FITS_to_floatbin_lock(const char *__restrict IDname,
                                           const char *__restrict fname);

/* =========================================
 *  V2 PARAMS
 * ======================================= */

static char p_in[FUNCTION_PARAMETER_STRMAXLEN]    = "im";
static char p_fname[FUNCTION_PARAMETER_STRMAXLEN] = "im.bin";

#define FPS_PARAMS(X)                                                              \
    X(".in_name", p_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".out_name", p_fname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output binary file")

MILK_PARAMS_DEFINE(FITS_to_floatbin_lock, FPS_PARAMS);

static MILK_HOT errno_t FITS_to_floatbin_lock_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    IMAGE_FORMAT_FITS_to_floatbin_lock(p_in, p_fname);
    return RETURN_SUCCESS;
}

imageID IMAGE_FORMAT_FITS_to_floatbin_lock(const char *__restrict IDname,
                                           const char *__restrict fname)
{
    imageID ID = -1;
    long    xsize, ysize;
    long    ii;
    int     fd;
    float  *valarray;

    ID    = image_ID(IDname, dcimg, dcnimg);
    xsize = dcimg[ID].md[0].size[0];
    ysize = dcimg[ID].md[0].size[1];

    valarray = (float *) malloc(sizeof(float) * xsize * ysize);
    if (valarray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    if (dcimg[ID].md[0].datatype == _DATATYPE_FLOAT)
    {
        printf("WRITING float array\n");
        for (ii = 0; ii < xsize * ysize; ii++)
        {
            valarray[ii] = dcimg[ID].array.F[ii];
        }
    }
    if (dcimg[ID].md[0].datatype == _DATATYPE_DOUBLE)
    {
        printf("WRITING double array\n");
        for (ii = 0; ii < xsize * ysize; ii++)
        {
            valarray[ii] = (float) dcimg[ID].array.D[ii];
        }
    }

    if ((fd = open(fname, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR)) == -1)
    {
        PRINT_ERROR("Cannot open file");
    }
    flock(fd, LOCK_EX);
    if (fd < 0)
    {
        printf("Error opening file: %s\n", strerror(errno));
    }

    if (write(fd, valarray, sizeof(float) * xsize * ysize) < 1)
    {
        PRINT_ERROR("write() returns <1 value");
    }
    //  for(ii=0;ii<xsize*ysize;ii++)
    //  printf("[%ld %f] ", ii, valarray[ii]);

    flock(fd, LOCK_UN);
    close(fd);

    free(valarray);

    return ID;
}

/* ================================================================
 * RUNNABLE DESCRIPTOR
 * ============================================================= */

const MILK_RUNNABLE runnable_image_format_FITS_to_floatbin_lock = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "writefloatlock",
                 .cmdkey      = "writefloatlock",
                 .description = "write float with file locking",
                 .description_long =
                     "Write a FITS image as a raw 32-bit float binary file with file locking for "
                     "safe concurrent access." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(FITS_to_floatbin_lock),
    .compute = FITS_to_floatbin_lock_compute,
};
