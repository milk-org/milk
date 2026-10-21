// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file FITS_to_ushortintbin_lock.c
 * @brief Write ushort binary with file locking
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
imageID IMAGE_FORMAT_FITS_to_ushortintbin_lock(const char *__restrict IDname,
                                               const char *__restrict fname);

static char p_in[FUNCTION_PARAMETER_STRMAXLEN]    = "im";
static char p_fname[FUNCTION_PARAMETER_STRMAXLEN] = "im.bin";

#define FPS_PARAMS(X)                                                              \
    X(".in_name", p_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".out_name", p_fname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output binary file")

MILK_PARAMS_DEFINE(FITS_to_ushortintbin_lock, FPS_PARAMS);

static MILK_HOT errno_t FITS_to_ushortintbin_lock_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    IMAGE_FORMAT_FITS_to_ushortintbin_lock(p_in, p_fname);
    return RETURN_SUCCESS;
}

imageID IMAGE_FORMAT_FITS_to_ushortintbin_lock(const char *__restrict IDname,
                                               const char *__restrict fname)
{
    imageID             ID;
    long                xsize, ysize;
    long                ii;
    int                 fd;
    unsigned short int *valarray;

    ID    = image_ID(IDname, dcimg, dcnimg);
    xsize = dcimg[ID].md[0].size[0];
    ysize = dcimg[ID].md[0].size[1];

    valarray = (unsigned short int *) malloc(sizeof(unsigned short int) * xsize * ysize);
    if (valarray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    if (dcimg[ID].md[0].datatype == _DATATYPE_FLOAT)
    {
        printf("float -> unsigned short int array\n");
        for (ii = 0; ii < xsize * ysize; ii++)
        {
            valarray[ii] = (unsigned short int) dcimg[ID].array.F[ii];
        }
    }
    if (dcimg[ID].md[0].datatype == _DATATYPE_DOUBLE)
    {
        printf("double -> unsigned short int array\n");
        for (ii = 0; ii < xsize * ysize; ii++)
        {
            valarray[ii] = (unsigned short int) dcimg[ID].array.D[ii];
        }
    }

    fd = open(fname, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    flock(fd, LOCK_EX);
    if (fd < 0)
    {
        perror("Error opening file");
        printf("Error opening file \"%s\": %s\n", fname, strerror(errno));
    }
    if (write(fd, valarray, sizeof(unsigned short int) * xsize * ysize) < 1)
    {
        PRINT_ERROR("write() returns <1 value");
    }
    flock(fd, LOCK_UN);
    close(fd);

    free(valarray);

    return ID;
}

/* ================================================================
 * RUNNABLE DESCRIPTOR
 * ============================================================= */

const MILK_RUNNABLE runnable_image_format_FITS_to_ushortintbin_lock = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "writeushortintlock",
                 .cmdkey      = "writeushortintlock",
                 .description = "write ushort with file locking",
                 .description_long =
                     "Write a FITS image as a raw unsigned short binary file with file locking for "
                     "safe concurrent access." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(FITS_to_ushortintbin_lock),
    .compute = FITS_to_ushortintbin_lock_compute,
};
