// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file read_binary32f.c
 * @brief Read 32-bit float RAW image
 */

#include <stdio.h>
#include <stdlib.h>

#include "milk_runnable.h"
#include "computeunit.h"

#include "COREMOD_memory/COREMOD_memory.h"

// Forward declaration
imageID IMAGE_FORMAT_read_binary32f(const char *__restrict fname,
                                    long xsize,
                                    long ysize,
                                    const char *__restrict IDname);

static char      p_fname[FUNCTION_PARAMETER_STRMAXLEN] = "im.bin";
static long long p_xsize                               = 512;
static long long p_ysize                               = 512;
static char      p_out[FUNCTION_PARAMETER_STRMAXLEN]   = "im";

#define FPS_PARAMS(X)                                                              \
    X(".in_fname", p_fname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "binary file") \
    X(".xsize", &p_xsize, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "x size")         \
    X(".ysize", &p_ysize, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "y size")         \
    X(".out_name", p_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")

MILK_PARAMS_DEFINE(read_binary32f, FPS_PARAMS);

static MILK_HOT errno_t read_binary32f_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    IMAGE_FORMAT_read_binary32f(p_fname, (long) p_xsize, (long) p_ysize, p_out);
    return RETURN_SUCCESS;
}

imageID IMAGE_FORMAT_read_binary32f(const char *__restrict fname,
                                    long xsize,
                                    long ysize,
                                    const char *__restrict IDname)
{
    DEBUG_TRACE_FSTART();

    FILE         *fp;
    float        *buffer;
    unsigned long fileLen;

    //Open file
    if ((fp = fopen(fname, "rb")) == NULL)
    {
        PRINT_ERROR("Cannot open file");
        return (0);
    }

    //Get file length
    fseek(fp, 0, SEEK_END);
    fileLen = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    //Allocate memory
    buffer = (float *) malloc(fileLen + 1);
    if (!buffer)
    {
        fprintf(stderr, "Memory error!");
        fclose(fp);
        return (0);
    }

    //Read file contents into buffer
    if (fread(buffer, fileLen, 1, fp) < 1)
    {
        PRINT_ERROR("fread() returns <1 value");
    }
    fclose(fp);

    IMGID imgout       = imgid_make_from_name_2D(IDname, xsize, ysize);
    imgout.mdt->shared = 0;
    imgout.im          = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    long i = 0;
    for (long jj = 0; jj < ysize; jj++)
    {
        for (long ii = 0; ii < xsize; ii++)
        {
            imgout.im->array.F[jj * xsize + ii] = buffer[i];
            i++;
        }
    }

    free(buffer);

    DEBUG_TRACE_FEXIT();
    return imgout.ID;
}

/* ================================================================
 * RUNNABLE DESCRIPTOR
 * ============================================================= */

const MILK_RUNNABLE runnable_image_format_read_binary32f = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "readb32fim",
                 .cmdkey      = "readb32fim",
                 .description = "read 32-bit float RAW image",
                 .description_long =
                     "Read a raw 32-bit float binary file into a shared memory image stream." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(read_binary32f),
    .compute = read_binary32f_compute,
};
