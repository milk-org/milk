// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    stream_merge.c
 * @brief   Merge n independently triggered streams
 *
 * Merges <shmname>_[0-N] into <shmname>,
 * assuming equal framerates.
 * Designed for parallel MVM computations.
 *
 * Uses FPS V2 framework.
 */

#include <stdio.h>
#include <stdlib.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "timeutils.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char    stream_basename[FUNCTION_PARAMETER_STRMAXLEN] = "stream";
static int32_t ptr_n_input                                   = 2;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                  \
    X(".stream_basename", stream_basename, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, \
      "output stream & input basename")                                                \
    X(".n_input", &ptr_n_input, FPTYPE_INT32, 1, FPFLAG_DEFAULT_INPUT,                 \
      "number of inputs to concatenate")


/* ================================================================
 * 5.  RUNNABLE
 * ============================================================= */

MILK_PARAMS_DEFINE(shmimmerge, FPS_PARAMS);

static MILK_HOT errno_t shmimmerge_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    int32_t n_input = ptr_n_input;

    IMGID *img_in_arr = (IMGID *) calloc(n_input, sizeof(IMGID));
    {
        char input_name[200];
        for (int ii = 0; ii < n_input; ++ii)
        {
            snprintf(input_name, sizeof(input_name), "%s_%d", stream_basename, ii);
            img_in_arr[ii] = imgid_make_from_name(input_name);
            resolveIMGID(&img_in_arr[ii], ERRMODE_ABORT, dcimg, dcnimg);
        }
    }

    IMGID img_out = imgid_make_from_name(stream_basename);
    resolveIMGID(&img_out, ERRMODE_WARN, dcimg, dcnimg);

    int32_t *offset_bytes = (int32_t *) calloc(n_input, sizeof(int32_t));
    if (offset_bytes == NULL)
    {
        PRINT_ERROR("calloc returns NULL pointer,"
                    " size %ld",
                    (long) (n_input * sizeof(int32_t)));
        abort();
    }

    int32_t *size_bytes = (int32_t *) calloc(n_input, sizeof(int32_t));
    if (size_bytes == NULL)
    {
        PRINT_ERROR("calloc returns NULL pointer,"
                    " size %ld",
                    (long) (n_input * sizeof(int32_t)));
        abort();
    }

    int32_t *sem_idxs = (int32_t *) calloc(n_input, sizeof(int32_t));
    if (sem_idxs == NULL)
    {
        PRINT_ERROR("calloc returns NULL pointer,"
                    " size %ld",
                    (long) (n_input * sizeof(int32_t)));
        abort();
    }

    {
        int acc = 0;
        for (int kk = 0; kk < n_input; ++kk)
        {
            offset_bytes[kk] = acc;
            size_bytes[kk] =
                img_in_arr[kk].mdt->size[0] * ImageStreamIO_typesize(img_in_arr[kk].mdt->datatype);
            acc += size_bytes[kk];

            sem_idxs[kk] = ImageStreamIO_getsemwaitindex(img_in_arr[kk].im, 0);
        }
    }

    MILK_PROCINFO_INIT(self)

    processinfo->triggermode = PROCESSINFO_TRIGGERMODE_IMMEDIATE;

    MILK_PROCINFO_LOOPSTART(self)
    {
        struct timespec t_spec1;
        struct timespec t_spec2;

        milk_clock_gettime(&t_spec1);
        t_spec2.tv_sec  = t_spec1.tv_sec + 1;
        t_spec2.tv_nsec = t_spec1.tv_nsec;

        for (int kk = 0; kk < n_input; kk++)
        {
            ImageStreamIO_semtimedwait(img_in_arr[kk].im, sem_idxs[kk], &t_spec2);
            ImageStreamIO_semflush(img_in_arr[kk].im, sem_idxs[kk]);
        }
        img_out.md->write = TRUE;
        for (int kk = 0; kk < n_input; kk++)
        {
            __builtin_memcpy(img_out.im->array.raw + offset_bytes[kk], img_in_arr[kk].im->array.raw,
                             size_bytes[kk]);
        }

        processinfo_update_output_stream(processinfo, img_out.im, NULL);
    }
    MILK_PROCINFO_END(self)

    for (int ii = 0; ii < n_input; ++ii)
    {
        imgid_free(&img_in_arr[ii]);
    }
    free(img_in_arr);
    imgid_free(&img_out);
    free(offset_bytes);
    free(size_bytes);
    free(sem_idxs);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_COREMOD_memory_shmimmerge = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "shmimmerge",
                 .cmdkey      = "shmimmerge",
                 .description = "Merge N in stream into out stream",
                 .description_long =
                     "Merge multiple 2D image streams into a single output stream by tiling "
                     "them side-by-side or stacking them into a 3D cube." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(shmimmerge),
    .compute = shmimmerge_compute,
};
