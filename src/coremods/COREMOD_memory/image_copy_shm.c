// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_copy_shm.c
 * @brief   copy image to shared memory
 *
 * Uses FPS V2 framework.
 */

#include <stdbool.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "create_image.h"
#include "read_shmim.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char inimname[FUNCTION_PARAMETER_STRMAXLEN]  = "imin";
static char outimname[FUNCTION_PARAMETER_STRMAXLEN] = "imout";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                  \
    X(".in_name", inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".out_name", outimname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output stream")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

errno_t image_copy_shm_IMGID(IMGID *img, IMGID *imgshm)
{
    resolveIMGID(img, ERRMODE_ABORT, dcimg, dcnimg);

    resolveIMGID(imgshm, ERRMODE_NULL, dcimg, dcnimg);
    if (imgshm->ID != -1)
    {
        if (imgid_compare_md(*img, *imgshm) > 0)
        {
            printf("Image %s already exist in shm,"
                   " but wrong size/format"
                   " -> deleting\n",
                   imgshm->name);
            ImageStreamIO_destroyIm(imgshm->im);
            imgshm->ID = -1;
        }
        else
        {
            printf("re-using existing shm %s\n", imgshm->name);
        }
    }

    if (imgshm->ID == -1)
    {
        imgid_copy(img, imgshm);
        imgshm->mdt->shared = 1;
        createimagefromIMGID(imgshm);
    }

    imgshm->md->write = 1;
    memcpy(imgshm->im->array.raw, img->im->array.raw,
           ImageStreamIO_typesize(img->md->datatype) * img->md->nelement);
    memcpy(imgshm->im->kw, img->im->kw, sizeof(IMAGE_KEYWORD) * img->md->NBkw);

    COREMOD_MEMORY_image_set_sempost_byID(imgshm->ID, -1);
    imgshm->md->cnt0++;
    imgshm->md->write = 0;

    return RETURN_SUCCESS;
}

errno_t image_copy_shm(const char *inname, const char *outname)
{
    IMGID imgin  = imgid_make_from_name(inname);
    IMGID imgshm = imgid_make_from_name(outname);

    errno_t ret = image_copy_shm_IMGID(&imgin, &imgshm);
    imgid_free(&imgin);
    imgid_free(&imgshm);
    return ret;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(image_copy_shm, FPS_PARAMS);


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t image_copy_shm_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    IMGID imgin  = imgid_make_from_name(inimname);
    IMGID imgshm = imgid_make_from_name(outimname);

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    image_copy_shm_IMGID(&imgin, &imgshm);

    MILK_PROCINFO_END(self)
    imgid_free(&imgin);
    imgid_free(&imgshm);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  RUNNABLE
 * ============================================================= */

const MILK_RUNNABLE runnable_COREMOD_memory_image_copy_shm = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "imcpshm",
                 .cmdkey      = "imcpshm",
                 .description = "copy image to shm",
                 .description_long =
                     "Copy an image from local memory to shared memory (/dev/shm), making it "
                     "accessible to other processes. Creates the shared memory segment if it does "
                     "not already exist." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(image_copy_shm),
    .compute = image_copy_shm_compute,
};
