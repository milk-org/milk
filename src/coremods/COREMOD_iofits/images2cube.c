// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file images2cube.c
 * @brief ==========================================
 */

/**
 * @file    images2cube.c
 */

#include <stdio.h>
#include <stdlib.h>

#include "milk_runnable.h"
#include "computeunit.h"
#include "COREMOD_memory/COREMOD_memory.h"

// ==========================================
// Forward declaration(s)
// ==========================================

/**
 * @brief Assemble multiple 2D images into a 3D FITS cube.
 *
 * Reads a list of FITS filenames and stacks them
 * along the third axis.
 */
errno_t images_to_cube(const char *restrict img_name,
                       long nbframes,
                       const char *restrict cube_name);

// ==========================================
// FPS V2
// ==========================================

static char    *imgname;
static int64_t *nbframes;
static char    *cubename;

#define FPS_PARAMS(X)                                                                          \
    X(".imgname", &imgname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "input image name format") \
    X(".nbframes", &nbframes, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "number of frames")       \
    X(".cubename", &cubename, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output cube name")

MILK_PARAMS_DEFINE(images2cube, FPS_PARAMS);


static MILK_HOT errno_t images2cube_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    return images_to_cube(imgname, *nbframes, cubename);
}

const MILK_RUNNABLE runnable_COREMOD_iofits_images2cube = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "imgs2cube",
                 .cmdkey           = "imgs2cube",
                 .description      = "combine individual images into cube",
                 .description_long = "Assemble multiple 2D FITS files into a single 3D cube. Input "
                                     "files must have identical dimensions. The z-axis corresponds "
                                     "to the file sequence." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(images2cube),
    .compute = images2cube_compute,
};

EXPORT_FPSEXEC(milk, "iofits-imgs2cube", runnable_COREMOD_iofits_images2cube);


// ==========================================
// Compute code
// ==========================================

/**
 * @brief Assemble multiple 2D images into a 3D FITS cube.
 *
 * Reads a list of FITS filenames and stacks them
 * along the third axis.
 */
errno_t images_to_cube(const char *restrict img_name, long nbframes, const char *restrict cube_name)
{
    DEBUG_TRACE_FSTART();
    char imname[STRINGMAXLEN_IMGNAME];

    long frame = 0;
    CREATE_IMAGENAME(imname, "%s%05ld", img_name, frame);

    IMGID img1 = imgid_make_from_name(imname);
    resolveIMGID(&img1, ERRMODE_WARN, dcimg, dcnimg);
    if (img1.ID == -1)
    {
        return RETURN_FAILURE;
    }

    uint32_t xsize = img1.md->size[0];
    uint32_t ysize = img1.md->size[1];

    printf("SIZE = %u %u %ld\n", xsize, ysize, nbframes);
    fflush(stdout);

    IMGID imgcube       = imgid_make_from_name_3D(cube_name, xsize, ysize, nbframes);
    imgcube.mdt->shared = 0;
    imgid_create_registered(&imgcube);

    for (uint32_t ii = 0; ii < xsize; ii++)
    {
        for (uint32_t jj = 0; jj < ysize; jj++)
        {
            imgcube.im->array.F[jj * xsize + ii] = img1.im->array.F[jj * xsize + ii];
        }
    }

    for (frame = 1; frame < nbframes; frame++)
    {
        WRITE_IMAGENAME(imname, "%s%05ld", img_name, frame);
        printf("Adding image %s -> %ld/%ld.."
               " ",
               img_name, frame, nbframes);
        fflush(stdout);

        img1 = imgid_make_from_name(imname);
        resolveIMGID(&img1, ERRMODE_NULL, dcimg, dcnimg);
        if (img1.ID == -1)
        {
            PRINT_ERROR("Image \"%s\" does not "
                        "exist - skipping",
                        imname);
        }
        else
        {
            if ((xsize != img1.md->size[0]) || (ysize != img1.md->size[1]))
            {
                PRINT_ERROR("Image has wrong size");
                return RETURN_FAILURE;
            }
            for (uint32_t ii = 0; ii < xsize; ii++)
            {
                for (uint32_t jj = 0; jj < ysize; jj++)
                {
                    imgcube.im->array.F[frame * xsize * ysize + jj * xsize + ii] =
                        img1.im->array.F[jj * xsize + ii];
                }
            }
        }
        printf("Done\n");
        fflush(stdout);
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
