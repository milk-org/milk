// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <stdbool.h>

#include "CommandLineInterface/CLIcore.h"

#include "create_image.h"
#include "read_shmim.h"

// Local variables pointers
static char *inimname;
static char *outimname;

static CLICMDARGDEF farg[] = {{
        CLIARG_IMG,
        ".in_name",
        "input image",
        "im1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimname,
        NULL
    },
    {
        CLIARG_STR,
        ".out_name",
        "output stream",
        "out1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimname,
        NULL
    }
};

// flag CLICMDFLAG_FPS enabled FPS capability
static CLICMDDATA CLIcmddata =
{
    "imcpshm", "copy image to shm", CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}

// copy image to shared memory
errno_t image_copy_shm_IMGID(
    IMGID *img,
    IMGID *imgshm
)
{
    resolveIMGID(img, ERRMODE_ABORT);

    // check if shared memory destination exists
    resolveIMGID(imgshm, ERRMODE_NULL);
    if( imgshm->ID != -1)
    {
        // image exists - checking if compatible size and type
        if( IMGIDmdcompare(*img, *imgshm) > 0 )
        {
            // image formats are incompatible
            // delete output
            printf("Image %s already exist in shm, but wrong size/format -> deleting\n", imgshm->name);

            ImageStreamIO_destroyIm(imgshm->im);
            imgshm->ID = -1;
        }
        else
        {
            printf("re-using existing shm %s\n", imgshm->name);
        }
    }

    if ( imgshm->ID == -1 )
    {
        copyIMGID( img, imgshm );
        imgshm->shared = 1;

        createimagefromIMGID(imgshm);
    }

    imgshm->md->write = 1;
    // copy data array
    memcpy(imgshm->im->array.raw,
           img->im->array.raw,
           ImageStreamIO_typesize(img->md->datatype)* img->md->nelement);
    // copy keywords
    memcpy(imgshm->im->kw, img->im->kw, sizeof(IMAGE_KEYWORD) * img->md->NBkw);

    COREMOD_MEMORY_image_set_sempost_byID(imgshm->ID, -1);
    imgshm->md->cnt0++;
    imgshm->md->write = 0;

    return RETURN_SUCCESS;
}

errno_t image_copy_shm(
    const char *inname,
    const char *outname
)
{
    IMGID imgin = mkIMGID_from_name(inname);
    IMGID imgshm = mkIMGID_from_name(outname);

    return image_copy_shm_IMGID(&imgin, &imgshm);
}

// adding INSERT_STD_PROCINFO statements enables processinfo support
static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    IMGID imgin = mkIMGID_from_name(inimname);
    IMGID imgshm = mkIMGID_from_name(outimname);

    image_copy_shm_IMGID(&imgin, &imgshm);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions

errno_t
CLIADDCMD_COREMOD_memory__image_copy_shm()
{
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
