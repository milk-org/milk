#include <stdbool.h>

#include "CLIcore.h"

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
    resolveIMGID(img, ERRMODE_ABORT, data.image, data.NB_MAX_IMAGE);

    // check if shared memory destination exists
    resolveIMGID(imgshm, ERRMODE_NULL, data.image, data.NB_MAX_IMAGE);
    if( imgshm->ID != -1)
    {
        // image exists - checking if compatible size and type
        if( imgid_compare_md(*img, *imgshm) > 0 )
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
        imgid_copy( img, imgshm );
        imgshm->mdt->shared = 1;

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
    IMGID imgin = imgid_make_from_name(inname);
    IMGID imgshm = imgid_make_from_name(outname);

    errno_t ret = image_copy_shm_IMGID(&imgin, &imgshm);
    imgid_free(&imgin);
    imgid_free(&imgshm);
    return ret;
}

// adding INSERT_STD_PROCINFO statements enables processinfo support
static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID imgin = imgid_make_from_name(inimname);
    IMGID imgshm = imgid_make_from_name(outimname);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    image_copy_shm_IMGID(&imgin, &imgshm);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    imgid_free(&imgin);
    imgid_free(&imgshm);

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
