/**
 * @file    image_copy_shm.c
 * @brief   copy image to shared memory
 *
 * Uses FPS V2 framework.
 */

#include <stdbool.h>

#include "CLIcore.h"
#include "fps.h"

#include "create_image.h"
#include "read_shmim.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "imcpshm",
    .cmdkey      = "imcpshm",
    .description = "copy image to shm"
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char *inimname  = NULL;
static char *outimname = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".in_name", &inimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input image") \
    X(".out_name", &outimname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output stream")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

errno_t image_copy_shm_IMGID(
    IMGID *img,
    IMGID *imgshm
)
{
    resolveIMGID(
        img, ERRMODE_ABORT,
        data.image, data.NB_MAX_IMAGE);

    resolveIMGID(
        imgshm, ERRMODE_NULL,
        data.image, data.NB_MAX_IMAGE);
    if(imgshm->ID != -1)
    {
        if(imgid_compare_md(*img, *imgshm) > 0)
        {
            printf(
                "Image %s already exist in shm,"
                " but wrong size/format"
                " -> deleting\n",
                imgshm->name);
            ImageStreamIO_destroyIm(imgshm->im);
            imgshm->ID = -1;
        }
        else
        {
            printf("re-using existing shm %s\n",
                   imgshm->name);
        }
    }

    if(imgshm->ID == -1)
    {
        imgid_copy(img, imgshm);
        imgshm->mdt->shared = 1;
        createimagefromIMGID(imgshm);
    }

    imgshm->md->write = 1;
    memcpy(imgshm->im->array.raw,
           img->im->array.raw,
           ImageStreamIO_typesize(
               img->md->datatype)
           * img->md->nelement);
    memcpy(imgshm->im->kw,
           img->im->kw,
           sizeof(IMAGE_KEYWORD)
           * img->md->NBkw);

    COREMOD_MEMORY_image_set_sempost_byID(
        imgshm->ID, -1);
    imgshm->md->cnt0++;
    imgshm->md->write = 0;

    return RETURN_SUCCESS;
}

errno_t image_copy_shm(
    const char *inname,
    const char *outname
)
{
    IMGID imgin =
        imgid_make_from_name(inname);
    IMGID imgshm =
        imgid_make_from_name(outname);

    errno_t ret = image_copy_shm_IMGID(
        &imgin, &imgshm);
    imgid_free(&imgin);
    imgid_free(&imgshm);
    return ret;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

static FPS_CLI_BINDING my_bindings[] = {
    FPS_PARAMS(FPS_X_BINDING)
};

static const int nb_bindings =
    sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = {
    FPS_PARAMS(FPS_X_FARG)
};

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata = {
#else
static CLICMDDATA CLIcmddata = {
#endif
    "",
    "",
    CLICMD_FIELDS_DEFAULTS
};

static CMDSETTINGS default_cmdsettings = {0};

static __attribute__((constructor))
void init_cmdsettings(void)
{
    strncpy(CLIcmddata.key,
            FPS_app_info.cmdkey,
            sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description,
            FPS_app_info.description,
            sizeof(CLIcmddata.description) - 1);
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings =
            &default_cmdsettings;
    }
}


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID imgin =
        imgid_make_from_name(inimname);
    IMGID imgshm =
        imgid_make_from_name(outimname);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    image_copy_shm_IMGID(&imgin, &imgshm);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    imgid_free(&imgin);
    imgid_free(&imgshm);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#ifndef FPS_STANDALONE
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_COREMOD_memory__image_copy_shm()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    FPS_app_info,
    FPS_PARAMS,
    compute_function)
#endif
