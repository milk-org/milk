/**
 * @file    image_setzero.c
 * @brief   Set all image pixels to zero
 *
 * Sets every element of a stream to zero.
 * Uses FPS V2 framework.
 */

#include <stdio.h>
#include <string.h>

#include "CLIcore.h"
#include "image_setzero.h"
#include "fps.h"
#include "fps_cli_binding.h"
#include "fps_cli_function.h"
#include "processinfo.h"
#include "ImageStreamIO.h"


/* ============================================ */
/* 1. FPS identity and local variables          */
/* ============================================ */

char *imsetzero_inimname = NULL;

static FPS_APP_INFO app_info = {
    .fps_name    = "imzero",
    .cmdkey      = "imzero",
    .description = "set all image pixels to zero"
};

static uint64_t processinfo_change_cnt_local;


/* ============================================ */
/* 2. Core computation                          */
/* ============================================ */

#ifndef FPS_STANDALONE
errno_t image_setzero_IMGID(IMGID *inimg)
{
    resolveIMGID(
        inimg, ERRMODE_ABORT,
        data.image, data.NB_MAX_IMAGE);
    memset(
        inimg->im->array.raw, 0,
        ImageStreamIO_typesize(
            inimg->md->datatype)
        * inimg->md->nelement);
    return RETURN_SUCCESS;
}

errno_t image_setzero(IMGID inimg)
{
    return image_setzero_IMGID(&inimg);
}
#endif

void image_setzero_compute(
    FUNCTION_PARAMETER_STRUCT *fps,
    PROCESSINFO               *processinfo,
    IMAGE                     *inimg
)
{
    if (fps && fps->md->processinfo_change_cnt
        != processinfo_change_cnt_local)
    {
        fps_to_processinfo(fps, processinfo);
        processinfo_change_cnt_local =
            fps->md->processinfo_change_cnt;
    }
    memset(
        inimg->array.raw, 0,
        ImageStreamIO_typesize(
            inimg->md[0].datatype)
        * inimg->md[0].nelement);
}


/* ============================================ */
/* 3. FPS bindings, farg, CLIcmddata            */
/* ============================================ */

static FPS_CLI_BINDING bindings[] = {
    IMSETZERO_PARAMS(FPS_X_BINDING)
};
static int nb_bindings =
    sizeof(bindings) / sizeof(bindings[0]);

static CLICMDARGDEF farg[] = {
    IMSETZERO_PARAMS(FPS_X_FARG)
};

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata = {
    "imzero",
    "set all image pixels to zero",
    CLICMD_FIELDS_DEFAULTS
};
static CMDSETTINGS default_cmdsettings = {0};
static __attribute__((constructor))
void init_cmdsettings_imzero(void)
{
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings =
            &default_cmdsettings;
    }
}
#else
static CLICMDDATA CLIcmddata = {
    "imzero",
    "set all image pixels to zero",
    CLICMD_FIELDS_DEFAULTS
};
#endif


/* ============================================ */
/* 4. compute_function wrapper                  */
/* ============================================ */

static errno_t compute_function()
{
    IMGID in =
        imgid_make_from_name(
            imsetzero_inimname);
    resolveIMGID(
        &in, ERRMODE_ABORT,
        data.image, data.NB_MAX_IMAGE);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    image_setzero_compute(
        data.fpsptr, processinfo, in.im);
    processinfo_update_output_stream(
        processinfo, in.im, NULL);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    return RETURN_SUCCESS;
}


/* ============================================ */
/* 5. CLI integration                           */
/* ============================================ */

#ifndef FPS_STANDALONE
static errno_t CLIfunction()
{
    return safe_fps_generic_CLIfunction(
        &app_info, farg, &CLIcmddata,
        bindings, nb_bindings,
        compute_function);
}

errno_t CLIADDCMD_COREMOD_arith__imsetzero()
{
    safe_fps_fill_farg_examples(
        farg, bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif


/* ============================================ */
/* 6. Standalone executable                     */
/* ============================================ */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    app_info,
    IMSETZERO_PARAMS,
    compute_function
)
#endif
