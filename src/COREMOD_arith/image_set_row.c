/**
 * @file    image_set_row.c
 * @brief   Set image row pixels to a value
 *
 * Sets all pixels in a specified row of an image
 * stream to a given value.
 * Uses FPS V2 framework.
 */

#include <stdio.h>
#include <string.h>

#include "CLIcore.h"
#include "image_set_row.h"
#include "fps.h"
#include "fps_cli_binding.h"
#include "fps_cli_function.h"
#include "processinfo.h"
#include "ImageStreamIO.h"


/* ============================================ */
/* 1. FPS identity and local variables          */
/* ============================================ */

char     *setrow_inimname = NULL;
float    *setrow_pixval   = NULL;
uint32_t *setrow_rowindex = NULL;

static FPS_APP_INFO app_info = {
    .fps_name    = "setrow",
    .cmdkey      = "setrow",
    .description = "set image row pixel values"
};

static uint64_t processinfo_change_cnt_local;


/* ============================================ */
/* 2. Core computation                          */
/* ============================================ */

errno_t image_set_row(
    IMGID    inimg,
    double   value,
    uint32_t rowindex
)
{
    if (rowindex >= inimg.md->size[1]) {
        return RETURN_FAILURE;
    }
    uint32_t xsize = inimg.md->size[0];
    switch (inimg.md->datatype) {
    case _DATATYPE_FLOAT:
        for (uint32_t i = 0; i < xsize; i++) {
            inimg.im->array.F[
                rowindex * xsize + i] =
                (float) value;
        }
        break;
    case _DATATYPE_DOUBLE:
        for (uint32_t i = 0; i < xsize; i++) {
            inimg.im->array.D[
                rowindex * xsize + i] = value;
        }
        break;
    }
    return RETURN_SUCCESS;
}

void image_set_row_compute(
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
    if (!setrow_pixval || !setrow_rowindex) {
        return;
    }
    IMGID id;
    id.im = inimg;
    id.md = &inimg->md[0];
    image_set_row(id, *setrow_pixval,
                  *setrow_rowindex);
}


/* ============================================ */
/* 3. FPS bindings, farg, CLIcmddata            */
/* ============================================ */

static FPS_CLI_BINDING bindings[] = {
    SETROW_PARAMS(FPS_X_BINDING)
};
static int nb_bindings =
    sizeof(bindings) / sizeof(bindings[0]);

static CLICMDARGDEF farg[] = {
    SETROW_PARAMS(FPS_X_FARG)
};

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata = {
    "setrow",
    "set image row pixel values",
    CLICMD_FIELDS_DEFAULTS
};
static CMDSETTINGS default_cmdsettings = {0};
static __attribute__((constructor))
void init_cmdsettings_setrow(void)
{
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings =
            &default_cmdsettings;
    }
}
#else
static CLICMDDATA CLIcmddata = {
    "setrow",
    "set image row pixel values",
    CLICMD_FIELDS_DEFAULTS
};
#endif


/* ============================================ */
/* 4. compute_function wrapper                  */
/* ============================================ */

static errno_t compute_function()
{
    IMGID in =
        imgid_make_from_name(setrow_inimname);
    resolveIMGID(
        &in, ERRMODE_ABORT,
        data.image, data.NB_MAX_IMAGE);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    image_set_row_compute(
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

errno_t CLIADDCMD_COREMOD_arith__imset_row()
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
    SETROW_PARAMS,
    compute_function
)
#endif
