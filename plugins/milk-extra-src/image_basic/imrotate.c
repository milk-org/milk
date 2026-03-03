/**
 * @file    imrotate.c
 * @brief   Rotate 2D image
 *
 * V2 FPS framework migration.
 */

#include "ImageStreamIO/ImageStruct.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "CLIcore.h"
#include "imrotate.h"
#include "fps.h"
#include "fps_cli_binding.h"
#include "fps_cli_function.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

#include "COREMOD_memory/COREMOD_memory.h"

char  *imrotate_inimname  = NULL;
char  *imrotate_outimname = NULL;
float *imrotate_angle     = NULL;


static void imrotate_step(
    IMAGE *imgin,
    IMAGE *imgout,
    float  angle
)
{
    uint32_t nx = imgin->md[0].size[0];
    uint32_t ny = imgin->md[0].size[1];
    float c = cos(angle);
    float s = sin(angle);
    for (uint32_t jj = 0; jj < ny; jj++) {
        for (uint32_t ii = 0; ii < nx; ii++) {
            long iis = (long)(nx / 2
                + (ii - (int)nx / 2) * c
                + (jj - (int)ny / 2) * s);
            long jjs = (long)(ny / 2
                - (ii - (int)nx / 2) * s
                + (jj - (int)ny / 2) * c);
            if ((iis >= 0) && (jjs >= 0)
                && (iis < (long)nx)
                && (jjs < (long)ny))
            {
                imgout->array.F[jj * nx + ii] =
                    imgin->array.F[jjs * nx + iis];
            }
            else {
                imgout->array.F[jj * nx + ii] =
                    0.0;
            }
        }
    }
}


#ifndef FPS_STANDALONE
imageID basic_rotate(
    const char *__restrict ID_name,
    const char *__restrict IDout_name,
    float angle
)
{
    IMGID in =
        imgid_make_from_name(ID_name);
    resolveIMGID(&in, ERRMODE_ABORT,
                 data.image, data.NB_MAX_IMAGE);
    IMGID out = stream_connect_create_2Df32(
        IDout_name,
        in.md->size[0], in.md->size[1]);
    imrotate_step(in.im, out.im, angle);
    ImageStreamIO_UpdateIm(out.im);
    return out.ID;
}
#endif


/* =========================================
 * V2 FPS-CLI integration
 * ========================================= */

static FPS_APP_INFO app_info = {
    .fps_name    = "rotateim",
    .cmdkey      = "rotateim",
    .description = "rotate 2D image",
};

static FPS_CLI_BINDING bindings[] = {
    IMROTATE_PARAMS(FPS_X_BINDING)
};
static int nb_bindings =
    sizeof(bindings) / sizeof(bindings[0]);

static CLICMDARGDEF farg[] = {
    IMROTATE_PARAMS(FPS_X_FARG)
};

#ifdef FPS_STANDALONE
static CLICMDDATA CLIcmddata;
__attribute__((constructor))
static void init_CLIcmddata(void)
{
    memset(&CLIcmddata, 0, sizeof(CLIcmddata));
    strncpy(CLIcmddata.key, app_info.cmdkey,
            sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description,
            app_info.description,
            sizeof(CLIcmddata.description) - 1);
}
#else
static CLICMDDATA CLIcmddata = {
    "rotateim",
    "rotate 2D image",
    CLICMD_FIELDS_DEFAULTS
};
#endif

static errno_t compute_function()
{
    basic_rotate(
        imrotate_inimname,
        imrotate_outimname,
        *imrotate_angle);
    return RETURN_SUCCESS;
}

static errno_t CLIfunction()
{
    return safe_fps_generic_CLIfunction(
        &app_info, farg, &CLIcmddata,
        bindings, nb_bindings,
        compute_function);
}

errno_t imrotate_addCLIcmd()
{
    safe_fps_fill_farg_examples(
        farg, bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    app_info,
    IMROTATE_PARAMS,
    compute_function
)
#endif
