/**
 * @file    imresize.c
 * @brief   Resize 2D image
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
#include "imresize.h"
#include "fps.h"
#include "fps_cli_binding.h"
#include "fps_cli_function.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

#include "COREMOD_memory/COREMOD_memory.h"

char *imresize_inimname  = NULL;
char *imresize_outimname = NULL;
long *imresize_xsize     = NULL;
long *imresize_ysize     = NULL;


static void imresize_step(
    IMAGE *imgin,
    IMAGE *imgout
)
{
    uint32_t nx_in  = imgin->md[0].size[0];
    uint32_t ny_in  = imgin->md[0].size[1];
    uint32_t nx_out = imgout->md[0].size[0];
    uint32_t ny_out = imgout->md[0].size[1];

    if (imgin->md[0].datatype != _DATATYPE_FLOAT)
    {
        return;
    }

    for (uint32_t ii = 0; ii < nx_out; ii++) {
        for (uint32_t jj = 0; jj < ny_out; jj++)
        {
            float xf1 =
                (float)ii * nx_in / nx_out;
            float yf1 =
                (float)jj * ny_in / ny_out;
            long ii1 = (long)xf1;
            long jj1 = (long)yf1;
            float uf = xf1 - (float)ii1;
            float tf = yf1 - (float)jj1;

            if ((ii1 >= 0)
                && (ii1 + 1 < (long)nx_in)
                && (jj1 >= 0)
                && (jj1 + 1 < (long)ny_in))
            {
                float v00 = imgin->array.F[
                    jj1 * nx_in + ii1];
                float v01 = imgin->array.F[
                    (jj1 + 1) * nx_in + ii1];
                float v10 = imgin->array.F[
                    jj1 * nx_in + ii1 + 1];
                float v11 = imgin->array.F[
                    (jj1 + 1) * nx_in + ii1 + 1];

                imgout->array.F[
                    jj * nx_out + ii] =
                    v00 * (1.0 - uf) * (1.0 - tf)
                    + v10 * uf * (1.0 - tf)
                    + v01 * (1.0 - uf) * tf
                    + v11 * uf * tf;
            }
            else {
                imgout->array.F[
                    jj * nx_out + ii] = 0.0;
            }
        }
    }
}


#ifndef FPS_STANDALONE
long basic_resizeim(
    const char *imname_in,
    const char *imname_out,
    long        xsizeout,
    long        ysizeout
)
{
    IMGID in =
        imgid_make_from_name(imname_in);
    resolveIMGID(&in, ERRMODE_ABORT,
                 data.image, data.NB_MAX_IMAGE);
    IMGID out = stream_connect_create_2Df32(
        imname_out, xsizeout, ysizeout);
    imresize_step(in.im, out.im);
    ImageStreamIO_UpdateIm(out.im);
    return 0;
}
#endif


/* =========================================
 * V2 FPS-CLI integration
 * ========================================= */

static FPS_APP_INFO app_info = {
    .fps_name    = "resizeim",
    .cmdkey      = "resizeim",
    .description = "resize 2D image",
};

static FPS_CLI_BINDING bindings[] = {
    IMRESIZE_PARAMS(FPS_X_BINDING)
};
static int nb_bindings =
    sizeof(bindings) / sizeof(bindings[0]);

static CLICMDARGDEF farg[] = {
    IMRESIZE_PARAMS(FPS_X_FARG)
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
    "resizeim",
    "resize 2D image",
    CLICMD_FIELDS_DEFAULTS
};
#endif

static errno_t compute_function()
{
    basic_resizeim(
        imresize_inimname,
        imresize_outimname,
        *imresize_xsize,
        *imresize_ysize);
    return RETURN_SUCCESS;
}

static errno_t CLIfunction()
{
    return safe_fps_generic_CLIfunction(
        &app_info, farg, &CLIcmddata,
        bindings, nb_bindings,
        compute_function);
}

errno_t imresize_addCLIcmd()
{
    safe_fps_fill_farg_examples(
        farg, bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    app_info,
    IMRESIZE_PARAMS,
    compute_function
)
#endif
