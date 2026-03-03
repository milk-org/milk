/**
 * @file    gaussfilter.c
 * @brief   Image filtering
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
#include "gaussfilter.h"
#include "fps.h"
#include "fps_cli_binding.h"
#include "fps_cli_function.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_memory/COREMOD_memory.h"

char  *gaussfilt_inimname   = NULL;
char  *gaussfilt_outimname  = NULL;
float *gaussfilt_sigma      = NULL;
int   *gaussfilt_filtersize = NULL;


static void gauss_filter_step(
    IMAGE *imgin,
    IMAGE *imgout,
    float  sigma,
    int    filter_size
)
{
    uint32_t nx = imgin->md[0].size[0];
    uint32_t ny = imgin->md[0].size[1];
    uint32_t nz = (imgin->md[0].naxis == 3)
        ? imgin->md[0].size[2] : 1;
    int fsize = filter_size;
    if (fsize > (int)nx / 2 - 1) {
        fsize = nx / 2 - 1;
    }
    if (fsize > (int)ny / 2 - 1) {
        fsize = ny / 2 - 1;
    }

    float *array = (float *) malloc(
        (2 * fsize + 1) * sizeof(float));
    float sum = 0.0;
    for (int i = 0; i < (2 * fsize + 1); i++) {
        array[i] = exp(
            -((i - fsize) * (i - fsize))
            / sigma / sigma);
        sum += array[i];
    }
    for (int i = 0; i < (2 * fsize + 1); i++) {
        array[i] /= sum;
    }

    float *tmp = (float *) calloc(
        nx * ny, sizeof(float));
    for (uint32_t k = 0; k < nz; k++) {
        float *pl_in =
            imgin->array.F + k * nx * ny;
        float *pl_out =
            imgout->array.F + k * nx * ny;
        memset(tmp, 0, nx * ny * sizeof(float));
        for (uint32_t j = 0; j < ny; j++) {
            for (uint32_t i = fsize;
                 i < nx - fsize; i++)
            {
                for (int ii = -fsize;
                     ii <= fsize; ii++)
                {
                    tmp[j * nx + i] +=
                        array[ii + fsize]
                        * pl_in[j * nx + i + ii];
                }
            }
        }
        for (uint32_t i = 0; i < nx; i++) {
            for (uint32_t j = fsize;
                 j < ny - fsize; j++)
            {
                float v = 0;
                for (int jj = -fsize;
                     jj <= fsize; jj++)
                {
                    v += array[jj + fsize]
                         * tmp[(j + jj) * nx + i];
                }
                pl_out[j * nx + i] = v;
            }
        }
    }
    free(tmp);
    free(array);
}


#ifndef FPS_STANDALONE
imageID gauss_filter(
    const char *ID_name,
    const char *out_name,
    float       sigma,
    int         filter_size
)
{
    IMGID in =
        imgid_make_from_name(ID_name);
    resolveIMGID(&in, ERRMODE_ABORT,
                 data.image, data.NB_MAX_IMAGE);
    IMGID out = stream_connect_create_2Df32(
        out_name,
        in.md->size[0], in.md->size[1]);
    gauss_filter_step(
        in.im, out.im, sigma, filter_size);
    ImageStreamIO_UpdateIm(out.im);
    return out.ID;
}
#endif


/* =========================================
 * V2 FPS-CLI integration
 * ========================================= */

static FPS_APP_INFO app_info = {
    .fps_name    = "gaussfilt",
    .cmdkey      = "gaussfilt",
    .description = "gaussian 2D filtering",
};

static FPS_CLI_BINDING bindings[] = {
    GAUSSFILT_PARAMS(FPS_X_BINDING)
};
static int nb_bindings =
    sizeof(bindings) / sizeof(bindings[0]);

static CLICMDARGDEF farg[] = {
    GAUSSFILT_PARAMS(FPS_X_FARG)
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
    "gaussfilt",
    "gaussian 2D filtering",
    CLICMD_FIELDS_DEFAULTS
};
#endif

static errno_t compute_function()
{
    gauss_filter(
        gaussfilt_inimname,
        gaussfilt_outimname,
        *gaussfilt_sigma,
        *gaussfilt_filtersize);
    return RETURN_SUCCESS;
}

static errno_t CLIfunction()
{
    return safe_fps_generic_CLIfunction(
        &app_info, farg, &CLIcmddata,
        bindings, nb_bindings,
        compute_function);
}

errno_t gaussfilter_addCLIcmd()
{
    safe_fps_fill_farg_examples(
        farg, bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    app_info,
    GAUSSFILT_PARAMS,
    compute_function
)
#endif
