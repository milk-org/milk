/**
 * @file    stream_ave.c
 * @brief   Average stream of images
 *
 * V2 FPS framework migration.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#include "CLIcore.h"
#include "stream_ave.h"
#include "fps.h"
#include "fps_cli_binding.h"
#include "fps_cli_function.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

char     *streamave_inimname    = NULL;
char     *streamave_outimave    = NULL;
uint32_t *streamave_outimshared = NULL;
char     *streamave_outimrms    = NULL;
uint64_t *streamave_NBcoadd     = NULL;
uint64_t *streamave_cntindex    = NULL;
uint64_t *streamave_compave     = NULL;
uint64_t *streamave_comprms     = NULL;

static uint64_t
    processinfo_change_cnt_local = 0;


/* =========================================
 * Compute kernel
 * ========================================= */

void stream_ave_compute(
    FUNCTION_PARAMETER_STRUCT *fps,
    PROCESSINFO               *processinfo,
    IMAGE                     *imgin,
    IMAGE                     *imgoutave,
    IMAGE                     *imgoutrms,
    double                    *imdataarray,
    double                    *imdataarrayPOW
)
{
    if (fps &&
        fps->md->processinfo_change_cnt
        != processinfo_change_cnt_local)
    {
        fps_to_processinfo(fps, processinfo);
        processinfo_change_cnt_local =
            fps->md->processinfo_change_cnt;
    }

    uint64_t xysize =
        imgin->md[0].size[0]
        * imgin->md[0].size[1];

    if (*streamave_cntindex == 0) {
        for (uint64_t i = 0; i < xysize; i++) {
            double v = 0;
            switch (imgin->md[0].datatype) {
            case _DATATYPE_FLOAT:
                v = imgin->array.F[i];
                break;
            case _DATATYPE_DOUBLE:
                v = imgin->array.D[i];
                break;
            }
            imdataarray[i] = v;
            if (*streamave_comprms) {
                imdataarrayPOW[i] = v * v;
            }
        }
    }
    else {
        for (uint64_t i = 0; i < xysize; i++) {
            double v = 0;
            switch (imgin->md[0].datatype) {
            case _DATATYPE_FLOAT:
                v = imgin->array.F[i];
                break;
            case _DATATYPE_DOUBLE:
                v = imgin->array.D[i];
                break;
            }
            imdataarray[i] += v;
            if (*streamave_comprms) {
                imdataarrayPOW[i] += v * v;
            }
        }
    }

    (*streamave_cntindex)++;

    if (*streamave_cntindex >= *streamave_NBcoadd)
    {
        if (*streamave_compave && imgoutave) {
            for (uint64_t i = 0; i < xysize; i++)
            {
                imgoutave->array.F[i] =
                    imdataarray[i]
                    / (*streamave_cntindex);
            }
            processinfo_update_output_stream(
                processinfo, imgoutave, NULL);
        }
        if (*streamave_comprms && imgoutrms) {
            for (uint64_t i = 0; i < xysize; i++)
            {
                imgoutrms->array.F[i] =
                    sqrt(imdataarrayPOW[i])
                    / (*streamave_cntindex);
            }
            processinfo_update_output_stream(
                processinfo, imgoutrms, NULL);
        }
        *streamave_cntindex = 0;
    }
}


/* =========================================
 * V2 FPS-CLI integration
 * ========================================= */

static FPS_APP_INFO app_info = {
    .fps_name    = "streamave",
    .cmdkey      = "streamave",
    .description = "average stream of images",
};

static FPS_CLI_BINDING bindings[] = {
    STREAMAVE_PARAMS(FPS_X_BINDING)
};
static int nb_bindings =
    sizeof(bindings) / sizeof(bindings[0]);

static CLICMDARGDEF farg[] = {
    STREAMAVE_PARAMS(FPS_X_FARG)
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
    "streamave",
    "average stream of images",
    CLICMD_FIELDS_DEFAULTS
};
#endif

static errno_t compute_function()
{
    IMGID in =
        imgid_make_from_name(streamave_inimname);
    resolveIMGID(&in, ERRMODE_ABORT,
                 data.image, data.NB_MAX_IMAGE);
    uint64_t xys =
        in.md[0].size[0] * in.md[0].size[1];
    double *d1 = malloc(sizeof(double) * xys);
    double *d2 = malloc(sizeof(double) * xys);
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    stream_ave_compute(
        data.fpsptr, processinfo,
        in.im, NULL, NULL, d1, d2);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    free(d1);
    free(d2);
    return RETURN_SUCCESS;
}

static errno_t CLIfunction()
{
    return safe_fps_generic_CLIfunction(
        &app_info, farg, &CLIcmddata,
        bindings, nb_bindings,
        compute_function);
}

errno_t CLIADDCMD_streamaverage()
{
    safe_fps_fill_farg_examples(
        farg, bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    app_info,
    STREAMAVE_PARAMS,
    compute_function
)
#endif