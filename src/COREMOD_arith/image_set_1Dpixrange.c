#include <string.h>

#include "CLIcore.h"
#include "image_set_1Dpixrange.h"
#include "fps.h"
#include "fps_cli_binding.h"
#include "fps_cli_function.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

char     *setpix1d_inimname = NULL;
float    *setpix1d_pixval   = NULL;
uint32_t *setpix1d_minindex = NULL;
uint32_t *setpix1d_maxindex = NULL;

static FPS_APP_INFO app_info = {
    .fps_name    = "setpix1D",
    .cmdkey      = "setpix1Drange",
    .description =
        "set image pixel value over range"
};

static uint64_t processinfo_change_cnt_local;

errno_t image_set_1Dpixrange(
    IMGID inimg, double value,
    uint32_t minindex, uint32_t maxindex
)
{
    if (maxindex > inimg.md->nelement) {
        maxindex = inimg.md->nelement;
    }
    if (minindex >= maxindex) {
        return RETURN_FAILURE;
    }
    switch (inimg.md->datatype) {
    case _DATATYPE_FLOAT:
        for (uint32_t i = minindex;
             i < maxindex; i++)
        {
            inimg.im->array.F[i] =
                (float) value;
        }
        break;
    case _DATATYPE_DOUBLE:
        for (uint32_t i = minindex;
             i < maxindex; i++)
        {
            inimg.im->array.D[i] = value;
        }
        break;
    }
    return RETURN_SUCCESS;
}

void image_set_1Dpixrange_compute(
    FUNCTION_PARAMETER_STRUCT *fps,
    PROCESSINFO *processinfo, IMAGE *inimg
)
{
    if (fps && fps->md->processinfo_change_cnt
        != processinfo_change_cnt_local)
    {
        fps_to_processinfo(fps, processinfo);
        processinfo_change_cnt_local =
            fps->md->processinfo_change_cnt;
    }
    if (!setpix1d_pixval
        || !setpix1d_minindex
        || !setpix1d_maxindex)
    {
        return;
    }
    IMGID id;
    id.im = inimg;
    id.md = &inimg->md[0];
    image_set_1Dpixrange(
        id, *setpix1d_pixval,
        *setpix1d_minindex,
        *setpix1d_maxindex);
}

static FPS_CLI_BINDING bindings[] = {
    SETPIX1D_PARAMS(FPS_X_BINDING)
};
static int nb_bindings =
    sizeof(bindings) / sizeof(bindings[0]);

static CLICMDARGDEF farg[] = {
    SETPIX1D_PARAMS(FPS_X_FARG)
};

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata = {
    "setpix1Drange",
    "set image pixel value over range",
    CLICMD_FIELDS_DEFAULTS
};
static CMDSETTINGS default_cmdsettings = {0};
static __attribute__((constructor))
void init_cmdsettings_setpix1D(void)
{
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings =
            &default_cmdsettings;
    }
}
#else
static CLICMDDATA CLIcmddata = {
    "setpix1Drange",
    "set image pixel value over range",
    CLICMD_FIELDS_DEFAULTS
};
#endif

static errno_t compute_function()
{
    IMGID in =
        imgid_make_from_name(
            setpix1d_inimname);
    resolveIMGID(
        &in, ERRMODE_ABORT,
        data.image, data.NB_MAX_IMAGE);
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    image_set_1Dpixrange_compute(
        data.fpsptr, processinfo, in.im);
    processinfo_update_output_stream(
        processinfo, in.im, NULL);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    return RETURN_SUCCESS;
}

#ifndef FPS_STANDALONE
static errno_t CLIfunction()
{
    return safe_fps_generic_CLIfunction(
        &app_info, farg, &CLIcmddata,
        bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_COREMOD_arith__imset_1Dpixrange()
{
    safe_fps_fill_farg_examples(
        farg, bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    app_info,
    SETPIX1D_PARAMS,
    compute_function
)
#endif
