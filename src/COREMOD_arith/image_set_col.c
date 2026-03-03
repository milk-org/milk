#include <string.h>

#include "CLIcore.h"
#include "image_set_col.h"
#include "fps.h"
#include "fps_cli_binding.h"
#include "fps_cli_function.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

char     *setcol_inimname = NULL;
float    *setcol_pixval   = NULL;
uint32_t *setcol_colindex = NULL;

static FPS_APP_INFO app_info = {
    .fps_name    = "setcol",
    .cmdkey      = "setcol",
    .description = "set image column pixel values"
};

static uint64_t processinfo_change_cnt_local;

errno_t image_set_col(
    IMGID    inimg,
    double   value,
    uint32_t colindex
)
{
    if (colindex >= inimg.md->size[0]) {
        return RETURN_FAILURE;
    }
    uint32_t xsize = inimg.md->size[0];
    uint32_t ysize = inimg.md->size[1];
    switch (inimg.md->datatype) {
    case _DATATYPE_FLOAT:
        for (uint32_t j = 0; j < ysize; j++) {
            inimg.im->array.F[
                j * xsize + colindex] =
                (float) value;
        }
        break;
    case _DATATYPE_DOUBLE:
        for (uint32_t j = 0; j < ysize; j++) {
            inimg.im->array.D[
                j * xsize + colindex] = value;
        }
        break;
    }
    return RETURN_SUCCESS;
}

void image_set_col_compute(
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
    if (!setcol_pixval || !setcol_colindex) {
        return;
    }
    IMGID id;
    id.im = inimg;
    id.md = &inimg->md[0];
    image_set_col(id, *setcol_pixval,
                  *setcol_colindex);
}

static FPS_CLI_BINDING bindings[] = {
    SETCOL_PARAMS(FPS_X_BINDING)
};
static int nb_bindings =
    sizeof(bindings) / sizeof(bindings[0]);

static CLICMDARGDEF farg[] = {
    SETCOL_PARAMS(FPS_X_FARG)
};

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata = {
    "setcol",
    "set image column pixel values",
    CLICMD_FIELDS_DEFAULTS
};
static CMDSETTINGS default_cmdsettings = {0};
static __attribute__((constructor))
void init_cmdsettings_setcol(void)
{
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings =
            &default_cmdsettings;
    }
}
#else
static CLICMDDATA CLIcmddata = {
    "setcol",
    "set image column pixel values",
    CLICMD_FIELDS_DEFAULTS
};
#endif

static errno_t compute_function()
{
    IMGID in =
        imgid_make_from_name(setcol_inimname);
    resolveIMGID(
        &in, ERRMODE_ABORT,
        data.image, data.NB_MAX_IMAGE);
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    image_set_col_compute(
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

errno_t CLIADDCMD_COREMOD_arith__imset_col()
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
    SETCOL_PARAMS,
    compute_function
)
#endif
