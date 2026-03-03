#include <string.h>

#include "CLIcore.h"
#include "image_set_3Daxes.h"
#include "fps.h"
#include "fps_cli_binding.h"
#include "fps_cli_function.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

char     *set3d_inimname = NULL;
uint32_t *set3d_size0    = NULL;
uint32_t *set3d_size1    = NULL;
uint32_t *set3d_size2    = NULL;

static FPS_APP_INFO app_info = {
    .fps_name    = "set3Daxes",
    .cmdkey      = "set3Daxes",
    .description = "set 3D image axes size"
};

static uint64_t processinfo_change_cnt_local;

errno_t image_set_3Daxes(
    IMGID inimg, uint32_t imsize0,
    uint32_t imsize1, uint32_t imsize2
)
{
    long nelem = inimg.md->nelement;
    uint32_t s0 = (imsize0 == 0)
        ? inimg.md->size[0] : imsize0;
    uint32_t s1 = (imsize1 == 0)
        ? ((inimg.md->naxis < 2)
           ? 1 : inimg.md->size[1])
        : imsize1;
    uint32_t s2 = (imsize2 == 0)
        ? ((inimg.md->naxis < 3)
           ? 1 : inimg.md->size[2])
        : imsize2;
    if ((long) s0 * s1 * s2 == nelem) {
        inimg.md->naxis = 3;
        inimg.md->size[0] = s0;
        inimg.md->size[1] = s1;
        inimg.md->size[2] = s2;
    }
    return RETURN_SUCCESS;
}

void image_set_3Daxes_compute(
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
    if (!set3d_size0 || !set3d_size1
        || !set3d_size2)
    {
        return;
    }
    IMGID id;
    id.im = inimg;
    id.md = &inimg->md[0];
    image_set_3Daxes(
        id, *set3d_size0,
        *set3d_size1, *set3d_size2);
}

static FPS_CLI_BINDING bindings[] = {
    SET3DAXES_PARAMS(FPS_X_BINDING)
};
static int nb_bindings =
    sizeof(bindings) / sizeof(bindings[0]);

static CLICMDARGDEF farg[] = {
    SET3DAXES_PARAMS(FPS_X_FARG)
};

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata = {
    "set3Daxes",
    "set 3D image axes size",
    CLICMD_FIELDS_DEFAULTS
};
static CMDSETTINGS default_cmdsettings = {0};
static __attribute__((constructor))
void init_cmdsettings_set3Daxes(void)
{
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings =
            &default_cmdsettings;
    }
}
#else
static CLICMDDATA CLIcmddata = {
    "set3Daxes",
    "set 3D image axes size",
    CLICMD_FIELDS_DEFAULTS
};
#endif

static errno_t compute_function()
{
    IMGID in =
        imgid_make_from_name(set3d_inimname);
    resolveIMGID(
        &in, ERRMODE_ABORT,
        data.image, data.NB_MAX_IMAGE);
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    image_set_3Daxes_compute(
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

errno_t CLIADDCMD_COREMOD_arith__imset_3Daxes()
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
    SET3DAXES_PARAMS,
    compute_function
)
#endif
