/**
 * @file    cubecollapse.c
 * @brief   Collapse a cube along z axis
 *
 * V2 FPS framework migration.
 */

#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "CLIcore.h"
#include "cubecollapse.h"
#include "fps.h"
#include "fps_cli_binding.h"
#include "fps_cli_function.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

#include "COREMOD_memory/COREMOD_memory.h"

char *cubecollapse_inimname = NULL;
char *cubecollapse_outimname = NULL;


static void cube_collapse_step(
    IMAGE *imgin,
    IMAGE *imgout
)
{
    uint32_t xsize = imgin->md[0].size[0];
    uint32_t ysize = imgin->md[0].size[1];
    uint32_t ksize = imgin->md[0].size[2];
    for (uint32_t i = 0; i < xsize * ysize; i++)
    {
        float v = 0.0;
        for (uint32_t k = 0; k < ksize; k++) {
            v += imgin->array.F[
                k * xsize * ysize + i];
        }
        imgout->array.F[i] = v;
    }
}


#ifndef FPS_STANDALONE
imageID cube_collapse(
    const char *__restrict ID_in_name,
    const char *__restrict ID_out_name
)
{
    IMGID in =
        imgid_make_from_name(ID_in_name);
    resolveIMGID(&in, ERRMODE_ABORT,
                 data.image, data.NB_MAX_IMAGE);
    IMGID out = stream_connect_create_2Df32(
        ID_out_name,
        in.md->size[0], in.md->size[1]);
    cube_collapse_step(in.im, out.im);
    ImageStreamIO_UpdateIm(out.im);
    return out.ID;
}
#endif


/* =========================================
 * V2 FPS-CLI integration
 * ========================================= */

static FPS_APP_INFO app_info = {
    .fps_name    = "cubecollapse",
    .cmdkey      = "cubecollapse",
    .description = "collapse a cube along z",
};

static FPS_CLI_BINDING bindings[] = {
    CUBECOLLAPSE_PARAMS(FPS_X_BINDING)
};
static int nb_bindings =
    sizeof(bindings) / sizeof(bindings[0]);

static CLICMDARGDEF farg[] = {
    CUBECOLLAPSE_PARAMS(FPS_X_FARG)
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
    "cubecollapse",
    "collapse a cube along z",
    CLICMD_FIELDS_DEFAULTS
};
#endif

static errno_t compute_function()
{
    cube_collapse(
        cubecollapse_inimname,
        cubecollapse_outimname);
    return RETURN_SUCCESS;
}

static errno_t CLIfunction()
{
    return safe_fps_generic_CLIfunction(
        &app_info, farg, &CLIcmddata,
        bindings, nb_bindings,
        compute_function);
}

errno_t __attribute__((cold))
cubecollapse_addCLIcmd()
{
    safe_fps_fill_farg_examples(
        farg, bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    app_info,
    CUBECOLLAPSE_PARAMS,
    compute_function
)
#endif
