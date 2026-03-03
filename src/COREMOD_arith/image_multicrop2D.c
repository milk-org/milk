#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "image_multicrop2D.h"
#include "fps.h"
#include "fps_cli_binding.h"
#include "fps_cli_function.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

char     *multicrop_insname  = NULL;
char     *multicrop_outsname = NULL;
uint32_t *multicrop_outxsize = NULL;
uint32_t *multicrop_outysize = NULL;

int64_t  *multicrop_wactive[
    MAXNB_CROPWINDOW];
int64_t  *multicrop_waddmode[
    MAXNB_CROPWINDOW];
uint32_t *multicrop_wcropxstart[
    MAXNB_CROPWINDOW];
uint32_t *multicrop_wcropxsize[
    MAXNB_CROPWINDOW];
uint32_t *multicrop_wcropystart[
    MAXNB_CROPWINDOW];
uint32_t *multicrop_wcropysize[
    MAXNB_CROPWINDOW];
uint32_t *multicrop_wbinfact[
    MAXNB_CROPWINDOW];
uint32_t *multicrop_wcropxpos[
    MAXNB_CROPWINDOW];
uint32_t *multicrop_wcropypos[
    MAXNB_CROPWINDOW];

static FPS_APP_INFO app_info = {
    .fps_name    = "multicrop",
    .cmdkey      = "multicrop2D",
    .description =
        "crop 2D image, multiple crops"
};

static uint64_t processinfo_change_cnt_local;

void image_multicrop2D_compute(
    FUNCTION_PARAMETER_STRUCT *fps,
    PROCESSINFO *processinfo,
    IMAGE *imgin, IMAGE *imgout
)
{
    if (fps && fps->md->processinfo_change_cnt
        != processinfo_change_cnt_local)
    {
        fps_to_processinfo(fps, processinfo);
        processinfo_change_cnt_local =
            fps->md->processinfo_change_cnt;
    }
    uint32_t ox = *multicrop_outxsize;
    uint32_t oy = *multicrop_outysize;
    size_t ts = ImageStreamIO_typesize(
        imgin->md[0].datatype);
    memset(imgout->array.raw, 0,
           ts * ox * oy);
    for (int w = 0;
         w < MAXNB_CROPWINDOW; w++)
    {
        if (multicrop_wactive[w]
            && *multicrop_wactive[w] == 1)
        {
            uint32_t xs =
                *multicrop_wcropxstart[w];
            uint32_t ys =
                *multicrop_wcropystart[w];
            uint32_t xw =
                *multicrop_wcropxsize[w];
            uint32_t yw =
                *multicrop_wcropysize[w];
            uint32_t xp =
                *multicrop_wcropxpos[w];
            uint32_t yp =
                *multicrop_wcropypos[w];
            uint32_t bf =
                *multicrop_wbinfact[w];
            if (bf < 1) {
                bf = 1;
            }
            uint32_t cxw = xw;
            if (xp + cxw / bf > ox) {
                cxw = (ox - xp) * bf;
            }
            if (xs + cxw
                > imgin->md[0].size[0])
            {
                cxw = imgin->md[0].size[0]
                      - xs;
            }
            uint32_t cyw = yw;
            if (yp + cyw / bf > oy) {
                cyw = (oy - yp) * bf;
            }
            if (ys + cyw
                > imgin->md[0].size[1])
            {
                cyw = imgin->md[0].size[1]
                      - ys;
            }
            for (uint32_t j = 0;
                 j < cyw; j++)
            {
                uint64_t ioff =
                    (uint64_t)(ys + j)
                    * imgin->md[0].size[0]
                    + xs;
                uint64_t ooff =
                    (uint64_t)(yp + j / bf)
                    * ox + xp;
                if (*multicrop_waddmode[w]
                    == 0)
                {
                    memcpy(
                        ((char *)
                         imgout->array.raw)
                        + ooff * ts,
                        ((char *)
                         imgin->array.raw)
                        + ioff * ts,
                        ts * (cxw / bf));
                }
                else if (
                    imgin->md[0].datatype
                    == _DATATYPE_FLOAT)
                {
                    for (uint32_t i = 0;
                         i < cxw; i++)
                    {
                        imgout->array.F[
                            ooff + i / bf]
                            += imgin->array.F[
                                ioff + i];
                    }
                }
            }
        }
    }
}

errno_t image_multicrop2D_validate()
{
    if (multicrop_outxsize
        && *multicrop_outxsize < 1)
    {
        *multicrop_outxsize = 1;
    }
    if (multicrop_outysize
        && *multicrop_outysize < 1)
    {
        *multicrop_outysize = 1;
    }
    return RETURN_SUCCESS;
}


static FPS_CLI_BINDING bindings[] = {
    MULTICROP2D_PARAMS(FPS_X_BINDING)
};
static int nb_bindings =
    sizeof(bindings) / sizeof(bindings[0]);

static CLICMDARGDEF farg[] = {
    MULTICROP2D_PARAMS(FPS_X_FARG)
};

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata = {
    "multicrop2D",
    "crop 2D image, multiple crops",
    CLICMD_FIELDS_DEFAULTS
};
static CMDSETTINGS default_cmdsettings = {0};
static __attribute__((constructor))
void init_cmdsettings_multicrop(void)
{
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings =
            &default_cmdsettings;
    }
}
#else
static CLICMDDATA CLIcmddata = {
    "multicrop2D",
    "crop 2D image, multiple crops",
    CLICMD_FIELDS_DEFAULTS
};
#endif

static errno_t compute_function()
{
    IMGID in = imgid_make_from_name(
        multicrop_insname);
    resolveIMGID(
        &in, ERRMODE_ABORT,
        data.image, data.NB_MAX_IMAGE);
    IMGID out = stream_connect_create_2D(
        multicrop_outsname,
        *multicrop_outxsize,
        *multicrop_outysize,
        in.md->datatype);
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    image_multicrop2D_compute(
        data.fpsptr, processinfo,
        in.im, out.im);
    processinfo_update_output_stream(
        processinfo, out.im, in.im);
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
CLIADDCMD_COREMODE_arith__multicrop2D()
{
    CLIcmddata.FPS_customCONFcheck =
        image_multicrop2D_validate;
    safe_fps_fill_farg_examples(
        farg, bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    app_info,
    MULTICROP2D_PARAMS,
    compute_function
)
#endif