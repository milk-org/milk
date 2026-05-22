/**
 * @file    image_multicrop2D.c
 * @brief   Multi-window 2D cropping from stream
 *
 * Uses FPS V2 framework.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "multicrop",
    .cmdkey           = "multicrop2D",
    .description      = "crop 2D image, multiple crops",
    .description_long = "Extract multiple rectangular sub-regions from a single 2D image, "
                        "specified by a list of origin coordinates and sizes. Outputs are written "
                        "as separate shared memory streams or assembled into a 3D cube."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

#define MAXNB_CROPWINDOW 8

static char     *multicrop_insname  = NULL;
static char     *multicrop_outsname = NULL;
static uint32_t *multicrop_outxsize = NULL;
static uint32_t *multicrop_outysize = NULL;

static int64_t  *multicrop_wactive[MAXNB_CROPWINDOW];
static int64_t  *multicrop_waddmode[MAXNB_CROPWINDOW];
static uint32_t *multicrop_wcropxstart[MAXNB_CROPWINDOW];
static uint32_t *multicrop_wcropxsize[MAXNB_CROPWINDOW];
static uint32_t *multicrop_wcropystart[MAXNB_CROPWINDOW];
static uint32_t *multicrop_wcropysize[MAXNB_CROPWINDOW];
static uint32_t *multicrop_wbinfact[MAXNB_CROPWINDOW];
static uint32_t *multicrop_wcropxpos[MAXNB_CROPWINDOW];
static uint32_t *multicrop_wcropypos[MAXNB_CROPWINDOW];


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define MULTICROP_WPARAMS(X, wn)                                                                  \
    X(".w" #wn ".active", &multicrop_wactive[wn], FPTYPE_ONOFF, 0, FPFLAG_DEFAULT_INPUT,          \
      "crop window active")                                                                       \
    X(".w" #wn ".addmode", &multicrop_waddmode[wn], FPTYPE_ONOFF, 0, FPFLAG_DEFAULT_INPUT,        \
      "1:add, 0:replace")                                                                         \
    X(".w" #wn ".cropxstart", &multicrop_wcropxstart[wn], FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT, \
      "crop x coord start")                                                                       \
    X(".w" #wn ".cropxsize", &multicrop_wcropxsize[wn], FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,   \
      "crop x coord size")                                                                        \
    X(".w" #wn ".cropystart", &multicrop_wcropystart[wn], FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT, \
      "crop y coord start")                                                                       \
    X(".w" #wn ".cropysize", &multicrop_wcropysize[wn], FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,   \
      "crop y coord size")                                                                        \
    X(".w" #wn ".cropxpos", &multicrop_wcropxpos[wn], FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,     \
      "x placement in output")                                                                    \
    X(".w" #wn ".cropypos", &multicrop_wcropypos[wn], FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,     \
      "y placement in output")                                                                    \
    X(".w" #wn ".cropbinfact", &multicrop_wbinfact[wn], FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,   \
      "binning factor")

#define FPS_PARAMS(X)                                                                            \
    X(".insname", &multicrop_insname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,                \
      "input stream name")                                                                       \
    X(".outsname", &multicrop_outsname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,              \
      "output stream name")                                                                      \
    X(".outxsize", &multicrop_outxsize, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "output x size") \
    X(".outysize", &multicrop_outysize, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "output y size") \
    MULTICROP_WPARAMS(X, 0)                                                                      \
    MULTICROP_WPARAMS(X, 1)                                                                      \
    MULTICROP_WPARAMS(X, 2)                                                                      \
    MULTICROP_WPARAMS(X, 3)                                                                      \
    MULTICROP_WPARAMS(X, 4)                                                                      \
    MULTICROP_WPARAMS(X, 5)                                                                      \
    MULTICROP_WPARAMS(X, 6)                                                                      \
    MULTICROP_WPARAMS(X, 7)


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

static MILK_HOT errno_t fpsexec(IMAGE *imgin, IMAGE *imgout)
{
    uint32_t ox = *multicrop_outxsize;
    uint32_t oy = *multicrop_outysize;
    size_t   ts = ImageStreamIO_typesize(imgin->md[0].datatype);

    memset(imgout->array.raw, 0, ts * ox * oy);

    for (int w = 0; w < MAXNB_CROPWINDOW; w++)
    {
        if (multicrop_wactive[w] && *multicrop_wactive[w] == 1)
        {
            uint32_t xs = *multicrop_wcropxstart[w];
            uint32_t ys = *multicrop_wcropystart[w];
            uint32_t xw = *multicrop_wcropxsize[w];
            uint32_t yw = *multicrop_wcropysize[w];
            uint32_t xp = *multicrop_wcropxpos[w];
            uint32_t yp = *multicrop_wcropypos[w];
            uint32_t bf = *multicrop_wbinfact[w];
            if (bf < 1)
            {
                bf = 1;
            }
            uint32_t cxw = xw;
            if (xp + cxw / bf > ox)
            {
                cxw = (ox - xp) * bf;
            }
            if (xs + cxw > imgin->md[0].size[0])
            {
                cxw = imgin->md[0].size[0] - xs;
            }
            uint32_t cyw = yw;
            if (yp + cyw / bf > oy)
            {
                cyw = (oy - yp) * bf;
            }
            if (ys + cyw > imgin->md[0].size[1])
            {
                cyw = imgin->md[0].size[1] - ys;
            }
            for (uint32_t j = 0; j < cyw; j++)
            {
                uint64_t ioff = (uint64_t) (ys + j) * imgin->md[0].size[0] + xs;
                uint64_t ooff = (uint64_t) (yp + j / bf) * ox + xp;
                if (*multicrop_waddmode[w] == 0)
                {
                    memcpy(((char *) imgout->array.raw) + ooff * ts,
                           ((char *) imgin->array.raw) + ioff * ts, ts * (cxw / bf));
                }
                else if (imgin->md[0].datatype == _DATATYPE_FLOAT)
                {
                    for (uint32_t i = 0; i < cxw; i++)
                    {
                        imgout->array.F[ooff + i / bf] += imgin->array.F[ioff + i];
                    }
                }
            }
        }
    }
    return RETURN_SUCCESS;
}

static errno_t __attribute__((unused)) multicrop2D_validate()
{
    if (multicrop_outxsize && *multicrop_outxsize < 1)
    {
        *multicrop_outxsize = 1;
    }
    if (multicrop_outysize && *multicrop_outysize < 1)
    {
        *multicrop_outysize = 1;
    }
    return RETURN_SUCCESS;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    IMGID in = imgid_make_from_name(multicrop_insname);
    resolveIMGID(&in, ERRMODE_ABORT, dcimg, dcnimg);
    IMGID out = stream_connect_create_2D(multicrop_outsname, *multicrop_outxsize,
                                         *multicrop_outysize, in.md->datatype);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START fpsexec(in.im, out.im);
    processinfo_update_output_stream(processinfo, out.im, in.im);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_COREMODE_arith__multicrop2D()
{
    CLIcmddata.FPS_customCONFcheck = multicrop2D_validate;
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC return RETURN_SUCCESS;
}
#endif


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)
#endif