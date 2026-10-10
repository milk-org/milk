// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    rt_legacy_passthru.c
 * @brief   Legacy counterpart of runnable_rt_passthru: out = in (fpsexec only, no CLI command)
 */

#include "CLIcore.h"
#include "computeunit.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "runnabletest/rt_kernels.h"

static FPS_APP_INFO FPS_app_info = { .fps_name = "rtlpassthru",
                                     .cmdkey   = "passthru",
                                     .description =
                                         "runnabletest legacy: out = in (fpsexec only)" };

static char inimname[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static char outimname[FUNCTION_PARAMETER_STRMAXLEN] = "";

#define FPS_PARAMS(X)                                                                     \
    X(".in_name", inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_TRIGGER_STREAM, "input") \
    X(".out_name", outimname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output")

// Only used by the standalone build of this file
static __attribute__((unused)) FPS_CLI_BINDING my_bindings[] = { FPS_PARAMS(FPS_X_BINDING) };

static const int __attribute__((unused)) nb_bindings =
    sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);

static __attribute__((unused)) CLICMDARGDEF farg[] = { FPS_PARAMS(FPS_X_FARG) };

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata = {
#else
static __attribute__((unused)) CLICMDDATA CLIcmddata = {
#endif
    "", "", CLICMD_FIELDS_DEFAULTS
};

FPS_CMDSETTINGS_INIT(dft, CLIcmddata, FPS_app_info)


static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID inimg = imgid_make_from_name(inimname);
    resolveIMGID(&inimg, ERRMODE_WARN, dcimg, dcnimg);
    if (inimg.ID == -1)
    {
        return RETURN_FAILURE;
    }

    IMGID outimg = imgid_make_from_name(outimname);
    imgid_copy(&inimg, &outimg);
    imcreateIMGID(&outimg);

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT
    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        rt_stream_affine(&inimg, &outimg, 1.0f, 0.0f);
        processinfo_update_output_stream(processinfo, outimg.im, inimg.im);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    imgid_free(&inimg);
    imgid_free(&outimg);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)
#endif
