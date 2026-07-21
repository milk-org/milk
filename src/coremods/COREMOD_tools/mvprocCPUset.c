// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mvprocCPUset.c
 * @brief CPU set and RT priority utilities
 *
 * Uses FPS V2 framework.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "COREMOD_memory/COREMOD_memory.h"
#include "fps.h"
#include "milk_rt.h"

/* ================================================================
 *  COMMON PARAMS
 * ============================================================= */

static long long p_pid                                = 0;
static char      p_name[FUNCTION_PARAMETER_STRMAXLEN] = "realtime";
static long long p_rtprio                             = 80;

/* ================================================================
 *  CMD 1: rtprio (1 arg)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_rtp = {
    .fps_name         = "rtprio",
    .cmdkey           = "rtprio",
    .description      = "set SCHED_FIFO priority",
    .description_long = "Move processes to specific CPU sets for core pinning and isolation."
};

#define FPS_PARAMS_RTP(X)

static CLICMDDATA CLIcmddata_rtp = { "", "", CLICMD_FIELDS_NOPARAM };
FPS_CMDSETTINGS_INIT(rtp, CLIcmddata_rtp, FPS_app_info_rtp)

static errno_t __attribute__((unused)) compute_rtp()
{
    milkrt_RTPrio(p_rtprio);
    return RETURN_SUCCESS;
}

/* ================================================================
 *  CMD 2: tsetpmove (1 arg)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_tset = {
    .fps_name         = "tsetpmove",
    .cmdkey           = "tsetpmove",
    .description      = "assign taskset to current process",
    .description_long = "Move processes to specific CPU sets for core pinning and isolation."
};

#define FPS_PARAMS_TSET(X) \
    X(".name", p_name, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "taskset spec list")

static CLICMDDATA CLIcmddata_tset = { "", "", CLICMD_FIELDS_NOPARAM };
FPS_CMDSETTINGS_INIT(tset, CLIcmddata_tset, FPS_app_info_tset)

static errno_t __attribute__((unused)) compute_tset()
{
    milkrt_Tset(p_name);
    return RETURN_SUCCESS;
}

/* ================================================================
 *  CMD 3: tsetpmoveext (2 args)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_tsete = {
    .fps_name         = "tsetpmoveext",
    .cmdkey           = "tsetpmoveext",
    .description      = "assign taskset for any process",
    .description_long = "Move processes to specific CPU sets for core pinning and isolation."
};

#define FPS_PARAMS_TSETE(X)                                         \
    X(".pid", &p_pid, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "PID") \
    X(".name", p_name, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "taskset spec list")

static CLICMDDATA CLIcmddata_tsete = { "", "", CLICMD_FIELDS_NOPARAM };
FPS_CMDSETTINGS_INIT(tsete, CLIcmddata_tsete, FPS_app_info_tsete)

static errno_t __attribute__((unused)) compute_tsete()
{
    milkrt_TsetExt(p_pid, p_name);
    return RETURN_SUCCESS;
}

/* ================================================================
 *  CMD 4: csetpmove (1 arg)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_cset = {
    .fps_name         = "csetpmove",
    .cmdkey           = "csetpmove",
    .description      = "move current process to CPU set",
    .description_long = "Move processes to specific CPU sets for core pinning and isolation."
};

static CLICMDDATA CLIcmddata_cset = { "", "", CLICMD_FIELDS_NOPARAM };
FPS_CMDSETTINGS_INIT(cset, CLIcmddata_cset, FPS_app_info_cset)

static errno_t __attribute__((unused)) compute_cset()
{
    milkrt_CPUset(p_name);
    return RETURN_SUCCESS;
}

/* ================================================================
 *  CMD 5: csetandprioext (3 args, primary)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "csetandprioext",
    .cmdkey           = "csetandprioext",
    .description      = "move PID to CPU set and assign "
                        "RT priority",
    .description_long = "Move processes to specific CPU sets for core pinning and isolation."
};

#define FPS_PARAMS(X)                                                          \
    X(".pid", &p_pid, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "PID")            \
    X(".name", p_name, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "CPU set name") \
    X(".rtprio", &p_rtprio, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "RT priority (0=ignore)")

FPS_V2_SECTION5(FPS_PARAMS)
static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();
    INSERT_STD_PROCINFO_COMPUTEFUNC_START milkrt_CPUsetExt(p_pid, p_name, p_rtprio);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END   DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

/* ================================================================
 *  REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)

/* bindings for single-param commands */
static FPS_CLI_BINDING bindings_rtp[]  = { FPS_PARAMS_RTP(FPS_X_BINDING) };
static const int       nb_bindings_rtp = sizeof(bindings_rtp) / sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF    farg_rtp[]      = { FPS_PARAMS_RTP(FPS_X_FARG) };

static FPS_CLI_BINDING bindings_tset[]  = { FPS_PARAMS_TSET(FPS_X_BINDING) };
static const int       nb_bindings_tset = sizeof(bindings_tset) / sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF    farg_tset[]      = { FPS_PARAMS_TSET(FPS_X_FARG) };

static FPS_CLI_BINDING bindings_tsete[]  = { FPS_PARAMS_TSETE(FPS_X_BINDING) };
static const int       nb_bindings_tsete = sizeof(bindings_tsete) / sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF    farg_tsete[]      = { FPS_PARAMS_TSETE(FPS_X_FARG) };

/* csetpmove reuses TSET bindings (1 str) */

static errno_t CLIfunction_rtp(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info_rtp, farg_rtp, &CLIcmddata_rtp, bindings_rtp,
                                        nb_bindings_rtp, compute_rtp);
}

static errno_t CLIfunction_tset(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info_tset, farg_tset, &CLIcmddata_tset,
                                        bindings_tset, nb_bindings_tset, compute_tset);
}

static errno_t CLIfunction_tsete(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info_tsete, farg_tsete, &CLIcmddata_tsete,
                                        bindings_tsete, nb_bindings_tsete, compute_tsete);
}

static errno_t CLIfunction_cset(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info_cset, farg_tset, &CLIcmddata_cset,
                                        bindings_tset, nb_bindings_tset, compute_cset);
}

static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_COREMOD_tools__mvprocCPUset()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    safe_fps_fill_farg_examples(farg_rtp, bindings_rtp, nb_bindings_rtp);
    safe_fps_fill_farg_examples(farg_tset, bindings_tset, nb_bindings_tset);
    safe_fps_fill_farg_examples(farg_tsete, bindings_tsete, nb_bindings_tsete);

    {
        int cmdi                   = RegisterCLIcmd(CLIcmddata_rtp, CLIfunction_rtp);
        CLIcmddata_rtp.cmdsettings = &data.cmd[cmdi].cmdsettings;
    }
    {
        int cmdi                    = RegisterCLIcmd(CLIcmddata_tset, CLIfunction_tset);
        CLIcmddata_tset.cmdsettings = &data.cmd[cmdi].cmdsettings;
    }
    {
        int cmdi                     = RegisterCLIcmd(CLIcmddata_tsete, CLIfunction_tsete);
        CLIcmddata_tsete.cmdsettings = &data.cmd[cmdi].cmdsettings;
    }
    {
        int cmdi                    = RegisterCLIcmd(CLIcmddata_cset, CLIfunction_cset);
        CLIcmddata_cset.cmdsettings = &data.cmd[cmdi].cmdsettings;
    }
    {
        int cmdi               = RegisterCLIcmd(CLIcmddata, CLIfunction);
        CLIcmddata.cmdsettings = &data.cmd[cmdi].cmdsettings;
    }

    return RETURN_SUCCESS;
}
#endif
