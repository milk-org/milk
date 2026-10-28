// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    computeunit.h
 * @brief   libcomputeunit umbrella header: FPS runtime API and standalone macros
 */

#ifndef COMPUTEUNIT_H
#define COMPUTEUNIT_H

#include "fps.h"
#include "libmilkdata/milkdata.h"

#include "fps_cli_binding.h"
#include "fps_cli_init.h"
#include "fps_cli_sync.h"
#include "fps_modulevars_resync.h"
#include "fps_lifecycle.h"

#include "fps_FPCONFsetup.h"
#include "fps_FPCONFloopstep.h"
#include "fps_FPCONFexit.h"
#include "fps_RUNexit.h"
#include "fps_CONFstart.h"
#include "fps_CONFstop.h"
#include "fps_RUNstart.h"
#include "fps_RUNstop.h"

#include "fps_processinfo.h"
#include "fps_processinfo_entries.h"
#include "fps_unit.h"

/**
 * @brief Command-line driver of every fpsexec; implemented in core/fps_standalone_main.c.
 *
 * @return process exit code
 */
int fpsexec_main_function(int              argc,
                          char            *argv[],
                          FPS_APP_INFO    *APP_INFO,
                          FPS_CLI_BINDING  bindings[],
                          int              nb_bindings,
                          CLICMDARGDEF     farg[],
                          fps_compute_fn   COMPUTE_FN,
                          fps_confcheck_fn CONFCHECK_FN);


/**
 * @brief Standard initialization preamble for FPSINIT function
 */
#define FPS_INIT_STD_PREAMBLE(VARfps, VARfps_name, VARkeywords, VARdescription, VARhelptext) \
    (VARfps) = function_parameter_FPCONFsetup(VARfps_name, FPSCMDCODE_FPSINIT);              \
    strncpy((VARfps).md->sourcefname, __FILE__, FPS_SRCDIR_STRLENMAX - 1);                   \
    (VARfps).md->sourceline = __LINE__;                                                      \
    if ((VARkeywords) != NULL)                                                               \
    {                                                                                        \
        strncpy((VARfps).md->keywordarray, (VARkeywords), FPS_KEYWORDARRAY_STRMAXLEN - 1);   \
    }                                                                                        \
    if ((VARdescription) != NULL)                                                            \
    {                                                                                        \
        strncpy((VARfps).md->description, (VARdescription), FPS_DESCR_STRMAXLEN - 1);        \
    }                                                                                        \
    strncpy((VARfps).md->helptext, (VARhelptext), FPS_HELPTEXT_STRMAXLEN - 1);

/**
 * @brief Standard ProcessInfo default settings for FPSINIT
 */
#define FPS_INIT_PROCINFO_DEFAULTS(VARfps, VARtriggerstream, VARtimeout_sec)                     \
    strncpy((VARfps).cmdset.triggerstreamname, (VARtriggerstream), STRINGMAXLEN_IMAGE_NAME - 1); \
    (VARfps).cmdset.procinfo_loopcntMax    = -1;                                                 \
    (VARfps).cmdset.triggermode            = PROCESSINFO_TRIGGERMODE_SEMAPHORE;                  \
    (VARfps).cmdset.triggertimeout.tv_sec  = (VARtimeout_sec);                                   \
    (VARfps).cmdset.triggertimeout.tv_nsec = 0;

/**
 * @brief Standard connection and parameter mapping for FPSRUN
 */
#define FPS_RUN_STD_PREAMBLE(VARfps_name, VARfps, BLOCK_VAR_MAP)                     \
    if (fps_connect(VARfps_name, &(VARfps), FPSCONNECT_RUN) == -1)                   \
    {                                                                                \
        PRINT_ERROR("Error: FPS '%s' not found. Run 'fpsinit' first.", VARfps_name); \
        return 1;                                                                    \
    }                                                                                \
    BLOCK_VAR_MAP

/**
 * @brief Standard setup for ProcessInfo in FPSRUN
 */
#define FPS_RUN_PROCESSINFO_SETUP(VARprocessinfo, VARfps_name, VARdesc_short, VARdesc_detail, \
                                  VARinput_image, VARfps)                                     \
    VARprocessinfo = processinfo_setup((char *) VARfps_name, VARdesc_short, VARdesc_detail,   \
                                       __FUNCTION__, __FILE__, __LINE__);                     \
    if (!VARprocessinfo)                                                                      \
        return 1;                                                                             \
    processinfo_CatchSignals();                                                               \
    processinfo_waitoninputstream_init(VARprocessinfo, VARinput_image,                        \
                                       ((VARinput_image) != NULL)                             \
                                           ? PROCESSINFO_TRIGGERMODE_SEMAPHORE                \
                                           : PROCESSINFO_TRIGGERMODE_IMMEDIATE,               \
                                       -1);                                                   \
    fps_to_processinfo(&(VARfps), VARprocessinfo);                                            \
    processinfo_loopstart(VARprocessinfo);                                                    \
    fps_unit_notify_ready();

/**
 * @brief Standard loop for FPSRUN
 */
#define FPS_RUN_PROCESSINFO_LOOP(VARprocessinfo, VARfps, VARinput_image, VARoutput_image,  \
                                 BLOCK_COMPUTE)                                            \
    int loopOK = 1;                                                                        \
    while (loopOK)                                                                         \
    {                                                                                      \
        loopOK = processinfo_loopstep(VARprocessinfo);                                     \
        if (!loopOK)                                                                       \
            break;                                                                         \
        fps_unit_tick(VARprocessinfo->loopcnt);                                            \
        processinfo_waitoninputstream(VARprocessinfo);                                     \
        if (processinfo_exit_requested(VARprocessinfo))                                    \
            break;                                                                         \
        if (VARprocessinfo->triggerstatus == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT)           \
            continue;                                                                      \
        processinfo_exec_start(VARprocessinfo);                                            \
        BLOCK_COMPUTE                                                                      \
        processinfo_exec_end(VARprocessinfo);                                              \
        processinfo_update_output_stream(VARprocessinfo, VARoutput_image, VARinput_image); \
    }                                                                                      \
    processinfo_cleanExit(VARprocessinfo);                                                 \
    fps_disconnect(&(VARfps));

#endif /* COMPUTEUNIT_H */
