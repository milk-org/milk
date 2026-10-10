// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FPS_H
#define FPS_H

#include <linux/limits.h> // For PATH_MAX (but maybe would belong in a global milk compilation header?) // For some reason this fails only for standalone execs, and even not all of them!!

/* Type definitions, structs, constants, and flags */
//#include "CLI"
#ifdef MILK_NO_CLI // The guard is necessary due to USE_CLI=ON conflict on milkdata.h
#    include "milkdata.h"
#endif

#include "fps_types.h"

/* Additional dependencies not in fps_types.h */
#include "timeutils.h"
#include "processinfo_signals.h"
#include "processtools.h"


typedef long imageID;
typedef long variableID;

#include "IMGID.h"

int function_parameter_printlist(FPS_PARAM *funcparamarray, long NBparamMAX);

#ifdef USE_NCURSES
errno_t functionparameter_CTRLscreen(uint32_t mode,
                                     char    *fpsnamemask,
                                     char    *fpsCTRLfifoname,
                                     double   timeout_sec);
#endif

FPS      function_parameter_FPCONFsetup(const char *fpsname, uint32_t mode);
FPS      function_parameter_FPCONFsetup_sized(const char *fpsname, uint32_t mode, long NBparamMAX);
uint16_t function_parameter_FPCONFloopstep(FPS *fps);
uint16_t function_parameter_FPCONFexit(FPS *fps);
uint16_t function_parameter_RUNexit(FPS *fps);


/* Core FPS operations (connect, params, entries) */
#include "fps_core.h"

/* Additional API sub-headers */
#include "fps_execFPScmd.h"
#include "fps_GetFileName.h"
#include "fps_getFPSargs.h"
#include "fps_load.h"
#include "fps_loadstream.h"
#include "fps_outlog.h"
// #include "fps_printlist.h" // Removed
#include "fps_save2disk.h"
#include "fps_scan.h"
#include "fps_shmdirname.h"
#include "fps_WriteParameterToDisk.h"

#include "fps_CONFstop.h"
#include "fps_RUNstop.h"
#include "fps_processinfo.h"
#include "fps_tmux.h"

#include "fps_processinfo_entries.h"

/* FPS-CLI unified framework (V2) */
#include "fps_cli_binding.h"
#include "fps_local_store.h"
#include "fps_cli_init.h"
#include "fps_cli_sync.h"
#include "fps_lifecycle.h"
#include "fps_cli_query.h"
#include "fps_cli_function.h"

/* Shared help-message color palette and flag init */
#include "milk_help.h"

// ===========================
// CONVENIENT MACROS FOR FPS
// ===========================

/** @defgroup fpsmacro          MACROS: Function parameter structure
 *
 * Frequently used function parameter structure (FPS) operations :
 * - Create / initialize FPS
 * - Add parameters to existing FPS
 *
 * @{
 */

/* Legacy color aliases -- map to milk_help.h palette.
 * Used by non-help code (e.g. runtime status lines).
 * New code should use MH_* macros instead.
 */
#ifndef COLORRESET
#    define COLORRESET MH_RST
#endif
#ifndef COLORARGCLI
#    define COLORARGCLI MH_DFLT
#endif
#ifndef COLORARGnotCLI
#    define COLORARGnotCLI MH_ARG
#endif
#ifndef COLORCOMMAND
#    define COLORCOMMAND MH_CMD
#endif
#ifndef COLORHEADER
#    define COLORHEADER MH_HDR
#endif
#ifndef COLOROPTION
#    define COLOROPTION MH_OPT
#endif
#ifndef COLORPRIMARY
#    define COLORPRIMARY MH_TITLE
#endif
#ifndef COLORNORMAL
#    define COLORNORMAL "\033[0;37m"
#    define COLORERROR MH_ERR
#endif

/*
 * ================================================================
 * FPS_MAIN_STANDALONE_V2 -- uses generic library functions
 *
 * Usage (new binding format):
 *     FPS_MAIN_STANDALONE_V2(
 *         FPS_app_info,
 *         MY_PARAMS,
 *         my_compute_function)
 *
 * The PARAMS_MACRO uses the 6-arg binding format:
 *     X(keyword, ptr, type, is_primary, fpflag, descr)
 * ================================================================
 */

/**
 * @brief Stamp out V2 section 5 boilerplate.
 *
 * Produces: my_bindings[], nb_bindings, farg[],
 * CLIcmddata, default_cmdsettings, and init_cmdsettings().
 *
 * Requires: FPS_app_info to be declared before this macro.
 *
 * Usage (replaces ~35 lines of copy-paste):
 * @code
 * FPS_V2_SECTION5(FPS_PARAMS)
 * @endcode
 */
#define FPS_V2_SECTION5(PARAMS_MACRO)                                                            \
    static                                                                                       \
        __attribute__((unused)) FPS_CLI_BINDING my_bindings[] = { PARAMS_MACRO(FPS_X_BINDING) }; \
    static const int __attribute__((unused))    nb_bindings =                                    \
        (int) (sizeof(my_bindings) / sizeof(FPS_CLI_BINDING));                                   \
    static __attribute__((unused)) CLICMDARGDEF farg[] = { PARAMS_MACRO(FPS_X_FARG) };           \
    FPS_V2_CLICMDDATA_DECL_                                                                      \
    static CMDSETTINGS                       default_cmdsettings_ = { 0 };                       \
    static __attribute__((constructor)) void init_cmdsettings_(void)                             \
    {                                                                                            \
        strncpy(CLIcmddata.key, FPS_app_info.cmdkey, sizeof(CLIcmddata.key) - 1);                \
        strncpy(CLIcmddata.description, FPS_app_info.description,                                \
                sizeof(CLIcmddata.description) - 1);                                             \
        if (CLIcmddata.cmdsettings == NULL)                                                      \
        {                                                                                        \
            CLIcmddata.cmdsettings = &default_cmdsettings_;                                      \
        }                                                                                        \
    }

/**
 * @brief Helper: CLIcmddata declaration
 */
#define FPS_V2_CLICMDDATA_DECL_ static CLICMDDATA CLIcmddata = { "", "", CLICMD_FIELDS_DEFAULTS };

#define _FPS_MAIN_STANDALONE_V2_IMPL(APP_INFO, PARAMS_MACRO, COMPUTE_FN, CONFCHECK_FN)      \
    int main(int argc, char *argv[])                                                        \
    {                                                                                       \
        MILK_EMBED_BUILD_TAG();                                                             \
        return fpsexec_main_function(argc, argv, &APP_INFO, my_bindings, nb_bindings, farg, \
                                     (COMPUTE_FN), (CONFCHECK_FN));                         \
    }

//#ifdef MILK_NO_CLI // The guard is necessary due to USE_CLI=ON conflict on milkdata.h
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
//#endif


/**
 * @brief V2 standalone macro with confcheck.
 *
 * Usage:
 *   FPS_MAIN_STANDALONE_V2_CONFCHECK(
 *       app_info, PARAMS, compute_fn,
 *       customCONFcheck)
 */
#define FPS_MAIN_STANDALONE_V2_CONFCHECK(APP_INFO, PARAMS_MACRO, COMPUTE_FN, CONFCHECK_FN) \
    _FPS_MAIN_STANDALONE_V2_IMPL(APP_INFO, PARAMS_MACRO, COMPUTE_FN, CONFCHECK_FN)


/**
 * @brief V2 standalone macro (no confcheck).
 *
 * For backward compatibility with existing
 * callers.  Passes NULL as confcheck.
 */
#define FPS_MAIN_STANDALONE_V2(APP_INFO, PARAMS_MACRO, COMPUTE_FN) \
    _FPS_MAIN_STANDALONE_V2_IMPL(APP_INFO, PARAMS_MACRO, COMPUTE_FN, NULL)

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
    processinfo_loopstart(VARprocessinfo);

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
        processinfo_waitoninputstream(VARprocessinfo);                                     \
        if (VARprocessinfo->triggerstatus == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT)           \
            continue;                                                                      \
        processinfo_exec_start(VARprocessinfo);                                            \
        BLOCK_COMPUTE                                                                      \
        processinfo_exec_end(VARprocessinfo);                                              \
        processinfo_update_output_stream(VARprocessinfo, VARoutput_image, VARinput_image); \
    }                                                                                      \
    processinfo_cleanExit(VARprocessinfo);                                                 \
    fps_disconnect(&(VARfps));

/** @} */ // end group fpsmacro

#endif // FPS_H
