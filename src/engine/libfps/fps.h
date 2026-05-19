#ifndef FPS_H
#define FPS_H

/* Type definitions, structs, constants, and flags */
#include "fps_types.h"

/* Additional dependencies not in fps_types.h */
#include "timeutils.h"
#include "processinfo_signals.h"
#include "processtools.h"

typedef long imageID;
typedef long variableID;

#include "IMGID.h"

int function_parameter_printlist(
    FPS_PARAM *funcparamarray,
    long      NBparamMAX);

#ifdef USE_NCURSES
errno_t functionparameter_CTRLscreen(
    uint32_t mode,
    char     *fpsnamemask,
    char     *fpsCTRLfifoname,
    double   timeout_sec);
#endif

FPS function_parameter_FPCONFsetup(
    const char *fpsname,
    uint32_t   mode);
FPS function_parameter_FPCONFsetup_sized(
    const char *fpsname,
    uint32_t   mode,
    long       NBparamMAX);
uint16_t function_parameter_FPCONFloopstep(
    FPS *fps);
uint16_t function_parameter_FPCONFexit(
    FPS *fps);
uint16_t function_parameter_RUNexit(
    FPS *fps);

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
#define COLORRESET     MH_RST
#endif
#ifndef COLORARGCLI
#define COLORARGCLI    MH_DFLT
#endif
#ifndef COLORARGnotCLI
#define COLORARGnotCLI MH_ARG
#endif
#ifndef COLORCOMMAND
#define COLORCOMMAND   MH_CMD
#endif
#ifndef COLORHEADER
#define COLORHEADER    MH_HDR
#endif
#ifndef COLOROPTION
#define COLOROPTION    MH_OPT
#endif
#ifndef COLORPRIMARY
#define COLORPRIMARY   MH_TITLE
#endif
#ifndef COLORNORMAL
#define COLORNORMAL    "\033[0;37m"
#define COLORERROR     MH_ERR
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
 * @brief Help-print expansion for the 6-arg binding format.
 */
/* ---- helper: fill type_str for V2 params ---- */
#define X_HELP_V2_FILL_TYPESTR_(type, ts) \
    do { \
        if (type == FPTYPE_INT32) \
            strcpy(ts, "INT32"); \
        else if (type == FPTYPE_UINT32) \
            strcpy(ts, "UINT32"); \
        else if (type == FPTYPE_INT64) \
            strcpy(ts, "INT64"); \
        else if (type == FPTYPE_UINT64) \
            strcpy(ts, "UINT64"); \
        else if (type == FPTYPE_FLOAT32) \
            strcpy(ts, "FLOAT32"); \
        else if (type == FPTYPE_FLOAT64) \
            strcpy(ts, "FLOAT64"); \
        else if (type == FPTYPE_ONOFF) \
            strcpy(ts, "ONOFF"); \
        else if (type == FPTYPE_STREAMNAME) \
            strcpy(ts, "STREAMNAME"); \
        else if (type == FPTYPE_FILENAME) \
            strcpy(ts, "FILENAME"); \
        else if (type == FPTYPE_FITSFILENAME) \
            strcpy(ts, "FITSFILE"); \
        else if (type == FPTYPE_EXECFILENAME) \
            strcpy(ts, "EXECFILE"); \
        else if (type == FPTYPE_DIRNAME) \
            strcpy(ts, "DIRNAME"); \
        else if (type == FPTYPE_FPSNAME) \
            strcpy(ts, "FPSNAME"); \
        else if (type == FPTYPE_PROCESS) \
            strcpy(ts, "PROCESS"); \
        else if (FPTYPE_IS_STRING(type)) \
            strcpy(ts, "STRING"); \
        else if (type == FPTYPE_PID) \
            strcpy(ts, "PID"); \
        else if (type == FPTYPE_TIMESPEC) \
            strcpy(ts, "TIMESPEC"); \
    } while (0)

/* ---- helper: fill val_str for V2 params ---- */
#define X_HELP_V2_FILL_VALSTR_(type, ptr, vs) \
    do { \
        if (type == FPTYPE_INT32) \
            sprintf(vs, "%d", *(int32_t*)ptr); \
        else if (type == FPTYPE_UINT32) \
            sprintf(vs, "%u", *(uint32_t*)ptr); \
        else if (type == FPTYPE_INT64) \
            sprintf(vs, "%ld", *(int64_t*)ptr); \
        else if (type == FPTYPE_UINT64) \
            sprintf(vs, "%lu", *(uint64_t*)ptr); \
        else if (type == FPTYPE_FLOAT32) \
            sprintf(vs, "%f", *(float*)ptr); \
        else if (type == FPTYPE_FLOAT64) \
            sprintf(vs, "%f", *(double*)ptr); \
        else if (type == FPTYPE_ONOFF) \
            sprintf(vs, "%s", \
                (*(int32_t*)ptr) \
                    ? "ON" : "OFF"); \
        else if (type == FPTYPE_PID) \
            sprintf(vs, "%d", \
                (int)*(pid_t*)ptr); \
        else if (type == FPTYPE_TIMESPEC) \
            sprintf(vs, "%ld.%09ld", \
                ((struct timespec*)ptr)->tv_sec,\
                ((struct timespec*)ptr) \
                    ->tv_nsec); \
        else if (FPTYPE_IS_STRING(type) || \
                 type == FPTYPE_STREAMNAME || \
                 type == FPTYPE_FILENAME || \
                 type == FPTYPE_FITSFILENAME || \
                 type == FPTYPE_EXECFILENAME || \
                 type == FPTYPE_DIRNAME || \
                 type == FPTYPE_FPSNAME || \
                 type == FPTYPE_PROCESS) \
            strncpy(vs, (char*)ptr, 63); \
    } while (0)

/**
 * @brief Measure column widths for 6-arg binding.
 */
#define X_HELP_MEASURE_V2(kw, ptr, type, \
                          is_primary, flag, desc) \
    { \
        const char *_kp = \
            (kw[0] == '.') ? &kw[1] : kw; \
        int _kl = (int) strlen(_kp); \
        if (_kl > col_kw_w) col_kw_w = _kl; \
        char _ts[20] = "???"; \
        X_HELP_V2_FILL_TYPESTR_(type, _ts); \
        int _tl = (int) strlen(_ts); \
        if (_tl > col_tp_w) col_tp_w = _tl; \
        char _vs[64] = ""; \
        X_HELP_V2_FILL_VALSTR_(type, ptr, _vs);\
        int _vl = (int) strlen(_vs); \
        if (_vl > col_df_w) col_df_w = _vl; \
    }

/**
 * @brief Help-print for the 6-arg binding format.
 *
 * Uses col_kw_w, col_tp_w, col_df_w (set by
 * X_HELP_MEASURE_V2) for dynamic column widths.
 */
#define X_HELP_PRINT_V2(kw, ptr, type, \
                         is_primary, flag, desc)\
    { \
        char cli_idx_str[8]; \
        char val_str[64] = ""; \
        char type_str[20] = "???"; \
        const char *disp_kw = \
            (kw[0] == '.') ? &kw[1] : kw; \
        if (is_primary) \
            sprintf(cli_idx_str, "%3d", \
                    CLIargcnt); \
        else \
            strcpy(cli_idx_str, " - "); \
        X_HELP_V2_FILL_TYPESTR_(type, type_str);\
        X_HELP_V2_FILL_VALSTR_(type, ptr, \
                               val_str); \
        const char *_trig_tag = ""; \
        const char *_trig_rst = ""; \
        if ((flag) & FPFLAG_TRIGGER_STREAM) { \
            if (show_help_color) { \
                _trig_tag = \
                    " \033[48;5;23m" \
                    "\033[38;2;80;220;220m" \
                    " [TRIGGER] " \
                    "\033[0m"; \
            } else { \
                _trig_tag = " [TRIGGER]"; \
            } \
        } \
        if (show_help_color) { \
            const char *_clr = is_primary \
                ? COLORPRIMARY : COLORARGnotCLI;\
            printf("  %s %s%-*s%s" \
                   " %-*s %-*s %s%s\n", \
                   cli_idx_str, \
                   _clr, col_kw_w, disp_kw, \
                   COLORRESET, \
                   col_tp_w, type_str, \
                   col_df_w, val_str, \
                   desc, _trig_tag); \
        } else { \
            printf("  %s %-*s %-*s" \
                   " %-*s %s%s\n", \
                   cli_idx_str, \
                   col_kw_w, disp_kw, \
                   col_tp_w, type_str, \
                   col_df_w, val_str, \
                   desc, _trig_tag); \
        } \
        (void) _trig_rst; \
        if (is_primary) CLIargcnt++; \
    }


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
#define FPS_V2_SECTION5(PARAMS_MACRO)               \
    static __attribute__((unused)) FPS_CLI_BINDING my_bindings[] = {         \
        PARAMS_MACRO(FPS_X_BINDING)                  \
    };                                               \
    static const int __attribute__((unused)) nb_bindings =                   \
        (int)(sizeof(my_bindings)                     \
            / sizeof(FPS_CLI_BINDING));              \
    static __attribute__((unused)) CLICMDARGDEF farg[] = {                   \
        PARAMS_MACRO(FPS_X_FARG)                     \
    };                                               \
    FPS_V2_CLICMDDATA_DECL_                          \
    static CMDSETTINGS default_cmdsettings_ = {0};   \
    static __attribute__((constructor))              \
    void init_cmdsettings_(void) {                   \
        strncpy(CLIcmddata.key,                      \
                FPS_app_info.cmdkey,                 \
                sizeof(CLIcmddata.key) - 1);         \
        strncpy(CLIcmddata.description,              \
                FPS_app_info.description,            \
                sizeof(CLIcmddata.description) - 1); \
        if (CLIcmddata.cmdsettings == NULL) {        \
            CLIcmddata.cmdsettings =                 \
                &default_cmdsettings_;               \
        }                                            \
    }

/**
 * @brief Helper: CLIcmddata declaration with
 * proper linkage based on FPS_STANDALONE.
 */
#ifdef FPS_STANDALONE
#define FPS_V2_CLICMDDATA_DECL_                      \
    CLICMDDATA CLIcmddata = {                        \
        "", "", CLICMD_FIELDS_DEFAULTS               \
    };
#else
#define FPS_V2_CLICMDDATA_DECL_                      \
    static CLICMDDATA CLIcmddata = {                 \
        "", "", CLICMD_FIELDS_DEFAULTS               \
    };
#endif


/*
 * ================================================================
 * MILK_EMBED_BUILD_TAG -- compile-time build metadata
 *
 * Embeds a sentinel string in every fpsexec binary
 * so milk-perfbench can detect PGO/LTO status at
 * runtime without needing debug symbols.
 *
 * Format (readable via `strings | grep MILK_BUILD`):
 *   \x1fMILK_BUILD:<flags>END
 *
 * where <flags> is a comma-separated list of:
 *   OPT=1        -- optimised (-O2/-O3)
 *   PGO=GENERATE -- pass-1 instrumented binary
 *   PGO=USE      -- pass-2 profile-optimised binary
 *   LTO=1        -- link-time optimisation enabled
 *   STATIC=1     -- static LTO archives used
 *
 * The \x1f (ASCII unit-separator) prefix ensures
 * the sentinel is not confused with other strings.
 * ================================================================
 */
#define MILK_EMBED_BUILD_TAG() \
    static const char \
        __attribute__((used, section(".rodata"))) \
        _milk_build_tag_[] = \
        "\x1fMILK_BUILD:" \
        "VER=1," \
        __DATE__ "T" __TIME__ "," \
        "CC=" __VERSION__ "," \
        "SRC=" __FILE__ "," \
        "BIN=" MILK_BUILD_BINNAME "," \
        "GCC=" \
        "ARCH=" MILK_BUILD_ARCH "," \
        "OPT=" MILK_BUILD_OPT_STR \
        MILK_BUILD_PGO_STR \
        MILK_BUILD_LTO_STR \
        "END"

/* Helper strings selected by cmake compile-time defines */
#if defined(__x86_64__) || defined(_M_X64)
# define MILK_BUILD_ARCH "x86_64"
#elif defined(__aarch64__)
# define MILK_BUILD_ARCH "aarch64"
#else
# define MILK_BUILD_ARCH "unknown"
#endif

#ifdef MILK_BUILD_OPT
# define MILK_BUILD_OPT_STR "3,"
#else
# define MILK_BUILD_OPT_STR "0,"
#endif

#ifdef MILK_BUILD_PGO_GENERATE
# define MILK_BUILD_PGO_STR "PGO=GENERATE,"
#elif defined(MILK_BUILD_PGO_USE)
# define MILK_BUILD_PGO_STR "PGO=USE,"
#else
# define MILK_BUILD_PGO_STR ""
#endif

#ifdef MILK_BUILD_LTO
# ifdef MILK_BUILD_STATIC
#  define MILK_BUILD_LTO_STR "LTO=STATIC,"
# else
#  define MILK_BUILD_LTO_STR "LTO=1,"
# endif
#else
# define MILK_BUILD_LTO_STR ""
#endif

/* MILK_BUILD_BINNAME is injected per-target by cmake */
#ifndef MILK_BUILD_BINNAME
# define MILK_BUILD_BINNAME "unknown"
#endif


#define _FPS_MAIN_STANDALONE_V2_IMPL( \
    APP_INFO, PARAMS_MACRO, COMPUTE_FN, \
    CONFCHECK_FN) \
int main(int argc, char *argv[]) { \
    MILK_EMBED_BUILD_TAG(); \
    /* Pre-getopt scan: -h1 and -h2 must be \
     * handled before anything else so -h1 is \
     * never split into -h + 1.              */ \
    for (int _i = 1; _i < argc; _i++) { \
        if (strcmp(argv[_i], "-h1") == 0 || \
            strcmp(argv[_i], \
                   "--help-oneline") == 0) { \
            printf("%s\n", \
                   (APP_INFO).description); \
            return 0; \
        } \
        if (strcmp(argv[_i], "-h2") == 0 || \
            strcmp(argv[_i], \
                   "--help-description") == 0) {\
            const char *_dl = \
                (APP_INFO).description_long; \
            printf("%s\n", \
                   _dl ? _dl \
                   : (APP_INFO).description); \
            return 0; \
        } \
    } \
    milk_data_init(); \
    extern void milkfps_set_image_array( \
        IMAGE *imarray, long nb_max); \
    milkfps_set_image_array( \
        milk_data.image, milk_data.NB_MAX_IMAGE); \
    fps_cli_set_standalone_args(argc, argv); \
    char fps_name[STRINGMAXLEN_FPS_NAME] = ""; \
    strncpy(fps_name, \
            (APP_INFO).fps_name, \
            STRINGMAXLEN_FPS_NAME - 1); \
    char arg_fps_name[STRINGMAXLEN_FPS_NAME] \
        = ""; \
    int use_tmux = 0; \
    int use_procinfo = 0; \
    if (getenv("MILK_FPSPROCINFO") != NULL) { \
        use_procinfo = 1; \
    } \
    int use_loop = 0; \
    double loop_delay = -1.0; \
    int show_help = 0; \
    int mh_color = 1; \
    char *command = NULL; \
    char *keywords = NULL; \
    char *description = NULL; \
    char *colon_pos = NULL; \
    FPS_CLI_BINDING my_bindings_[] = { \
        PARAMS_MACRO(FPS_X_BINDING) \
    }; \
    int nb_bindings_ = sizeof(my_bindings_) \
                     / sizeof(FPS_CLI_BINDING); \
    CLICMDARGDEF farg_[] = { \
        PARAMS_MACRO(FPS_X_FARG) \
    }; \
    (void) farg_; \
    for (int ii = 1; ii < argc; ii++) { \
        if (strcmp(argv[ii], "-h") == 0 || \
            strcmp(argv[ii], "--help") == 0) { \
            show_help = 1; \
        } else if ( \
            strcmp(argv[ii], "-hm") == 0 || \
            strcmp(argv[ii], \
                   "--help-mono") == 0) { \
            show_help = 1; \
            mh_color = 0; \
        } else if (strcmp(argv[ii], \
                   "-tmux") == 0) { \
            use_tmux = 1; \
        } else if (strcmp(argv[ii], \
                   "-procinfo") == 0 || \
            strcmp(argv[ii], \
                   "--procinfo") == 0) { \
            use_procinfo = 1; \
        } else if (strcmp(argv[ii], \
                   "-loops") == 0 || \
            strcmp(argv[ii], \
                   "--loops") == 0) { \
            use_loop = 1; \
            use_procinfo = 1; \
        } else if (strcmp(argv[ii], \
                    "-loopd") == 0 || \
            strcmp(argv[ii], \
                    "--loopd") == 0) { \
            use_loop = 2; \
            use_procinfo = 1; \
            if (ii + 1 < argc) { \
                char c0 = argv[ii + 1][0]; \
                if ((c0 >= '0' && c0 <= '9') \
                    || c0 == '.' || c0 == '-') {\
                    loop_delay = atof(argv[++ii]);\
                } else { \
                    loop_delay = 0.0; \
                } \
            } else { \
                loop_delay = 0.0; \
            } \
        } else if ((strcmp(argv[ii], "-k") == 0 ||\
            strcmp(argv[ii], \
                   "--keywords") == 0) \
            && ii + 1 < argc) { \
            keywords = argv[++ii]; \
        } else if ((strcmp(argv[ii], "-d") == 0 ||\
            strcmp(argv[ii], \
                   "--description") == 0) \
            && ii + 1 < argc) { \
            description = argv[++ii]; \
        } else if ((strcmp(argv[ii], "-n") == 0 ||\
            strcmp(argv[ii], \
                   "--name") == 0) \
            && ii + 1 < argc) { \
            strncpy(arg_fps_name, argv[++ii], \
                    STRINGMAXLEN_FPS_NAME - 1); \
        } else if (command == NULL) { \
            command = argv[ii]; \
            if ((colon_pos = strchr(command, \
                                    ':')) \
                != NULL) { \
                *colon_pos = '\0'; \
                strncpy(arg_fps_name, command, \
                    STRINGMAXLEN_FPS_NAME - 1); \
                command = colon_pos + 1; \
            } \
        } \
    } \
    if (command == NULL) { \
        command = "run"; \
    } else { \
        if (strcmp(command, "fpsinit") != 0 && \
            strcmp(command, "fps") != 0 && \
            strcmp(command, "fpslist") != 0 && \
            strcmp(command, "confstart") != 0 && \
            strcmp(command, "confstep") != 0 && \
            strcmp(command, "confstop") != 0 && \
            strcmp(command, "runstart") != 0 && \
            strcmp(command, "runstop") != 0 && \
            strcmp(command, "exec") != 0 && \
            strcmp(command, "set") != 0 && \
            strcmp(command, "run") != 0) { \
            fprintf(stderr, \
                    MH_ERR "Error:" MH_RST \
                    " '%s' is not a" \
                    " valid command. Run with" \
                    " -h for help.\n", \
                    command); \
            return 1; \
        } \
    } \
    if (strlen(arg_fps_name) > 0) { \
        strncpy(fps_name, arg_fps_name, \
                STRINGMAXLEN_FPS_NAME - 1); \
    } \
    (void)keywords; (void)description; \
    if (show_help || (argc < 2)) { \
        /* TTY auto-detect for -h */ \
        if (show_help && mh_color && \
            !isatty(STDOUT_FILENO)) { \
            mh_color = 0; \
        } \
        /* ---- BANNER ---- */ \
        milk_help_banner(argv[0], \
            (APP_INFO).description, mh_color); \
        /* ---- USAGE ---- */ \
        milk_help_section("Usage", mh_color); \
        printf("  %s %s %s%s\n\n", \
               argv[0], \
               MH(MH_OPT, "[options]"), \
               MH(MH_OPT, "[fpsname:]"), \
               MH(MH_CMD, "<command>")); \
        /* ---- DESCRIPTION ---- */ \
        { \
            const char *_dl = \
                (APP_INFO).description_long; \
            const char *_desc = \
                _dl ? _dl \
                : (APP_INFO).description; \
            milk_help_section("Description", \
                              mh_color); \
            printf("  %s\n\n", _desc); \
        } \
        /* ---- FPS NOTE ---- */ \
        if (mh_color) { \
            printf("  " MH_NOTE \
                   "This is a fpsexec command." \
                   " Command parameters map" \
                   " to FPS parameters." MH_RST \
                   "\n  Run " MH_CMD \
                   "milk-fpsexec-help" MH_RST \
                   " for detailed FPS" \
                   " framework info.\n\n"); \
        } else { \
            printf("  NOTE: This is a fpsexec" \
                   " command. Command" \
                   " parameters map to FPS" \
                   " parameters.\n" \
                   "  Run milk-fpsexec-help" \
                   " for detailed FPS" \
                   " framework info.\n\n"); \
        } \
        /* ---- OPTIONS ---- */ \
        milk_help_section("Options", mh_color); \
        printf("  %s         Show this help\n", \
               MH(MH_OPT, "-h, --help")); \
        printf("  %s              One-line" \
               " description\n", \
               MH(MH_OPT, "-h1")); \
        printf("  %s              Verbose" \
               " description\n", \
               MH(MH_OPT, "-h2")); \
        printf("  %s              Monochrome" \
               " help\n", \
               MH(MH_OPT, "-hm")); \
        printf("  %s            Run inside" \
               " tmux session\n", \
               MH(MH_OPT, "-tmux")); \
        printf("  %s        Enable" \
               " processinfo\n", \
               MH(MH_OPT, "-procinfo")); \
        printf("  %s           Infinite loop," \
               " semaphore trigger\n", \
               MH(MH_OPT, "-loops")); \
        printf("  %s %s      Infinite loop," \
               " delay trigger\n", \
               MH(MH_OPT, "-loopd"), \
               MH(MH_ARG, "SEC")); \
        printf("  %s %s        Set FPS" \
               " instance name\n\n", \
               MH(MH_OPT, "-n"), \
               MH(MH_ARG, "NAME")); \
        /* ---- COMMANDS ---- */ \
        milk_help_section("Commands", mh_color);\
        printf("  %s          Create" \
               " the FPS\n", \
               MH(MH_CMD, "fpsinit")); \
        printf("  %s              Print FPS" \
               " content\n", \
               MH(MH_CMD, "fps")); \
        printf("  %s          List matching" \
               " FPS instances\n", \
               MH(MH_CMD, "fpslist")); \
        printf("  %s        Configuration" \
               " loop\n", \
               MH(MH_CMD, "confstart")); \
        printf("  %s         Single config" \
               " step\n", \
               MH(MH_CMD, "confstep")); \
        printf("  %s         Stop config" \
               " loop\n", \
               MH(MH_CMD, "confstop")); \
        printf("  %s         Main processing" \
               " loop\n", \
               MH(MH_CMD, "runstart")); \
        printf("  %s          Stop" \
               " processing loop\n", \
               MH(MH_CMD, "runstop")); \
        printf("  %s %s      Set positional" \
               " args (. to skip)\n", \
               MH(MH_CMD, "set"), \
               MH(MH_OPT, "[args]")); \
        printf("  %s %s     Auto-init + set" \
               " args + run\n\n", \
               MH(MH_CMD, "exec"), \
               MH(MH_OPT, "[args]")); \
        /* ---- PARAMETERS ---- */ \
        int col_kw_w = 7; \
        int col_tp_w = 4; \
        int col_df_w = 7; \
        PARAMS_MACRO(X_HELP_MEASURE_V2) \
        milk_help_section("Parameters", \
                          mh_color); \
        printf("  %-3s %-*s %-*s %-*s %s\n", \
               "Idx", \
               col_kw_w, "Keyword", \
               col_tp_w, "Type", \
               col_df_w, "Default", \
               "Description"); \
        printf("  %-3s %-*s %-*s %-*s %s\n", \
               "---", \
               col_kw_w, "-------", \
               col_tp_w, "----", \
               col_df_w, "-------", \
               "-----------"); \
        int CLIargcnt = 0; \
        (void) CLIargcnt; \
        int show_help_color = mh_color; \
        (void) show_help_color; \
        PARAMS_MACRO(X_HELP_PRINT_V2) \
        printf("\n"); \
        /* ---- SEE ALSO ---- */ \
        { \
            const char *_sa[] = { \
                "milk-fpsexec-help:print detailed FPS framework info", \
                "milk-fpsCTRL:launch the FPS dashboard TUI", \
                "milk-fpsexec-list:list installed fpsexec commands" \
            }; \
            milk_help_see_also(_sa, 3, \
                               mh_color); \
        } \
        printf("\n"); \
        return 0; \
    } \
    if (command == NULL) { \
        fprintf(stderr, "Error: Missing " \
                "command argument.\n"); \
        return 1; \
    } \
    if (strcmp(command, "exec") != 0) \
        printf("FPS " COLORCOMMAND "%s" \
               COLORRESET " %s\n", \
               fps_name, command); \
    if (strcmp(command, "fps") == 0) { \
        FPS fps; \
        if (fps_connect( \
                fps_name, &fps, \
                FPSCONNECT_SIMPLE) == -1) { \
            fprintf(stderr, \
                    "Error: cannot connect to " \
                    "FPS '%s'.\n", fps_name); \
            return 1; \
        } \
        function_parameter_print_info( \
            &fps, 0, 0); \
        fps_disconnect( \
            &fps); \
        return 0; \
    } else if (strcmp(command, \
                      "fpslist") == 0) { \
        FPS *fpsarray = \
            (FPS *) \
            calloc(NB_FPS_MAX, \
                   sizeof( \
                   FPS));\
        if (fpsarray == NULL) return 1; \
        for (int ii = 0; ii < NB_FPS_MAX; ii++) \
            fpsarray[ii].SMfd = -1; \
        KEYWORD_TREE_NODE *keywnode = \
            (KEYWORD_TREE_NODE *) \
            calloc(NB_KEYWNODE_MAX, \
                   sizeof(KEYWORD_TREE_NODE)); \
        if (keywnode == NULL) { \
            free(fpsarray); return 1; } \
        int NBkwn = 0, NBfps = 0; \
        long NBpindex = 0; \
        functionparameter_scan_fps(0, "_ALL", \
            fpsarray, keywnode, \
            &NBkwn, &NBfps, &NBpindex, 0); \
        if (NBfps > 0) { \
            char *eb = strrchr(argv[0], '/'); \
            if (eb) eb++; else eb = argv[0]; \
            int found = 0; \
            for (int ii = 0; ii < NBfps; ii++) { \
                char *fb = strrchr( \
                    fpsarray[ii].md \
                        ->execfullpath, '/'); \
                if (fb) fb++; \
                else fb = fpsarray[ii].md \
                    ->execfullpath; \
                if (strcmp(eb, fb) == 0) { \
                    if (!found) { \
                        printf("%-30s %-10s " \
                               "%s\n", \
                               "FPS Name", \
                               "Status", \
                               "Description"); \
                        printf("----------" \
                               "----------" \
                               "----------" \
                               "----------\n");\
                        found = 1; \
                    } \
                    char ss[32] = "UNKNOWN"; \
                    if (fpsarray[ii].md->status \
                        & FUNCTION_PARAMETER_STRUCT_STATUS_CONF) \
                        strcpy(ss, "CONF"); \
                    else if (fpsarray[ii].md \
                        ->status \
                        & FUNCTION_PARAMETER_STRUCT_STATUS_RUN) \
                        strcpy(ss, "RUN"); \
                    printf("%-30s %-10s %s\n", \
                           fpsarray[ii].md->name,\
                           ss, \
                           fpsarray[ii].md \
                               ->description); \
                } \
                fps_disconnect( \
                    &fpsarray[ii]); \
            } \
            if (!found) \
                printf("No matching FPS for " \
                       "'%s'.\n", eb); \
        } else { \
            printf("No FPS found.\n"); \
        } \
        free(keywnode); \
        free(fpsarray); \
        return 0; \
    } \
    if (use_tmux) { \
        char path[1024]; \
        if (functionparameter_FPS_get_executable_path( \
                path, sizeof(path)) == NULL) { \
            if (realpath(argv[0], path) == NULL)\
                strncpy(path, argv[0], 1023); \
        } \
        char name_arg[256] = ""; \
        if (strcmp(fps_name, \
                  (APP_INFO).fps_name) != 0) { \
            snprintf(name_arg, \
                     sizeof(name_arg), \
                     " %s:%s", \
                     fps_name, command); \
        } else { \
            snprintf(name_arg, \
                     sizeof(name_arg), \
                     " %s", command); \
        } \
        functionparameter_FPS_tmux_standalone_setup( \
            fps_name); \
        if (strcmp(command, "exec") == 0) { \
            { \
                FPS fc_; \
                if (fps_name[0] != '_') { \
                    if (fps_connect( \
                            fps_name, &fc_, \
                            FPSCONNECT_SIMPLE) \
                        == -1) \
                    { \
                        printf("FPS " COLORCOMMAND \
                               "%s" COLORRESET \
                               " exec -> " \
                               "\033[33m" "NEW" \
                               COLORRESET "\n", \
                               fps_name); \
                        fps_generic_init(fps_name,\
                            (FPS_APP_INFO *) \
                            &(APP_INFO), \
                            my_bindings_, \
                            nb_bindings_, \
                            use_procinfo); \
                    } else { \
                        printf("FPS " COLORCOMMAND \
                               "%s" COLORRESET \
                               " exec -> " \
                               COLORCOMMAND \
                               "REUSE" \
                               COLORRESET "\n", \
                               fps_name); \
                        fps_disconnect( \
                            &fc_); \
                    } \
                } \
            } \
            /* Set CLI args into FPS before \
             * dispatching runstart to tmux */ \
            { \
                FPS fs_; \
                if (fps_connect(\
                        fps_name, &fs_, \
                        FPSCONNECT_SIMPLE) != -1) \
                { \
                    fps_process_cli_and_sync( \
                        &fs_, farg_, \
                        my_bindings_, \
                        nb_bindings_); \
                    if (use_loop == 1) { \
                        fps_loop_override_trigger(\
                            &fs_, my_bindings_, \
                            nb_bindings_); \
                    } else if (use_loop == 2) { \
                        fps_loop_override_delay(\
                            &fs_, loop_delay); \
                    } \
                    fps_disconnect(\
                        &fs_); \
                } \
            } \
            char run_arg[512] = ""; \
            if (use_procinfo) { \
                strncat(run_arg, " -procinfo", \
                        sizeof(run_arg) \
                        - strlen(run_arg) - 1); \
            } \
            if (use_loop == 1) { \
                strncat(run_arg, " -loops", \
                        sizeof(run_arg) \
                        - strlen(run_arg) - 1); \
            } else if (use_loop == 2) { \
                char _ld[64]; \
                snprintf(_ld, sizeof(_ld), \
                         " -loopd %.6f", \
                         loop_delay); \
                strncat(run_arg, _ld, \
                        sizeof(run_arg) \
                        - strlen(run_arg) - 1); \
            } \
            if (strcmp(fps_name, \
                      (APP_INFO).fps_name) != 0)\
            { \
                { size_t _l = strlen(run_arg); \
                  snprintf(run_arg + _l, \
                           sizeof(run_arg) - _l,\
                           " %s:runstart", \
                           fps_name); } \
            } else { \
                strncat(run_arg, " runstart", \
                        sizeof(run_arg) \
                        - strlen(run_arg) - 1); \
            } \
            functionparameter_FPS_tmux_send_dispatch( \
                fps_name, "runstart", path, \
                run_arg); \
            return 0; \
        } \
        if (functionparameter_FPS_tmux_send_dispatch( \
                fps_name, command, path, \
                name_arg) == 0) { \
            return 0; \
        } \
    } \
    if (strcmp(command, "fpsinit") == 0) { \
        int rc_ = fps_generic_init(fps_name, \
            (FPS_APP_INFO *)&(APP_INFO), \
            my_bindings_, nb_bindings_, \
            use_procinfo); \
        if (use_loop == 1 \
            && !fps_check_has_trigger_binding(\
                my_bindings_, \
                nb_bindings_)) { \
            fprintf(stderr, \
                "\033[1;33mWARNING" \
                "\033[0m [-loops] No" \
                " trigger stream" \
                " binding found \xe2\x80\x94" \
                " semaphore trigger" \
                " will not be" \
                " configured.\n" \
                "  Flag a stream" \
                " parameter with" \
                " FPFLAG_TRIGGER" \
                "_STREAM.\n"); \
        } \
        /* Sync CLI args into FPS before \
         * applying loop overrides. */ \
        if (use_loop && rc_ == 0) { \
            FPS fp_; \
            if (fps_connect(\
                    fps_name, &fp_, \
                    FPSCONNECT_SIMPLE) != -1) {\
                fps_process_cli_and_sync( \
                    &fp_, farg_, \
                    my_bindings_, nb_bindings_);\
                if (use_loop == 1) { \
                    fps_loop_override_trigger(\
                        &fp_, my_bindings_, \
                        nb_bindings_); \
                } else if (use_loop == 2) { \
                    fps_loop_override_delay( \
                        &fp_, loop_delay); \
                } \
                fps_disconnect(\
                    &fp_); \
            } \
        } \
        return rc_; \
    } else if (strcmp(command, \
                      "confstart") == 0) { \
        return fps_generic_conf_cb( \
            fps_name, 1, CONFCHECK_FN); \
    } else if (strcmp(command, \
                      "confstep") == 0) { \
        return fps_generic_conf_cb( \
            fps_name, 0, CONFCHECK_FN); \
    } else if (strcmp(command, \
                      "confstop") == 0) { \
        return fps_generic_confstop(fps_name); \
    } else if (strcmp(command, "set") == 0) { \
        FPS fps; \
        if (fps_connect( \
                fps_name, &fps, \
                FPSCONNECT_SIMPLE) == -1) { \
            fprintf(stderr, \
                    "Error: FPS '%s' not found." \
                    " Run fpsinit first.\n", \
                    fps_name); \
            return 1; \
        } \
        fps_process_cli_and_sync( \
            &fps, farg_, \
            my_bindings_, nb_bindings_); \
        fps_disconnect( \
            &fps); \
        printf("FPS " COLORCOMMAND "%s" \
               COLORRESET " set done\n", \
               fps_name); \
        return 0; \
    } else if (strcmp(command, "exec") == 0) { \
        /* Auto-init if FPS doesn't exist yet, \
         * then run. fps_name goes to shared mem \
         * when name lacks _ prefix. */ \
        { \
            FPS fps_chk_; \
            if (fps_name[0] != '_') { \
                if (fps_connect( \
                        fps_name, &fps_chk_, \
                        FPSCONNECT_SIMPLE) == -1) \
                { \
                    printf("FPS " COLORCOMMAND \
                           "%s" COLORRESET \
                           " exec -> " \
                           "\033[33m" "NEW" \
                           COLORRESET "\n", \
                           fps_name); \
                    fps_generic_init(fps_name, \
                        (FPS_APP_INFO *)&(APP_INFO), \
                        my_bindings_, nb_bindings_, \
                        use_procinfo); \
                } else { \
                    printf("FPS " COLORCOMMAND \
                           "%s" COLORRESET \
                           " exec -> " \
                           COLORCOMMAND "REUSE" \
                           COLORRESET "\n", \
                           fps_name); \
                    fps_disconnect( \
                        &fps_chk_); \
                } \
            } \
        } \
        /* Sync CLI args into FPS before \
         * applying loop overrides, so that \
         * trigger stream names are available. */ \
        { \
            FPS fps_sync_; \
            if (fps_connect(\
                    fps_name, &fps_sync_, \
                    FPSCONNECT_SIMPLE) != -1) { \
                fps_process_cli_and_sync( \
                    &fps_sync_, farg_, \
                    my_bindings_, nb_bindings_);\
                fps_disconnect(\
                    &fps_sync_); \
            } \
        } \
        /* Apply -loops/-loopd overrides */ \
        if (use_loop == 1) { \
            FPS fps_lp_; \
            if (fps_connect(\
                    fps_name, &fps_lp_, \
                    FPSCONNECT_SIMPLE) != -1) { \
                fps_loop_override_trigger( \
                    &fps_lp_, my_bindings_, \
                    nb_bindings_); \
                fps_disconnect(\
                    &fps_lp_); \
            } \
        } else if (use_loop == 2) { \
            FPS fps_lp_; \
            if (fps_connect(\
                    fps_name, &fps_lp_, \
                    FPSCONNECT_SIMPLE) != -1) { \
                fps_loop_override_delay( \
                    &fps_lp_, loop_delay); \
                fps_disconnect(\
                    &fps_lp_); \
            } \
        } \
        return fps_generic_run(fps_name, \
            (FPS_APP_INFO *)&(APP_INFO), \
            farg_, my_bindings_, nb_bindings_, \
            COMPUTE_FN); \
    } else if (strcmp(command, \
                      "runstart") == 0 || \
               strcmp(command, "run") == 0) { \
        /* Sync CLI args into FPS before \
         * applying loop overrides. */ \
        { \
            FPS fps_sync_; \
            if (fps_connect(\
                    fps_name, &fps_sync_, \
                    FPSCONNECT_SIMPLE) != -1) { \
                fps_process_cli_and_sync( \
                    &fps_sync_, farg_, \
                    my_bindings_, nb_bindings_);\
                fps_disconnect(\
                    &fps_sync_); \
            } \
        } \
        /* Apply -loops/-loopd overrides */ \
        if (use_loop == 1) { \
            FPS fps_lp_; \
            if (fps_connect(\
                    fps_name, &fps_lp_, \
                    FPSCONNECT_SIMPLE) != -1) { \
                fps_loop_override_trigger( \
                    &fps_lp_, my_bindings_, \
                    nb_bindings_); \
                fps_disconnect(\
                    &fps_lp_); \
            } \
        } else if (use_loop == 2) { \
            FPS fps_lp_; \
            if (fps_connect(\
                    fps_name, &fps_lp_, \
                    FPSCONNECT_SIMPLE) != -1) { \
                fps_loop_override_delay( \
                    &fps_lp_, loop_delay); \
                fps_disconnect(\
                    &fps_lp_); \
            } \
        } \
        return fps_generic_run(fps_name, \
            (FPS_APP_INFO *)&(APP_INFO), \
            farg_, my_bindings_, nb_bindings_, \
            COMPUTE_FN); \
    } else if (strcmp(command, \
                      "runstop") == 0) { \
        return fps_generic_runstop(fps_name); \
    } \
    fprintf(stderr, \
            "Invalid command: %s\n", command); \
    return 1; \
}


/**
 * @brief V2 standalone macro with confcheck.
 *
 * Usage:
 *   FPS_MAIN_STANDALONE_V2_CONFCHECK(
 *       app_info, PARAMS, compute_fn,
 *       customCONFcheck)
 */
#define FPS_MAIN_STANDALONE_V2_CONFCHECK( \
    APP_INFO, PARAMS_MACRO, COMPUTE_FN, \
    CONFCHECK_FN) \
    _FPS_MAIN_STANDALONE_V2_IMPL( \
        APP_INFO, PARAMS_MACRO, COMPUTE_FN, \
        CONFCHECK_FN)


/**
 * @brief V2 standalone macro (no confcheck).
 *
 * For backward compatibility with existing
 * callers.  Passes NULL as confcheck.
 */
#define FPS_MAIN_STANDALONE_V2( \
    APP_INFO, PARAMS_MACRO, COMPUTE_FN) \
    _FPS_MAIN_STANDALONE_V2_IMPL( \
        APP_INFO, PARAMS_MACRO, COMPUTE_FN, \
        NULL)

/**
 * @brief Standard initialization preamble for FPSINIT function
 */
#define FPS_INIT_STD_PREAMBLE(VARfps, VARfps_name, VARkeywords, VARdescription, VARhelptext) \
    (VARfps) = function_parameter_FPCONFsetup(VARfps_name, FPSCMDCODE_FPSINIT); \
    strncpy((VARfps).md->sourcefname, __FILE__, FPS_SRCDIR_STRLENMAX - 1); \
    (VARfps).md->sourceline = __LINE__; \
    if ((VARkeywords) != NULL) { \
        strncpy((VARfps).md->keywordarray, (VARkeywords), FPS_KEYWORDARRAY_STRMAXLEN - 1); \
    } \
    if ((VARdescription) != NULL) { \
        strncpy((VARfps).md->description, (VARdescription), FPS_DESCR_STRMAXLEN - 1); \
    } \
    strncpy((VARfps).md->helptext, (VARhelptext), FPS_HELPTEXT_STRMAXLEN - 1);

/**
 * @brief Standard ProcessInfo default settings for FPSINIT
 */
#define FPS_INIT_PROCINFO_DEFAULTS(VARfps, VARtriggerstream, VARtimeout_sec) \
    strncpy((VARfps).cmdset.triggerstreamname, (VARtriggerstream), STRINGMAXLEN_IMAGE_NAME - 1); \
    (VARfps).cmdset.procinfo_loopcntMax = -1; \
    (VARfps).cmdset.triggermode = PROCESSINFO_TRIGGERMODE_SEMAPHORE; \
    (VARfps).cmdset.triggertimeout.tv_sec = (VARtimeout_sec); \
    (VARfps).cmdset.triggertimeout.tv_nsec = 0;

/**
 * @brief Standard body for FPSCONF function
 *
 * @param VARfps_name Name of the FPS
 * @param VARloop Loop flag (1 for loop, 0 for single step)
 * @param BLOCK_VAR_MAP Code block to map parameters (e.g. { ptr = ...; })
 * @param BLOCK_VALIDATE Code block to validate parameters (e.g. { validate(); })
 */
#define FPS_CONF_STD_BODY(VARfps_name, VARloop, BLOCK_VAR_MAP, BLOCK_VALIDATE) \
    FPS fps; \
    if (VARloop) { \
        printf("Starting configuration process loop for '%s'\n", VARfps_name); \
        fps = function_parameter_FPCONFsetup(VARfps_name, FPSCMDCODE_CONFSTART); \
        BLOCK_VAR_MAP \
        while (fps.localstatus & FPS_LOCALSTATUS_CONFLOOP) { \
            if (function_parameter_FPCONFloopstep(&fps)) { \
                BLOCK_VALIDATE \
                functionparameter_CheckParametersAll(&fps); \
            } \
            usleep(10000); \
        } \
    } else { \
        printf("Running single configuration step for '%s'\n", VARfps_name); \
        fps = function_parameter_FPCONFsetup(VARfps_name, FPSCMDCODE_FPSINIT); \
        BLOCK_VAR_MAP \
        function_parameter_FPCONFloopstep(&fps); \
        BLOCK_VALIDATE \
        functionparameter_CheckParametersAll(&fps); \
    } \
    function_parameter_FPCONFexit(&fps);

/**
 * @brief Standard connection and parameter mapping for FPSRUN
 */
#define FPS_RUN_STD_PREAMBLE(VARfps_name, VARfps, BLOCK_VAR_MAP) \
    if (fps_connect(VARfps_name, &(VARfps), FPSCONNECT_RUN) == -1) { \
        PRINT_ERROR("Error: FPS '%s' not found. Run 'fpsinit' first.", VARfps_name); \
        return 1; \
    } \
    BLOCK_VAR_MAP

/**
 * @brief Standard setup for ProcessInfo in FPSRUN
 */
#define FPS_RUN_PROCESSINFO_SETUP(VARprocessinfo, VARfps_name, VARdesc_short, VARdesc_detail, VARinput_image, VARfps) \
    VARprocessinfo = processinfo_setup((char*)VARfps_name, VARdesc_short, VARdesc_detail, __FUNCTION__, __FILE__, __LINE__); \
    if (!VARprocessinfo) return 1; \
    processinfo_CatchSignals(); \
    processinfo_waitoninputstream_init(VARprocessinfo, VARinput_image, ((VARinput_image) != NULL) ? PROCESSINFO_TRIGGERMODE_SEMAPHORE : PROCESSINFO_TRIGGERMODE_IMMEDIATE, -1); \
    fps_to_processinfo(&(VARfps), VARprocessinfo); \
    processinfo_loopstart(VARprocessinfo);

/**
 * @brief Standard loop for FPSRUN
 */
#define FPS_RUN_PROCESSINFO_LOOP(VARprocessinfo, VARfps, VARinput_image, VARoutput_image, BLOCK_COMPUTE) \
    int loopOK = 1; \
    while(loopOK) { \
        loopOK = processinfo_loopstep(VARprocessinfo); \
        if(!loopOK) break; \
        processinfo_waitoninputstream(VARprocessinfo); \
        if (VARprocessinfo->triggerstatus == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT) continue; \
        processinfo_exec_start(VARprocessinfo); \
        BLOCK_COMPUTE \
        processinfo_exec_end(VARprocessinfo); \
        processinfo_update_output_stream(VARprocessinfo, VARoutput_image, VARinput_image); \
    } \
    processinfo_cleanExit(VARprocessinfo); \
    fps_disconnect(&(VARfps));

/** @} */ // end group fpsmacro

#endif // FPS_H
