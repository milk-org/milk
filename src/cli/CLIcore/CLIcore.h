// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    CLIcore.h
 * @brief   Command line interface
 *
 * Command line interface (CLI) definitions and function prototypes
 *
 * @defgroup errcheckmacro     MACROS: Error checking
 * @defgroup debugmacro        MACROS: Debugging
 * @defgroup procinfomacro     MACROS: Process control
 *
 */


#ifndef _CLICORE_H
#define _CLICORE_H

/* When building without CLI (MILK_NO_CLI), redirect
 * to the standalone header that provides all types
 * and macros as stubs, without linking CLIcore.
 */
#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else /* full CLI mode */


#    include <stdio.h>
#    include <stdint.h>
#    include <stdarg.h>
#    include <string.h>
#    include <sys/types.h>

#    include "milk_config.h"

/* Core data structure (MILK_DATA) and macros */
#    include "libmilkdata/milkdata.h"

#    include "ImageStreamIO/ImageStreamIO.h"
#    include "ImageStreamIO/ImageStruct.h"

#    include <fps.h>
#    include <computeunit.h>
#    include <processtools.h>
#    include "timeutils.h"


#    include "CLIcore_checkargs.h"
#    include "CLIcore_help.h"
#    include "CLIcore_modules.h"

#    include "milkDebugTools.h"
#    include "milk_types.h"

// Initialize CLI
errno_t CLI_startup();

/// important directories and info
extern pid_t CLIPID;         // command line interface PID
extern char  DocDir[200];    // location of documentation
extern char  SrcDir[200];    // location of source
extern char  BuildFile[200]; // file name for source
extern char  BuildDate[200];
extern char  BuildTime[200];

extern int C_ERRNO; // C errno (from errno.h)

#    ifdef USE_NCURSES
errno_t functionparameter_CTRLscreen(uint32_t mode,
                                     char    *fpsnamemask,
                                     char    *fpsCTRLfifoname,
                                     double   timeout_sec);

errno_t processinfo_CTRLscreen();
#    else
/**
 * functionparameter_CTRLscreen - No-op FPS TUI stub when built without ncurses
 * @mode:            TUI display mode (unused)
 * @fpsnamemask:     FPS name filter mask (unused)
 * @fpsCTRLfifoname: Control FIFO path (unused)
 * @timeout_sec:     Refresh timeout in seconds (unused)
 *
 * Return: 0.
 */
static inline errno_t functionparameter_CTRLscreen(uint32_t mode,
                                                   char    *fpsnamemask,
                                                   char    *fpsCTRLfifoname,
                                                   double   timeout_sec)
{
    (void) mode;
    (void) fpsnamemask;
    (void) fpsCTRLfifoname;
    (void) timeout_sec;
    return 0;
}

/**
 * processinfo_CTRLscreen - No-op processinfo TUI stub when built without ncurses
 *
 * Return: 0.
 */
static inline errno_t processinfo_CTRLscreen()
{
    return 0;
}

/**
 * TUI_printfw - Fallback formatted stdout writer when built without ncurses
 * @fmt: `printf`-style format string
 */
static inline void TUI_printfw(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}

/**
 * TUI_newline - Emit a newline to stdout when built without ncurses
 */
static inline void TUI_newline()
{
    printf("\n");
}

/**
 * screenprint_setreverse - No-op reverse-video attribute setter without ncurses
 */
static inline void screenprint_setreverse()
{
}

/**
 * screenprint_unsetreverse - No-op reverse-video attribute resetter without ncurses
 */
static inline void screenprint_unsetreverse()
{
}

/**
 * screenprint_setcolor - No-op color-pair setter without ncurses
 * @pair: Color pair index (unused)
 */
static inline void screenprint_setcolor(int pair)
{
    (void) pair;
}

/**
 * screenprint_unsetcolor - No-op color-pair resetter without ncurses
 * @pair: Color pair index (unused)
 */
static inline void screenprint_unsetcolor(int pair)
{
    (void) pair;
}

/**
 * TUI_set_screenprintmode - No-op screen print mode selector without ncurses
 * @mode: Screen print mode (unused)
 */
static inline void TUI_set_screenprintmode(int mode)
{
    (void) mode;
}

/**
 * TUI_init_terminal - No-op terminal initializer without ncurses
 * @wrow: Output row count pointer (unused)
 * @wcol: Output column count pointer (unused)
 *
 * Return: 0.
 */
static inline errno_t TUI_init_terminal(short unsigned int *wrow, short unsigned int *wcol)
{
    (void) wrow;
    (void) wcol;
    return 0;
}

/**
 * get_singlechar_nonblock - Non-blocking character input stub without ncurses
 *
 * Return: Always -1 (no key available).
 */
static inline int get_singlechar_nonblock()
{
    return -1;
}

/**
 * TUI_exit - No-op terminal teardown stub without ncurses
 *
 * Return: 0.
 */
static inline errno_t TUI_exit()
{
    return 0;
}
#    endif


//
//  ************ lib module init **********************************
//

/* MODULE_DEPS, MILK_MODULE_INFO and MILK_MODULE_RUNNABLES (engine tier) */
#    include "milk_module_info.h"

#    define MAX_NB_FRAMENAME_CHAR 500
#    define MAX_NB_EXCLUSIONS 40

// declare a boolean type "BOOL"
// TRUE and FALSE improve code readability
//
typedef uint_fast8_t BOOL;
#    define FALSE 0
#    define TRUE 1

#    define DATA_NB_MAX_COMMAND 2000
#    define DATA_NB_MAX_MODULE 200

//Need to install process with setuid.  Then, so you aren't running privileged all the time do this:
extern uid_t euid_real;
extern uid_t euid_called;
extern uid_t suid;

/*^-----------------------------------------------------------------------------
| commands available through the CLI
+-----------------------------------------------------------------------------*/

#    define STRINGMAXLEN_MODULE_NAME 100
#    define STRINGMAXLEN_MODULE_SHORTNAME 50
#    define STRINGMAXLEN_MODULE_LOADNAME 500
#    define STRINGMAXLEN_MODULE_SOFILENAME 1000
#    define STRINGMAXLEN_MODULE_PACKAGENAME 50
#    define STRINGMAXLEN_MODULE_INFOSTRING 1000
#    define STRINGMAXLEN_MODULE_DATESTRING 20
#    define STRINGMAXLEN_MODULE_TIMESTRING 20

#    define MODULE_TYPE_UNUSED 0
#    define MODULE_TYPE_STARTUP 1
#    define MODULE_TYPE_CUSTOMLOAD 2

/** Maximum number of declared dependencies per module */
#    define MODULE_MAX_DEPS 16

typedef struct
{
    int type;

    char name[STRINGMAXLEN_MODULE_NAME]; // module name

    // short name. If non-empty, access functions as <shortname>.<functionname>
    char shortname[STRINGMAXLEN_MODULE_SHORTNAME];

    char loadname[STRINGMAXLEN_MODULE_LOADNAME];
    char sofilename[STRINGMAXLEN_MODULE_SOFILENAME];

    // package to which module belongs
    char package[STRINGMAXLEN_MODULE_PACKAGENAME];
    int  versionmajor; // package version
    int  versionminor;
    int  versionpatch;

    char info[STRINGMAXLEN_MODULE_INFOSTRING]; // short description

    char datestring[STRINGMAXLEN_MODULE_DATESTRING]; // Compilation date
    char timestring[STRINGMAXLEN_MODULE_TIMESTRING]; // Compilation time

    void *DLib_handle;

    /** Number of declared dependencies */
    int nbdep;
    /** Dependency load names (mload convention) */
    char depname[MODULE_MAX_DEPS][STRINGMAXLEN_MODULE_LOADNAME];

} MODULE;

#    define STRINGMAXLEN_CMD_KEY 100
#    define STRINGMAXLEN_CMD_INFO 1000
#    define STRINGMAXLEN_CMD_SYNTAX 1000
#    define STRINGMAXLEN_CMD_EXAMPLE 1000
#    define STRINGMAXLEN_CMD_CCALL 1000
#    define STRINGMAXLEN_CMD_SRCFILE 1000
typedef struct
{
    char key[STRINGMAXLEN_CMD_KEY]; // command keyword

    // module
    char module[STRINGMAXLEN_MODULE_NAME]; // module name
    // index of module to which command belongs
    // set to -1 if does not belong to any module
    long moduleindex;
    char srcfile[STRINGMAXLEN_CMD_SRCFILE]; // module source filename

    // command function pointer
    errno_t (*fp)();

    // descriptor this command was registered from (NULL for legacy commands)
    const struct MILK_RUNNABLE_ *runnable;

    char info[STRINGMAXLEN_CMD_INFO];       // short description/help
    char syntax[STRINGMAXLEN_CMD_SYNTAX];   // command syntax
    char example[STRINGMAXLEN_CMD_EXAMPLE]; // command example
    char Ccall[STRINGMAXLEN_CMD_CCALL];

    // command arguments and parameters
    int nbarg;   // Number of visible CLI arguments
    int nbparam; // Total number of parameters (visible + hidden)

    CLICMDARGDATA *argdata; // arguments and parameters to function

    const char *outvars; // calculator variables it sets for the caller (CLICMDDATA.outvars)

    // defines static function capabilities and behavior
    //uint64_t flags;

    // dynamic settings for function
    CMDSETTINGS cmdsettings;

    uint32_t callcount;
} CMD;

// The command line is parsed and

// cmdargtoken type
// 0 : unsolved
// 1 : floating point (double precision)
// 2 : long
// 3 : string
// 4 : existing image
// 5 : command

#    define CMDARGTOKEN_TYPE_UNSOLVED 0
#    define CMDARGTOKEN_TYPE_FLOAT 1
#    define CMDARGTOKEN_TYPE_LONG 2
#    define CMDARGTOKEN_TYPE_STRING 3
#    define CMDARGTOKEN_TYPE_EXISTINGIMAGE 4
#    define CMDARGTOKEN_TYPE_COMMAND 5
#    define CMDARGTOKEN_TYPE_RAWSTRING 6

#    define STRINGMAXLEN_CMDARGTOKEN_VAL 200
typedef struct
{
    int type;
    struct
    {
        double numf;
        long   numl;
        char   string[STRINGMAXLEN_CMDARGTOKEN_VAL];
    } val;
} CMDARGTOKEN;

extern uint8_t TYPESIZE[32];

/* VARIABLE, CODETESTPOINT, and related constants
 * are now defined in milkdata.h */


#    define CLI_MAX_ALIASES 128
#    define CLI_ALIAS_NAMELEN 64
#    define CLI_ALIAS_CMDLEN 512

/* Tab completion modes for data.CLImatchMode */
#    define CLICOMPLETIONMODE_COMMANDS 0
#    define CLICOMPLETIONMODE_IMAGES 1
#    define CLICOMPLETIONMODE_CMDARGS 2
#    define CLICOMPLETIONMODE_FILES 3
#    define CLICOMPLETIONMODE_FPSPARAMS 4
#    define CLICOMPLETIONMODE_VARS_FPS 5
#    define CLICOMPLETIONMODE_VARS_SEQ 6
#    define CLICOMPLETIONMODE_VARS_STREAM 7
#    define CLICOMPLETIONMODE_VARS_ENV 8
#    define CLICOMPLETIONMODE_VARS_PROC 9

void cli_menu_display_matches(char **matches, int num_matches, int max_length);
int  cli_custom_getc(FILE *stream);

typedef struct
{
    char name[CLI_ALIAS_NAMELEN];
    char cmd[CLI_ALIAS_CMDLEN];
} CLI_ALIAS;

/**
 * @brief Extended data structure for CLI programs
 *
 * MILK_DATA core contains all core fields (images,
 * FPS, signals, etc.). CLI-specific fields follow.
 * Standalone programs use milk_data directly and
 * never instantiate DATA.
 */
typedef struct
{
    /* Core data (shared with standalone programs) */
    MILK_DATA core;

    // LOGGING (CLI-specific)
    // =================================================

    int  CLIloopON;
    int  CLIlogON;
    char CLIlogname[STRINGMAXLEN_FULLFILENAME];

    // COMMAND LINE INTERFACE (CLI)
    // =================================================

    int      fifoON;
    int      fifofd;
    char     processname[STRINGMAXLEN_PROCESSNAME];
    char     processname0[STRINGMAXLEN_PROCESSNAME];
    int      processnameflag;
    char     fifoname[STRINGMAXLEN_FULLFILENAME];
    uint32_t NBcmd;

    CMD cmd[DATA_NB_MAX_COMMAND];

    char        CLIcmdline[STRINGMAXLEN_CLICMDLINE];
    int         CLIexecuteCMDready;
    int         CLImatchMode;
    int         parseerror;
    int         echo_input;
    int         autocomplete;
    int         autocomplete_history;
    int         autocomplete_arghint;
    int         autocomplete_fuzzy;
    int         syntax_highlight;
    int         syntax_diagnostics;
    int         auto_indent;
    int         show_match;
    int         print_cmd_timing;
    char        last_argument[200];
    long        cmdNBarg;
    CMDARGTOKEN cmdargtoken[NB_ARG_MAX];

    long    cmdindex;
    long    calctmp_imindex;
    int     CMDexecuted;
    errno_t CMDerrstatus;

    // SESSION IDENTITY
    // =================================================

    char            session_id[128];
    char            session_tty[64];
    struct timespec session_start;

    // MODULES
    // =================================================

    long NBmodule;

    MODULE module[DATA_NB_MAX_MODULE];

    long moduleindex;
    int  moduletype;
    char modulename[STRINGMAXLEN_MODULE_NAME];
    char moduleloadname[STRINGMAXLEN_MODULE_LOADNAME];
    char modulesofilename[STRINGMAXLEN_MODULE_SOFILENAME];
    char moduleshortname[STRINGMAXLEN_MODULE_SHORTNAME];
    char moduleshortname_default[STRINGMAXLEN_MODULE_SHORTNAME];
    char moduledatestring[STRINGMAXLEN_MODULE_DATESTRING];
    char moduletimestring[STRINGMAXLEN_MODULE_TIMESTRING];

    /** Transient dep count for module being registered */
    int module_nbdep;
    /** Transient dep names for module being registered */
    char module_depname[MODULE_MAX_DEPS][STRINGMAXLEN_MODULE_LOADNAME];

    // COMMAND ALIASES
    // =================================================

    int       NBalias;
    CLI_ALIAS alias[CLI_MAX_ALIASES];

} DATA;

#    ifdef MILK_BASH_CLIENT /* libmilkbash, the bash client: no engine state, no images of its own */
extern DATA data;
#        define MILK_LOCAL_IMAGES ((IMAGE *) NULL)
#        define MILK_NB_LOCAL_IMAGES 0L
#    else
__attribute__((weak)) DATA data = { .core = { 0 } };
#        define MILK_LOCAL_IMAGES (data.core.image)
#        define MILK_NB_LOCAL_IMAGES (data.core.NB_MAX_IMAGE)
#    endif

#    include "CLIcore_utils.h"

errno_t set_signal_catch();

void sig_handler(int signo);

errno_t runCLI(int argc, char *argv[], char *promptstring);

errno_t CLI_execute_line();

#endif /* !MILK_NO_CLI — full CLI mode */

#endif /* _CLICORE_H */
