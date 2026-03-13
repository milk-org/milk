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
#include "CLIcore_standalone.h"
#else /* full CLI mode */


#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#include <sys/types.h>

#include "config.h"

/* Core data structure (MILK_DATA) and macros */
#include "libmilkdata/milkdata.h"

#include "ImageStreamIO/ImageStreamIO.h"
#include "ImageStreamIO/ImageStruct.h"

#include <fps.h>
#include <processtools.h>
#include "timeutils.h"
#include "streamCTRL/streamCTRL_TUI.h"

#include "CLIcore_checkargs.h"
#include "CLIcore_help.h"
#include "CLIcore_modules.h"

#include "milkDebugTools.h"

#define PI 3.14159265358979323846264338328

/// Size of array CLICOREVARRAY
#define SZ_CLICOREVARRAY 1000

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

#ifdef USE_NCURSES
errno_t functionparameter_CTRLscreen(uint32_t mode,
                                     char    *fpsnamemask,
                                     char    *fpsCTRLfifoname,
                                     double  timeout_sec);

errno_t processinfo_CTRLscreen();
#else
static inline errno_t functionparameter_CTRLscreen(uint32_t mode, char *fpsnamemask, char *fpsCTRLfifoname, double timeout_sec) {
    (void)mode; (void)fpsnamemask; (void)fpsCTRLfifoname; (void)timeout_sec; return 0;
}
static inline errno_t processinfo_CTRLscreen() { return 0; }
static inline void TUI_printfw(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}
static inline void TUI_newline() { printf("\n"); }
static inline void screenprint_setreverse() {}
static inline void screenprint_unsetreverse() {}
static inline void screenprint_setcolor(int pair) { (void)pair; }
static inline void screenprint_unsetcolor(int pair) { (void)pair; }
static inline void TUI_set_screenprintmode(int mode) { (void)mode; }
static inline errno_t TUI_init_terminal(short unsigned int *wrow, short unsigned int *wcol) { (void)wrow; (void)wcol; return 0; }
static inline int get_singlechar_nonblock() { return -1; }
static inline errno_t TUI_exit() { return 0; }
#endif


#define STRINGMAXLEN_CLISTARTUPFILENAME 200

#define STRINGMAXLEN_CLIPROMPT 200


/* #define DEBUG */
#define CFITSEXIT                                                              \
    printf("Program abnormally terminated, File \"%s\", line %d\n",            \
           __FILE__,                                                           \
           __LINE__);                                                          \
    exit(0)

#ifdef DEBUG
#define nmalloc(f, type, n)                                                    \
    f = (type *) malloc(sizeof(type) * n);                                     \
    if (f == NULL)                                                             \
    {                                                                          \
        printf("ERROR: pointer \"" #f "\" allocation failed\n");               \
        exit(0);                                                               \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        printf("\nMALLOC: \"" #f "\" allocated\n");                            \
    }
#define nfree(f)                                                               \
    free(f);                                                                   \
    printf("\nMALLOC: \"" #f "\" freed\n");
#else
#define nmalloc(f, type, n)                                                    \
    f = (type *) malloc(sizeof(type) * n);                                     \
    if (f == NULL)                                                             \
    {                                                                          \
        printf("ERROR: pointer \"" #f "\" allocation failed\n");               \
        exit(0);                                                               \
    }
#define nfree(f) free(f);
#endif

#define TEST_ALLOC(f)                                                          \
    if (f == NULL)                                                             \
    {                                                                          \
        printf("ERROR: pointer \"" #f "\" allocation failed\n");               \
        exit(0);                                                               \
    }

#define NB_ARG_MAX 100

//
//  ************ lib module init **********************************
//

/** @brief Initialize module
 */
#define INIT_MODULE_LIB(modname)                                               \
    static errno_t init_module_CLI(); /* forward declaration */                \
    static int     INITSTATUS_##modname = 0;                                   \
    void __attribute__((constructor)) libinit_##modname()                      \
    {                                                                          \
        if (INITSTATUS_##modname == 0) /* only run once */                     \
        {                                                                      \
            strncpy(data.moduleshortname_default, MODULE_SHORTNAME_DEFAULT, STRINGMAXLEN_MODULE_SHORTNAME-1);    \
            strncpy(data.moduledatestring, __DATE__, STRINGMAXLEN_MODULE_DATESTRING-1);                           \
            strncpy(data.moduletimestring, __TIME__, STRINGMAXLEN_MODULE_TIMESTRING-1);                           \
            strncpy(data.modulename, (#modname), STRINGMAXLEN_MODULE_NAME);                               \
            RegisterModule(__FILE__,                                           \
                           PROJECT_NAME,                                       \
                           MODULE_DESCRIPTION,                                 \
                           VERSION_MAJOR,                                      \
                           VERSION_MINOR,                                      \
                           VERSION_PATCH);                                     \
            init_module_CLI();                                                 \
            INITSTATUS_##modname = 1;                                          \
            strncpy(data.modulename, "", STRINGMAXLEN_MODULE_NAME-1);              /* reset after use */    \
            strncpy(data.moduleshortname_default, "", STRINGMAXLEN_MODULE_SHORTNAME-1); /* reset after use */    \
            strncpy(data.moduleshortname, "", STRINGMAXLEN_MODULE_SHORTNAME-1);         /* reset after use */    \
        }                                                                      \
    }                                                                          \
    void __attribute__((destructor)) libclose_##modname()                      \
    {                                                                          \
        if (INITSTATUS_##modname == 1)                                         \
        {                                                                      \
        }                                                                      \
    }

#define MAX_NB_FRAMENAME_CHAR 500
#define MAX_NB_EXCLUSIONS     40

// declare a boolean type "BOOL"
// TRUE and FALSE improve code readability
//
typedef uint_fast8_t BOOL;
#define FALSE 0
#define TRUE  1

#define DATA_NB_MAX_COMMAND 2000
#define DATA_NB_MAX_MODULE  200

//Need to install process with setuid.  Then, so you aren't running privileged all the time do this:
extern uid_t euid_real;
extern uid_t euid_called;
extern uid_t suid;

/*^-----------------------------------------------------------------------------
| commands available through the CLI
+-----------------------------------------------------------------------------*/

#define STRINGMAXLEN_MODULE_NAME        100
#define STRINGMAXLEN_MODULE_SHORTNAME   50
#define STRINGMAXLEN_MODULE_LOADNAME    500
#define STRINGMAXLEN_MODULE_SOFILENAME  1000
#define STRINGMAXLEN_MODULE_PACKAGENAME 50
#define STRINGMAXLEN_MODULE_INFOSTRING  1000
#define STRINGMAXLEN_MODULE_DATESTRING  20
#define STRINGMAXLEN_MODULE_TIMESTRING  20

#define MODULE_TYPE_UNUSED     0
#define MODULE_TYPE_STARTUP    1
#define MODULE_TYPE_CUSTOMLOAD 2

typedef struct
{
    int type;

    char name[STRINGMAXLEN_MODULE_NAME]; // module name

    // short name. If non-empty, access functions as <shortname>.<functionname>
    char shortname[STRINGMAXLEN_MODULE_SHORTNAME];

    char loadname[STRINGMAXLEN_MODULE_LOADNAME];
    char sofilename[STRINGMAXLEN_MODULE_SOFILENAME];

    // package to which module belongs
    char package [STRINGMAXLEN_MODULE_PACKAGENAME];
    int versionmajor;                      // package version
    int versionminor;
    int versionpatch;

    char info[STRINGMAXLEN_MODULE_INFOSTRING]; // short description

    char datestring[STRINGMAXLEN_MODULE_DATESTRING]; // Compilation date
    char timestring[STRINGMAXLEN_MODULE_TIMESTRING]; // Compilation time

    void *DLib_handle;

} MODULE;

#define STRINGMAXLEN_CMD_KEY     100
#define STRINGMAXLEN_CMD_INFO    1000
#define STRINGMAXLEN_CMD_SYNTAX  1000
#define STRINGMAXLEN_CMD_EXAMPLE 1000
#define STRINGMAXLEN_CMD_CCALL   1000
#define STRINGMAXLEN_CMD_SRCFILE 1000
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

    char info[STRINGMAXLEN_CMD_INFO];       // short description/help
    char syntax[STRINGMAXLEN_CMD_SYNTAX];   // command syntax
    char example[STRINGMAXLEN_CMD_EXAMPLE]; // command example
    char Ccall[STRINGMAXLEN_CMD_CCALL];

    // command arguments and parameters
    int nbarg;   // Number of visible CLI arguments
    int nbparam; // Total number of parameters (visible + hidden)

    CLICMDARGDATA *argdata; // arguments and parameters to function

    // defines static function capabilities and behavior
    //uint64_t flags;

    // dynamic settings for function
    CMDSETTINGS cmdsettings;
} CMD;

// The command line is parsed and

// cmdargtoken type
// 0 : unsolved
// 1 : floating point (double precision)
// 2 : long
// 3 : string
// 4 : existing image
// 5 : command

#define CMDARGTOKEN_TYPE_UNSOLVED      0
#define CMDARGTOKEN_TYPE_FLOAT         1
#define CMDARGTOKEN_TYPE_LONG          2
#define CMDARGTOKEN_TYPE_STRING        3
#define CMDARGTOKEN_TYPE_EXISTINGIMAGE 4
#define CMDARGTOKEN_TYPE_COMMAND       5
#define CMDARGTOKEN_TYPE_RAWSTRING     6

#define STRINGMAXLEN_CMDARGTOKEN_VAL 200
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

    int          CLIloopON;
    int          CLIlogON;
    char         CLIlogname[STRINGMAXLEN_FULLFILENAME];

    // COMMAND LINE INTERFACE (CLI)
    // =================================================

    int      fifoON;
    char     processname[STRINGMAXLEN_PROCESSNAME];
    char     processname0[STRINGMAXLEN_PROCESSNAME];
    int      processnameflag;
    char     fifoname[STRINGMAXLEN_FULLFILENAME];
    uint32_t NBcmd;

    CMD cmd[DATA_NB_MAX_COMMAND];

    char CLIcmdline[STRINGMAXLEN_CLICMDLINE];
    int  CLIexecuteCMDready;
    int  CLImatchMode;
    int parseerror;
    int autocomplete;
    int autocomplete_history;
    int autocomplete_arghint;
    int autocomplete_fuzzy;
    long        cmdNBarg;
    CMDARGTOKEN cmdargtoken[NB_ARG_MAX];

    long cmdindex;
    long calctmp_imindex;
    int CMDexecuted;
    errno_t CMDerrstatus;

    // MODULES
    // =================================================

    long NBmodule;

    MODULE module[DATA_NB_MAX_MODULE];

    long moduleindex;
    int  moduletype;
    char modulename[STRINGMAXLEN_MODULE_NAME];
    char moduleloadname[STRINGMAXLEN_MODULE_LOADNAME];
    char modulesofilename[
        STRINGMAXLEN_MODULE_SOFILENAME];
    char moduleshortname[
        STRINGMAXLEN_MODULE_SHORTNAME];
    char moduleshortname_default[
        STRINGMAXLEN_MODULE_SHORTNAME];
    char moduledatestring[
        STRINGMAXLEN_MODULE_DATESTRING];
    char moduletimestring[
        STRINGMAXLEN_MODULE_TIMESTRING];

} DATA;

extern DATA data;

#include "CLIcore_utils.h"

errno_t set_signal_catch();

void sig_handler(int signo);

/*
errno_t RegisterModule(
    const char *restrict FileName,
    const char *restrict PackageName,
    const char *restrict InfoString,
    int versionmajor,
    int versionminor,
    int versionpatch
);

uint32_t RegisterCLIcommand(
    const char *restrict CLIkey,
    const char *restrict CLImodulesrc,
    errno_t (*CLIfptr)(),
    const char *restrict CLIinfo,
    const char *restrict CLIsyntax,
    const char *restrict CLIexample,
    const char *restrict CLICcall
);
*/

errno_t runCLItest(int argc, char *argv[], char *promptstring);

errno_t runCLI(int argc, char *argv[], char *promptstring);

errno_t CLI_execute_line();

errno_t write_process_log();

#endif /* !MILK_NO_CLI — full CLI mode */

#endif /* _CLICORE_H */
