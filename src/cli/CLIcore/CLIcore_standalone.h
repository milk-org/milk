/**
 * @file    CLIcore_standalone.h
 * @brief   Minimal CLI types for standalone builds
 *
 * Provides all types, macros, and stubs that
 * are needed to compile milk modules WITHOUT
 * linking against CLIcore. Used when building
 * with -DMILK_NO_CLI.
 *
 * Modules compiled with this header can provide
 * computation functions that standalone fpsexec
 * programs call, but CLI registration code
 * (INIT_MODULE_LIB, RegisterCLIcommand, etc.)
 * becomes no-op stubs.
 */

#ifndef CLICORE_STANDALONE_H
#define CLICORE_STANDALONE_H

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

/* CLIcore_checkargs.h provides types (CLICMDARGDEF,
 * CLICMDDATA, etc.) and guards its function decls
 * behind #ifndef MILK_NO_CLI, so no conflict with
 * our static inline stubs below.
 */
#include "CLIcore_checkargs.h"

#include "milkDebugTools.h"

#define PI 3.14159265358979323846264338328

#define SZ_CLICOREVARRAY 1000

/* =====================================
 * Stubs for CLI-only declarations
 * ===================================== */

static inline errno_t CLI_startup(void)
{
    return 0;
}

/* These are normally extern in CLIcore.h,
 * but standalone builds don't link CLIcore,
 * so we provide local storage. */
static pid_t CLIPID __attribute__((unused));
static char  DocDir[200] __attribute__((unused));
static char  SrcDir[200] __attribute__((unused));
static char  BuildFile[200] __attribute__((unused));
static char  BuildDate[200] __attribute__((unused));
static char  BuildTime[200] __attribute__((unused));
static int   C_ERRNO __attribute__((unused));
static uid_t euid_real __attribute__((unused));
static uid_t euid_called __attribute__((unused));
static uid_t suid __attribute__((unused));
static uint8_t TYPESIZE[32] __attribute__((unused));


/* =====================================
 * ncurses stubs (always stub in no-CLI)
 * ===================================== */

static inline errno_t
functionparameter_CTRLscreen(
    uint32_t mode __attribute__((unused)),
    char *fpsnamemask __attribute__((unused)),
    char *fpsCTRLfifoname __attribute__((unused)),
    double timeout_sec __attribute__((unused)))
{
    return 0;
}

static inline errno_t
processinfo_CTRLscreen(void)
{
    return 0;
}

static inline void
TUI_printfw(
    const char *fmt __attribute__((unused)),
    ...)
{
}

static inline void TUI_newline(void) {}
static inline void screenprint_setreverse(void) {}
static inline void screenprint_unsetreverse(void) {}

static inline void
screenprint_setcolor(int p __attribute__((unused)))
{
}

static inline void
screenprint_unsetcolor(int p __attribute__((unused)))
{
}

static inline void
TUI_set_screenprintmode(
    int m __attribute__((unused)))
{
}

static inline errno_t
TUI_init_terminal(
    short unsigned int *wrow __attribute__((unused)),
    short unsigned int *wcol __attribute__((unused)))
{
    return 0;
}

static inline int
get_singlechar_nonblock(void)
{
    return -1;
}

static inline errno_t TUI_exit(void) { return 0; }


/* =====================================
 * String length constants
 * ===================================== */

#define STRINGMAXLEN_CLISTARTUPFILENAME 200
#define STRINGMAXLEN_CLIPROMPT          200

#define CFITSEXIT                        \
    printf("Abnormal termination, "      \
           "File \"%s\", line %d\n",     \
           __FILE__, __LINE__);          \
    exit(0)

#ifdef DEBUG
#define nmalloc(f, type, n)              \
    f = (type *) malloc(sizeof(type)*n); \
    if (f == NULL) {                     \
        printf("ERROR: \"" #f            \
               "\" alloc failed\n");     \
        exit(0);                         \
    } else {                             \
        printf("\nMALLOC: \"" #f          \
               "\" allocated\n");        \
    }
#define nfree(f)                         \
    free(f);                             \
    printf("\nMALLOC: \"" #f             \
           "\" freed\n");
#else
#define nmalloc(f, type, n)              \
    f = (type *) malloc(sizeof(type)*n); \
    if (f == NULL) {                     \
        printf("ERROR: \"" #f            \
               "\" alloc failed\n");     \
        exit(0);                         \
    }
#define nfree(f) free(f);
#endif

#define TEST_ALLOC(f)                    \
    if (f == NULL) {                     \
        printf("ERROR: \"" #f            \
               "\" alloc failed\n");     \
        exit(0);                         \
    }

#define NB_ARG_MAX 100


/* =====================================
 * Module init — no-op in standalone
 * ===================================== */

#define INIT_MODULE_LIB(modname)            \
    static errno_t init_module_CLI(void);   \
    static int INITSTATUS_##modname = 0;


/* =====================================
 * Type definitions
 * ===================================== */

#define MAX_NB_FRAMENAME_CHAR 500
#define MAX_NB_EXCLUSIONS     40

typedef uint_fast8_t BOOL;
#define FALSE 0
#define TRUE  1

#define DATA_NB_MAX_COMMAND 2000
#define DATA_NB_MAX_MODULE  200

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
    int  type;
    char name[STRINGMAXLEN_MODULE_NAME];
    char shortname[STRINGMAXLEN_MODULE_SHORTNAME];
    char loadname[STRINGMAXLEN_MODULE_LOADNAME];
    char sofilename[
        STRINGMAXLEN_MODULE_SOFILENAME];
    char package[
        STRINGMAXLEN_MODULE_PACKAGENAME];
    int  versionmajor;
    int  versionminor;
    int  versionpatch;
    char info[STRINGMAXLEN_MODULE_INFOSTRING];
    char datestring[
        STRINGMAXLEN_MODULE_DATESTRING];
    char timestring[
        STRINGMAXLEN_MODULE_TIMESTRING];
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
    char key[STRINGMAXLEN_CMD_KEY];
    char module[STRINGMAXLEN_MODULE_NAME];
    long moduleindex;
    char srcfile[STRINGMAXLEN_CMD_SRCFILE];
    errno_t (*fp)();
    char info[STRINGMAXLEN_CMD_INFO];
    char syntax[STRINGMAXLEN_CMD_SYNTAX];
    char example[STRINGMAXLEN_CMD_EXAMPLE];
    char Ccall[STRINGMAXLEN_CMD_CCALL];
    int  nbarg;
    int  nbparam;
    CLICMDARGDATA *argdata;
    CMDSETTINGS    cmdsettings;
} CMD;


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


/* =====================================
 * DATA struct (CLI-extended MILK_DATA)
 * ===================================== */

typedef struct
{
    MILK_DATA core;

    int  CLIloopON;
    int  CLIlogON;
    char CLIlogname[STRINGMAXLEN_FULLFILENAME];

    int      fifoON;
    int      fifofd;
    char     processname[STRINGMAXLEN_PROCESSNAME];
    char     processname0[STRINGMAXLEN_PROCESSNAME];
    int      processnameflag;
    char     fifoname[STRINGMAXLEN_FULLFILENAME];
    uint32_t NBcmd;

    CMD cmd[DATA_NB_MAX_COMMAND];

    char CLIcmdline[STRINGMAXLEN_CLICMDLINE];
    int  CLIexecuteCMDready;
    int  CLImatchMode;
    int  parseerror;
    int  autocomplete;
    int  autocomplete_history;
    int  autocomplete_arghint;
    int  autocomplete_fuzzy;
    long        cmdNBarg;
    CMDARGTOKEN cmdargtoken[NB_ARG_MAX];

    long    cmdindex;
    long    calctmp_imindex;
    int     CMDexecuted;
    errno_t CMDerrstatus;

    // SESSION IDENTITY
    char            session_id[64];
    char            session_tty[64];
    struct timespec session_start;

    long    NBmodule;
    MODULE  module[DATA_NB_MAX_MODULE];

    long moduleindex;
    int  moduletype;
    char modulename[STRINGMAXLEN_MODULE_NAME];
    char moduleloadname[
        STRINGMAXLEN_MODULE_LOADNAME];
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


/* Global DATA instance — standalone provides
 * its own storage in fps_standalone_data.c */
extern DATA data;


/* =====================================
 * CLI function stubs (no-op)
 * ===================================== */

static inline int
CLI_checkarg(
    int argnum __attribute__((unused)),
    uint32_t argtype __attribute__((unused)))
{
    return 1; /* always "fail" — prevents
                 legacy CLI wrappers from
                 executing in standalone */
}

static inline int
CLI_checkarg_noerrmsg(
    int argnum __attribute__((unused)),
    uint32_t argtype __attribute__((unused)))
{
    return 1;
}

static inline errno_t
CLI_checkarg_array(
    CLICMDARGDEF *fca __attribute__((unused)),
    int nbarg __attribute__((unused)))
{
    return 1;
}

static inline int
CLIargs_to_FPSparams_setval(
    CLICMDARGDEF *fca __attribute__((unused)),
    int n __attribute__((unused)),
    FUNCTION_PARAMETER_STRUCT *fps
    __attribute__((unused)))
{
    return 0;
}

static inline int
CMDargs_to_FPSparams_create(
    FUNCTION_PARAMETER_STRUCT *fps
    __attribute__((unused)))
{
    return 0;
}

static inline void *
get_farg_ptr(
    char *tag __attribute__((unused)),
    long *fpsi __attribute__((unused)))
{
    return NULL;
}

static inline errno_t
set_signal_catch(void)
{
    return 0;
}

static inline void
sig_handler(int signo __attribute__((unused)))
{
}

static inline errno_t
RegisterModule(
    const char *f __attribute__((unused)),
    const char *p __attribute__((unused)),
    const char *i __attribute__((unused)),
    int ma __attribute__((unused)),
    int mi __attribute__((unused)),
    int pa __attribute__((unused)))
{
    return 0;
}

static inline uint32_t
RegisterCLIcmd(
    CLICMDDATA cd __attribute__((unused)),
    errno_t (*fp)(void) __attribute__((unused)))
{
    return 0;
}

static inline uint32_t
RegisterCLIcommand(
    const char *k __attribute__((unused)),
    const char *s __attribute__((unused)),
    errno_t (*fp)() __attribute__((unused)),
    const char *i __attribute__((unused)),
    const char *sy __attribute__((unused)),
    const char *e __attribute__((unused)),
    const char *c __attribute__((unused)))
{
    return 0;
}


/* =====================================
 * CLIcore_utils.h macros subset
 *
 * These macros are used by computation
 * files and must be available in
 * standalone builds.
 * ===================================== */

#ifdef __cplusplus
typedef const char *CONST_WORD;
#else
typedef const char *__restrict CONST_WORD;
#endif

#include "CLIcore/IMGID.h"
#include "COREMOD_memory/COREMOD_memory.h"

#define CLICMD_FIELDS_FPSPROC          \
    __FILE__,                          \
    sizeof(farg) / sizeof(CLICMDARGDEF), \
    farg,                              \
    CLICMDFLAG_FPS | CLICMDFLAG_PROCINFO, \
    NULL, NULL, NULL

#define CLICMD_FIELDS_DEFAULTS         \
    __FILE__,                          \
    sizeof(farg) / sizeof(CLICMDARGDEF), \
    farg, CLICMDFLAG_FPS,              \
    NULL, NULL, NULL

#define CLICMD_FIELDS_NOFPS            \
    __FILE__,                          \
    sizeof(farg) / sizeof(CLICMDARGDEF), \
    farg, 0, NULL, NULL, NULL

#define CLICMD_FIELDS_NOPARAM          \
    __FILE__, 0, NULL,                 \
    CLICMDFLAG_FPS, NULL, NULL, NULL

#define RETURN_CLICHECKARGARRAY_SUCCESS      0
#define RETURN_CLICHECKARGARRAY_FAILURE      1
#define RETURN_CLICHECKARGARRAY_FUNCPARAMSET 2
#define RETURN_CLICHECKARGARRAY_HELP         3

#define HELPDETAILSSTRINGSTART \
    "------- DETAILS ------"
#define HELPDETAILSSTRINGEND \
    "-------- END ---------"


typedef struct
{
    char *name;
} LOCVAR_INIMG;

#define FARG_INPUTIM(imkey) \
    {                       \
        CLIARG_STR,         \
        "." #imkey ".name", \
        "input image",      \
        #imkey,             \
        (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT), \
        (void **) &imkey.name  \
    }


typedef struct
{
    char     *name;
    uint32_t *xsize;
    uint32_t *ysize;
    uint32_t *datatype;
    uint32_t *shared;
    uint32_t *NBkw;
    uint32_t *CBsize;
} LOCVAR_OUTIMG2D;

#define FARG_OUTIM_NAME(imkey)     \
    {CLIARG_STR,                   \
     "." #imkey ".name",           \
     "output image",               \
     #imkey,                       \
     (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT), \
     (void **) &imkey.name,        \
     NULL}

#define FARG_OUTIM_XSIZE(imkey)    \
    {CLIARG_UINT32,                \
     "." #imkey ".xsize",          \
     "x size", "256",              \
     (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT), \
     (void **) &imkey.xsize,       \
     NULL}

#define FARG_OUTIM_YSIZE(imkey)    \
    {CLIARG_UINT32,                \
     "." #imkey ".ysize",          \
     "y size", "256",              \
     (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT), \
     (void **) &imkey.ysize,       \
     NULL}

#define FARG_OUTIM_SHARED(imkey)   \
    {CLIARG_UINT32,                \
     "." #imkey ".shared",         \
     "shared flag", "0",           \
     FPFLAG_DEFAULT_INPUT,         \
     (void **) &imkey.shared,      \
     NULL}

#define FARG_OUTIM_NBKW(imkey)     \
    {CLIARG_UINT32,                \
     "." #imkey ".NBkw",           \
     "number keywords", "10",      \
     FPFLAG_DEFAULT_INPUT,         \
     (void **) &imkey.NBkw,        \
     NULL}

#define FARG_OUTIM_CBSIZE(imkey)   \
    {CLIARG_UINT32,                \
     "." #imkey ".CBsize",         \
     "circ buffer size", "0",      \
     FPFLAG_DEFAULT_INPUT,         \
     (void **) &imkey.CBsize,      \
     NULL}

#define FARG_OUTIM2D(imkey)        \
    FARG_OUTIM_NAME(imkey),        \
    FARG_OUTIM_XSIZE(imkey),       \
    FARG_OUTIM_YSIZE(imkey),       \
    FARG_OUTIM_SHARED(imkey),      \
    FARG_OUTIM_NBKW(imkey),        \
    FARG_OUTIM_CBSIZE(imkey)

#define FARG_OUTIM2DCREATE(imkey, img, dt) \
    IMGID img = imgid_make_from_name(    \
        imkey.name);                     \
    img.mdt->shared = *imkey.shared;     \
    img.mdt->NBkw   = *imkey.NBkw;      \
    img.mdt->CBsize = *imkey.CBsize;     \
    if (*imkey.shared == 1) {            \
        img = stream_connect_create_2D(  \
            imkey.name,                  \
            *imkey.xsize,                \
            *imkey.ysize, dt);           \
    } else {                             \
        img.mdt->naxis   = 2;            \
        img.mdt->size[0] = *imkey.xsize; \
        img.mdt->size[1] = *imkey.ysize; \
        img.mdt->datatype = dt;          \
        createimagefromIMGID(&img);       \
    }                                    \
    imcreateIMGID(&img);


#define STD_FARG_LINKfunction            \
    for (int argi = 0;                   \
         argi < (int)(sizeof(farg) /     \
             sizeof(CLICMDARGDEF));      \
         argi++)                         \
    {                                    \
        long  fpsi = -1;                 \
        void *ptr  = get_farg_ptr(       \
            farg[argi].fpstag, &fpsi);   \
        *(farg[argi].valptr) = ptr;      \
        if (farg[argi].indexptr != NULL)  \
        {                                \
            *(farg[argi].indexptr) = fpsi;\
        }                                \
    }


/* INSERT_STD_* macros — stub versions
 * for standalone compilation. The real
 * versions are in CLIcore_utils.h and
 * reference CLI functions. */

#define INSERT_STD_CLIfunction            \
    static errno_t __attribute__((unused)) CLIfunction(void)     \
    {                                    \
        (void) farg;                     \
        (void) CLIcmddata;               \
        return RETURN_SUCCESS;           \
    }

#define INSERT_STD_FPSCONFfunction        \
    static errno_t FPSCONFfunction(void) \
    { return RETURN_SUCCESS; }

#define INSERT_STD_FPSRUNfunction         \
    static errno_t FPSRUNfunction(void)  \
    { return RETURN_SUCCESS; }

#define INSERT_STD_FPSCLIfunction         \
    /* already provided by CLIfunction */

#define INSERT_STD_FPSCLIfunctions        \
    INSERT_STD_FPSCONFfunction            \
    INSERT_STD_FPSRUNfunction             \
    INSERT_STD_CLIfunction

#define INSERT_STD_FPSCONFfunction_DynamicSize \
    INSERT_STD_FPSCONFfunction

#define INSERT_STD_FPSCLIfunctions_DynamicSize \
    INSERT_STD_FPSCLIfunctions

#define INSERT_STD_CLIREGISTERFUNC {}

/* Process info macros — these are used by
 * standalone code so we provide the real
 * versions via CLIcore_utils.h-compatible
 * macros that reference CLIcmddata. */

#define INSERT_STD_PROCINFO_COMPUTEFUNC_INIT \
    int          processloopOK = 1;      \
    PROCESSINFO *processinfo   = NULL;   \
    CLIcmddata.cmdsettings->            \
        triggertimeout.tv_sec  = 2;      \
    CLIcmddata.cmdsettings->            \
        triggertimeout.tv_nsec = 0;      \
    if (dcfpsptr != NULL)               \
    {                                    \
        CLIcmddata.cmdsettings->        \
            RT_priority =                \
            dcfpsptr->cmdset.RT_priority;\
        CLIcmddata.cmdsettings->        \
            procinfo_loopcntMax =        \
            dcfpsptr->cmdset             \
                .procinfo_loopcntMax;    \
        CLIcmddata.cmdsettings->        \
            triggermode =                \
            dcfpsptr->cmdset             \
                .triggermode;            \
        strncpy(CLIcmddata.cmdsettings  \
                    ->triggerstreamname, \
                dcfpsptr->cmdset         \
                    .triggerstreamname,  \
                STRINGMAXLEN_IMAGE_NAME  \
                    - 1);                \
        CLIcmddata.cmdsettings->        \
            semindexrequested =          \
            dcfpsptr->cmdset             \
                .semindexrequested;      \
        CLIcmddata.cmdsettings->        \
            triggerdelay.tv_sec =        \
            dcfpsptr->cmdset             \
                .triggerdelay.tv_sec;    \
        CLIcmddata.cmdsettings->        \
            triggerdelay.tv_nsec =       \
            dcfpsptr->cmdset             \
                .triggerdelay.tv_nsec;   \
        CLIcmddata.cmdsettings->        \
            triggertimeout.tv_sec =      \
            dcfpsptr->cmdset             \
                .triggertimeout.tv_sec;  \
        CLIcmddata.cmdsettings->        \
            triggertimeout.tv_nsec =     \
            dcfpsptr->cmdset             \
                .triggertimeout.tv_nsec; \
    }                                    \
    if (CLIcmddata.cmdsettings->flags   \
        & CLICMDFLAG_PROCINFO)           \
    {                                    \
        char pinfodescr[200];            \
        int slen = snprintf(pinfodescr,  \
            200, "function %.10s",       \
            CLIcmddata.key);             \
        if (slen < 1 || slen >= 200) {   \
            abort();                     \
        }                                \
        if (dcfpsptr != NULL) {         \
            const char *_piname =        \
                (dcfpsname[0] != '\0')   \
                ? dcfpsname              \
                : dcfpsptr->md->name;    \
            processinfo =                \
                processinfo_setup(       \
                    (char*)_piname,      \
                    pinfodescr,          \
                    "startup",           \
                    __FUNCTION__,        \
                    __FILE__, __LINE__); \
            fps_to_processinfo(          \
                dcfpsptr,               \
                processinfo);            \
        } else {                         \
            processinfo =                \
                processinfo_setup(       \
                    CLIcmddata.key,      \
                    pinfodescr,          \
                    "startup",           \
                    __FUNCTION__,        \
                    __FILE__, __LINE__); \
        }                                \
        processinfo->loopcntMax =        \
            CLIcmddata.cmdsettings      \
                ->procinfo_loopcntMax;   \
        processinfo->triggerstreamID     \
            = -2;                        \
        processinfo->triggermode =       \
            CLIcmddata.cmdsettings      \
                ->triggermode;           \
        strncpy(processinfo              \
                    ->triggerstreamname, \
                CLIcmddata.cmdsettings  \
                    ->triggerstreamname, \
                STRINGMAXLEN_IMAGE_NAME  \
                    - 1);                \
        processinfo->triggerdelay =      \
            CLIcmddata.cmdsettings      \
                ->triggerdelay;          \
        processinfo->triggertimeout =    \
            CLIcmddata.cmdsettings      \
                ->triggertimeout;        \
        processinfo->triggerstreamID =   \
            image_ID(processinfo         \
                         ->triggerstreamname, \
                     dcimg, dcnimg);     \
        FUNC_CHECK_RETURN(               \
            processinfo_waitoninputstream_init( \
                processinfo,             \
                (processinfo             \
                         ->triggerstreamID \
                     > -1                \
                     ? &dcimg[processinfo \
                                  ->triggerstreamID] \
                     : NULL),            \
                CLIcmddata.cmdsettings  \
                    ->triggermode,        \
                CLIcmddata.cmdsettings  \
                    ->semindexrequested));\
        processinfo->RT_priority =       \
            CLIcmddata.cmdsettings      \
                ->RT_priority;           \
        processinfo->CPUmask =           \
            CLIcmddata.cmdsettings      \
                ->CPUmask;               \
        processinfo->MeasureTiming =     \
            CLIcmddata.cmdsettings      \
                ->procinfo_MeasureTiming;\
        processinfo_loopstart(           \
            processinfo);                \
    }


#define INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART \
    while (processloopOK == 1)           \
    {                                    \
        if (CLIcmddata.cmdsettings->flags \
            & CLICMDFLAG_PROCINFO) {     \
            processloopOK =              \
                processinfo_loopstep(    \
                    processinfo);        \
            processinfo_waitoninputstream(\
                processinfo);            \
            if (processinfo              \
                    ->triggerstatus ==   \
                PROCESSINFO_TRIGGERSTATUS_TIMEDOUT \
                && processinfo           \
                       ->triggermode ==   \
                   PROCESSINFO_TRIGGERMODE_SEMAPHORE) \
            {                            \
                continue;                \
            }                            \
            processinfo_exec_start(      \
                processinfo);            \
        } else {                         \
            processloopOK = 0;           \
        }                                \
        int processcompstatus = 1;       \
        if (CLIcmddata.cmdsettings->flags \
            & CLICMDFLAG_PROCINFO) {     \
            processcompstatus =          \
                processinfo_compute_status( \
                    processinfo);        \
        }                                \
        if (processcompstatus == 1) {


#define INSERT_STD_PROCINFO_COMPUTEFUNC_START \
    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT \
    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART


#define INSERT_STD_PROCINFO_COMPUTEFUNC_END \
    }                                    \
    if (CLIcmddata.cmdsettings->flags   \
        & CLICMDFLAG_PROCINFO) {         \
        if (processinfo != NULL) {       \
            if (dcfpsptr != NULL) {     \
                if (dcfpsptr->cmdset    \
                        .triggermodeptr  \
                    != NULL) {           \
                    processinfo          \
                        ->triggermode =  \
                        *dcfpsptr       \
                             ->cmdset    \
                             .triggermodeptr; \
                }                        \
                if (dcfpsptr->cmdset    \
                        .procinfo_loopcntMax_ptr \
                    != NULL) {           \
                    processinfo          \
                        ->loopcntMax =   \
                        *dcfpsptr       \
                             ->cmdset    \
                             .procinfo_loopcntMax_ptr; \
                }                        \
                if (dcfpsptr->cmdset    \
                        .triggerdelayptr \
                    != NULL) {           \
                    processinfo          \
                        ->triggerdelay = \
                        dcfpsptr        \
                            ->cmdset     \
                            .triggerdelayptr[0]; \
                }                        \
                if (dcfpsptr->cmdset    \
                        .triggertimeoutptr \
                    != NULL) {           \
                    processinfo          \
                        ->triggertimeout = \
                        dcfpsptr        \
                            ->cmdset     \
                            .triggertimeoutptr[0]; \
                }                        \
            }                            \
        }                                \
        if (processinfo != NULL) {       \
            processinfo_exec_end(        \
                processinfo);            \
        }                                \
    }                                    \
    }                                    \
    if (CLIcmddata.cmdsettings->flags   \
        & CLICMDFLAG_PROCINFO) {         \
        processinfo_cleanExit(           \
            processinfo);                \
    }


#endif /* CLICORE_STANDALONE_H */
