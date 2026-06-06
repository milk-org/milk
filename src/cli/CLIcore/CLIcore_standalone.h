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
static pid_t   CLIPID __attribute__((unused));
static char    DocDir[200] __attribute__((unused));
static char    SrcDir[200] __attribute__((unused));
static char    BuildFile[200] __attribute__((unused));
static char    BuildDate[200] __attribute__((unused));
static char    BuildTime[200] __attribute__((unused));
static int     C_ERRNO __attribute__((unused));
static uid_t   euid_real __attribute__((unused));
static uid_t   euid_called __attribute__((unused));
static uid_t   suid __attribute__((unused));
static uint8_t TYPESIZE[32] __attribute__((unused));


/* TUI stubs moved to fps_standalone_data.c */


/* =====================================
 * String length constants
 * ===================================== */

#define STRINGMAXLEN_CLISTARTUPFILENAME 200
#define STRINGMAXLEN_CLIPROMPT 200

#define CFITSEXIT                    \
    printf("Abnormal termination, "  \
           "File \"%s\", line %d\n", \
           __FILE__, __LINE__);      \
    exit(0)

#ifdef DEBUG
#    define nmalloc(f, type, n)                         \
        f = (type *) calloc(n, sizeof(type));           \
        if (f == NULL)                                  \
        {                                               \
            printf("ERROR: \"" #f "\" alloc failed\n"); \
            exit(0);                                    \
        }                                               \
        else                                            \
        {                                               \
            printf("\nMALLOC: \"" #f "\" allocated\n"); \
        }
#    define nfree(f) \
        free(f);     \
        printf("\nMALLOC: \"" #f "\" freed\n");
#else
#    define nmalloc(f, type, n)                         \
        f = (type *) calloc(n, sizeof(type));           \
        if (f == NULL)                                  \
        {                                               \
            printf("ERROR: \"" #f "\" alloc failed\n"); \
            exit(0);                                    \
        }
#    define nfree(f) free(f);
#endif

#define TEST_ALLOC(f)                               \
    if (f == NULL)                                  \
    {                                               \
        printf("ERROR: \"" #f "\" alloc failed\n"); \
        exit(0);                                    \
    }

#define NB_ARG_MAX 100


/* =====================================
 * Module init — no-op in standalone
 * ===================================== */

/* MODULE_DEPS — standalone stub.
 * Defines the arrays (they exist but are
 * never iterated since INIT_MODULE_LIB is
 * a no-op in standalone builds).
 */
#define MODULE_DEPS(...)                                                         \
    static const char *_module_deps[] __attribute__((unused)) = { __VA_ARGS__ }; \
    static const int   _module_ndeps __attribute__((unused)) =                   \
        (int) (sizeof(_module_deps) / sizeof(_module_deps[0]));                  \
    static const int _module_deps_defined __attribute__((unused)) = 1

#define INIT_MODULE_LIB(modname)                              \
    static errno_t                     init_module_CLI(void); \
    static int __attribute__((unused)) INITSTATUS_##modname = 0;

/* INIT_MODULE_LIB_DEPS — same as INIT_MODULE_LIB
 * in standalone builds (no dep loading).
 */
#define INIT_MODULE_LIB_DEPS(modname)                         \
    static errno_t                     init_module_CLI(void); \
    static int __attribute__((unused)) INITSTATUS_##modname = 0;


/* =====================================
 * Type definitions
 * ===================================== */

typedef uint_fast8_t BOOL;
#define FALSE 0
#define TRUE 1

#define DATA_NB_MAX_COMMAND 1
#define DATA_NB_MAX_MODULE 1

#define STRINGMAXLEN_MODULE_NAME 100
#define STRINGMAXLEN_CMD_KEY 100

typedef struct
{
    char name[STRINGMAXLEN_MODULE_NAME];
} MODULE;

typedef struct
{
    char        key[STRINGMAXLEN_CMD_KEY];
    CMDSETTINGS cmdsettings;
} CMD;

typedef struct
{
    int type;
    struct
    {
        double numf;
        long   numl;
        char   string[10];
    } val;
} CMDARGTOKEN;

/* =====================================
 * DATA struct (CLI-extended MILK_DATA)
 *
 * Minimally defined for standalone compute
 * builds to compile dummy CLI registrations.
 * ===================================== */

typedef struct
{
    MILK_DATA   core;
    CMD         cmd[DATA_NB_MAX_COMMAND];
    CMDARGTOKEN cmdargtoken[2]; // minimal
    MODULE      module[DATA_NB_MAX_MODULE];
    long        cmdNBarg;
    long        cmdindex;
    char        processname[STRINGMAXLEN_PROCESSNAME];
} DATA;

extern DATA data;

/* =====================================
 * CLI function stubs (no-op)
 * ===================================== */

static inline int CLI_checkarg(int      argnum __attribute__((unused)),
                               uint32_t argtype __attribute__((unused)))
{
    return 1; /* always "fail" — prevents
                 legacy CLI wrappers from
                 executing in standalone */
}

static inline int CLI_checkarg_noerrmsg(int      argnum __attribute__((unused)),
                                        uint32_t argtype __attribute__((unused)))
{
    return 1;
}

static inline errno_t CLI_checkarg_array(CLICMDARGDEF *fca __attribute__((unused)),
                                         int           nbarg __attribute__((unused)))
{
    return 1;
}

static inline int CLIargs_to_FPSparams_setval(CLICMDARGDEF *fca __attribute__((unused)),
                                              int           n __attribute__((unused)),
                                              FPS          *fps __attribute__((unused)))
{
    return 0;
}

static inline int CMDargs_to_FPSparams_create(FPS *fps __attribute__((unused)))
{
    return 0;
}

static inline void *get_farg_ptr(char *tag __attribute__((unused)),
                                 long *fpsi __attribute__((unused)))
{
    return NULL;
}

static inline errno_t set_signal_catch(void)
{
    return 0;
}

static inline void sig_handler(int signo __attribute__((unused)))
{
}

static inline errno_t RegisterModule(const char *f __attribute__((unused)),
                                     const char *p __attribute__((unused)),
                                     const char *i __attribute__((unused)),
                                     int         ma __attribute__((unused)),
                                     int         mi __attribute__((unused)),
                                     int         pa __attribute__((unused)))
{
    return 0;
}

static inline uint32_t RegisterCLIcmd(CLICMDDATA cd __attribute__((unused)),
                                      errno_t (*fp)(void) __attribute__((unused)))
{
    return 0;
}

static inline uint32_t RegisterCLIcommand(const char *k __attribute__((unused)),
                                          const char *s __attribute__((unused)),
                                          errno_t (*fp)() __attribute__((unused)),
                                          const char *i __attribute__((unused)),
                                          const char *sy __attribute__((unused)),
                                          const char *e __attribute__((unused)),
                                          const char *c __attribute__((unused)))
{
    return 0;
}

#include "milk_types.h"
#include "fps_procinfo_macros.h"

#endif /* CLICORE_STANDALONE_H */
