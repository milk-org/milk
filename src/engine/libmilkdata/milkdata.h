/**
 * @file    milkdata.h
 * @brief   Core milk data structures
 *
 * Defines MILK_DATA -- the minimal data structure
 * needed by all milk programs (CLI and standalone).
 * The CLI extends this with CLI-specific fields
 * in the DATA struct (see CLIcore.h).
 */

#ifndef MILKDATA_H
#define MILKDATA_H

#include <signal.h>
#include <stdint.h>
#include <sys/types.h>

#include "milkDebugTools.h"

#include "ImageStreamIO/ImageStreamIO.h"
#include "ImageStreamIO/ImageStruct.h"

#define STRINGMAXLEN_FPS_NAME 100

#include "libfps/fps_types.h"
struct PROCESSINFO;

#ifndef __STDC_LIB_EXT1__
typedef int errno_t;
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338328
#endif

/* Static allocation limits */
#define STATIC_NB_MAX_IMAGE    520
#define STATIC_NB_MAX_VARIABLE 5030

/* In STATIC mode, arrays are fixed-size */
/* #define DATA_STATIC_ALLOC */

/* Maximum number of entries in trace array */
#define CODETESTPOINTARRAY_NBCNT 100000

/* Maximum function stack depth */
#define MAXNB_FUNCSTACK 100
#define STRINGMAXLEN_FUNCSTAK_FUNCNAME 100


/**
 * @brief Variable structure for internal use
 */
typedef struct
{
    int  used;
    char name[80];
    int  type; /** 0: double, 1: long, 2: string */
    union
    {
        double f;
        long   l;
        char   s[80];
    } value;
    char comment[200];
} VARIABLE;


/**
 * @brief Code test point for tracing
 */
typedef struct
{
    uint64_t loopcnt;
    int      line;
    char     file[STRINGMAXLEN_FULLFILENAME];
    char     func[STRINGMAXLEN_FUNCTIONNAME];

    int  funclevel;
    long funccallcnt;

    char funcstack[MAXNB_FUNCSTACK]
        [STRINGMAXLEN_FUNCSTAK_FUNCNAME];
    long fcntstack[MAXNB_FUNCSTACK];
    int  linestack[MAXNB_FUNCSTACK];

    char            msg[STRINGMAXLEN_FUNCTIONARGS];
    struct timespec time;
} CODETESTPOINT;


/**
 * @brief Core milk data structure
 *
 * Contains all state shared between CLI and standalone
 * programs: image arrays, FPS arrays, signals, config.
 * The CLI-only DATA struct embeds this as its first
 * member (field 'core').
 */
typedef struct
{
    /* Package info */
    char package_name[100];
    int  package_version_major;
    int  package_version_minor;
    int  package_version_patch;
    char package_version[100];
    char configdir[STRINGMAXLEN_DIRNAME];
    char sourcedir[STRINGMAXLEN_DIRNAME];
    char installdir[STRINGMAXLEN_DIRNAME];

    char shmdir[STRINGMAXLEN_DIRNAME];
    char shmsemdirname[STRINGMAXLEN_DIRNAME];

    /* Signals */
    struct sigaction sigact;

    int signal_USR1;
    int signal_USR2;
    int signal_TERM;
    int signal_INT;
    int signal_SEGV;
    int signal_ABRT;
    int signal_BUS;
    int signal_HUP;
    int signal_PIPE;

    /* Test points (runtime tracing) */
    CODETESTPOINT  testpoint;
    CODETESTPOINT *testpointarray;
    int            testpointarrayinit;
    uint64_t       testpointloopcnt;
    uint64_t       testpointcnt;

    /* Program status */
    int progStatus;

    /* Real-time priority UIDs */
    uid_t ruid;
    uid_t euid;
    uid_t suid;

    /* Operation mode */
    int Debug;
    int quiet;
    int errorexit;
    int exitcode;

    int    overwrite;
    int    rmSHMfile;
    double INVRANDMAX;
    void  *rndgen;
    int    precision;

    /* Process monitoring */
    int          processinfo;
    int          processinfoActive;
    struct PROCESSINFO *pinfo;

    /* FPS */
    long                       NB_MAX_FPS;
    FPS *fpsarray;
    FPS *fpsptr;
    char     FPS_name[STRINGMAXLEN_FPS_NAME];
    long     FPS_TIMESTAMP;
    uint32_t FPS_CMDCODE;

    /* Images */
    long NB_MAX_IMAGE;
#ifdef DATA_STATIC_ALLOC
    IMAGE image[STATIC_NB_MAX_IMAGE];
#else
    IMAGE *image;
#endif
    int MEM_MONITOR;
    int SHARED_DFT;

    /* Variables */
    long NB_MAX_VARIABLE;
#ifdef DATA_STATIC_ALLOC
    VARIABLE variable[STATIC_NB_MAX_VARIABLE];
#else
    VARIABLE *variable;
#endif

    /* Convenience storage */
    float  FLOATARRAY[1000];
    double DOUBLEARRAY[1000];
    char   SAVEDIR[STRINGMAXLEN_DIRNAME];

    int retvalue;
    int status0;
    int status1;

} MILK_DATA;


/**
 * @brief Global core data instance
 *
 * In standalone programs, this is the primary global.
 * In CLI programs, data.core aliases this via macro.
 */
extern MILK_DATA milk_data;


/* ========================================
 * Core initialization
 * ======================================== */

/**
 * @brief Initialize milk_data arrays
 *
 * Allocates image, variable, and FPS arrays.
 * Must be called early in main().
 */
errno_t milk_data_init(void);


/* ========================================
 * Shorthand access macros (dc = data core)
 * ======================================== */

#include "milkdata_macros.h"

/* ========================================
 * Pure-C random number generator
 *
 * Replaces GSL RNG. Uses xorshift64* for
 * uniform generation, Box-Muller for
 * Gaussian, Knuth for Poisson.
 * ======================================== */

/**
 * @brief Allocate and seed the global RNG
 *
 * Stores a MILK_RNG in milk_data.rndgen.
 * @param seed  Seed value (e.g. time(NULL))
 */
void milk_rng_init(uint64_t seed);

/** @brief Free the global RNG */
void milk_rng_free(void);

/** @brief Uniform random in [0, 1) */
double milk_rng_uniform(void);

/** @brief Gaussian random N(0, sigma) */
double milk_rng_gaussian(double sigma);

/** @brief Poisson random with mean mu */
long milk_rng_poisson(double mu);


#endif /* MILKDATA_H */
