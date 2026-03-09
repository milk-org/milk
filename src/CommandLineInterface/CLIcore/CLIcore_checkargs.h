/**
 * @file CLIcore_checkargs.h
 *
 * @brief Check CLI command line arguments
 *
 */

#ifndef CLICORE_CHECKARGS_H
#define CLICORE_CHECKARGS_H

#include "../cmdsettings.h"

#include "libfps/fps.h"

// testing argument type for command line interface
// CLI ARGS consist of two 16-bit fields
// lower 16-bit is format input type in CLI
// higher 16-bit can be more specific and used for conversion

#define CLIARG_MISSING      0x00000000
#define CLIARG_FLOAT        FPTYPE_FLOAT64 // floating point number, defaults to float64
#define CLIARG_LONG         FPTYPE_INT64 // integer, default to int64
#define CLIARG_STR_NOT_IMG  FPTYPE_STRING_NOT_STREAM // string, not existing image
#define CLIARG_IMG          FPTYPE_STREAMNAME // existing image or stream
#define CLIARG_STR          FPTYPE_STRING // string
#define CLIARG_FILENAME     FPTYPE_FILENAME
#define CLIARG_FITSFILENAME FPTYPE_FITSFILENAME
#define CLIARG_FPSNAME      FPTYPE_FPSNAME


#define CLIARG_FLOAT32 FPTYPE_FLOAT32 // same as float
#define CLIARG_FLOAT64 FPTYPE_FLOAT64 // same as double

// integer types
#define CLIARG_ONOFF  FPTYPE_ONOFF
#define CLIARG_INT32  FPTYPE_INT32
#define CLIARG_UINT32 FPTYPE_UINT32
#define CLIARG_INT64  FPTYPE_INT64 // same as LONG
#define CLIARG_UINT64 FPTYPE_UINT64

// image/stream types
#define CLIARG_STREAM FPTYPE_STREAMNAME // stream


#define STRINGMAXLEN_FPSCLIARG_TAG       100
#define STRINGMAXLEN_FPSCLIARG_DESCR     100
#define STRINGMAXLEN_FPSCLIARG_EXAMPLE   100
#define STRINGMAXLEN_FPSCLIARG_LASTENTRY 100

typedef struct
{
    // Type is one of FPTYPE_XXXX
    uint64_t type;

    // tag is hierarchical set of words separated by dot: "word1.word2.word3"
    char fpstag[STRINGMAXLEN_FPSCLIARG_TAG];

    // short description of argument
    char descr[STRINGMAXLEN_FPSCLIARG_DESCR];

    // example value, will be used as default
    char example[STRINGMAXLEN_FPSCLIARG_EXAMPLE];

    // see FPFLAG_  in function_parameters.h
    uint64_t fpflag;

    // pointer to value
    void **valptr;

    // pointer to parameter index in fps
    long *indexptr;

} CLICMDARGDEF;

typedef struct
{
    uint64_t type;
    struct
    {
        double numf;
        long   numl;
        char   string[200];
    } val;
} CMDARGVAL;

#define STRINGMAXLEN_CLICMDARG 256
typedef struct
{
    uint64_t type;   // Command line argument type
    char     fpstag[STRINGMAXLEN_FPSCLIARG_TAG];
    char     descr[STRINGMAXLEN_FPSCLIARG_DESCR];
    char     example[STRINGMAXLEN_FPSCLIARG_EXAMPLE];
    uint64_t fpflag;
    union
    {
        float  f32;
        double f64;

        int32_t  i32;
        int64_t  i64;
        uint32_t ui32;
        uint64_t ui64;

        char s[STRINGMAXLEN_CLICMDARG];
    } val;
} CLICMDARGDATA;


#define CLICMDDATA_KEY_STRLENMAX            100
#define CLICMDDATA_DESCRIPTION_STRLENMAX    100
#define CLICMDDATA_SOURCEFILENAME_STRLENMAX 100

typedef struct
{
    char key[CLICMDDATA_KEY_STRLENMAX];
    char description[CLICMDDATA_DESCRIPTION_STRLENMAX];
    char sourcefilename[CLICMDDATA_SOURCEFILENAME_STRLENMAX];

    int           nbarg;
    CLICMDARGDEF *funcfpscliarg;

    uint64_t flags; // controls function behavior and capabilities
    // see CLICMDFLAGS for details

    // pointer to CMD struct initialized by CLI function registration
    CMDSETTINGS *cmdsettings;

    // pointer to optional custom FPS conf setup function
    errno_t (*FPS_customCONFsetup)();

    // pointer to optional custom FPS conf check function
    errno_t (*FPS_customCONFcheck)();

} CLICMDDATA;

#define CLICMD_SUCCESS     0
#define CLICMD_INVALID_ARG 1
#define CLICMD_ERROR       2

/* Function declarations — only in full CLI mode.
 * In standalone (MILK_NO_CLI), these are provided
 * as static inline stubs in CLIcore_standalone.h.
 */
#ifndef MILK_NO_CLI

int CLI_checkarg(int argnum, uint32_t argtype);

int CLI_checkarg_noerrmsg(
    int argnum, uint32_t argtype);

errno_t CLI_checkarg_array(
    CLICMDARGDEF fpscliarg[], int nbarg);

int CLIargs_to_FPSparams_setval(
    CLICMDARGDEF               fpscliarg[],
    int                        nbarg,
    FUNCTION_PARAMETER_STRUCT *fps);

int CMDargs_to_FPSparams_create(
    FUNCTION_PARAMETER_STRUCT *fps);

void *get_farg_ptr(char *tag, long *fpsi);

#endif /* !MILK_NO_CLI */

#endif
