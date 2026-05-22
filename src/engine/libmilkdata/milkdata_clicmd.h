/**
 * @file milkdata_clicmd.h
 *
 * @brief MILK Data structures for CLI command arguments and registration
 *
 * Extracted from CLIcore_checkargs.h to allow the engine tier
 * (specifically libfps) to register parameters and structure
 * function arguments without depending on the interactive
 * CLIcore or readline libraries.
 */

#ifndef MILKDATA_CLICMD_H
#define MILKDATA_CLICMD_H

#include <stdint.h>
#include "milkDebugTools.h" // For errno_t

// Forward declaration of CMDSETTINGS (defined in cmdsettings.h)
typedef struct CMDSETTINGS CMDSETTINGS;

// testing argument type for command line interface
// CLI ARGS consist of two 16-bit fields
// lower 16-bit is format input type in CLI
// higher 16-bit can be more specific and used for conversion

#define CLIARG_MISSING 0x00000000
#define CLIARG_FLOAT 0x00000040        // FPTYPE_FLOAT64
#define CLIARG_LONG 0x00000008         // FPTYPE_INT64
#define CLIARG_STR_NOT_IMG 0x00040000  // FPTYPE_STRING_NOT_STREAM
#define CLIARG_IMG 0x00002000          // FPTYPE_STREAMNAME
#define CLIARG_STR 0x00004000          // FPTYPE_STRING
#define CLIARG_FILENAME 0x00000200     // FPTYPE_FILENAME
#define CLIARG_FITSFILENAME 0x00000400 // FPTYPE_FITSFILENAME
#define CLIARG_FPSNAME 0x00020000      // FPTYPE_FPSNAME

#define CLIARG_FLOAT32 0x00000020 // FPTYPE_FLOAT32
#define CLIARG_FLOAT64 0x00000040 // FPTYPE_FLOAT64

// integer types
#define CLIARG_ONOFF 0x00008000  // FPTYPE_ONOFF
#define CLIARG_INT32 0x00000002  // FPTYPE_INT32
#define CLIARG_UINT32 0x00000004 // FPTYPE_UINT32
#define CLIARG_INT64 0x00000008  // FPTYPE_INT64
#define CLIARG_UINT64 0x00000010 // FPTYPE_UINT64

// image/stream types
#define CLIARG_STREAM 0x00002000 // FPTYPE_STREAMNAME

// Convenience fpflag defaults for CLICMDARGDEF initializers
#define CLIARG_VISIBLE_DEFAULT 0x0000000000000002 // FPFLAG_DEFAULT_INPUT
#define CLIARG_HIDDEN_DEFAULT \
    (0x0000000000000002 & ~0x0000000000000001)   // FPFLAG_DEFAULT_INPUT & ~FPFLAG_VISIBLE
#define CLIARG_OUTPUT_DEFAULT 0x0000000000000004 // FPFLAG_DEFAULT_OUTPUT

#define STRINGMAXLEN_FPSCLIARG_TAG 100
#define STRINGMAXLEN_FPSCLIARG_DESCR 100
#define STRINGMAXLEN_FPSCLIARG_EXAMPLE 100
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
    uint64_t type; // Command line argument type
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

#define CLICMDDATA_KEY_STRLENMAX 100
#define CLICMDDATA_DESCRIPTION_STRLENMAX 200
#define CLICMDDATA_SOURCEFILENAME_STRLENMAX 1000

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

#endif // MILKDATA_CLICMD_H
