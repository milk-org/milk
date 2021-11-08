/**
 * @file CLIcore_checkargs.h
 *
 * @brief Check CLI command line arguments
 *
 */

#ifndef CLICORE_CHECKARGS_H
#define CLICORE_CHECKARGS_H


#include "cmdsettings.h"


// testing argument type for command line interface
// CLI ARGS consist of two 16-bit fields
// lower 16-bit is format input type in CLI
// higher 16-bit can be more specific and used for conversion

#define CLIARG_MISSING          0x00000000
#define CLIARG_FLOAT            0x00000001  // floating point number, defaults to float64
#define CLIARG_LONG             0x00000002  // integer, default to int64
#define CLIARG_STR_NOT_IMG      0x00000003  // string, not existing image
#define CLIARG_IMG              0x00000004  // existing image
#define CLIARG_STR              0x00000005  // string

#define CLIARG_FLOAT32          0x00010001
#define CLIARG_FLOAT64          0x00020001 // same as FLOAT

#define CLIARG_INT32            0x00010002
#define CLIARG_UINT32           0x00110002
#define CLIARG_INT64            0x00020002 // same as LONG
#define CLIARG_UINT64           0x00120002




#define STRINGMAXLEN_FPSCLIARG_TAG       100
#define STRINGMAXLEN_FPSCLIARG_DESCR     100
#define STRINGMAXLEN_FPSCLIARG_EXAMPLE   100
#define STRINGMAXLEN_FPSCLIARG_LASTENTRY 100



#define CLICMDARG_FLAG_DEFAULT 0x00000000

#define CLICMDARG_FLAG_NOCLI 0x00000001 // 1 if argument is not part or CLI call
// If set to 1, the argument value is not specified as part of the
// command line function call in the CLI
#define CLICMDARG_FLAG_NOFPS 0x00000002 // 1 if argument is not part or FPS


typedef struct
{
    // Type is one of CLIARG_XXXX
    int type;

    // tag is hierarchical set of words separated by dot: "word1.word2.word3"
    char fpstag[STRINGMAXLEN_FPSCLIARG_TAG];

    // short description of argument
    char descr[STRINGMAXLEN_FPSCLIARG_DESCR];

    // example value, will be used as default
    char example[STRINGMAXLEN_FPSCLIARG_EXAMPLE];

    // CLICMDARG flag
    uint64_t flag;

    // see FPTYPE_ in function_parameters.h
    uint64_t fptype;

    // see FPFLAG_  in function_parameters.h
    uint64_t fpflag;

    void **valptr;

} CLICMDARGDEF;


typedef struct
{
    int type;
    struct
    {
        double numf;
        long numl;
        char string[200];
    } val;
} CMDARGVAL;


#define STRINGMAXLEN_CLICMDARG 256
typedef struct
{
    int type;
    char fpstag[STRINGMAXLEN_FPSCLIARG_TAG];
    char descr[STRINGMAXLEN_FPSCLIARG_DESCR];
    char example[STRINGMAXLEN_FPSCLIARG_EXAMPLE];
    uint64_t flag;
    union
    {
        double f;
        long l;
        char s[STRINGMAXLEN_CLICMDARG];
    } val;
} CLICMDARGDATA;








typedef struct
{
    char key[100];
    char description[100];
    char sourcefilename[100];

    int nbarg;
    CLICMDARGDEF *funcfpscliarg;

    uint64_t flags; // controls function behavior and capabilities
    // see CLICMDFLAGS for details

    // pointer to CMD struct initialized by CLI function registration
    CMDSETTINGS *cmdsettings;

    errno_t (*FPS_customCONFsetup)
    ();    // pointer to optional custom FPS conf setup function
    errno_t (*FPS_customCONFcheck)
    ();    // pointer to optional custom FPS conf check function

} CLICMDDATA;



#define CLICMD_SUCCESS          0
#define CLICMD_INVALID_ARG      1
#define CLICMD_ERROR            2









//int CLI_checkarg0(int argnum, int argtype, int errmsg);

int CLI_checkarg(int argnum, uint32_t argtype);

int CLI_checkarg_noerrmsg(int argnum, uint32_t argtype);

errno_t CLI_checkarg_array(
    CLICMDARGDEF fpscliarg[],
    int nbarg
);



int CLIargs_to_FPSparams_setval(
    CLICMDARGDEF fpscliarg[],
    int nbarg,
    FUNCTION_PARAMETER_STRUCT *fps
);

int CMDargs_to_FPSparams_create(
    FUNCTION_PARAMETER_STRUCT *fps
);



void *get_farg_ptr(
    char *tag
);



#endif
