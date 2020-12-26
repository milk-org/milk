/**
 * @file CLIcore_checkargs.h
 * 
 * @brief Check CLI command line arguments
 *
 */

#ifndef CLICORE_CHECKARGS_H
#define CLICORE_CHECKARGS_H







// testing argument type for command line interface
#define CLIARG_FLOAT            1  // floating point number
#define CLIARG_LONG             2  // integer (int or long)
#define CLIARG_STR_NOT_IMG      3  // string, not existing image
#define CLIARG_IMG              4  // existing image
#define CLIARG_STR              5  // string


typedef struct
{
	int type;
	char fpstag[100];
	char descr[100];
	char example[100];
} FPSCLIARG;



#define CLICMD_SUCCESS          0
#define CLICMD_INVALID_ARG      1
#define CLICMD_ERROR            2









int CLI_checkarg0(int argnum, int argtype, int errmsg);

int CLI_checkarg(int argnum, int argtype);

int CLI_checkarg_noerrmsg(int argnum, int argtype);

errno_t CLI_checkarg_array(
    FPSCLIARG fpscliarg[],
    int nbarg
);


#endif
