/**
 * @file CLIcore_checkargs.h
 * 
 * @brief Check CLI command line arguments
 *
 */


#ifndef CLICORE_CHECKARGS_H

#define CLICORE_CHECKARGS_H


int CLI_checkarg0(int argnum, int argtype, int errmsg);

int CLI_checkarg(int argnum, int argtype);

int CLI_checkarg_noerrmsg(int argnum, int argtype);


#endif
