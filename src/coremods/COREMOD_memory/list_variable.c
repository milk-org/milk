/**
 * @file    list_variable.c
 * @brief   list variables
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif

// ==========================================
// Forward declaration(s)
// ==========================================

errno_t list_variable_ID();

errno_t list_variable_ID_file(const char *fname);

// ==========================================
#ifndef MILK_NO_CLI
// Command line interface wrapper function(s)
// ==========================================

static errno_t list_variable_ID_file__cli()
{
    if(CLI_checkarg(1, CLIARG_STR_NOT_IMG) == 0)
    {
        list_variable_ID_file(data.cmdargtoken[1].val.string);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

// ==========================================
// Register CLI command(s)
// ==========================================

errno_t list_variable_addCLIcmd()
{

    RegisterCLIcommand("listvar",
                       __FILE__,
                       list_variable_ID,
                       "list variables in memory",
                       "no argument",
                       "listvar",
                       "int list_variable_ID()");

    RegisterCLIcommand("listvarf",
                       __FILE__,
                       list_variable_ID_file__cli,
                       "list variables in memory, write to file",
                       "<file name>",
                       "listvarf var.txt",
                       "int list_variable_ID_file()");

    return RETURN_SUCCESS;
}
#endif /* MILK_NO_CLI */
errno_t list_variable_ID()
{
    variableID i;

    for(i = 0; i < dcnvar; i++)
        if(dcvar[i].used == 1)
        {
            printf("%4ld %16s %25.18g\n",
                   i,
                   dcvar[i].name,
                   dcvar[i].value.f);
        }

    return RETURN_SUCCESS;
}

errno_t list_variable_ID_file(const char *fname)
{
    imageID i;
    FILE   *fp;

    fp = fopen(fname, "w");
    for(i = 0; i < dcnvar; i++)
        if(dcvar[i].used == 1)
        {
            fprintf(fp,
                    "%s=%.18g\n",
                    dcvar[i].name,
                    dcvar[i].value.f);
        }

    fclose(fp);

    return RETURN_SUCCESS;
}

