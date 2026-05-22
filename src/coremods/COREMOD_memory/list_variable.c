/**
 * @file    list_variable.c
 * @brief   list variables
 *
 * Uses FPS V2 framework.
 */

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include <regex.h>
#include <string.h>


/* ================================================================
 *  COMPUTATION LOGIC
 * ============================================================= */

/**
 * @brief List variables, optionally filtered by regex
 *
 * @param regexstr  POSIX extended regex pattern to match
 *                  variable names. NULL or empty string
 *                  matches all variables.
 */
errno_t list_variable_ID(const char *regexstr)
{
    regex_t re;
    int     use_regex = 0;

    if (regexstr != NULL && regexstr[0] != '\0')
    {
        int rc = regcomp(&re, regexstr, REG_EXTENDED | REG_NOSUB);
        if (rc != 0)
        {
            char errbuf[128];
            regerror(rc, &re, errbuf, sizeof(errbuf));
            PRINT_ERROR("bad regex \"%s\": %s", regexstr, errbuf);
            return RETURN_FAILURE;
        }
        use_regex = 1;
    }

    for (variableID i = 0; i < dcnvar; i++)
    {
        if (dcvar[i].used != 1)
        {
            continue;
        }

        if (use_regex)
        {
            if (regexec(&re, dcvar[i].name, 0, NULL, 0) != 0)
            {
                continue;
            }
        }

        printf("%4ld %16s ", i, dcvar[i].name);
        if (dcvar[i].type == 0)
        {
            printf("%25.18g\n", dcvar[i].value.f);
        }
        else if (dcvar[i].type == 1)
        {
            printf("%25ld\n", dcvar[i].value.l);
        }
        else if (dcvar[i].type == 2)
        {
            printf("%25s\n", dcvar[i].value.s);
        }
    }

    if (use_regex)
    {
        regfree(&re);
    }

    return RETURN_SUCCESS;
}

/**
 * @brief List variables to file
 */
errno_t list_variable_ID_file(const char *fname)
{
    imageID i;
    FILE   *fp;

    fp = fopen(fname, "w");
    for (i = 0; i < dcnvar; i++)
    {
        if (dcvar[i].used == 1)
        {
            if (dcvar[i].type == 0)
            {
                fprintf(fp, "%s=%.18g\n", dcvar[i].name, dcvar[i].value.f);
            }
            else if (dcvar[i].type == 1)
            {
                fprintf(fp, "%s=%ld\n", dcvar[i].name, dcvar[i].value.l);
            }
            else if (dcvar[i].type == 2)
            {
                fprintf(fp, "%s=%s\n", dcvar[i].name, dcvar[i].value.s);
            }
        }
    }

    fclose(fp);

    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 1: listvar — REGISTRATION
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_listvar = {
    .fps_name         = "listvar",
    .cmdkey           = "listvar",
    .description      = "list variables in memory",
    .description_long = "List all variables currently defined in the process memory space, showing "
                        "name, type, and value."
};

static char p_regex[FUNCTION_PARAMETER_STRMAXLEN] = "";

#define FPS_PARAMS_listvar(X) \
    X(".regex", p_regex, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "regex filter (empty = all)")

static FPS_CLI_BINDING my_bindings_listvar[] = { FPS_PARAMS_listvar(FPS_X_BINDING) };

static const int __attribute__((unused)) nb_bindings_listvar =
    sizeof(my_bindings_listvar) / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg_listvar[] = { FPS_PARAMS_listvar(FPS_X_FARG) };

static CLICMDDATA CLIcmddata_listvar = {
    "",   "",   __FILE__, sizeof(farg_listvar) / sizeof(CLICMDARGDEF), farg_listvar, CLICMDFLAG_FPS,
    NULL, NULL, NULL
};

FPS_CMDSETTINGS_INIT(dft1, CLIcmddata_listvar, FPS_app_info_listvar)

static errno_t __attribute__((unused)) compute_listvar()
{
    FUNC_CHECK_RETURN(list_variable_ID(p_regex));
    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 2: listvarf — REGISTRATION
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_listvarf = {
    .fps_name         = "listvarf",
    .cmdkey           = "listvarf",
    .description      = "list variables to file",
    .description_long = "List all variables currently defined in the process memory space, showing "
                        "name, type, and value."
};

static char p_fname[FUNCTION_PARAMETER_STRMAXLEN] = "var.txt";

#define FPS_PARAMS_listvarf(X) \
    X(".fname", p_fname, FPTYPE_FILENAME, 1, FPFLAG_DEFAULT_INPUT, "output filename")

static FPS_CLI_BINDING my_bindings2[] = { FPS_PARAMS_listvarf(FPS_X_BINDING) };

static const int __attribute__((unused)) nb_bindings2 =
    sizeof(my_bindings2) / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = { FPS_PARAMS_listvarf(FPS_X_FARG) };

static CLICMDDATA CLIcmddata = { "", "", CLICMD_FIELDS_DEFAULTS };

FPS_CMDSETTINGS_INIT(dft2, CLIcmddata, FPS_app_info_listvarf)

static errno_t __attribute__((unused)) compute_listvarf()
{
    DEBUG_TRACE_FSTART();
    INSERT_STD_PROCINFO_COMPUTEFUNC_START FUNC_CHECK_RETURN(list_variable_ID_file(p_fname));
    INSERT_STD_PROCINFO_COMPUTEFUNC_END   DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 *  REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)

static errno_t CLIfunction_listvar(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info_listvar, farg_listvar, &CLIcmddata_listvar,
                                        my_bindings_listvar, nb_bindings_listvar, compute_listvar);
}

static errno_t CLIfunction_listvarf(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info_listvarf, farg, &CLIcmddata, my_bindings2,
                                        nb_bindings2, compute_listvarf);
}

errno_t CLIADDCMD_COREMOD_memory__list_variable()
{
    {
        safe_fps_fill_farg_examples(farg_listvar, my_bindings_listvar, nb_bindings_listvar);

        int cmdi                       = RegisterCLIcmd(CLIcmddata_listvar, CLIfunction_listvar);
        CLIcmddata_listvar.cmdsettings = &data.cmd[cmdi].cmdsettings;
    }

    {
        safe_fps_fill_farg_examples(farg, my_bindings2, nb_bindings2);

        int cmdi               = RegisterCLIcmd(CLIcmddata, CLIfunction_listvarf);
        CLIcmddata.cmdsettings = &data.cmd[cmdi].cmdsettings;
    }

    return RETURN_SUCCESS;
}
#endif
