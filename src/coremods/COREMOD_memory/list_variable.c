/**
 * @file    list_variable.c
 * @brief   list variables
 *
 * Uses FPS V2 framework.
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"


/* ================================================================
 *  COMPUTATION LOGIC
 * ============================================================= */

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


/* ================================================================
 *  CMD 1: listvar — REGISTRATION
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_listvar = {
    .fps_name    = "listvar",
    .cmdkey      = "listvar",
    .description = "list variables in memory"
};

static CLICMDDATA CLIcmddata_listvar = {
    "",
    "",
    CLICMD_FIELDS_NOPARAM
};

FPS_CMDSETTINGS_INIT(dft1, CLIcmddata_listvar, FPS_app_info_listvar)

static errno_t __attribute__((unused)) compute_listvar()
{
    FUNC_CHECK_RETURN(list_variable_ID());
    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 2: listvarf — REGISTRATION
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_listvarf = {
    .fps_name    = "listvarf",
    .cmdkey      = "listvarf",
    .description =
        "list variables to file"
};

static char p_fname[FUNCTION_PARAMETER_STRMAXLEN]
    = "var.txt";

#define FPS_PARAMS_listvarf(X) \
    X(".fname", p_fname, \
      FPTYPE_FILENAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output filename")

static FPS_CLI_BINDING my_bindings2[] = {
    FPS_PARAMS_listvarf(FPS_X_BINDING)
};

static const int __attribute__((unused)) nb_bindings2 =
    sizeof(my_bindings2) /
    sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = {
    FPS_PARAMS_listvarf(FPS_X_FARG)
};

static CLICMDDATA CLIcmddata = {
    "",
    "",
    CLICMD_FIELDS_DEFAULTS
};

FPS_CMDSETTINGS_INIT(dft2, CLIcmddata, FPS_app_info_listvarf)

static errno_t __attribute__((unused)) compute_listvarf()
{
    DEBUG_TRACE_FSTART();
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    FUNC_CHECK_RETURN(
        list_variable_ID_file(p_fname));
    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 *  REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)

static errno_t CLIfunction_listvar(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_listvar,
        NULL, &CLIcmddata_listvar,
        NULL, 0,
        compute_listvar);
}

static errno_t CLIfunction_listvarf(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_listvarf,
        farg, &CLIcmddata,
        my_bindings2, nb_bindings2,
        compute_listvarf);
}

errno_t
CLIADDCMD_COREMOD_memory__list_variable()
{
    {
        int cmdi = RegisterCLIcmd(
            CLIcmddata_listvar,
            CLIfunction_listvar);
        CLIcmddata_listvar.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }

    {
        safe_fps_fill_farg_examples(
            farg, my_bindings2,
            nb_bindings2);

        int cmdi = RegisterCLIcmd(
            CLIcmddata,
            CLIfunction_listvarf);
        CLIcmddata.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }

    return RETURN_SUCCESS;
}
#endif
