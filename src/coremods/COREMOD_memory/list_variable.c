// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    list_variable.c
 * @brief   list variables
 *
 * Uses FPS V2 framework.
 */

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include <regex.h>
#include <stdio.h>
#include <string.h>

#include "list_variable.h"


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
    if (fp == NULL)
    {
        PRINT_ERROR("cannot open file \"%s\"", fname);
        return RETURN_FAILURE;
    }
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
 *  CMD 1: listvar
 * ============================================================= */

static char p_regex[FUNCTION_PARAMETER_STRMAXLEN] = "";

#define FPS_PARAMS_listvar(X) \
    X(".regex", p_regex, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "regex filter (empty = all)")

MILK_PARAMS_DEFINE(listvar, FPS_PARAMS_listvar);

static MILK_HOT errno_t listvar_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    FUNC_CHECK_RETURN(list_variable_ID(p_regex));
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_COREMOD_memory_listvar = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "listvar",
                 .cmdkey      = "listvar",
                 .description = "list variables in memory",
                 .description_long =
                     "List all variables currently defined in the process memory space, showing "
                     "name, type, and value." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(listvar),
    .compute = listvar_compute,
};


/* ================================================================
 *  CMD 2: listvarf
 * ============================================================= */

static char p_fname[FUNCTION_PARAMETER_STRMAXLEN] = "var.txt";

#define FPS_PARAMS_listvarf(X) \
    X(".fname", p_fname, FPTYPE_FILENAME, 1, FPFLAG_DEFAULT_INPUT, "output filename")

MILK_PARAMS_DEFINE(listvarf, FPS_PARAMS_listvarf);

static MILK_HOT errno_t listvarf_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    FUNC_CHECK_RETURN(list_variable_ID_file(p_fname));
    MILK_PROCINFO_END(self)
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_COREMOD_memory_listvarf = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "listvarf",
                 .cmdkey      = "listvarf",
                 .description = "list variables to file",
                 .description_long =
                     "List all variables currently defined in the process memory space, showing "
                     "name, type, and value." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(listvarf),
    .compute = listvarf_compute,
};
