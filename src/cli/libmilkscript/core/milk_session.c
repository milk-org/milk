// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_session.c
 * @brief Built-in module `milk`, engine root commands and default modules (see milk_session.h).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "CLIcore_help.h"
#include "CLIcore_modules.h"
#include "CLIcore_script.h"
#include "milk_cmd_index.h"
#include "milk_config.h"
#include "milk_session.h"

/* Loaded in every front-end after the core modules, unless $MILKCLI_ADD_LIBS says otherwise */
static const char *const default_extra_modules =
    "milklinalgebra,milkfft,milkimagebasic,milkimagefilter,milkimageformat,milkimagegen,"
    "milkimgreduce,"
    "milkinfo,milkkdtree,milklinARfilterPred,milklinoptimtools,milk_module_example,milkpsf,"
    "milkstatistic,milkZernikePolyn,milksequencer";

static int g_milk_midx = -1;

void milk_module_begin(void)
{
    milk_reserve_milk_shortname(1);
    if (g_milk_midx < 0)
    {
        strncpy(data.modulename, "milk", STRINGMAXLEN_MODULE_NAME - 1);
        strncpy(data.moduleshortname, "milk", STRINGMAXLEN_MODULE_SHORTNAME - 1);
        strncpy(data.moduledatestring, __DATE__, STRINGMAXLEN_MODULE_DATESTRING - 1);
        strncpy(data.moduletimestring, __TIME__, STRINGMAXLEN_MODULE_TIMESTRING - 1);
        RegisterModule(__FILE__, "milk", "milk engine utilities", MILK_VERSION_MAJOR,
                       MILK_VERSION_MINOR, MILK_VERSION_PATCH);
        g_milk_midx = (int) data.moduleindex;
    }
    data.moduleindex = g_milk_midx;
    strncpy(data.modulename, "milk", STRINGMAXLEN_MODULE_NAME - 1);
}

void milk_module_end(void)
{
    data.moduleindex        = -1;
    data.modulename[0]      = '\0';
    data.moduleshortname[0] = '\0';
    milk_reserve_milk_shortname(0);
}

void milk_session_register_engine(void)
{
    milk_module_begin();
    RegisterCLIcommand("ci", __FILE__, printInfo, "print version, build settings and memory use",
                       "no argument", "milk.ci", "int printInfo()");
    RegisterCLIcommand("fpsdump", __FILE__, cli_cmd_fpsdump, "dump FPS parameters as key=value",
                       "[-t] [--json] <fpsname>", "milk.fpsdump loopctrl", "cli_cmd_fpsdump()");
    milk_module_end();

    RegisterCLIcommand("help", __FILE__, help, "help on topics, keywords, modules and commands",
                       "[subject] [--json|--porcelain] | -k <regex> | --tui",
                       "help iofits.loadfits", "int help()");
}

void milk_session_load_modules(void)
{
    load_module_shared("milkCOREMODmemory");
#ifdef USE_CFITSIO
    load_module_shared("milkCOREMODiofits");
#endif
    load_module_shared("milkCOREMODarith");
    load_module_shared("milkCOREMODtools");

    const char *extra = getenv("MILKCLI_ADD_LIBS");
    if (extra == NULL)
    {
        extra = default_extra_modules;
    }
    if (extra[0] == '\0' || strcmp(extra, "none") == 0)
    {
        return;
    }
    char  list[2048];
    char *save = NULL;
    snprintf(list, sizeof(list), "%s", extra);
    for (char *lib = strtok_r(list, " ,;", &save); lib != NULL; lib = strtok_r(NULL, " ,;", &save))
    {
        if (load_module_shared(lib) != RETURN_SUCCESS)
        {
            fprintf(stderr, "milk: could not load module '%s'\n", lib);
        }
    }
}
