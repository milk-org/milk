// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_builtin_module.c
 * @brief Built-in module `milk`, root commands and default modules (see milk_builtin_module.h).
 *
 * The commands are MILK_RUNNABLE definitions, registered with milk_register_runnable().
 * milk_module_begin() registers module `milk` once, under its reserved short name, and makes it
 * the module of the commands registered until milk_module_end(). The core modules load in a fixed
 * order: memory, iofits (when built with cfitsio), arith, tools. The other installed modules
 * load when one of their commands is first used (milk_catalog.h); $MILKCLI_ADD_LIBS names
 * modules to load at start.
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
#include "milk_runnable.h"
#include "milk_builtin_module.h"

static int g_milk_midx = -1;

#define SESSION_CLI_CMD(tag, keystr, descstr, fn)                     \
    MILK_RUNNABLE_RT_DEFINE(tag);                                     \
    static const MILK_RUNNABLE unit_##tag = {                         \
        .kind     = MILK_RUNNABLE_CLI,                                \
        .app      = { .cmdkey = (keystr), .description = (descstr) }, \
        .srcfile  = __FILE__,                                         \
        .cli_func = (fn),                                             \
        .rt       = &tag##_rt,                                        \
    }

SESSION_CLI_CMD(ci, "ci", "print version, build settings and memory use", printInfo);
SESSION_CLI_CMD(fpsdump, "fpsdump", "dump FPS parameters as key=value", cli_cmd_fpsdump);
SESSION_CLI_CMD(help, "help", "help on topics, keywords, modules and commands", help);

/**
 * milk_module_begin - Activate the built-in `"milk"` module context for registration
 *
 * Temporarily unlocks the reserved `"milk"` shortname, registers the `"milk"` module
 * slot on first call, and sets `data.moduleindex` so subsequent `milk_register_runnable`
 * calls register commands under `milk.<cmd>`.
 *
 * Return: void
 */
void milk_module_begin(void)
{
    milk_reserve_milk_shortname(1);
    if (g_milk_midx < 0)
    {
        strncpy(data.modulename, "milk", STRINGMAXLEN_MODULE_NAME - 1);
        strncpy(data.moduleshortname, "milk", STRINGMAXLEN_MODULE_SHORTNAME - 1);
        strncpy(data.moduledatestring, __DATE__, STRINGMAXLEN_MODULE_DATESTRING - 1);
        strncpy(data.moduletimestring, __TIME__, STRINGMAXLEN_MODULE_TIMESTRING - 1);
        RegisterModule(__FILE__, "milk", "milk built-in utilities", MILK_VERSION_MAJOR,
                       MILK_VERSION_MINOR, MILK_VERSION_PATCH);
        g_milk_midx = (int) data.moduleindex;
    }
    data.moduleindex = g_milk_midx;
    strncpy(data.modulename, "milk", STRINGMAXLEN_MODULE_NAME - 1);
}

/**
 * milk_module_end - Close the built-in `"milk"` module registration context
 *
 * Resets transient module registration fields in `data` and re-locks the `"milk"`
 * shortname so user modules cannot claim it.
 *
 * Return: void
 */
void milk_module_end(void)
{
    data.moduleindex        = -1;
    data.modulename[0]      = '\0';
    data.moduleshortname[0] = '\0';
    milk_reserve_milk_shortname(0);
}

/**
 * milk_register_builtin_commands - Register built-in `milk.*` and root commands
 *
 * Registers `milk.ci`, `milk.fpsdump`, and the root `help` command shared across
 * all CLI and scripting front-ends via `MILK_RUNNABLE` descriptors.
 *
 * Return: void
 */
void milk_register_builtin_commands(void)
{
    milk_module_begin();
    milk_register_runnable(&unit_ci);
    milk_register_runnable(&unit_fpsdump);
    milk_module_end();

    milk_register_runnable(&unit_help);
}

/**
 * milk_session_load_modules - Load core modules and configured extra modules at startup
 *
 * Loads the `COREMOD` shared libraries (`memory`, `iofits`, `arith`, `tools`) followed
 * by the comma/space/semicolon-separated list in `$MILKCLI_ADD_LIBS`, if any ("none" names
 * none). Every other installed module loads on first use of one of its commands.
 *
 * Return: void
 */
void milk_session_load_modules(void)
{
    load_module_shared("milkCOREMODmemory");
#ifdef USE_CFITSIO
    load_module_shared("milkCOREMODiofits");
#endif
    load_module_shared("milkCOREMODarith");
    load_module_shared("milkCOREMODtools");

    const char *extra = getenv("MILKCLI_ADD_LIBS");
    if (extra == NULL || extra[0] == '\0' || strcmp(extra, "none") == 0)
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
