// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milkscript_api.c
 * @brief Public API implementation for the scripting engine
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/types.h>
#include <unistd.h>

#include "CLIcore.h"
#include "CLIcore_UI_execute.h"
#include "CLIcore_datainit.h"
#include "CLIcore_modules.h"
#include "CLIcore_setSHMdir.h"
#include "milk_bash_engine.h"
#include "milkscript.h"
#include "milk_session.h"

/**
 * @brief Initialize the milk scripting engine.
 */
errno_t milkscript_init(int argc, char **argv)
{
    if (argc > 0 && argv && argv[0])
    {
        strncpy(data.processname, argv[0], STRINGMAXLEN_PROCESSNAME - 1);
    }
    else
    {
        strncpy(data.processname, "milk-script", STRINGMAXLEN_PROCESSNAME - 1);
    }

    CLI_runtime_init();

    data.autocomplete         = 0;
    data.autocomplete_history = 0;
    data.autocomplete_arghint = 0;
    data.autocomplete_fuzzy   = 0;

    setSHMdir();
    CLI_data_init();

    milk_session_register_engine(); /* the reserved module `milk` comes first */
    milk_session_load_modules();

    return RETURN_SUCCESS;
}

/**
 * @brief Execute a single milk script command string.
 */
errno_t milkscript_execute(const char *cmdline)
{
    if (!cmdline)
    {
        return -1;
    }
    strncpy(data.CLIcmdline, cmdline, STRINGMAXLEN_CLICMDLINE - 1);
    data.CLIcmdline[STRINGMAXLEN_CLICMDLINE - 1] = '\0';
    return CLI_execute_line();
}

/**
 * @brief Run a milk script stream in the bash co-process engine.
 */
errno_t milkscript_run(FILE *fp)
{
    if (!fp)
    {
        return -1;
    }
    size_t cap = 8192;
    size_t len = 0;
    char  *buf = (char *) malloc(cap);
    if (buf == NULL)
    {
        return -1;
    }

    char line[STRINGMAXLEN_CLICMDLINE];
    int  first_line = 1;
    while (fgets(line, sizeof(line), fp))
    {
        size_t l = strlen(line);
        if (data.echo_input && l > 0)
        {
            int is_shebang = (first_line && line[0] == '#' && line[1] == '!');
            if (!is_shebang)
            {
                printf("\033[2;36m%s\033[0m", line);
                if (line[l - 1] != '\n')
                {
                    printf("\n");
                }
            }
        }
        first_line = 0;
        if (len + l + 1 > cap)
        {
            size_t ncap = (cap + l + 1) * 2;
            char  *nbuf = (char *) realloc(buf, ncap);
            if (nbuf == NULL)
            {
                free(buf);
                return -1;
            }
            buf = nbuf;
            cap = ncap;
        }
        memcpy(buf + len, line, l);
        len += l;
    }
    buf[len] = '\0';

    int exit_req = 0;
    int rc       = milk_bash_engine_exec(buf, &exit_req);
    free(buf);
    dcexitcode = rc;
    return (rc == 0) ? 0 : 1;
}

void milkscript_cleanup(void)
{
    milk_bash_engine_shutdown();
}
