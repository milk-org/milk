// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_script.h
 * @brief CLI variable storage and runtime inspection declarations
 */

#ifndef CLICORE_SCRIPT_H
#define CLICORE_SCRIPT_H

#include "CLIcore.h"

/* ============================================================
 *  CLI Variables
 * ============================================================
 */

#define CLI_MAX_VARS 256
#define CLI_VAR_NAMELEN 64
#define CLI_VAR_VALLEN 512

typedef struct
{
    char name[CLI_VAR_NAMELEN];
    char val[CLI_VAR_VALLEN];
    int  used;
    int  readonly; /* 1 if marked readonly */
    int  type;     /* 0: double, 1: long, 2: string */
    union
    {
        double f;
        long   l;
    } num;
} CLI_VAR;

extern CLI_VAR cli_vars[CLI_MAX_VARS];
extern int     cli_last_retval;
extern int     cli_float_digits;

const char *cli_var_get(const char *name);

void cli_var_set(const char *name, const char *val);

void cli_var_unset(const char *name);

const char *cli_var_lookup(const char *name);

void cli_export_vars_to_env(void);

/* ============================================================
 *  System Inspection & Process Control Commands
 * ============================================================
 */

errno_t cli_cmd_fpslist(void);
errno_t cli_cmd_fpsdump(void);
errno_t cli_cmd_streamlist(void);
errno_t cli_cmd_proclist(void);
errno_t cli_cmd_milkquery(void);

int  cli_intercept_cmd_procctl(const char *p);
int  cli_intercept_cmd_procwait(const char *p);
int  cli_intercept_cmd_procstat(const char *p);
int  cli_intercept_cmd_dpdigits(const char *p);
void cli_listim_glob(const char *pat);
int  milk_ls_exec(int argc, char **argv);

#endif /* CLICORE_SCRIPT_H */
