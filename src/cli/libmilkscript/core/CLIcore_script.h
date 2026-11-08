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
#define CLI_VAR_NAMELEN 80 /* as calculator variable and image names */
#define CLI_VAR_VALLEN 512

/* A string variable of the calculator (its numbers live in dcvar) */
typedef struct
{
    char name[CLI_VAR_NAMELEN];
    char val[CLI_VAR_VALLEN];
    int  used;
    int  readonly; /* 1 if marked readonly */
} CLI_VAR;

extern CLI_VAR cli_vars[CLI_MAX_VARS];
extern int     cli_last_retval;
extern int     cli_float_digits;
extern int     cli_calc_mode;

const char *cli_var_get(const char *name);

/** cli_var_set() - set string variable @name to @val (NULL removes it). */
void cli_var_set(const char *name, const char *val);

void cli_export_vars_to_env(void);

/* ============================================================
 *  System Inspection & Process Control Commands
 * ============================================================
 */

errno_t cli_cmd_fpsdump(void);

void cli_listim_glob(const char *pat);
int  milk_ls_exec(int argc, char **argv);

#endif /* CLICORE_SCRIPT_H */
