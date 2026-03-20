/**
 * @file CLIcore_script.h
 * @brief CLI scripting engine — variables, FPS access,
 *        flow control, functions
 */

#ifndef CLICORE_SCRIPT_H
#define CLICORE_SCRIPT_H

#include "CLIcore.h"

/* ---- CLI Variable Storage ---- */

#define CLI_VAR_NAMELEN   64
#define CLI_VAR_VALLEN   512
#define CLI_MAX_VARS     256

typedef struct
{
    char name[CLI_VAR_NAMELEN];
    char val[CLI_VAR_VALLEN];
    int  used;
} CLI_VAR;

/* Global variable table */
extern CLI_VAR cli_vars[CLI_MAX_VARS];
extern int     cli_last_retval;

/* ---- Variable Functions ---- */

/** Look up a CLI variable by name.
 *  Returns pointer to value string, or NULL. */
const char *cli_var_get(const char *name);

/** Set a CLI variable. Creates if new. */
void cli_var_set(
    const char *name,
    const char *val
);

/** Remove a CLI variable. */
void cli_var_unset(const char *name);

/* ---- CLI Commands ---- */

/** unset VAR — remove a variable */
errno_t cli_cmd_unset(void);

/** vars — list all variables */
errno_t cli_cmd_vars(void);

/** echo — print arguments */
errno_t cli_cmd_echo(void);

/* ---- Expansion Functions ---- */

/** Expand CLI variables ($VAR, ${VAR}, $?)
 *  before falling through to env vars.
 *  Called from cli_expand_env(). */
const char *cli_var_lookup(const char *name);

/** Expand @fpsname.param tokens in place. */
void cli_expand_fpsvar(
    char *line,
    int   maxlen
);

/** Expand $(( expr )) arithmetic in place. */
void cli_expand_arith(
    char *line,
    int   maxlen
);

/** Check if line is a variable assignment
 *  (VAR=val). Returns 1 if handled. */
int cli_try_var_assign(const char *line);

#endif /* CLICORE_SCRIPT_H */
