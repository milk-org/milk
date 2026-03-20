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

extern CLI_VAR cli_vars[CLI_MAX_VARS];
extern int     cli_last_retval;
extern int     cli_return_flag;

/* ---- set flags ---- */
extern int     cli_flag_errexit;  /* set -e */
extern int     cli_flag_xtrace;   /* set -x */

/* ---- Array Variables ---- */

#define CLI_MAX_ARRAYS    64
#define CLI_ARRAY_MAXELEM 256

typedef struct
{
    char name[CLI_VAR_NAMELEN];
    char elem[CLI_ARRAY_MAXELEM][
        CLI_VAR_VALLEN];
    int  nelem;
    int  used;
} CLI_ARRAY;

extern CLI_ARRAY cli_arrays[CLI_MAX_ARRAYS];

/* ---- Variable Functions ---- */

/** Look up a CLI variable by name. */
const char *cli_var_get(const char *name);

/** Set a CLI variable. Creates if new. */
void cli_var_set(
    const char *name,
    const char *val
);

/** Remove a CLI variable. */
void cli_var_unset(const char *name);

/** Array assignment: arr=(val1 val2 ...) */
int cli_try_array_assign(const char *line);

/* ---- CLI Commands ---- */

/** unset VAR — remove a variable */
errno_t cli_cmd_unset(void);

/** vars — list all variables */
errno_t cli_cmd_vars(void);

/** echo — print arguments */
errno_t cli_cmd_echo(void);

/** fpsset — write FPS parameter */
errno_t cli_cmd_fpsset(void);

/* ---- Expansion Functions ---- */

/** Unified variable lookup: CLI > special > env */
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

/** Expand $VAR and ${VAR} in place.
 *  Defined in CLIcore_UI.c. */
void cli_expand_env(
    char *line,
    int   maxlen
);

/** Check if line is a variable assignment
 *  (VAR=val). Returns 1 if handled. */
int cli_try_var_assign(const char *line);

/* ---- Block Accumulator (flow control) ---- */

#define CLI_BLOCK_MAXLINES  1024
#define CLI_BLOCK_MAXDEPTH    16

/** Block types */
enum
{
    CLI_BLOCK_NONE = 0,
    CLI_BLOCK_IF,
    CLI_BLOCK_WHILE,
    CLI_BLOCK_FOR,
    CLI_BLOCK_FUNC,
    CLI_BLOCK_CASE,
    CLI_BLOCK_SELECT
};

/** Block accumulator state */
typedef struct
{
    int  type;
    int  depth;
    char lines[CLI_BLOCK_MAXLINES][
        STRINGMAXLEN_CLICMDLINE];
    int  nlines;
    int  active;
} CLI_BLOCK;

extern CLI_BLOCK cli_block_stack[CLI_BLOCK_MAXDEPTH];
extern int       cli_block_level;

/** Intercept a line for block accumulation.
 *  Returns 1 if consumed, 0 if not. */
int cli_script_intercept(const char *line);

/** Evaluate a test expression [ ... ].
 *  Returns 1 if true, 0 if false. */
int cli_eval_test(const char *expr);

/** Execute a list of lines as a block. */
void cli_exec_lines(
    char lines[][STRINGMAXLEN_CLICMDLINE],
    int  nlines
);

/* ---- User-Defined Functions ---- */

#define CLI_FUNC_MAXARGS  10
#define CLI_MAX_FUNCS     64
#define CLI_FUNC_NAMELEN  64

typedef struct
{
    char name[CLI_FUNC_NAMELEN];
    char body[CLI_BLOCK_MAXLINES][
        STRINGMAXLEN_CLICMDLINE];
    int  nbody;
    int  used;
} CLI_FUNC;

extern CLI_FUNC cli_funcs[CLI_MAX_FUNCS];

/** Look up user function by name. */
CLI_FUNC *cli_func_find(const char *name);

/** Try to call a user-defined function.
 *  Returns 1 if matched and called. */
int cli_try_func_call(const char *line);

/* ---- Trap Handlers ---- */

#define CLI_TRAP_MAXSIGS 8
#define CLI_TRAP_CMDLEN  512

typedef struct
{
    int  signum;
    char cmd[CLI_TRAP_CMDLEN];
    int  used;
} CLI_TRAP;

extern CLI_TRAP cli_traps[CLI_TRAP_MAXSIGS];

/** Execute trap handlers for given signal. */
void cli_trap_run(int signum);

/** Run EXIT traps (called at script end). */
void cli_trap_run_exit(void);

#endif /* CLICORE_SCRIPT_H */
