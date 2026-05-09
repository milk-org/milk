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
    int  type;  /**< 0: double, 1: long, 2: string */
    union
    {
        double f;
        long   l;
    } num;
    int  used;
} CLI_VAR;

extern CLI_VAR cli_vars[CLI_MAX_VARS];
extern int     cli_last_retval;
extern int     cli_return_flag;
extern int     cli_break_flag;
extern int     cli_continue_flag;

/* ---- set flags ---- */
extern int     cli_flag_errexit;  /* set -e */
extern int     cli_flag_xtrace;   /* set -x */

/** Number of significant digits for
 *  double display (1–17, default 15). */
extern int     cli_float_digits;

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

/* ---- Associative Array Variables ---- */

#define CLI_ASSOC_MAXELEM 128
#define CLI_MAX_ASSOC      32

typedef struct
{
    char name[CLI_VAR_NAMELEN];
    char keys[CLI_ASSOC_MAXELEM][
        CLI_VAR_NAMELEN];
    char vals[CLI_ASSOC_MAXELEM][
        CLI_VAR_VALLEN];
    int  nelem;
    int  used;
} CLI_ASSOC_ARRAY;

extern CLI_ASSOC_ARRAY
    cli_assoc[CLI_MAX_ASSOC];

/* ---- Local Variable Scope Stack ---- */

#define CLI_MAX_LOCAL_DEPTH 32
#define CLI_MAX_LOCALS_PER_FUNC 64

typedef struct {
    char name[CLI_VAR_NAMELEN];
    char val[CLI_VAR_VALLEN];
    int  was_used;
} CLI_LOCAL_SHADOW;

extern CLI_LOCAL_SHADOW cli_local_shadows[CLI_MAX_LOCAL_DEPTH][CLI_MAX_LOCALS_PER_FUNC];
extern int cli_local_shadow_count[CLI_MAX_LOCAL_DEPTH];
extern int cli_local_depth;

/* Aliases use CLI_ALIAS from CLIcore.h
 * stored in data.alias[] */


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

/** Set an FPS parameter by name.
 *  Returns 0 on success, -1 on error. */
int cli_fps_set_param(
    const char *fpsname,
    const char *pname,
    const char *valstr
);

/** read — read a line into a variable */
errno_t cli_cmd_read(void);

/** export — push CLI var to environ */
errno_t cli_cmd_export(void);

/** shift — rotate positional parameters */
errno_t cli_cmd_shift(void);

/** printf — formatted output */
errno_t cli_cmd_printf(void);

/** fpslist — list live FPS instances */
errno_t cli_cmd_fpslist(void);

/** fpsdump — dump FPS params as key=value */
errno_t cli_cmd_fpsdump(void);

/** streamlist — list live SHM streams */
errno_t cli_cmd_streamlist(void);

/** proclist — list active processes */
errno_t cli_cmd_proclist(void);

/** milkquery — unified JSON system snapshot */
errno_t cli_cmd_milkquery(void);

/** defer — push cleanup command to stack */
errno_t cli_cmd_defer(void);

/** Run deferred cleanup commands (LIFO) */
void    cli_defer_run(void);

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

/** Export CLI variables to environment (for wordexp and shell sync) */
void cli_export_vars_to_env(void);

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
    CLI_BLOCK_SELECT,
    CLI_BLOCK_UNTIL
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

int cli_trap_signum(const char *name);

/** Evaluate a test expression [ ... ].
 *  Returns 1 if true, 0 if false. */
int cli_eval_test(const char *expr);

/** Execute a list of lines as a block. */
void cli_exec_lines(
    char lines[][STRINGMAXLEN_CLICMDLINE],
    int  nlines
);

/* -- Cross-file helpers (CLIcore_script_*.c) -- */

/** Strip leading whitespace. */
const char *strip_ws(const char *s);

/** Test if line starts with given prefix. */
int starts_with(
    const char *line,
    const char *prefix
);

/** Evaluate a condition line for if/elif. */
int eval_cond_line(
    const char *raw,
    int skip
);

/* CLIcore_script_flow.c */
void cli_exec_block_if(
    char lines[][STRINGMAXLEN_CLICMDLINE],
    int nlines
);
void cli_exec_block_while(
    char lines[][STRINGMAXLEN_CLICMDLINE],
    int nlines
);
void cli_exec_block_for(
    char lines[][STRINGMAXLEN_CLICMDLINE],
    int nlines
);
void cli_exec_block_select(
    char lines[][STRINGMAXLEN_CLICMDLINE],
    int nlines
);

/* CLIcore_script_case.c */
void cli_func_define(
    const char *name,
    char body[][STRINGMAXLEN_CLICMDLINE],
    int nbody
);
void cli_exec_block_case(
    char lines[][STRINGMAXLEN_CLICMDLINE],
    int nlines
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

/* ---- Engine Event Traps ---- */

#define CLI_ENGINE_TRAP_MAX    16
#define CLI_ETRAP_DEFAULT_MS  100

/** Engine event trap types */
enum
{
    CLI_ETRAP_STREAM = 0,
    CLI_ETRAP_FPS    = 1,
    CLI_ETRAP_PROC   = 2
};

/** Comparison operators for FPS traps */
enum
{
    CLI_ETRAP_OP_EQ = 0,
    CLI_ETRAP_OP_NE = 1,
    CLI_ETRAP_OP_GE = 2,
    CLI_ETRAP_OP_LE = 3
};

/**
 * @brief Engine event trap entry
 *
 * Registered via trap 'cmd' STREAM:name etc.
 * Polled non-blockingly between CLI commands.
 */
typedef struct
{
    int    used;
    int    type;          /**< CLI_ETRAP_* */
    char   target[128];   /**< stream/fps/proc */
    char   param[64];     /**< FPS param name */
    int    op;            /**< comparison op */
    int    has_cmp;       /**< 1 if op+value given */
    double cmpval;        /**< comparison value */
    int    proc_state;    /**< target loopstat */
    char   cmd[CLI_TRAP_CMDLEN];

    /* Throttle */
    long   min_interval_ms; /**< min ms between */
    int    max_fires;     /**< -1 = unlimited */
    int    fire_count;

    /* Runtime state */
    int      connected;
    IMAGE    img;
    uint64_t last_cnt0;
    FPS fps;
    char     last_fpsval[256];
    struct timespec last_fire_ts;
} CLI_ENGINE_TRAP;

extern CLI_ENGINE_TRAP
    cli_engine_traps[CLI_ENGINE_TRAP_MAX];

/** Poll all engine traps (non-blocking). */
void cli_engine_traps_poll(void);

/** Disconnect and clean up all engine traps. */
void cli_engine_traps_cleanup(void);

/* ---- Source Location Tracking ---- */

#define CLI_SRC_STACK_DEPTH 16

/**
 * @brief Source file location for error
 *        context display.
 */
typedef struct
{
    char file[256];
    int  line;
} CLI_SRC_LOC;

extern CLI_SRC_LOC
    cli_src_stack[CLI_SRC_STACK_DEPTH];
extern int cli_src_depth;

/** Print source location stack trace. */
void cli_print_source_trace(void);

#endif /* CLICORE_SCRIPT_H */
