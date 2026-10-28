// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_prog_internal.h
 * @brief Internal structs and cross-file prototypes for multi-statement calculator programs.
 */

#ifndef MILK_CLI_CALC_PROG_INTERNAL_H
#define MILK_CLI_CALC_PROG_INTERNAL_H

#include <stddef.h>

#include "CLIcore.h"
#include "cli_calc_prog.h"

#define PROG_MAXSTMT 128
#define HIST_MAX 8
#define FOR_MAX_ITER 100000
#define HIST_MARK '\001' /* first character of an appended ring-buffer update */

enum
{
    STMT_NORMAL,
    STMT_SKIP,      /* `skip`: end this tick here */
    STMT_POST,      /* `post(name)`: post that output this tick even if no statement wrote it */
    STMT_HIST,      /* a ring-buffer update appended by rewrite_history(): runs even after `skip` */
    STMT_NOP,       /* `def f(x) = ...` or statement merged into preceding kept program */
    STMT_IF_BEGIN,  /* `if <cond> {`: evaluates guard; if false, jumps to `jump_idx` */
    STMT_ELSE_JUMP, /* end of taken `if`/`elif` block: jumps to `jump_idx` (after `if` chain) */
    STMT_FOR_BEGIN, /* `for k = a:b[:s] {`: checks loop bound; if done, jumps to `jump_idx` */
    STMT_FOR_END    /* end of `for` body: increments induction variable and jumps to `jump_idx` */
};

typedef struct
{
    char *text;
    void *kept;       /* fused program, or NULL: evaluated as text */
    char *guard;      /* `if <cond>:` as the statement `__g<i> = (<cond>)`, or NULL */
    void *guard_kept; /* its kept form, or NULL: evaluated as text */
    long  guard_vid;  /* the variable __g<i> the guard sets */
    char *for_step;   /* `for` step statement `__fstep<i> = (<step>)`, or NULL */
    void *for_step_kept;
    long  for_step_vid;
    long  for_var_vid;
    int   for_iters;
    int   jump_idx;    /* target statement index for STMT_IF_BEGIN / STMT_ELSE_JUMP / STMT_FOR_* */
    int   block_id;    /* basic-block ID so cross-statement fusion never crosses control flow */
    int   is_local;    /* 1 for `local x = ...` or `L:x = ...` (do not promote to SHM) */
    int   kind;        /* STMT_* */
    int   merged_into; /* 1-based index of the statement whose fused pass runs this one, or 0 */
    char  post[STRINGMAXLEN_IMAGE_NAME];
    char  dst_name[STRINGMAXLEN_IMAGE_NAME];
    char  why[200]; /* why the statement is not kept (evaluated as text), "" when kept */
} prog_stmt;

struct cli_calc_prog
{
    prog_stmt         st[PROG_MAXSTMT];
    int               n;
    int               next_block_id;
    cli_calc_binding *vars;
    long             *vid; /* calculator variable of each binding */
    unsigned char    *assigned;
    int               nvars;
    void             *processinfo; /* the unit's PROCESSINFO for output posts, or NULL */
};

int        cli_calc_prog_split(cli_calc_prog *p, const char *text, char *err, size_t errsz);
int        cli_calc_prog_parse_multi_targets(const char *s,
                                             char        names[][STRINGMAXLEN_IMAGE_NAME],
                                             int         is_local[],
                                             int         max_n);
int        cli_calc_prog_classify_target(cli_calc_prog          *p,
                                         int                     stmt_idx,
                                         const char             *stmt,
                                         const cli_calc_binding *vars,
                                         char                   *name,
                                         size_t                  namesz,
                                         int                    *out_op,
                                         int                    *out_bind,
                                         char                   *err,
                                         size_t                  errsz);
int        cli_calc_prog_parse_prefix(cli_calc_prog *p, int i, char *err, size_t errsz);
int        cli_calc_prog_text_mentions_ident(const char *text, const char *ident);
imageID    cli_calc_prog_connect_stream(const char *name);
void       cli_calc_prog_merge_basic_blocks(cli_calc_prog *p);
char      *cli_calc_prog_rewrite_history(const char *text, char *err, size_t errsz);
extern int cli_calc_prog_default_shared;
extern int cli_calc_prog_quiet; /* 1: no 'statement not kept' notes at compile */

#endif /* MILK_CLI_CALC_PROG_INTERNAL_H */
