// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_cmd_args.h
 * @brief Per-call command arguments: the call's meta-options, session defaults, presets, typed values.
 *
 * Every call starts from the declared defaults overlaid with the session defaults (`cmd --default .tag=v`);
 * nothing carries over from one call to the next. CLI_checkarg_array() maps the call's tokens onto the
 * parameters.
 */

#ifndef MILK_CMD_ARGS_H
#define MILK_CMD_ARGS_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "libmilkdata/milkdata_clicmd.h"
#include "milk_argval.h"

/* Where a parameter's value comes from in the current call (low bits of milk_call_t.src[]) */
enum
{
    MILK_ARGSRC_DECL,    /* declared default */
    MILK_ARGSRC_SESSION, /* session default */
    MILK_ARGSRC_FPS,     /* current value of the shared FPS the call runs on */
    MILK_ARGSRC_PRESET,  /* --load=NAME */
    MILK_ARGSRC_DOT,     /* `.`: the default, given explicitly */
    MILK_ARGSRC_ARG,     /* positional value */
    MILK_ARGSRC_TAG      /* .tag=value */
};
#define MILK_ARGSRC_NAMED 0x80 /* flag: given as .tag=... */

typedef struct
{
    int    cmdi;        /* data.cmd[] index */
    int    set_default; /* --default: store the values given as session defaults, do not run */
    int    no_run;      /* --no-run: resolve and print the call, do not run */
    char   save[64];    /* --save=NAME, "" when absent */
    char   load[64];    /* --load=NAME, "" when absent */
    int    posonly;     /* tokens from this index on are positional (after "--") */
    char **raw;         /* raw[i]: argument string of token i */
    int    nraw;
    unsigned char *src; /* MILK_ARGSRC_* of each parameter */
    int            usage_error;
} milk_call_t;

/** milk_args_begin() - start a call of data.cmd[@cmdi]: take its meta-options out of @argv, reset its values.
 * Returns 0, or 2 after reporting a usage error. */
int milk_args_begin(int cmdi, int *argc, char **argv);

/** milk_args_end() - end the call: restore the defaults; 1 if it failed on its arguments (exit status 2). */
int milk_args_end(void);

/** milk_args_call() - the call of data.cmd[@cmdi] in progress, or NULL. */
milk_call_t *milk_args_call(int cmdi);

/** milk_args_named() - 1 if @s is a named argument `.tag=value`. */
int milk_args_named(const char *s);

/** milk_args_find() - index of the parameter of @cmdi named @tag (@len chars, no leading dot), or -1. */
int milk_args_find(int cmdi, const char *tag, size_t len);

/** milk_args_register() - data.cmd[@cmdi] was just registered: declared defaults, no session defaults. */
void milk_args_register(int cmdi);

/** milk_args_reset() - data.cmd[@cmdi]'s values := declared defaults overlaid with the session defaults. */
void milk_args_reset(int cmdi);

/** milk_args_tag() - @fpstag without its leading dot (the name written after `.` in `.name=value`). */
const char *milk_args_tag(const char *fpstag);

/** milk_args_session_set() - set (@val != NULL) or clear the session default of parameter @argi of @cmdi. */
int milk_args_session_set(int cmdi, int argi, const char *val);

/** milk_args_session_get() - the session default of parameter @argi of @cmdi, or NULL. */
const char *milk_args_session_get(int cmdi, int argi);

/** milk_args_preset_save() - write data.cmd[@cmdi]'s current values as preset @name; 0 or -1. */
int milk_args_preset_save(int cmdi, const char *name);

/** milk_args_preset_load() - apply preset @call->load to the call's values; 0, or -1 after reporting why. */
int milk_args_preset_load(milk_call_t *call);

/** milk_args_preset_list() - print `  Presets: NAME ...` for the saved presets of @cmdi, if any. */
void milk_args_preset_list(int cmdi, FILE *fp);

/** milk_args_print_call() - print `key .tag=value ...` (shell-quoted); @session: session defaults only. */
void milk_args_print_call(int cmdi, int session, FILE *fp);

#endif /* MILK_CMD_ARGS_H */
