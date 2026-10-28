// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_bash_defer.h
 * @brief Internal ABI types, hook helpers, and `@defer` support for `libmilkbash`.
 */

#ifndef MILK_BASH_DEFER_H
#define MILK_BASH_DEFER_H

#include <signal.h>

#include "milk_client.h"

/* ============================================================
 *  Minimal Vendored GNU Bash Loadable Builtin ABI
 * ============================================================ */

typedef struct word_desc
{
    char *word;
    int   flags;
} WORD_DESC;

typedef struct word_list
{
    struct word_list *next;
    WORD_DESC        *word;
} WORD_LIST;

typedef int sh_builtin_func_t(WORD_LIST *);

struct builtin
{
    char              *name;
    sh_builtin_func_t *function;
    int                flags;
    char *const       *long_doc;
    const char        *short_doc;
    char              *handle;
};

#define BUILTIN_ENABLED 0x01
#define EXECUTION_SUCCESS 0
#define EXECUTION_FAILURE 1

extern volatile sig_atomic_t g_sigint_pending;

/** Populate a milk client hook table with host Bash callbacks. */
void make_bash_hooks(milk_client_hooks_t *hooks);

/** Convert a Bash WORD_LIST linked list into a heap-allocated C argv array. */
int wordlist_to_argv(WORD_LIST *list, char ***out_argv);

/** Mark file descriptors close-on-exec (`milkcloexec` builtin). */
int milkcloexec_builtin(WORD_LIST *list);

/** Translate a `.milk` script or interactive line (`milktranslate` builtin). */
int milktranslate_builtin(WORD_LIST *list);

/** Register or run scoped `@defer` commands (`milkdefer` builtin). */
int milkdefer_builtin(WORD_LIST *list);

/** Parse and bind function/script arguments (`milkargs` builtin). */
int milkargs_builtin(WORD_LIST *list);

/**
 * milk_bash_defer_argv() - the milkdefer builtin: `[--] CMD...`, `--exit CMD...`,
 * `--restore SPEC...`, and for milk.bash `--return U`, `--orphans`, `--exit-run`
 * (status 1: evaluate __MILK_DRUN_SH).
 */
int milk_bash_defer_argv(int argc, char **argv);

#endif /* MILK_BASH_DEFER_H */
