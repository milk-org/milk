// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_keywords.h
 * @brief The language keyword table (`@` words) and the bash words the REPL highlights and
 *        completes.
 *
 * Purpose:     each `@` keyword is defined once (where it works, how it is implemented, its help
 *              and its completion), and every consumer derives from that one table.
 * Does:        holds the keyword table and the bash word list; looks keywords up; prints a
 *              keyword's help, its usage line on a usage error, and the keyword list; tells
 *              whether a keyword's arguments are calculator text.
 * Used by:     the milk.bash keyword functions (`milkls --keywords --bash`); help
 *              (CLIcore_help_command.c, CLIcore_help_topics.c); completion and highlighting
 *              (milk_server_complete.c, CLIcore_UI_completion_tab.c); the reserved-name check
 *              (milk_cmd_index.c); milk-call (milk_call_main.c); the libmilkbash builtins and
 *              client (bash/, client/, ipc/milk_srvlog.c); the server's usage errors.
 * Uses:        libc only.
 * See also:    milk_keywords.c; module commands come from the command registry (data.cmd) and
 *              entity namespaces from milk_entity_ns.h; docs/guide/scripting.md, "Keywords".
 * Constraints: compiled into libmilkbash.so, the thin client every shell loads: no server code
 *              or state.
 */

#ifndef MILK_KEYWORDS_H
#define MILK_KEYWORDS_H

#include <stdio.h>

/* Front-ends a keyword works in */
#define MILK_KW_CLI 0x1    /* milk-cli: prompt, -s, -c, piped input */
#define MILK_KW_SCRIPT 0x2 /* milk-script */
#define MILK_KW_BASH 0x4   /* plain bash after `source milk.bash` */
#define MILK_KW_ALL (MILK_KW_CLI | MILK_KW_SCRIPT | MILK_KW_BASH)

/* How `@name` is implemented */
typedef enum
{
    MILK_KW_BUILTIN, /* the libmilkbash builtin milk<name> */
    MILK_KW_SHELL,   /* the milk.bash function _milk_<name>_impl */
    MILK_KW_SERVER   /* `milkcall <name>`: a command handled by the milk-cli server */
} milk_kw_impl_t;

typedef struct
{
    const char    *name; /* without the sigil */
    milk_kw_impl_t impl;
    unsigned       avail;    /* MILK_KW_* front-ends */
    int            callable; /* implemented by milk-call (gets a bin-farm link) */
    const char    *synopsis;
    const char    *summary;
    const char    *details;  /* "" when none */
    const char    *examples; /* "" when none */
    const char    *options;  /* completion: its options, "-v --json ..." (NULL: none) */
    const char    *words;    /* completion: the words of its first argument, then "<w>: ..." lines:
                                the words after <w> ("*:" after any other first argument) */
} milk_keyword_t;

/** Language keywords, terminated by an entry with name == NULL. Functions, not exported arrays: an
 * executable would otherwise copy-relocate the arrays at their link-time size. */
const milk_keyword_t *milk_lexicon_keywords(void);

/** milk_lexicon_keyword() - the keyword called @name (with or without the sigil), or NULL. */
const milk_keyword_t *milk_lexicon_keyword(const char *name);

/** milk_lexicon_frontend() - MILK_KW_* bit of the running front-end, from $_MILK_FRONTEND (cli|script). */
unsigned milk_lexicon_frontend(const char *frontend);

/** milk_lexicon_print_help() - the help of keyword @k (`@help <kw>`, `@<kw> --help`). */
void milk_lexicon_print_help(const milk_keyword_t *k, FILE *fp);

/** milk_lexicon_print_list() - the keywords, one per line: name and summary, or (@synopsis) synopsis and
 * summary; @callable_only: those milk-call implements. */
void milk_lexicon_print_list(FILE *fp, int callable_only, int synopsis);

/** milk_lexicon_usage() - print "usage: <synopsis>" of keyword @name (a usage error); return 2. */
int milk_lexicon_usage(const char *name, FILE *fp);

/** milk_lexicon_help_arg() - 1 if @arg asks keyword @k for its help: `--help`, or `-h` unless @k takes an
 * expression (`@calc -h` computes -h). */
int milk_lexicon_help_arg(const milk_keyword_t *k, const char *arg);

/** milk_lexicon_takes_expr() - 1 if the arguments of keyword @k are calculator text (@calc @test
 * @assert). */
int milk_lexicon_takes_expr(const milk_keyword_t *k);

/** Bash reserved words and builtins that the REPL highlights and completes, NULL-terminated. */
const char *const *milk_lexicon_bash_words(void);

#endif /* MILK_KEYWORDS_H */
