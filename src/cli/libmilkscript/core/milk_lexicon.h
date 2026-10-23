// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_lexicon.h
 * @brief The language keyword table (`@` words) and the bash words the REPL highlights and completes.
 *
 * Every consumer derives from this table: the milk.bash keyword functions (`milkls --keywords --bash`),
 * keyword help, completion, highlighting and the exclusion lists. Registered milk commands come from
 * the command registry (data.cmd); entity namespaces from milk_entity_ns.h.
 */

#ifndef MILK_LEXICON_H
#define MILK_LEXICON_H

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
} milk_keyword_t;

/** Language keywords, terminated by an entry with name == NULL. Functions, not exported arrays: an
 * executable would otherwise copy-relocate the arrays at their link-time size. */
const milk_keyword_t *milk_lexicon_keywords(void);

/** milk_lexicon_keyword() - the keyword called @name (with or without the sigil), or NULL. */
const milk_keyword_t *milk_lexicon_keyword(const char *name);

/** milk_lexicon_frontend() - MILK_KW_* bit of the running front-end, from $_MILK_ENGINE (cli|script). */
unsigned milk_lexicon_frontend(const char *engine);

/** milk_lexicon_print_help() - the help of keyword @k (`@help <kw>`, `@<kw> --help`). */
void milk_lexicon_print_help(const milk_keyword_t *k, FILE *fp);

/** milk_lexicon_help_arg() - 1 if @arg asks keyword @k for its help: `--help`, or `-h` unless @k takes an
 * expression (`@calc -h` computes -h). */
int milk_lexicon_help_arg(const milk_keyword_t *k, const char *arg);

/** Bash reserved words and builtins that the REPL highlights and completes, NULL-terminated. */
const char *const *milk_lexicon_bash_words(void);

#endif /* MILK_LEXICON_H */
