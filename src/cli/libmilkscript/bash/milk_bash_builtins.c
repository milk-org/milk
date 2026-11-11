// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_bash_builtins.c
 * @brief Loadable GNU Bash builtins for milk (`enable -f libmilkbash.so ...`)
 *
 * Purpose:     the keywords that read or write shared memory and evaluate scalars, and the calls
 *              to the session, run in the shell's own process: no fork and no shell function
 *              per call, and results go straight into shell variables.
 * Does:        defines the `struct builtin` that bash looks up for each name: milkcall,
 *              milkcomplete, milkshutdown, milksession, milkcloexec, milktranslate, and, for each
 *              keyword in MILK_BUILTIN_KEYWORDS, milk<kw> and the keyword builtin @<kw> (loaded
 *              as kw<kw>), whose --help prints the keyword's help. Each runs the client command
 *              of its name (milk_client_cmd()) through client_builtin(), which raises again a
 *              SIGINT that the client caught once bash's handler is back. Bash's `help` shows
 *              the keyword table's summary and synopsis.
 * Used by:     bash, through `enable -f` in scripts/milk.bash: every shell that sources it,
 *              including the co-process of milk-cli and milk-script and the milk.script unit.
 * Uses:        the client calls (milk_client.h), the bash ABI and hooks (milk_bash_defer.h),
 *              the keyword table (milk_keywords.h).
 * See also:    milk_bash_hooks.c, milk_bash_defer.c; the milkbash target in
 *              src/cli/libmilkscript/CMakeLists.txt; docs/guide/scripting.md, "Keywords".
 * Constraints: compiled into libmilkbash.so only, with hidden visibility: bash finds a builtin
 *              by its `<name>_struct` symbol, the only ones exported. MILK_BUILTIN_KEYWORDS must
 *              name the keyword table's MILK_KW_BUILTIN entries, as __MILK_KWBUILTINS in
 *              milk.bash does.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "milk_bash_defer.h"
#include "milk_client.h"
#include "milk_keywords.h"

/* ============================================================
 *  Client Commands as Builtins
 * ============================================================ */

/**
 * client_builtin - Run client command @name (milk_client_cmd()) as a bash builtin
 * @name: Command name: `get` for `milkget` and `@get`
 * @list: The builtin's arguments
 *
 * The command gets the words as its argv and bash's variables through the hooks: no fork, and
 * results go straight into shell variables. A SIGINT the client caught is re-raised once
 * bash's handler is back, so bash reacts to it (INT trap, or aborting the command line) as if
 * it had received it.
 *
 * Return: The command's exit status
 */
static int client_builtin(const char *name, WORD_LIST *list)
{
    char              **argv = NULL;
    int                 argc = wordlist_to_argv(list, &argv);
    milk_client_hooks_t hooks;
    make_bash_hooks(&hooks);
    int rc = milk_client_cmd(name)->fn((argv != NULL) ? argc : 0, argv, &hooks);
    free(argv);
    if (g_sigint_pending)
    {
        g_sigint_pending = 0;
        raise(SIGINT);
    }
    return rc;
}

#define MILK_CLIENT_BUILTIN(name)                          \
    static int milk##name##_builtin_entry(WORD_LIST *list) \
    {                                                      \
        return client_builtin(#name, list);                \
    }
MILK_CLIENT_BUILTIN(call)
MILK_CLIENT_BUILTIN(get)
MILK_CLIENT_BUILTIN(set)
MILK_CLIENT_BUILTIN(calc)
MILK_CLIENT_BUILTIN(test)
MILK_CLIENT_BUILTIN(exists)
MILK_CLIENT_BUILTIN(waitfor)
MILK_CLIENT_BUILTIN(waitany)
MILK_CLIENT_BUILTIN(grab)
MILK_CLIENT_BUILTIN(complete)
MILK_CLIENT_BUILTIN(shutdown)
MILK_CLIENT_BUILTIN(session)
MILK_CLIENT_BUILTIN(log)
MILK_CLIENT_BUILTIN(ls)
MILK_CLIENT_BUILTIN(sleep)

/* ============================================================
 *  Builtin Registration Structs for `enable -f`
 * ============================================================ */

static char *const milkcall_doc[]      = { "Dispatch a milk command to milk-cli --server.", NULL };
static char *const milkcomplete_doc[]  = { "Query milk tab-completion candidates.", NULL };
static char *const milkshutdown_doc[]  = { "Shut down the session milk-cli --server.", NULL };
static char *const milksession_doc[]   = { "Manage milk sessions (@session).", NULL };
static char *const milkcloexec_doc[]   = { "Mark file descriptors close-on-exec.", NULL };
static char *const milktranslate_doc[] = { "Translate a milk script to source, or a typed line.",
                                           NULL };

/**
 * keyword_builtin - Dispatch `@<kw> args`, printing keyword help on `-h`/`--help`
 * @kw:   Keyword name without leading `@`
 * @fn:   Underlying builtin entry function
 * @list: Arguments passed to the keyword builtin
 *
 * The keywords implemented by a builtin are loaded as `kw<kw>` and named `@<kw>`
 * (`enable -f libmilkbash.so kwget` defines `@get`) so calls avoid shell function overhead.
 *
 * Return: Execution status of the keyword builtin
 */
static int keyword_builtin(const char *kw, sh_builtin_func_t *fn, WORD_LIST *list)
{
    const milk_keyword_t *k = NULL;
    if (list != NULL && list->word != NULL && list->word->word != NULL &&
        list->word->word[0] == '-' && (k = milk_lexicon_keyword(kw)) != NULL &&
        milk_lexicon_help_arg(k, list->word->word))
    {
        milk_lexicon_print_help(k, stdout);
        fflush(stdout);
        return EXECUTION_SUCCESS;
    }
    return fn(list);
}

/* The keywords bash runs as builtins, the keyword table's MILK_KW_BUILTIN entries: X(name,
 * function) makes milk<name> and the keyword builtin @<name>, whose help bash's `help` shows comes
 * from the keyword table */
#define MILK_BUILTIN_KEYWORDS(X)          \
    X(calc, milkcalc_builtin_entry)       \
    X(test, milktest_builtin_entry)       \
    X(get, milkget_builtin_entry)         \
    X(set, milkset_builtin_entry)         \
    X(exists, milkexists_builtin_entry)   \
    X(ls, milkls_builtin_entry)           \
    X(waitfor, milkwaitfor_builtin_entry) \
    X(waitany, milkwaitany_builtin_entry) \
    X(sleep, milksleep_builtin_entry)     \
    X(grab, milkgrab_builtin_entry)       \
    X(log, milklog_builtin_entry)         \
    X(defer, milkdefer_builtin)           \
    X(args, milkargs_builtin)

#define MILK_KEYWORD_BUILTIN(kw, fn)                 \
    static int kw##_keyword_builtin(WORD_LIST *list) \
    {                                                \
        return keyword_builtin(#kw, fn, list);       \
    }
MILK_BUILTIN_KEYWORDS(MILK_KEYWORD_BUILTIN)

/* until the constructor below sets the keyword table's text */
static char *const lexicon_doc_pending[] = { "A milk @ keyword (@help <keyword>).", NULL };

/* Built with hidden visibility: bash finds each builtin by its `<name>_struct` symbol */
#pragma GCC visibility push(default)
struct builtin milkcall_struct = { "milkcall",
                                   milkcall_builtin_entry,
                                   BUILTIN_ENABLED,
                                   milkcall_doc,
                                   "milkcall [--async] [--frontend] [--nowait] <cmd> [args...]",
                                   0 };


struct builtin milkcomplete_struct = {
    "milkcomplete",   milkcomplete_builtin_entry,        BUILTIN_ENABLED,
    milkcomplete_doc, "milkcomplete <prefix> [cmd|arg]", 0
};

struct builtin milkshutdown_struct = { "milkshutdown",  milkshutdown_builtin_entry,
                                       BUILTIN_ENABLED, milkshutdown_doc,
                                       "milkshutdown",  0 };

struct builtin milksession_struct = { "milksession",   milksession_builtin_entry, BUILTIN_ENABLED,
                                      milksession_doc, "milksession [verb ...]",  0 };


struct builtin milkcloexec_struct = { "milkcloexec",
                                      milkcloexec_builtin,
                                      BUILTIN_ENABLED,
                                      milkcloexec_doc,
                                      "milkcloexec FD... | --watcher PARENT",
                                      0 };

struct builtin milktranslate_struct = { "milktranslate",
                                        milktranslate_builtin,
                                        BUILTIN_ENABLED,
                                        milktranslate_doc,
                                        "milktranslate -v VAR FILE | -s VAR TEXT | --done PATH",
                                        0 };

/* milk<kw> and the keyword builtin @<kw>, loaded as kw<kw> (bash takes the name from the struct) */
#define MILK_KEYWORD_STRUCTS(name, fn)                                                         \
    struct builtin milk##name##_struct = { "milk" #name,        fn,           BUILTIN_ENABLED, \
                                           lexicon_doc_pending, "milk" #name, 0 };             \
    struct builtin kw##name##_struct   = { "@" #name,       name##_keyword_builtin,            \
                                           BUILTIN_ENABLED, lexicon_doc_pending,               \
                                           "@" #name,       0 };
MILK_BUILTIN_KEYWORDS(MILK_KEYWORD_STRUCTS)
#pragma GCC visibility pop

/**
 * lexicon_doc - Give builtins @plain (milk<kw>) and @kw (@<kw>) the keyword's summary and synopsis
 * @plain: Plain builtin
 * @kw:    Keyword builtin
 * @name:  Keyword name
 * @doc:   Two-entry storage for the long help ({summary, NULL})
 *
 * Return: void
 */
static void lexicon_doc(struct builtin *plain, struct builtin *kw, const char *name, char *doc[2])
{
    const milk_keyword_t *k = milk_lexicon_keyword(name);
    if (k != NULL)
    {
        doc[0]          = (char *) k->summary;
        doc[1]          = NULL;
        plain->long_doc = kw->long_doc = doc;
        plain->short_doc = kw->short_doc = k->synopsis;
    }
}

/* bash's `help @get` and `help milkget` print the keyword table's text: set before bash reads the
 * structs */
#define MILK_KEYWORD_DOC(name, fn)                                         \
    {                                                                      \
        static char *doc[2];                                               \
        lexicon_doc(&milk##name##_struct, &kw##name##_struct, #name, doc); \
    }
__attribute__((constructor)) static void builtin_docs_from_lexicon(void)
{
    MILK_BUILTIN_KEYWORDS(MILK_KEYWORD_DOC)
}
