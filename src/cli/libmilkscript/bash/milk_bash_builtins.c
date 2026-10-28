// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_bash_builtins.c
 * @brief Loadable GNU Bash builtins for milk (`enable -f libmilkbash.so ...`)
 *
 * Exports loadable builtins (`milkcall`, `milkget`, `milkset`, `milkcalc`, `milktest`,
 * `milkexists`, `milkwaitfor`, `milkwaitany`, `milkgrab`, `milkcomplete`, `milkshutdown`,
 * `milksession`, `milklog`, `milkls`, `milksleep`, `milkdefer`, `milkargs`, `milkcloexec`,
 * `milktranslate`) and `@<kw>` keyword builtins (`kw<kw>_struct`).
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
#include "milk_lexicon.h"

/* ============================================================
 *  IPC-Backed Builtin Entry Points
 * ============================================================ */

/**
 * milkcall_builtin - Bash builtin entry point for `milkcall`
 * @list: Arguments passed to the builtin as a Bash WORD_LIST
 *
 * Dispatches a milk CLI command (optionally with `--async`) to the session's
 * `milk-cli --server` daemon over IPC using in-process Bash variable hooks.
 *
 * Return: Command execution status (`EXECUTION_SUCCESS` or error code)
 */
static int milkcall_builtin(WORD_LIST *list)
{
    char **argv = NULL;
    int    argc = wordlist_to_argv(list, &argv);
    if (argc == 0 || argv == NULL)
    {
        free(argv);
        return EXECUTION_SUCCESS;
    }

    int flags     = 0;
    int start_idx = 0;
    for (; start_idx < argc && argv[start_idx][0] == '-'; start_idx++)
    {
        if (strcmp(argv[start_idx], "--async") == 0)
        {
            flags |= MILK_CALL_ASYNC;
        }
        else if (strcmp(argv[start_idx], "--engine") == 0)
        {
            flags |= MILK_CALL_ENGINE;
        }
        else if (strcmp(argv[start_idx], "--nowait") == 0)
        {
            flags |= MILK_CALL_NOWAIT;
        }
        else
        {
            break;
        }
    }
    if (start_idx >= argc)
    {
        free(argv);
        return EXECUTION_SUCCESS;
    }

    milk_client_hooks_t hooks;
    make_bash_hooks(&hooks);

    int rc = milk_client_call(argc - start_idx, argv + start_idx, flags, &hooks);
    free(argv);
    return rc;
}

/**
 * milkget_builtin - Bash builtin entry point for `milkget` (`@get`)
 * @list: Arguments passed to the builtin as a Bash WORD_LIST
 *
 * Queries stream, FPS or processinfo properties directly in-process
 * or via the session server and optionally binds results to shell variables.
 *
 * Return: 0 on success, non-zero on lookup or syntax failure
 */
static int milkget_builtin(WORD_LIST *list)
{
    char              **argv = NULL;
    int                 argc = wordlist_to_argv(list, &argv);
    milk_client_hooks_t hooks;
    make_bash_hooks(&hooks);
    int rc = milk_client_get_argv(argc, argv, &hooks);
    free(argv);
    return rc;
}

/**
 * milkset_builtin - Bash builtin entry point for `milkset` (`@set`)
 * @list: Arguments passed to the builtin as a Bash WORD_LIST
 *
 * Modifies FPS parameters, stream FITS keywords, or processinfo controls from
 * the shell without spawning an external process.
 *
 * Return: 0 on success, `EXECUTION_FAILURE` or error code on failure
 */
static int milkset_builtin(WORD_LIST *list)
{
    char **argv = NULL;
    int    argc = wordlist_to_argv(list, &argv);
    if (argc < 1 || argv == NULL)
    {
        free(argv);
        fprintf(stderr, "Usage: @set <entity> [=] [value] [comment] [...]\n");
        return EXECUTION_FAILURE;
    }

    milk_client_hooks_t hooks;
    make_bash_hooks(&hooks);
    int rc = milk_client_set_argv(argc, argv, &hooks);
    free(argv);
    return rc;
}

/**
 * milkcalc_builtin - Bash builtin entry point for `milkcalc` (`@calc`)
 * @list: Arguments passed to the builtin as a Bash WORD_LIST
 *
 * Evaluates a mathematical or image arithmetic expression via the milk client
 * and server, binding any resulting variables directly in the host shell.
 *
 * Return: 0 on success, non-zero on evaluation error
 */
static int milkcalc_builtin(WORD_LIST *list)
{
    char **argv = NULL;
    int    argc = wordlist_to_argv(list, &argv);
    if (argc == 0 || argv == NULL)
    {
        free(argv);
        return EXECUTION_SUCCESS;
    }
    milk_client_hooks_t hooks;
    make_bash_hooks(&hooks);
    int rc = milk_client_calc_argv(argc, argv, &hooks);
    free(argv);
    return rc;
}

/**
 * milktest_builtin - Bash builtin entry point for `milktest` (`@test`)
 * @list: Arguments passed to the builtin as a Bash WORD_LIST
 *
 * Concatenates @list into a boolean expression and evaluates it via
 * `milk_client_test` for use in shell conditionals (`if @test ...`).
 *
 * Return: 0 if the condition is true, 1 if false, 2 on error
 */
static int milktest_builtin(WORD_LIST *list)
{
    char **argv = NULL;
    int    argc = wordlist_to_argv(list, &argv);
    if (argc == 0 || argv == NULL)
    {
        free(argv);
        return 1;
    }

    size_t tot = 1;
    for (int i = 0; i < argc; i++)
    {
        tot += strlen(argv[i]) + 1;
    }
    char *expr = (char *) malloc(tot);
    if (expr == NULL)
    {
        free(argv);
        return 2;
    }
    expr[0]    = '\0';
    size_t pos = 0;
    for (int i = 0; i < argc; i++)
    {
        int n = snprintf(expr + pos, tot - pos, "%s%s", (i > 0) ? " " : "", argv[i]);
        if (n > 0)
        {
            pos += (size_t) n;
        }
    }

    milk_client_hooks_t hooks;
    make_bash_hooks(&hooks);
    int rc = milk_client_test(expr, &hooks);
    free(expr);
    free(argv);
    return rc;
}

/**
 * milkexists_builtin - Bash builtin entry point for `milkexists` (`@exists`)
 * @list: Arguments passed to the builtin as a Bash WORD_LIST
 *
 * Checks whether a milk stream, FPS or processinfo entity exists
 * (or is actively running when `-r` / `--running` is passed).
 *
 * Return: 0 if the entity exists (and is running if `-r`), 1 otherwise
 */
static int milkexists_builtin(WORD_LIST *list)
{
    char **argv = NULL;
    int    argc = wordlist_to_argv(list, &argv);
    if (argc == 0 || argv == NULL)
    {
        free(argv);
        return 1;
    }

    int         check_running = 0;
    const char *spec          = NULL;
    for (int i = 0; i < argc; i++)
    {
        if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--running") == 0)
        {
            check_running = 1;
        }
        else if (spec == NULL)
        {
            spec = argv[i];
        }
    }

    milk_client_hooks_t hooks;
    make_bash_hooks(&hooks);
    int rc = milk_client_exists(spec, check_running, &hooks);
    free(argv);
    return rc;
}

/**
 * milkwaitfor_builtin - Bash builtin entry point for `milkwaitfor` (`@waitfor`)
 * @list: Arguments passed to the builtin as a Bash WORD_LIST
 *
 * Waits in-process or via the session server for a stream to advance by N frames
 * or for an entity/boolean condition to be satisfied.
 *
 * Return: 0 on condition met, 124 on timeout, 130 on SIGINT, 1 on error
 */
static int milkwaitfor_builtin(WORD_LIST *list)
{
    char              **argv = NULL;
    int                 argc = wordlist_to_argv(list, &argv);
    milk_client_hooks_t hooks;
    make_bash_hooks(&hooks);
    int rc = milk_client_waitfor(argc, argv, &hooks);
    free(argv);
    return rc;
}

/**
 * milkwaitany_builtin - Bash builtin entry point for `milkwaitany` (`@waitany`)
 * @list: Arguments passed to the builtin as a Bash WORD_LIST
 *
 * Blocks until any one of the specified shared-memory streams updates and outputs
 * or binds the triggered stream's name.
 *
 * Return: 0 when a stream updates, 124 on timeout, 130 on SIGINT, 1 on error
 */
static int milkwaitany_builtin(WORD_LIST *list)
{
    char              **argv = NULL;
    int                 argc = wordlist_to_argv(list, &argv);
    milk_client_hooks_t hooks;
    make_bash_hooks(&hooks);
    int rc = milk_client_waitany(argc, argv, &hooks);
    free(argv);
    return rc;
}

/**
 * milkgrab_builtin - Bash builtin entry point for `milkgrab` (`@grab`)
 * @list: Arguments passed to the builtin as a Bash WORD_LIST
 *
 * Captures one or more consecutive frames from a stream into an image, 3D cube,
 * FITS file, or stdout via `milk_client_grab`.
 *
 * Return: 0 on success, 124 on timeout, 130 on SIGINT, 1 on error
 */
static int milkgrab_builtin(WORD_LIST *list)
{
    char              **argv = NULL;
    int                 argc = wordlist_to_argv(list, &argv);
    milk_client_hooks_t hooks;
    make_bash_hooks(&hooks);
    int rc = milk_client_grab(argc, argv, &hooks);
    free(argv);
    return rc;
}

/**
 * milkcomplete_builtin - Bash builtin entry point for `milkcomplete`
 * @list: Arguments passed to the builtin as a Bash WORD_LIST (`<prefix> [cmd|arg]`)
 *
 * Queries tab-completion candidates from the session server for interactive
 * readline completion in `milk.bash` and `milk-cli`.
 *
 * Return: 0 on success, non-zero on error
 */
static int milkcomplete_builtin(WORD_LIST *list)
{
    char      **argv      = NULL;
    int         argc      = wordlist_to_argv(list, &argv);
    const char *prefix    = (argc >= 1) ? argv[0] : "";
    const char *comp_type = (argc >= 2) ? argv[1] : "arg";

    milk_client_hooks_t hooks;
    make_bash_hooks(&hooks);
    int rc = milk_client_complete(prefix, comp_type, &hooks);
    free(argv);
    return rc;
}

/**
 * milkshutdown_builtin - Bash builtin entry point for `milkshutdown`
 * @list: Unused argument list
 *
 * Closes the shell's lifetime owner pipe and requests clean termination of the
 * session's `milk-cli --server` daemon.
 *
 * Return: 0 on success, non-zero on error
 */
static int milkshutdown_builtin(WORD_LIST *list)
{
    (void) list;
    milk_client_hooks_t hooks;
    make_bash_hooks(&hooks);
    return milk_client_shutdown(&hooks);
}

/**
 * milksession_builtin - Bash builtin entry point for `milksession` (`@session`)
 * @list: Arguments passed to the builtin as a Bash WORD_LIST
 *
 * Shows, starts, switches and ends sessions (milk-cli servers) from the current Bash shell.
 *
 * Return: 0 on success, non-zero on error
 */
static int milksession_builtin(WORD_LIST *list)
{
    char              **argv = NULL;
    int                 argc = wordlist_to_argv(list, &argv);
    milk_client_hooks_t hooks;
    make_bash_hooks(&hooks);
    int rc = milk_client_session(argc, argv, &hooks);
    free(argv);
    return rc;
}

/**
 * milklog_builtin - Bash builtin entry point for `milklog` (`@log`)
 * @list: Arguments passed to the builtin as a Bash WORD_LIST
 *
 * Queries, filters, or live-tails the shared-memory command ring-buffer log of
 * the active or specified milk server.
 *
 * Return: 0 on success, non-zero on error
 */
static int milklog_builtin(WORD_LIST *list)
{
    char              **argv = NULL;
    int                 argc = wordlist_to_argv(list, &argv);
    milk_client_hooks_t hooks;
    make_bash_hooks(&hooks);
    int rc = milk_client_log(argc, argv, &hooks);
    free(argv);
    return rc;
}

/**
 * milksleep_builtin - Bash builtin entry point for `milksleep` (`@sleep`)
 * @list: Arguments (`SECONDS` or `--every T VAR` / `--hz F VAR`)
 *
 * Sleeps in-process via `milk_client_sleep_argv` while allowing `SIGINT` to reach Bash.
 *
 * Return: 0 on completion, 130 on SIGINT, non-zero on error
 */
static int milksleep_builtin(WORD_LIST *list)
{
    char              **argv = NULL;
    int                 argc = wordlist_to_argv(list, &argv);
    milk_client_hooks_t hooks;
    make_bash_hooks(&hooks);
    int rc = milk_client_sleep_argv(argc, argv, &hooks);
    free(argv);
    return rc;
}

/**
 * milkls_builtin - Bash builtin entry point for `milkls` (`@ls`)
 * @list: Arguments passed to the builtin as a Bash WORD_LIST
 *
 * Lists active streams, local images, FPS instances and processinfo entries via
 * `milk_client_ls`.
 *
 * Return: 0 on success, non-zero on error
 */
static int milkls_builtin(WORD_LIST *list)
{
    char              **argv = NULL;
    int                 argc = wordlist_to_argv(list, &argv);
    milk_client_hooks_t hooks;
    make_bash_hooks(&hooks);
    int rc = milk_client_ls(argc, argv, &hooks);
    free(argv);
    return rc;
}

/* ============================================================
 *  Builtin Registration Structs for `enable -f`
 * ============================================================ */

static char *const milkcall_doc[]    = { "Dispatch a milk command to milk-cli --server.", NULL };
static char *const milkget_doc[]     = { "Query a milk stream/FPS/procinfo/seq property.", NULL };
static char *const milkset_doc[]     = { "Set a milk FPS parameter or stream FITS keyword.", NULL };
static char *const milkcalc_doc[]    = { "Evaluate a milk @calc expression.", NULL };
static char *const milktest_doc[]    = { "Evaluate a milk @test boolean expression.", NULL };
static char *const milkexists_doc[]  = { "Check if a milk entity exists or is running.", NULL };
static char *const milkwaitfor_doc[] = { "Wait for stream frames or an entity condition.", NULL };
static char *const milkwaitany_doc[] = { "Wait for any listed stream to update.", NULL };
static char *const milkgrab_doc[] = { "Capture consecutive stream frames into a 3D cube.", NULL };
static char *const milkcomplete_doc[] = { "Query milk tab-completion candidates.", NULL };
static char *const milkshutdown_doc[] = { "Shut down the session milk-cli --server.", NULL };
static char *const milksession_doc[]  = { "Manage milk sessions (@session).", NULL };
static char *const milklog_doc[]   = { "Inspect or search the milk server SHM log (@log).", NULL };
static char *const milkls_doc[]    = { "List active milk entities (@ls).", NULL };
static char *const milkdefer_doc[] = { "Run a command when the calling function or file returns.",
                                       NULL };
static char *const milkargs_doc[]  = { "Declare and check the arguments of a function or script.",
                                       NULL };
static char *const milkcloexec_doc[] = { "Mark file descriptors close-on-exec.", NULL };
static char *const milksleep_doc[]   = { "Sleep for SECONDS (fractional) without forking.", NULL };
static char *const milktranslate_doc[] = { "Translate a milk script to source, or a typed line.",
                                           NULL };

/* Builtin entry points: a SIGINT the client caught is re-raised once bash's handler is back, so
 * bash reacts to it (INT trap, or aborting the command line) as if it had received it. */
#define MILK_BUILTIN_ENTRY(fn)             \
    static int fn##_entry(WORD_LIST *list) \
    {                                      \
        int rc = fn(list);                 \
        if (g_sigint_pending)              \
        {                                  \
            g_sigint_pending = 0;          \
            raise(SIGINT);                 \
        }                                  \
        return rc;                         \
    }
MILK_BUILTIN_ENTRY(milkcall_builtin)
MILK_BUILTIN_ENTRY(milkget_builtin)
MILK_BUILTIN_ENTRY(milkset_builtin)
MILK_BUILTIN_ENTRY(milkcalc_builtin)
MILK_BUILTIN_ENTRY(milktest_builtin)
MILK_BUILTIN_ENTRY(milkexists_builtin)
MILK_BUILTIN_ENTRY(milkwaitfor_builtin)
MILK_BUILTIN_ENTRY(milkwaitany_builtin)
MILK_BUILTIN_ENTRY(milkgrab_builtin)
MILK_BUILTIN_ENTRY(milkcomplete_builtin)
MILK_BUILTIN_ENTRY(milkshutdown_builtin)
MILK_BUILTIN_ENTRY(milksession_builtin)
MILK_BUILTIN_ENTRY(milklog_builtin)
MILK_BUILTIN_ENTRY(milkls_builtin)
MILK_BUILTIN_ENTRY(milksleep_builtin)

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

#define MILK_KEYWORD_BUILTIN(kw, fn)                 \
    static int kw##_keyword_builtin(WORD_LIST *list) \
    {                                                \
        return keyword_builtin(#kw, fn, list);       \
    }
MILK_KEYWORD_BUILTIN(calc, milkcalc_builtin_entry)
MILK_KEYWORD_BUILTIN(test, milktest_builtin_entry)
MILK_KEYWORD_BUILTIN(get, milkget_builtin_entry)
MILK_KEYWORD_BUILTIN(set, milkset_builtin_entry)
MILK_KEYWORD_BUILTIN(exists, milkexists_builtin_entry)
MILK_KEYWORD_BUILTIN(ls, milkls_builtin_entry)
MILK_KEYWORD_BUILTIN(waitfor, milkwaitfor_builtin_entry)
MILK_KEYWORD_BUILTIN(waitany, milkwaitany_builtin_entry)
MILK_KEYWORD_BUILTIN(sleep, milksleep_builtin_entry)
MILK_KEYWORD_BUILTIN(grab, milkgrab_builtin_entry)
MILK_KEYWORD_BUILTIN(log, milklog_builtin_entry)
MILK_KEYWORD_BUILTIN(defer, milkdefer_builtin)
MILK_KEYWORD_BUILTIN(args, milkargs_builtin)

/* Built with hidden visibility: bash finds each builtin by its `<name>_struct` symbol */
#pragma GCC visibility push(default)
struct builtin milkcall_struct = { "milkcall",
                                   milkcall_builtin_entry,
                                   BUILTIN_ENABLED,
                                   milkcall_doc,
                                   "milkcall [--async] [--engine] [--nowait] <cmd> [args...]",
                                   0 };

struct builtin milkget_struct = { "milkget",
                                  milkget_builtin_entry,
                                  BUILTIN_ENABLED,
                                  milkget_doc,
                                  "milkget [-v var|-a arr|-A assoc|--json] <spec>",
                                  0 };

struct builtin milkset_struct = { "milkset",
                                  milkset_builtin_entry,
                                  BUILTIN_ENABLED,
                                  milkset_doc,
                                  "milkset <entity> [val] [comment]",
                                  0 };

struct builtin milkcalc_struct = { "milkcalc",   milkcalc_builtin_entry,     BUILTIN_ENABLED,
                                   milkcalc_doc, "milkcalc [-v var] <expr>", 0 };

struct builtin milktest_struct = { "milktest",   milktest_builtin_entry, BUILTIN_ENABLED,
                                   milktest_doc, "milktest <expr>",      0 };

struct builtin milkexists_struct = { "milkexists",   milkexists_builtin_entry,   BUILTIN_ENABLED,
                                     milkexists_doc, "milkexists [-r] <entity>", 0 };

struct builtin milkwaitfor_struct = { "milkwaitfor",
                                      milkwaitfor_builtin_entry,
                                      BUILTIN_ENABLED,
                                      milkwaitfor_doc,
                                      "milkwaitfor <target> [nframes] [--timeout <sec>]",
                                      0 };

struct builtin milkwaitany_struct = { "milkwaitany",
                                      milkwaitany_builtin_entry,
                                      BUILTIN_ENABLED,
                                      milkwaitany_doc,
                                      "milkwaitany <s1> <s2> [...] [--timeout <sec>]",
                                      0 };

struct builtin milkgrab_struct = {
    "milkgrab",
    milkgrab_builtin_entry,
    BUILTIN_ENABLED,
    milkgrab_doc,
    "milkgrab [--fits <f>] [--timestamps <t>] <stream> [nframes] [out_img] [--timeout <sec>]",
    0
};

struct builtin milkcomplete_struct = {
    "milkcomplete",   milkcomplete_builtin_entry,        BUILTIN_ENABLED,
    milkcomplete_doc, "milkcomplete <prefix> [cmd|arg]", 0
};

struct builtin milkshutdown_struct = { "milkshutdown",  milkshutdown_builtin_entry,
                                       BUILTIN_ENABLED, milkshutdown_doc,
                                       "milkshutdown",  0 };

struct builtin milksession_struct = { "milksession",   milksession_builtin_entry, BUILTIN_ENABLED,
                                      milksession_doc, "milksession [verb ...]",  0 };

struct builtin milklog_struct = { "milklog",   milklog_builtin_entry,  BUILTIN_ENABLED,
                                  milklog_doc, "milklog [options...]", 0 };

struct builtin milkls_struct = { "milkls",
                                 milkls_builtin_entry,
                                 BUILTIN_ENABLED,
                                 milkls_doc,
                                 "milkls [-l|-q|--json|--local] [pat]",
                                 0 };

struct builtin milkargs_struct = {
    "milkargs", milkargs_builtin, BUILTIN_ENABLED, milkargs_doc, "milkargs SPEC... -- VALUE...", 0
};

struct builtin milkdefer_struct = { "milkdefer",
                                    milkdefer_builtin,
                                    BUILTIN_ENABLED,
                                    milkdefer_doc,
                                    "milkdefer [--exit] CMD... | --restore SPEC...",
                                    0 };

struct builtin milksleep_struct = { "milksleep",
                                    milksleep_builtin_entry,
                                    BUILTIN_ENABLED,
                                    milksleep_doc,
                                    "milksleep SECONDS | --every T|--hz F [--late L] VAR",
                                    0 };

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

/* the keyword builtins: `@<kw>`, loaded as kw<kw> (bash takes the name from the struct) */
#define MILK_KEYWORD_STRUCT(name, doc)                                     \
    struct builtin kw##name##_struct = { "@" #name,                        \
                                         name##_keyword_builtin,           \
                                         BUILTIN_ENABLED,                  \
                                         doc,                              \
                                         "@" #name " (@" #name " --help)", \
                                         0 }
MILK_KEYWORD_STRUCT(calc, milkcalc_doc);
MILK_KEYWORD_STRUCT(test, milktest_doc);
MILK_KEYWORD_STRUCT(get, milkget_doc);
MILK_KEYWORD_STRUCT(set, milkset_doc);
MILK_KEYWORD_STRUCT(exists, milkexists_doc);
MILK_KEYWORD_STRUCT(ls, milkls_doc);
MILK_KEYWORD_STRUCT(waitfor, milkwaitfor_doc);
MILK_KEYWORD_STRUCT(waitany, milkwaitany_doc);
MILK_KEYWORD_STRUCT(sleep, milksleep_doc);
MILK_KEYWORD_STRUCT(grab, milkgrab_doc);
MILK_KEYWORD_STRUCT(log, milklog_doc);
MILK_KEYWORD_STRUCT(defer, milkdefer_doc);
MILK_KEYWORD_STRUCT(args, milkargs_doc);
#pragma GCC visibility pop
