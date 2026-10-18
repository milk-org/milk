// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_bash_builtins.c
 * @brief Loadable GNU Bash builtins for milk (`enable -f libmilkbash.so ...`)
 *
 * Exports loadable builtins:
 *   milkcall, milkget, milkset, milkcalc, milktest,
 *   milkexists, milkwaitfor, milkgrab, milkcomplete, milkshutdown, milkdefer, milkargs
 *
 * Uses a minimal vendored GNU Bash ABI definition (stable across Bash 3.2 - 5.3+)
 * and resolves host bash symbols via dlsym(RTLD_DEFAULT, ...) so no external
 * bash headers are required at build time.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "milk_bash_defer.h"
#include "milk_bash_engine.h"
#include "milk_lexicon.h"
#include "milk_client.h"
#include "milkdata.h"

__attribute__((constructor(101))) static void milkbash_init_quiet(void)
{
    milk_data.quiet = 1;
}

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

/* ============================================================
 *  Dynamic resolution of host Bash C functions
 * ============================================================ */

typedef void *(*bash_bind_variable_fn)(const char *, char *, int);
typedef const char *(*bash_get_string_value_fn)(const char *);
typedef int (*bash_parse_and_execute_fn)(char *, const char *, int);

static bash_bind_variable_fn     fn_bind_variable     = NULL;
static bash_get_string_value_fn  fn_get_string_value  = NULL;
static bash_parse_and_execute_fn fn_parse_and_execute = NULL;
static int                       g_syms_resolved      = 0;

static void resolve_bash_symbols(void)
{
    if (g_syms_resolved)
    {
        return;
    }
    g_syms_resolved      = 1;
    fn_bind_variable     = (bash_bind_variable_fn) dlsym(RTLD_DEFAULT, "bind_variable");
    fn_get_string_value  = (bash_get_string_value_fn) dlsym(RTLD_DEFAULT, "get_string_value");
    fn_parse_and_execute = (bash_parse_and_execute_fn) dlsym(RTLD_DEFAULT, "parse_and_execute");
}

static const char *bash_hook_get_var(const char *name, void *ctx)
{
    (void) ctx;
    resolve_bash_symbols();
    if (fn_get_string_value != NULL)
    {
        return fn_get_string_value(name);
    }
    return getenv(name);
}

static void bash_hook_set_var(const char *name, const char *val, void *ctx)
{
    (void) ctx;
    resolve_bash_symbols();
    if (fn_bind_variable != NULL && name != NULL)
    {
        fn_bind_variable(name, (char *) (val ? val : ""), 0);
    }
    else if (name != NULL)
    {
        setenv(name, val ? val : "", 1);
    }
}

static void bash_hook_def_cmd(const char *cmdname, void *ctx)
{
    (void) ctx;
    resolve_bash_symbols();
    if (fn_parse_and_execute == NULL || cmdname == NULL || cmdname[0] == '\0')
    {
        return;
    }
    if (strchr(cmdname, '.') == NULL) /* module commands only: module.command */
    {
        return;
    }
    char *script = (char *) malloc(512);
    if (script != NULL)
    {
        snprintf(script, 512, "function %s { milkcall %s \"$@\"; }", cmdname, cmdname);
        /* SEVAL_NONINT (0x01) | SEVAL_NOHIST (0x04) = 0x05 */
        fn_parse_and_execute(script, "milk_defcmd", 0x05);
    }
}

static volatile sig_atomic_t g_sigint_pending = 0; /* SIGINT taken by the client, owed to bash */

static void bash_hook_on_sigint(void *ctx)
{
    (void) ctx;
    g_sigint_pending = 1;
}

static void bash_hook_eval(char *script, void *ctx)
{
    (void) ctx;
    resolve_bash_symbols();
    if (fn_parse_and_execute != NULL)
    {
        fn_parse_and_execute(strdup(script), "milk_bind", 0x05); /* bash frees the string */
    }
}

static void make_bash_hooks(milk_client_hooks_t *hooks)
{
    resolve_bash_symbols();
    hooks->get_var   = bash_hook_get_var;
    hooks->set_var   = bash_hook_set_var;
    hooks->def_cmd   = bash_hook_def_cmd;
    hooks->on_sigint = bash_hook_on_sigint;
    hooks->eval      = bash_hook_eval;
    hooks->ctx       = NULL;
}

static int wordlist_to_argv(WORD_LIST *list, char ***out_argv)
{
    int count = 0;
    for (WORD_LIST *w = list; w != NULL; w = w->next)
    {
        count++;
    }
    char **argv = (char **) calloc((size_t) count + 1, sizeof(char *));
    if (argv == NULL)
    {
        *out_argv = NULL;
        return 0;
    }
    int idx = 0;
    for (WORD_LIST *w = list; w != NULL; w = w->next)
    {
        argv[idx++] = (w->word && w->word->word) ? w->word->word : "";
    }
    argv[idx] = NULL;
    *out_argv = argv;
    return count;
}

/* ============================================================
 *  Builtin Entry Points
 * ============================================================ */

static int milkcall_builtin(WORD_LIST *list)
{
    char **argv = NULL;
    int    argc = wordlist_to_argv(list, &argv);
    if (argc == 0 || argv == NULL)
    {
        free(argv);
        return EXECUTION_SUCCESS;
    }

    int async_flag = 0;
    int start_idx  = 0;
    if (strcmp(argv[0], "--async") == 0)
    {
        async_flag = 1;
        start_idx  = 1;
    }
    if (start_idx >= argc)
    {
        free(argv);
        return EXECUTION_SUCCESS;
    }

    milk_client_hooks_t hooks;
    make_bash_hooks(&hooks);

    int rc = milk_client_call(argc - start_idx, argv + start_idx, async_flag, &hooks);
    free(argv);
    return rc;
}

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

static int milkshutdown_builtin(WORD_LIST *list)
{
    (void) list;
    milk_client_hooks_t hooks;
    make_bash_hooks(&hooks);
    return milk_client_shutdown(&hooks);
}

static int milkserver_builtin(WORD_LIST *list)
{
    char              **argv = NULL;
    int                 argc = wordlist_to_argv(list, &argv);
    milk_client_hooks_t hooks;
    make_bash_hooks(&hooks);
    int rc = milk_client_server(argc, argv, &hooks);
    free(argv);
    return rc;
}

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

/** milksleep SECONDS | --every T VAR - sleep in-process (milk_client_sleep_argv()); Ctrl-C reaches bash. */
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

/** milkcloexec FD... - mark the engine's pipe fds close-on-exec so bash's children do not inherit them. */
static int milkcloexec_builtin(WORD_LIST *list)
{
    int rc = 0;
    for (WORD_LIST *w = list; w != NULL; w = w->next)
    {
        char *end = NULL;
        long  fd  = strtol(w->word->word, &end, 10);
        int fl = (end != w->word->word && *end == '\0' && fd >= 0) ? fcntl((int) fd, F_GETFD) : -1;
        if (fl < 0 || fcntl((int) fd, F_SETFD, fl | FD_CLOEXEC) != 0)
        {
            rc = 1;
        }
    }
    return rc;
}

/** sync_funsub_env() - set MILK_BASH_NO_FUNSUB from the running shell's version (a value the user set wins). */
static void sync_funsub_env(void)
{
    const char *forced = bash_hook_get_var("MILK_BASH_NO_FUNSUB", NULL);
    const char *ver    = bash_hook_get_var("BASH_VERSION", NULL);
    int         major  = 0;
    int         minor  = 0;

    if (forced != NULL && forced[0] != '\0')
    {
        setenv("MILK_BASH_NO_FUNSUB", forced, 1);
    }
    else if (ver != NULL && sscanf(ver, "%d.%d", &major, &minor) == 2)
    {
        setenv("MILK_BASH_NO_FUNSUB", (major > 5 || (major == 5 && minor >= 3)) ? "0" : "1", 1);
    }
}

/**
 * milktranslate_builtin() - `milktranslate -v VAR FILE`: translate a milk script into a private
 * command file and bind its path to VAR (no fork); `milktranslate --done PATH` removes it.
 */
static int milktranslate_builtin(WORD_LIST *list)
{
    char **argv = NULL;
    int    argc = wordlist_to_argv(list, &argv);
    int    rc   = EXECUTION_FAILURE;
    if (argc == 2 && strcmp(argv[0], "--done") == 0)
    {
        if (milk_bash_cmdfile_owned(argv[1]))
        {
            unlink(argv[1]);
        }
        free(argv);
        return EXECUTION_SUCCESS;
    }
    if (argc == 3 && strcmp(argv[0], "-s") == 0)
    {
        /* a line typed at the prompt (@config inline), left as is while it is incomplete */
        sync_funsub_env();
        char *out =
            milk_shell_incomplete(argv[2]) ? NULL : milk_translate_shorthand_to_bash(argv[2], 1);
        bash_hook_set_var(argv[1], (out != NULL) ? out : argv[2], NULL);
        free(out);
        free(argv);
        return EXECUTION_SUCCESS;
    }
    if (argc != 3 || strcmp(argv[0], "-v") != 0)
    {
        fprintf(stderr, "Usage: milktranslate -v VAR FILE | -s VAR TEXT | --done PATH\n");
        free(argv);
        return 2;
    }
    FILE *fp = fopen(argv[2], "r");
    if (fp == NULL)
    {
        fprintf(stderr, "milktranslate: cannot open '%s'\n", argv[2]);
        free(argv);
        return EXECUTION_FAILURE;
    }
    size_t cap = 4096;
    size_t len = 0;
    char  *src = (char *) malloc(cap);
    size_t nr;
    while (src != NULL && (nr = fread(src + len, 1, cap - len - 1, fp)) > 0)
    {
        len += nr;
        if (len + 1 == cap)
        {
            char *ns = (char *) realloc(src, cap * 2);
            if (ns == NULL)
            {
                break;
            }
            src = ns;
            cap *= 2;
        }
    }
    fclose(fp);
    if (src == NULL)
    {
        free(argv);
        return EXECUTION_FAILURE;
    }
    src[len] = '\0';

    sync_funsub_env();
    char *out = milk_translate_shorthand_to_bash(src, 1);
    free(src);
    const char *base = strrchr(argv[2], '/'); /* bash error messages name the script */
    char        path[512];
    if (out != NULL &&
        milk_bash_cmdfile_path((base != NULL) ? base + 1 : argv[2], path, sizeof(path)) == 0)
    {
        FILE *wf = fopen(path, "w");
        if (wf != NULL && fputs(out, wf) >= 0 && fclose(wf) == 0)
        {
            bash_hook_set_var(argv[1], path, NULL);
            rc = EXECUTION_SUCCESS;
        }
        else if (wf != NULL)
        {
            fclose(wf);
        }
    }
    if (rc != EXECUTION_SUCCESS)
    {
        fprintf(stderr, "milktranslate: cannot translate '%s'\n", argv[2]);
    }
    free(out);
    free(argv);
    return rc;
}

static int milkdefer_builtin(WORD_LIST *list)
{
    char **argv = NULL;
    int    argc = wordlist_to_argv(list, &argv);
    int    rc   = milk_bash_defer_argv(argc, argv);
    free(argv);
    return rc;
}

/**
 * milkargs_builtin() - `@args SPEC... -- VALUE...`: the values become locals of the calling function
 * (globals at a script's top level); after -h or a usage error the caller ends, as `eval return N` would.
 */
static int milkargs_builtin(WORD_LIST *list)
{
    char              **argv = NULL;
    int                 argc = wordlist_to_argv(list, &argv);
    const int          *vc   = (const int *) dlsym(RTLD_DEFAULT, "variable_context");
    const int          *sl   = (const int *) dlsym(RTLD_DEFAULT, "sourcelevel");
    int                 infn = (vc != NULL && *vc > 0);
    const char         *fn   = bash_hook_get_var(infn ? "FUNCNAME" : "BASH_ARGV0", NULL);
    milk_client_hooks_t hooks;
    char                fname[256];
    char                end[32];

    if (fn == NULL || fn[0] == '\0' || strcmp(fn, "source") == 0) /* a sourced file: its name */
    {
        fn = bash_hook_get_var("BASH_SOURCE", NULL);
    }
    snprintf(fname, sizeof(fname), "%s",
             (fn == NULL || fn[0] == '\0') ? "@args"
                                           : (strrchr(fn, '/') ? strrchr(fn, '/') + 1 : fn));
    make_bash_hooks(&hooks);
    int rc = milk_client_args(argc, argv, fname, infn, &hooks);
    free(argv);
    if (rc == 0)
    {
        return 0;
    }
    rc = (rc == 1) ? 0 : 2;
    snprintf(end, sizeof(end), (infn || (sl != NULL && *sl > 0)) ? "return %d" : "exit %d", rc);
    bash_hook_eval(end, NULL); /* does not come back */
    return rc;
}

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
static char *const milkserver_doc[]   = { "Manage milk-cli servers (@server).", NULL };
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

/* Builtin entry points: a SIGINT the client caught is re-raised once bash's handler is back, so bash
 * reacts to it (INT trap, or aborting the command line) as if it had received it. */
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
MILK_BUILTIN_ENTRY(milkserver_builtin)
MILK_BUILTIN_ENTRY(milklog_builtin)
MILK_BUILTIN_ENTRY(milkls_builtin)
MILK_BUILTIN_ENTRY(milksleep_builtin)

/**
 * keyword_builtin() - `@<kw> args`: the keyword's help (milk_lexicon_help_arg()), else builtin @fn.
 *
 * The keywords that a builtin implements are builtins named `@<kw>` themselves, loaded as `kw<kw>`
 * (`enable -f libmilkbash.so kwget` defines `@get`): a call then costs no shell function.
 */
static int keyword_builtin(const char *kw, sh_builtin_func_t *fn, WORD_LIST *list)
{
    const milk_keyword_t *k = NULL;
    if (list != NULL && list->word->word[0] == '-' && (k = milk_lexicon_keyword(kw)) != NULL &&
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

/* The library is built with hidden visibility: bash finds each builtin by its `<name>_struct` symbol */
#pragma GCC visibility push(default)
struct builtin milkcall_struct = { "milkcall",   milkcall_builtin_entry,     BUILTIN_ENABLED,
                                   milkcall_doc, "milkcall <cmd> [args...]", 0 };

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

struct builtin milkserver_struct = { "milkserver",   milkserver_builtin_entry,      BUILTIN_ENABLED,
                                     milkserver_doc, "milkserver [subcommand ...]", 0 };

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

struct builtin milkcloexec_struct = { "milkcloexec",   milkcloexec_builtin, BUILTIN_ENABLED,
                                      milkcloexec_doc, "milkcloexec FD...", 0 };

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
