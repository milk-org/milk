// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_bash_builtins.c
 * @brief Loadable GNU Bash builtins for milk (`enable -f libmilkbash.so ...`)
 *
 * Exports loadable builtins:
 *   milkcall, milkget, milkset, milkcalc, milktest,
 *   milkexists, milkwaitfor, milkgrab, milkcomplete, milkshutdown
 *
 * Uses a minimal vendored GNU Bash ABI definition (stable across Bash 3.2 - 5.3+)
 * and resolves host bash symbols via dlsym(RTLD_DEFAULT, ...) so no external
 * bash headers are required at build time.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
typedef void *(*bash_find_variable_fn)(const char *);
typedef int (*bash_parse_and_execute_fn)(char *, const char *, int);

static bash_bind_variable_fn     fn_bind_variable     = NULL;
static bash_get_string_value_fn  fn_get_string_value  = NULL;
static bash_find_variable_fn     fn_find_variable     = NULL;
static bash_parse_and_execute_fn fn_parse_and_execute = NULL;
static volatile int             *ptr_interrupt_state  = NULL;
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
    fn_find_variable     = (bash_find_variable_fn) dlsym(RTLD_DEFAULT, "find_variable");
    fn_parse_and_execute = (bash_parse_and_execute_fn) dlsym(RTLD_DEFAULT, "parse_and_execute");
    ptr_interrupt_state  = (volatile int *) dlsym(RTLD_DEFAULT, "interrupt_state");
}

/**
 * @brief Detect whether we are executing inside a Bash 5.3+ `${| ...; }` funsub.
 *
 * Inside `${| ...; }`, Bash creates a local placeholder variable `REPLY` with a NULL
 * value (`find_variable("REPLY") != NULL && get_string_value("REPLY") == NULL`).
 */
static int in_bash_funsub(void)
{
    resolve_bash_symbols();
    if (fn_find_variable != NULL && fn_get_string_value != NULL)
    {
        if (fn_find_variable("REPLY") != NULL && fn_get_string_value("REPLY") == NULL)
        {
            return 1;
        }
    }
    return 0;
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
    const char *sigil = bash_hook_get_var("MILKCLI_SIGIL", NULL);
    if (sigil == NULL || sigil[0] == '\0')
    {
        sigil = "@";
    }
    char *script = (char *) malloc(512);
    if (script != NULL)
    {
        snprintf(script, 512, "%s%s() { milkcall \"%s\" \"$@\"; }", sigil, cmdname, cmdname);
        /* SEVAL_NONINT (0x01) | SEVAL_NOHIST (0x04) = 0x05 */
        fn_parse_and_execute(script, "milk_defcmd", 0x05);
    }
}

static void bash_hook_on_sigint(void *ctx)
{
    (void) ctx;
    resolve_bash_symbols();
    if (ptr_interrupt_state != NULL)
    {
        (*ptr_interrupt_state)++;
    }
}

static void make_bash_hooks(milk_client_hooks_t *hooks)
{
    resolve_bash_symbols();
    hooks->get_var   = bash_hook_get_var;
    hooks->set_var   = bash_hook_set_var;
    hooks->def_cmd   = bash_hook_def_cmd;
    hooks->on_sigint = bash_hook_on_sigint;
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
    char **argv = NULL;
    int    argc = wordlist_to_argv(list, &argv);
    if (argc == 0 || argv == NULL)
    {
        free(argv);
        fprintf(stderr, "Usage: @get [-v var] <entity>\n");
        return EXECUTION_FAILURE;
    }

    const char *varname = NULL;
    const char *spec    = NULL;
    for (int i = 0; i < argc; i++)
    {
        if (strcmp(argv[i], "-v") == 0 && i + 1 < argc)
        {
            varname = argv[++i];
        }
        else if (spec == NULL)
        {
            spec = argv[i];
        }
    }

    if (varname == NULL && in_bash_funsub())
    {
        varname = "REPLY";
    }

    milk_client_hooks_t hooks;
    make_bash_hooks(&hooks);
    int rc = milk_client_get(spec, varname, &hooks);
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
        fprintf(stderr, "Usage: @set <entity> [=] [value] [comment]\n");
        return EXECUTION_FAILURE;
    }

    const char *spec    = argv[0];
    int         v_idx   = (argc >= 3 && strcmp(argv[1], "=") == 0) ? 2 : 1;
    const char *val     = (argc > v_idx) ? argv[v_idx] : NULL;
    const char *comment = (argc > v_idx + 1) ? argv[v_idx + 1] : NULL;

    milk_client_hooks_t hooks;
    make_bash_hooks(&hooks);
    int rc = milk_client_set(spec, val, comment, &hooks);
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

    const char *varname   = NULL;
    int         start_idx = 0;
    if (strcmp(argv[0], "-v") == 0 && argc >= 2)
    {
        varname   = argv[1];
        start_idx = 2;
    }
    else if (in_bash_funsub())
    {
        varname = "REPLY";
    }

    size_t tot = 1;
    for (int i = start_idx; i < argc; i++)
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
    for (int i = start_idx; i < argc; i++)
    {
        int n = snprintf(expr + pos, tot - pos, "%s%s", (i > start_idx) ? " " : "", argv[i]);
        if (n > 0)
        {
            pos += (size_t) n;
        }
    }

    milk_client_hooks_t hooks;
    make_bash_hooks(&hooks);
    int rc = milk_client_calc(expr, varname, &hooks);
    free(expr);
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
static char *const milkgrab_doc[] = { "Capture consecutive stream frames into a 3D cube.", NULL };
static char *const milkcomplete_doc[] = { "Query milk tab-completion candidates.", NULL };
static char *const milkshutdown_doc[] = { "Shut down the session milk-cli --server.", NULL };
static char *const milkserver_doc[]   = { "Manage milk-cli servers (@server).", NULL };
static char *const milklog_doc[] = { "Inspect or search the milk server SHM log (@log).", NULL };

struct builtin milkcall_struct = { "milkcall",   milkcall_builtin,           BUILTIN_ENABLED,
                                   milkcall_doc, "milkcall <cmd> [args...]", 0 };

struct builtin milkget_struct = {
    "milkget", milkget_builtin, BUILTIN_ENABLED, milkget_doc, "milkget [-v var] <entity>", 0
};

struct builtin milkset_struct = {
    "milkset", milkset_builtin, BUILTIN_ENABLED, milkset_doc, "milkset <entity> [val] [comment]", 0
};

struct builtin milkcalc_struct = { "milkcalc",   milkcalc_builtin,           BUILTIN_ENABLED,
                                   milkcalc_doc, "milkcalc [-v var] <expr>", 0 };

struct builtin milktest_struct = { "milktest",   milktest_builtin,  BUILTIN_ENABLED,
                                   milktest_doc, "milktest <expr>", 0 };

struct builtin milkexists_struct = { "milkexists",   milkexists_builtin,         BUILTIN_ENABLED,
                                     milkexists_doc, "milkexists [-r] <entity>", 0 };

struct builtin milkwaitfor_struct = { "milkwaitfor",
                                      milkwaitfor_builtin,
                                      BUILTIN_ENABLED,
                                      milkwaitfor_doc,
                                      "milkwaitfor <target> [nframes] [--timeout <sec>]",
                                      0 };

struct builtin milkgrab_struct = { "milkgrab",
                                   milkgrab_builtin,
                                   BUILTIN_ENABLED,
                                   milkgrab_doc,
                                   "milkgrab <stream> <nframes> <out_cube> [--timeout <sec>]",
                                   0 };

struct builtin milkcomplete_struct = { "milkcomplete",
                                       milkcomplete_builtin,
                                       BUILTIN_ENABLED,
                                       milkcomplete_doc,
                                       "milkcomplete <prefix> [cmd|arg]",
                                       0 };

struct builtin milkshutdown_struct = { "milkshutdown",   milkshutdown_builtin, BUILTIN_ENABLED,
                                       milkshutdown_doc, "milkshutdown",       0 };

struct builtin milkserver_struct = { "milkserver",
                                     milkserver_builtin,
                                     BUILTIN_ENABLED,
                                     milkserver_doc,
                                     "milkserver [subcommand ...]",
                                     0 };

struct builtin milklog_struct = { "milklog",   milklog_builtin,        BUILTIN_ENABLED,
                                  milklog_doc, "milklog [options...]", 0 };
