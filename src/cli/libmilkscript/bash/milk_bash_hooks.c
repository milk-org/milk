// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_bash_hooks.c
 * @brief Host GNU Bash symbol resolution, client callbacks, and pure-shell loadable builtins.
 *
 * Resolves host Bash C functions (`bind_variable`, `get_string_value`, `parse_and_execute`)
 * via `dlsym(RTLD_DEFAULT, ...)`, provides `milk_client_hooks_t` callbacks for in-process
 * variable binding and command definition, and implements the pure-shell builtins
 * (`milkcloexec`, `milktranslate`, `milkdefer`, `milkargs`).
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <dlfcn.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <unistd.h>

#include "milk_bash_defer.h"
#include "milk_bash_engine.h"
#include "milk_client.h"
#include "milk_client_internal.h"
#include "milkdata.h"

/**
 * milkbash_init_quiet - Suppress milk startup banners when loaded as a Bash builtin
 *
 * Sets the global quiet flag during shared library initialization so that loading
 * libmilkbash.so via `enable -f` does not pollute the interactive shell's output.
 *
 * Return: void
 */
__attribute__((constructor(101))) static void milkbash_init_quiet(void)
{
    milk_data.quiet = 1;
}

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

volatile sig_atomic_t g_sigint_pending = 0; /* SIGINT taken by the client, owed to bash */

/**
 * resolve_bash_symbols - Lazily resolve host Bash C API symbols via dlsym
 *
 * Looks up `bind_variable`, `get_string_value`, and `parse_and_execute` in the
 * hosting Bash process image using `RTLD_DEFAULT`. Avoids a compile-time dependency
 * on version-specific GNU Bash internal headers while allowing builtins to read and
 * mutate shell state directly.
 *
 * Return: void
 */
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

/**
 * bash_hook_get_var - Read a shell or environment variable from the host Bash process
 * @name: Variable name to look up
 * @ctx:  Unused hook context pointer
 *
 * Queries the host Bash symbol table via `get_string_value` so unexported shell
 * variables are visible to milk client operations, falling back to `getenv()` if
 * the Bash symbol is unavailable.
 *
 * Return: Pointer to variable value string, or NULL if unset
 */
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

/**
 * bash_hook_set_var - Bind a shell variable in the host Bash process
 * @name: Variable name to assign
 * @val:  String value to bind (NULL is treated as empty string)
 * @ctx:  Unused hook context pointer
 *
 * Binds @name directly in the running shell's variable table via `bind_variable`
 * so milk commands can return values into shell variables without command
 * substitution or subshell forks. Falls back to `setenv()` if unavailable.
 *
 * Return: void
 */
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

/**
 * bash_hook_def_cmd - Define a shell wrapper function for a milk module command
 * @cmdname: Dotted module command name (e.g., `module.command`)
 * @ctx:     Unused hook context pointer
 *
 * Evaluates a non-interactive, no-history function definition in the host shell
 * that forwards invocations of @cmdname to `milkcall`, enabling newly loaded
 * server commands to be called directly as first-class shell commands.
 *
 * Return: void
 */
static void bash_hook_def_cmd(const char *cmdname, void *ctx)
{
    (void) ctx;
    resolve_bash_symbols();
    if (fn_parse_and_execute == NULL || cmdname == NULL || cmdname[0] == '\0')
    {
        return;
    }
    size_t clen = strlen(cmdname);
    if (strchr(cmdname, '.') == NULL || clen > 200) /* module commands only: module.command */
    {
        return;
    }
    char *script = (char *) malloc(512);
    if (script != NULL)
    {
        int n = snprintf(script, 512, "function %s { milkcall %s \"$@\"; }", cmdname, cmdname);
        if (n < 0 || n >= 512)
        {
            free(script);
            return;
        }
        /* SEVAL_NONINT (0x01) | SEVAL_NOHIST (0x04) = 0x05 */
        fn_parse_and_execute(script, "milk_defcmd", 0x05);
    }
}

/**
 * bash_hook_on_sigint - Record that a SIGINT was caught during a client operation
 * @ctx: Unused hook context pointer
 *
 * Marks a pending SIGINT flag so the builtin entry wrapper can re-raise SIGINT
 * after restoring Bash's own signal handlers, ensuring Bash traps and script
 * abort semantics behave as expected.
 *
 * Return: void
 */
static void bash_hook_on_sigint(void *ctx)
{
    (void) ctx;
    g_sigint_pending = 1;
}

/**
 * bash_hook_eval - Evaluate a Bash script snippet in the current shell frame
 * @script: Null-terminated shell code string to execute
 * @ctx:    Unused hook context pointer
 *
 * Passes a heap-duplicated copy of @script to Bash's `parse_and_execute` with
 * non-interactive and no-history flags, used to bind array variables or run
 * caller-scoped `local` / `return` statements.
 *
 * Return: void
 */
static void bash_hook_eval(char *script, void *ctx)
{
    (void) ctx;
    resolve_bash_symbols();
    if (fn_parse_and_execute != NULL && script != NULL)
    {
        char *dup = strdup(script);
        if (dup != NULL)
        {
            fn_parse_and_execute(dup, "milk_bind", 0x05); /* bash frees the string */
        }
    }
}

/**
 * make_bash_hooks - Populate a milk client hook table with host Bash callbacks
 * @hooks: Output hook structure to initialize
 *
 * Resolves host Bash symbols and wires variable access, command definition,
 * SIGINT notification, and script evaluation callbacks into @hooks for use by
 * loadable builtin implementations.
 *
 * Return: void
 */
void make_bash_hooks(milk_client_hooks_t *hooks)
{
    resolve_bash_symbols();
    hooks->get_var   = bash_hook_get_var;
    hooks->set_var   = bash_hook_set_var;
    hooks->def_cmd   = bash_hook_def_cmd;
    hooks->on_sigint = bash_hook_on_sigint;
    hooks->eval      = bash_hook_eval;
    hooks->ctx       = NULL;
}

/**
 * wordlist_to_argv - Convert a Bash WORD_LIST linked list into a C argv array
 * @list:     Head of the Bash word list passed to a builtin
 * @out_argv: Output pointer receiving the newly allocated NULL-terminated argv array
 *
 * Counts words in @list and allocates a pointer array referencing the underlying
 * word strings without copying characters. The caller must `free(*out_argv)`.
 *
 * Return: Number of arguments in @out_argv, or 0 on empty list or allocation failure
 */
int wordlist_to_argv(WORD_LIST *list, char ***out_argv)
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
 *  Pure-Bash Loadable Builtins
 * ============================================================ */

/*
 * --watcher PARENT: this subshell is an @on watcher started by PARENT. It does not hold the shell's
 * session (its owner pipe), and it ends with PARENT. Return: 1 if PARENT has gone already
 */
static int watcher_init(WORD_LIST *arg)
{
    pid_t parent = (arg != NULL && arg->word != NULL) ? (pid_t) atol(arg->word->word) : 0;

    milk_client_close_owner_pipe();
    if (prctl(PR_SET_PDEATHSIG, SIGTERM) != 0)
    {
        return 1;
    }
    return (parent > 0 && getppid() != parent) ? 1 : 0;
}

/**
 * milkcloexec_builtin - Mark file descriptors close-on-exec (`milkcloexec` builtin)
 * @list: File descriptor numbers passed as a Bash `WORD_LIST`, or `--watcher PARENT`
 *
 * Marks the engine's pipe file descriptors `FD_CLOEXEC` so child processes spawned
 * by Bash do not inherit them.
 *
 * Return: 0 if all file descriptors were updated, 1 if any failed
 */
int milkcloexec_builtin(WORD_LIST *list)
{
    int rc = 0;
    if (list != NULL && list->word != NULL && strcmp(list->word->word, "--watcher") == 0)
    {
        return watcher_init(list->next);
    }
    for (WORD_LIST *w = list; w != NULL; w = w->next)
    {
        if (w->word == NULL || w->word->word == NULL)
        {
            rc = 1;
            continue;
        }
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

/**
 * sync_funsub_env - Set `MILK_BASH_NO_FUNSUB` from the running shell's `BASH_VERSION`
 *
 * Honors any non-empty user setting of `MILK_BASH_NO_FUNSUB` and otherwise enables
 * `${| ...; }` function substitutions when the hosting Bash is 5.3 or newer.
 *
 * Return: void
 */
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
 * milktranslate_builtin - Translate a milk script or interactive line (`milktranslate` builtin)
 * @list: Arguments (`-v VAR FILE`, `-s VAR TEXT`, or `--done PATH`)
 *
 * Translates a `.milk` script into a private command file and binds its path to `VAR`
 * without forking, translates a single interactive readline buffer (`-s`), or removes
 * a completed command file (`--done`).
 *
 * Return: `EXECUTION_SUCCESS` on success, `EXECUTION_FAILURE` on error, 2 on bad usage
 */
int milktranslate_builtin(WORD_LIST *list)
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
                free(src);
                fclose(fp);
                free(argv);
                return EXECUTION_FAILURE;
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

/**
 * milkdefer_builtin - Bash builtin entry point for `milkdefer` (`@defer`)
 * @list: Arguments passed to the builtin as a Bash WORD_LIST
 *
 * Registers cleanup commands to run when the current function, sourced file, or
 * shell session exits, or evaluates pending deferred commands on `RETURN`/`EXIT`.
 *
 * Return: 0 on success, non-zero when staging trap commands or on error
 */
int milkdefer_builtin(WORD_LIST *list)
{
    char **argv = NULL;
    int    argc = wordlist_to_argv(list, &argv);
    int    rc   = milk_bash_defer_argv(argc, argv);
    free(argv);
    return rc;
}

/**
 * milkargs_builtin - Parse and bind function or script arguments (`@args` builtin)
 * @list: Argument specifications followed by `--` and actual values
 *
 * Binds parsed values as local variables of the calling function (or globals at a
 * script's top level, including `@source`'d scripts) and exits the caller on `-h`
 * or usage error.
 *
 * Return: 0 on success, 2 on usage error
 */
int milkargs_builtin(WORD_LIST *list)
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
        fn   = bash_hook_get_var("BASH_SOURCE", NULL);
        infn = 0;
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
