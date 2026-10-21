// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_bash_defer.c
 * @brief `@defer` in the shell: cleanup entries that a RETURN trap runs when their scope ends
 *
 * A scope is the frame of the function or @source'd file that registered an entry, kept as its height:
 * the length of FUNCNAME seen from that frame. Height 0 is the session, run at EXIT. The entries live
 * here, per process. The RETURN trap calls `milkdefer --return "$_"`, which binds __MILK_DRUN_SH to a
 * script that the trap evaluates in the returning frame (its locals and arguments). scripts/milk.bash
 * has the same logic in bash for shells without this library.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "milk_bash_defer.h"

/* the members of bash's SHELL_VAR, unchanged since bash 3 */
typedef struct bash_var
{
    char *name;
    char *value; /* an ARRAY * for an indexed array */
    char *exportstr;
    void *dynamic_value;
    void *assign_func;
    int   attributes;
    int   context;
} bash_var_t;

#define BASH_ATT_ARRAY 0x4

static bash_var_t *(*fn_find_variable)(const char *);
static char *(*fn_array_reference)(void *, intmax_t);
static const char *(*fn_get_string_value)(const char *);
static void *(*fn_bind_variable)(const char *, char *, int);
static int (*fn_parse_and_execute)(char *, const char *, int);
static void (*fn_set_signal)(int, const char *);
static void (*fn_restore_default_signal)(int);
static int g_return_sig = -1;

typedef struct
{
    char *cmd;
    int   height;
} defer_entry_t;

static defer_entry_t *g_stack; /* scoped entries: scopes nest, so the innermost ones are on top */
static int            g_n;
static int            g_cap;
static char         **g_exit; /* session entries */
static int            g_en;
static int            g_ecap;
static pid_t          g_pid;       /* the process the entries belong to */
static int            g_sf_h = -1; /* the last `source` end seen: height, command and $_ */
static char          *g_sf_cmd;
static char          *g_sf_u;

static const char RETURN_TRAP[] = "milkdefer --return \"$_\" || eval \"${__MILK_DRUN_SH}\"";

/** resolve() - the bash functions used here (dlsym, as for the other builtins); -1 if one is missing. */
static int resolve(void)
{
    int (*decode_signal)(const char *, int);

    if (fn_find_variable != NULL)
    {
        return 0;
    }
    fn_array_reference  = (char *(*) (void *, intmax_t)) dlsym(RTLD_DEFAULT, "array_reference");
    fn_get_string_value = (const char *(*) (const char *) ) dlsym(RTLD_DEFAULT, "get_string_value");
    fn_bind_variable =
        (void *(*) (const char *, char *, int) ) dlsym(RTLD_DEFAULT, "bind_variable");
    fn_parse_and_execute =
        (int (*)(char *, const char *, int)) dlsym(RTLD_DEFAULT, "parse_and_execute");
    fn_set_signal             = (void (*)(int, const char *)) dlsym(RTLD_DEFAULT, "set_signal");
    fn_restore_default_signal = (void (*)(int)) dlsym(RTLD_DEFAULT, "restore_default_signal");
    decode_signal             = (int (*)(const char *, int)) dlsym(RTLD_DEFAULT, "decode_signal");
    if (fn_array_reference == NULL || fn_get_string_value == NULL || fn_bind_variable == NULL ||
        fn_parse_and_execute == NULL || fn_set_signal == NULL ||
        fn_restore_default_signal == NULL || decode_signal == NULL ||
        (g_return_sig = decode_signal("RETURN", 0)) < 0)
    {
        return -1;
    }
    fn_find_variable = (bash_var_t * (*) (const char *) ) dlsym(RTLD_DEFAULT, "find_variable");
    return (fn_find_variable != NULL) ? 0 : -1;
}

/** sh_eval() - run a line of bash in the current frame; its status. */
static int sh_eval(const char *script)
{
    char *s = strdup(script); /* bash frees it */
    return (s != NULL) ? fn_parse_and_execute(s, "milkdefer", 0x05)
                       : 1; /* SEVAL_NONINT | SEVAL_NOHIST */
}

/**
 * bash_array - Look up an indexed Bash array variable's internal ARRAY pointer
 * @name: Name of the Bash array variable (e.g., `FUNCNAME` or `BASH_LINENO`)
 *
 * Queries Bash's symbol table via `find_variable` and verifies `BASH_ATT_ARRAY`
 * is set before returning the raw `ARRAY *` stored in the variable's value slot.
 *
 * Return: Pointer to Bash's internal ARRAY structure, or NULL if not an array
 */
static void *bash_array(const char *name)
{
    bash_var_t *v = fn_find_variable(name);
    return (v != NULL && (v->attributes & BASH_ATT_ARRAY)) ? v->value : NULL;
}

/**
 * frames() - FUNCNAME's length: the height of the running frame. *scope: the height of the first frame
 * that is a function or an @source'd file and not milk's, 0 for the session. A plain `source` is no
 * scope; the script's own frame is the bottom `main`, called from line 0.
 */
static int frames(int *scope)
{
    void *fa = bash_array("FUNCNAME");
    int   n  = 0;

    while (fa != NULL && fn_array_reference(fa, n) != NULL)
    {
        n++;
    }
    if (scope == NULL)
    {
        return n;
    }
    void *la = bash_array("BASH_LINENO");
    *scope   = 0;
    for (int i = 0; i < n; i++)
    {
        const char *f = fn_array_reference(fa, i);
        if (strcmp(f, "_milk_source_impl") != 0)
        {
            if (strcmp(f, "source") == 0 || strncmp(f, "_milk_", 6) == 0 ||
                strncmp(f, "__milk_", 7) == 0 || f[0] == '@')
            {
                continue;
            }
            const char *l = (strcmp(f, "main") == 0 && la != NULL) ? fn_array_reference(la, i) : "";
            if (l == NULL || strcmp(l, "0") == 0)
            {
                continue;
            }
        }
        *scope = n - i;
        break;
    }
    return n;
}

/** own() - make the entries this process's: a subshell drops those it inherited (its parent runs them). */
static void own(void)
{
    pid_t pid = getpid();

    if (g_pid == pid)
    {
        return;
    }
    for (int i = 0; i < g_n; i++)
    {
        free(g_stack[i].cmd);
    }
    for (int i = 0; i < g_en; i++)
    {
        free(g_exit[i]);
    }
    g_n    = 0;
    g_en   = 0;
    g_sf_h = -1;
    g_pid  = pid;
    sh_eval("_milk_defer_init");
}

/** push() - register cmd (taken) in the scope at height h, 0: the session. */
static int push(char *cmd, int h)
{
    if (h > 0)
    {
        if (g_n == g_cap)
        {
            int            cap = g_cap ? 2 * g_cap : 16;
            defer_entry_t *s   = realloc(g_stack, (size_t) cap * sizeof(*s));
            if (s == NULL)
            {
                free(cmd);
                return 1;
            }
            g_stack = s;
            g_cap   = cap;
        }
        g_stack[g_n].cmd    = cmd;
        g_stack[g_n].height = h;
        g_n++;
        fn_set_signal(g_return_sig,
                      RETURN_TRAP); /* the frames running now fire it when they return */
        return 0;
    }
    if (g_en == g_ecap)
    {
        int    cap = g_ecap ? 2 * g_ecap : 16;
        char **e   = realloc(g_exit, (size_t) cap * sizeof(*e));
        if (e == NULL)
        {
            free(cmd);
            return 1;
        }
        g_exit = e;
        g_ecap = cap;
    }
    g_exit[g_en++] = cmd;
    sh_eval("_milk_defer_exit_trap"); /* again: a user `trap ... EXIT` may have replaced it */
    return 0;
}

/** quote() - append s to p in single quotes; the end. */
static char *quote(char *p, const char *s)
{
    *p++ = '\'';
    for (; *s != '\0'; s++)
    {
        if (*s == '\'')
        {
            p = stpcpy(p, "'\\''");
        }
        else
        {
            *p++ = *s;
        }
    }
    *p++ = '\'';
    return p;
}

/** join() - the words of argv as one command string, each quoted. */
static char *join(int argc, char **argv)
{
    size_t len = 1;
    for (int i = 0; i < argc; i++)
    {
        len += 4 * strlen(argv[i]) + 3;
    }
    char *s = malloc(len);
    if (s != NULL)
    {
        char *p = s;
        for (int i = 0; i < argc; i++)
        {
            p    = quote(p, argv[i]);
            *p++ = ' ';
        }
        *p = '\0';
    }
    return s;
}

/**
 * run() - move the entries of the scopes at height h or above (ended), innermost first, then with
 * with_exit the session's, into __MILK_DRUN_SH as `eval '<cmd>' || :` lines; status 1 if there are any.
 */
static int run(int h, int with_exit)
{
    char **cmds = malloc(((size_t) g_n + (size_t) g_en + 1) * sizeof(*cmds));
    int    k    = 0;
    size_t len  = 1;

    if (cmds == NULL)
    {
        return 0;
    }
    while (g_n > 0 && g_stack[g_n - 1].height >= h)
    {
        cmds[k++] = g_stack[--g_n].cmd;
    }
    if (g_n == 0)
    {
        fn_restore_default_signal(g_return_sig);
    }
    while (with_exit && g_en > 0)
    {
        cmds[k++] = g_exit[--g_en];
    }
    for (int i = 0; i < k; i++)
    {
        len += 4 * strlen(cmds[i]) + 16;
    }
    char *s = (k > 0) ? malloc(len + 16) : NULL;
    if (s != NULL)
    {
        char *p =
            stpcpy(s, "__MILK_DRUN_SH=\n"); /* run once, even if milkdefer is gone (@unload) */
        for (int i = 0; i < k; i++)
        {
            p = quote(stpcpy(p, "eval "), cmds[i]);
            p = stpcpy(p, " || :\n");
        }
        *p = '\0';
        fn_bind_variable("__MILK_DRUN_SH", s, 0);
        free(s);
    }
    for (int i = 0; i < k; i++)
    {
        free(cmds[i]);
    }
    free(cmds);
    return s != NULL;
}

/**
 * is_source - Check whether a Bash command string is a `source` or `.` invocation
 * @cmd: Command string from `BASH_COMMAND`
 *
 * Detects whether a `RETURN` trap was triggered by the completion of a plain
 * `source` or `.` command (with optional `builtin` or `command` prefix) so its
 * caller's deferred actions are not prematurely executed.
 *
 * Return: 1 if @cmd starts with a source builtin invocation, 0 otherwise
 */
static int is_source(const char *cmd)
{
    static const char *const pre[] = { "source ",         ". ",
                                       "builtin source ", "builtin . ",
                                       "command source ", "command . " };

    for (size_t i = 0; i < sizeof(pre) / sizeof(pre[0]); i++)
    {
        if (strncmp(cmd, pre[i], strlen(pre[i])) == 0)
        {
            return 1;
        }
    }
    return 0;
}

/** defer_return() - the RETURN trap (u: $_ then): run the entries of the frame that returns, if any. */
static int defer_return(const char *u)
{
    if (g_n == 0 || g_pid != getpid())
    {
        return 0;
    }
    int h = frames(NULL);
    if (h == 0 || h > g_stack[g_n - 1].height)
    {
        return 0;
    }
    const char *cmd = fn_get_string_value("BASH_COMMAND");
    if (cmd != NULL && is_source(cmd))
    {
        /* A sourced file ended, with its caller's FUNCNAME; or that caller returns right after the
         * source. Only then has $_ changed (to the source's last argument) since the file ended. */
        if (g_sf_h != h || strcmp(g_sf_cmd, cmd) != 0 || strcmp(g_sf_u, u) == 0)
        {
            free(g_sf_cmd);
            free(g_sf_u);
            g_sf_cmd = strdup(cmd);
            g_sf_u   = strdup(u);
            g_sf_h   = (g_sf_cmd != NULL && g_sf_u != NULL) ? h : -1;
            return 0;
        }
    }
    g_sf_h = -1;
    return run(h, 0);
}

/**
 * milk_bash_defer_argv - Register or execute scoped `@defer` cleanup commands in Bash
 * @argc: Number of argument tokens
 * @argv: Argument array (flags such as `--return`, `--exit`, `--restore`, or command words)
 *
 * Implements the `milkdefer` loadable builtin. Registers cleanup commands tied to
 * either the current function/`@source` call-frame height or the shell session exit,
 * and populates `__MILK_DRUN_SH` when invoked from the `RETURN` or `EXIT` trap.
 *
 * Return: 0 on success (or when no deferred commands need to run), 1 when trap
 *         commands were staged into `__MILK_DRUN_SH` or on error, 2 on bad usage
 */
int milk_bash_defer_argv(int argc, char **argv)
{
    int session = 0;

    if (resolve() != 0)
    {
        fprintf(stderr, "@defer: this bash lacks a function milkdefer needs\n");
        return 1;
    }
    if (argc > 0 && strcmp(argv[0], "--return") == 0)
    {
        return defer_return((argc > 1) ? argv[1] : "");
    }
    if (argc > 0 && strcmp(argv[0], "--orphans") == 0) /* before a prompt or milk-cli command */
    {
        return (g_n > 0 && g_pid == getpid()) ? run(1, 0) : 0;
    }
    if (argc > 0 && strcmp(argv[0], "--exit-run") == 0)
    {
        return (g_pid == getpid()) ? run(0, 1) : 0;
    }
    if (argc > 0 && strcmp(argv[0], "--restore") == 0)
    {
        char *q  = join(argc - 1, argv + 1);
        char *s  = (q != NULL) ? malloc(strlen(q) + 32) : NULL;
        int   rc = 1;
        if (s != NULL)
        {
            stpcpy(stpcpy(s, "_milk_defer_restore "), q);
            rc = sh_eval(s);
        }
        free(s);
        free(q);
        return rc;
    }
    if (argc > 0 && (strcmp(argv[0], "--exit") == 0 || strcmp(argv[0], "--") == 0))
    {
        session = (strcmp(argv[0], "--exit") == 0);
        argc--;
        argv++;
    }
    else if (argc > 0 && argv[0][0] == '-' && argv[0][1] != '\0')
    {
        fprintf(stderr, "Usage: @defer [--exit] <command...> | @defer --restore <spec>...\n");
        return 2;
    }
    if (argc == 0)
    {
        return 0;
    }
    char *cmd = (argc == 1) ? strdup(argv[0]) : join(argc, argv);
    int   h   = 0;
    if (cmd == NULL)
    {
        return 1;
    }
    own();
    if (!session)
    {
        frames(&h);
    }
    return push(cmd, h);
}
