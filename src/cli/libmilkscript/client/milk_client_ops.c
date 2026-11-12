// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_client_ops.c
 * @brief The client calls that run module commands and read or write values: `@get`, `@set`
 *        (and its undo journal), `@calc`, `@test`, `@exists`.
 *
 * Purpose:     the frequent calls need no request: the shell reads and writes shared memory and
 *              evaluates scalar statements in its own process; the session does the rest.
 * Does:        sends a module command (CALL) with the caller's stdin, stdout, stderr and working
 *              directory (SCM_RIGHTS), a piped stdin read into a memory file first; reads
 *              (@get: -v, -a, -A, --watch) and writes (@set: --restore, --tui) entities in shared
 *              memory, and asks the session for the rest (local images); records the values
 *              each @set changed in the shell's __MILK_UNDO, for `@restore --undo`;
 *              evaluates @calc and @test in the shell when every statement is scalar
 *              (mcalc_local()), else in the session; tells whether an entity exists or runs.
 * Used by:     the bash builtins (milk_bash_builtins.c), milk-call (milk_call_main.c);
 *              milk_client_stream.c and milk_sessions.c (milk_client_call()). Declared in
 *              milk_client.h and milk_client_internal.h.
 * Uses:        milk_shm_prop.h (libmilkprop: entities in shared memory), mcalc.h (the in-shell
 *              scalar evaluator), milk_client_internal.h (the connection), milk_ipc_common.h
 *              (the caller's variables), milk_srvlog.h (a @set done in the shell goes to the
 *              session's log).
 * See also:    milk_client_stream.c (the other keywords); docs/guide/scripting.md, "Entities"
 *              and "Cost Model".
 * Constraints: also compiled into libmilkbash.so, the thin client every shell loads.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "milk_keywords.h"
#include "CLIcore_script.h"
#include "mcalc.h"
#include "milk_client.h"
#include "milk_client_internal.h"
#include "milk_entity_ns.h"
#include "milk_ipc_common.h"
#include "milk_proto.h"
#include "milk_shm_prop.h"
#include "milk_srvlog.h"

/**
 * drain_stdin_to_memfd - Buffer piped stdin into an anonymous memfd before connecting to the server
 * @argc: Command argument count
 * @argv: Command argument vector
 *
 * Reading stdin prior to opening the single-threaded server connection prevents
 * deadlocks in pipelines such as `@grab --stdout im1 | ... | @replay --stdin im2`,
 * while supporting arbitrary-length text and binary (`--raw`) payloads.
 *
 * Return: Seeked memfd file descriptor on success, or -1 if stdin is not read by @argv.
 */
static int drain_stdin_to_memfd(int argc, char **argv)
{
    if (argc < 1 || isatty(STDIN_FILENO))
    {
        return -1;
    }
    const char *c0 = argv[0];
    if (c0[0] == '@')
    {
        c0++;
    }
    /* streamwrite (@replay --stdin) reads stdin only when told to with a '-' argument */
    int is_sw = 0;
    for (int i = 1; strcmp(c0, "streamwrite") == 0 && i < argc; i++)
    {
        is_sw |= (strcmp(argv[i], "-") == 0);
    }
    int is_batch = (strcmp(c0, "pipe") == 0 && argc == 1);
    if (!is_sw && !is_batch)
    {
        return -1;
    }

    int mfd = memfd_create("milk_sw_stdin", MFD_CLOEXEC);
    if (mfd < 0)
    {
        return -1;
    }

    uint8_t buf[8192];
    size_t  total = 0;
    ssize_t nr;
    while ((nr = read(STDIN_FILENO, buf, sizeof(buf))) > 0)
    {
        ssize_t off = 0;
        while (off < nr)
        {
            ssize_t nw = write(mfd, buf + off, (size_t) (nr - off));
            if (nw <= 0)
            {
                break;
            }
            off += nw;
        }
        total += (size_t) nr;
    }

    if (total == 0)
    {
        close(mfd);
        return -1;
    }
    lseek(mfd, 0, SEEK_SET);
    return mfd;
}

/**
 * client_call_sync - Synchronous wrapper around `milk_client_call()` for `milk_client_capture_array()`
 * @argc:  Argument count
 * @argv:  Argument vector
 * @hooks: Optional shell variable hooks
 *
 * Return: Command exit status.
 */
static int client_call_sync(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    return milk_client_call(argc, argv, 0, hooks);
}

/**
 * milk_client_call_as - Dispatch `@cmd args` as server command @as with option @opt stripped
 * @as:         Target server command name (`streamcat`, `streamwrite`, `fparam`)
 * @argc:       Source argument count
 * @argv:       Source argument vector
 * @opt:        Option flag to strip (`--stdout`, `--stdin`, `--tui`)
 * @last:       Optional trailing token to append if not already present (e.g. `"-"`)
 * @async_flag: Non-zero for asynchronous execution
 * @hooks:      Optional shell variable hooks
 *
 * Translates high-level CLI flags (`@grab --stdout`, `@replay --stdin`, `@set --tui`)
 * into their underlying server commands.
 *
 * Return: Command exit status.
 */
int milk_client_call_as(const char                *as,
                        int                        argc,
                        char                     **argv,
                        const char                *opt,
                        const char                *last,
                        int                        async_flag,
                        const milk_client_hooks_t *hooks)
{
    char **av = calloc((size_t) argc + 3, sizeof(char *));
    int    n  = 0;
    int    rc;

    if (av == NULL)
    {
        return 1;
    }
    av[n++] = (char *) as;
    for (int i = 0; i < argc; i++)
    {
        if (strcmp(argv[i], opt) != 0)
        {
            av[n++] = argv[i];
        }
    }
    if (last != NULL && strcmp(av[n - 1], last) != 0)
    {
        av[n++] = (char *) last;
    }
    rc = milk_client_call(n, av, async_flag, hooks);
    free(av);
    return rc;
}

/**
 * milk_client_has_arg - Check whether option string @opt is present in `@argv[0..@argc)`
 * @argc: Argument count
 * @argv: Argument vector
 * @opt:  Option string to match
 *
 * Return: 1 if @opt is present, 0 otherwise.
 */
int milk_client_has_arg(int argc, char **argv, const char *opt)
{
    for (int i = 0; i < argc; i++)
    {
        if (strcmp(argv[i], opt) == 0)
        {
            return 1;
        }
    }
    return 0;
}

/**
 * milk_client_call - Dispatch a CLI command to `milk-cli --server` with caller stdio/cwd fds
 * @argc:       Argument count (`argv[0]` is the command name)
 * @argv:       Argument vector
 * @flags:      MILK_CALL_ASYNC (run asynchronously on the server), MILK_CALL_FRONTEND, MILK_CALL_NOWAIT
 * @hooks:      Optional shell variable hooks
 *
 * Packs `argv`, propagated environment variables, and referenced shell variables
 * into a `MILK_MSG_CALL` frame and passes caller stdio + working-directory file
 * descriptors over `SCM_RIGHTS`.
 *
 * Return: Server command exit status, or `MILK_CLIENT_RC_TRANSPORT` on connection failure.
 */
int milk_client_call(int argc, char **argv, int flags, const milk_client_hooks_t *hooks)
{
    int async_flag = (flags & MILK_CALL_ASYNC) != 0;
    if (argc <= 0 || argv == NULL || argv[0] == NULL)
    {
        return 0;
    }

    const char *c0 = argv[0];
    if (c0[0] == '@')
    {
        c0++;
    }
    if (strcmp(c0, "replay") == 0 && milk_client_has_arg(argc - 1, argv + 1, "--stdin"))
    {
        return milk_client_call_as("streamwrite", argc - 1, argv + 1, "--stdin", "-", async_flag,
                                   hooks);
    }
    int arc;
    if (!async_flag && strcmp(c0, "streamcat") == 0 &&
        (arc = milk_client_capture_array(client_call_sync, argc, argv, NULL, hooks)) >= 0)
    {
        return arc;
    }

    int sw_stdin_fd = drain_stdin_to_memfd(argc, argv);

    char        sockpath[256];
    const char *esock =
        (flags & MILK_CALL_FRONTEND) ? milk_ipc_get_var(hooks, "_MILK_FRONTEND_SOCK") : NULL;
    int fd;
    if (esock != NULL && esock[0] != '\0')
    {
        snprintf(sockpath, sizeof(sockpath), "%s",
                 esock); // milk-cli's own: never started from here
        if ((fd = milk_client_connect(sockpath)) < 0)
        {
            fprintf(stderr, "milk: milk-cli's session does not answer (%s)\n", sockpath);
        }
    }
    else
    {
        fd = milk_client_connect_or_start(hooks, sockpath, sizeof(sockpath));
    }
    if (fd < 0)
    {
        if (sw_stdin_fd >= 0)
        {
            close(sw_stdin_fd);
        }
        return MILK_CLIENT_RC_TRANSPORT;
    }

    /* only calc text reads shell variables: `n` in `mem.mk2Dim im n 4` is no variable */
    milk_packed_vars_t pv;
    pv.count = 0;
    milk_client_pack_digits(&pv, hooks);
    for (int i = 1; i < argc && (strcmp(c0, "assert") == 0 || strcmp(c0, "pipe") == 0); i++)
    {
        milk_client_collect_identifiers(&pv, hooks, argv[i]);
    }

    static const char *const env_keys[] = { "MILK_SHM_DIR", "MILK_QUIET", "NO_COLOR",
                                            "MILK_FLOAT_DIGITS", NULL };
    char                     env_kv[8][256];
    uint32_t                 nenv = 0;
    for (int i = 0; env_keys[i] != NULL && nenv < 8; i++)
    {
        const char *ev = milk_ipc_get_var(hooks, env_keys[i]);
        if (ev != NULL)
        {
            snprintf(env_kv[nenv], sizeof(env_kv[nenv]), "%s=%s", env_keys[i], ev);
            nenv++;
        }
    }

    size_t cap = sizeof(uint32_t) * 3 + 256;
    for (int i = 0; i < argc; i++)
    {
        cap += strlen(argv[i]) + 1;
    }
    for (uint32_t i = 0; i < nenv; i++)
    {
        cap += strlen(env_kv[i]) + 1;
    }
    for (int i = 0; i < pv.count; i++)
    {
        cap += strlen(pv.names[i]) + 1 + strlen(pv.vals[i]) + 1;
    }

    uint8_t *buf = (uint8_t *) malloc(cap);
    if (buf == NULL)
    {
        if (sw_stdin_fd >= 0)
        {
            close(sw_stdin_fd);
        }
        milk_client_end(fd);
        return 1;
    }

    uint32_t off   = 0;
    uint32_t uargc = (uint32_t) argc;
    memcpy(buf + off, &uargc, sizeof(uargc));
    off += (uint32_t) sizeof(uargc);
    for (int i = 0; i < argc; i++)
    {
        size_t l = strlen(argv[i]) + 1;
        memcpy(buf + off, argv[i], l);
        off += (uint32_t) l;
    }

    memcpy(buf + off, &nenv, sizeof(nenv));
    off += (uint32_t) sizeof(nenv);
    for (uint32_t i = 0; i < nenv; i++)
    {
        size_t l = strlen(env_kv[i]) + 1;
        memcpy(buf + off, env_kv[i], l);
        off += (uint32_t) l;
    }

    uint32_t nvars = (uint32_t) pv.count;
    memcpy(buf + off, &nvars, sizeof(nvars));
    off += (uint32_t) sizeof(nvars);
    for (int i = 0; i < pv.count; i++)
    {
        size_t nl = strlen(pv.names[i]) + 1;
        size_t vl = strlen(pv.vals[i]) + 1;
        memcpy(buf + off, pv.names[i], nl);
        off += (uint32_t) nl;
        memcpy(buf + off, pv.vals[i], vl);
        off += (uint32_t) vl;
    }

    int cwd_fd = open(".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    int fds[4] = { (sw_stdin_fd >= 0) ? sw_stdin_fd : STDIN_FILENO, STDOUT_FILENO, STDERR_FILENO,
                   cwd_fd };
    uint8_t msg_flags = (uint8_t) ((async_flag ? MILK_FLAG_ASYNC : 0) |
                                   ((flags & MILK_CALL_NOWAIT) ? MILK_FLAG_NOWAIT : 0));
    int     sret      = milk_proto_send_with_fds(fd, MILK_MSG_CALL, msg_flags, buf, off, fds,
                                                 (cwd_fd >= 0) ? 4 : 3);
    if (sw_stdin_fd >= 0)
    {
        close(sw_stdin_fd);
    }
    if (cwd_fd >= 0)
    {
        close(cwd_fd);
    }
    free(buf);

    if (sret != 0)
    {
        milk_client_end(fd);
        return 1;
    }

    int rc = milk_client_await_response(fd, sockpath, hooks, NULL, 0, NULL);
    milk_client_end(fd);
    return rc;
}

/**
 * milk_client_get_buf - Read entity property @spec into caller buffer @out
 * @spec:  Entity property specification (`s.im.cnt0`, `f.name.param`, ...)
 * @out:   Destination character buffer
 * @outsz: Capacity of @out in bytes
 * @hooks: Optional shell variable hooks
 *
 * Attempts a zero-RPC `/dev/shm` read via `milk_shm_prop_get()` first and falls
 * back to `MILK_MSG_GET` on the server for server-local (`l.*`) or complex entities.
 *
 * Return: 0 on success, non-zero on error.
 */
int milk_client_get_buf(const char *spec, char *out, size_t outsz, const milk_client_hooks_t *hooks)
{
    if (spec == NULL || spec[0] == '\0' || out == NULL || outsz == 0)
    {
        return 1;
    }
    out[0] = '\0';
    milk_ipc_sync_shm_env(hooks);

    if (milk_shm_prop_get(spec, out, outsz) == 0)
    {
        return 0;
    }
    if (milk_entity_ns_removed(strncmp(spec, "json:", 5) == 0 ? spec + 5 : spec) != NULL)
    {
        return 1; /* milk_shm_prop_get gave the hint; the server knows no more */
    }

    char sockpath[256];
    int  fd = milk_client_connect_or_start(hooks, sockpath, sizeof(sockpath));
    if (fd < 0)
    {
        return MILK_CLIENT_RC_TRANSPORT;
    }

    uint32_t slen = (uint32_t) (strlen(spec) + 1);
    if (milk_proto_send(fd, MILK_MSG_GET, 0, spec, slen) != 0)
    {
        milk_client_end(fd);
        return 1;
    }

    int got_val = 0;
    int rc      = milk_client_await_response(fd, sockpath, hooks, out, outsz, &got_val);
    milk_client_end(fd);
    return (rc == 0 && got_val) ? 0 : 1;
}

/**
 * milk_client_get - Read entity property @spec and print it or bind it to @varname
 * @spec:    Entity property specification
 * @varname: Optional shell variable name to bind (or NULL to print to stdout)
 * @hooks:   Optional shell variable hooks
 *
 * Return: 0 on success, non-zero on error.
 */
int milk_client_get(const char *spec, const char *varname, const milk_client_hooks_t *hooks)
{
    char val[16384];
    int  rc = milk_client_get_buf(spec, val, sizeof(val), hooks);
    if (rc == 0)
    {
        if (varname != NULL && varname[0] != '\0')
        {
            milk_ipc_set_var(hooks, varname, val, 0);
        }
        else
        {
            printf("%s\n", val);
            fflush(stdout);
        }
    }
    return rc;
}

/**
 * milk_client_set - Write entity property @spec (`@set`), using `/dev/shm` fast path first
 * @spec:    Entity specification (`f.name.param=val` or `f.name.param`)
 * @val:     Optional value string when not embedded in @spec
 * @comment: Optional FITS keyword comment
 * @hooks:   Optional shell variable hooks
 *
 * Writes directly to `/dev/shm` via `milk_shm_prop_set()` when possible (and records
 * the operation in the server ring log), sending `MILK_MSG_SET` to the server for local
 * images, globs, and streams that are not in shared memory (they may be the server's
 * local images). A write that fails here is not sent again: the server would only fail
 * the same way and print its error a second time.
 *
 * Return: 0 on success, non-zero on error.
 */
int milk_client_set(const char                *spec,
                    const char                *val,
                    const char                *comment,
                    const milk_client_hooks_t *hooks)
{
    if (spec == NULL || spec[0] == '\0')
    {
        milk_lexicon_usage("set", stderr);
        return 1;
    }
    if (strchr(spec, '.') == NULL)
    {
        fprintf(stderr, "Error: @set: invalid entity path '%s' (expected <entity>.<prop>)\n", spec);
        return 1;
    }
    milk_ipc_sync_shm_env(hooks);

    char s_norm[512];
    milk_normalize_entity_spec(spec, s_norm, sizeof(s_norm));
    milk_ns_kind_t nk = milk_entity_ns_kind(s_norm, NULL);
    int            is_local_or_glob =
        (nk == MILK_NS_LOCAL ||
         (nk == MILK_NS_STREAM && (strchr(s_norm, '*') != NULL || strchr(s_norm, '?') != NULL)));

    int here = !is_local_or_glob;
    if (here && nk == MILK_NS_STREAM)
    {
        char ent[512];
        snprintf(ent, sizeof(ent), "%s", s_norm);
        char *d1 = strchr(ent, '.');
        char *d2 = (d1 != NULL) ? strchr(d1 + 1, '.') : NULL;
        if (d2 != NULL)
        {
            *d2 = '\0'; // s.<name>
        }
        here = milk_shm_prop_exists(ent, 0);
    }

    struct timespec ts0;
    clock_gettime(CLOCK_REALTIME, &ts0);

    int lrc = here ? milk_shm_prop_set(spec, val, comment) : -1;
    if (lrc == MILK_SHM_PROP_ACTION_FAILED)
    {
        return 1; // a unit start or stop failed here, with its messages: not again by the server
    }
    if (lrc == 0)
    {
        struct timespec ts1;
        clock_gettime(CLOCK_REALTIME, &ts1);
        int64_t dur = ((int64_t) ts1.tv_sec - (int64_t) ts0.tv_sec) * 1000000000LL +
                      ((int64_t) ts1.tv_nsec - (int64_t) ts0.tv_nsec);
        char    sockpath[256];
        milk_ipc_resolve_sockpath(hooks, sockpath, sizeof(sockpath), 1);
        char lcmd[256];
        if (val != NULL)
        {
            snprintf(lcmd, sizeof(lcmd), "@set %s = %s", spec, val);
        }
        else
        {
            snprintf(lcmd, sizeof(lcmd), "@set %s", spec);
        }
        milk_srvlog_client_record(sockpath, &ts0, (dur > 0) ? (uint64_t) dur : 0, MILK_MSG_SET, 0,
                                  lcmd, "");
        return 0;
    }
    if (here)
    {
        return 1; // it failed here with its message; the server runs the same write
    }

    char sockpath[256];
    int  fd = milk_client_connect_or_start(hooks, sockpath, sizeof(sockpath));
    if (fd < 0)
    {
        return MILK_CLIENT_RC_TRANSPORT;
    }

    size_t   l1  = strlen(spec) + 1;
    size_t   l3  = (comment != NULL) ? (strlen(comment) + 1) : 0;
    size_t   l2  = (val != NULL) ? (strlen(val) + 1)
                                 : (l3 > 0 ? 1 : 0); /* empty slot keeps comment in place */
    uint32_t tot = (uint32_t) (l1 + l2 + l3);

    uint8_t *buf = (uint8_t *) malloc(tot);
    if (buf == NULL)
    {
        milk_client_end(fd);
        return 1;
    }
    memcpy(buf, spec, l1);
    if (l2 > 0)
    {
        memcpy(buf + l1, (val != NULL) ? val : "", l2);
    }
    if (l3 > 0)
    {
        memcpy(buf + l1 + l2, comment, l3);
    }

    int fds[3] = { STDIN_FILENO, STDOUT_FILENO, STDERR_FILENO };
    int sret   = milk_proto_send_with_fds(fd, MILK_MSG_SET, 0, buf, tot, fds, 3);
    free(buf);
    if (sret != 0)
    {
        milk_client_end(fd);
        return 1;
    }

    int rc = milk_client_await_response(fd, sockpath, hooks, NULL, 0, NULL);
    milk_client_end(fd);
    return rc;
}

/**
 * assoc_spec - Normalize an `@get -A` target @spec into its `.kv` key-value listing path
 * @spec:  Input entity specification (`s.cam`, `s.cam.kw`, `f.x.*`, `f.x.list`, ...)
 * @out:   Destination buffer for the `.kv` path
 * @outsz: Capacity of @out in bytes
 */
static void assoc_spec(const char *spec, char *out, size_t outsz)
{
    while (*spec == '@')
    {
        spec++;
    }
    size_t      slen = strlen(spec);
    const char *dot1 = strchr(spec, '.');
    if (slen > 5 && strcmp(spec + slen - 5, ".kw.*") == 0)
    {
        snprintf(out, outsz, "%.*s.kw.kv", (int) (slen - 5), spec);
    }
    else if (slen > 3 && strcmp(spec + slen - 3, ".kw") == 0)
    {
        snprintf(out, outsz, "%s.kv", spec);
    }
    else if (slen > 2 && strcmp(spec + slen - 2, ".*") == 0)
    {
        snprintf(out, outsz, "%.*s.kv", (int) (slen - 2), spec);
    }
    else if (slen > 5 && strcmp(spec + slen - 5, ".list") == 0)
    {
        snprintf(out, outsz, "%.*s.kv", (int) (slen - 5), spec);
    }
    else if (dot1 != NULL && strpbrk(dot1 + 1, ".*?") == NULL)
    {
        snprintf(out, outsz, "%s.kv", spec);
    }
    else
    {
        snprintf(out, outsz, "%s", spec);
    }
}

/**
 * client_bind_array - Bind newline/space-delimited @text to a shell indexed or associative array
 * @hooks: Shell variable/eval hooks
 * @name:  Target shell array variable name
 * @assoc: Non-zero for `declare -gA` associative array, 0 for indexed array
 * @text:  Output text to parse into array elements
 */
static void client_bind_array(const milk_client_hooks_t *hooks,
                              const char                *name,
                              int                        assoc,
                              char                      *text)
{
    char *script = milk_ipc_bind_script(name, assoc, text);
    if (script != NULL)
    {
        milk_ipc_eval(hooks, script);
        free(script);
    }
}

/**
 * get_specs - Execute a batch of `@get spec...` and `@get name=spec...` arguments
 * @n:         Number of specifications in @specs
 * @specs:     Array of `spec` or `name=spec` strings
 * @json_mode: Non-zero to prefix plain specs with `json:`
 * @hooks:     Optional shell variable hooks
 *
 * Return: 0 if all reads succeeded, 1 if any read failed, or 2 on invalid variable name.
 */
static int get_specs(int n, char **specs, int json_mode, const milk_client_hooks_t *hooks)
{
    int rc = 0;
    for (int i = 0; i < n; i++)
    {
        char        js[512];
        const char *eq = strchr(specs[i], '=');
        char        name[128];
        const char *spec = specs[i];
        if (eq != NULL) /* name=spec: entity paths never contain '=' */
        {
            snprintf(name, sizeof(name), "%.*s", (int) (eq - specs[i]), specs[i]);
            if (!milk_ipc_is_identifier(name))
            {
                fprintf(stderr, "milk: @get: '%s' is no variable name\n", name);
                return 2;
            }
            spec = eq + 1;
        }
        if (json_mode && eq == NULL)
        {
            snprintf(js, sizeof(js), "json:%s", spec);
            spec = js;
        }
        rc |= (milk_client_get(spec, (eq != NULL) ? name : NULL, hooks) != 0);
    }
    return rc;
}

static volatile sig_atomic_t g_watch_stop = 0;

static void watch_sigint(int signo)
{
    (void) signo;
    g_watch_stop = 1;
}

/*
 * @get --watch [T] <spec>...: the values every @period s until Ctrl-C, redrawn in place at a
 * terminal, else one tab-separated line per refresh. Return: 0 once Ctrl-C ends it, 1 if a spec
 * fails at the first refresh
 */
static int get_watch(int                        n,
                     char                     **specs,
                     int                        json_mode,
                     double                     period,
                     const milk_client_hooks_t *hooks)
{
    struct sigaction sa;
    struct sigaction old;
    char             js[512];
    int              tty   = isatty(STDOUT_FILENO);
    int              lines = 0; /* printed by the last refresh, at a terminal */
    int              rc    = 0;
    char            *val   = (char *) malloc(16384);

    if (val == NULL)
    {
        return 1;
    }
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = watch_sigint; /* no SA_RESTART: the sleep ends at once */
    sigemptyset(&sa.sa_mask);
    g_watch_stop = 0;
    sigaction(SIGINT, &sa, &old);
    for (int first = 1; !g_watch_stop; first = 0)
    {
        if (tty && lines > 0)
        {
            printf("\033[%dA\r", lines);
        }
        lines = 0;
        for (int i = 0; i < n && !g_watch_stop; i++)
        {
            const char *spec = specs[i];
            if (json_mode)
            {
                snprintf(js, sizeof(js), "json:%s", specs[i]);
                spec = js;
            }
            if (milk_client_get_buf(spec, val, 16384, hooks) != 0)
            {
                if (first)
                {
                    rc           = 1;
                    g_watch_stop = 1;
                    break;
                }
                snprintf(val, 16384, "?");
            }
            if (tty)
            {
                printf("\033[2K%s = ", specs[i]);
                for (char *p = val; *p != '\0'; p++)
                {
                    fputc(*p, stdout);
                    if (*p == '\n' && p[1] != '\0')
                    {
                        fputs("\033[2K  ", stdout);
                        lines++;
                    }
                }
                fputc('\n', stdout);
                lines++;
            }
            else
            {
                for (char *p = val; *p != '\0'; p++)
                {
                    *p = (*p == '\n' || *p == '\t') ? ' ' : *p;
                }
                printf("%s%s", (i > 0) ? "\t" : "", val);
            }
        }
        if (rc == 0)
        {
            fputs(tty ? "\033[J" : "\n", stdout); /* a shorter refresh leaves no old lines */
        }
        fflush(stdout);
        struct timespec ts = { (time_t) period,
                               (long) ((period - (double) (time_t) period) * 1e9) };
        while (!g_watch_stop && nanosleep(&ts, &ts) != 0 && errno == EINTR)
        {
        }
    }
    sigaction(SIGINT, &old, NULL);
    g_watch_stop = 0;
    free(val);
    return rc;
}

/**
 * milk_client_get_argv - Parse and execute `@get` command-line arguments
 * @argc:  Argument count
 * @argv:  Argument vector (`--json`, `-v var`, `-a arr`, `-A assoc`, `<entity>...`)
 * @hooks: Optional shell variable hooks
 *
 * Return: 0 on success, non-zero on error.
 */
int milk_client_get_argv(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    milk_kw_args_t ka;
    if (milk_kw_parse("get", 0, &argc, argv, &ka))
    {
        return ka.rc;
    }
    const char *varname   = milk_kw_opt(&ka, "-v");
    const char *arrname   = milk_kw_opt(&ka, "-a");
    const char *assocname = milk_kw_opt(&ka, "-A");
    const char *spec      = NULL;
    const char *wt        = milk_kw_opt(&ka, "--watch"); /* --watch [T]: the period */
    int         json_mode = (milk_kw_opt(&ka, "--json") != NULL);
    int         nspec     = argc;
    char      **specs     = argv;
    char       *end       = NULL;
    double      watch     = (wt == NULL) ? 0.0 : (wt[0] == '\0') ? 0.5 : strtod(wt, &end);
    watch                 = (end != NULL && (*end != '\0' || watch <= 0.0)) ? -1.0 : watch;
    for (int i = 0; watch != 0.0 && i < nspec; i++)
    {
        watch = (strchr(specs[i], '=') != NULL) ? -1.0 : watch;
    }
    if (watch != 0.0)
    {
        int rc =
            (watch < 0.0 || nspec == 0 || varname != NULL || arrname != NULL || assocname != NULL)
                ? -1
                : get_watch(nspec, specs, json_mode, watch, hooks);
        if (rc < 0)
        {
            milk_lexicon_usage("get", stderr);
            return 2;
        }
        return rc;
    }
    spec = (nspec > 0) ? specs[0] : NULL;
    if (nspec == 0 || ((varname != NULL || arrname != NULL || assocname != NULL) &&
                       (nspec > 1 || strchr(spec, '=') != NULL)))
    {
        milk_lexicon_usage("get", stderr);
        return 2;
    }
    if (varname == NULL && arrname == NULL && assocname == NULL)
    {
        int rc = get_specs(nspec, specs, json_mode, hooks);
        return rc;
    }

    char eff_spec[512];
    if (assocname != NULL)
    {
        assoc_spec(spec, eff_spec, sizeof(eff_spec));
        spec = eff_spec;
    }
    else if (json_mode)
    {
        snprintf(eff_spec, sizeof(eff_spec), "json:%s", spec);
        spec = eff_spec;
    }
    if (arrname == NULL && assocname == NULL)
    {
        return milk_client_get(spec, varname, hooks);
    }

    char *val = (char *) malloc(16384);
    if (val == NULL)
    {
        return 1;
    }
    int rc = milk_client_get_buf(spec, val, 16384, hooks);
    if (rc == 0)
    {
        client_bind_array(hooks, arrname != NULL ? arrname : assocname, arrname == NULL, val);
    }
    free(val);
    return rc;
}

/**
 * milk_client_capture_array - Run @fn with `-a <arr>` stripped, capture stdout, and bind to `<arr>`
 * @fn:    Command callback (`client_call_sync` or `milk_client_ls`)
 * @argc:  Argument count
 * @argv:  Argument vector
 * @extra: Optional extra flag to append when capturing (e.g. `"-1"` for `@ls -a`)
 * @hooks: Shell variable hooks
 *
 * Used by `@grab --stdout -a arr` and `@ls -a arr` to capture stdout into an
 * anonymous memfd and bind the resulting words to a bash array without a subshell.
 *
 * Return: Command exit status if `-a <arr>` was present, or -1 if `@argv` has no `-a`.
 */
int milk_client_capture_array(int (*fn)(int, char **, const milk_client_hooks_t *),
                              int                        argc,
                              char                     **argv,
                              const char                *extra,
                              const milk_client_hooks_t *hooks)
{
    const char *arrname = NULL;
    char       *fargv[argc + 2];
    int         fargc = 0;
    for (int i = 0; i < argc; i++)
    {
        if (arrname == NULL && strcmp(argv[i], "-a") == 0 && i + 1 < argc)
        {
            arrname = argv[++i];
        }
        else
        {
            fargv[fargc++] = argv[i];
        }
    }
    if (arrname == NULL)
    {
        return -1;
    }
    if (extra != NULL)
    {
        fargv[fargc++] = (char *) extra;
    }
    fargv[fargc] = NULL;

    int mfd       = memfd_create("milk_arr", MFD_CLOEXEC);
    int saved_out = (mfd >= 0) ? dup(STDOUT_FILENO) : -1;
    if (mfd >= 0 && saved_out >= 0)
    {
        fflush(stdout);
        dup2(mfd, STDOUT_FILENO);
    }
    int rc = fn(fargc, fargv, hooks);
    if (saved_out >= 0)
    {
        fflush(stdout);
        dup2(saved_out, STDOUT_FILENO);
        close(saved_out);
    }
    if (mfd < 0)
    {
        return rc;
    }
    off_t sz = lseek(mfd, 0, SEEK_END);
    lseek(mfd, 0, SEEK_SET);
    char *obuf = (rc == 0 && sz >= 0) ? (char *) malloc((size_t) sz + 1) : NULL;
    if (obuf != NULL)
    {
        ssize_t nr              = read(mfd, obuf, (size_t) sz);
        obuf[(nr > 0) ? nr : 0] = '\0';
        client_bind_array(hooks, arrname, 0, obuf);
        free(obuf);
    }
    close(mfd);
    return rc;
}

/**
 * sh_quote - Append @s single-quoted for bash evaluation onto @out
 * @out: Destination NUL-terminated buffer
 * @sz:  Total capacity of @out in bytes
 * @s:   String to quote and append
 *
 * Return: 0 on success, -1 if @out lacks capacity.
 */
static int sh_quote(char *out, size_t sz, const char *s)
{
    size_t n = strlen(out);
    if (n + 3 > sz)
    {
        return -1;
    }
    out[n++] = '\'';
    for (; *s != '\0'; s++)
    {
        if (n + 6 > sz)
        {
            return -1;
        }
        if (*s == '\'')
        {
            memcpy(out + n, "'\\''", 4);
            n += 4;
        }
        else
        {
            out[n++] = *s;
        }
    }
    out[n++] = '\'';
    out[n]   = '\0';
    return 0;
}

/**
 * set_restore - Snapshot current entity values before `@set --restore` and register a `@defer` restore command
 * @argc:  Assignment argument count
 * @argv:  Assignment argument vector
 * @hooks: Shell variable/eval hooks
 *
 * Reads the pre-write value of each assigned entity and registers a
 * `__MILK_NOLOG=1 milkset ...` command in the caller's `@defer` scope so old
 * values are automatically restored when the enclosing function or script exits.
 *
 * Return: 0 on success, non-zero if any pre-write value cannot be read.
 */
static int set_restore(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    char cmd[16384] = "__MILK_NOLOG=1 milkset";
    int  i          = 0;
    while (i < argc)
    {
        milk_set_item_t it;
        milk_set_item(argc, argv, &i, &it);
        const char *eq = strchr(it.spec, '=');
        char        name[512];
        char        old[4096];
        char        kv[4700];
        snprintf(name, sizeof(name), "%.*s",
                 (int) ((eq != NULL) ? (size_t) (eq - it.spec) : strlen(it.spec)), it.spec);
        if (eq == NULL && it.val == NULL) /* an action: nothing to put back */
        {
            continue;
        }
        if (strpbrk(name, "*?[") != NULL)
        {
            fprintf(stderr, "milk: @set --restore %s: for a glob, use @defer --restore\n", name);
            return 2;
        }
        if (milk_client_get_buf(name, old, sizeof(old), hooks) != 0)
        {
            fprintf(stderr, "milk: @set --restore: cannot read %s\n", name);
            return 1;
        }
        snprintf(kv, sizeof(kv), "%s=%s", name, old);
        if (strlen(cmd) + 1 >= sizeof(cmd) ||
            (strcat(cmd, " "), sh_quote(cmd, sizeof(cmd), kv)) != 0)
        {
            fprintf(stderr, "milk: @set --restore: too many values\n");
            return 1;
        }
    }
    char script[sizeof(cmd) * 2 + 32] = "_milk_defer_here ";
    if (sh_quote(script, sizeof(script), cmd) == 0)
    {
        milk_ipc_eval(hooks, script);
    }
    return 0;
}

/* @restore --undo: each @set call adds an entry to the shell's __MILK_UNDO array, a
 * "<spec>\t<old>\t<new>" line per value it changed */
typedef struct
{
    char  *buf;
    size_t len;
    size_t cap;
} undo_text_t;

/**
 * undo_add - Append @n bytes from @line to growable undo buffer @u
 * @u:    Target undo buffer
 * @line: Byte slice to append
 * @n:    Length of @line in bytes
 */
static void undo_add(undo_text_t *u, const char *line, size_t n)
{
    if (u->len + n + 1 > u->cap)
    {
        size_t cap = (u->cap + n + 1) * 2;
        char  *b   = realloc(u->buf, cap);
        if (b == NULL)
        {
            return;
        }
        u->buf = b;
        u->cap = cap;
    }
    memcpy(u->buf + u->len, line, n);
    u->len += n;
    u->buf[u->len] = '\0';
}

/**
 * undo_loggable - Check whether entity @name is a reversible parameter/keyword for `@restore --undo`
 * @name: Entity path without `=value`
 *
 * Excludes state transitions and actions (`conf`, `run`, `status`, process controls)
 * so `@restore --undo` only reverts data-bearing FPS parameters and stream keywords.
 *
 * Return: 1 if changes to @name should be recorded in `__MILK_UNDO`, 0 otherwise.
 */
static int undo_loggable(const char *name)
{
    const char    *rest = name;
    milk_ns_kind_t k    = milk_entity_ns_kind(name, &rest);
    const char    *last = strrchr(name, '.');

    if (k == MILK_NS_STREAM)
    {
        return strstr(rest, ".kw.") != NULL;
    }
    return k == MILK_NS_FPS && last != NULL && strcmp(last, ".conf") != 0 &&
           strcmp(last, ".run") != 0 && strcmp(last, ".status") != 0;
}

/**
 * undo_values - Format `<spec>\t<value>\n` lines for @name (expanding globs) from `/dev/shm`
 * @name:  Entity specification (single path or glob)
 * @out:   Destination buffer
 * @sz:    Capacity of @out in bytes
 * @hooks: Shell variable hooks
 *
 * Return: 0 on success, -1 if @name cannot be read from `/dev/shm`.
 */
static int undo_values(const char *name, char *out, size_t sz, const milk_client_hooks_t *hooks)
{
    char   raw[8192];
    size_t o = 0;

    milk_ipc_sync_shm_env(hooks);
    if (milk_shm_prop_get(name, raw, sizeof(raw)) !=
        0) /* shared memory only: no request to the server */
    {
        return -1;
    }
    if (strpbrk(name, "*?[") == NULL)
    {
        raw[strcspn(raw, "\n")] = '\0';
        snprintf(out, sz, "%s\t%s\n", name, raw);
        return 0;
    }
    size_t      nl   = strlen(name);
    int         tail = (nl > 2 && strcmp(name + nl - 2, ".*") == 0 &&
                        strpbrk(name, "*?[") == name + nl - 1); /* <entity>.* */
    const char *ent  = strchr(name, '.');
    const char *att  = (ent != NULL) ? strchr(ent + 1, '.') : NULL;
    if (!tail && (ent == NULL || att == NULL || strpbrk(att, "*?[") != NULL))
    {
        return -1;
    }
    out[0] = '\0';
    for (char *line = strtok(raw, "\n"); line != NULL && o < sz; line = strtok(NULL, "\n"))
    {
        char *eq = strchr(line, '=');
        if (eq == NULL)
        {
            continue;
        }
        *eq   = '\0';
        int w = tail ? snprintf(out + o, sz - o, "%.*s%s\t%s\n", (int) (nl - 1), name, line, eq + 1)
                     : snprintf(out + o, sz - o, "%.*s.%s%s\t%s\n", (int) (ent - name), name, line,
                                att, eq + 1);
        o += (w > 0) ? (size_t) w : 0;
    }
    return (o < sz) ? 0 : -1;
}

/**
 * undo_pair - Append `<spec>\t<old>\t<new>\n` lines to @u for every spec in @before modified in @after
 * @u:      Target undo buffer
 * @before: Tab-separated `<spec>\t<old>\n` lines captured before `@set`
 * @after:  Tab-separated `<spec>\t<new>\n` lines captured after `@set`
 */
static void undo_pair(undo_text_t *u, char *before, const char *after)
{
    for (char *line = strtok(before, "\n"); line != NULL; line = strtok(NULL, "\n"))
    {
        char *tab = strchr(line, '\t');
        if (tab == NULL)
        {
            continue;
        }
        size_t      sl  = (size_t) (tab - line);
        const char *a   = after;
        const char *now = NULL;
        while (a != NULL && *a != '\0' && now == NULL)
        {
            if (strncmp(a, line, sl + 1) == 0)
            {
                now = a + sl + 1;
            }
            a = strchr(a, '\n');
            a = (a != NULL) ? a + 1 : NULL;
        }
        size_t nn = (now != NULL) ? strcspn(now, "\n") : 0;
        if (now != NULL && (strlen(tab + 1) != nn || strncmp(tab + 1, now, nn) != 0))
        {
            undo_add(u, line, strlen(line));
            undo_add(u, "\t", 1);
            undo_add(u, now, nn);
            undo_add(u, "\n", 1);
        }
    }
}

static undo_text_t
           g_undo_pending; /* entries of this shell not yet in __MILK_UNDO, '\036' after each */
static int g_undo_npending;

/**
 * undo_flush - Flush queued undo records in `g_undo_pending` into the shell's `__MILK_UNDO` array
 * @hooks: Shell variable/eval hooks
 *
 * Batches `__MILK_UNDO` updates across `@set` calls and caps the history ring at
 * 1,000-2,000 entries to bound shell memory consumption.
 */
static void undo_flush(const milk_client_hooks_t *hooks)
{
    if (g_undo_pending.len == 0)
    {
        return;
    }
    char *script = malloc(2 * g_undo_pending.len + 160);
    if (script != NULL)
    {
        char *p = stpcpy(script, "__MILK_UNDO+=($'");
        for (const char *c = g_undo_pending.buf; *c != '\0'; c++)
        {
            p = (*c == '\036' && c[1] == '\0') ? p
                : (*c == '\036')               ? stpcpy(p, "' $'")
                : (*c == '\\')                 ? stpcpy(p, "\\\\")
                : (*c == '\'')                 ? stpcpy(p, "\\'")
                : (*c == '\n')                 ? stpcpy(p, "\\n")
                : (*c == '\t')                 ? stpcpy(p, "\\t")
                                               : (*p = *c, p + 1);
        }
        stpcpy(
            p,
            "'); (( ${#__MILK_UNDO[@]} <= 2000 )) || __MILK_UNDO=(\"${__MILK_UNDO[@]: -1000}\")");
        milk_ipc_eval(hooks, script);
        free(script);
    }
    g_undo_pending.len    = 0;
    g_undo_pending.buf[0] = '\0';
    g_undo_npending       = 0;
}

/**
 * undo_commit - Queue the undo entry @u from a single `@set` call and flush if running out-of-process
 * @u:     Undo entry collected during `milk_client_set_argv()`
 * @hooks: Shell variable/eval hooks
 */
static void undo_commit(undo_text_t *u, const milk_client_hooks_t *hooks)
{
    if (u->len == 0)
    {
        return;
    }
    u->buf[--u->len] = '\0'; /* the last newline */
    undo_add(&g_undo_pending, u->buf, u->len);
    undo_add(&g_undo_pending, "\036", 1);
    if (++g_undo_npending > 2000) /* never flushed: keep the last 1000 */
    {
        char *c = g_undo_pending.buf;
        for (int k = 0; k < 1000 && c != NULL; k++)
        {
            c = strchr(c, '\036');
            c = (c != NULL) ? c + 1 : NULL;
        }
        if (c != NULL)
        {
            g_undo_pending.len -= (size_t) (c - g_undo_pending.buf);
            memmove(g_undo_pending.buf, c, g_undo_pending.len + 1);
            g_undo_npending -= 1000;
        }
    }
    if (hooks == NULL || hooks->eval == NULL)
    {
        undo_flush(hooks);
    }
}

/**
 * milk_client_set_argv - Parse and execute `@set` command-line arguments (`--tui`, `--restore`, assignments)
 * @argc:  Argument count
 * @argv:  Argument vector
 * @hooks: Optional shell variable/eval hooks
 *
 * Return: 0 on success, non-zero on error.
 */
int milk_client_set_argv(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    milk_kw_args_t ka;
    if (milk_kw_parse("set", MILK_KW_OPTS_FIRST, &argc, argv, &ka))
    {
        return ka.rc;
    }
    if (milk_kw_opt(&ka, "--tui") != NULL)
    {
        return milk_client_call_as("fparam", argc, argv, "--tui", NULL, 0, hooks);
    }
    if (milk_kw_opt(&ka, "--flush-undo") != NULL) /* for @restore --undo */
    {
        undo_flush(hooks);
        return 0;
    }
    if (milk_kw_opt(&ka, "--restore") !=
        NULL) /* put the old values back when the caller's scope ends */
    {
        int rr = set_restore(argc, argv, hooks);
        if (rr != 0 || argc == 0)
        {
            return rr;
        }
    }
    else if (argc < 1)
    {
        return milk_lexicon_usage("set", stderr);
    }

    int         rc   = 0;
    int         i    = 0;
    undo_text_t undo = { NULL, 0, 0 };
    const char *nl = milk_ipc_get_var(hooks, "__MILK_NOLOG"); /* the writes of an undo or restore */
    int         nolog = (nl != NULL && nl[0] != '\0');
    while (i < argc)
    {
        milk_set_item_t it;
        milk_set_item(argc, argv, &i, &it);

        /* a value (not an action) that @restore --undo can set back: its values before and after */
        char        name[512];
        char        before[8192];
        const char *eq  = strchr(it.spec, '=');
        int         log = (eq != NULL || it.val != NULL) && !nolog;
        snprintf(name, sizeof(name), "%.*s",
                 (int) ((eq != NULL) ? (size_t) (eq - it.spec) : strlen(it.spec)), it.spec);
        log = log && undo_loggable(name) && undo_values(name, before, sizeof(before), hooks) == 0;
        if (milk_client_set(it.spec, it.val, it.comment, hooks) != 0)
        {
            rc = 1;
        }
        else if (log)
        {
            char after[8192];
            if (undo_values(name, after, sizeof(after), hooks) == 0)
            {
                undo_pair(&undo, before, after);
            }
        }
    }
    undo_commit(&undo, hooks);
    free(undo.buf);
    return rc;
}

/**
 * names_ok - Validate a comma-separated list of shell variable identifiers (`@calc -v a,b`)
 * @names: Comma-separated identifier list
 *
 * Return: 1 if every comma-separated token is a valid shell identifier, 0 otherwise.
 */
static int names_ok(const char *names)
{
    char nm[128];
    for (const char *p = names;;)
    {
        size_t l = strcspn(p, ",");
        if (l == 0 || l >= sizeof(nm))
        {
            return 0;
        }
        snprintf(nm, sizeof(nm), "%.*s", (int) l, p);
        if (!milk_ipc_is_identifier(nm))
        {
            return 0;
        }
        if (p[l] == '\0')
        {
            return 1;
        }
        p += l + 1;
    }
}

/**
 * bind_values - Bind comma-separated shell variables @names to the trailing @n statement results @v
 * @hooks: Shell variable hooks
 * @names: Comma-separated variable names (`"a,b,..."`)
 * @v:     Array of @n formatted result strings
 * @n:     Number of statement values in @v
 *
 * Right-aligns the `k` variable names against the `n` statement outputs so
 * `@calc -v x "tmp = 2; tmp * 3"` binds `x` to the final statement's value.
 */
static void bind_values(const milk_client_hooks_t *hooks, const char *names, char **v, int n)
{
    int k = 1;
    for (const char *c = names; *c != '\0'; c++)
    {
        k += (*c == ',');
    }
    char nm[128];
    for (const char *p = names;; k--)
    {
        size_t l = strcspn(p, ",");
        snprintf(nm, sizeof(nm), "%.*s", (int) l, p);
        milk_ipc_set_var(hooks, nm, (n - k >= 0) ? v[n - k] : "", 0);
        if (p[l] == '\0')
        {
            return;
        }
        p += l + 1;
    }
}

/**
 * calc_local_list - Attempt to evaluate `;`-separated scalar statements of @expr in-process
 * @expr:   Calculator expression string
 * @digits: Decimal precision (`1..17`)
 * @vals:   Output array of formatted scalar result strings
 * @maxn:   Maximum number of statements supported in @vals
 * @truth:  Output boolean truth value of the last statement
 *
 * Evaluates pure scalar arithmetic expressions and `/dev/shm` property reads
 * directly in the client process via `mcalc_local()`, avoiding a socket
 * round-trip to the server.
 *
 * Return: Number of evaluated statements (`> 0`), or -1 if server evaluation is required.
 */
static int calc_local_list(const char *expr, int digits, char vals[][256], int maxn, int *truth)
{
    char stmt[1024];
    int  n = 0;

    if (strpbrk(expr, "\"'") != NULL)
    {
        return -1;
    }
    for (const char *p = expr;;)
    {
        size_t l = strcspn(p, ";");
        if (l >= sizeof(stmt))
        {
            return -1;
        }
        memcpy(stmt, p, l);
        stmt[l] = '\0';
        if (stmt[strspn(stmt, " \t")] != '\0' &&
            (n == maxn || !mcalc_local(stmt, digits, vals[n++], 256, truth)))
        {
            return -1;
        }
        if (p[l] == '\0')
        {
            return n;
        }
        p += l + 1;
    }
}

/**
 * client_calc_or_test - Shared implementation of `@calc` and `@test`
 * @expr:    Calculator or boolean expression string
 * @is_test: Non-zero for `@test` (exit status 0/1, no stdout output)
 * @xflags:  `MILK_FLAG_REPL` (interactive calc mode), `MILK_FLAG_EXPLAIN` (`--explain`), or 0
 * @names:   Optional comma-separated variable names for `-v` binding
 * @arr:     Optional array variable name for `-a` pixel binding
 * @hooks:   Optional shell variable hooks
 *
 * Tries the zero-RPC `calc_local_list()` scalar evaluator first and falls back
 * to `MILK_MSG_CALC` / `MILK_MSG_TEST` on the server when images or stateful
 * calculator variables are involved.
 *
 * Return: 0 on success (or true condition), 1 on false condition, 2 on evaluation error.
 */
static int client_calc_or_test(const char                *expr,
                               int                        is_test,
                               uint8_t                    xflags,
                               const char                *names,
                               const char                *arr,
                               const milk_client_hooks_t *hooks)
{
    if (expr == NULL || expr[0] == '\0')
    {
        return is_test ? 1 : 0;
    }
    if ((names != NULL && !names_ok(names)) || (arr != NULL && !milk_ipc_is_identifier(arr)))
    {
        fprintf(stderr, "milk: @calc: '%s' is no variable name\n", (arr != NULL) ? arr : names);
        return 2;
    }

    milk_ipc_sync_shm_env(hooks); /* local reads of s.cam.cnt0 use the shell's MILK_SHM_DIR */
    const char *dp = milk_ipc_get_var(hooks, "MILK_DPDIGITS"); /* sent back by @config digits */
    long        nd = (dp != NULL) ? strtol(dp, NULL, 10) : 0;
    char        lv[16][256];
    char       *lp[16];
    int         truth = 0;
    int n = (arr == NULL && xflags == 0)
                ? calc_local_list(expr, (nd >= 1 && nd <= 17) ? (int) nd : 15, lv, 16, &truth)
                : -1;
    if (n > 0 && (!is_test || n == 1)) /* nothing to look up on the server: evaluated here */
    {
        if (is_test)
        {
            return truth ? 0 : 1;
        }
        for (int i = 0; i < n; i++)
        {
            lp[i] = lv[i];
            if (names == NULL)
            {
                printf("%s\n", lv[i]);
            }
        }
        if (names != NULL)
        {
            bind_values(hooks, names, lp, n);
        }
        fflush(stdout);
        return 0;
    }

    char sockpath[256];
    int  fd = milk_client_connect_or_start(hooks, sockpath, sizeof(sockpath));
    if (fd < 0)
    {
        return MILK_CLIENT_RC_TRANSPORT;
    }

    milk_packed_vars_t pv;
    pv.count = 0;
    milk_client_pack_digits(&pv, hooks);
    milk_client_collect_identifiers(&pv, hooks, expr);

    size_t elen = strlen(expr) + 1;
    size_t cap  = elen + sizeof(uint32_t);
    for (int i = 0; i < pv.count; i++)
    {
        cap += strlen(pv.names[i]) + 1 + strlen(pv.vals[i]) + 1;
    }

    uint8_t *buf = (uint8_t *) malloc(cap);
    if (buf == NULL)
    {
        milk_client_end(fd);
        return 2;
    }

    uint32_t off = 0;
    memcpy(buf + off, expr, elen);
    off += (uint32_t) elen;

    uint32_t nvars = (uint32_t) pv.count;
    memcpy(buf + off, &nvars, sizeof(nvars));
    off += (uint32_t) sizeof(nvars);
    for (int i = 0; i < pv.count; i++)
    {
        size_t nl = strlen(pv.names[i]) + 1;
        size_t vl = strlen(pv.vals[i]) + 1;
        memcpy(buf + off, pv.names[i], nl);
        off += (uint32_t) nl;
        memcpy(buf + off, pv.vals[i], vl);
        off += (uint32_t) vl;
    }

    /* the values to bind go to a memory file: a pipe read after DONE would block a large -a output */
    int bind      = !is_test && (names != NULL || arr != NULL);
    int out_fd    = bind ? memfd_create("milk_calc_out", MFD_CLOEXEC) : STDOUT_FILENO;
    int cwd_fd    = open(".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    int fds[4]    = { STDIN_FILENO, (out_fd >= 0) ? out_fd : STDOUT_FILENO, STDERR_FILENO, cwd_fd };
    uint8_t mtype = is_test ? MILK_MSG_TEST : MILK_MSG_CALC;
    uint8_t flags = !bind ? xflags : (arr != NULL) ? MILK_FLAG_PIXELS : MILK_FLAG_BIND;
    int     sret = milk_proto_send_with_fds(fd, mtype, flags, buf, off, fds, (cwd_fd >= 0) ? 4 : 3);
    if (cwd_fd >= 0)
    {
        close(cwd_fd);
    }
    free(buf);

    int rc = (sret != 0) ? 2 : milk_client_await_response(fd, sockpath, hooks, NULL, 0, NULL);
    milk_client_end(fd);

    if (bind && out_fd >= 0)
    {
        off_t sz   = lseek(out_fd, 0, SEEK_END);
        char *text = (rc == 0 && sz >= 0) ? (char *) malloc((size_t) sz + 1) : NULL;
        if (text != NULL && pread(out_fd, text, (size_t) sz, 0) == (ssize_t) sz)
        {
            char **lines = (char **) malloc(((size_t) sz + 1) * sizeof(char *));
            int    nl    = 0;
            text[sz]     = '\0';
            for (char *l = text; lines != NULL && *l != '\0';) /* one line per statement */
            {
                char *e     = strchr(l, '\n');
                lines[nl++] = l;
                if (e == NULL)
                {
                    break;
                }
                *e = '\0';
                l  = e + 1;
            }
            if (lines != NULL && names != NULL)
            {
                bind_values(hooks, names, lines, nl);
            }
            else if (lines != NULL && nl > 0)
            {
                client_bind_array(hooks, arr, 0, lines[nl - 1]);
            }
            free(lines);
        }
        free(text);
        close(out_fd);
    }
    return rc;
}

/** calc_libs() - append the `--lib <file>` values of @a to @libs (@n held, 16 at most) */
static int calc_libs(const milk_kw_args_t *a, const char **libs, int n)
{
    for (int j = 0; j < a->nocc; j++)
    {
        if (strcmp(a->k->opts[a->occ[j].opt].names, "--lib") == 0 && n < 16)
        {
            libs[n++] = a->occ[j].val;
        }
    }
    return n;
}

/**
 * milk_client_calc_argv - Parse and execute `@calc` command-line arguments (`--repl`, `-v`, `-a`, `<expr>...`)
 * @argc:  Argument count
 * @argv:  Argument vector
 * @hooks: Optional shell variable hooks
 *
 * Return: 0 on success, non-zero on error.
 */
int milk_client_calc_argv(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    milk_kw_args_t ka;
    milk_kw_args_t kl; /* --list [<pat>] [--json] [--lib <file>]: in any order */
    const char    *libs[16];
    const char    *pat = NULL;
    if (milk_kw_parse("calc", MILK_KW_OPTS_FIRST, &argc, argv, &ka))
    {
        return ka.rc;
    }
    const char *names     = milk_kw_opt(&ka, "-v");
    const char *arr       = milk_kw_opt(&ka, "-a");
    int         repl      = (milk_kw_opt(&ka, "--repl") != NULL);
    int         explain   = (milk_kw_opt(&ka, "--explain") != NULL);
    int         list_mode = (milk_kw_opt(&ka, "--list") != NULL);
    int         json_mode = (milk_kw_opt(&ka, "--json") != NULL);
    int         nlibs     = calc_libs(&ka, libs, 0);
    int         i         = 0;
    if (list_mode)
    {
        if (milk_kw_parse("calc", 0, &argc, argv, &kl))
        {
            return kl.rc;
        }
        json_mode |= (milk_kw_opt(&kl, "--json") != NULL);
        nlibs = calc_libs(&kl, libs, nlibs);
        pat   = (argc > 0 && argv[0][0] != '-') ? argv[0] : NULL;
        char sockpath[256];
        milk_ipc_resolve_sockpath(hooks, sockpath, sizeof(sockpath), 1);
        int srv_fd = (nlibs == 0) ? milk_ipc_connect_once(sockpath) : -1;
        if (srv_fd >= 0)
        {
            close(srv_fd);
        }
        const char *env_libs = getenv("MILK_CALC_LIBS");
        if (nlibs == 0 && srv_fd < 0 && (env_libs == NULL || env_libs[0] == '\0'))
        {
            return mcalc_print_catalog(pat, json_mode);
        }
    }
    size_t tot = 64 + (size_t) nlibs * (PATH_MAX + 16);
    for (int j = i; j < argc; j++)
    {
        tot += strlen(argv[j]) + 1;
    }
    char *expr = (char *) malloc(tot);
    if (expr == NULL)
    {
        return 2;
    }
    expr[0] = '\0';
    for (int k = 0; k < nlibs; k++)
    {
        char        rpath[PATH_MAX];
        const char *use = (realpath(libs[k], rpath) != NULL) ? rpath : libs[k];
        size_t      cur = strlen(expr);
        snprintf(expr + cur, tot - cur, "%sloadlib \"%s\"", (cur > 0) ? "; " : "", use);
    }
    if (list_mode)
    {
        size_t cur = strlen(expr);
        snprintf(expr + cur, tot - cur, "%s__list_catalog%s%s%s", (cur > 0) ? "; " : "",
                 json_mode ? " --json" : "", (pat != NULL) ? " " : "", (pat != NULL) ? pat : "");
    }
    else
    {
        if (explain && nlibs > 0) /* load them first: --explain runs nothing */
        {
            int lrc = client_calc_or_test(expr, 0, 0, NULL, NULL, hooks);
            if (lrc != 0)
            {
                free(expr);
                return lrc;
            }
            expr[0] = '\0';
        }
        for (int j = i; j < argc; j++)
        {
            strcat(expr, (j == i && expr[0] != '\0') ? "; " : (j > i) ? " " : "");
            strcat(expr, argv[j]);
        }
    }
    int rc = client_calc_or_test(expr, 0,
                                 explain ? MILK_FLAG_EXPLAIN
                                 : repl  ? MILK_FLAG_REPL
                                         : 0,
                                 names, arr, hooks);
    free(expr);
    return rc;
}

/**
 * milk_client_test - Evaluate boolean condition @expr (`@test`)
 * @expr:  Boolean condition expression
 * @hooks: Optional shell variable hooks
 *
 * Return: 0 if @expr evaluates to true, 1 if false, 2 on syntax/evaluation error.
 */
int milk_client_test(const char *expr, const milk_client_hooks_t *hooks)
{
    return client_calc_or_test(expr, 1, 0, NULL, NULL, hooks);
}

/**
 * milk_client_exists - Check whether entity @spec exists or is actively running (`@exists [-r]`)
 * @spec:          Entity specification (`s.im`, `f.fps`, `p.proc`, `l.im`)
 * @check_running: Non-zero for `@exists -r`, 0 for `@exists`
 * @hooks:         Optional shell variable hooks
 *
 * Checks `/dev/shm` directly first via `milk_shm_prop_exists()`, and queries a
 * connected server if @spec may name a server-local image (`l.*`).
 *
 * Return: 0 if the entity exists (or is running), 1 otherwise.
 */
int milk_client_exists(const char *spec, int check_running, const milk_client_hooks_t *hooks)
{
    if (spec == NULL || spec[0] == '\0')
    {
        return 1;
    }
    milk_ipc_sync_shm_env(hooks);

    if (milk_shm_prop_exists(spec, check_running))
    {
        return 0;
    }

    char s_norm[256];
    milk_normalize_entity_spec(spec, s_norm, sizeof(s_norm));
    milk_ns_kind_t ek = milk_entity_ns_kind(s_norm, NULL);
    if (!check_running &&
        (ek == MILK_NS_STREAM || ek == MILK_NS_LOCAL || strchr(s_norm, '.') == NULL))
    {
        char sockpath[256];
        milk_ipc_resolve_sockpath(hooks, sockpath, sizeof(sockpath), 1);
        int fd = milk_client_connect(sockpath);
        if (fd >= 0)
        {
            size_t   slen = strlen(s_norm) + 1;
            uint8_t *pl   = (uint8_t *) malloc(slen + 1);
            if (pl != NULL)
            {
                pl[0] = check_running ? 'R' : 'E';
                memcpy(pl + 1, s_norm, slen);
                if (milk_proto_send(fd, MILK_MSG_EXISTS, 0, pl, (uint32_t) (slen + 1)) == 0)
                {
                    int rc = milk_client_await_response(fd, sockpath, hooks, NULL, 0, NULL);
                    free(pl);
                    milk_client_end(fd);
                    return rc;
                }
                free(pl);
            }
            milk_client_end(fd);
        }
    }
    return 1;
}

/* ============================================================
 *  Commands by Name
 * ============================================================ */

/* call [--async] [--frontend] [--nowait] <cmd> [args...]; no command is not an error */
static int cmd_call(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    int flags = 0;
    int i     = 0;
    for (; i < argc; i++)
    {
        int f = (strcmp(argv[i], "--async") == 0)      ? MILK_CALL_ASYNC
                : (strcmp(argv[i], "--frontend") == 0) ? MILK_CALL_FRONTEND
                : (strcmp(argv[i], "--nowait") == 0)   ? MILK_CALL_NOWAIT
                                                       : 0;
        if (f == 0)
        {
            break;
        }
        flags |= f;
    }
    return (i < argc) ? milk_client_call(argc - i, argv + i, flags, hooks) : 0;
}

static int cmd_set(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    return (argc < 1) ? milk_lexicon_usage("set", stderr) : milk_client_set_argv(argc, argv, hooks);
}

/* milk-cli takes a bare @calc itself (calc mode) */
static int cmd_calc(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    return (argc < 1) ? milk_lexicon_usage("calc", stderr)
                      : milk_client_calc_argv(argc, argv, hooks);
}

/* test <words...>: the words, joined by spaces, are the condition */
static int cmd_test(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    if (argc < 1)
    {
        return milk_lexicon_usage("test", stderr);
    }
    size_t tot = 1;
    for (int i = 0; i < argc; i++)
    {
        tot += strlen(argv[i]) + 1;
    }
    char *expr = (char *) malloc(tot);
    if (expr == NULL)
    {
        return 2;
    }
    size_t pos = 0;
    expr[0]    = '\0';
    for (int i = 0; i < argc; i++)
    {
        pos += (size_t) snprintf(expr + pos, tot - pos, "%s%s", (i > 0) ? " " : "", argv[i]);
    }
    int rc = milk_client_test(expr, hooks);
    free(expr);
    return rc;
}

/* exists [-r] <entity>: one entity */
static int cmd_exists(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    milk_kw_args_t ka;
    if (milk_kw_parse("exists", 0, &argc, argv, &ka))
    {
        return ka.rc;
    }
    return (argc != 1) ? milk_lexicon_usage("exists", stderr)
                       : milk_client_exists(argv[0], milk_kw_opt(&ka, "-r") != NULL, hooks);
}

/* complete <prefix> [cmd|arg] */
static int cmd_complete(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    return milk_client_complete((argc >= 1) ? argv[0] : "", (argc >= 2) ? argv[1] : "arg", hooks);
}

static int cmd_shutdown(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    (void) argc;
    (void) argv;
    return milk_client_shutdown(hooks);
}

/**
 * milk_client_cmd - Look up a client command by name
 * @name: `get`, `calc`, `waitfor`, ...: the bash builtin `milk<name>` and `milk-call <name>`
 *
 * Return: The command, or NULL when the client has none of that name
 */
const milk_client_cmd_t *milk_client_cmd(const char *name)
{
    static const milk_client_cmd_t cmds[] = {
        { "call", cmd_call },
        { "get", milk_client_get_argv },
        { "set", cmd_set },
        { "calc", cmd_calc },
        { "test", cmd_test },
        { "exists", cmd_exists },
        { "waitfor", milk_client_waitfor },
        { "waitany", milk_client_waitany },
        { "sleep", milk_client_sleep_argv },
        { "grab", milk_client_grab },
        { "complete", cmd_complete },
        { "session", milk_client_session },
        { "log", milk_client_log },
        { "ls", milk_client_ls },
        { "shutdown", cmd_shutdown },
        { "unload", cmd_shutdown },
    };
    for (size_t i = 0; name != NULL && i < sizeof(cmds) / sizeof(cmds[0]); i++)
    {
        if (strcmp(name, cmds[i].name) == 0)
        {
            return &cmds[i];
        }
    }
    return NULL;
}
