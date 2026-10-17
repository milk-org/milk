// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_client_ops.c
 * @brief Public client operations (`milk_client_*`) for bash integration.
 *
 * Role & Goal:
 *   Implements all public client operations declared in `milk_client.h`:
 *     - `milk_client_call()`     : Dispatch CLI commands with SCM_RIGHTS fds
 *     - `milk_client_get()`      : Fast `/dev/shm` read with server fallback
 *     - `milk_client_set()`      : Fast `/dev/shm` write with server fallback
 *     - `milk_client_calc()`     : Evaluate math/array expressions via server
 *     - `milk_client_test()`     : Boolean condition evaluation via server
 *     - `milk_client_exists()`   : Entity existence / running checks
 *     - `milk_client_waitfor()`  : Stream semaphore / condition wait
 *     - `milk_client_grab()`     : N-frame stream capture into 3D cube
 *     - `milk_client_complete()` : Interactive tab completion query
 *     - `milk_client_shutdown()` : Clean server shutdown
 *     - `milk_client_server()`   : Multi-server manager (`@server`)
 *     - `milk_client_log()`      : Command/output ring-buffer log (`@log`)
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <ctype.h>
#include <dirent.h>
#include <fcntl.h>
#include <math.h>
#include <semaphore.h>
#include <sys/eventfd.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "CLIcore_script.h"
#include "cli_calc_scalar.h"
#include "milk_client.h"
#include "milk_lexicon.h"
#include "milk_renames.h"
#include "milk_client_internal.h"
#include "milk_entity_ns.h"
#include "milk_shm_prop.h"
#include "milk_ipc_common.h"
#include "milk_proto.h"
#include "milk_srvlog.h"
#include "milk_waitfor.h"

/**
 * @brief Drain client stdin into an anonymous memfd for `@replay --stdin` pipelines.
 *
 * Reading stdin before connecting to the single-threaded server prevents deadlocks
 * in pipelines such as `@grab --stdout im1 | ... | @replay --stdin im2`, while supporting
 * both text and binary (`--raw`) streams of any size.
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
    // streamwrite (@replay --stdin) reads stdin only when told to with a '-' argument
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
 * @brief Dispatch a milk command to `milk-cli --server` with the caller's fds.
 */
static int client_call_sync(int argc, char **argv, const milk_client_hooks_t *hooks);
static int client_capture_array(int (*fn)(int, char **, const milk_client_hooks_t *),
                                int                        argc,
                                char                     **argv,
                                const char                *extra,
                                const milk_client_hooks_t *hooks);

/**
 * call_as() - run `@cmd args` without the option @opt, as engine command @as (`@grab --stdout` is
 * streamcat, `@replay --stdin` streamwrite reading '-', `@set --tui` the fparam editor).
 */
static int call_as(const char                *as,
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

static int has_arg(int argc, char **argv, const char *opt)
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

int milk_client_call(int argc, char **argv, int async_flag, const milk_client_hooks_t *hooks)
{
    if (argc <= 0 || argv == NULL || argv[0] == NULL)
    {
        return 0;
    }

    const char *c0 = argv[0];
    if (c0[0] == '@')
    {
        c0++;
    }
    if (strcmp(c0, "replay") == 0 && has_arg(argc - 1, argv + 1, "--stdin"))
    {
        return call_as("streamwrite", argc - 1, argv + 1, "--stdin", "-", async_flag, hooks);
    }
    int arc;
    if (!async_flag && strcmp(c0, "streamcat") == 0 &&
        (arc = client_capture_array(client_call_sync, argc, argv, NULL, hooks)) >= 0)
    {
        return arc;
    }

    int sw_stdin_fd = drain_stdin_to_memfd(argc, argv);

    char sockpath[256];
    int  fd = milk_client_connect_or_start(hooks, sockpath, sizeof(sockpath));
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
        close(fd);
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

    int cwd_fd    = open(".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    int fds[4]    = { (sw_stdin_fd >= 0) ? sw_stdin_fd : STDIN_FILENO, STDOUT_FILENO, STDERR_FILENO,
                      cwd_fd };
    uint8_t flags = async_flag ? MILK_FLAG_ASYNC : 0;
    int     sret =
        milk_proto_send_with_fds(fd, MILK_MSG_CALL, flags, buf, off, fds, (cwd_fd >= 0) ? 4 : 3);
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
        close(fd);
        return 1;
    }

    int rc = milk_client_await_response(fd, sockpath, hooks, NULL, 0, NULL);
    close(fd);
    return rc;
}

/**
 * @brief Read a live entity property (`@get`) into @out, using `/dev/shm` fast path first.
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

    char sockpath[256];
    int  fd = milk_client_connect_or_start(hooks, sockpath, sizeof(sockpath));
    if (fd < 0)
    {
        return MILK_CLIENT_RC_TRANSPORT;
    }

    uint32_t slen = (uint32_t) (strlen(spec) + 1);
    if (milk_proto_send(fd, MILK_MSG_GET, 0, spec, slen) != 0)
    {
        close(fd);
        return 1;
    }

    int got_val = 0;
    int rc      = milk_client_await_response(fd, sockpath, hooks, out, outsz, &got_val);
    close(fd);
    return (rc == 0 && got_val) ? 0 : 1;
}

/**
 * @brief Read a live entity property (`@get`), using `/dev/shm` fast path first.
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
 * @brief Write a live entity property (`@set`), using `/dev/shm` fast path first.
 */
int milk_client_set(const char                *spec,
                    const char                *val,
                    const char                *comment,
                    const milk_client_hooks_t *hooks)
{
    if (spec == NULL || spec[0] == '\0')
    {
        fprintf(stderr, "Usage: @set <entity> [=] [value] [comment]\n");
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

    struct timespec ts0;
    clock_gettime(CLOCK_REALTIME, &ts0);

    if (!is_local_or_glob && milk_shm_prop_set(spec, val, comment) == 0)
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

    char sockpath[256];
    int  fd = milk_client_connect_or_start(hooks, sockpath, sizeof(sockpath));
    if (fd < 0)
    {
        return MILK_CLIENT_RC_TRANSPORT;
    }

    size_t l1 = strlen(spec) + 1;
    size_t l3 = (comment != NULL) ? (strlen(comment) + 1) : 0;
    size_t l2 =
        (val != NULL) ? (strlen(val) + 1) : (l3 > 0 ? 1 : 0); // empty slot keeps comment in place
    uint32_t tot = (uint32_t) (l1 + l2 + l3);

    uint8_t *buf = (uint8_t *) malloc(tot);
    if (buf == NULL)
    {
        close(fd);
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
        close(fd);
        return 1;
    }

    int rc = milk_client_await_response(fd, sockpath, hooks, NULL, 0, NULL);
    close(fd);
    return rc;
}

/** assoc_spec() - the `.kv` listing behind an `@get -A` spec (`s.cam`, `s.cam.kw`, `f.x.*`, `f.x.list`, ...). */
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

/** client_bind_array() - bind @text to shell array @name (indexed, or associative when @assoc). */
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

int milk_client_get_argv(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    const char *varname   = NULL;
    const char *arrname   = NULL;
    const char *assocname = NULL;
    const char *spec      = NULL;
    int         json_mode = 0;
    for (int i = 0; i < argc; i++)
    {
        if (strcmp(argv[i], "--json") == 0 || strcmp(argv[i], "-j") == 0)
        {
            json_mode = 1;
        }
        else if (strcmp(argv[i], "-v") == 0 && i + 1 < argc)
        {
            varname = argv[++i];
        }
        else if (strcmp(argv[i], "-a") == 0 && i + 1 < argc)
        {
            arrname = argv[++i];
        }
        else if (strcmp(argv[i], "-A") == 0 && i + 1 < argc)
        {
            assocname = argv[++i];
        }
        else if (spec == NULL)
        {
            spec = argv[i];
        }
    }
    if (spec == NULL)
    {
        fprintf(stderr, "Usage: @get [--json|-j] [-v var | -a arr | -A assoc] <entity>\n");
        return 1;
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

static int client_call_sync(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    return milk_client_call(argc, argv, 0, hooks);
}

/**
 * client_capture_array() - `-a <arr>` of @grab --stdout and @ls: run @fn on @argv without the option (plus @extra,
 * may be NULL) with stdout captured, then bind the output words to array @arrname. -1 if @argv has no `-a`.
 */
static int client_capture_array(int (*fn)(int, char **, const milk_client_hooks_t *),
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

static int is_entity_spec_token(const char *s)
{
    if (s == NULL)
    {
        return 0;
    }
    char norm[256];
    milk_normalize_entity_spec(s, norm, sizeof(norm));
    const char *rest = norm;
    return milk_entity_ns_kind(norm, &rest) != MILK_NS_NONE && strchr(rest, '.') != NULL;
}

/**
 * @brief Parse and execute one or more `@set` assignments from `argv`.
 */
int milk_client_set_argv(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    if (argc < 1 || argv == NULL || argv[0] == NULL)
    {
        fprintf(stderr, "Usage: @set <entity> [=] [value] [comment] [...]\n");
        return 1;
    }
    if (strcmp(argv[0], "--tui") == 0)
    {
        return call_as("fparam", argc, argv, "--tui", NULL, 0, hooks);
    }

    int rc = 0;
    int i  = 0;
    while (i < argc)
    {
        const char *spec    = argv[i++];
        const char *val     = NULL;
        const char *comment = NULL;

        if (strchr(spec, '=') != NULL)
        {
            if (i < argc && !is_entity_spec_token(argv[i]))
            {
                comment = argv[i++];
            }
        }
        else
        {
            if (i < argc && strcmp(argv[i], "=") == 0)
            {
                i++;
                if (i < argc)
                {
                    val = argv[i++];
                }
            }
            else if (i < argc && !is_entity_spec_token(argv[i]))
            {
                val = argv[i++];
            }
            if (i < argc && !is_entity_spec_token(argv[i]))
            {
                comment = argv[i++];
            }
        }

        if (milk_client_set(spec, val, comment, hooks) != 0)
        {
            rc = 1;
        }
    }
    return rc;
}

/**
 * @brief Shared implementation for `@calc` and `@test` expression evaluation.
 */
static int client_calc_or_test(const char                *expr,
                               int                        is_test,
                               const char                *varname,
                               const milk_client_hooks_t *hooks)
{
    if (expr == NULL || expr[0] == '\0')
    {
        return is_test ? 1 : 0;
    }

    const char *dp = milk_ipc_get_var(hooks, "MILK_DPDIGITS"); /* sent back by @config digits */
    long        nd = (dp != NULL) ? strtol(dp, NULL, 10) : 0;
    char        res[256];
    int         truth = 0;
    if (cli_calc_local(expr, (nd >= 1 && nd <= 17) ? (int) nd : 15, res, sizeof(res), &truth))
    {
        if (is_test) /* nothing to look up on the server: evaluated here */
        {
            return truth ? 0 : 1;
        }
        if (varname != NULL && varname[0] != '\0')
        {
            milk_ipc_set_var(hooks, varname, res, 0);
        }
        else
        {
            printf("%s\n", res);
            fflush(stdout);
        }
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
        close(fd);
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

    int cap_pipe[2] = { -1, -1 };
    int out_fd      = STDOUT_FILENO;
    if (!is_test && varname != NULL && varname[0] != '\0')
    {
        if (pipe(cap_pipe) == 0)
        {
            out_fd = cap_pipe[1];
        }
    }

    int     cwd_fd = open(".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    int     fds[4] = { STDIN_FILENO, out_fd, STDERR_FILENO, cwd_fd };
    uint8_t mtype  = is_test ? MILK_MSG_TEST : MILK_MSG_CALC;
    int     sret   = milk_proto_send_with_fds(fd, mtype, 0, buf, off, fds, (cwd_fd >= 0) ? 4 : 3);
    if (cwd_fd >= 0)
    {
        close(cwd_fd);
    }
    if (cap_pipe[1] >= 0)
    {
        close(cap_pipe[1]);
    }
    free(buf);

    if (sret != 0)
    {
        if (cap_pipe[0] >= 0)
        {
            close(cap_pipe[0]);
        }
        close(fd);
        return 2;
    }

    int rc = milk_client_await_response(fd, sockpath, hooks, NULL, 0, NULL);
    close(fd);

    if (cap_pipe[0] >= 0)
    {
        size_t  cap = 512;
        size_t  cn  = 0;
        char   *cb  = (char *) malloc(cap);
        ssize_t nr;
        while (cb != NULL && (nr = read(cap_pipe[0], cb + cn, cap - cn - 1)) > 0) // whole value
        {
            cn += (size_t) nr;
            if (cn + 1 == cap)
            {
                char *nb = (char *) realloc(cb, cap * 2);
                if (nb == NULL)
                {
                    break;
                }
                cb = nb;
                cap *= 2;
            }
        }
        close(cap_pipe[0]);
        if (cb != NULL &&
            (cn > 0 || rc == 0)) /* nothing printed (an assignment): empty, like @{...} */
        {
            cb[cn] = '\0';
            while (cn > 0 && (cb[cn - 1] == '\n' || cb[cn - 1] == '\r'))
            {
                cb[--cn] = '\0';
            }
            milk_ipc_set_var(hooks, varname, cb, 0);
        }
        free(cb);
    }
    return rc;
}

int milk_client_calc(const char *expr, const char *varname, const milk_client_hooks_t *hooks)
{
    return client_calc_or_test(expr, 0, varname, hooks);
}

int milk_client_test(const char *expr, const milk_client_hooks_t *hooks)
{
    return client_calc_or_test(expr, 1, NULL, hooks);
}

/**
 * @brief Check whether a milk entity exists (or is actively running).
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
        int fd = milk_ipc_connect_once(sockpath);
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
                    close(fd);
                    return rc;
                }
                free(pl);
            }
            close(fd);
        }
    }
    return 1;
}

static volatile sig_atomic_t g_wait_sigint = 0;

static void client_wait_sigint_handler(int sig)
{
    (void) sig;
    g_wait_sigint = 1;
}

static int client_wait_cancelled(void *ctx)
{
    (void) ctx;
    return g_wait_sigint;
}

/** client_wait_local() - run a `@waitfor` stream/entity wait in this process; MILK_WAIT_ABSENT: ask the server. */
static int client_wait_local(const milk_wait_target_t  *wt,
                             const char                *arg2,
                             const char                *sem_str,
                             const char                *cnt_str,
                             double                     timeout_sec,
                             const milk_client_hooks_t *hooks)
{
    struct sigaction sa_int, old_int;
    memset(&sa_int, 0, sizeof(sa_int));
    sa_int.sa_handler = client_wait_sigint_handler;
    sigemptyset(&sa_int.sa_mask);
    g_wait_sigint = 0;
    sigaction(SIGINT, &sa_int, &old_int);

    milk_wait_ops_t ops = { .cancelled = client_wait_cancelled };
    double          t0  = milk_wait_now();
    int             rc;
    if (wt->kind == MILK_WAIT_EQ)
    {
        rc = milk_waitfor_eq(wt->lhs, wt->expect, t0, timeout_sec, &ops);
    }
    else
    {
        int      req_sem = (sem_str && sem_str[0]) ? (int) strtol(sem_str, NULL, 10) : -1;
        uint64_t abs_cnt = (cnt_str && cnt_str[0]) ? (uint64_t) strtoull(cnt_str, NULL, 10) : 0;
        rc = milk_waitfor_stream(wt->stream, strtol(arg2, NULL, 10), req_sem,
                                 (cnt_str && cnt_str[0]) ? &abs_cnt : NULL, 0, t0, timeout_sec,
                                 &ops);
    }
    sigaction(SIGINT, &old_int, NULL);
    if (rc == MILK_WAIT_CANCEL && hooks && hooks->on_sigint)
    {
        hooks->on_sigint(hooks->ctx);
    }
    return rc;
}

/**
 * @brief Wait for stream frames or a boolean condition (`@waitfor`).
 */
int milk_client_waitfor(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    milk_ipc_sync_shm_env(hooks);

    const char *target   = NULL;
    const char *arg2     = "1";
    int         has_arg2 = 0;
    const char *timeout  = "0";
    const char *sem_str  = NULL;
    const char *cnt_str  = NULL;

    for (int i = 0; i < argc; i++)
    {
        if ((strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--timeout") == 0) && i + 1 < argc)
        {
            timeout = argv[++i];
        }
        else if (strncmp(argv[i], "--timeout=", 10) == 0)
        {
            timeout = argv[i] + 10;
        }
        else if ((strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--sem") == 0) && i + 1 < argc)
        {
            sem_str = argv[++i];
        }
        else if (strncmp(argv[i], "--sem=", 6) == 0)
        {
            sem_str = argv[i] + 6;
        }
        else if (strcmp(argv[i], "--cnt") == 0 && i + 1 < argc)
        {
            cnt_str = argv[++i];
        }
        else if (strncmp(argv[i], "--cnt=", 6) == 0)
        {
            cnt_str = argv[i] + 6;
        }
        else if (target == NULL)
        {
            target = argv[i];
        }
        else
        {
            arg2     = argv[i];
            has_arg2 = 1;
        }
    }

    if (target == NULL || target[0] == '\0')
    {
        fprintf(stderr, "Usage: @waitfor [-t|--timeout <sec>] [-s <sem>] [--cnt <target>] "
                        "<stream|condition> [nframes]\n");
        return 1;
    }

    milk_wait_target_t wt;
    if (milk_waitfor_parse(target, has_arg2 ? arg2 : NULL, &wt) == MILK_WAIT_BAD)
    {
        fprintf(stderr, "@waitfor: invalid target '%s'\n", target);
        return 1;
    }

    /* No RPC for shm entities and existing streams; local images, variables and expressions need the server */
    int here = (wt.kind == MILK_WAIT_STREAM && !wt.local);
    if (wt.kind == MILK_WAIT_EQ)
    {
        char val[256];
        here = (milk_shm_prop_get(wt.lhs, val, sizeof(val)) == 0) ||
               (is_entity_spec_token(wt.lhs) && milk_entity_ns_kind(wt.lhs, NULL) != MILK_NS_LOCAL);
    }
    if (here)
    {
        int rc = client_wait_local(&wt, arg2, sem_str, cnt_str, strtod(timeout, NULL), hooks);
        if (rc != MILK_WAIT_ABSENT)
        {
            return rc;
        }
    }

    char arg2_buf[128];
    snprintf(arg2_buf, sizeof(arg2_buf), "%s", arg2);
    if (sem_str != NULL && sem_str[0] != '\0')
    {
        size_t cur = strlen(arg2_buf);
        snprintf(arg2_buf + cur, sizeof(arg2_buf) - cur, ";sem=%s", sem_str);
    }
    if (cnt_str != NULL && cnt_str[0] != '\0')
    {
        size_t cur = strlen(arg2_buf);
        snprintf(arg2_buf + cur, sizeof(arg2_buf) - cur, ";cnt=%s", cnt_str);
    }

    char sockpath[256];
    int  fd = milk_client_connect_or_start(hooks, sockpath, sizeof(sockpath));
    if (fd < 0)
    {
        return MILK_CLIENT_RC_TRANSPORT;
    }

    size_t   l1  = strlen(target) + 1;
    size_t   l2  = strlen(arg2_buf) + 1;
    size_t   l3  = strlen(timeout) + 1;
    uint32_t tot = (uint32_t) (l1 + l2 + l3);

    uint8_t *buf = (uint8_t *) malloc(tot);
    if (buf == NULL)
    {
        close(fd);
        return 1;
    }
    memcpy(buf, target, l1);
    memcpy(buf + l1, arg2_buf, l2);
    memcpy(buf + l1 + l2, timeout, l3);

    int sret = milk_proto_send(fd, MILK_MSG_WAITFOR, 0, buf, tot);
    free(buf);
    if (sret != 0)
    {
        close(fd);
        return 1;
    }

    int rc = milk_client_await_response(fd, sockpath, hooks, NULL, 0, NULL);
    close(fd);
    return rc;
}

/**
 * @brief Wait for any of the given streams to advance its `cnt0` counter (`@waitany`).
 *
 * Uses zero-fork `/dev/shm` inspection (with `ImageStreamIO_openIm`) so waiting
 * on multiple streams has sub-millisecond latency without spawning subshells.
 */
/* @waitany: one thread per stream blocks on that stream's semaphore and signals an eventfd. */
typedef struct
{
    IMAGE                 *im;
    uint64_t               init_cnt;
    int                    semindex;
    int                    efd;
    volatile sig_atomic_t *stop;
} sem_waiter_arg_t;

typedef struct
{
    int                   efd;
    volatile sig_atomic_t stop;
    int                   n;
    int                   active[64];
    int                   semindex[64];
    pthread_t             tid[64];
    sem_waiter_arg_t      arg[64];
} sem_waiters_t;

static void *sem_waiter_main(void *p)
{
    sem_waiter_arg_t *a = (sem_waiter_arg_t *) p;
    while (!*a->stop)
    {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += 1; /* only bounds how long a stop request can wait */
        sem_timedwait(a->im->semptr[a->semindex], &ts);
        if (!*a->stop && a->im->md->cnt0 != a->init_cnt)
        {
            uint64_t one = 1;
            ssize_t  wr  = write(a->efd, &one, sizeof(one));
            (void) wr;
            break;
        }
    }
    return NULL;
}

/** sem_waiters_start() - start a waiter for each open stream that has semaphores (sw->efd < 0: none). */
static void sem_waiters_start(sem_waiters_t  *sw,
                              IMAGE          *imgs,
                              const int      *opened,
                              const uint64_t *init_cnt,
                              int             n)
{
    memset(sw, 0, sizeof(*sw));
    sw->n   = n;
    sw->efd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    if (sw->efd < 0)
    {
        return;
    }
    sigset_t all, old;
    sigfillset(&all);
    pthread_sigmask(SIG_BLOCK, &all, &old); /* signals stay with bash's main thread */
    for (int i = 0; i < n; i++)
    {
        if (!opened[i] || imgs[i].md == NULL || imgs[i].md->sem <= 0)
        {
            continue;
        }
        int si = ImageStreamIO_getsemwaitindex(&imgs[i], -1);
        if (si < 0 || si >= imgs[i].md->sem)
        {
            continue;
        }
        while (sem_trywait(imgs[i].semptr[si]) == 0)
        {
        }
        sw->semindex[i] = si;
        sw->arg[i]      = (sem_waiter_arg_t) { &imgs[i], init_cnt[i], si, sw->efd, &sw->stop };
        sw->active[i]   = (pthread_create(&sw->tid[i], NULL, sem_waiter_main, &sw->arg[i]) == 0);
        if (!sw->active[i])
        {
            imgs[i].semReadPID[si] = 0;
        }
    }
    pthread_sigmask(SIG_SETMASK, &old, NULL);
}

/** sem_waiters_stop() - wake and join the waiters, release their semaphores. */
static void sem_waiters_stop(sem_waiters_t *sw, IMAGE *imgs)
{
    sw->stop = 1;
    for (int i = 0; i < sw->n; i++)
    {
        if (sw->active[i])
        {
            sem_post(imgs[i].semptr[sw->semindex[i]]); /* our own semaphore: nobody else reads it */
            pthread_join(sw->tid[i], NULL);
            imgs[i].semReadPID[sw->semindex[i]] = 0;
        }
    }
    if (sw->efd >= 0)
    {
        close(sw->efd);
    }
}

int milk_client_waitany(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    milk_ipc_sync_shm_env(hooks);

    const char *streams[64];
    int         nstreams    = 0;
    double      timeout_sec = 0.0;

    for (int i = 0; i < argc; i++)
    {
        if ((strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--timeout") == 0) && i + 1 < argc)
        {
            timeout_sec = strtod(argv[++i], NULL);
        }
        else if (strncmp(argv[i], "--timeout=", 10) == 0)
        {
            timeout_sec = strtod(argv[i] + 10, NULL);
        }
        else if (nstreams < 64)
        {
            const char *s = argv[i];
            while (*s == '@')
            {
                s++;
            }
            const char    *rest = s;
            milk_ns_kind_t k    = milk_entity_ns_kind(s, &rest);
            if (strncmp(s, "S:", 2) == 0 || strncmp(s, "L:", 2) == 0)
            {
                s += 2;
            }
            else if (k == MILK_NS_STREAM || k == MILK_NS_LOCAL)
            {
                s = rest;
            }
            if (s[0] != '\0')
            {
                streams[nstreams++] = s;
            }
        }
    }

    if (nstreams == 0)
    {
        fprintf(stderr, "Usage: @waitany <stream1> <stream2> ... [--timeout <sec>]\n");
        return 1;
    }

    IMAGE    imgs[64];
    int      opened[64];
    uint64_t init_cnt[64];
    int      has_init[64];
    memset(imgs, 0, sizeof(imgs));
    memset(opened, 0, sizeof(opened));
    memset(init_cnt, 0, sizeof(init_cnt));
    memset(has_init, 0, sizeof(has_init));

    for (int i = 0; i < nstreams; i++)
    {
        if (ImageStreamIO_openIm(&imgs[i], streams[i]) == IMAGESTREAMIO_SUCCESS &&
            imgs[i].md != NULL)
        {
            opened[i]   = 1;
            init_cnt[i] = imgs[i].md->cnt0;
            has_init[i] = 1;
        }
        else
        {
            char spec[280];
            char val[64];
            snprintf(spec, sizeof(spec), "s.%s.cnt0", streams[i]);
            if (milk_shm_prop_get(spec, val, sizeof(val)) == 0)
            {
                init_cnt[i] = (uint64_t) strtoull(val, NULL, 10);
                has_init[i] = 1;
            }
        }
    }

    struct sigaction sa_int, old_int;
    memset(&sa_int, 0, sizeof(sa_int));
    sa_int.sa_handler = client_wait_sigint_handler;
    sigemptyset(&sa_int.sa_mask);
    g_wait_sigint = 0;
    sigaction(SIGINT, &sa_int, &old_int);

    sem_waiters_t sw;
    sem_waiters_start(&sw, imgs, opened, init_cnt, nstreams);

    double t_start  = milk_wait_now();
    int    has_timo = (timeout_sec > 0.0);
    int    rc       = 124;
    int    iter     = 0;
    int    hit      = -1;

    while (hit < 0)
    {
        if (g_wait_sigint)
        {
            if (hooks && hooks->on_sigint)
            {
                hooks->on_sigint(hooks->ctx);
            }
            rc = 130;
            break;
        }

        int need_poll =
            0; /* a stream not yet created, without semaphores, or local (server-side) */
        for (int i = 0; i < nstreams && hit < 0; i++)
        {
            if (!opened[i] && ImageStreamIO_openIm(&imgs[i], streams[i]) == IMAGESTREAMIO_SUCCESS &&
                imgs[i].md != NULL)
            {
                opened[i] = 1;
                if (!has_init[i])
                {
                    hit = i; /* created while we waited */
                    break;
                }
            }
            if (opened[i] && imgs[i].md != NULL)
            {
                if (imgs[i].md->cnt0 != init_cnt[i])
                {
                    hit = i;
                }
                need_poll |= (sw.efd < 0 || !sw.active[i]);
                continue;
            }
            need_poll = 1;
            char spec[280];
            char val[64];
            snprintf(spec, sizeof(spec), "s.%s.cnt0", streams[i]);
            if (milk_shm_prop_get(spec, val, sizeof(val)) == 0)
            {
                uint64_t c = (uint64_t) strtoull(val, NULL, 10);
                if (!has_init[i] || c != init_cnt[i])
                {
                    hit = i;
                }
            }
        }
        if (hit >= 0)
        {
            rc = 0;
            break;
        }

        double left = has_timo ? timeout_sec - (milk_wait_now() - t_start) : 1.0;
        if (left <= 0.0)
        {
            rc = 124;
            break;
        }
        if (!need_poll)
        {
            /* every stream has a semaphore waiter: sleep until one fires (or a signal, or the timeout) */
            struct pollfd pfd = { sw.efd, POLLIN, 0 };
            int           tms = (left < 1.0) ? (int) (left * 1000.0) + 1 : 1000;
            if (poll(&pfd, 1, tms) > 0)
            {
                uint64_t v;
                ssize_t  rr = read(sw.efd, &v, sizeof(v));
                (void) rr;
            }
            continue;
        }
        long            ns  = (iter++ < 40) ? 50000L : 200000L; /* 50 us then 200 us */
        struct timespec slp = { 0, ns };
        nanosleep(&slp, NULL);
    }

    sem_waiters_stop(&sw, imgs);
    sigaction(SIGINT, &old_int, NULL);
    for (int j = 0; j < nstreams; j++)
    {
        if (opened[j])
        {
            ImageStreamIO_closeIm(&imgs[j]);
        }
    }
    if (hit >= 0)
    {
        milk_ipc_set_var(hooks, "MILK_TRIGGER", streams[hit], 0);
        setenv("MILK_TRIGGER", streams[hit], 1);
        printf("%s\n", streams[hit]);
        fflush(stdout);
    }
    return rc;
}

static void resolve_client_path(const char *in, char *out, size_t outsz)
{
    if (in == NULL || in[0] == '\0')
    {
        out[0] = '\0';
        return;
    }
    const char *p    = (in[0] == '!') ? (in + 1) : in;
    const char *bang = (in[0] == '!') ? "!" : "";
    if (p[0] == '/')
    {
        snprintf(out, outsz, "%s", in);
        return;
    }
    char cwd[2048];
    if (getcwd(cwd, sizeof(cwd)) != NULL)
    {
        snprintf(out, outsz, "%s%s/%s", bang, cwd, p);
    }
    else
    {
        snprintf(out, outsz, "%s", in);
    }
}

/**
 * @brief Capture 1 or N consecutive frames from a stream into an image/3D cube or FITS file (`@grab`).
 */
int milk_client_grab(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    if (has_arg(argc, argv, "--stdout"))
    {
        return call_as("streamcat", argc, argv, "--stdout", NULL, 0, hooks);
    }

    const char *sname    = NULL;
    const char *nframes  = NULL;
    const char *outname  = NULL;
    const char *timeout  = "0";
    const char *fits_raw = NULL;
    const char *ts_raw   = NULL;

    for (int i = 0; i < argc; i++)
    {
        if ((strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--timeout") == 0) && i + 1 < argc)
        {
            timeout = argv[++i];
        }
        else if (strncmp(argv[i], "--timeout=", 10) == 0)
        {
            timeout = argv[i] + 10;
        }
        else if ((strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--fits") == 0) && i + 1 < argc)
        {
            fits_raw = argv[++i];
        }
        else if (strncmp(argv[i], "--fits=", 7) == 0)
        {
            fits_raw = argv[i] + 7;
        }
        else if ((strcmp(argv[i], "--timestamps") == 0 || strcmp(argv[i], "--ts") == 0) &&
                 i + 1 < argc)
        {
            ts_raw = argv[++i];
        }
        else if (strncmp(argv[i], "--timestamps=", 13) == 0)
        {
            ts_raw = argv[i] + 13;
        }
        else if (sname == NULL)
        {
            sname = argv[i];
        }
        else if (nframes == NULL)
        {
            nframes = argv[i];
        }
        else if (outname == NULL)
        {
            outname = argv[i];
        }
    }

    if (outname == NULL && sname != NULL)
    {
        if (fits_raw != NULL)
        {
            if (nframes == NULL)
            {
                nframes = "1";
                outname = "";
            }
            else
            {
                int all_digits = (nframes[0] != '\0');
                for (const char *p = nframes; *p != '\0'; p++)
                {
                    if (!isdigit((unsigned char) *p))
                    {
                        all_digits = 0;
                        break;
                    }
                }
                if (all_digits)
                {
                    outname = "";
                }
                else
                {
                    outname = nframes;
                    nframes = "1";
                }
            }
        }
        else if (nframes != NULL)
        {
            /* Support 2-arg form: @grab <src_stream> <dst_img> [-t <sec>] */
            outname = nframes;
            nframes = "1";
        }
    }

    if (sname == NULL || nframes == NULL || outname == NULL ||
        (outname[0] == '\0' && fits_raw == NULL))
    {
        fprintf(stderr,
                "Usage: @grab [-t|--timeout <sec>] [--fits <out.fits>] [--timestamps <out.txt>] "
                "<stream> [nframes] [out_img]\n");
        return 1;
    }

    char fits_abs[4096];
    char ts_abs[4096];
    resolve_client_path(fits_raw, fits_abs, sizeof(fits_abs));
    resolve_client_path(ts_raw, ts_abs, sizeof(ts_abs));

    char sockpath[256];
    int  fd = milk_client_connect_or_start(hooks, sockpath, sizeof(sockpath));
    if (fd < 0)
    {
        return MILK_CLIENT_RC_TRANSPORT;
    }

    size_t   l1  = strlen(sname) + 1;
    size_t   l2  = strlen(nframes) + 1;
    size_t   l3  = strlen(outname) + 1;
    size_t   l4  = strlen(timeout) + 1;
    size_t   l5  = strlen(fits_abs) + 1;
    size_t   l6  = strlen(ts_abs) + 1;
    uint32_t tot = (uint32_t) (l1 + l2 + l3 + l4 + l5 + l6);

    uint8_t *buf = (uint8_t *) malloc(tot);
    if (buf == NULL)
    {
        close(fd);
        return 1;
    }
    memcpy(buf, sname, l1);
    memcpy(buf + l1, nframes, l2);
    memcpy(buf + l1 + l2, outname, l3);
    memcpy(buf + l1 + l2 + l3, timeout, l4);
    memcpy(buf + l1 + l2 + l3 + l4, fits_abs, l5);
    memcpy(buf + l1 + l2 + l3 + l4 + l5, ts_abs, l6);

    int sret = milk_proto_send(fd, MILK_MSG_GRAB, 0, buf, tot);
    free(buf);
    if (sret != 0)
    {
        close(fd);
        return 1;
    }

    int rc = milk_client_await_response(fd, sockpath, hooks, NULL, 0, NULL);
    close(fd);
    return rc;
}

static void complete_local_files(const char *prefix, int fits_only)
{
    char        dirpart[512];
    const char *base     = prefix;
    const char *last_sep = strrchr(prefix, '/');
    if (last_sep != NULL)
    {
        size_t dlen = (size_t) (last_sep - prefix + 1);
        if (dlen >= sizeof(dirpart))
        {
            dlen = sizeof(dirpart) - 1;
        }
        memcpy(dirpart, prefix, dlen);
        dirpart[dlen] = '\0';
        base          = last_sep + 1;
    }
    else
    {
        dirpart[0] = '\0';
    }

    size_t blen    = strlen(base);
    DIR   *dp      = opendir(dirpart[0] ? dirpart : ".");
    int    emitted = 0;
    if (dp != NULL)
    {
        struct dirent *de;
        while ((de = readdir(dp)) != NULL)
        {
            if (de->d_name[0] == '.' && base[0] != '.')
            {
                continue;
            }
            if (strncmp(de->d_name, base, blen) != 0)
            {
                continue;
            }
            char fullpath[1024];
            snprintf(fullpath, sizeof(fullpath), "%s%s", dirpart, de->d_name);
            struct stat st;
            int         is_dir = (stat(fullpath, &st) == 0 && S_ISDIR(st.st_mode));
            if (fits_only && !is_dir)
            {
                size_t nlen = strlen(de->d_name);
                int    ok   = 0;
                if ((nlen > 5 && strcasecmp(de->d_name + nlen - 5, ".fits") == 0) ||
                    (nlen > 4 && strcasecmp(de->d_name + nlen - 4, ".fit") == 0) ||
                    (nlen > 4 && strcasecmp(de->d_name + nlen - 4, ".fts") == 0) ||
                    (nlen > 8 && strcasecmp(de->d_name + nlen - 8, ".fits.gz") == 0))
                {
                    ok = 1;
                }
                if (!ok)
                {
                    continue;
                }
            }
            printf("%s%s%s\n", dirpart, de->d_name, is_dir ? "/" : "");
            emitted++;
        }
        closedir(dp);
    }
    if (fits_only && emitted == 0)
    {
        complete_local_files(prefix, 0);
    }
    fflush(stdout);
}

/**
 * @brief Query completion candidates from the server (`@complete`).
 */
int milk_client_complete(const char                *prefix,
                         const char                *comp_type,
                         const milk_client_hooks_t *hooks)
{
    if (prefix == NULL)
    {
        prefix = "";
    }
    if (comp_type == NULL)
    {
        comp_type = "arg";
    }

    char sockpath[256];
    int  fd = milk_client_connect_or_start(hooks, sockpath, sizeof(sockpath));
    if (fd < 0)
    {
        return MILK_CLIENT_RC_TRANSPORT;
    }

    size_t   l1  = strlen(prefix) + 1;
    size_t   l2  = strlen(comp_type) + 1;
    uint32_t tot = (uint32_t) (l1 + l2);

    uint8_t *buf = (uint8_t *) malloc(tot);
    if (buf == NULL)
    {
        close(fd);
        return 1;
    }
    memcpy(buf, prefix, l1);
    memcpy(buf + l1, comp_type, l2);

    int sret = milk_proto_send(fd, MILK_MSG_COMPLETE, 0, buf, tot);
    free(buf);
    if (sret != 0)
    {
        close(fd);
        return 1;
    }

    char out[32768];
    int  got_val = 0;
    int  rc      = milk_client_await_response(fd, sockpath, hooks, out, sizeof(out), &got_val);
    close(fd);
    if (rc == 0 && got_val)
    {
        if (strncmp(out, "__MILK_COMPLETE_FITS__", 22) == 0)
        {
            complete_local_files(prefix, 1);
        }
        else if (strncmp(out, "__MILK_COMPLETE_FILES__", 23) == 0)
        {
            complete_local_files(prefix, 0);
        }
        else if (out[0] != '\0')
        {
            fputs(out, stdout);
            size_t olen = strlen(out);
            if (olen > 0 && out[olen - 1] != '\n')
            {
                fputc('\n', stdout);
            }
            fflush(stdout);
        }
    }
    return rc;
}

/**
 * @brief Shut down the current shell's server (`@shutdown`).
 */
int milk_client_shutdown(const milk_client_hooks_t *hooks)
{
    char sockpath[256];
    milk_ipc_resolve_sockpath(hooks, sockpath, sizeof(sockpath), 1);
    int fd = milk_ipc_connect_once(sockpath);
    if (fd < 0)
    {
        milk_client_close_owner_pipe();
        return 0;
    }

    milk_proto_send(fd, MILK_MSG_SHUTDOWN, 0, NULL, 0);
    int rc = milk_client_await_response(fd, sockpath, hooks, NULL, 0, NULL);
    close(fd);
    milk_client_close_owner_pipe();
    return rc;
}

int milk_client_server(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    milk_ipc_sync_shm_env(hooks);
    return milk_srvlog_cmd_server(argc, argv, hooks);
}

extern int milk_ls_exec(int argc, char **argv);

int milk_client_log(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    milk_ipc_sync_shm_env(hooks);
    char sockpath[256];
    milk_ipc_resolve_sockpath(hooks, sockpath, sizeof(sockpath), 1);
    return milk_srvlog_cmd_log(sockpath, argc, argv, hooks);
}

/** kw_where() - human description of the front-ends in @avail (MILK_KW_* bits). */
static const char *kw_where(unsigned avail)
{
    if (avail == MILK_KW_CLI)
    {
        return "only in the milk-cli REPL";
    }
    if (avail == MILK_KW_BASH)
    {
        return "only in a bash shell that sourced milk.bash";
    }
    return "in milk-cli and in milk.bash shells";
}

/**
 * print_keywords() - write the keyword table to @fp as "name impl avail" lines or, with @bash, as the `@`
 * function definitions of the running front-end ($_MILK_ENGINE); unavailable keywords fail with status 2.
 */
static void print_keywords(FILE *fp, int bash, const milk_client_hooks_t *hooks)
{
    static const char *const impl_names[] = { "builtin", "shell", "server" };
    unsigned                 fe = milk_lexicon_frontend(milk_ipc_get_var(hooks, "_MILK_ENGINE"));

    for (const milk_keyword_t *k = milk_lexicon_keywords(); k->name != NULL; k++)
    {
        if (!bash)
        {
            fprintf(fp, "%s %s %s%s%s\n", k->name, impl_names[k->impl],
                    (k->avail & MILK_KW_CLI) ? "c" : "", (k->avail & MILK_KW_SCRIPT) ? "s" : "",
                    (k->avail & MILK_KW_BASH) ? "b" : "");
        }
        else if (!(k->avail & fe))
        {
            fprintf(fp, "function @%s { echo \"@%s: not available here (%s)\" >&2; return 2; }\n",
                    k->name, k->name, kw_where(k->avail));
        }
        else
        {
            const char *pre  = (k->impl == MILK_KW_BUILTIN) ? "milk"
                               : (k->impl == MILK_KW_SHELL) ? "_milk_"
                                                            : "milkcall ";
            const char *post = (k->impl == MILK_KW_SHELL) ? "_impl" : "";
            /* `case` costs less than [[ ]] on every call; with libmilkbash loaded, milk.bash replaces the
             * functions of the builtin keywords by the @<name> builtins */
            fprintf(
                fp,
                "function @%s { case ${1-} in --help%s) milkls --help-keyword %s; return ;; esac; "
                "%s%s%s \"$@\"; }\n",
                k->name, milk_lexicon_help_arg(k, "-h") ? " | -h" : "", k->name, pre, k->name,
                post);
        }
    }
}

/** define_keywords() - define the `@` keyword functions and the __MILK_KWSET name set in the calling shell. */
static int define_keywords(const milk_client_hooks_t *hooks)
{
    char  *buf = NULL;
    size_t len = 0;
    FILE  *fp  = open_memstream(&buf, &len);
    if (fp == NULL)
    {
        return 1;
    }
    print_keywords(fp, 1, hooks);
    fprintf(fp, "declare -gA __MILK_KWSET=(");
    for (const milk_keyword_t *k = milk_lexicon_keywords(); k->name != NULL; k++)
    {
        fprintf(fp, "[%s]=1 ", k->name);
    }
    fprintf(fp, ")\n");
    fclose(fp);
    milk_ipc_eval(hooks, buf);
    free(buf);
    return 0;
}

/** define_namespaces() - fill __MILK_NS_KIND (entity namespace -> kind) in the calling shell. */
static int define_namespaces(const milk_client_hooks_t *hooks)
{
    char  *buf = NULL;
    size_t len = 0;
    FILE  *fp  = open_memstream(&buf, &len);
    if (fp == NULL)
    {
        return 1;
    }
    fprintf(fp, "declare -gA __MILK_NS_KIND=(");
    for (int i = 0; milk_entity_ns_table[i].name != NULL; i++)
    {
        fprintf(fp, "[%s]=%s ", milk_entity_ns_table[i].name,
                milk_ns_kind_names[milk_entity_ns_table[i].kind]);
    }
    fprintf(fp, ")\n");
    fclose(fp);
    milk_ipc_eval(hooks, buf);
    free(buf);
    return 0;
}

int milk_client_ls(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    if (argc == 2 && strcmp(argv[0], "--namespaces") == 0 && strcmp(argv[1], "--define") == 0)
    {
        return define_namespaces(hooks); /* milk.bash, without a fork */
    }
    if (argc == 1 && strcmp(argv[0], "--namespaces") == 0) /* "name kind" lines */
    {
        for (int i = 0; milk_entity_ns_table[i].name != NULL; i++)
        {
            printf("%s %s\n", milk_entity_ns_table[i].name,
                   milk_ns_kind_names[milk_entity_ns_table[i].kind]);
        }
        return 0;
    }
    int arc = client_capture_array(milk_client_ls, argc, argv, "-1", hooks);
    if (arc >= 0)
    {
        return arc;
    }
    if (argc >= 2 && strcmp(argv[0], "--rename") == 0) /* removed spelling -> replacement (hints) */
    {
        const char *nw = milk_rename_cmd(argv[1], NULL);
        if (nw != NULL)
        {
            printf("%s\n", nw);
        }
        return (nw != NULL) ? 0 : 1;
    }
    if (argc >= 2 && strcmp(argv[0], "--help-keyword") == 0)
    {
        const milk_keyword_t *k = milk_lexicon_keyword(argv[1]);
        if (k != NULL)
        {
            milk_lexicon_print_help(k, stdout);
        }
        return (k != NULL) ? 0 : 1;
    }
    for (int i = 0; i < argc; i++)
    {
        if (strcmp(argv[i], "--keywords") ==
            0) /* --bash: definitions; --define: evaluate them (milk.bash) */
        {
            int bash   = 0;
            int define = 0;
            for (int j = 0; j < argc; j++)
            {
                bash |= (strcmp(argv[j], "--bash") == 0);
                define |= (strcmp(argv[j], "--define") == 0);
            }
            if (define)
            {
                return define_keywords(hooks);
            }
            print_keywords(stdout, bash, hooks);
            return 0;
        }
    }
    milk_ipc_sync_shm_env(hooks);
    for (int i = 0; i < argc; i++)
    {
        const char *a = argv[i];
        if (a == NULL)
        {
            continue;
        }
        while (*a == '@')
        {
            a++;
        }
        if (strcmp(a, "--local") == 0 || milk_entity_ns_lookup(a, strlen(a)) == MILK_NS_LOCAL ||
            milk_entity_ns_kind(a, NULL) == MILK_NS_LOCAL || strncmp(a, "L:", 2) == 0)
        {
            char **cargv = (char **) calloc((size_t) argc + 2, sizeof(char *));
            if (cargv == NULL)
            {
                return 1;
            }
            cargv[0] = (char *) "ls";
            for (int j = 0; j < argc; j++)
            {
                cargv[j + 1] = argv[j];
            }
            int rc = milk_client_call(argc + 1, cargv, 0, hooks);
            free(cargv);
            return rc;
        }
    }
    return milk_ls_exec(argc, argv);
}
