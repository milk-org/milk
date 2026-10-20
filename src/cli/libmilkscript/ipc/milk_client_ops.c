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
#include <errno.h>
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

/**
 * get_specs() - `@get spec...` and `@get name=spec...`: print each plain spec's value (one per line) and
 * bind each named one; 0, or 1 if any read failed (the others are still done).
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

int milk_client_get_argv(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    const char *varname   = NULL;
    const char *arrname   = NULL;
    const char *assocname = NULL;
    const char *spec      = NULL;
    int         json_mode = 0;
    int         nspec     = 0;
    char      **specs     = (char **) calloc((size_t) argc + 1, sizeof(char *));
    if (specs == NULL)
    {
        return 1;
    }
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
        else
        {
            specs[nspec++] = argv[i];
        }
    }
    spec = specs[0];
    if (nspec == 0 || ((varname != NULL || arrname != NULL || assocname != NULL) &&
                       (nspec > 1 || strchr(spec, '=') != NULL)))
    {
        fprintf(stderr, "Usage: @get [--json|-j] [-v var | -a arr | -A assoc] <entity> | "
                        "@get [<name>=]<entity>...\n");
        free(specs);
        return 2;
    }
    if (varname == NULL && arrname == NULL && assocname == NULL)
    {
        int rc = get_specs(nspec, specs, json_mode, hooks);
        free(specs);
        return rc;
    }
    free(specs);

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
/** sh_quote() - append @s single-quoted for the shell to the string @out of size @sz; -1 if it does not fit. */
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
 * set_restore() - `@set --restore <assignments>`: read the current value of every entity assigned (before
 * anything is written) and register the `milkset` that puts them back in the caller's @defer scope; 0, or
 * non-zero after a message (nothing registered).
 */
static int set_restore(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    char cmd[16384] = "__MILK_NOLOG=1 milkset";
    int  i          = 0;
    while (i < argc) /* the assignment grammar of milk_client_set_argv() */
    {
        const char *spec = argv[i++];
        const char *eq   = strchr(spec, '=');
        size_t      sl   = (eq != NULL) ? (size_t) (eq - spec) : strlen(spec);
        int         has  = (eq != NULL);
        if (eq == NULL && i < argc && strcmp(argv[i], "=") == 0)
        {
            has = (++i < argc);
            i += has;
        }
        else if (eq == NULL && i < argc && !is_entity_spec_token(argv[i]))
        {
            has = 1;
            i++;
        }
        if (i < argc && !is_entity_spec_token(argv[i])) /* a comment */
        {
            i++;
        }
        char name[512];
        char old[4096];
        char kv[4700];
        snprintf(name, sizeof(name), "%.*s", (int) sl, spec);
        if (!has) /* an action: nothing to put back */
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

/** undo_loggable() - 1 if @name is a value @restore --undo can set back: an FPS parameter or a stream
 * keyword; not an action (conf, run, process or sequencer controls). */
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

/** undo_values() - "<spec>\t<value>\n" lines for @name, one per match of a glob; 0 or -1. A glob in the
 * entity gives <ns>.<match>.<rest>; <entity>.* gives <entity>.<match>. */
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

/** undo_pair() - add "<spec>\t<old>\t<new>" for each spec of @before whose value @after changed. */
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

/** undo_flush() - append the pending entries to __MILK_UNDO (the last 1000 to 2000 calls are kept). */
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

/** undo_commit() - keep the entry @u of this @set call: until the next flush in a shell (a builtin), at
 * once from milk-call. */
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
    if (strcmp(argv[0], "--flush-undo") == 0) /* for @restore --undo */
    {
        undo_flush(hooks);
        return 0;
    }
    if (strcmp(argv[0], "--restore") ==
        0) /* put the old values back when the caller's scope ends */
    {
        int rr = set_restore(argc - 1, argv + 1, hooks);
        if (rr != 0 || argc == 1)
        {
            return rr;
        }
        argc--;
        argv++;
    }

    int         rc   = 0;
    int         i    = 0;
    undo_text_t undo = { NULL, 0, 0 };
    const char *nl = milk_ipc_get_var(hooks, "__MILK_NOLOG"); /* the writes of an undo or restore */
    int         nolog = (nl != NULL && nl[0] != '\0');
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

        /* a value (not an action) that @restore --undo can set back: its values before and after */
        char        name[512];
        char        before[8192];
        const char *eq  = strchr(spec, '=');
        int         log = (eq != NULL || val != NULL) && !nolog;
        snprintf(name, sizeof(name), "%.*s",
                 (int) ((eq != NULL) ? (size_t) (eq - spec) : strlen(spec)), spec);
        log = log && undo_loggable(name) && undo_values(name, before, sizeof(before), hooks) == 0;
        if (milk_client_set(spec, val, comment, hooks) != 0)
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

/** names_ok() - 1 if @names is a comma-separated list of shell variable names (`@calc -v a,b`). */
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

/** bind_values() - set the variables @names to the last of the @n values @v, right-aligned: the k names
 * get the values of the last k statements (a name without one is emptied). */
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

/** calc_local_list() - evaluate the `;`-separated statements of @expr in this process (cli_calc_local());
 * their number, values in @vals, or -1 if one needs the engine. */
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
            (n == maxn || !cli_calc_local(stmt, digits, vals[n++], 256, truth)))
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
 * client_calc_or_test() - @calc / @test @expr. @names ("a,b,...") binds the values of the last statements,
 * @arr the pixels of an image result (or a number); otherwise the values of non-assignments are printed.
 */
static int client_calc_or_test(const char                *expr,
                               int                        is_test,
                               int                        repl,
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

    const char *dp = milk_ipc_get_var(hooks, "MILK_DPDIGITS"); /* sent back by @config digits */
    long        nd = (dp != NULL) ? strtol(dp, NULL, 10) : 0;
    char        lv[16][256];
    char       *lp[16];
    int         truth = 0;
    int n = (arr == NULL && !repl)
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

    /* the values to bind go to a memory file: a pipe read after DONE would block a large -a output */
    int bind      = !is_test && (names != NULL || arr != NULL);
    int out_fd    = bind ? memfd_create("milk_calc_out", MFD_CLOEXEC) : STDOUT_FILENO;
    int cwd_fd    = open(".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    int fds[4]    = { STDIN_FILENO, (out_fd >= 0) ? out_fd : STDOUT_FILENO, STDERR_FILENO, cwd_fd };
    uint8_t mtype = is_test ? MILK_MSG_TEST : MILK_MSG_CALC;
    uint8_t flags = !bind           ? (repl ? MILK_FLAG_REPL : 0)
                    : (arr != NULL) ? MILK_FLAG_PIXELS
                                    : MILK_FLAG_BIND;
    int     sret = milk_proto_send_with_fds(fd, mtype, flags, buf, off, fds, (cwd_fd >= 0) ? 4 : 3);
    if (cwd_fd >= 0)
    {
        close(cwd_fd);
    }
    free(buf);

    int rc = (sret != 0) ? 2 : milk_client_await_response(fd, sockpath, hooks, NULL, 0, NULL);
    close(fd);

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

int milk_client_calc(const char *expr, const char *varname, const milk_client_hooks_t *hooks)
{
    return client_calc_or_test(expr, 0, 0, varname, NULL, hooks);
}

int milk_client_calc_argv(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    const char *names = NULL;
    const char *arr   = NULL;
    int         repl  = 0;
    int         i     = 0;
    while (i < argc)
    {
        if (strcmp(argv[i], "--repl") == 0)
        {
            repl = 1;
            i++;
        }
        else if (i + 1 < argc && (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "-a") == 0))
        {
            *((argv[i][1] == 'v') ? &names : &arr) = argv[i + 1];
            i += 2;
        }
        else
        {
            break;
        }
    }
    size_t tot = 1;
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
    for (int j = i; j < argc; j++)
    {
        strcat(expr, (j > i) ? " " : "");
        strcat(expr, argv[j]);
    }
    int rc = client_calc_or_test(expr, 0, repl, names, arr, hooks);
    free(expr);
    return rc;
}

int milk_client_test(const char *expr, const milk_client_hooks_t *hooks)
{
    return client_calc_or_test(expr, 1, 0, NULL, NULL, hooks);
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

static int client_wait_expr(const char                *expr,
                            int                        next,
                            double                     timeout_sec,
                            const milk_client_hooks_t *hooks);

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
    int         next     = 0;

    for (int i = 0; i < argc; i++)
    {
        if (strcmp(argv[i], "--next") == 0)
        {
            next = 1;
        }
        else if ((strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--timeout") == 0) && i + 1 < argc)
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
                        "<stream|condition> [nframes] | @waitfor [--next] \"<expr>\"\n");
        return 1;
    }

    milk_wait_target_t wt;
    if (milk_waitfor_parse(target, has_arg2 ? arg2 : NULL, &wt) == MILK_WAIT_BAD)
    {
        fprintf(stderr, "@waitfor: invalid target '%s'\n", target);
        return 1;
    }
    if (wt.kind == MILK_WAIT_EXPR)
    {
        while (*target == '@')
        {
            target++;
        }
        return client_wait_expr(target, next, strtod(timeout, NULL), hooks);
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

#define WAIT_EXPR_MAXS 16

/**
 * expr_streams() - the shared streams that @expr names (`s.<name>.<prop>`, or an image name that has a
 * shm file), opened into @imgs; *@other is set if it also names FPS, process or sequencer values.
 */
static int expr_streams(const char *expr, IMAGE *imgs, int *other)
{
    int n  = 0;
    *other = 0;
    for (const char *p = expr; *p != '\0';)
    {
        if (*p == '"' || *p == '\'') /* a string literal */
        {
            const char *q = strchr(p + 1, *p);
            p             = (q != NULL) ? q + 1 : p + strlen(p);
            continue;
        }
        if (!(isalpha((unsigned char) *p) || *p == '_') ||
            (p > expr && (isalnum((unsigned char) p[-1]) || p[-1] == '.')))
        {
            p++;
            continue;
        }
        char   id[160];
        size_t len = strspn(p, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.");
        snprintf(id, sizeof(id), "%.*s", (int) ((len < sizeof(id)) ? len : sizeof(id) - 1), p);
        p += len;
        const char    *after = p + strspn(p, " \t");
        const char    *rest  = id;
        milk_ns_kind_t k     = milk_entity_ns_kind(id, &rest);
        if (*after == '(' || (k == MILK_NS_NONE && strchr(id, '.') != NULL))
        {
            continue; /* a function, or not an entity */
        }
        if (k != MILK_NS_NONE && k != MILK_NS_STREAM)
        {
            *other = 1; /* FPS, process, sequencer or local image: no semaphore */
            continue;
        }
        memmove(id, rest, strlen(rest) + 1);
        id[strcspn(id, ".")] = '\0';
        int dup              = 0;
        for (int i = 0; i < n && !dup; i++)
        {
            dup = (strcmp(imgs[i].name, id) == 0);
        }
        if (!dup && n < WAIT_EXPR_MAXS && milk_waitfor_open(&imgs[n], id) == 0)
        {
            if (imgs[n].md != NULL && imgs[n].md->sem > 0)
            {
                n++;
            }
            else
            {
                ImageStreamIO_closeIm(&imgs[n]);
                *other = 1; /* no semaphore to sleep on */
            }
        }
    }
    return n;
}

/**
 * client_wait_expr() - `@waitfor "<expr>"`: evaluate it (in the shell when it is scalar, else in the
 * server) each time a stream it names posts a frame, polling as well when it names no stream or FPS,
 * process or sequencer values. @next: only from the next frame on. 0 once true, 124 on timeout, 130 on
 * Ctrl-C, the evaluation's status on an error.
 */
static int client_wait_expr(const char                *expr,
                            int                        next,
                            double                     timeout_sec,
                            const milk_client_hooks_t *hooks)
{
    IMAGE    imgs[WAIT_EXPR_MAXS];
    int      opened[WAIT_EXPR_MAXS];
    uint64_t cnt[WAIT_EXPR_MAXS];
    int      other = 0;
    int      iter  = 0;
    int      rc    = 1;

    milk_ipc_sync_shm_env(hooks);
    memset(imgs, 0, sizeof(imgs));
    int n = expr_streams(expr, imgs, &other);
    for (int i = 0; i < n; i++)
    {
        opened[i] = 1;
        cnt[i]    = imgs[i].md->cnt0;
    }

    struct sigaction sa_int, old_int;
    memset(&sa_int, 0, sizeof(sa_int));
    sa_int.sa_handler = client_wait_sigint_handler;
    sigemptyset(&sa_int.sa_mask);
    g_wait_sigint = 0;
    sigaction(SIGINT, &sa_int, &old_int);
    double t0 = milk_wait_now();

    for (;;)
    {
        if (!next)
        {
            for (int i = 0; i < n; i++)
            {
                cnt[i] =
                    imgs[i].md->cnt0; /* frames that come during the evaluation wake the wait */
            }
            rc = milk_client_test(expr, hooks);
            if (rc != 1)
            {
                break;
            }
        }
        next = 0;
        if (g_wait_sigint)
        {
            rc = 130;
            break;
        }
        double left = (timeout_sec > 0.0) ? timeout_sec - (milk_wait_now() - t0) : 1.0;
        if (left <= 0.0)
        {
            rc = 124;
            break;
        }
        if (n == 0 || other)
        {
            milk_wait_backoff(&iter);
            continue;
        }
        sem_waiters_t sw;
        int           hit = 0;
        sem_waiters_start(&sw, imgs, opened, cnt, n);
        for (int i = 0; i < n && !hit; i++)
        {
            hit = (imgs[i].md->cnt0 != cnt[i]);
        }
        if (!hit && sw.efd >= 0)
        {
            struct pollfd pfd = { sw.efd, POLLIN, 0 };
            poll(&pfd, 1, (left < 1.0) ? (int) (left * 1000.0) + 1 : 1000);
        }
        sem_waiters_stop(&sw, imgs);
        for (int i = 0; i < n && !hit; i++)
        {
            hit = (imgs[i].md->cnt0 != cnt[i]);
        }
        next = !hit; /* no new frame (timeout, signal): nothing to evaluate */
    }

    sigaction(SIGINT, &old_int, NULL);
    if (rc == 130 && hooks != NULL && hooks->on_sigint != NULL)
    {
        hooks->on_sigint(hooks->ctx);
    }
    for (int i = 0; i < n; i++)
    {
        ImageStreamIO_closeIm(&imgs[i]);
    }
    return rc;
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
        if (milk_waitfor_open(&imgs[i], streams[i]) == 0)
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
            if (!opened[i] && milk_waitfor_open(&imgs[i], streams[i]) == 0)
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

static volatile sig_atomic_t g_sleep_int = 0;

static void sleep_on_int(int signo)
{
    (void) signo;
    g_sleep_int = 1;
}

/** sleep_until() - sleep to CLOCK_MONOTONIC time @dl (ns); 130 if Ctrl-C ended it, passed on to the shell. */
static int sleep_until(int64_t dl, const milk_client_hooks_t *hooks)
{
    struct timespec  ts = { (time_t) (dl / 1000000000), (long) (dl % 1000000000) };
    struct sigaction sa;
    struct sigaction old;

    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = sleep_on_int;
    sigemptyset(&sa.sa_mask);
    g_sleep_int = 0;
    sigaction(SIGINT, &sa, &old);
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL) == EINTR && !g_sleep_int)
    {
    }
    sigaction(SIGINT, &old, NULL);
    if (!g_sleep_int)
    {
        return 0;
    }
    if (hooks != NULL && hooks->on_sigint != NULL)
    {
        hooks->on_sigint(hooks->ctx);
    }
    else
    {
        raise(SIGINT);
    }
    return 130;
}

static int64_t mono_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (int64_t) t.tv_sec * 1000000000 + t.tv_nsec;
}

/** sleep_seconds() - @s as seconds (0 to 1e9), or -1. */
static double sleep_seconds(const char *s)
{
    char  *end = NULL;
    double v   = strtod(s, &end);
    return (end != s && *end == '\0' && v >= 0.0 && v < 1.0e9) ? v : -1.0;
}

int milk_client_sleep_argv(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    static const char usage[] =
        "Usage: @sleep <seconds> | @sleep --every <seconds>|--hz <rate> [--late <var>] <var>\n";
    double      per  = -1.0;
    const char *var  = NULL;
    const char *late = NULL;

    if (argc == 1 && sleep_seconds(argv[0]) >= 0.0)
    {
        return sleep_until(mono_ns() + (int64_t) (sleep_seconds(argv[0]) * 1.0e9), hooks);
    }
    for (int i = 0; i < argc; i++)
    {
        if ((strcmp(argv[i], "--every") == 0 || strcmp(argv[i], "--hz") == 0) && i + 1 < argc &&
            per < 0.0)
        {
            per = sleep_seconds(argv[i + 1]);
            per = (argv[i][2] == 'h') ? ((per > 0.0) ? 1.0 / per : -1.0) : per;
            i++;
            if (per <= 0.0)
            {
                fputs(usage, stderr);
                return 2;
            }
        }
        else if (strcmp(argv[i], "--late") == 0 && i + 1 < argc && late == NULL)
        {
            late = argv[++i];
        }
        else if (var == NULL && argv[i][0] != '-')
        {
            var = argv[i];
        }
        else
        {
            fputs(usage, stderr);
            return 2;
        }
    }
    if (per <= 0.0 || var == NULL || !milk_ipc_is_identifier(var) ||
        (late != NULL && !milk_ipc_is_identifier(late)))
    {
        fputs(usage, stderr);
        return 2;
    }

    /* the deadline: VAR (ns), or one period from now; without the shell's variables (milk-call), the
     * fallback shim passes VAR's value as __MILK_SLEEP_VAR */
    int64_t     p   = (int64_t) llround(per * 1.0e9);
    int64_t     now = mono_ns();
    int64_t     dl  = now + p;
    const char *cur = (hooks != NULL && hooks->get_var != NULL) ? milk_ipc_get_var(hooks, var)
                                                                : getenv("__MILK_SLEEP_VAR");
    if (p < 1)
    {
        p = 1;
    }
    if (cur != NULL && cur[0] != '\0')
    {
        char *end = NULL;
        dl        = (int64_t) strtoll(cur, &end, 10);
        if (*end != '\0')
        {
            fprintf(stderr, "@sleep: $%s is not a deadline (integer ns): %s\n", var, cur);
            return 2;
        }
    }
    int64_t missed = 0;
    if (dl > now)
    {
        int rc = sleep_until(dl, hooks);
        if (rc != 0)
        {
            return rc;
        }
    }
    else
    {
        missed = (now - dl) / p; /* late: serve this tick now, skip the ones missed */
    }
    char buf[32];
    snprintf(buf, sizeof(buf), "%lld", (long long) (dl + (missed + 1) * p));
    milk_ipc_set_var(hooks, var, buf, 0);
    if (late != NULL)
    {
        snprintf(buf, sizeof(buf), "%lld", (long long) missed);
        milk_ipc_set_var(hooks, late, buf, 0);
    }
    return 0;
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
