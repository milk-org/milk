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
#include "milk_client.h"
#include "milk_client_internal.h"
#include "milk_entity_ns.h"
#include "milk_shm_prop.h"
#include "milk_ipc_common.h"
#include "milk_proto.h"
#include "milk_srvlog.h"

/**
 * @brief Drain client stdin into an anonymous memfd for `@streamwrite` pipelines.
 *
 * Reading stdin before connecting to the single-threaded server prevents deadlocks
 * in pipelines such as `@streamcat im1 | ... | @streamwrite im2`, while supporting
 * both text and binary (`--raw`) streams of any size.
 */
static int drain_stdin_to_memfd(int argc, char **argv)
{
    if (argc < 1 || isatty(STDIN_FILENO))
    {
        return -1;
    }
    const char *c0 = argv[0];
    if (c0[0] == '@' || c0[0] == ',')
    {
        c0++;
    }
    int is_sw    = (strcmp(c0, "streamwrite") == 0 && argc >= 2);
    int is_batch = ((strcmp(c0, "batch") == 0 || strcmp(c0, "pipe") == 0) && argc == 1);
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
 * @brief Dispatch a milk command to `milk-cli --server` with fds and synced vars.
 */
int milk_client_call(int argc, char **argv, int async_flag, const milk_client_hooks_t *hooks)
{
    if (argc <= 0 || argv == NULL || argv[0] == NULL)
    {
        return 0;
    }

    const char *c0 = argv[0];
    if (c0[0] == '@' || c0[0] == ',')
    {
        c0++;
    }
    if (!async_flag && strcmp(c0, "procwait") == 0 && argc >= 2)
    {
        char        ptarget[280];
        const char *raw_p = argv[1];
        while (*raw_p == '@')
        {
            raw_p++;
        }
        if (strncmp(raw_p, "p.", 2) == 0 || strncmp(raw_p, "proc.", 5) == 0 ||
            strncmp(raw_p, "procinfo.", 9) == 0)
        {
            snprintf(ptarget, sizeof(ptarget), "%s", raw_p);
        }
        else
        {
            snprintf(ptarget, sizeof(ptarget), "p.%s", raw_p);
        }
        const char *st   = (argc >= 3) ? argv[2] : "STOPPED";
        const char *timo = (argc >= 4) ? argv[3] : "30";
        char       *wargv[5];
        wargv[0] = (char *) "-t";
        wargv[1] = (char *) timo;
        wargv[2] = ptarget;
        wargv[3] = (char *) st;
        wargv[4] = NULL;
        int wrc  = milk_client_waitfor(4, wargv, hooks);
        return (wrc == 0) ? 0 : 1;
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
        return 1;
    }

    milk_packed_vars_t pv;
    pv.count = 0;
    for (int i = 1; i < argc; i++)
    {
        milk_client_collect_identifiers(&pv, hooks, argv[i]);
    }
    milk_client_collect_synced_vars(&pv, hooks);

    static const char *const env_keys[] = { "MILK_SHM_DIR",      "MILK_QUIET",    "NO_COLOR",
                                            "MILK_FLOAT_DIGITS", "MILKCLI_SIGIL", NULL };
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
        return 1;
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
    int is_local_or_glob =
        ((strncmp(s_norm, "l.", 2) == 0 || strncmp(s_norm, "local.", 6) == 0) ||
         ((strncmp(s_norm, "s.", 2) == 0 || strncmp(s_norm, "stream.", 7) == 0) &&
          (strchr(s_norm, '*') != NULL || strchr(s_norm, '?') != NULL)));

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
        return 1;
    }

    size_t   l1  = strlen(spec) + 1;
    size_t   l2  = (val != NULL) ? (strlen(val) + 1) : 0;
    size_t   l3  = (comment != NULL) ? (strlen(comment) + 1) : 0;
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
        memcpy(buf + l1, val, l2);
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

static int is_entity_spec_token(const char *s)
{
    if (s == NULL)
    {
        return 0;
    }
    char norm[256];
    milk_normalize_entity_spec(s, norm, sizeof(norm));
    static const char *const pfx[] = { "s.",   "l.",      "f.",     "p.",    "q.", "fps.",
                                       "seq.", "stream.", "local.", "proc.", NULL };
    for (int k = 0; pfx[k] != NULL; k++)
    {
        size_t plen = strlen(pfx[k]);
        if (strncmp(norm, pfx[k], plen) == 0 && strchr(norm + plen, '.') != NULL)
        {
            return 1;
        }
    }
    return 0;
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

    char sockpath[256];
    int  fd = milk_client_connect_or_start(hooks, sockpath, sizeof(sockpath));
    if (fd < 0)
    {
        return 2;
    }

    milk_packed_vars_t pv;
    pv.count = 0;
    milk_client_collect_identifiers(&pv, hooks, expr);
    milk_client_collect_synced_vars(&pv, hooks);

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
        char    cbuf[512];
        ssize_t cn = read(cap_pipe[0], cbuf, sizeof(cbuf) - 1);
        close(cap_pipe[0]);
        if (cn > 0)
        {
            cbuf[cn] = '\0';
            while (cn > 0 && (cbuf[cn - 1] == '\n' || cbuf[cn - 1] == '\r'))
            {
                cbuf[--cn] = '\0';
            }
            milk_ipc_set_var(hooks, varname, cbuf, 0);
        }
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
    if (!check_running && (strncmp(s_norm, "s.", 2) == 0 || strncmp(s_norm, "stream.", 7) == 0 ||
                           strncmp(s_norm, "l.", 2) == 0 || strncmp(s_norm, "local.", 6) == 0 ||
                           strchr(s_norm, '.') == NULL))
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

static double client_monotonic_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1.0e-9 * (double) ts.tv_nsec;
}

static int client_status_matches(const char *val, const char *expect)
{
    if (strcmp(val, expect) == 0)
    {
        return 1;
    }
    if ((strcmp(val, "RUN") == 0 && strcmp(expect, "RUNNING") == 0) ||
        (strcmp(val, "RUNNING") == 0 && strcmp(expect, "RUN") == 0))
    {
        return 1;
    }
    char  *e1 = NULL;
    char  *e2 = NULL;
    double d1 = strtod(val, &e1);
    double d2 = strtod(expect, &e2);
    if (val[0] != '\0' && expect[0] != '\0' && *e1 == '\0' && *e2 == '\0' &&
        fabs(d1 - d2) <= 1.0e-9 * (1.0 + fabs(d2)))
    {
        return 1;
    }
    return 0;
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

    const char *tclean = target;
    while (*tclean == '@')
    {
        tclean++;
    }
    double timeout_sec = strtod(timeout, NULL);
    int    has_timo    = (timeout_sec > 0.0);

    int is_complex_expr = (strchr(tclean, '<') != NULL || strchr(tclean, '>') != NULL ||
                           strstr(tclean, "==") != NULL || strstr(tclean, "!=") != NULL ||
                           strstr(tclean, "&&") != NULL || strstr(tclean, "||") != NULL);

    /* Fast Path 0: Client-side process state wait (@waitfor p.<name> [state]) */
    const char *eq = (!is_complex_expr) ? strchr(tclean, '=') : NULL;
    char        proc_eq_buf[320];
    if (!is_complex_expr && eq == NULL &&
        (strncmp(tclean, "p.", 2) == 0 || strncmp(tclean, "proc.", 5) == 0 ||
         strncmp(tclean, "procinfo.", 9) == 0))
    {
        const char *dot1 = strchr(tclean, '.');
        if (dot1 != NULL && dot1[1] != '\0' && strchr(dot1 + 1, '.') == NULL)
        {
            const char *st      = has_arg2 ? arg2 : "STOPPED";
            const char *norm_st = st;
            if (strcasecmp(st, "ACTIVE") == 0 || strcasecmp(st, "RUN") == 0 ||
                strcasecmp(st, "RUNNING") == 0 || strcmp(st, "1") == 0)
            {
                norm_st = "ACTIVE";
            }
            else if (strcasecmp(st, "PAUSE") == 0 || strcasecmp(st, "PAUSED") == 0 ||
                     strcmp(st, "2") == 0)
            {
                norm_st = "PAUSED";
            }
            else if (strcasecmp(st, "STOP") == 0 || strcasecmp(st, "STOPPED") == 0 ||
                     strcasecmp(st, "EXIT") == 0 || strcmp(st, "3") == 0)
            {
                norm_st = "STOPPED";
            }
            else if (strcasecmp(st, "INIT") == 0 || strcmp(st, "0") == 0)
            {
                norm_st = "INIT";
            }
            else if (strcasecmp(st, "ERROR") == 0 || strcmp(st, "4") == 0)
            {
                norm_st = "ERROR";
            }
            snprintf(proc_eq_buf, sizeof(proc_eq_buf), "p.%s.status=%s", dot1 + 1, norm_st);
            tclean = proc_eq_buf;
            eq     = strchr(tclean, '=');
        }
    }

    /* Fast Path 1: Client-side equality condition wait (e.g. f.name.param=val, p.name.status=STOPPED) */
    if (eq != NULL)
    {
        char   lhs[256];
        size_t lhslen = (size_t) (eq - tclean);
        if (lhslen > 0 && lhslen < sizeof(lhs))
        {
            memcpy(lhs, tclean, lhslen);
            lhs[lhslen]        = '\0';
            const char *expect = eq + 1;
            char        val[256];
            int         is_shm_ent = (is_entity_spec_token(lhs) && strncmp(lhs, "l.", 2) != 0 &&
                                      strncmp(lhs, "local.", 6) != 0);
            if (milk_shm_prop_get(lhs, val, sizeof(val)) == 0 || is_shm_ent)
            {
                if (val[0] != '\0' && client_status_matches(val, expect))
                {
                    return 0;
                }
                struct sigaction sa_int, old_int;
                memset(&sa_int, 0, sizeof(sa_int));
                sa_int.sa_handler = client_wait_sigint_handler;
                sigemptyset(&sa_int.sa_mask);
                g_wait_sigint = 0;
                sigaction(SIGINT, &sa_int, &old_int);

                double t_start = client_monotonic_sec();
                int    rc      = 0;
                int    iter    = 0;
                for (;;)
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
                    if (milk_shm_prop_get(lhs, val, sizeof(val)) == 0 &&
                        client_status_matches(val, expect))
                    {
                        rc = 0;
                        break;
                    }
                    if (has_timo && (client_monotonic_sec() - t_start) >= timeout_sec)
                    {
                        rc = 124;
                        break;
                    }
                    long            ns  = (iter++ < 20) ? 1000000L : 5000000L;
                    struct timespec slp = { 0, ns };
                    nanosleep(&slp, NULL);
                }
                sigaction(SIGINT, &old_int, NULL);
                return rc;
            }
        }
    }

    /* Fast Path 2: Client-side /dev/shm semaphore wait (no server fork) */
    if (!is_complex_expr && eq == NULL)
    {
        const char *sname = tclean;
        if (strncmp(sname, "S:", 2) == 0)
        {
            sname += 2;
        }
        else if (strncmp(sname, "s.", 2) == 0)
        {
            sname += 2;
        }
        else if (strncmp(sname, "stream.", 7) == 0)
        {
            sname += 7;
        }

        IMAGE img;
        memset(&img, 0, sizeof(img));
        if (sname[0] != '\0' && strncmp(tclean, "L:", 2) != 0 && strncmp(tclean, "l.", 2) != 0 &&
            ImageStreamIO_openIm(&img, sname) == IMAGESTREAMIO_SUCCESS && img.md != NULL)
        {
            long nframes = strtol(arg2, NULL, 10);
            if (nframes < 1)
            {
                nframes = 1;
            }
            int      req_sem     = (sem_str && sem_str[0]) ? (int) strtol(sem_str, NULL, 10) : -1;
            int      has_abs_cnt = (cnt_str && cnt_str[0]) ? 1 : 0;
            uint64_t abs_cnt     = has_abs_cnt ? (uint64_t) strtoull(cnt_str, NULL, 10) : 0;

            uint64_t start_cnt0 = img.md->cnt0;
            uint64_t target_cnt = has_abs_cnt ? abs_cnt : (start_cnt0 + (uint64_t) nframes);
            int      semindex = (req_sem >= 0) ? req_sem : ImageStreamIO_getsemwaitindex(&img, -1);

            if (semindex >= 0 && semindex < img.md->sem)
            {
                while (sem_trywait(img.semptr[semindex]) == 0)
                {
                }
            }

            struct sigaction sa_int, old_int;
            memset(&sa_int, 0, sizeof(sa_int));
            sa_int.sa_handler = client_wait_sigint_handler;
            sigemptyset(&sa_int.sa_mask);
            g_wait_sigint = 0;
            sigaction(SIGINT, &sa_int, &old_int);

            double t_start = client_monotonic_sec();
            int    rc      = 0;
            while (img.md->cnt0 < target_cnt)
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
                double elapsed = client_monotonic_sec() - t_start;
                if (has_timo && elapsed >= timeout_sec)
                {
                    rc = 124;
                    break;
                }
                if (semindex >= 0 && semindex < img.md->sem)
                {
                    double rem_sec = has_timo ? (timeout_sec - elapsed) : 0.05;
                    if (rem_sec > 0.05)
                    {
                        rem_sec = 0.05;
                    }
                    if (rem_sec < 0.0001)
                    {
                        rem_sec = 0.0001;
                    }
                    struct timespec ts;
                    clock_gettime(CLOCK_REALTIME, &ts);
                    long add_ns = (long) (rem_sec * 1.0e9);
                    ts.tv_sec += add_ns / 1000000000L;
                    ts.tv_nsec += add_ns % 1000000000L;
                    if (ts.tv_nsec >= 1000000000L)
                    {
                        ts.tv_sec += 1;
                        ts.tv_nsec -= 1000000000L;
                    }
                    sem_timedwait(img.semptr[semindex], &ts);
                }
                else
                {
                    struct timespec slp = { 0, 1000000L };
                    nanosleep(&slp, NULL);
                }
            }

            sigaction(SIGINT, &old_int, NULL);
            if (req_sem < 0 && semindex >= 0 && semindex < img.md->sem)
            {
                img.semReadPID[semindex] = 0;
            }
            ImageStreamIO_closeIm(&img);
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
        return 1;
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
            if (strncmp(s, "S:", 2) == 0 || strncmp(s, "L:", 2) == 0 || strncmp(s, "s.", 2) == 0 ||
                strncmp(s, "l.", 2) == 0)
            {
                s += 2;
            }
            else if (strncmp(s, "stream.", 7) == 0)
            {
                s += 7;
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

    double t_start  = client_monotonic_sec();
    int    has_timo = (timeout_sec > 0.0);
    int    rc       = 124;
    int    iter     = 0;

    for (;;)
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

        for (int i = 0; i < nstreams; i++)
        {
            if (!opened[i])
            {
                if (ImageStreamIO_openIm(&imgs[i], streams[i]) == IMAGESTREAMIO_SUCCESS &&
                    imgs[i].md != NULL)
                {
                    opened[i] = 1;
                    if (!has_init[i])
                    {
                        sigaction(SIGINT, &old_int, NULL);
                        for (int j = 0; j < nstreams; j++)
                        {
                            if (opened[j])
                            {
                                ImageStreamIO_closeIm(&imgs[j]);
                            }
                        }
                        milk_ipc_set_var(hooks, "MILK_TRIGGER", streams[i], 0);
                        setenv("MILK_TRIGGER", streams[i], 1);
                        printf("%s\n", streams[i]);
                        fflush(stdout);
                        return 0;
                    }
                }
            }
            if (opened[i] && imgs[i].md != NULL)
            {
                if (imgs[i].md->cnt0 != init_cnt[i])
                {
                    sigaction(SIGINT, &old_int, NULL);
                    for (int j = 0; j < nstreams; j++)
                    {
                        if (opened[j])
                        {
                            ImageStreamIO_closeIm(&imgs[j]);
                        }
                    }
                    milk_ipc_set_var(hooks, "MILK_TRIGGER", streams[i], 0);
                    setenv("MILK_TRIGGER", streams[i], 1);
                    printf("%s\n", streams[i]);
                    fflush(stdout);
                    return 0;
                }
            }
            else
            {
                char spec[280];
                char val[64];
                snprintf(spec, sizeof(spec), "s.%s.cnt0", streams[i]);
                if (milk_shm_prop_get(spec, val, sizeof(val)) == 0)
                {
                    uint64_t c = (uint64_t) strtoull(val, NULL, 10);
                    if (!has_init[i] || c != init_cnt[i])
                    {
                        sigaction(SIGINT, &old_int, NULL);
                        for (int j = 0; j < nstreams; j++)
                        {
                            if (opened[j])
                            {
                                ImageStreamIO_closeIm(&imgs[j]);
                            }
                        }
                        milk_ipc_set_var(hooks, "MILK_TRIGGER", streams[i], 0);
                        setenv("MILK_TRIGGER", streams[i], 1);
                        printf("%s\n", streams[i]);
                        fflush(stdout);
                        return 0;
                    }
                }
            }
        }

        if (has_timo && (client_monotonic_sec() - t_start) >= timeout_sec)
        {
            rc = 124;
            break;
        }

        long            ns  = (iter++ < 40) ? 50000L : 200000L; /* 50 us then 200 us */
        struct timespec slp = { 0, ns };
        nanosleep(&slp, NULL);
    }

    sigaction(SIGINT, &old_int, NULL);
    for (int j = 0; j < nstreams; j++)
    {
        if (opened[j])
        {
            ImageStreamIO_closeIm(&imgs[j]);
        }
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
        return 1;
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
        return 1;
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

int milk_client_ls(int argc, char **argv, const milk_client_hooks_t *hooks)
{
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
        if (strcmp(a, "--local") == 0 || strcmp(a, "l") == 0 || strcmp(a, "local") == 0 ||
            strncmp(a, "l.", 2) == 0 || strncmp(a, "local.", 6) == 0 || strncmp(a, "L:", 2) == 0)
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
