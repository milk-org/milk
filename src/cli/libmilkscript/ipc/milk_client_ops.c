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

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "milk_client.h"
#include "milk_client_internal.h"
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
    if (argc < 2 || isatty(STDIN_FILENO))
    {
        return -1;
    }
    const char *c0 = argv[0];
    if (c0[0] == '@' || c0[0] == ',')
    {
        c0++;
    }
    if (strcmp(c0, "streamwrite") != 0)
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
 * @brief Read a live entity property (`@get`), using `/dev/shm` fast path first.
 */
int milk_client_get(const char *spec, const char *varname, const milk_client_hooks_t *hooks)
{
    if (spec == NULL || spec[0] == '\0')
    {
        return 1;
    }
    milk_ipc_sync_shm_env(hooks);

    char val[1024];
    val[0] = '\0';

    if (milk_shm_prop_get(spec, val, sizeof(val)) == 0)
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
    int rc      = milk_client_await_response(fd, sockpath, hooks, val, sizeof(val), &got_val);
    close(fd);

    if (rc == 0 && got_val)
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

    struct timespec ts0;
    clock_gettime(CLOCK_REALTIME, &ts0);

    if (milk_shm_prop_set(spec, val, comment) == 0)
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

    const char *s = spec;
    while (*s == '@')
    {
        s++;
    }
    if (!check_running &&
        (strncmp(s, "s.", 2) == 0 || strncmp(s, "stream.", 7) == 0 || strchr(s, '.') == NULL))
    {
        char sockpath[256];
        milk_ipc_resolve_sockpath(hooks, sockpath, sizeof(sockpath), 1);
        int fd = milk_ipc_connect_once(sockpath);
        if (fd >= 0)
        {
            size_t   slen = strlen(spec) + 1;
            uint8_t *pl   = (uint8_t *) malloc(slen + 1);
            if (pl != NULL)
            {
                pl[0] = check_running ? 'R' : 'E';
                memcpy(pl + 1, spec, slen);
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

/**
 * @brief Wait for stream frames or a boolean condition (`@waitfor`).
 */
int milk_client_waitfor(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    const char *target  = NULL;
    const char *arg2    = "1";
    const char *timeout = "0";
    const char *sem_str = NULL;
    const char *cnt_str = NULL;

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
            arg2 = argv[i];
        }
    }

    if (target == NULL || target[0] == '\0')
    {
        fprintf(stderr, "Usage: @waitfor [-t|--timeout <sec>] [-s <sem>] [--cnt <target>] "
                        "<stream|condition> [nframes]\n");
        return 1;
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
            if (strncmp(s, "s.", 2) == 0)
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

    struct timespec ts0;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    double t_start  = (double) ts0.tv_sec + 1.0e-9 * (double) ts0.tv_nsec;
    int    has_timo = (timeout_sec > 0.0);
    int    rc       = 124;

    for (;;)
    {
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
                        /* Newly created stream counts as updated */
                        for (int j = 0; j < nstreams; j++)
                        {
                            if (opened[j])
                            {
                                ImageStreamIO_closeIm(&imgs[j]);
                            }
                        }
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
                    for (int j = 0; j < nstreams; j++)
                    {
                        if (opened[j])
                        {
                            ImageStreamIO_closeIm(&imgs[j]);
                        }
                    }
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
                        for (int j = 0; j < nstreams; j++)
                        {
                            if (opened[j])
                            {
                                ImageStreamIO_closeIm(&imgs[j]);
                            }
                        }
                        printf("%s\n", streams[i]);
                        fflush(stdout);
                        return 0;
                    }
                }
            }
        }

        if (has_timo)
        {
            struct timespec ts_now;
            clock_gettime(CLOCK_MONOTONIC, &ts_now);
            double now = (double) ts_now.tv_sec + 1.0e-9 * (double) ts_now.tv_nsec;
            if ((now - t_start) >= timeout_sec)
            {
                rc = 124;
                break;
            }
        }

        struct timespec slp = { 0, 1000000L }; /* 1 ms */
        nanosleep(&slp, NULL);
    }

    for (int j = 0; j < nstreams; j++)
    {
        if (opened[j])
        {
            ImageStreamIO_closeIm(&imgs[j]);
        }
    }
    return rc;
}

/**
 * @brief Capture 1 or N consecutive frames from a stream into an image/3D cube (`@grab`).
 */
int milk_client_grab(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    const char *sname   = NULL;
    const char *nframes = NULL;
    const char *outname = NULL;
    const char *timeout = "0";

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

    /* Support 2-arg form: @grab <src_stream> <dst_img> [-t <sec>] */
    if (sname != NULL && nframes != NULL && outname == NULL)
    {
        outname = nframes;
        nframes = "1";
    }

    if (sname == NULL || nframes == NULL || outname == NULL)
    {
        fprintf(stderr, "Usage: @grab [-t|--timeout <sec>] <stream> [nframes] <out_img>\n");
        return 1;
    }

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
    uint32_t tot = (uint32_t) (l1 + l2 + l3 + l4);

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

    int rc = milk_client_await_response(fd, sockpath, hooks, NULL, 0, NULL);
    close(fd);
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

int milk_client_log(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    milk_ipc_sync_shm_env(hooks);
    char sockpath[256];
    milk_ipc_resolve_sockpath(hooks, sockpath, sizeof(sockpath), 1);
    return milk_srvlog_cmd_log(sockpath, argc, argv, hooks);
}
