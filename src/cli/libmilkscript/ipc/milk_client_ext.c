// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_client_ext.c
 * @brief Client synchronization (`@waitfor`/`@waitany`/`@sleep`), capture (`@grab`), completion, and discovery (`@ls`).
 *
 * Role & Goal:
 *   Implements the stream synchronization, timing, frame capture, tab completion,
 *   server lifecycle, and entity/keyword discovery operations declared in `milk_client.h`:
 *   - `milk_client_waitfor()`  : Semaphore/property/predicate wait (`@waitfor`)
 *   - `milk_client_waitany()`  : Multi-stream eventfd + semaphore wait (`@waitany`)
 *   - `milk_client_sleep_argv()`: Monotonic sleep and periodic rate-limiter (`@sleep`)
 *   - `milk_client_grab()`     : N-frame stream capture to 3D cube or FITS (`@grab`)
 *   - `milk_client_complete()` : Interactive tab completion query (`@complete`)
 *   - `milk_client_shutdown()` : Clean server shutdown (`@shutdown`)
 *   - `milk_client_session()`  : Sessions (`@session`)
 *   - `milk_client_log()`      : Command/output ring-buffer log (`@log`)
 *   - `milk_client_ls()`       : Entity, keyword, and namespace introspection (`@ls`)
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <poll.h>
#include <pthread.h>
#include <semaphore.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/eventfd.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "milk_client.h"
#include "milk_client_internal.h"
#include "milk_entity_ns.h"
#include "milk_shm_cache.h"
#include "milk_ipc_common.h"
#include "milk_lexicon.h"
#include "milk_proto.h"
#include "milk_renames.h"
#include "milk_shm_prop.h"
#include "milk_srvlog.h"
#include "milk_waitfor.h"

static volatile sig_atomic_t g_wait_sigint = 0;

/**
 * client_wait_sigint_handler - Record `SIGINT` receipt during `@waitfor` / `@waitany`
 * @sig: Signal number (unused)
 */
static void client_wait_sigint_handler(int sig)
{
    (void) sig;
    g_wait_sigint = 1;
}

/**
 * client_wait_cancelled - Callback for `milk_wait_ops_t` checking whether `SIGINT` occurred
 * @ctx: Opaque context pointer (unused)
 *
 * Return: Non-zero if interrupted by `SIGINT`, 0 otherwise.
 */
static int client_wait_cancelled(void *ctx)
{
    (void) ctx;
    return g_wait_sigint;
}

/**
 * client_wait_local - Execute a `@waitfor` stream or `/dev/shm` property wait in the client process
 * @wt:          Parsed wait target (`MILK_WAIT_STREAM` or `MILK_WAIT_EQ`)
 * @arg2:        Frame count delta string
 * @sem_str:     Optional semaphore index string
 * @cnt_str:     Optional absolute `cnt0` target string
 * @timeout_sec: Timeout in seconds (`0` = infinite)
 * @hooks:       Optional shell variable/signal hooks
 *
 * Avoids a server round-trip when waiting on `/dev/shm` streams or entity properties.
 *
 * Return: 0 on success, 124 on timeout, 130 on Ctrl-C, or `MILK_WAIT_ABSENT` if server fallback is needed.
 */
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
        rc = milk_waitfor_eq(wt->lhs, wt->any ? NULL : wt->expect, t0, timeout_sec, &ops);
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
 * milk_client_waitfor - Wait for stream frames, property equality, or a boolean expression (`@waitfor`)
 * @argc:  Argument count
 * @argv:  Argument vector (`-t <sec>`, `-s <sem>`, `--cnt <n>`, `--next`, `<target>`, `[nframes]`)
 * @hooks: Optional shell variable/signal hooks
 *
 * Return: 0 when satisfied, 124 on timeout, 130 on Ctrl-C, or non-zero on error.
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
    const char *gone = milk_entity_ns_removed(target);
    if (gone != NULL) /* q.x...: nothing will ever match */
    {
        fprintf(stderr, "milk: %s: %s\n", target, gone);
        return 1;
    }

    milk_wait_target_t wt;
    if (milk_waitfor_parse(target, has_arg2 ? arg2 : NULL, &wt) == MILK_WAIT_BAD)
    {
        fprintf(stderr, "@waitfor: invalid target '%s'%s%s\n", target, wt.why[0] ? ": " : "",
                wt.why);
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
               (milk_client_is_entity_spec_token(wt.lhs) &&
                milk_entity_ns_kind(wt.lhs, NULL) != MILK_NS_LOCAL);
    }
    if (here)
    {
        int rc = client_wait_local(&wt, arg2, sem_str, cnt_str, strtod(timeout, NULL), hooks);
        if (rc != MILK_WAIT_ABSENT)
        {
            return (rc == MILK_WAIT_FULL) ? 1 : rc;
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

/**
 * sem_waiter_main - Worker thread blocking on a stream semaphore and signaling an `eventfd`
 * @p: Pointer to `sem_waiter_arg_t`
 *
 * Waits on `im->semptr[semindex]` and writes `1` to `a->efd` as soon as `cnt0`
 * advances past `init_cnt` or exits when `*a->stop` is set.
 *
 * Return: NULL.
 */
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

/**
 * sem_waiters_start - Spawn semaphore waiter threads for all open streams with semaphores
 * @sw:       Waiter state structure to initialize
 * @imgs:     Array of `IMAGE` handles
 * @opened:   Per-stream flag indicating whether `imgs[i]` is open
 * @init_cnt: Initial `cnt0` snapshot per stream
 * @n:        Number of streams
 */
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
        int si =
            milk_shm_sem_claim(&imgs[i]); /* a slot of its own, apart from any pinned waiter's */
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
            milk_shm_sem_release(&imgs[i], si);
        }
    }
    pthread_sigmask(SIG_SETMASK, &old, NULL);
}

/**
 * sem_waiters_stop - Wake and join all active semaphore waiter threads and release their semaphores
 * @sw:   Waiter state structure
 * @imgs: Array of `IMAGE` handles
 */
static void sem_waiters_stop(sem_waiters_t *sw, IMAGE *imgs)
{
    sw->stop = 1;
    for (int i = 0; i < sw->n; i++)
    {
        if (sw->active[i])
        {
            sem_post(imgs[i].semptr[sw->semindex[i]]); /* our own semaphore: nobody else reads it */
            pthread_join(sw->tid[i], NULL);
            milk_shm_sem_release(&imgs[i], sw->semindex[i]);
        }
    }
    if (sw->efd >= 0)
    {
        close(sw->efd);
    }
}

#define WAIT_EXPR_MAXS 16

/**
 * expr_streams - Open `/dev/shm` streams referenced inside predicate expression @expr
 * @expr:  Predicate expression string
 * @imgs:  Output array of opened `IMAGE` structures (up to `WAIT_EXPR_MAXS`)
 * @other: Output flag set to 1 if @expr also references non-stream entities (FPS, proc, local)
 *
 * Scans @expr for stream identifiers so `client_wait_expr()` can sleep directly on
 * their semaphores instead of busy-polling when the predicate depends only on streams.
 *
 * Return: Number of streams opened into @imgs.
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
            *other = 1; /* FPS, process or local image: no semaphore */
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
 * client_wait_expr - Wait until calculator predicate @expr evaluates to true (`@waitfor "<expr>"`)
 * @expr:        Boolean expression to evaluate
 * @next:        Non-zero (`--next`) to wait for a new stream frame before the first evaluation
 * @timeout_sec: Timeout in seconds (`0` = infinite)
 * @hooks:       Optional shell variable/signal hooks
 *
 * Sleeps on stream semaphores when @expr depends solely on `/dev/shm` streams,
 * or uses adaptive backoff polling when @expr also references FPS/process properties.
 *
 * Return: 0 once true, 124 on timeout, 130 on Ctrl-C, or evaluation error status.
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

/**
 * milk_client_waitany - Wait for any of the given streams to advance its `cnt0` counter (`@waitany`)
 * @argc:  Argument count
 * @argv:  Argument vector (`<stream1> <stream2> ... [-t|--timeout <sec>]`)
 * @hooks: Optional shell variable/signal hooks
 *
 * Uses per-stream semaphore waiter threads signaling a shared `eventfd` so waiting
 * on multiple streams has sub-millisecond latency without spawning subshells.
 * Sets `MILK_TRIGGER` and prints the triggered stream name on success.
 *
 * Return: 0 on trigger, 124 on timeout, 130 on Ctrl-C, or 1 on usage error.
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

/**
 * resolve_client_path - Resolve a relative client file path against `getcwd()` before sending to server
 * @in:    Input path (optionally prefixed with `'!'` for FITS overwrite)
 * @out:   Destination buffer for absolute path
 * @outsz: Capacity of @out in bytes
 *
 * Ensures `@grab --fits rel.fits` writes relative to the client's working directory
 * even if the background server daemon started elsewhere.
 */
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

/**
 * sleep_on_int - Signal handler recording `SIGINT` during `@sleep`
 * @signo: Signal number (unused)
 */
static void sleep_on_int(int signo)
{
    (void) signo;
    g_sleep_int = 1;
}

/**
 * sleep_until - Sleep until `CLOCK_MONOTONIC` deadline @dl (in nanoseconds)
 * @dl:    Absolute `CLOCK_MONOTONIC` deadline in nanoseconds
 * @hooks: Optional shell signal hooks
 *
 * Return: 0 on completion, 130 if interrupted by `SIGINT`.
 */
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

/**
 * mono_ns - Read current `CLOCK_MONOTONIC` timestamp in nanoseconds
 *
 * Return: Monotonic time in nanoseconds.
 */
static int64_t mono_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (int64_t) t.tv_sec * 1000000000 + t.tv_nsec;
}

/**
 * sleep_seconds - Parse @s as a non-negative duration in seconds (`0 .. 1e9`)
 * @s: Input string
 *
 * Return: Duration in seconds, or `-1.0` if invalid.
 */
static double sleep_seconds(const char *s)
{
    char  *end = NULL;
    double v   = strtod(s, &end);
    return (end != s && *end == '\0' && v >= 0.0 && v < 1.0e9) ? v : -1.0;
}

/**
 * milk_client_sleep_argv - High-resolution monotonic sleep and drift-free periodic timer (`@sleep`)
 * @argc:  Argument count
 * @argv:  Argument vector (`<seconds>` or `--every <sec>|--hz <rate> [--late <var>] <var>`)
 * @hooks: Optional shell variable/signal hooks
 *
 * Return: 0 on completion, 130 on Ctrl-C, or 2 on usage error.
 */
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
 * milk_client_grab - Capture 1 or N consecutive frames from a stream into an image/cube or FITS file (`@grab`)
 * @argc:  Argument count
 * @argv:  Argument vector
 * @hooks: Optional shell variable hooks
 *
 * Dispatches `--stdout` to `streamcat` and otherwise sends a `MILK_MSG_GRAB`
 * request to the server.
 *
 * Return: 0 on success, non-zero on error.
 */
int milk_client_grab(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    if (milk_client_has_arg(argc, argv, "--stdout"))
    {
        return milk_client_call_as("streamcat", argc, argv, "--stdout", NULL, 0, hooks);
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

/**
 * complete_local_files - Print local filesystem completion candidates matching @prefix
 * @prefix:    Path prefix to complete
 * @fits_only: Non-zero to prefer `.fits`/`.fit`/`.fts`/`.fits.gz` files (and directories)
 *
 * Executed in the client process when the server responds with
 * `__MILK_COMPLETE_FITS__` or `__MILK_COMPLETE_FILES__` so completions reflect
 * the client's current working directory.
 */
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
 * milk_client_complete - Query tab-completion candidates from the server (`@complete`)
 * @prefix:    Token prefix being completed
 * @comp_type: Completion context (`"cmd"`, `"arg"`, `"arg:<cmd>:<idx>"`)
 * @hooks:     Optional shell variable hooks
 *
 * Return: 0 on success, non-zero on error.
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
 * milk_client_shutdown - Shut down the current shell's background server (`@shutdown`)
 * @hooks: Optional shell variable hooks
 *
 * Return: 0 on clean shutdown (or if no server was running), non-zero on error.
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

/**
 * milk_client_session - Sessions (`@session`)
 * @argc:  Argument count
 * @argv:  Argument vector
 * @hooks: Optional shell variable hooks
 *
 * Return: 0 on success, non-zero on error.
 */
int milk_client_session(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    milk_ipc_sync_shm_env(hooks);
    return milk_srvlog_cmd_session(argc, argv, hooks);
}

extern int milk_ls_exec(int argc, char **argv);

/**
 * milk_client_log - Inspect or clear the server's shared-memory command/output ring log (`@log`)
 * @argc:  Argument count
 * @argv:  Argument vector
 * @hooks: Optional shell variable hooks
 *
 * Return: 0 on success, non-zero on error.
 */
int milk_client_log(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    milk_ipc_sync_shm_env(hooks);
    char sockpath[256];
    milk_ipc_resolve_sockpath(hooks, sockpath, sizeof(sockpath), 1);
    return milk_srvlog_cmd_log(sockpath, argc, argv, hooks);
}

/**
 * kw_where - Return a human-readable description of front-ends matching `MILK_KW_*` mask @avail
 * @avail: Bitmask of `MILK_KW_CLI`, `MILK_KW_SCRIPT`, `MILK_KW_BASH`
 *
 * Return: Static descriptive string for diagnostic messages.
 */
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
 * print_keywords - Write the `@` keyword table or bash function wrappers to @fp
 * @fp:    Output stream
 * @bash:  Non-zero to emit `function @<name> { ... }` definitions for `milk.bash`
 * @hooks: Optional shell variable hooks (used to read `_MILK_ENGINE`)
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

/**
 * define_keywords - Define `@` keyword functions and `__MILK_KWSET` in the calling shell without forking
 * @hooks: Shell eval hooks
 *
 * Return: 0 on success, 1 on allocation error.
 */
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

/**
 * define_namespaces - Populate the `__MILK_NS_KIND` associative array in the calling shell
 * @hooks: Shell eval hooks
 *
 * Return: 0 on success, 1 on allocation error.
 */
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

/**
 * milk_client_ls - List `/dev/shm` entities, server-local images, keywords, or namespaces (`@ls`)
 * @argc:  Argument count
 * @argv:  Argument vector
 * @hooks: Optional shell variable/eval hooks
 *
 * Handles `--namespaces`, `--keywords`, `--rename`, `--help-keyword`, and `-a <arr>`
 * directly in the client, forwards `--local` / `l.*` queries to the server, and
 * delegates `/dev/shm` entity listings to `milk_ls_exec()`.
 *
 * Return: 0 on success, non-zero on error.
 */
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
    int arc = milk_client_capture_array(milk_client_ls, argc, argv, "-1", hooks);
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
