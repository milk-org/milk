// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_server.c
 * @brief Milk UNIX domain socket listener, connection handler, and SHM log tee pump (see
 *        milk_server.h).
 *
 * The listening socket and the connections shells keep are in one epoll set. Requests that touch
 * milk state run one at a time, under a recursive command lock: a command gets the caller's stdin,
 * stdout, stderr and directory, passed with the request and swapped in process-wide. PING,
 * SHUTDOWN, COMPLETE, GET and EXISTS are answered at once. WAITFOR and GRAB run in a thread of this
 * process, so that they see live state; --async calls and @replay run in a forked child, on a copy
 * of it. When the output of a CALL is not a terminal, one pump thread forwards it and keeps its
 * start for the session's log.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#ifdef _OPENMP
#    include <omp.h>
#endif

#include "CLIcore.h"
#include "libmilkcommon/milk_omp_team.h"
#include "libmilkcommon/milk_compiler.h"
#include "milk_safepoint.h"
#include "milk_shm_prop.h"
#include "milk_ipc_common.h"
#include "milk_proto.h"
#include "milk_catalog.h"
#include "milk_cmd_index.h"
#include "milk_detach.h"
#include "milk_sdnotify.h"
#include "milk_server.h"
#include "milk_server_internal.h"
#include "milk_srvlog.h"

#define MILK_SERVER_REQ_TIMEOUT_MS 2000 // max time to receive a request frame

#ifndef F_SETPIPE_SZ
#    define F_SETPIPE_SZ 1031
#endif

milk_srvlog_shm_t *g_srvlog_shm           = NULL;
char               g_srvlog_sockpath[256] = "";

volatile sig_atomic_t g_milk_server_stop = 0;

/* Idle timeout: when a request other than a ping last came (CLOCK_MONOTONIC ns), and the stream
 * requests still running */
static uint64_t g_last_request_ns = 0;
static int      g_stream_workers  = 0;

static void mark_request(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    __atomic_store_n(&g_last_request_ns, (uint64_t) t.tv_sec * 1000000000ULL + (uint64_t) t.tv_nsec,
                     __ATOMIC_RELEASE);
}

static pthread_mutex_t g_cmd_mutex;
static pthread_once_t  g_cmd_mutex_once = PTHREAD_ONCE_INIT;

/**
 * cmd_mutex_init - Initialize the recursive server command serialization mutex
 *
 * Configures `g_cmd_mutex` as `PTHREAD_MUTEX_RECURSIVE` so nested command
 * dispatches on the same thread do not deadlock while worker threads are
 * serialized against process-wide state and stdio changes.
 *
 * Return: void
 */
static void cmd_mutex_init(void)
{
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&g_cmd_mutex, &attr);
    pthread_mutexattr_destroy(&attr);
}

/**
 * milk_server_cmd_lock - Acquire the recursive server command mutex
 *
 * Serializes access to process-wide `data` state, `dcvar` tables, and temporary
 * stdio/cwd redirections across the main server loop and detached stream worker
 * threads.
 *
 * Return: void
 */
void milk_server_cmd_lock(void)
{
    pthread_once(&g_cmd_mutex_once, cmd_mutex_init);
    pthread_mutex_lock(&g_cmd_mutex);
}

/**
 * milk_server_cmd_unlock - Release the recursive server command mutex
 *
 * Releases the lock acquired by `milk_server_cmd_lock`.
 *
 * Return: void
 */
void milk_server_cmd_unlock(void)
{
    pthread_mutex_unlock(&g_cmd_mutex);
}

/* ---------------------------------------------------------------------------------------------- */
/* Connections kept between requests                                                              */
/* ---------------------------------------------------------------------------------------------- */

/* A shell keeps its connection between requests (milk_client_request()). The listening socket and the
 * kept connections are in one epoll set, which the loops poll (milk_server_ready_fd()). A kept
 * connection is armed once (EPOLLONESHOT): its request disarms it, and it is armed again when it is
 * kept after the reply. g_kept lists the kept ones, which a forked child closes. */
#define MILK_SERVER_KEPT_MAX 256  /* connections kept at most: others close after their request */
#define MILK_SERVER_KEPT_FDS 4096 /* and their fd numbers are below this */
static int      g_epfd = -1;
static uint64_t g_kept[MILK_SERVER_KEPT_FDS / 64];
static int      g_nkept = 0;

int milk_server_ready_fd(void)
{
    return g_epfd;
}

int milk_server_can_keep(int fd)
{
    return g_epfd >= 0 && fd >= 0 && fd < MILK_SERVER_KEPT_FDS && g_nkept < MILK_SERVER_KEPT_MAX;
}

void milk_server_send_done(int client_fd, int keep, const void *pl, uint32_t len)
{
    milk_proto_send(client_fd, MILK_MSG_DONE, keep ? MILK_FLAG_KEEP : 0, pl, len);
}

/** conn_arm() - watch @fd in the epoll set: once (@oneshot, a connection), or always (listening). */
static int conn_arm(int fd, int oneshot)
{
    struct epoll_event ev = { .events  = EPOLLIN | (oneshot ? (EPOLLRDHUP | EPOLLONESHOT) : 0),
                              .data.fd = fd };
    return (epoll_ctl(g_epfd, EPOLL_CTL_MOD, fd, &ev) == 0 ||
            (errno == ENOENT && epoll_ctl(g_epfd, EPOLL_CTL_ADD, fd, &ev) == 0))
               ? 0
               : -1;
}

/** conn_take() - kept connection @fd has a request: its handler owns it now. */
static void conn_take(int fd)
{
    if (fd < MILK_SERVER_KEPT_FDS && (g_kept[fd / 64] & (1ULL << (fd % 64))) != 0)
    {
        g_kept[fd / 64] &= ~(1ULL << (fd % 64));
        g_nkept--;
    }
}

void milk_server_conn_done(int fd, int outcome)
{
    /* kept as the reply said, even when others filled the set meanwhile (milk_server_can_keep()) */
    if (outcome == MILK_CONN_KEEP && fd < MILK_SERVER_KEPT_FDS && conn_arm(fd, 1) == 0)
    {
        g_kept[fd / 64] |= 1ULL << (fd % 64);
        g_nkept++;
    }
    else if (outcome != MILK_CONN_TAKEN)
    {
        if (g_epfd >= 0)
        {
            epoll_ctl(g_epfd, EPOLL_CTL_DEL, fd, NULL);
        }
        close(fd);
    }
}

int milk_server_take_ready(int listen_fd, int *fds, int max)
{
    struct epoll_event ev[16];
    int                n = (g_epfd >= 0) ? epoll_wait(g_epfd, ev, (max < 16) ? max : 16, 0) : 0;
    int                k = 0;
    for (int i = 0; i < n; i++)
    {
        int fd = ev[i].data.fd;
        if (fd == listen_fd)
        {
            fd = accept4(listen_fd, NULL, NULL, SOCK_CLOEXEC);
            if (fd >= 0 && !milk_ipc_peer_is_self(fd))
            {
                close(fd);
                fd = -1;
            }
        }
        else
        {
            conn_take(fd);
        }
        if (fd >= 0)
        {
            fds[k++] = fd;
        }
    }
    return k;
}

static int g_listen_released; /* the socket name is no longer this server's: do not unlink it */

void milk_server_stop_listening(int listen_fd)
{
    if (g_srvlog_shm == NULL || (g_srvlog_shm->flags & MILK_SRVFLAG_INPROC))
    {
        return; /* milk-cli's session: the line runs on, its commands sent here, until its prompt */
    }
    int d = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (listen_fd >= 0 && d >= 0 && g_epfd >= 0)
    {
        epoll_ctl(g_epfd, EPOLL_CTL_DEL, listen_fd, NULL);
        dup3(d, listen_fd, O_CLOEXEC); /* its name is free: the next request starts a new server */
        if (g_srvlog_sockpath[0] != '@')
        {
            unlink(g_srvlog_sockpath); /* now, not at exit, where it may be the new server's */
        }
        g_listen_released = 1;
    }
    if (d >= 0)
    {
        close(d);
    }
    for (int w = 0; w < MILK_SERVER_KEPT_FDS / 64; w++) /* the shells see them end */
    {
        for (uint64_t b = g_kept[w]; b != 0; b &= b - 1)
        {
            milk_server_conn_done(w * 64 + __builtin_ctzll(b), MILK_CONN_CLOSE);
        }
        g_kept[w] = 0;
    }
    g_nkept = 0;
}

/** conns_atfork_child() - in a forked child: the kept connections and their set are the parent's. */
static void conns_atfork_child(void)
{
    for (int w = 0; w < MILK_SERVER_KEPT_FDS / 64; w++)
    {
        for (uint64_t b = g_kept[w]; b != 0; b &= b - 1)
        {
            close(w * 64 + __builtin_ctzll(b));
        }
    }
    memset(g_kept, 0, sizeof(g_kept));
    g_nkept = 0;
    if (g_epfd >= 0)
    {
        close(g_epfd);
        g_epfd = -1;
    }
}

/**
 * milk_child_reset_omp - Reset OpenMP thread pool and mutex state in forked children
 *
 * Invoked as a `pthread_atfork` child handler so forked `--async` and `@replay`
 * worker processes reset OpenMP to a single thread, reinitialize `g_cmd_mutex`
 * in an unlocked state, and clear inherited watcher state and kept connections.
 *
 * Return: void
 */
static void milk_child_reset_omp(void)
{
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    milk_omp_force_serial(); /* a num_threads() clause would override the line above */
    cmd_mutex_init();        /* the forking thread may not be the one holding it: start unlocked */
    milk_server_watcher_atfork_child();
    conns_atfork_child();
}

/** stream_worker() - serve one WAITFOR/GRAB request inline, then close its connection. */
static void *stream_worker(void *arg)
{
    int fd    = (int) (intptr_t) arg;
    int dummy = 0;
    milk_server_handle_client(fd, -1, &dummy, 0);
    close(fd);
    mark_request();
    __atomic_sub_fetch(&g_stream_workers, 1, __ATOMIC_RELEASE);
    return NULL;
}

/**
 * milk_server_spawn_stream_worker - Spawn a detached thread to service a blocking stream request
 * @client_fd: Connected client socket file descriptor (ownership transferred on success)
 *
 * Launches a detached worker thread with all signals blocked so long-running
 * `MILK_MSG_WAITFOR` and `MILK_MSG_GRAB` requests execute in-process (sharing live
 * local images and streams) without blocking concurrent server requests.
 *
 * Return: 0 if the worker thread was spawned, -1 on failure
 */
int milk_server_spawn_stream_worker(int client_fd)
{
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    sigset_t all, old; /* signals belong to the main thread (sig_handler may siglongjmp) */
    sigfillset(&all);
    pthread_sigmask(SIG_BLOCK, &all, &old);
    pthread_t tid;
    __atomic_add_fetch(&g_stream_workers, 1,
                       __ATOMIC_ACQ_REL); // a waiting client holds the session
    int rc = pthread_create(&tid, &attr, stream_worker, (void *) (intptr_t) client_fd);
    if (rc != 0)
    {
        __atomic_sub_fetch(&g_stream_workers, 1, __ATOMIC_RELEASE);
    }
    pthread_sigmask(SIG_SETMASK, &old, NULL);
    pthread_attr_destroy(&attr);
    return (rc == 0) ? 0 : -1;
}

/** peek_request_type() - type of the request frame waiting on @fd, without consuming it (-1 if none). */
static int peek_request_type(int fd)
{
    milk_frame_hdr_t ph;
    if (recv(fd, &ph, sizeof(ph), MSG_PEEK | MSG_WAITALL) != (ssize_t) sizeof(ph))
    {
        return -1;
    }
    return ph.type;
}

static pthread_once_t g_atfork_once = PTHREAD_ONCE_INIT;

/**
 * register_atfork_handler - Register the child post-fork cleanup handler once per process
 *
 * Installs `milk_child_reset_omp` via `pthread_atfork` so any subsequent `fork()`
 * in the server safely resets OpenMP and synchronization primitives in the child.
 *
 * Return: void
 */
static void register_atfork_handler(void)
{
    pthread_atfork(NULL, NULL, milk_child_reset_omp);
    milk_safe_point_fn = milk_server_safe_point; /* a command answers waiting reads there */
}

/**
 * @brief Refresh module/image/variable counts and working directory in the SHM log header.
 */
void milk_server_refresh_log_stats(void)
{
    if (g_srvlog_shm == NULL)
    {
        return;
    }

    uint32_t nb_images = 0;
    if (dcimg != NULL)
    {
        for (long i = 0; i < dcnimg; i++)
        {
            if (dcimg[i].used)
            {
                nb_images++;
            }
        }
    }

    uint32_t nb_vars = 0;
    if (dcvar != NULL)
    {
        for (long i = 0; i < dcnvar; i++)
        {
            if (dcvar[i].used)
            {
                nb_vars++;
            }
        }
    }

    char cwd[256] = "";
    if (getcwd(cwd, sizeof(cwd)) == NULL)
    {
        cwd[0] = '\0';
    }

    milk_srvlog_server_update_stats(g_srvlog_shm, (uint32_t) data.NBmodule, nb_images, nb_vars,
                                    cwd);
}

/**
 * @brief Set additional flags on the active server SHM log header.
 *
 * @param flags Bitmask of `MILK_SRVFLAG_*` flags to OR into `shm->flags`
 */
void milk_server_set_log_flags(uint32_t flags)
{
    if (g_srvlog_shm != NULL)
    {
        g_srvlog_shm->flags |= flags;
    }
}

const char *milk_server_session_name(void)
{
    static char  name[64];
    unsigned int uid = 0;

    if (g_srvlog_sockpath[0] == '\0')
    {
        return "";
    }
    milk_srvlog_parse_sock(g_srvlog_sockpath, &uid, name, sizeof(name));
    return name;
}

/**
 * @brief Close (and optionally unlink) the active server SHM log.
 *
 * @param unlink_shm If non-zero, unlink the SHM file from `/dev/shm`
 */
void milk_server_close_log(int unlink_shm)
{
    if (g_srvlog_shm != NULL)
    {
        milk_srvlog_server_close(g_srvlog_shm, unlink_shm);
        g_srvlog_shm = NULL;
    }
}

typedef struct
{
    int    out_r;
    int    err_r;
    int    client_out;
    int    client_err;
    char   cap_buf[MILK_SRVLOG_OUT_MAX];
    size_t cap_len;
} tee_pump_ctx_t;

/* A CALL whose stdout is no terminal writes into pipes that one thread, for the server's life,
 * forwards to the client's stdout and stderr, keeping the start for the log. The pipes are the
 * CALL's own: a process its command left running cannot write into a later CALL's output. */
static tee_pump_ctx_t g_tee;         /* the CALL being captured */
static int            g_tee_go = -1; /* eventfd: +1 when a CALL starts, +1 when its command ends */
static int            g_tee_done = -1; /* eventfd: the pump forwarded the CALL's output */
static pid_t g_tee_pid = 0; /* the process the pump runs in (a forked child starts its own) */

/**
 * tee_drain_fd - Drain available bytes from a captured pipe into client stdio and the log buffer
 * @rfd:       Non-blocking read end of the server stdout/stderr capture pipe
 * @wfd:       Client destination file descriptor to forward output to
 * @ctx:       Tee pump context holding the ring-buffer log snippet accumulator
 * @open_flag: In/out flag cleared to 0 when @rfd reaches EOF
 *
 * Reads all currently available bytes from @rfd, forwards them completely to @wfd,
 * and appends up to `MILK_SRVLOG_OUT_MAX - 1` bytes into `ctx->cap_buf` for the
 * shared-memory command log.
 *
 * Return: void
 */
static void tee_drain_fd(int rfd, int wfd, tee_pump_ctx_t *ctx, int *open_flag)
{
    char buf[4096];
    for (;;)
    {
        ssize_t n = read(rfd, buf, sizeof(buf));
        if (n > 0)
        {
            if (wfd >= 0)
            {
                ssize_t off = 0;
                while (off < n)
                {
                    ssize_t w = write(wfd, buf + off, (size_t) (n - off));
                    if (w < 0)
                    {
                        if (errno == EINTR)
                        {
                            continue;
                        }
                        break;
                    }
                    off += w;
                }
            }
            if (ctx->cap_len + 1 < sizeof(ctx->cap_buf))
            {
                size_t avail = sizeof(ctx->cap_buf) - 1 - ctx->cap_len;
                size_t to_cp = ((size_t) n < avail) ? (size_t) n : avail;
                memcpy(ctx->cap_buf + ctx->cap_len, buf, to_cp);
                ctx->cap_len += to_cp;
                ctx->cap_buf[ctx->cap_len] = '\0';
            }
        }
        else if (n == 0)
        {
            *open_flag = 0;
            break;
        }
        else
        {
            if (errno == EINTR)
            {
                continue;
            }
            break;
        }
    }
}

/**
 * tee_pump - Forward a CALL's output pipes to its client while capturing the log snippet
 * @ctx:   The CALL
 * @ended: 1 if its command already ended (both g_tee_go signals came at once)
 *
 * Forwards until the command ends (g_tee_go: its pipes may reach their end before, when the server
 * puts its stdio back), then drains what is left.
 */
static void tee_pump(tee_pump_ctx_t *ctx, int ended)
{
    int out_open = 1;
    int err_open = 1;

    while (!ended)
    {
        struct pollfd pfds[3] = { { out_open ? ctx->out_r : -1, POLLIN, 0 },
                                  { err_open ? ctx->err_r : -1, POLLIN, 0 },
                                  { g_tee_go, POLLIN, 0 } };
        if (poll(pfds, 3, -1) < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            break;
        }
        if (pfds[0].revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL))
        {
            tee_drain_fd(ctx->out_r, ctx->client_out, ctx, &out_open);
            out_open = out_open && !(pfds[0].revents & POLLNVAL);
        }
        if (pfds[1].revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL))
        {
            tee_drain_fd(ctx->err_r, ctx->client_err, ctx, &err_open);
            err_open = err_open && !(pfds[1].revents & POLLNVAL);
        }
        if (pfds[2].revents & POLLIN)
        {
            uint64_t v;
            ended = (read(g_tee_go, &v, sizeof(v)) == (ssize_t) sizeof(v));
        }
    }
    if (out_open)
    {
        tee_drain_fd(ctx->out_r, ctx->client_out, ctx, &out_open);
    }
    if (err_open)
    {
        tee_drain_fd(ctx->err_r, ctx->client_err, ctx, &err_open);
    }
}

/** tee_main() - the pump thread: serves each captured CALL in turn (tee_pump()). */
static void *tee_main(void *arg)
{
    (void) arg;
    for (;;)
    {
        uint64_t v   = 0;
        uint64_t one = 1;
        if (read(g_tee_go, &v, sizeof(v)) != (ssize_t) sizeof(v))
        {
            continue;
        }
        tee_pump(&g_tee, v > 1);
        ssize_t w = write(g_tee_done, &one, sizeof(one));
        (void) w;
    }
    return NULL;
}

/** tee_start() - the pump thread of this process, started on first use; 0, or -1 if unavailable. */
static int tee_start(void)
{
    if (g_tee_pid == getpid())
    {
        return 0;
    }
    if (g_tee_go >= 0) /* a forked child: the parent's */
    {
        close(g_tee_go);
        close(g_tee_done);
    }
    g_tee_go   = eventfd(0, EFD_CLOEXEC);
    g_tee_done = eventfd(0, EFD_CLOEXEC);
    pthread_t tid;
    sigset_t  all, old; /* signals stay with the main thread (sig_handler may siglongjmp) */
    sigfillset(&all);
    pthread_sigmask(SIG_BLOCK, &all, &old);
    int ok = (g_tee_go >= 0 && g_tee_done >= 0 && pthread_create(&tid, NULL, tee_main, NULL) == 0);
    pthread_sigmask(SIG_SETMASK, &old, NULL);
    if (!ok)
    {
        return -1;
    }
    pthread_detach(tid);
    g_tee_pid = getpid();
    return 0;
}

/**
 * diff_timespec_ns - Compute the non-negative nanosecond difference between two timestamps
 * @t0: Start timestamp
 * @t1: End timestamp
 *
 * Calculates `(@t1 - @t0)` in nanoseconds, clamping negative differences to 0.
 *
 * Return: Elapsed duration in nanoseconds
 */
static uint64_t diff_timespec_ns(const struct timespec *t0, const struct timespec *t1)
{
    int64_t ds  = (int64_t) t1->tv_sec - (int64_t) t0->tv_sec;
    int64_t dns = (int64_t) t1->tv_nsec - (int64_t) t0->tv_nsec;
    int64_t tot = ds * 1000000000LL + dns;
    return (tot > 0) ? (uint64_t) tot : 0;
}

/** refresh_log_stats_soon() - refresh the SHM log header at most every 100 ms (per-request path). */
static void refresh_log_stats_soon(void)
{
    static struct timespec last = { 0, 0 };
    struct timespec        now;
    clock_gettime(CLOCK_MONOTONIC_COARSE, &now);
    if ((now.tv_sec - last.tv_sec) * 1000000000L + (now.tv_nsec - last.tv_nsec) >= 100000000L)
    {
        last = now;
        milk_server_refresh_log_stats();
    }
}

volatile sig_atomic_t g_srv_children = 0; /* a worker was forked and may need reaping */

/** reap_children() - collect exited workers; also every 64 requests for children of commands. */
static void reap_children(void)
{
    static unsigned int nreq = 0;
    if (!g_srv_children && (++nreq & 63U) != 0)
    {
        return;
    }
    pid_t r;
    while ((r = waitpid(0, NULL, WNOHANG)) > 0)
    {
    }
    if (r < 0 && errno == ECHILD)
    {
        g_srv_children = 0;
    }
}

/**
 * @brief Handle a fast, non-stdio protocol message (`PING`, `SHUTDOWN`, `COMPLETE`, `GET`, `EXISTS`).
 *
 * @return 1 if message was handled, 0 otherwise
 */
static int handle_fast_message(int                     client_fd,
                               int                     listen_fd,
                               int                     keep,
                               const milk_frame_hdr_t *hdr,
                               const uint8_t          *payload,
                               int                    *shutdown_flag)
{
    int32_t rc = 0;

    if (hdr->type == MILK_MSG_PING)
    {
        milk_server_send_done(client_fd, keep, &rc, sizeof(rc));
        return 1;
    }

    if (hdr->type == MILK_MSG_SHUTDOWN)
    {
        *shutdown_flag = 1;
        milk_server_stop_listening(listen_fd);
        milk_server_send_done(client_fd, 0, &rc, sizeof(rc));
        return 1;
    }

    if (hdr->type == MILK_MSG_COMPLETE)
    {
        uint32_t    off    = 0;
        const char *prefix = milk_pl_next(payload, hdr->len, &off, "");
        milk_server_handle_complete(client_fd, keep, prefix,
                                    milk_pl_next(payload, hdr->len, &off, "arg"));
        return 1;
    }

    if (hdr->type == MILK_MSG_GET)
    {
        char val[16384];
        if (payload != NULL && milk_shm_prop_get((const char *) payload, val, sizeof(val)) == 0)
        {
            milk_proto_send(client_fd, MILK_MSG_VALUE, 0, val, (uint32_t) (strlen(val) + 1));
            rc = 0;
        }
        else
        {
            rc = 1;
        }
        milk_server_send_done(client_fd, keep, &rc, sizeof(rc));
        return 1;
    }

    if (hdr->type == MILK_MSG_EXISTS)
    {
        if (payload != NULL && hdr->len >= 2)
        {
            int         check_run = (payload[0] == 'R') ? 1 : 0;
            const char *spec      = (const char *) (payload + 1);
            rc                    = milk_shm_prop_exists(spec, check_run) ? 0 : 1;
        }
        else
        {
            rc = 1;
        }
        milk_server_send_done(client_fd, keep, &rc, sizeof(rc));
        return 1;
    }

    return 0;
}

void milk_server_serve_fast(int cfd)
{
    milk_frame_hdr_t hdr;
    uint8_t         *payload = NULL;
    int              fds[4]  = { -1, -1, -1, -1 };
    int              dummy   = 0;

    if (milk_proto_recv(cfd, &hdr, &payload, fds) == 0)
    {
        mark_request();
        handle_fast_message(cfd, -1, 0, &hdr, payload, &dummy);
    }
    for (int i = 0; i < 4; i++)
    {
        if (fds[i] >= 0)
        {
            close(fds[i]);
        }
    }
    free(payload);
}

/**
 * @brief Handle `MILK_MSG_WAITFOR` or `MILK_MSG_GRAB` messages.
 *
 * @return 1 if handled, 0 otherwise
 */
static int handle_wait_or_grab_message(int                     client_fd,
                                       int                     keep,
                                       pid_t                   client_pid,
                                       const milk_frame_hdr_t *hdr,
                                       const uint8_t          *payload,
                                       const struct timespec  *ts_start)
{
    int32_t  rc  = 1;
    uint32_t len = hdr->len;
    uint32_t off = 0;
    char     log_cmd[MILK_SRVLOG_CMD_MAX];

    if (hdr->type != MILK_MSG_WAITFOR && hdr->type != MILK_MSG_GRAB)
    {
        return 0;
    }
    const char *name = milk_pl_next(payload, len, &off, NULL); /* the target, or the stream */
    snprintf(log_cmd, sizeof(log_cmd), "%s",
             (hdr->type == MILK_MSG_WAITFOR) ? "@waitfor" : "@grab");
    if (name != NULL && hdr->type == MILK_MSG_WAITFOR)
    {
        const char *arg2 = milk_pl_next(payload, len, &off, "");
        double      tsec = strtod(milk_pl_next(payload, len, &off, "0"), NULL);
        snprintf(log_cmd, sizeof(log_cmd), "@waitfor %s%s%s", name, arg2[0] ? " " : "", arg2);
        rc = milk_server_waitfor(client_fd, name, arg2, tsec);
    }
    else if (name != NULL)
    {
        long        nf      = strtol(milk_pl_next(payload, len, &off, "1"), NULL, 10);
        const char *outname = milk_pl_next(payload, len, &off, "");
        double      tsec    = strtod(milk_pl_next(payload, len, &off, "0"), NULL);
        const char *fits_p  = milk_pl_next(payload, len, &off, "");
        const char *ts_p    = milk_pl_next(payload, len, &off, "");
        snprintf(log_cmd, sizeof(log_cmd), "@grab %s %ld%s%s", name, nf, outname[0] ? " " : "",
                 outname);
        rc = milk_server_grab(client_fd, name, nf, outname, tsec, fits_p, ts_p);
    }
    if (g_srvlog_shm != NULL)
    {
        struct timespec ts_end;
        clock_gettime(CLOCK_REALTIME, &ts_end);
        milk_srvlog_record(g_srvlog_shm, ts_start, diff_timespec_ns(ts_start, &ts_end), client_pid,
                           hdr->type, rc, log_cmd, "");
    }
    milk_server_send_done(client_fd, keep, &rc, sizeof(rc));
    return 1;
}

/**
 * @brief Service the request on client connection @client_fd.
 *
 * @param client_fd     Connected client socket fd
 * @param listen_fd     Server listening socket fd (or -1 to handle everything inline)
 * @param shutdown_flag Set to 1 if client requested server shutdown
 * @param keep          1: the connection may stay open for the shell's next request
 * @return MILK_CONN_KEEP (the reply said so), MILK_CONN_TAKEN (a worker thread has @client_fd) or
 *         MILK_CONN_CLOSE (the caller closes it)
 */
int milk_server_handle_client(int client_fd, int listen_fd, int *shutdown_flag, int keep)
{
    milk_frame_hdr_t hdr;
    uint8_t         *payload     = NULL;
    int              recv_fds[4] = { -1, -1, -1, -1 };

    /* the request frame must arrive promptly: a stalled client must not block the server */
    struct timeval req_to = { MILK_SERVER_REQ_TIMEOUT_MS / 1000,
                              (MILK_SERVER_REQ_TIMEOUT_MS % 1000) * 1000 };
    setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &req_to, sizeof(req_to));
    if (listen_fd >= 0)
    {
        /* long waits run in a thread of this process: they see (and may create) live state */
        int ptype = peek_request_type(client_fd);
        if ((ptype == MILK_MSG_WAITFOR || ptype == MILK_MSG_GRAB) &&
            milk_server_spawn_stream_worker(client_fd) == 0)
        {
            return MILK_CONN_TAKEN;
        }
    }
    int rr = milk_proto_recv(client_fd, &hdr, &payload, recv_fds);
    if (rr == -2)
    {
        int32_t vrc = 255; // our version in the reply header tells the client what we speak
        milk_server_send_done(client_fd, 0, &vrc, sizeof(vrc));
    }
    if (rr != 0)
    {
        return MILK_CONN_CLOSE; // also the end of a kept connection: the shell closed it
    }
    if (hdr.type == MILK_MSG_CANCEL || hdr.type == MILK_MSG_WINCH) // sent as its request ended
    {
        free(payload);
        return keep ? MILK_CONN_KEEP : MILK_CONN_CLOSE;
    }
    if (hdr.type != MILK_MSG_PING && hdr.type != MILK_MSG_SHUTDOWN)
    {
        mark_request();
    }
    struct timeval no_to = { 0, 0 };
    setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &no_to, sizeof(no_to));

    pid_t        client_pid = getpid();
    struct ucred cred;
    socklen_t    clen = sizeof(cred);
    if (getsockopt(client_fd, SOL_SOCKET, SO_PEERCRED, &cred, &clen) == 0 && cred.pid > 0)
    {
        client_pid = cred.pid;
    }

    int is_replay_call = 0;
    if (hdr.type == MILK_MSG_CALL && payload != NULL && hdr.len > sizeof(uint32_t))
    {
        const char *cmd0 = (const char *) (payload + sizeof(uint32_t));
        if (cmd0[0] == '@')
        {
            cmd0++;
        }
        if (strcmp(cmd0, "replay") == 0)
        {
            is_replay_call = 1;
        }
    }

    /* --async and @replay need their own stdio while the server goes on: they run in a fork, on a
     * snapshot of the state (images and variables they create are not kept); a module loaded on
     * first use is loaded here, before the fork, so that the server keeps it */
    int is_forked_worker = 0;
    if ((hdr.flags & MILK_FLAG_ASYNC) && hdr.type == MILK_MSG_CALL && payload != NULL &&
        hdr.len > sizeof(uint32_t) && listen_fd >= 0)
    {
        const char *cmd0 = (const char *) (payload + sizeof(uint32_t));
        cmd0 += (cmd0[0] == '@');
        milk_server_cmd_lock();
        if (milk_cmd_lookup(cmd0) < 0)
        {
            milk_catalog_autoload(cmd0);
        }
        milk_server_cmd_unlock();
    }
    if (((hdr.flags & MILK_FLAG_ASYNC) || is_replay_call) && listen_fd >= 0)
    {
        pid_t pid = fork();
        if (pid > 0)
        {
            g_srv_children = 1;
            for (int i = 0; i < 4; i++)
            {
                if (recv_fds[i] >= 0)
                {
                    close(recv_fds[i]);
                }
            }
            free(payload);
            return MILK_CONN_CLOSE; // the child answers, and the connection ends with it
        }
        else if (pid == 0)
        {
            is_forked_worker = 1;
            keep             = 0;
            close(listen_fd);
            listen_fd = -1;
        }
    }

    milk_server_cmd_lock();
    int fast = handle_fast_message(client_fd, listen_fd, keep, &hdr, payload, shutdown_flag);
    keep     = keep && !*shutdown_flag;
    milk_server_cmd_unlock();
    if (fast)
    {
        goto cleanup;
    }

    struct timespec ts_start;
    clock_gettime(CLOCK_REALTIME, &ts_start);

    if (handle_wait_or_grab_message(client_fd, keep, client_pid, &hdr, payload, &ts_start))
    {
        goto cleanup;
    }

    /* Commands that may read/write stdio or cwd: SET, CALC, TEST, CALL */
    {
        milk_server_cmd_lock(); /* stdio/cwd are swapped process-wide: one command at a time */
        int saved_stdin  = -1;
        int saved_stdout = -1;
        int saved_stderr = -1;
        int saved_cwd    = -1;

        int is_streamcat = 0;
        if (hdr.type == MILK_MSG_CALL && payload != NULL && hdr.len > sizeof(uint32_t))
        {
            const char *first_arg = (const char *) (payload + sizeof(uint32_t));
            if (strcmp(first_arg, "streamcat") == 0 || strcmp(first_arg, "@streamcat") == 0)
            {
                is_streamcat = 1;
            }
        }

        // a terminal stays a terminal (TUIs, isatty checks): only capture output going elsewhere
        // only CALLs are captured for the log (calc/test/set output is small and on the hot path)
        int use_tee     = (g_srvlog_shm != NULL && hdr.type == MILK_MSG_CALL && !is_streamcat &&
                           !isatty(recv_fds[1]))
                              ? 1
                              : 0;
        int out_pipe[2] = { -1, -1 };
        int err_pipe[2] = { -1, -1 };
        int tee_running = 0;

        if (recv_fds[0] >= 0)
        {
            saved_stdin = fcntl(STDIN_FILENO, F_DUPFD_CLOEXEC, 3);
            dup2(recv_fds[0], STDIN_FILENO);
            close(recv_fds[0]);
            recv_fds[0] = -1;
        }

        if (use_tee && recv_fds[1] >= 0 && recv_fds[2] >= 0 && tee_start() == 0 &&
            pipe2(out_pipe, O_CLOEXEC | O_NONBLOCK) == 0 &&
            pipe2(err_pipe, O_CLOEXEC | O_NONBLOCK) == 0)
        {
            fcntl(out_pipe[1], F_SETFL, 0);
            fcntl(err_pipe[1], F_SETFL, 0);

            struct stat st;
            if (fstat(recv_fds[1], &st) == 0 && S_ISFIFO(st.st_mode))
            {
                fcntl(recv_fds[1], F_SETPIPE_SZ, 1048576);
            }

            g_tee.out_r      = out_pipe[0];
            g_tee.err_r      = err_pipe[0];
            g_tee.client_out = recv_fds[1];
            g_tee.client_err = recv_fds[2];
            g_tee.cap_len    = 0;
            g_tee.cap_buf[0] = '\0';
            recv_fds[1]      = -1;
            recv_fds[2]      = -1;

            saved_stdout = fcntl(STDOUT_FILENO, F_DUPFD_CLOEXEC, 3);
            saved_stderr = fcntl(STDERR_FILENO, F_DUPFD_CLOEXEC, 3);
            dup2(out_pipe[1], STDOUT_FILENO);
            dup2(err_pipe[1], STDERR_FILENO);
            close(out_pipe[1]);
            close(err_pipe[1]);
            out_pipe[1] = -1;
            err_pipe[1] = -1;

            uint64_t one = 1; /* the pump forwards this CALL's output now */
            tee_running  = (write(g_tee_go, &one, sizeof(one)) == (ssize_t) sizeof(one));
        }
        else
        {
            if (out_pipe[0] >= 0)
            {
                close(out_pipe[0]);
                close(out_pipe[1]);
            }
            if (err_pipe[0] >= 0)
            {
                close(err_pipe[0]);
                close(err_pipe[1]);
            }
            if (recv_fds[1] >= 0)
            {
                struct stat st;
                if (fstat(recv_fds[1], &st) == 0 && S_ISFIFO(st.st_mode))
                {
                    fcntl(recv_fds[1], F_SETPIPE_SZ, 1048576);
                }
                saved_stdout = fcntl(STDOUT_FILENO, F_DUPFD_CLOEXEC, 3);
                dup2(recv_fds[1], STDOUT_FILENO);
                close(recv_fds[1]);
                recv_fds[1] = -1;
            }
            if (recv_fds[2] >= 0)
            {
                saved_stderr = fcntl(STDERR_FILENO, F_DUPFD_CLOEXEC, 3);
                dup2(recv_fds[2], STDERR_FILENO);
                close(recv_fds[2]);
                recv_fds[2] = -1;
            }
        }

        if (recv_fds[3] >= 0)
        {
            struct stat cst, ost;
            if (fstat(recv_fds[3], &cst) != 0 || stat(".", &ost) != 0 || cst.st_dev != ost.st_dev ||
                cst.st_ino != ost.st_ino) /* usually the client is in our directory already */
            {
                saved_cwd = open(".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
                if (fchdir(recv_fds[3]) != 0)
                {
                }
            }
            close(recv_fds[3]);
            recv_fds[3] = -1;
        }

        char    log_cmd[MILK_SRVLOG_CMD_MAX] = "";
        int32_t rc                           = 0;

        if (hdr.type == MILK_MSG_SET)
        {
            rc = milk_server_exec_set(payload, hdr.len, log_cmd, sizeof(log_cmd));
        }
        else if (hdr.type == MILK_MSG_CALC || hdr.type == MILK_MSG_TEST)
        {
            rc = milk_server_exec_calc_or_test(client_fd, listen_fd,
                                               (hdr.type == MILK_MSG_TEST) ? 1 : 0, hdr.flags,
                                               payload, hdr.len, log_cmd, sizeof(log_cmd));
        }
        else if (hdr.type == MILK_MSG_CALL)
        {
            rc = milk_server_exec_call(client_fd, listen_fd, payload, hdr.len, log_cmd,
                                       sizeof(log_cmd));
        }

        fflush(stdout);
        fflush(stderr);

        if (saved_stdin >= 0)
        {
            dup2(saved_stdin, STDIN_FILENO);
            close(saved_stdin);
        }
        if (saved_stdout >= 0)
        {
            dup2(saved_stdout, STDOUT_FILENO);
            close(saved_stdout);
        }
        if (saved_stderr >= 0)
        {
            dup2(saved_stderr, STDERR_FILENO);
            close(saved_stderr);
        }

        if (tee_running) /* the command ended: the pump drains its pipes, then stops */
        {
            uint64_t one = 1;
            uint64_t v;
            ssize_t  w = write(g_tee_go, &one, sizeof(one));
            (void) w;
            while (read(g_tee_done, &v, sizeof(v)) < 0 && errno == EINTR)
            {
            }
            close(g_tee.out_r);
            close(g_tee.err_r);
            close(g_tee.client_out);
            close(g_tee.client_err);
        }

        refresh_log_stats_soon();

        if (saved_cwd >= 0)
        {
            if (fchdir(saved_cwd) != 0)
            {
            }
            close(saved_cwd);
        }

        if (g_srvlog_shm != NULL && log_cmd[0] != '\0' && strcmp(log_cmd, "log") != 0 &&
            strncmp(log_cmd, "log ", 4) != 0 && strcmp(log_cmd, "session") != 0 &&
            strncmp(log_cmd, "session ", 8) != 0)
        {
            struct timespec ts_end;
            clock_gettime(CLOCK_REALTIME, &ts_end);
            milk_srvlog_record(g_srvlog_shm, &ts_start, diff_timespec_ns(&ts_start, &ts_end),
                               client_pid, hdr.type, rc, log_cmd, tee_running ? g_tee.cap_buf : "");
        }
        milk_server_cmd_unlock();

        /* a CALL also reports the registry generation: shells resync their functions on change; the
         * connection of a request that crashed ends with it (its frames may be cut) */
        int32_t done[2]       = { rc, (int32_t) milk_registry_gen() };
        keep                  = keep && !g_milk_server_crashed;
        g_milk_server_crashed = 0;
        milk_server_send_done(client_fd, keep, done,
                              (hdr.type == MILK_MSG_CALL) ? sizeof(done) : sizeof(rc));
    } // SET / CALC / TEST / CALL block

cleanup:
    for (int i = 0; i < 4; i++)
    {
        if (recv_fds[i] >= 0)
        {
            close(recv_fds[i]);
        }
    }
    free(payload);

    if (is_forked_worker)
    {
        close(client_fd);
        _exit(0);
    }
    return keep ? MILK_CONN_KEEP : MILK_CONN_CLOSE;
}

/**
 * @brief Create, bind, and listen on the milk UNIX domain socket.
 *
 * @param sockpath Socket path (`"@abstract"` or `"/path/to/sock"`)
 * @return Listening socket fd (>= 0) on success, -1 on error
 */
int milk_server_create_listener(const char *sockpath)
{
    pthread_once(&g_atfork_once, register_atfork_handler);

    if (sockpath == NULL || sockpath[0] == '\0')
    {
        fprintf(stderr, "ERROR: milk-cli --server requires a socket path\n");
        errno = EINVAL;
        return -1;
    }

    struct sockaddr_un addr;
    socklen_t          addrlen = 0;
    if (milk_sockaddr_init(sockpath, &addr, &addrlen) != 0)
    {
        fprintf(stderr, "ERROR: invalid socket path '%s'\n", sockpath);
        errno = EINVAL;
        return -1;
    }

    if (sockpath[0] != '@')
    {
        unlink(sockpath);
    }

    int listen_fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (listen_fd < 0)
    {
        perror("milk-cli --server: socket");
        return -1;
    }

    if (bind(listen_fd, (struct sockaddr *) &addr, addrlen) != 0)
    {
        int err = errno; // the reason goes back to the starter (milk_server_run())
        perror("milk-cli --server: bind");
        close(listen_fd);
        errno = err;
        return -1;
    }

    if (sockpath[0] != '@')
    {
        chmod(sockpath, 0600);
    }

    if (listen(listen_fd, 64) != 0)
    {
        int err = errno;
        perror("milk-cli --server: listen");
        close(listen_fd);
        errno = err;
        return -1;
    }

    setenv("MILKCLI_SOCK", sockpath, 1);

    snprintf(g_srvlog_sockpath, sizeof(g_srvlog_sockpath), "%s", sockpath);
    uint32_t srv_flags = 0;
    if (getenv("MILK_SRV_NAMED") != NULL)
    {
        srv_flags |= MILK_SRVFLAG_NAMED;
    }
    if (g_srvlog_shm != NULL)
    {
        milk_srvlog_server_close(g_srvlog_shm, 0);
    }
    g_srvlog_shm = milk_srvlog_server_init(sockpath, 0, srv_flags);
    milk_server_refresh_log_stats();

    if (g_epfd < 0)
    {
        g_epfd = epoll_create1(EPOLL_CLOEXEC);
    }
    if (g_epfd < 0 || conn_arm(listen_fd, 0) != 0)
    {
        perror("milk-cli --server: epoll");
        close(listen_fd);
        return -1;
    }
    return listen_fd;
}

/**
 * @brief Accept and service one pending connection on @listen_fd.
 *
 * @param listen_fd     Listening UNIX domain socket fd
 * @param shutdown_flag Set to 1 if client requested server shutdown
 */
void milk_server_accept_and_handle(int listen_fd, int *shutdown_flag)
{
    int dummy = 0;
    int fds[16];
    shutdown_flag = (shutdown_flag != NULL) ? shutdown_flag : &dummy;
    reap_children();
    int n = milk_server_take_ready(listen_fd, fds, 16);
    for (int i = 0; i < n; i++)
    {
        int keep = milk_server_can_keep(fds[i]);
        milk_server_conn_done(fds[i],
                              milk_server_handle_client(fds[i], listen_fd, shutdown_flag, keep));
        milk_server_serve_waiting(listen_fd, shutdown_flag);
    }
}

/* SIGTERM, a service manager's stop (and the second step of @session end's ladder): the loop ends
 * once the command running, if any, returns */
static void milk_server_on_term(int signo)
{
    (void) signo;
    g_milk_server_stop = 1;
}

/* Seconds since the session was last used: a request, or an entry of its log (a @set that a shell
 * did itself writes there); 0 while a stream request (@waitfor, @grab) runs */
static double idle_for(void)
{
    struct timespec now;
    struct timespec real;

    if (__atomic_load_n(&g_stream_workers, __ATOMIC_ACQUIRE) > 0)
    {
        return 0.0;
    }
    clock_gettime(CLOCK_MONOTONIC, &now);
    double idle = 1e-9 * (double) ((uint64_t) now.tv_sec * 1000000000ULL + (uint64_t) now.tv_nsec -
                                   __atomic_load_n(&g_last_request_ns, __ATOMIC_ACQUIRE));
    uint64_t seq =
        (g_srvlog_shm != NULL) ? __atomic_load_n(&g_srvlog_shm->write_seq, __ATOMIC_ACQUIRE) : 0;
    if (seq > 0)
    {
        const milk_srvlog_entry_t *e = &g_srvlog_shm->entries[(seq - 1) % MILK_SRVLOG_CAP];
        clock_gettime(CLOCK_REALTIME, &real);
        double since = (double) (real.tv_sec - e->ts_start.tv_sec) +
                       1e-9 * (double) (real.tv_nsec - e->ts_start.tv_nsec) -
                       1e-9 * (double) e->duration_ns;
        idle         = (since < idle) ? since : idle;
    }
    return idle;
}

/**
 * @brief Run the milk UNIX domain socket command server loop.
 *
 * @param sockpath  Socket path (`"@abstract"` or `"/path/to/sock"`)
 * @param owner_fd  Optional fd held by owning shell (-1 if none)
 * @param ready_fd  Optional ready pipe (milk_detach.h): `READY=1` once listening, else `ERRNO=` (-1 if none)
 * @param owner_pid The owner to record, when it is not the session's name (0: none)
 * @param idle_s    End after this many seconds without a command (0: never)
 * @return 0 on clean shutdown, non-zero on error
 */
int milk_server_run(const char *sockpath,
                    int         owner_fd,
                    int         ready_fd,
                    pid_t       owner_pid,
                    unsigned    idle_s)
{
    char msg[600];
    // a service manager that waits for readiness (systemd's Type=notify) is told the same
    int svc = milknotify_open();

    errno         = 0;
    int listen_fd = milk_server_create_listener(sockpath);
    if (listen_fd < 0)
    {
        snprintf(msg, sizeof(msg), "ERRNO=%d\nSTATUS=cannot listen on %s\n",
                 (errno != 0) ? errno : EINVAL, sockpath);
        milknotify_send(ready_fd, msg);
        milknotify_send(svc, msg);
        return 1;
    }

    if (g_srvlog_shm != NULL && owner_pid > 0)
    {
        milk_srvlog_set_owner(g_srvlog_shm, owner_pid);
    }
    if (g_srvlog_shm != NULL)
    {
        g_srvlog_shm->idle_s = idle_s;
    }
    mark_request();
    uint64_t next_scan_ns = 0; // once idle, when to look again for shells that use the session

    snprintf(msg, sizeof(msg), "MAINPID=%d\nREADY=1\n", (int) getpid());
    if (ready_fd >= 0)
    {
        milknotify_send(ready_fd, msg);
        close(ready_fd);
    }
    snprintf(msg, sizeof(msg), "READY=1\nSTATUS=listening on %s\n", sockpath);
    milknotify_send(svc, msg);

    struct sigaction term;
    memset(&term, 0, sizeof(term));
    term.sa_handler = milk_server_on_term; // no SA_RESTART: poll() returns, the loop sees the flag
    sigemptyset(&term.sa_mask);
    sigaction(SIGTERM, &term, NULL);

    int shutdown_flag = 0;
    while (!shutdown_flag && !g_milk_server_stop)
    {
        struct pollfd pfds[2];
        int           npfds = 1;
        pfds[0].fd          = g_epfd; // the listening socket and the kept connections
        pfds[0].events      = POLLIN;
        pfds[0].revents     = 0;

        if (owner_fd >= 0)
        {
            pfds[1].fd      = owner_fd;
            pfds[1].events  = POLLIN | POLLHUP | POLLERR;
            pfds[1].revents = 0;
            npfds           = 2;
        }

        int pr = poll(pfds, (nfds_t) npfds, 1000);
        if (pr < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            break;
        }

        if (npfds == 2 && (pfds[1].revents & (POLLHUP | POLLERR | POLLIN)))
        {
            struct pollfd drain_pfd = { g_epfd, POLLIN, 0 };
            if (poll(&drain_pfd, 1, 50) <= 0)
            {
                break;
            }
        }

        if (pfds[0].revents & POLLIN)
        {
            milk_server_accept_and_handle(listen_fd, &shutdown_flag);
        }
        if (idle_s > 0 && !shutdown_flag && idle_for() >= (double) idle_s)
        {
            struct timespec t;
            clock_gettime(CLOCK_MONOTONIC, &t);
            uint64_t now = (uint64_t) t.tv_sec * 1000000000ULL + (uint64_t) t.tv_nsec;
            if (now < next_scan_ns)
            {
                continue; // a shell used it at the last look
            }
            if (milk_srvlog_session_shells(sockpath) > 0)
            {
                next_scan_ns = now + 1000000000ULL * ((idle_s < 5) ? idle_s : 5);
                continue;
            }
            fprintf(stderr,
                    "milk: session %s ends: no command for %u s, and no shell uses it (--idle)\n",
                    sockpath, idle_s);
            break;
        }
    }

    milknotify_send(svc, "STOPPING=1\n");
    if (svc >= 0)
    {
        close(svc);
    }
    milk_server_close_log(0);

    close(listen_fd);
    if (sockpath[0] != '@' && !g_listen_released)
    {
        unlink(sockpath);
    }
    return 0;
}
