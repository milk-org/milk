// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_server.c
 * @brief Milk UNIX domain socket listener, connection handler, and SHM log tee pump.
 *
 * Role & Goal:
 *   Implements the UNIX domain socket server event loop (`milk_server_run`),
 *   listener socket creation (`milk_server_create_listener`), per-connection
 *   framing and SCM_RIGHTS stdio/cwd redirection (`milk_server_handle_client`),
 *   and the background tee pump thread that forwards command stdout/stderr to
 *   the client while simultaneously capturing a snippet for the `/dev/shm`
 *   server ring-buffer log.
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
#include "libmilkcommon/milk_compiler.h"
#include "milk_shm_prop.h"
#include "milk_ipc_common.h"
#include "milk_proto.h"
#include "milk_catalog.h"
#include "milk_cmd_index.h"
#include "milk_detach.h"
#include "milk_notify.h"
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

/**
 * milk_child_reset_omp - Reset OpenMP thread pool and mutex state in forked children
 *
 * Invoked as a `pthread_atfork` child handler so forked `--async` and `@replay`
 * worker processes reset OpenMP to a single thread, reinitialize `g_cmd_mutex`
 * in an unlocked state, and clear inherited watcher state.
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
}

/** stream_worker() - serve one WAITFOR/GRAB request inline, then close its connection. */
static void *stream_worker(void *arg)
{
    int fd    = (int) (intptr_t) arg;
    int dummy = 0;
    milk_server_handle_client(fd, -1, &dummy);
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
    if (data.core.image != NULL)
    {
        for (long i = 0; i < data.core.NB_MAX_IMAGE; i++)
        {
            if (data.core.image[i].used)
            {
                nb_images++;
            }
        }
    }

    uint32_t nb_vars = 0;
    if (data.core.variable != NULL)
    {
        for (long i = 0; i < data.core.NB_MAX_VARIABLE; i++)
        {
            if (data.core.variable[i].used)
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
    int    stop_r;
    int    client_out;
    int    client_err;
    char   cap_buf[MILK_SRVLOG_OUT_MAX];
    size_t cap_len;
} tee_pump_ctx_t;

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
 * @brief Pump stdout/stderr pipes to client fds while capturing output for SHM logging.
 */
static void *tee_pump_thread(void *arg)
{
    tee_pump_ctx_t *ctx      = (tee_pump_ctx_t *) arg;
    int             out_open = (ctx->out_r >= 0) ? 1 : 0;
    int             err_open = (ctx->err_r >= 0) ? 1 : 0;

    while (out_open || err_open)
    {
        struct pollfd pfds[3];
        int           npfds   = 0;
        int           out_idx = -1;
        int           err_idx = -1;
        int           stp_idx = -1;

        if (out_open)
        {
            out_idx               = npfds++;
            pfds[out_idx].fd      = ctx->out_r;
            pfds[out_idx].events  = POLLIN;
            pfds[out_idx].revents = 0;
        }
        if (err_open)
        {
            err_idx               = npfds++;
            pfds[err_idx].fd      = ctx->err_r;
            pfds[err_idx].events  = POLLIN;
            pfds[err_idx].revents = 0;
        }
        if (ctx->stop_r >= 0)
        {
            stp_idx               = npfds++;
            pfds[stp_idx].fd      = ctx->stop_r;
            pfds[stp_idx].events  = POLLIN | POLLHUP;
            pfds[stp_idx].revents = 0;
        }

        int pr = poll(pfds, (nfds_t) npfds, -1);
        if (pr < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            break;
        }

        if (out_idx >= 0 && (pfds[out_idx].revents & (POLLIN | POLLHUP | POLLERR)))
        {
            tee_drain_fd(ctx->out_r, ctx->client_out, ctx, &out_open);
        }
        if (err_idx >= 0 && (pfds[err_idx].revents & (POLLIN | POLLHUP | POLLERR)))
        {
            tee_drain_fd(ctx->err_r, ctx->client_err, ctx, &err_open);
        }
        if (stp_idx >= 0 && (pfds[stp_idx].revents & (POLLIN | POLLHUP | POLLERR)))
        {
            if (out_open)
            {
                tee_drain_fd(ctx->out_r, ctx->client_out, ctx, &out_open);
            }
            if (err_open)
            {
                tee_drain_fd(ctx->err_r, ctx->client_err, ctx, &err_open);
            }
            break;
        }
    }

    return NULL;
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
                               const milk_frame_hdr_t *hdr,
                               const uint8_t          *payload,
                               int                    *shutdown_flag)
{
    int32_t rc = 0;

    if (hdr->type == MILK_MSG_PING)
    {
        milk_proto_send(client_fd, MILK_MSG_DONE, 0, &rc, sizeof(rc));
        return 1;
    }

    if (hdr->type == MILK_MSG_SHUTDOWN)
    {
        *shutdown_flag = 1;
        milk_proto_send(client_fd, MILK_MSG_DONE, 0, &rc, sizeof(rc));
        return 1;
    }

    if (hdr->type == MILK_MSG_COMPLETE)
    {
        const char *prefix = (payload != NULL) ? (const char *) payload : "";
        size_t      plen   = strlen(prefix) + 1;
        const char *comp_type =
            (payload != NULL && plen < hdr->len) ? (const char *) (payload + plen) : "arg";
        milk_server_handle_complete(client_fd, prefix, comp_type);
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
        milk_proto_send(client_fd, MILK_MSG_DONE, 0, &rc, sizeof(rc));
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
        milk_proto_send(client_fd, MILK_MSG_DONE, 0, &rc, sizeof(rc));
        return 1;
    }

    return 0;
}

/**
 * @brief Handle `MILK_MSG_WAITFOR` or `MILK_MSG_GRAB` messages.
 *
 * @return 1 if handled, 0 otherwise
 */
static int handle_wait_or_grab_message(int                     client_fd,
                                       pid_t                   client_pid,
                                       const milk_frame_hdr_t *hdr,
                                       const uint8_t          *payload,
                                       const struct timespec  *ts_start)
{
    int32_t rc = 0;

    if (hdr->type == MILK_MSG_WAITFOR)
    {
        char log_cmd[MILK_SRVLOG_CMD_MAX] = "@waitfor";
        if (payload != NULL && hdr->len > 0)
        {
            const char *target = (const char *) payload;
            size_t      o1     = strlen(target) + 1;
            const char *arg2   = (o1 < hdr->len) ? (const char *) (payload + o1) : "";
            size_t      o2     = o1 + strlen(arg2) + 1;
            const char *tstr   = (o2 < hdr->len) ? (const char *) (payload + o2) : "0";
            double      tsec   = strtod(tstr, NULL);
            snprintf(log_cmd, sizeof(log_cmd), "@waitfor %s%s%s", target, arg2[0] ? " " : "", arg2);
            rc = milk_server_waitfor(client_fd, target, arg2, tsec);
        }
        else
        {
            rc = 1;
        }
        if (g_srvlog_shm != NULL)
        {
            struct timespec ts_end;
            clock_gettime(CLOCK_REALTIME, &ts_end);
            milk_srvlog_record(g_srvlog_shm, ts_start, diff_timespec_ns(ts_start, &ts_end),
                               client_pid, hdr->type, rc, log_cmd, "");
        }
        milk_proto_send(client_fd, MILK_MSG_DONE, 0, &rc, sizeof(rc));
        return 1;
    }

    if (hdr->type == MILK_MSG_GRAB)
    {
        char log_cmd[MILK_SRVLOG_CMD_MAX] = "@grab";
        if (payload != NULL && hdr->len > 0)
        {
            const char *sname   = (const char *) payload;
            size_t      o1      = strlen(sname) + 1;
            const char *nfstr   = (o1 < hdr->len) ? (const char *) (payload + o1) : "1";
            size_t      o2      = o1 + strlen(nfstr) + 1;
            const char *outname = (o2 < hdr->len) ? (const char *) (payload + o2) : "";
            size_t      o3      = o2 + strlen(outname) + 1;
            const char *tstr    = (o3 < hdr->len) ? (const char *) (payload + o3) : "0";
            size_t      o4      = o3 + strlen(tstr) + 1;
            const char *fits_p  = (o4 < hdr->len) ? (const char *) (payload + o4) : "";
            size_t      o5      = o4 + strlen(fits_p) + 1;
            const char *ts_p    = (o5 < hdr->len) ? (const char *) (payload + o5) : "";
            long        nf      = strtol(nfstr, NULL, 10);
            double      tsec    = strtod(tstr, NULL);
            snprintf(log_cmd, sizeof(log_cmd), "@grab %s %ld%s%s", sname, nf, outname[0] ? " " : "",
                     outname);
            rc = milk_server_grab(client_fd, sname, nf, outname, tsec, fits_p, ts_p);
        }
        else
        {
            rc = 1;
        }
        if (g_srvlog_shm != NULL)
        {
            struct timespec ts_end;
            clock_gettime(CLOCK_REALTIME, &ts_end);
            milk_srvlog_record(g_srvlog_shm, ts_start, diff_timespec_ns(ts_start, &ts_end),
                               client_pid, hdr->type, rc, log_cmd, "");
        }
        milk_proto_send(client_fd, MILK_MSG_DONE, 0, &rc, sizeof(rc));
        return 1;
    }

    return 0;
}

/**
 * @brief Service a single client connection on @client_fd.
 *
 * @param client_fd     Connected client socket fd
 * @param listen_fd     Server listening socket fd (or -1 to handle everything inline)
 * @param shutdown_flag Set to 1 if client requested server shutdown
 * @return 1 if @client_fd now belongs to a worker thread (the caller must not close it), else 0
 */
int milk_server_handle_client(int client_fd, int listen_fd, int *shutdown_flag)
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
            return 1;
        }
    }
    int rr = milk_proto_recv(client_fd, &hdr, &payload, recv_fds);
    if (rr == -2)
    {
        int32_t vrc = 255; // our version in the reply header tells the client what we speak
        milk_proto_send(client_fd, MILK_MSG_DONE, 0, &vrc, sizeof(vrc));
    }
    if (rr != 0)
    {
        return 0;
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
            return 0;
        }
        else if (pid == 0)
        {
            is_forked_worker = 1;
            close(listen_fd);
            listen_fd = -1;
        }
    }

    milk_server_cmd_lock();
    int fast = handle_fast_message(client_fd, &hdr, payload, shutdown_flag);
    milk_server_cmd_unlock();
    if (fast)
    {
        goto cleanup;
    }

    struct timespec ts_start;
    clock_gettime(CLOCK_REALTIME, &ts_start);

    if (handle_wait_or_grab_message(client_fd, client_pid, &hdr, payload, &ts_start))
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
        int use_tee      = (g_srvlog_shm != NULL && hdr.type == MILK_MSG_CALL && !is_streamcat &&
                            !isatty(recv_fds[1]))
                               ? 1
                               : 0;
        int out_pipe[2]  = { -1, -1 };
        int err_pipe[2]  = { -1, -1 };
        int stop_pipe[2] = { -1, -1 };
        tee_pump_ctx_t tee_ctx;
        pthread_t      tee_tid;
        int            tee_running = 0;

        memset(&tee_ctx, 0, sizeof(tee_ctx));
        tee_ctx.out_r      = -1;
        tee_ctx.err_r      = -1;
        tee_ctx.stop_r     = -1;
        tee_ctx.client_out = -1;
        tee_ctx.client_err = -1;

        if (recv_fds[0] >= 0)
        {
            saved_stdin = fcntl(STDIN_FILENO, F_DUPFD_CLOEXEC, 3);
            dup2(recv_fds[0], STDIN_FILENO);
            close(recv_fds[0]);
            recv_fds[0] = -1;
        }

        if (use_tee && recv_fds[1] >= 0 && recv_fds[2] >= 0 &&
            pipe2(out_pipe, O_CLOEXEC | O_NONBLOCK) == 0 &&
            pipe2(err_pipe, O_CLOEXEC | O_NONBLOCK) == 0 &&
            pipe2(stop_pipe, O_CLOEXEC | O_NONBLOCK) == 0)
        {
            fcntl(out_pipe[1], F_SETFL, 0);
            fcntl(err_pipe[1], F_SETFL, 0);

            struct stat st;
            if (fstat(recv_fds[1], &st) == 0 && S_ISFIFO(st.st_mode))
            {
                fcntl(recv_fds[1], F_SETPIPE_SZ, 1048576);
            }

            tee_ctx.out_r      = out_pipe[0];
            tee_ctx.err_r      = err_pipe[0];
            tee_ctx.stop_r     = stop_pipe[0];
            tee_ctx.client_out = recv_fds[1];
            tee_ctx.client_err = recv_fds[2];
            recv_fds[1]        = -1;
            recv_fds[2]        = -1;

            saved_stdout = fcntl(STDOUT_FILENO, F_DUPFD_CLOEXEC, 3);
            saved_stderr = fcntl(STDERR_FILENO, F_DUPFD_CLOEXEC, 3);
            dup2(out_pipe[1], STDOUT_FILENO);
            dup2(err_pipe[1], STDERR_FILENO);
            close(out_pipe[1]);
            close(err_pipe[1]);
            out_pipe[1] = -1;
            err_pipe[1] = -1;

            sigset_t all, old; /* signals stay with the main thread (sig_handler may siglongjmp) */
            sigfillset(&all);
            pthread_sigmask(SIG_BLOCK, &all, &old);
            if (pthread_create(&tee_tid, NULL, tee_pump_thread, &tee_ctx) == 0)
            {
                tee_running = 1;
            }
            pthread_sigmask(SIG_SETMASK, &old, NULL);
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
            if (stop_pipe[0] >= 0)
            {
                close(stop_pipe[0]);
                close(stop_pipe[1]);
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
            rc = milk_server_exec_calc_or_test(client_fd, (hdr.type == MILK_MSG_TEST) ? 1 : 0,
                                               hdr.flags, payload, hdr.len, log_cmd,
                                               sizeof(log_cmd));
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

        if (tee_running)
        {
            close(stop_pipe[1]);
            stop_pipe[1] = -1;
            pthread_join(tee_tid, NULL);
            close(out_pipe[0]);
            close(err_pipe[0]);
            close(stop_pipe[0]);
            close(tee_ctx.client_out);
            close(tee_ctx.client_err);
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
                               client_pid, hdr.type, rc, log_cmd, tee_ctx.cap_buf);
        }
        milk_server_cmd_unlock();

        /* a CALL also reports the registry generation: shells resync their functions on change */
        int32_t done[2] = { rc, (int32_t) milk_registry_gen() };
        milk_proto_send(client_fd, MILK_MSG_DONE, 0, done,
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
    return 0;
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
    milk_server_init_dcvar_cache();

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
    reap_children();

    int client_fd = accept4(listen_fd, NULL, NULL, SOCK_CLOEXEC);
    if (client_fd < 0)
    {
        return;
    }

    if (!milk_ipc_peer_is_self(client_fd))
    {
        close(client_fd);
        return;
    }

    int dummy = 0;
    if (!milk_server_handle_client(client_fd, listen_fd,
                                   shutdown_flag != NULL ? shutdown_flag : &dummy))
    {
        close(client_fd);
    }
    milk_server_serve_waiting(listen_fd, shutdown_flag != NULL ? shutdown_flag : &dummy);
}

/* SIGTERM, a service manager's stop (and the second step of @session end's ladder): the loop ends
 * once the command running, if any, returns */
static void milk_server_on_term(int signo)
{
    (void) signo;
    g_milk_server_stop = 1;
}

/* Seconds since the session was last used: a request, or an entry of its log (the clients' fast @get
 * and @set write there); 0 while a stream request (@waitfor, @grab) runs */
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
        milkdetach_notify(ready_fd, msg);
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
        milkdetach_notify(ready_fd, msg);
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
        pfds[0].fd          = listen_fd;
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
            struct pollfd drain_pfd = { listen_fd, POLLIN, 0 };
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
    if (sockpath[0] != '@')
    {
        unlink(sockpath);
    }
    return 0;
}
