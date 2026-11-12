// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_client_conn.c
 * @brief The client's connection to its session: kept connections, starting the shell's own
 *        session, and the reply loop.
 *
 * Purpose:     a milk command costs one request on a connection the process already holds, and
 *              a shell's own session starts by itself on the shell's first command.
 * Does:        keeps a process's connections (to the shell's session and to its front end's)
 *              open between requests while the replies say MILK_FLAG_KEEP; starts the shell's
 *              own session (`milk-cli -q --server`) when none answers, tied to the shell by an
 *              owner pipe, or by its pid from a subshell or milk-call; never starts a named
 *              session; packs the shell variables a calculator expression names, and
 *              MILK_DPDIGITS; sends every request (milk_client_request()) and reads its replies
 *              until DONE, forwarding Ctrl-C (CANCEL) and window size changes (WINCH); defines the functions of new module commands when
 *              the session's registry generation changes.
 * Used by:     milk_client_ops.c and milk_client_stream.c (every request); milk_bash_hooks.c.
 *              Declared in milk_client_internal.h.
 * Uses:        milk_proto.h, milk_ipc_common.h (the socket of the shell's session, connecting,
 *              starting a server), milk_srvlog.h (session names).
 * See also:    milk_server.c (the session's side); docs/guide/sessions.md, "Using milk from Plain
 *              Bash" and "Concurrent Commands Across Shells".
 * Constraints: also compiled into libmilkbash.so, the thin client every shell loads. A forked
 *              subshell opens connections of its own: two processes must not share one.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include "milk_client_internal.h"
#include "milk_ipc_common.h"
#include "milk_proto.h"
#include "milk_srvlog.h"


static pid_t g_owner_shell_pid = 0;
static int   g_owner_pipe_wfd  = -1;

static volatile sig_atomic_t g_client_sigint   = 0;
static volatile sig_atomic_t g_client_sigwinch = 0;

static uint32_t g_seen_gen = 0; /* registry generation of the server's last CALL reply */

/**
 * client_sigint_handler - Async-signal-safe SIGINT handler for active client calls
 * @sig: Signal number (`SIGINT`)
 *
 * Sets `g_client_sigint` so the response poll loop in `await_response`
 * can forward `MILK_MSG_CANCEL` to the server instead of terminating the client
 * process immediately.
 *
 * Return: void
 */
static void client_sigint_handler(int sig)
{
    (void) sig;
    g_client_sigint = 1;
}

/**
 * client_sigwinch_handler - Async-signal-safe SIGWINCH handler for active client calls
 * @sig: Signal number (`SIGWINCH`)
 *
 * Sets `g_client_sigwinch` so the response loop notifies the server of terminal
 * window size changes via `MILK_MSG_WINCH`.
 *
 * Return: void
 */
static void client_sigwinch_handler(int sig)
{
    (void) sig;
    g_client_sigwinch = 1;
}

/**
 * @brief Lazily spawn `milk-cli -q --server` for @sockpath if not yet running.
 *
 * @param hooks    Optional shell callback hooks
 * @param sockpath UNIX socket path to bind
 * @return 0 once server is ready, -1 on error
 */
static int autostart_server(const milk_client_hooks_t *hooks, const char *sockpath)
{
    char sname[64] = "";
    milk_srvlog_parse_sock(sockpath, NULL, sname, sizeof(sname));
    int is_private = (sname[0] == '\0' || milk_ipc_is_numeric_str(sname));

    int   owner_rfd = -1;
    pid_t owner_pid = 0; // daemon lifetime tied to this pid when no owner pipe can be held
    if (is_private && hooks != NULL && hooks->get_var != NULL)
    {
        if (g_owner_shell_pid == 0)
        {
            // the main shell (pid in _MILK_LOCAL_SOCK) owns it, not a $(...) subshell that runs first
            const char *ls    = milk_ipc_get_var(hooks, "_MILK_LOCAL_SOCK");
            const char *dash  = (ls != NULL) ? strrchr(ls, '-') : NULL;
            pid_t       sp    = (dash != NULL) ? (pid_t) atol(dash + 1) : 0;
            g_owner_shell_pid = (sp > 0) ? sp : getpid();
        }
        if (getpid() != g_owner_shell_pid)
        {
            owner_pid = g_owner_shell_pid;
        }
        else
        {
            int owner_pipe[2] = { -1, -1 };
            if (pipe(owner_pipe) == 0)
            {
                if (g_owner_pipe_wfd >= 0)
                {
                    close(g_owner_pipe_wfd);
                }
                g_owner_pipe_wfd = fcntl(owner_pipe[1], F_DUPFD_CLOEXEC, 100);
                if (g_owner_pipe_wfd < 0)
                {
                    g_owner_pipe_wfd = owner_pipe[1];
                    fcntl(g_owner_pipe_wfd, F_SETFD, FD_CLOEXEC);
                }
                else
                {
                    close(owner_pipe[1]);
                }
                owner_rfd = owner_pipe[0];
            }
        }
    }

    char binpath[512];
    milk_ipc_resolve_cli_bin(hooks, binpath, sizeof(binpath));

    if (is_private && (hooks == NULL || hooks->get_var == NULL))
    {
        owner_pid = getppid(); // plain milk-call: live as long as the calling shell
    }
    if (owner_rfd < 0 && owner_pid > 0 && kill(owner_pid, 0) != 0 && errno == ESRCH)
    {
        fprintf(stderr, "milk: session %s has ended with its shell (%d)\n", sname, (int) owner_pid);
        return -1;
    }
    milk_spawn_opts_t opts = { .owner_rfd = owner_rfd,
                               .owner_pid = owner_pid,
                               .is_named  = !is_private };
    int               rc   = milk_ipc_spawn_daemon(binpath, sockpath, &opts);
    if (owner_rfd >= 0)
    {
        close(owner_rfd);
    }
    return rc;
}

/* The connections of this process to its sessions (the shell's and its front end's), kept
 * between requests while the replies say MILK_FLAG_KEEP: a request then costs no connect() and no
 * accept(). A forked subshell opens its own: two processes must not mix their frames on one. The
 * connection is close-on-exec, which makes it one of the shell's own fds for bash: bash undoes a
 * script's `exec N>` onto it. So it sits near the top of the fd limit, never on a number a script
 * uses (none is kept when it cannot), and is used again only while that fd still is the socket. */
typedef struct
{
    int      fd;
    pid_t    pid;  /* the process that connected */
    ino_t    ino;  /* the socket, to tell it from another file on the same fd number */
    int      keep; /* the last reply on it said MILK_FLAG_KEEP */
    uint64_t used; /* when it was last used, to replace the oldest */
    char     path[256];
} client_conn_t;

static client_conn_t g_conns[2]   = { { .fd = -1 }, { .fd = -1 } };
static uint64_t      g_conn_clock = 0;

/** conn_is_ours() - 1 if the fd of @c still is its socket. */
static int conn_is_ours(const client_conn_t *c)
{
    struct stat st;
    return c->fd >= 0 && fstat(c->fd, &st) == 0 && S_ISSOCK(st.st_mode) && st.st_ino == c->ino;
}

/** conn_forget() - drop kept connection @c: close its fd if it still is the socket. */
static void conn_forget(client_conn_t *c)
{
    if (conn_is_ours(c))
    {
        close(c->fd);
    }
    c->fd = -1;
}

/**
 * @brief Close the shell-lifetime owner pipe and any kept sockets in the current process.
 */
void milk_client_close_owner_pipe(void)
{
    if (g_owner_pipe_wfd >= 0)
    {
        close(g_owner_pipe_wfd);
        g_owner_pipe_wfd = -1;
    }
    for (int i = 0; i < 2; i++)
    {
        conn_forget(&g_conns[i]);
    }
}

/** conn_of() - the kept connection of this process on @fd, or NULL. */
static client_conn_t *conn_of(int fd)
{
    for (int i = 0; i < 2 && fd >= 0; i++)
    {
        if (g_conns[i].fd == fd && g_conns[i].pid == getpid())
        {
            return &g_conns[i];
        }
    }
    return NULL;
}

/** conn_high_fd() - move connection @fd to fd 1000 (3/4 of a lower limit); -1 if it cannot. */
static int conn_high_fd(int fd)
{
    struct rlimit rl;
    int           lo = 1000;
    if (getrlimit(RLIMIT_NOFILE, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY && rl.rlim_cur <= 1024)
    {
        lo = (int) (rl.rlim_cur * 3 / 4);
    }
    int hi = (lo >= 64) ? fcntl(fd, F_DUPFD_CLOEXEC, lo) : -1;
    if (hi >= 0)
    {
        close(fd);
    }
    return hi;
}

/** milk_client_connect() - this process's connection to @sockpath: the kept one if the server kept
 * it too (no hang-up, nothing unread), else a new one; -1 if it cannot connect. */
int milk_client_connect(const char *sockpath)
{
    client_conn_t *c = NULL;
    for (int i = 0; i < 2; i++)
    {
        if (g_conns[i].fd >= 0 && g_conns[i].pid != getpid())
        {
            conn_forget(&g_conns[i]);
        }
        struct pollfd p = { g_conns[i].fd, POLLIN | POLLRDHUP, 0 };
        if (g_conns[i].fd < 0 || strcmp(g_conns[i].path, sockpath) != 0)
        {
            continue;
        }
        if (g_conns[i].keep && conn_is_ours(&g_conns[i]) && poll(&p, 1, 0) == 0)
        {
            g_conns[i].keep = 0; /* until this request's reply says MILK_FLAG_KEEP */
            g_conns[i].used = ++g_conn_clock;
            return g_conns[i].fd;
        }
        conn_forget(&g_conns[i]);
    }
    int         fd = milk_ipc_connect_once(sockpath);
    int         hi = (fd >= 0) ? conn_high_fd(fd) : -1;
    struct stat st;
    if (hi < 0 || fstat(hi, &st) != 0)
    {
        return (hi >= 0) ? hi : fd; /* not kept: conn_end() closes it */
    }
    fd = hi;
    c  = (g_conns[0].fd < 0 || (g_conns[1].fd >= 0 && g_conns[0].used < g_conns[1].used))
             ? &g_conns[0]
             : &g_conns[1];
    conn_forget(c);
    *c = (client_conn_t) { .fd = fd, .pid = getpid(), .ino = st.st_ino, .used = ++g_conn_clock };
    snprintf(c->path, sizeof(c->path), "%s", sockpath);
    return fd;
}

/** conn_end() - the request on @fd is over: keep the connection if its reply said so, else
 * close it. */
static void conn_end(int fd)
{
    client_conn_t *c = conn_of(fd);
    if (c == NULL)
    {
        if (fd >= 0)
        {
            close(fd);
        }
    }
    else if (!c->keep)
    {
        conn_forget(c);
    }
}

/**
 * @brief Connect to the active `milk-cli --server`, auto-starting it on demand.
 *
 * The connection is kept for the next request (milk_client_connect(), conn_end()).
 *
 * @param hooks        Optional shell callback hooks
 * @return Connected socket fd (>= 0), or -1 on error
 */
int milk_client_connect_or_start(const milk_client_hooks_t *hooks)
{
    char sockpath[256];
    milk_ipc_resolve_sockpath(hooks, sockpath, sizeof(sockpath), 1);
    int fd = milk_client_connect(sockpath);
    if (fd >= 0)
    {
        return fd;
    }

    char sname[64] = "";
    milk_srvlog_parse_sock(sockpath, NULL, sname, sizeof(sname));
    if (sname[0] != '\0' && !milk_ipc_is_numeric_str(sname)) // only @session new starts a named one
    {
        fprintf(stderr, "milk: session %s has ended: @session new %s, or @session leave\n", sname,
                sname);
        return -1;
    }
    if (autostart_server(hooks, sockpath) == 0)
    {
        fd = milk_client_connect(sockpath);
        if (fd >= 0)
        {
            return fd;
        }
    }
    fprintf(stderr, "ERROR: milk: unable to connect to or start milk-cli --server on '%s'\n",
            sockpath);
    return -1;
}

/**
 * try_add_packed_var - Look up an identifier in the shell and append it to a packed variable set
 * @pv:    Packed variable accumulator structure
 * @hooks: Optional shell callback hooks for reading unexported variables
 * @ident: Candidate variable identifier extracted from an expression
 *
 * Deduplicates @ident against entries already present in @pv and, if defined in
 * the calling shell with a value shorter than 128 bytes, records its name and value
 * so the server can evaluate `@calc` / `@test` expressions referencing local shell
 * variables.
 *
 * Return: void
 */
static void try_add_packed_var(milk_packed_vars_t        *pv,
                               const milk_client_hooks_t *hooks,
                               const char                *ident)
{
    if (pv->count >= 256 || ident == NULL || ident[0] == '\0' || strlen(ident) >= 64)
    {
        return;
    }
    for (int i = 0; i < pv->count; i++)
    {
        if (strcmp(pv->names[i], ident) == 0)
        {
            return;
        }
    }
    const char *val = milk_ipc_get_var(hooks, ident);
    if (val != NULL && strlen(val) < sizeof(pv->vals[0]))
    {
        snprintf(pv->names[pv->count], sizeof(pv->names[0]), "%s", ident);
        snprintf(pv->vals[pv->count], sizeof(pv->vals[0]), "%s", val);
        pv->count++;
    }
}

/**
 * milk_client_pack_digits - Pack the shell's MILK_DPDIGITS: the session prints with its digits
 * @pv:    Packed variable accumulator
 * @hooks: Optional shell callback hooks
 *
 * Return: void
 */
void milk_client_pack_digits(milk_packed_vars_t *pv, const milk_client_hooks_t *hooks)
{
    try_add_packed_var(pv, hooks, "MILK_DPDIGITS");
}

/**
 * @brief Extract identifier tokens from @s and pack any defined shell variables.
 *
 * @param pv    Packed variable accumulator
 * @param hooks Optional shell callback hooks
 * @param s     Input expression or argument string
 */
void milk_client_collect_identifiers(milk_packed_vars_t        *pv,
                                     const milk_client_hooks_t *hooks,
                                     const char                *s)
{
    if (s == NULL)
    {
        return;
    }
    const char *p    = s;
    int         in_q = 0;
    while (*p != '\0')
    {
        if (in_q)
        {
            if (*p == '\\' && p[1] != '\0')
            {
                p += 2;
                continue;
            }
            if (*p == '"')
            {
                in_q = 0;
            }
            p++;
            continue;
        }
        if (*p == '"')
        {
            in_q = 1;
            p++;
            continue;
        }
        if (*p == '#')
        {
            while (*p != '\0' && *p != '\n')
            {
                p++;
            }
            continue;
        }
        if (isalpha((unsigned char) *p) || *p == '_')
        {
            const char *start = p++;
            while (isalnum((unsigned char) *p) || *p == '_')
            {
                p++;
            }
            size_t len = (size_t) (p - start);
            if (len > 0 && len < 64)
            {
                char ident[64];
                memcpy(ident, start, len);
                ident[len] = '\0';
                try_add_packed_var(pv, hooks, ident);
            }
        }
        else
        {
            p++;
        }
    }
}

/**
 * resync_commands() - define a function for every module command of the server: it loaded commands
 * the catalog lacks (registry generation changed).
 */
static void resync_commands(const milk_client_hooks_t *hooks)
{
    char           out[32768];
    milk_payload_t p = { 0 };

    if (hooks == NULL || hooks->def_cmd == NULL)
    {
        return;
    }
    milk_pl_str(&p, ""); /* COMPLETE: prefix "", type "cmd" */
    milk_pl_str(&p, "cmd");
    if (milk_client_request(milk_client_connect_or_start(hooks), hooks, MILK_MSG_COMPLETE, 0, &p,
                            NULL, 0, out, sizeof(out)) == 0)
    {
        for (char *save = NULL, *k = strtok_r(out, "\n", &save); k != NULL;
             k = strtok_r(NULL, "\n", &save))
        {
            hooks->def_cmd(k, hooks->ctx);
        }
    }
}

/* A second Ctrl-C this long after the first, still without an answer, stops the wait */
#define MILK_CLIENT_GIVE_UP_S 2.0

/**
 * @brief Receive and process server response frames until `MILK_MSG_DONE`.
 *
 * Forwards `SIGINT` (`MILK_MSG_CANCEL`) and `SIGWINCH` (`MILK_MSG_WINCH`)
 * over the active socket connection while waiting.
 *
 * @param fd         Connected socket fd
 * @param hooks      Optional shell callback hooks
 * @param value_buf  Optional buffer for `MILK_MSG_VALUE` (NULL to print to stdout)
 * @param value_sz   Capacity of @value_buf
 * @return Exit status returned by the server (or 130 on SIGINT)
 */
static int await_response(int                        fd,
                          const milk_client_hooks_t *hooks,
                          char                      *value_buf,
                          size_t                     value_sz)
{
    if (value_buf != NULL && value_sz > 0)
    {
        value_buf[0] = '\0';
    }

    struct sigaction sa_int, old_int;
    struct sigaction sa_winch, old_winch;
    memset(&sa_int, 0, sizeof(sa_int));
    memset(&sa_winch, 0, sizeof(sa_winch));
    sa_int.sa_handler   = client_sigint_handler;
    sa_winch.sa_handler = client_sigwinch_handler;
    sigemptyset(&sa_int.sa_mask);
    sigemptyset(&sa_winch.sa_mask);

    g_client_sigint   = 0;
    g_client_sigwinch = 0;
    int had_sigint    = 0;

    sigaction(SIGINT, &sa_int, &old_int);
    sigaction(SIGWINCH, &sa_winch, &old_winch);

    int32_t final_rc = MILK_CLIENT_RC_TRANSPORT; // until DONE arrives, the transport failed
    int     resync   = 0;
    int     queued   = 0;   // the request waits for another shell's command
    double  cancel_t = 0.0; // when Ctrl-C was first forwarded (0: not yet)

    for (;;)
    {
        if (g_client_sigint)
        {
            g_client_sigint = 0;
            had_sigint      = 1;
            if (queued)
            {
                break; // not started: hanging up drops it
            }
            struct timespec ts;
            clock_gettime(CLOCK_MONOTONIC, &ts);
            double now = (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
            if (cancel_t > 0.0 && now - cancel_t >= MILK_CLIENT_GIVE_UP_S)
            {
                // the session did not answer the first Ctrl-C (stopped, or stuck): the shell
                // gets its prompt back, and the connection is not used again
                fprintf(stderr, "milk: no answer from the session: not waiting for it (the "
                                "command may still run there)\n");
                client_conn_t *c = conn_of(fd);
                if (c != NULL)
                {
                    c->keep = 0;
                }
                break;
            }
            if (cancel_t == 0.0)
            {
                cancel_t = now;
            }
            milk_proto_send(fd, MILK_MSG_CANCEL, 0, NULL, 0);
        }
        if (g_client_sigwinch)
        {
            g_client_sigwinch = 0;
            milk_proto_send(fd, MILK_MSG_WINCH, 0, NULL, 0);
        }

        struct pollfd pfd = { fd, POLLIN | POLLHUP | POLLERR, 0 };
        int           pr  = poll(&pfd, 1, -1);
        if (pr < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            fprintf(stderr, "milk: lost connection to server: %s\n", strerror(errno));
            break;
        }

        milk_frame_hdr_t hdr;
        uint8_t         *pl = NULL;
        int              rr = milk_proto_recv(fd, &hdr, &pl, NULL);
        if (rr != 0)
        {
            fprintf(stderr, (rr == -2)
                                ? "milk: server speaks another protocol version; restart it\n"
                                : "milk: lost connection to server\n");
            final_rc = MILK_CLIENT_RC_TRANSPORT;
            break;
        }

        if (hdr.type == MILK_MSG_VALUE && pl != NULL && hdr.len > 0 &&
            (hdr.flags & MILK_FLAG_ERRTEXT) != 0)
        {
            fprintf(stderr, "%s\n",
                    (const char *) pl); /* the request ran where our stderr is not */
        }
        else if (hdr.type == MILK_MSG_VALUE && pl != NULL && hdr.len > 0)
        {
            if (value_buf != NULL && value_sz > 0)
            {
                snprintf(value_buf, value_sz, "%s", (const char *) pl);
            }
            else
            {
                fputs((const char *) pl, stdout);
                size_t sl = strlen((const char *) pl);
                if (sl == 0 || ((const char *) pl)[sl - 1] != '\n')
                {
                    fputc('\n', stdout);
                }
                fflush(stdout);
            }
        }
        else if (hdr.type == MILK_MSG_SETVAR && pl != NULL && hdr.len > 0)
        {
            uint32_t    off   = 0;
            const char *vname = milk_pl_next(pl, hdr.len, &off, "");
            milk_ipc_set_var(hooks, vname, milk_pl_next(pl, hdr.len, &off, ""), 0);
        }
        else if (hdr.type == MILK_MSG_DEFCMD && pl != NULL && hdr.len > 0 && hooks != NULL &&
                 hooks->def_cmd != NULL)
        {
            hooks->def_cmd((const char *) pl, hooks->ctx);
        }
        else if (hdr.type == MILK_MSG_QUEUED)
        {
            queued          = (pl != NULL && hdr.len > 0); // empty: it runs now
            const char *tab = queued ? strchr((const char *) pl, '\t') : NULL;
            if (tab != NULL && isatty(STDERR_FILENO))
            {
                fprintf(stderr, "milk: waiting for '%.*s' (another shell's, running for %s s)\n",
                        (int) (tab - (const char *) pl), (const char *) pl, tab + 1);
            }
        }
        else if (hdr.type == MILK_MSG_DONE)
        {
            client_conn_t *c = conn_of(fd);
            if (c != NULL)
            {
                c->keep = (hdr.flags & MILK_FLAG_KEEP) != 0; /* else conn_end() closes it */
            }
            if (pl != NULL && hdr.len >= sizeof(int32_t))
            {
                memcpy(&final_rc, pl, sizeof(int32_t));
            }
            else
            {
                final_rc = 0;
            }
            if (pl != NULL && hdr.len >= 2 * sizeof(int32_t)) /* a CALL: registry generation */
            {
                uint32_t gen;
                memcpy(&gen, pl + sizeof(int32_t), sizeof(gen));
                resync     = (gen != g_seen_gen);
                g_seen_gen = gen;
            }
            free(pl);
            break;
        }
        free(pl);
    } // for (;;)

    sigaction(SIGINT, &old_int, NULL);
    sigaction(SIGWINCH, &old_winch, NULL);
    had_sigint =
        had_sigint || g_client_sigint; /* arrived after the last check (e.g. right after DONE) */
    g_client_sigint = 0;

    if (had_sigint)
    {
        if (hooks != NULL && hooks->on_sigint != NULL)
        {
            hooks->on_sigint(hooks->ctx);
        }
        if (final_rc == 0 ||
            final_rc == MILK_CLIENT_RC_TRANSPORT) // cancelled, possibly before DONE was sent
        {
            final_rc = 130;
        }
    }
    if (resync)
    {
        resync_commands(hooks);
    }
    return (int) final_rc;
}

void milk_client_pack_vars(milk_payload_t *p, const milk_packed_vars_t *pv)
{
    milk_pl_u32(p, (uint32_t) pv->count);
    for (int i = 0; i < pv->count; i++)
    {
        milk_pl_str(p, pv->names[i]);
        milk_pl_str(p, pv->vals[i]);
    }
}

int milk_client_request(int                        fd,
                        const milk_client_hooks_t *hooks,
                        uint8_t                    type,
                        uint8_t                    flags,
                        milk_payload_t            *p,
                        const int                 *fds,
                        int                        nfds,
                        char                      *out,
                        size_t                     outsz)
{
    int rc = MILK_CLIENT_RC_TRANSPORT;
    if (fd >= 0 && !p->bad &&
        ((nfds > 0) ? milk_proto_send_with_fds(fd, type, flags, p->buf, p->len, fds, nfds)
                    : milk_proto_send(fd, type, flags, p->buf, p->len)) == 0)
    {
        rc = await_response(fd, hooks, out, outsz);
    }
    else if (fd >= 0)
    {
        fprintf(stderr, "milk: the request could not be sent to the session\n");
    }
    free(p->buf);
    conn_end(fd);
    return rc;
}
