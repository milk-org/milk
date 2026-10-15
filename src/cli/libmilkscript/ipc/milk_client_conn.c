// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_client_conn.c
 * @brief Client connection management, lazy daemon auto-start, and response loop.
 *
 * Role & Goal:
 *   Manages UNIX domain socket connections from bash builtins (`libmilkbash.so`)
 *   and `milk-call` to `milk-cli --server`. Handles lazy on-demand spawning of
 *   the per-shell server daemon with lifetime pipe binding, tracks synced shell
 *   variables across invocations, and runs the response frame loop with `SIGINT`
 *   and `SIGWINCH` forwarding.
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
#include <unistd.h>

#include "milk_client_internal.h"
#include "milk_ipc_common.h"
#include "milk_proto.h"
#include "milk_srvlog.h"

#define MAX_SYNCED_VARS 256
static char g_synced_vars[MAX_SYNCED_VARS][64];
static int  g_nb_synced_vars = 0;

static pid_t g_owner_shell_pid = 0;
static int   g_owner_pipe_wfd  = -1;

static volatile sig_atomic_t g_client_sigint   = 0;
static volatile sig_atomic_t g_client_sigwinch = 0;

static void client_sigint_handler(int sig)
{
    (void) sig;
    g_client_sigint = 1;
}

static void client_sigwinch_handler(int sig)
{
    (void) sig;
    g_client_sigwinch = 1;
}

/**
 * @brief Record a variable name returned by the server via `MILK_MSG_SETVAR`.
 *
 * @param name Variable name
 */
static void record_synced_var(const char *name)
{
    if (name == NULL || name[0] == '\0' || strlen(name) >= 64)
    {
        return;
    }
    for (int i = 0; i < g_nb_synced_vars; i++)
    {
        if (strcmp(g_synced_vars[i], name) == 0)
        {
            return;
        }
    }
    if (g_nb_synced_vars < MAX_SYNCED_VARS)
    {
        snprintf(g_synced_vars[g_nb_synced_vars], 64, "%s", name);
        g_nb_synced_vars++;
    }
}

/**
 * @brief Close the shell-lifetime owner pipe if held by the current process.
 */
void milk_client_close_owner_pipe(void)
{
    if (g_owner_pipe_wfd >= 0)
    {
        close(g_owner_pipe_wfd);
        g_owner_pipe_wfd = -1;
    }
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

    int owner_rfd = -1;
    if (is_private && hooks != NULL && hooks->get_var != NULL)
    {
        if (g_owner_shell_pid == 0)
        {
            g_owner_shell_pid = getpid();
        }
        if (getpid() == g_owner_shell_pid)
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

    int rc = milk_ipc_spawn_daemon(binpath, sockpath, owner_rfd, g_owner_pipe_wfd, !is_private);
    if (owner_rfd >= 0)
    {
        close(owner_rfd);
    }
    return rc;
}

/**
 * @brief Connect to the active `milk-cli --server`, auto-starting it on demand.
 *
 * @param hooks        Optional shell callback hooks
 * @param sockpath_out Output buffer receiving the resolved socket path
 * @param sockpath_sz  Size of @sockpath_out in bytes
 * @return Connected socket fd (>= 0), or -1 on error
 */
int milk_client_connect_or_start(const milk_client_hooks_t *hooks,
                                 char                      *sockpath_out,
                                 size_t                     sockpath_sz)
{
    milk_ipc_resolve_sockpath(hooks, sockpath_out, sockpath_sz, 1);
    int fd = milk_ipc_connect_once(sockpath_out);
    if (fd >= 0)
    {
        return fd;
    }

    if (autostart_server(hooks, sockpath_out) == 0)
    {
        fd = milk_ipc_connect_once(sockpath_out);
        if (fd >= 0)
        {
            return fd;
        }
    }
    fprintf(stderr, "ERROR: milk: unable to connect to or start milk-cli --server on '%s'\n",
            sockpath_out);
    return -1;
}

/**
 * @brief Check whether a string is a valid floating-point numeric literal.
 */
static int is_numeric_literal(const char *s)
{
    if (s == NULL || s[0] == '\0')
    {
        return 0;
    }
    char *end = NULL;
    errno     = 0;
    (void) strtod(s, &end);
    return (errno == 0 && end != s && *end == '\0') ? 1 : 0;
}

static void try_add_packed_var(milk_packed_vars_t        *pv,
                               const milk_client_hooks_t *hooks,
                               const char                *ident,
                               int                        allow_any)
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
    if (val != NULL && (allow_any || is_numeric_literal(val)) && strlen(val) < 128)
    {
        snprintf(pv->names[pv->count], 64, "%s", ident);
        snprintf(pv->vals[pv->count], 128, "%s", val);
        pv->count++;
    }
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
    const char *p = s;
    while (*p != '\0')
    {
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
                try_add_packed_var(pv, hooks, ident, 1);
            }
        }
        else
        {
            p++;
        }
    }
}

/**
 * @brief Pack numeric values of any variables previously synced from the server.
 *
 * @param pv    Packed variable accumulator
 * @param hooks Optional shell callback hooks
 */
void milk_client_collect_synced_vars(milk_packed_vars_t *pv, const milk_client_hooks_t *hooks)
{
    for (int i = 0; i < g_nb_synced_vars; i++)
    {
        try_add_packed_var(pv, hooks, g_synced_vars[i], 0);
    }
}

/**
 * @brief Receive and process server response frames until `MILK_MSG_DONE`.
 *
 * Forwards `SIGINT` (`MILK_MSG_CANCEL`) and `SIGWINCH` (`MILK_MSG_WINCH`)
 * over the active socket connection while waiting.
 *
 * @param fd         Connected socket fd
 * @param sockpath   Socket path string
 * @param hooks      Optional shell callback hooks
 * @param value_buf  Optional buffer for `MILK_MSG_VALUE` (NULL to print to stdout)
 * @param value_sz   Capacity of @value_buf
 * @param got_value  Optional flag set to 1 if a `MILK_MSG_VALUE` frame arrived
 * @return Exit status returned by the server (or 130 on SIGINT)
 */
int milk_client_await_response(int                        fd,
                               const char                *sockpath,
                               const milk_client_hooks_t *hooks,
                               char                      *value_buf,
                               size_t                     value_sz,
                               int                       *got_value)
{
    (void) sockpath;
    if (got_value != NULL)
    {
        *got_value = 0;
    }
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

    int32_t final_rc = 1;

    for (;;)
    {
        if (g_client_sigint)
        {
            g_client_sigint = 0;
            had_sigint      = 1;
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
            break;
        }

        milk_frame_hdr_t hdr;
        uint8_t         *pl = NULL;
        if (milk_proto_recv(fd, &hdr, &pl, NULL) != 0)
        {
            break;
        }

        if (hdr.type == MILK_MSG_VALUE && pl != NULL && hdr.len > 0)
        {
            if (value_buf != NULL && value_sz > 0)
            {
                snprintf(value_buf, value_sz, "%s", (const char *) pl);
                if (got_value != NULL)
                {
                    *got_value = 1;
                }
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
            const char *vname = (const char *) pl;
            size_t      o1    = strlen(vname) + 1;
            const char *vval  = (o1 < hdr.len) ? (const char *) (pl + o1) : "";
            milk_ipc_set_var(hooks, vname, vval, 0);
            record_synced_var(vname);
        }
        else if (hdr.type == MILK_MSG_DEFCMD && pl != NULL && hdr.len > 0 && hooks != NULL &&
                 hooks->def_cmd != NULL)
        {
            hooks->def_cmd((const char *) pl, hooks->ctx);
        }
        else if (hdr.type == MILK_MSG_DONE)
        {
            if (pl != NULL && hdr.len >= sizeof(int32_t))
            {
                memcpy(&final_rc, pl, sizeof(int32_t));
            }
            else
            {
                final_rc = 0;
            }
            free(pl);
            break;
        }
        free(pl);
    } // for (;;)

    sigaction(SIGINT, &old_int, NULL);
    sigaction(SIGWINCH, &old_winch, NULL);

    if (had_sigint)
    {
        if (hooks != NULL && hooks->on_sigint != NULL)
        {
            hooks->on_sigint(hooks->ctx);
        }
        if (final_rc == 0)
        {
            final_rc = 130;
        }
    }
    return (int) final_rc;
}
