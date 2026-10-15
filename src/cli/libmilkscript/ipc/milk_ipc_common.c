// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_ipc_common.c
 * @brief Shared socket, environment, binary lookup, and daemon-spawning helpers.
 *
 * Role & Goal:
 *   Implements the common IPC utilities declared in `milk_ipc_common.h` so
 *   that `milk_client_conn.c`, `milk_client_ops.c`, `milk_srvlog_shm.c`, and
 *   `milk_srv_manager.c` share a single implementation of socket addressing,
 *   shell variable propagation, server ping/shutdown, and double-fork daemon
 *   launching.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include "milk_ipc_common.h"
#include "milk_proto.h"

/**
 * @brief Read a variable from the calling shell (via hooks) or `getenv()`.
 *
 * @param hooks Optional shell callback hooks (may be NULL)
 * @param name  Variable name
 * @return Value string, or NULL if unset
 */
const char *milk_ipc_get_var(const milk_client_hooks_t *hooks, const char *name)
{
    if (hooks != NULL && hooks->get_var != NULL)
    {
        const char *v = hooks->get_var(name, hooks->ctx);
        if (v != NULL)
        {
            return v;
        }
    }
    return getenv(name);
}

static void get_shell_override_file(const milk_client_hooks_t *hooks, char *out, size_t outsz)
{
    int         shell_pid  = 0;
    const char *local_sock = milk_ipc_get_var(hooks, "_MILK_LOCAL_SOCK");
    if (local_sock != NULL && local_sock[0] != '\0')
    {
        const char *last_dash = strrchr(local_sock, '-');
        if (last_dash != NULL && isdigit((unsigned char) last_dash[1]))
        {
            shell_pid = atoi(last_dash + 1);
        }
    }
    if (shell_pid <= 0)
    {
        shell_pid = (hooks != NULL && hooks->get_var != NULL) ? (int) getpid() : (int) getppid();
    }
    const char *tmpdir = getenv("TMPDIR");
    if (tmpdir == NULL || tmpdir[0] == '\0')
    {
        tmpdir = "/tmp";
    }
    snprintf(out, outsz, "%s/.milk_active_sock_%u_%d", tmpdir, (unsigned) getuid(), shell_pid);
}

/**
 * @brief Set a variable in the calling shell, process environment, and eval FD.
 *
 * @param hooks             Optional shell callback hooks (may be NULL)
 * @param name              Variable name
 * @param val               Value string
 * @param export_in_eval_fd If non-zero, prefix `export ` on `MILK_CALL_EVAL_FD`
 */
void milk_ipc_set_var(const milk_client_hooks_t *hooks,
                      const char                *name,
                      const char                *val,
                      int                        export_in_eval_fd)
{
    const char *safe_val = (val != NULL) ? val : "";
    if (name != NULL && strcmp(name, "MILKCLI_SOCK") == 0)
    {
        char ovfile[512];
        char def_local[128];
        int  shell_pid =
            (hooks != NULL && hooks->get_var != NULL) ? (int) getpid() : (int) getppid();
        const char *local_sock = milk_ipc_get_var(hooks, "_MILK_LOCAL_SOCK");
        snprintf(def_local, sizeof(def_local), "@milk-cli-%u-%d", (unsigned) getuid(), shell_pid);
        get_shell_override_file(hooks, ovfile, sizeof(ovfile));
        if (safe_val[0] == '\0' || strcmp(safe_val, def_local) == 0 ||
            (local_sock != NULL && strcmp(safe_val, local_sock) == 0))
        {
            unlink(ovfile);
        }
        else
        {
            int fd = open(ovfile, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
            if (fd >= 0)
            {
                if (write(fd, safe_val, strlen(safe_val)) < 0)
                {
                }
                close(fd);
            }
        }
    }

    if (hooks != NULL && hooks->set_var != NULL)
    {
        hooks->set_var(name, safe_val, hooks->ctx);
        if (!export_in_eval_fd)
        {
            return;
        }
    }

    const char *eval_fd_str = getenv("MILK_CALL_EVAL_FD");
    if (eval_fd_str != NULL && eval_fd_str[0] != '\0')
    {
        int efd = atoi(eval_fd_str);
        if (efd > 2)
        {
            char line[512];
            int  n = snprintf(line, sizeof(line), "%s%s='%s'\n", export_in_eval_fd ? "export " : "",
                              name, safe_val);
            if (n > 0 && n < (int) sizeof(line))
            {
                if (write(efd, line, (size_t) n) < 0)
                {
                }
            }
        }
    }
    setenv(name, safe_val, 1);
}

/**
 * @brief Synchronize `MILK_SHM_DIR` from shell hooks into `setenv()`.
 *
 * @param hooks Optional shell callback hooks (may be NULL)
 */
void milk_ipc_sync_shm_env(const milk_client_hooks_t *hooks)
{
    if (hooks != NULL && hooks->get_var != NULL)
    {
        const char *sd = hooks->get_var("MILK_SHM_DIR", hooks->ctx);
        if (sd != NULL && sd[0] != '\0')
        {
            setenv("MILK_SHM_DIR", sd, 1);
        }
    }
}

/**
 * @brief Resolve the active `MILKCLI_SOCK` address for the current shell.
 *
 * @param hooks            Optional shell callback hooks (may be NULL)
 * @param out              Output socket path buffer
 * @param outsz            Size of @out in bytes
 * @param persist_if_unset If non-zero and `MILKCLI_SOCK` was unset, store the
 *                         computed default back into the shell environment
 */
void milk_ipc_resolve_sockpath(const milk_client_hooks_t *hooks,
                               char                      *out,
                               size_t                     outsz,
                               int                        persist_if_unset)
{
    char ovfile[512];
    get_shell_override_file(hooks, ovfile, sizeof(ovfile));
    int fd = open(ovfile, O_RDONLY | O_CLOEXEC);
    if (fd >= 0)
    {
        char    buf[256];
        ssize_t n = read(fd, buf, sizeof(buf) - 1);
        close(fd);
        if (n > 0)
        {
            buf[n] = '\0';
            while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r'))
            {
                buf[--n] = '\0';
            }
            if (buf[0] != '\0')
            {
                snprintf(out, outsz, "%s", buf);
                if (hooks != NULL && hooks->set_var != NULL)
                {
                    hooks->set_var("MILKCLI_SOCK", out, hooks->ctx);
                }
                setenv("MILKCLI_SOCK", out, 1);
                return;
            }
        }
    }

    const char *s = milk_ipc_get_var(hooks, "MILKCLI_SOCK");
    if (s != NULL && s[0] != '\0')
    {
        snprintf(out, outsz, "%s", s);
        return;
    }
    pid_t ref_pid = (hooks != NULL && hooks->get_var != NULL) ? getpid() : getppid();
    snprintf(out, outsz, "@milk-cli-%u-%d", (unsigned) getuid(), (int) ref_pid);
    if (persist_if_unset)
    {
        if (hooks != NULL && hooks->set_var != NULL)
        {
            hooks->set_var("MILKCLI_SOCK", out, hooks->ctx);
        }
        setenv("MILKCLI_SOCK", out, 1);
    }
}

/**
 * @brief Locate the `milk-cli` executable path using `MILKCLI_BIN` or `MILK_INSTALLDIR`.
 *
 * @param hooks Optional shell callback hooks (may be NULL)
 * @param out   Output path buffer
 * @param outsz Size of @out in bytes
 */
void milk_ipc_resolve_cli_bin(const milk_client_hooks_t *hooks, char *out, size_t outsz)
{
    const char *bin = milk_ipc_get_var(hooks, "MILKCLI_BIN");
    if (bin != NULL && bin[0] != '\0' && access(bin, X_OK) == 0)
    {
        snprintf(out, outsz, "%s", bin);
        return;
    }
    const char *inst = milk_ipc_get_var(hooks, "MILK_INSTALLDIR");
    if (inst != NULL && inst[0] != '\0')
    {
        char cand[512];
        snprintf(cand, sizeof(cand), "%s/bin/milk-cli", inst);
        if (access(cand, X_OK) == 0)
        {
            snprintf(out, outsz, "%s", cand);
            return;
        }
    }
#ifdef INSTALLDIR
    {
        char cand[512];
        snprintf(cand, sizeof(cand), "%s/bin/milk-cli", INSTALLDIR);
        if (access(cand, X_OK) == 0)
        {
            snprintf(out, outsz, "%s", cand);
            return;
        }
    }
#endif
    snprintf(out, outsz, "milk-cli");
}

/**
 * @brief Check whether a string consists solely of ASCII decimal digits.
 *
 * @param s Input string
 * @return 1 if non-empty and all digits, 0 otherwise
 */
int milk_ipc_is_numeric_str(const char *s)
{
    if (s == NULL || s[0] == '\0')
    {
        return 0;
    }
    for (size_t i = 0; s[i] != '\0'; i++)
    {
        if (!isdigit((unsigned char) s[i]))
        {
            return 0;
        }
    }
    return 1;
}

/**
 * @brief Attempt a single non-retrying UNIX domain socket `connect()`.
 *
 * @param sockpath Abstract (`@name`) or filesystem UNIX socket path
 * @return Connected socket fd (>= 0), or -1 on failure
 */
int milk_ipc_connect_once(const char *sockpath)
{
    struct sockaddr_un addr;
    socklen_t          addrlen = 0;
    if (milk_sockaddr_init(sockpath, &addr, &addrlen) != 0)
    {
        return -1;
    }
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0)
    {
        return -1;
    }
    if (connect(fd, (struct sockaddr *) &addr, addrlen) != 0)
    {
        close(fd);
        return -1;
    }
    return fd;
}

/**
 * @brief Send `MILK_MSG_PING` to @sockpath and verify `MILK_MSG_DONE`.
 *
 * @param sockpath Target UNIX socket path
 * @return 1 if server responded to ping, 0 otherwise
 */
int milk_ipc_ping_socket(const char *sockpath)
{
    int fd = milk_ipc_connect_once(sockpath);
    if (fd < 0)
    {
        return 0;
    }
    int ok = 0;
    if (milk_proto_send(fd, MILK_MSG_PING, 0, NULL, 0) == 0)
    {
        milk_frame_hdr_t hdr;
        uint8_t         *pl = NULL;
        if (milk_proto_recv(fd, &hdr, &pl, NULL) == 0 && hdr.type == MILK_MSG_DONE)
        {
            ok = 1;
        }
        free(pl);
    }
    close(fd);
    return ok;
}

/**
 * @brief Send `MILK_MSG_SHUTDOWN` to @sockpath and wait for acknowledgment.
 *
 * @param sockpath Target UNIX socket path
 * @return 1 if shutdown frame was delivered, 0 if server was unreachable
 */
int milk_ipc_shutdown_socket(const char *sockpath)
{
    int fd = milk_ipc_connect_once(sockpath);
    if (fd < 0)
    {
        return 0;
    }
    milk_proto_send(fd, MILK_MSG_SHUTDOWN, 0, NULL, 0);
    milk_frame_hdr_t hdr;
    uint8_t         *pl = NULL;
    (void) milk_proto_recv(fd, &hdr, &pl, NULL);
    free(pl);
    close(fd);
    return 1;
}

/**
 * @brief Double-fork and `execlp` a background `milk-cli -q --server` daemon.
 *
 * @param binpath      Path or name of `milk-cli` binary
 * @param sockpath     UNIX socket path to pass to `--server`
 * @param owner_rfd    Read end of owner lifetime pipe (-1 if none)
 * @param close_in_chd Optional extra fd to close in child (-1 if none)
 * @param is_named     If non-zero, set `MILK_SRV_NAMED=1` in daemon environment
 * @return 0 once daemon signals readiness, -1 on error or timeout
 */
int milk_ipc_spawn_daemon(const char *binpath,
                          const char *sockpath,
                          int         owner_rfd,
                          int         close_in_chd,
                          int         is_named)
{
    int ready_pipe[2] = { -1, -1 };
    if (pipe(ready_pipe) != 0)
    {
        return -1;
    }
    fcntl(ready_pipe[0], F_SETFD, FD_CLOEXEC);

    pid_t pid = fork();
    if (pid < 0)
    {
        close(ready_pipe[0]);
        close(ready_pipe[1]);
        return -1;
    }

    if (pid == 0)
    {
        /* Double-fork so the server daemon is reparented to init/subreaper and
         * never interferes with the calling shell's wait() / job table */
        close(ready_pipe[0]);
        if (close_in_chd >= 0)
        {
            close(close_in_chd);
        }
        setsid();

        pid_t pid2 = fork();
        if (pid2 < 0)
        {
            _exit(1);
        }
        if (pid2 > 0)
        {
            _exit(0);
        }

        int devnull = open("/dev/null", O_RDWR);
        if (devnull >= 0)
        {
            dup2(devnull, STDIN_FILENO);
            dup2(devnull, STDOUT_FILENO);
            dup2(devnull, STDERR_FILENO);
            if (devnull > 2)
            {
                close(devnull);
            }
        }

        if (is_named)
        {
            setenv("MILK_SRV_NAMED", "1", 1);
        }

        char rfd_str[32];
        snprintf(rfd_str, sizeof(rfd_str), "%d", ready_pipe[1]);

        if (owner_rfd >= 0)
        {
            char ofd_str[32];
            snprintf(ofd_str, sizeof(ofd_str), "%d", owner_rfd);
            execlp(binpath, binpath, "-q", "--server", sockpath, "--ready-fd", rfd_str,
                   "--owner-fd", ofd_str, (char *) NULL);
        }
        else
        {
            execlp(binpath, binpath, "-q", "--server", sockpath, "--ready-fd", rfd_str,
                   (char *) NULL);
        }
        _exit(127);
    }

    close(ready_pipe[1]);
    while (waitpid(pid, NULL, 0) < 0 && errno == EINTR)
    {
    }

    struct pollfd pfd = { ready_pipe[0], POLLIN, 0 };
    int           pr  = poll(&pfd, 1, 5000);
    char          rbuf[16];
    ssize_t       rn = 0;
    if (pr > 0 && (pfd.revents & POLLIN))
    {
        rn = read(ready_pipe[0], rbuf, sizeof(rbuf));
    }
    close(ready_pipe[0]);
    return (rn > 0) ? 0 : -1;
}
