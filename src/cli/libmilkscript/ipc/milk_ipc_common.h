// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_ipc_common.h
 * @brief Shared socket, environment, binary lookup, and daemon-spawning helpers.
 *
 * Role & Goal:
 *   Consolidates all low-level IPC and shell-hook primitives shared between
 *   the milk client (`milk_client_conn.c`, `milk_client_ops.c`) and the
 *   multi-server lifecycle manager (`milk_srv_manager.c`, `milk_srvlog_shm.c`).
 *   Eliminates duplicated socket resolution, binary discovery, shell variable
 *   propagation (`MILK_CALL_EVAL_FD`), ping/shutdown handshakes, and
 *   double-fork daemon spawning.
 */

#ifndef MILK_IPC_COMMON_H
#define MILK_IPC_COMMON_H

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#include "milk_client.h"

/**
 * @brief Read a variable from the calling shell (via hooks) or `getenv()`.
 *
 * @param hooks Optional shell callback hooks (may be NULL)
 * @param name  Variable name
 * @return Value string, or NULL if unset
 */
const char *milk_ipc_get_var(const milk_client_hooks_t *hooks, const char *name);

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
                      int                        export_in_eval_fd);

char *milk_ipc_bind_script(const char *name, int assoc, char *text);

/** milk_ipc_is_identifier() - 1 if @name matches [A-Za-z_][A-Za-z0-9_]* (a shell variable name). */
int milk_ipc_is_identifier(const char *name);

void milk_ipc_eval(const milk_client_hooks_t *hooks, char *script);

/**
 * @brief Synchronize `MILK_SHM_DIR` from shell hooks into `setenv()`.
 *
 * @param hooks Optional shell callback hooks (may be NULL)
 */
void milk_ipc_sync_shm_env(const milk_client_hooks_t *hooks);

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
                               int                        persist_if_unset);

/**
 * @brief Locate the `milk-cli` executable path using `MILKCLI_BIN` or `MILK_INSTALLDIR`.
 *
 * @param hooks Optional shell callback hooks (may be NULL)
 * @param out   Output path buffer
 * @param outsz Size of @out in bytes
 */
void milk_ipc_resolve_cli_bin(const milk_client_hooks_t *hooks, char *out, size_t outsz);

/**
 * @brief Check whether a string consists solely of ASCII decimal digits.
 *
 * @param s Input string
 * @return 1 if non-empty and all digits, 0 otherwise
 */
int milk_ipc_is_numeric_str(const char *s);

/**
 * @brief Return 1 if the process at the other end of UNIX socket @fd runs under our uid.
 */
int milk_ipc_peer_is_self(int fd);

/**
 * @brief Attempt a single non-retrying UNIX domain socket `connect()`.
 *
 * @param sockpath Abstract (`@name`) or filesystem UNIX socket path
 * @return Connected socket fd (>= 0), or -1 on failure
 */
int milk_ipc_connect_once(const char *sockpath);

/**
 * @brief Send `MILK_MSG_PING` to @sockpath and verify `MILK_MSG_DONE` within 2 s.
 *
 * @param sockpath Target UNIX socket path
 * @return 1 if server responded to ping, 0 otherwise
 */
int milk_ipc_ping_socket(const char *sockpath);

/**
 * @brief Send `MILK_MSG_SHUTDOWN` to @sockpath and wait at most 2 s for acknowledgment.
 *
 * @param sockpath Target UNIX socket path
 * @return 1 if shutdown frame was delivered, 0 if server was unreachable
 */
int milk_ipc_shutdown_socket(const char *sockpath);

/* How milk_ipc_spawn_daemon() ties a server to its owner, and what the server records */
typedef struct
{
    int      owner_rfd;   /* read end of the owner pipe, passed as fd 4 (-1: none) */
    pid_t    owner_pid;   /* without a pipe, it exits when this process exits (0: none) */
    pid_t    creator_pid; /* a named session's owner: recorded, not tied (0: none) */
    unsigned idle_s;      /* it ends after this long without a command (0: never) */
    int      is_named;    /* MILK_SRV_NAMED=1 in its environment */
} milk_spawn_opts_t;

/**
 * @brief Start a background `milk-cli -q --server` on @sockpath detached (milk_detach.h), its output
 * in the file next to its log, and wait up to 5 s for `READY=1`; a failure is printed with the
 * server's reason, or the end of its output. Return: 0 once it is ready, -1
 */
int milk_ipc_spawn_daemon(const char *binpath, const char *sockpath, const milk_spawn_opts_t *opts);

/** milk_ipc_shell_pid() - the shell a command comes from: the one `_MILK_LOCAL_SOCK` names, else
 * this process (a builtin) or its parent (milk-call). */
pid_t milk_ipc_shell_pid(const milk_client_hooks_t *hooks);

#endif /* MILK_IPC_COMMON_H */
