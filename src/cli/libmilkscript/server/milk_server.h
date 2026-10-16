// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_server.h
 * @brief Milk UNIX domain socket command server public API.
 *
 * Role & Goal:
 *   Declares the entry points for starting and driving the persistent
 *   milk-cli command server over a UNIX domain socket (`MILKCLI_SOCK`).
 *   The server hosts all loaded milk modules, shared-memory streams, FPS
 *   connections, and local variables/images in memory, servicing client
 *   requests from interactive bash sessions, `.milk` scripts, and `milk-call`.
 */

#ifndef MILK_SERVER_H
#define MILK_SERVER_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "milk_shm_prop.h"

/**
 * @brief Dump all registered canonical and unambiguous short command names.
 *
 * @param fp Output stream
 */
void milk_server_dump_commands(FILE *fp);

/**
 * @brief Create, bind, and listen on the milk UNIX domain socket.
 *
 * Also sets MILKCLI_SOCK in the environment and snapshots initial dcvar
 * entries into the server cache.
 *
 * @param sockpath Socket path ("@abstract" or "/path/to/sock")
 * @return Listening socket fd (>= 0) on success, -1 on error
 */
int milk_server_create_listener(const char *sockpath);

/**
 * @brief Accept and service one pending connection on @listen_fd.
 *
 * @param listen_fd     Listening UNIX domain socket fd
 * @param shutdown_flag Set to 1 if client requested server shutdown
 */
void milk_server_accept_and_handle(int listen_fd, int *shutdown_flag);

/** Return 1 (once) if a CALL was aborted by SIGINT since the last query. */
int milk_server_take_call_interrupt(void);

/**
 * @brief Run the milk UNIX domain socket command server loop.
 *
 * @param sockpath Socket path ("@abstract" or "/path/to/sock")
 * @param owner_fd Optional fd held by owning shell (-1 if none)
 * @param ready_fd Optional fd to write "READY\n" once listening (-1 if none)
 * @return 0 on clean shutdown, non-zero on error
 */
int milk_server_run(const char *sockpath, int owner_fd, int ready_fd);

/**
 * @brief Set additional flags on the active server SHM log header.
 *
 * @param flags Bitmask of MILK_SRVFLAG_* flags to OR into shm->flags
 */
void milk_server_set_log_flags(uint32_t flags);

/**
 * @brief Close (and optionally unlink) the active server SHM log.
 *
 * @param unlink_shm If non-zero, unlink the SHM file from /dev/shm
 */
void milk_server_close_log(int unlink_shm);

#endif /* MILK_SERVER_H */
