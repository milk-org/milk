// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_server_internal.h
 * @brief Internal declarations shared across milk_server sub-modules.
 *
 * Role & Goal:
 *   Connects the server listener/connection loop (`milk_server.c`),
 *   command & expression dispatcher (`milk_server_dispatch.c`),
 *   variable synchronization & entity expansion (`milk_server_vars.c`),
 *   blocking stream wait/grab/I/O operations (`milk_server_stream.c`),
 *   and command listing/tab-completion (`milk_server_io.c`).
 */

#ifndef MILK_SERVER_INTERNAL_H
#define MILK_SERVER_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "milk_server.h"
#include "milk_srvlog.h"

/**
 * @brief Snapshot of a single server variable (`dcvar` or `cli_vars`) before command execution.
 */
typedef struct
{
    int    used;
    int    type;
    double f;
    long   l;
    char   s[80];
    char   name[80];
} dcvar_snap_t;

extern milk_srvlog_shm_t *g_srvlog_shm;
extern char               g_srvlog_sockpath[256];

/* --- milk_server_vars.c --- */

void milk_server_init_dcvar_cache(void);

void milk_server_sync_bash_vars(const uint8_t *buf,
                                uint32_t       len,
                                uint32_t       nvars,
                                uint32_t      *offset_io);

dcvar_snap_t *milk_server_snapshot_dcvar(long *out_nmax);

void milk_server_send_modified_dcvars(int client_fd, const dcvar_snap_t *snap, long snap_nmax);

int milk_server_is_entity_ns_prefix(const char *s, size_t nslen);

void milk_server_expand_entities_in_expr(const char *in, char *out, size_t outsz);

/* --- milk_server_stream.c --- */

int milk_server_waitfor(int client_fd, const char *target, const char *arg2, double timeout_sec);

int milk_server_grab(int         client_fd,
                     const char *sname_raw,
                     long        nframes,
                     const char *outname,
                     double      timeout_sec,
                     const char *fits_path,
                     const char *ts_path);

int milk_server_replay(int argc, char **argv);

int milk_server_streamcat(int argc, char **argv);

int milk_server_streamwrite(int argc, char **argv);

/* --- milk_server_io.c --- */

void milk_server_handle_complete(int client_fd, const char *prefix, const char *comp_type);

/* --- milk_server_dispatch.c --- */

int milk_server_dispatch_argv(int argc, char **argv);

int milk_server_exec_set(const uint8_t *payload, uint32_t len, char *log_cmd, size_t log_cmd_sz);

int milk_server_exec_calc_or_test(int            client_fd,
                                  int            is_test,
                                  const uint8_t *payload,
                                  uint32_t       len,
                                  char          *log_cmd,
                                  size_t         log_cmd_sz);

int milk_server_exec_call(int            client_fd,
                          int            listen_fd,
                          const uint8_t *payload,
                          uint32_t       len,
                          char          *log_cmd,
                          size_t         log_cmd_sz);

/* --- milk_server.c --- */

void milk_server_refresh_log_stats(void);

void milk_server_handle_client(int client_fd, int listen_fd, int *shutdown_flag);

#endif /* MILK_SERVER_INTERNAL_H */
