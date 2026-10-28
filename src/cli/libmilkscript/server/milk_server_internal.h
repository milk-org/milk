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

#include <signal.h>
#include <stddef.h>
#include <stdint.h>

#include "milk_server.h"
#include "milk_srvlog.h"


extern milk_srvlog_shm_t *g_srvlog_shm;
extern char               g_srvlog_sockpath[256];

/* --- milk_server_vars.c --- */

void milk_server_init_dcvar_cache(void);

void milk_server_sync_bash_vars(const uint8_t *buf,
                                uint32_t       len,
                                uint32_t       nvars,
                                uint32_t      *offset_io,
                                int            is_repl);

#define MILK_OUTVARS_MAX 32 /* declared outputs tracked per command */

/** milk_server_outvars_snap() - fingerprint the declared outputs @outvars (CLICMDDATA.outvars) before
 * their command runs, into @h[MILK_OUTVARS_MAX]; returns how many. */
int milk_server_outvars_snap(const char *outvars, uint64_t *h);

/** milk_server_outvars_diff() - after the command: queue the outputs whose fingerprint changed. */
void milk_server_outvars_diff(const char *outvars, const uint64_t *h, int n);

/** milk_server_outvars_send() - send the queued outputs to the caller as SETVAR (none if @client_fd < 0)
 * and forget them. */
void milk_server_outvars_send(int client_fd);


void milk_server_expand_entities_in_expr(const char *in, char *out, size_t outsz);

void milk_server_mark_expr_vars_synced(const char *expr);

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

int milk_server_resolve_cmd_index(const char *name, int *out_ambig);

int milk_server_set_argv(int argc, char **argv);

int milk_server_dispatch_argv(int argc, char **argv);

/** milk_server_module() - `@module`; milk_server_config() - `@config` (its server settings). */
int milk_server_module(int argc, char **argv);
int milk_server_config(int argc, char **argv);

/** milk_server_history() - run `@history <sub>` (*@handled set), or leave plain `@history` to the caller. */
int milk_server_history(int argc, char **argv, int *handled);

/** milk_server_session_save() - `@session save PATH`: this session as a milk script, images in PATH.d/. */
int milk_server_session_save(const char *path);

/** milk_server_serve_waiting() - serve, in order, the requests that waited for the CALL that just ran. */
void milk_server_serve_waiting(int listen_fd, int *shutdown_flag);

int milk_server_exec_set(const uint8_t *payload, uint32_t len, char *log_cmd, size_t log_cmd_sz);

int milk_server_exec_call(int            client_fd,
                          int            listen_fd,
                          const uint8_t *payload,
                          uint32_t       len,
                          char          *log_cmd,
                          size_t         log_cmd_sz);

/* --- milk_server_eval.c --- */

int milk_server_assert(int argc, char **argv);

int milk_server_batch_or_pipe(int argc, char **argv);

int milk_server_exec_calc_or_test(int            client_fd,
                                  int            is_test,
                                  uint8_t        flags,
                                  const uint8_t *payload,
                                  uint32_t       len,
                                  char          *log_cmd,
                                  size_t         log_cmd_sz);

/* --- milk_server.c --- */

void milk_server_refresh_log_stats(void);

int milk_server_handle_client(int client_fd, int listen_fd, int *shutdown_flag);

/** Serializes everything that touches milk state (data.*, calc, stdio/cwd swaps); recursive. */
void milk_server_cmd_lock(void);
void milk_server_cmd_unlock(void);

/** Run WAITFOR/GRAB on @client_fd in a detached thread that then owns the fd; 0 on success. */
int milk_server_spawn_stream_worker(int client_fd);

/** Set when SHUTDOWN arrives outside the main loop (e.g. during a CALL); the main loop stops on it. */
extern volatile sig_atomic_t g_milk_server_stop;

/** Reset the CALL watcher in a forked child (its thread is not copied). */
void milk_server_watcher_atfork_child(void);

/** Set after forking a worker; the server then reaps exited children. */
extern volatile sig_atomic_t g_srv_children;

#endif /* MILK_SERVER_INTERNAL_H */
