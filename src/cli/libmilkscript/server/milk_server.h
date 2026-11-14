// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_server.h
 * @brief Milk UNIX domain socket command server public API: the process of a session.
 *
 * Purpose:     the state that outlives a command (loaded modules, local images, calculator
 *              variables, settings) lives in one process, the session's server, which the
 *              shells that use the session send their commands to (socket $MILKCLI_SOCK).
 * Does:        declares the server's entry points: listen and run the loop until the shutdown
 *              command, SIGTERM, the end of the owner or --idle (`milk-cli --server`); or serve
 *              requests from another loop (milk_server_ready_fd(),
 *              milk_server_accept_and_handle()), as milk-cli and milk-script do for their own
 *              session; run the lines of milk-cli's calc mode in its process; the session's
 *              log flags and name; milk_complete(), the one completer.
 * Used by:     CLIcore.c (`milk-cli --server`); milk_bash_coproc.c (the session of milk-cli and
 *              milk-script, served while their bash runs); milk_bash_exec.c (calc mode);
 *              CLIcore_UI_completion_tab.c (milk_complete()); CLIcore_UI_history.c,
 *              CLIcore_UI_history_display.c.
 * See also:    milk_server.c, milk_server_internal.h (the request handlers);
 *              docs/guide/sessions.md, "Concurrent Commands Across Shells".
 */

#ifndef MILK_SERVER_H
#define MILK_SERVER_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/types.h>

#include "milk_shm_prop.h"


/**
 * @brief Create, bind, and listen on the milk UNIX domain socket.
 *
 * Also sets MILKCLI_SOCK in the environment. Poll milk_server_ready_fd(), not the listening socket:
 * a shell keeps its connection between requests.
 *
 * @param sockpath Socket path ("@abstract" or "/path/to/sock")
 * @return Listening socket fd (>= 0) on success, -1 on error
 */
int milk_server_create_listener(const char *sockpath);

/** milk_server_ready_fd() - readable when a shell connects, or sends a request on the connection it
 * keeps; then call milk_server_accept_and_handle(). -1 before milk_server_create_listener(). */
int milk_server_ready_fd(void);

/**
 * @brief Serve what milk_server_ready_fd() reports: new connections and the requests of kept ones.
 *
 * @param listen_fd     Listening UNIX domain socket fd
 * @param shutdown_flag Set to 1 if client requested server shutdown
 */
void milk_server_accept_and_handle(int listen_fd, int *shutdown_flag);

/** Return 1 (once) if a CALL was aborted by SIGINT since the last query. */
int milk_server_take_call_interrupt(void);

/**
 * @brief Run the milk UNIX domain socket command server loop: until its shutdown command, SIGTERM,
 * the end of its owner (@owner_fd), or @idle_s seconds without a command.
 */
int milk_server_run(const char *sockpath,
                    int         owner_fd,
                    int         ready_fd,
                    pid_t       owner_pid,
                    unsigned    idle_s);

/**
 * @brief Set additional flags on the active server SHM log header.
 *
 * @param flags Bitmask of MILK_SRVFLAG_* flags to OR into shm->flags
 */
void milk_server_set_log_flags(uint32_t flags);

/** milk_server_session_name() - the name of the session this process serves ("" before it listens). */
const char *milk_server_session_name(void);

/**
 * @brief Close (and optionally unlink) the active server SHM log.
 *
 * @param unlink_shm If non-zero, unlink the SHM file from /dev/shm
 */
void milk_server_close_log(int unlink_shm);

void milk_server_reset_bash_epoch(void);
void milk_server_bump_bash_epoch(void);
int  milk_server_can_exec_calc_inproc(const char *expr);
int  milk_server_exec_calc_repl_inproc(const char *expr);

/** milk_complete() - the completion candidates of word @prefix, one per line, into @out, for
 * @comp_type ("cmd", "arg:<cmd>:<cword>[:<previous word>]", "calc", ...); the length written. */
size_t milk_complete(const char *prefix, const char *comp_type, char *out, size_t outsz);

/** milk_server_split_words() - split command line @step into at most @max_args unquoted words,
 * kept in @arena; the number of words. */
int milk_server_split_words(const char *step,
                            char      **argv_out,
                            int         max_args,
                            char       *arena,
                            size_t      arena_sz);

#endif /* MILK_SERVER_H */
