// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_srvlog.h
 * @brief Shared-memory server telemetry, multi-server management (@server),
 *        and high-performance searchable command/output ring-buffer log (@log).
 */

#ifndef MILK_SRVLOG_H
#define MILK_SRVLOG_H

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>
#include <time.h>

#include "milk_client.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define MILK_SRVLOG_MAGIC 0x4D534C47U /* "MSLG" */
#define MILK_SRVLOG_VERSION 1U
#define MILK_SRVLOG_CAP 1024U
#define MILK_SRVLOG_CMD_MAX 256
#define MILK_SRVLOG_OUT_MAX 512

#define MILK_SRVFLAG_NAMED 0x01U  /* Persistent named server (survives shell exit) */
#define MILK_SRVFLAG_TMUX 0x02U   /* Hosted inside tmux session milk_srv_<name>    */
#define MILK_SRVFLAG_INPROC 0x04U /* In-process server inside interactive milk-cli */

/**
 * @brief Single command/output entry in the per-server ring buffer.
 */
typedef struct
{
    uint64_t        seq;         /* Monotonic sequence number (1-based; 0 = empty) */
    struct timespec ts_start;    /* Wall-clock start timestamp (CLOCK_REALTIME)    */
    uint64_t        duration_ns; /* Execution duration in nanoseconds              */
    pid_t           client_pid;  /* Calling shell/client PID                       */
    uint8_t         msg_type;    /* Protocol opcode (MILK_MSG_CALL, CALC, etc.)    */
    int32_t         rc;          /* Command exit code (0 = success, != 0 = error)  */
    char            cmd[256];    /* Input command or expression                    */
    char            output[512]; /* Captured stdout + stderr snippet               */
} milk_srvlog_entry_t;

/**
 * @brief Shared-memory telemetry and circular log layout for a milk-cli server.
 */
typedef struct
{
    uint32_t            magic;
    uint32_t            version;
    pid_t               srv_pid;
    pid_t               owner_pid;
    uint32_t            flags;
    struct timespec     ts_created;
    char                srv_name[64];
    char                sockpath[128];
    char                tmux_session[64];
    char                cwd[256];
    uint32_t            nb_modules;
    uint32_t            nb_images;
    uint32_t            nb_vars;
    uint64_t            total_cmds;
    uint64_t            total_errors;
    uint64_t            write_seq;
    milk_srvlog_entry_t entries[MILK_SRVLOG_CAP];
} milk_srvlog_shm_t;

/**
 * @brief Parse a MILKCLI_SOCK string into its numeric UID and logical server name.
 */
void milk_srvlog_parse_sock(const char   *sockpath,
                            unsigned int *out_uid,
                            char         *out_name,
                            size_t        namesz);

/**
 * @brief Build a canonical MILKCLI_SOCK string from a logical server name or socket.
 */
void milk_srvlog_make_sock(const char *name_or_sock, char *out_sock, size_t socksz);

/**
 * @brief Resolve the filesystem path to a server's .log.shm file.
 */
int milk_srvlog_shm_path(const char *sockpath, char *out_path, size_t pathsz);

/**
 * @brief Create or attach to a server's shared-memory log and initialize server metadata.
 */
milk_srvlog_shm_t *milk_srvlog_server_init(const char *sockpath, pid_t owner_pid, uint32_t flags);

/**
 * @brief Update live server entity counts and working directory in shared memory.
 */
void milk_srvlog_server_update_stats(milk_srvlog_shm_t *shm,
                                     uint32_t           nb_modules,
                                     uint32_t           nb_images,
                                     uint32_t           nb_vars,
                                     const char        *cwd);

/**
 * @brief Append a completed command and its output snippet to the ring buffer.
 */
void milk_srvlog_record(milk_srvlog_shm_t     *shm,
                        const struct timespec *ts_start,
                        uint64_t               duration_ns,
                        pid_t                  client_pid,
                        uint8_t                msg_type,
                        int32_t                rc,
                        const char            *cmd,
                        const char            *output);

/**
 * @brief Append a client-side fast-path operation (@get/@set) to the session's SHM log.
 */
void milk_srvlog_client_record(const char            *sockpath,
                               const struct timespec *ts_start,
                               uint64_t               duration_ns,
                               uint8_t                msg_type,
                               int32_t                rc,
                               const char            *cmd,
                               const char            *output);

/**
 * @brief Unmap the server's shared-memory log and optionally unlink the backing file.
 */
void milk_srvlog_server_close(milk_srvlog_shm_t *shm, int unlink_file);

/**
 * @brief Execute the `@log` / `@server log` query, search, or live-follow command.
 */
int milk_srvlog_cmd_log(const char                *default_sockpath,
                        int                        argc,
                        char                     **argv,
                        const milk_client_hooks_t *hooks);

/**
 * @brief Execute the `@server` management command (status, list, start, use, local, stop, attach).
 */
int milk_srvlog_cmd_server(int argc, char **argv, const milk_client_hooks_t *hooks);

#ifdef __cplusplus
}
#endif

#endif /* MILK_SRVLOG_H */
