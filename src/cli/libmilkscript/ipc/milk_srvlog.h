// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_srvlog.h
 * @brief Shared-memory server telemetry, sessions (@session),
 *        and high-performance searchable command/output ring-buffer log (@log).
 */

#ifndef MILK_SRVLOG_H
#define MILK_SRVLOG_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/types.h>
#include <time.h>

#include "milk_client.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define MILK_SRVLOG_MAGIC 0x4D534C47U /* "MSLG" */
#define MILK_SRVLOG_VERSION 2U        /* 2: owner_ident, idle_s */
#define MILK_SRVLOG_CAP 1024U
#define MILK_SRVLOG_CMD_MAX 256
#define MILK_SRVLOG_OUT_MAX 512

#define MILK_SRVFLAG_NAMED 0x01U  /* Persistent named server (survives shell exit) */
#define MILK_SRVFLAG_INPROC 0x04U /* In-process server inside interactive milk-cli */
#define MILK_SRVFLAG_SCRIPT 0x08U /* ... inside milk-script */
/* 0x02 marked a server hosted in tmux (`@server start --tmux`, removed) */

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
    uint32_t        magic;
    uint32_t        version;
    pid_t           srv_pid;
    pid_t           owner_pid;
    uint32_t        flags;
    struct timespec ts_created;
    char            srv_name[64];
    char            sockpath[128];
    char            ident[64]; /* srv_pid's start ticks and boot id, "%llu %s" (milk_detach.h) */
    char            owner_ident[64]; /* the same for owner_pid ("" when unknown) */
    uint32_t        idle_s;          /* it ends after this long without a command (0: never) */
    char            cwd[256];
    uint32_t        nb_modules;
    uint32_t        nb_images;
    uint32_t        nb_vars;
    uint64_t        total_cmds;
    uint64_t        total_errors;
    uint64_t        write_seq;
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
 * @brief Path of a server's output file, next to its log (milk-srv.<uid>.<name>.out).
 */
int milk_srvlog_out_path(const char *sockpath, char *out_path, size_t pathsz);

/**
 * @brief Create or attach to a server's shared-memory log and initialize server metadata.
 */
milk_srvlog_shm_t *milk_srvlog_server_init(const char *sockpath, pid_t owner_pid, uint32_t flags);

/**
 * @brief Record @pid, with its start ticks and boot id, as the owner of the session (0: none).
 */
void milk_srvlog_set_owner(milk_srvlog_shm_t *shm, pid_t pid);

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
 * @brief Execute the `@log` / `@session log` query, search, or live-follow command.
 */
int milk_srvlog_cmd_log(const char                *default_sockpath,
                        int                        argc,
                        char                     **argv,
                        const milk_client_hooks_t *hooks);

/**
 * @brief The number of live shells that use session @sockpath (their override file names it).
 */
int milk_srvlog_session_shells(const char *sockpath);

/**
 * @brief Write "# unit NAME: running|stopped" to @fp for each unit whose last run or conf names
 * session @name. Return: how many
 */
int milk_srvlog_session_list_units(const char *name, FILE *fp);

/**
 * @brief Session @name's socket in @sock, after starting it if it does not run (`milk-cli --session`;
 * @bin: milk-cli, NULL to look it up). Return: 0, 1 if it does not start, 2 for a bad name
 */
int milk_srvlog_session_join(const char *name, const char *bin, char *sock, size_t socksz);

/**
 * @brief Execute `@session` (status, ls, new, use, leave, who, end, prune, save, log).
 */
int milk_srvlog_cmd_session(int argc, char **argv, const milk_client_hooks_t *hooks);

#ifdef __cplusplus
}
#endif

#endif /* MILK_SRVLOG_H */
