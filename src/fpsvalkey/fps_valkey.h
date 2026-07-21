// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_valkey.h
 * @brief   FPS-to-Valkey bidirectional sync API
 *
 * Provides push (local SHM -> Valkey) and pull
 * (Valkey -> local SHM via Pub/Sub) for FPS
 * parameters. Uses two valkeyContext connections:
 * one synchronous for commands, one blocking in a
 * dedicated pthread for PSUBSCRIBE.
 */

#ifndef FPS_VALKEY_H
#define FPS_VALKEY_H

#include <pthread.h>
#include <valkey/valkey.h>

#include "fps.h"

#define FPS_VALKEY_HOSTNAME_LEN 64
#define FPS_VALKEY_PREFIX_LEN 128
#define FPS_VALKEY_MSGBUF_LEN 512

/**
 * @brief Valkey connection context
 *
 * Holds two connections (command + subscriber),
 * hostname for echo prevention, and thread state.
 */
typedef struct
{
    valkeyContext *cmd_ctx;
    valkeyContext *sub_ctx;

    pthread_t    sub_thread;
    volatile int sub_running;

    char hostname[FPS_VALKEY_HOSTNAME_LEN];
    char prefix[FPS_VALKEY_PREFIX_LEN];

    char server[FPS_VALKEY_HOSTNAME_LEN];
    int  port;
} FPS_VALKEY_CTX;

/**
 * @brief Connect to Valkey server
 *
 * Opens two connections (cmd + sub). Populates
 * hostname via gethostname().
 *
 * @param[out] vctx    Context to initialize
 * @param[in]  server  Valkey server address
 * @param[in]  port    Valkey server port
 * @return 0 on success, -1 on failure
 */
int fps_valkey_connect(FPS_VALKEY_CTX *vctx, const char *server, int port);

/**
 * @brief Disconnect and free Valkey resources
 *
 * Stops subscriber thread and frees both
 * connections.
 *
 * @param[in,out] vctx  Context to tear down
 */
void fps_valkey_disconnect(FPS_VALKEY_CTX *vctx);

/**
 * @brief Push one parameter to Valkey
 *
 * Writes HSET (value, type, cnt0) and PUBLISHes
 * a notification message on fps_update:<fpsname>.
 *
 * @param[in] vctx     Valkey context
 * @param[in] fpsname  FPS instance name
 * @param[in] keyword  Parameter keyword
 * @param[in] value    Value as string
 * @param[in] typestr  Type name (e.g. "FLOAT64")
 * @param[in] cnt0     Change counter
 * @return 0 on success, -1 on error
 */
int fps_valkey_push_param(FPS_VALKEY_CTX *vctx,
                          const char     *fpsname,
                          const char     *keyword,
                          const char     *value,
                          const char     *typestr,
                          long            cnt0);

/**
 * @brief Push FPS metadata to Valkey
 *
 * Writes _status, _confpid, _runpid, _lastsync
 * fields into the FPS hash.
 *
 * @param[in] vctx     Valkey context
 * @param[in] fpsname  FPS instance name
 * @param[in] md       FPS metadata struct
 * @return 0 on success, -1 on error
 */
int fps_valkey_push_metadata(FPS_VALKEY_CTX               *vctx,
                             const char                   *fpsname,
                             FUNCTION_PARAMETER_STRUCT_MD *md);

/**
 * @brief Register FPS in Valkey fps_list set
 *
 * @param[in] vctx     Valkey context
 * @param[in] fpsname  FPS instance name
 * @return 0 on success, -1 on error
 */
int fps_valkey_register_fps(FPS_VALKEY_CTX *vctx, const char *fpsname);

/**
 * @brief Unregister FPS from Valkey
 *
 * Removes from fps_list set and deletes the
 * FPS hash key.
 *
 * @param[in] vctx     Valkey context
 * @param[in] fpsname  FPS instance name
 * @return 0 on success, -1 on error
 */
int fps_valkey_unregister_fps(FPS_VALKEY_CTX *vctx, const char *fpsname);

/**
 * @brief Start subscriber thread
 *
 * Sends PSUBSCRIBE fps_update:* on sub_ctx and
 * spawns a pthread that blocks on valkeyGetReply.
 * Incoming messages are applied to local FPS SHM.
 *
 * @param[in,out] vctx  Valkey context
 * @return 0 on success, -1 on error
 */
int fps_valkey_sub_start(FPS_VALKEY_CTX *vctx);

/**
 * @brief Stop subscriber thread
 *
 * Signals thread to stop, closes sub_ctx to
 * unblock valkeyGetReply, and joins the thread.
 *
 * @param[in,out] vctx  Valkey context
 */
void fps_valkey_sub_stop(FPS_VALKEY_CTX *vctx);

#endif /* FPS_VALKEY_H */
