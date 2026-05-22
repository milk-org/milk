/**
 * @file    fps_valkey.c
 * @brief   FPS-to-Valkey bidirectional sync
 *
 * Implements push (local SHM -> Valkey via HSET +
 * PUBLISH) and pull (Valkey -> local SHM via a
 * PSUBSCRIBE subscriber thread).
 *
 * ## Push path (main thread)
 * On each detected local parameter change:
 *  1. HSET fps:<host>:<name> <keyword> <value>
 *  2. HSET fps:<host>:<name> _type.<keyword> <type>
 *  3. HSET fps:<host>:<name> _cnt0.<keyword> <cnt0>
 *  4. PUBLISH fps_update:<name> "<host> <kw> ..."
 *
 * ## Pull path (subscriber thread)
 * The subscriber thread blocks on valkeyGetReply()
 * after PSUBSCRIBE fps_update:*. On message:
 *  1. Parse "<host> <keyword> <value> <type> <cnt0>"
 *  2. Skip if host == our hostname (echo prevent)
 *  3. Connect to local FPS SHM, find param index
 *  4. Write value (type-aware), bump cnt0, signal
 *  5. Disconnect
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

#include "fps_valkey.h"
#include "fps_connect.h"
#include "fps_disconnect.h"
#include "fps_GetParamIndex.h"
#include "fps_GetTypeString.h"


/* ============================================
 * Internal helpers
 * ============================================ */

/**
 * @brief Check cmd_ctx for error and reconnect
 *
 * @param vctx  Valkey context
 * @return 0 if OK or reconnected, -1 on failure
 */
static int check_and_reconnect(FPS_VALKEY_CTX *vctx)
{
    if (vctx->cmd_ctx == NULL || vctx->cmd_ctx->err)
    {
        fprintf(stderr, "[fpsvalkey] cmd connection error: %s\n",
                vctx->cmd_ctx ? vctx->cmd_ctx->errstr : "NULL context");

        if (vctx->cmd_ctx)
        {
            valkeyFree(vctx->cmd_ctx);
        }

        vctx->cmd_ctx = valkeyConnect(vctx->server, vctx->port);

        if (vctx->cmd_ctx == NULL || vctx->cmd_ctx->err)
        {
            fprintf(stderr, "[fpsvalkey] reconnect failed\n");
            return -1;
        }

        fprintf(stderr, "[fpsvalkey] reconnected cmd_ctx\n");
    }
    return 0;
}


/**
 * @brief Get simple value string for a parameter
 *
 * Produces a plain value string (no stream info)
 * suitable for Valkey storage and Pub/Sub messages.
 *
 * @param fp      Parameter entry
 * @param buf     Output buffer
 * @param buflen  Buffer length
 */
static void param_value_to_string(FPS_PARAM *fp, char *buf, int buflen)
{
    switch (fp->type)
    {
    case FPTYPE_INT32:
        snprintf(buf, buflen, "%d", fp->val.i32[0]);
        break;
    case FPTYPE_UINT32:
        snprintf(buf, buflen, "%u", fp->val.ui32[0]);
        break;
    case FPTYPE_INT64:
        snprintf(buf, buflen, "%ld", fp->val.i64[0]);
        break;
    case FPTYPE_UINT64:
        snprintf(buf, buflen, "%lu", fp->val.ui64[0]);
        break;
    case FPTYPE_FLOAT32:
        snprintf(buf, buflen, "%.10g", fp->val.f32[0]);
        break;
    case FPTYPE_FLOAT64:
        snprintf(buf, buflen, "%.17g", fp->val.f64[0]);
        break;
    case FPTYPE_PID:
        snprintf(buf, buflen, "%ld", (long) fp->val.pid[0]);
        break;
    case FPTYPE_TIMESPEC:
        snprintf(buf, buflen, "%ld.%09ld", fp->val.ts[0].tv_sec, fp->val.ts[0].tv_nsec);
        break;
    case FPTYPE_ONOFF:
        snprintf(buf, buflen, "%s", fp->val.i64[0] ? "ON" : "OFF");
        break;
    case FPTYPE_STRING:
    case FPTYPE_FILENAME:
    case FPTYPE_FITSFILENAME:
    case FPTYPE_EXECFILENAME:
    case FPTYPE_DIRNAME:
    case FPTYPE_STREAMNAME:
    case FPTYPE_FPSNAME:
    case FPTYPE_PROCESS:
    case FPTYPE_STRING_NOT_STREAM:
        snprintf(buf, buflen, "%s", fp->val.string[0]);
        break;
    default:
        snprintf(buf, buflen, "?");
        break;
    }
}


/* ============================================
 * Subscriber thread
 * ============================================ */

/**
 * @brief Subscriber thread entry point
 *
 * Blocks on valkeyGetReply() in a loop. On each
 * incoming Pub/Sub message (PMESSAGE), parses
 * the payload and applies the value to local
 * FPS shared memory.
 *
 * Message format:
 *   "<hostname> <fpsname> <keyword> <value> <type>"
 *
 * @param arg  Pointer to FPS_VALKEY_CTX
 * @return NULL
 */
static void *subscriber_loop(void *arg)
{
    FPS_VALKEY_CTX *vctx  = (FPS_VALKEY_CTX *) arg;
    valkeyReply    *reply = NULL;

    while (vctx->sub_running)
    {
        int rc = valkeyGetReply(vctx->sub_ctx, (void **) &reply);

        if (rc != 0 || reply == NULL)
        {
            if (vctx->sub_running)
            {
                fprintf(stderr, "[fpsvalkey] sub read error, "
                                "exiting subscriber\n");
            }
            break;
        }

        /*
         * PMESSAGE reply is an array:
         *  [0] "pmessage"
         *  [1] pattern ("fps_update:*")
         *  [2] channel ("fps_update:<fpsname>")
         *  [3] message payload
         */
        if (reply->type == VALKEY_REPLY_ARRAY && reply->elements >= 4 &&
            reply->element[0]->type == VALKEY_REPLY_STRING &&
            strcmp(reply->element[0]->str, "pmessage") == 0)
        {
            const char *payload = reply->element[3]->str;

            /* Parse: host fpsname keyword value type */
            char msg_host[FPS_VALKEY_HOSTNAME_LEN];
            char msg_fpsname[STRINGMAXLEN_FPS_NAME];
            char msg_keyword[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN *
                             FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
            char msg_value[256];
            char msg_type[32];

            int nf = sscanf(payload, "%63s %99s %1279s %255s %31s", msg_host, msg_fpsname,
                            msg_keyword, msg_value, msg_type);

            if (nf < 5)
            {
                freeReplyObject(reply);
                continue;
            }

            /* Skip our own messages */
            if (strcmp(msg_host, vctx->hostname) == 0)
            {
                freeReplyObject(reply);
                continue;
            }

            /*
             * Apply to local SHM:
             * 1. Connect to the FPS
             * 2. Find param by keyword
             * 3. Write value
             * 4. Bump cnt0, signal UPDATE
             * 5. Disconnect
             */
            FPS  fps;
            long nbp = fps_connect(msg_fpsname, &fps, FPSCONNECT_SIMPLE);

            if (nbp == -1)
            {
                /* FPS not found locally — skip */
                freeReplyObject(reply);
                continue;
            }
            fps.NBparam = nbp;

            long pidx = functionparameter_GetParamIndex(&fps, msg_keyword);

            if (pidx == -1)
            {
                fps_disconnect(&fps);
                freeReplyObject(reply);
                continue;
            }

            int ret = functionparameter_SetParamValue_fromString(&fps, pidx, msg_value);

            if (ret == RETURN_SUCCESS)
            {
                printf("[fpsvalkey] PULL %s %s.%s"
                       " = %s (from %s)\n",
                       msg_type, msg_fpsname, msg_keyword, msg_value, msg_host);
                fflush(stdout);
            }

            fps_disconnect(&fps);
        }

        freeReplyObject(reply);
    }

    return NULL;
}


/* ============================================
 * Public API
 * ============================================ */

int fps_valkey_connect(FPS_VALKEY_CTX *vctx, const char *server, int port)
{
    memset(vctx, 0, sizeof(FPS_VALKEY_CTX));

    strncpy(vctx->server, server, FPS_VALKEY_HOSTNAME_LEN - 1);
    vctx->port        = port;
    vctx->sub_running = 0;

    /* Get local hostname */
    if (gethostname(vctx->hostname, FPS_VALKEY_HOSTNAME_LEN - 1) != 0)
    {
        strncpy(vctx->hostname, "unknown", FPS_VALKEY_HOSTNAME_LEN - 1);
    }

    /* Build prefix: "fps:<hostname>:" */
    snprintf(vctx->prefix, FPS_VALKEY_PREFIX_LEN, "fps:%s:", vctx->hostname);

    /* Command connection */
    struct timeval tv = { 2, 0 }; /* 2 sec timeout */
    vctx->cmd_ctx     = valkeyConnectWithTimeout(server, port, tv);
    if (vctx->cmd_ctx == NULL || vctx->cmd_ctx->err)
    {
        fprintf(stderr, "[fpsvalkey] cmd connect failed: %s\n",
                vctx->cmd_ctx ? vctx->cmd_ctx->errstr : "alloc error");
        return -1;
    }

    /* Subscriber connection */
    vctx->sub_ctx = valkeyConnectWithTimeout(server, port, tv);
    if (vctx->sub_ctx == NULL || vctx->sub_ctx->err)
    {
        fprintf(stderr, "[fpsvalkey] sub connect failed: %s\n",
                vctx->sub_ctx ? vctx->sub_ctx->errstr : "alloc error");
        valkeyFree(vctx->cmd_ctx);
        vctx->cmd_ctx = NULL;
        return -1;
    }

    printf("[fpsvalkey] Connected to %s:%d "
           "(hostname=%s)\n",
           server, port, vctx->hostname);

    return 0;
}


void fps_valkey_disconnect(FPS_VALKEY_CTX *vctx)
{
    fps_valkey_sub_stop(vctx);

    if (vctx->cmd_ctx)
    {
        valkeyFree(vctx->cmd_ctx);
        vctx->cmd_ctx = NULL;
    }

    printf("[fpsvalkey] Disconnected\n");
}


int fps_valkey_push_param(FPS_VALKEY_CTX *vctx,
                          const char     *fpsname,
                          const char     *keyword,
                          const char     *value,
                          const char     *typestr,
                          long            cnt0)
{
    if (check_and_reconnect(vctx) != 0)
    {
        return -1;
    }

    char hashkey[256];
    snprintf(hashkey, sizeof(hashkey), "%s%s", vctx->prefix, fpsname);

    char cnt0_field[320];
    snprintf(cnt0_field, sizeof(cnt0_field), "_cnt0.%s", keyword);

    char type_field[320];
    snprintf(type_field, sizeof(type_field), "_type.%s", keyword);

    char cnt0_str[32];
    snprintf(cnt0_str, sizeof(cnt0_str), "%ld", cnt0);

    /* Pipeline: 4 commands */
    valkeyAppendCommand(vctx->cmd_ctx, "HSET %s %s %s", hashkey, keyword, value);

    valkeyAppendCommand(vctx->cmd_ctx, "HSET %s %s %s", hashkey, type_field, typestr);

    valkeyAppendCommand(vctx->cmd_ctx, "HSET %s %s %s", hashkey, cnt0_field, cnt0_str);

    /* PUBLISH channel message */
    char channel[256];
    snprintf(channel, sizeof(channel), "fps_update:%s", fpsname);

    char msg[FPS_VALKEY_MSGBUF_LEN];
    snprintf(msg, sizeof(msg), "%s %s %s %s %s", vctx->hostname, fpsname, keyword, value, typestr);

    valkeyAppendCommand(vctx->cmd_ctx, "PUBLISH %s %s", channel, msg);

    /* Collect replies */
    valkeyReply *reply;
    for (int i = 0; i < 4; i++)
    {
        if (valkeyGetReply(vctx->cmd_ctx, (void **) &reply) != 0)
        {
            fprintf(stderr,
                    "[fpsvalkey] push pipeline "
                    "error at step %d\n",
                    i);
            return -1;
        }
        freeReplyObject(reply);
    }

    return 0;
}


int fps_valkey_push_metadata(FPS_VALKEY_CTX               *vctx,
                             const char                   *fpsname,
                             FUNCTION_PARAMETER_STRUCT_MD *md)
{
    if (check_and_reconnect(vctx) != 0)
    {
        return -1;
    }

    char hashkey[256];
    snprintf(hashkey, sizeof(hashkey), "%s%s", vctx->prefix, fpsname);

    /* UTC timestamp */
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm *ut = gmtime(&ts.tv_sec);
    char       tsbuf[64];
    snprintf(tsbuf, sizeof(tsbuf), "%04d-%02d-%02dT%02d:%02d:%02dZ", ut->tm_year + 1900,
             ut->tm_mon + 1, ut->tm_mday, ut->tm_hour, ut->tm_min, ut->tm_sec);

    char status_str[32];
    snprintf(status_str, sizeof(status_str), "0x%04x", md->status);

    char confpid_str[32];
    snprintf(confpid_str, sizeof(confpid_str), "%ld", (long) md->confpid);

    char runpid_str[32];
    snprintf(runpid_str, sizeof(runpid_str), "%ld", (long) md->runpid);

    /* Pipeline 4 HSETs */
    valkeyAppendCommand(vctx->cmd_ctx, "HSET %s _status %s", hashkey, status_str);

    valkeyAppendCommand(vctx->cmd_ctx, "HSET %s _confpid %s", hashkey, confpid_str);

    valkeyAppendCommand(vctx->cmd_ctx, "HSET %s _runpid %s", hashkey, runpid_str);

    valkeyAppendCommand(vctx->cmd_ctx, "HSET %s _lastsync %s", hashkey, tsbuf);

    valkeyReply *reply;
    for (int i = 0; i < 4; i++)
    {
        if (valkeyGetReply(vctx->cmd_ctx, (void **) &reply) != 0)
        {
            return -1;
        }
        freeReplyObject(reply);
    }

    return 0;
}


int fps_valkey_register_fps(FPS_VALKEY_CTX *vctx, const char *fpsname)
{
    if (check_and_reconnect(vctx) != 0)
    {
        return -1;
    }

    char setkey[256];
    snprintf(setkey, sizeof(setkey), "fps_list:%s", vctx->hostname);

    valkeyReply *reply =
        (valkeyReply *) valkeyCommand(vctx->cmd_ctx, "SADD %s %s", setkey, fpsname);

    if (reply == NULL)
    {
        return -1;
    }
    freeReplyObject(reply);

    return 0;
}


int fps_valkey_unregister_fps(FPS_VALKEY_CTX *vctx, const char *fpsname)
{
    if (check_and_reconnect(vctx) != 0)
    {
        return -1;
    }

    char setkey[256];
    snprintf(setkey, sizeof(setkey), "fps_list:%s", vctx->hostname);

    char hashkey[256];
    snprintf(hashkey, sizeof(hashkey), "%s%s", vctx->prefix, fpsname);

    /* Pipeline: SREM + DEL */
    valkeyAppendCommand(vctx->cmd_ctx, "SREM %s %s", setkey, fpsname);
    valkeyAppendCommand(vctx->cmd_ctx, "DEL %s", hashkey);

    valkeyReply *reply;
    for (int i = 0; i < 2; i++)
    {
        if (valkeyGetReply(vctx->cmd_ctx, (void **) &reply) != 0)
        {
            return -1;
        }
        freeReplyObject(reply);
    }

    return 0;
}


int fps_valkey_sub_start(FPS_VALKEY_CTX *vctx)
{
    if (vctx->sub_ctx == NULL)
    {
        fprintf(stderr, "[fpsvalkey] no sub connection\n");
        return -1;
    }

    /* Send PSUBSCRIBE */
    valkeyReply *reply = (valkeyReply *) valkeyCommand(vctx->sub_ctx, "PSUBSCRIBE fps_update:*");

    if (reply == NULL || vctx->sub_ctx->err)
    {
        fprintf(stderr, "[fpsvalkey] PSUBSCRIBE failed: %s\n", vctx->sub_ctx->errstr);
        if (reply)
        {
            freeReplyObject(reply);
        }
        return -1;
    }
    freeReplyObject(reply);

    vctx->sub_running = 1;

    int rc = pthread_create(&vctx->sub_thread, NULL, subscriber_loop, vctx);
    if (rc != 0)
    {
        fprintf(stderr, "[fpsvalkey] pthread_create failed\n");
        vctx->sub_running = 0;
        return -1;
    }

    printf("[fpsvalkey] Subscriber thread started\n");
    return 0;
}


void fps_valkey_sub_stop(FPS_VALKEY_CTX *vctx)
{
    if (!vctx->sub_running)
    {
        return;
    }

    vctx->sub_running = 0;

    /*
     * Close the sub connection to unblock
     * valkeyGetReply() in the subscriber thread.
     */
    if (vctx->sub_ctx)
    {
        valkeyFree(vctx->sub_ctx);
        vctx->sub_ctx = NULL;
    }

    pthread_join(vctx->sub_thread, NULL);

    printf("[fpsvalkey] Subscriber thread stopped\n");
}
