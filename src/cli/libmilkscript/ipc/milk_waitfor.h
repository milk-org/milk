// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_waitfor.h
 * @brief `@waitfor` target parsing, state aliasing and wait loops shared by the client and the
 *        server.
 *
 * Purpose:     `@waitfor` reads a target and waits for it the same way in a shell and in a
 *              session.
 * Does:        classifies a target: frames of a stream, an entity value (`f.loop.gain=0.5`; a
 *              unit state for `f.<name> [state]`, a process state for `p.<name> [state]`, with
 *              their aliases: RUN or 1 is ACTIVE) or a calculator expression; matches a value
 *              (numbers by value); waits for frames on a stream semaphore, first for the stream
 *              to be created if need be; polls an entity value until it matches. A wait ends
 *              after its timeout (124) or when its caller cancels it (130).
 * Used by:     the client (milk_client_stream.c), which waits in the shell for streams and
 *              shared-memory entities, with no request; the server (milk_server_stream.c), for
 *              what only it can read (local images, variables).
 * Uses:        ImageStreamIO, libmilkprop (milk_shm_cache.h: pinned streams, milk_shm_prop.h:
 *              entity values, milk_entity_ns.h).
 * See also:    milk_waitfor.c; docs/guide/scripting.md, "Waiting, Events and Cleanup".
 * Constraints: compiled into libmilkbash.so, the thin client every shell loads.
 */

#ifndef MILK_WAITFOR_H
#define MILK_WAITFOR_H

#include <stddef.h>
#include <stdint.h>

#include "ImageStreamIO/ImageStruct.h"

/** Return codes beyond 0 (condition met): */
#define MILK_WAIT_TIMEOUT 124
#define MILK_WAIT_CANCEL 130
#define MILK_WAIT_ABSENT (-1) /**< stream does not exist and the caller asked not to wait for it */
#define MILK_WAIT_FULL (-2)   /**< every stream-cache slot of this process is pinned by a waiter */

typedef enum
{
    MILK_WAIT_BAD = 0, /**< malformed target */
    MILK_WAIT_EXPR,    /**< boolean calc expression (`<`, `>`, `==`, `!=`, `&&`, `||`) */
    MILK_WAIT_EQ, /**< `entity=value`, or `p.<name> [state]` / `f.<name> [state]` rewritten to a property */
    MILK_WAIT_STREAM /**< frame count on a stream */
} milk_wait_kind_t;

typedef struct
{
    milk_wait_kind_t kind;
    char             lhs[256];    /**< EQ: entity to read */
    char             expect[256]; /**< EQ: value to wait for */
    int              any;         /**< EQ: any value, i.e. the entity exists (expect unused) */
    char             why[160];    /**< BAD: what is wrong with the target */
    char             stream[256]; /**< STREAM: stream name without its namespace prefix */
    int              local;       /**< STREAM: target names a local (`l.`, `L:`) image */
} milk_wait_target_t;

/** Callbacks of a wait loop; all optional. lock/unlock bracket each entity read (server command lock). */
typedef struct
{
    int (*cancelled)(void *ctx); /**< non-zero stops the wait with MILK_WAIT_CANCEL */
    void (*lock)(void *ctx);
    void (*unlock)(void *ctx);
    void *ctx;
} milk_wait_ops_t;

/** milk_waitfor_parse() - classify @target into @t; @state is the optional state of `p.<name>` or `f.<name>`. */
milk_wait_kind_t milk_waitfor_parse(const char *target, const char *state, milk_wait_target_t *t);

int milk_waitfor_value_matches(const char *val, const char *expect);

double milk_wait_now(void);

void milk_wait_backoff(int *iter);

/** milk_waitfor_eq() - poll entity @lhs until it matches @expect (NULL: until it exists). */
int milk_waitfor_eq(const char            *lhs,
                    const char            *expect,
                    double                 t_start,
                    double                 timeout_sec,
                    const milk_wait_ops_t *ops);

/** milk_waitfor_open() - open stream @sname into @img if its file exists (ImageStreamIO warns about a missing
 * one); 0 when open. */
int milk_waitfor_open(IMAGE *img, const char *sname);

int milk_waitfor_stream(const char            *sname,
                        long                   nframes,
                        int                    req_sem,
                        const uint64_t        *abs_cnt,
                        int                    wait_create,
                        double                 t_start,
                        double                 timeout_sec,
                        const milk_wait_ops_t *ops);

#endif /* MILK_WAITFOR_H */
