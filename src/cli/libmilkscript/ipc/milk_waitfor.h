// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_waitfor.h
 * @brief `@waitfor` target parsing, state aliasing and wait loops shared by the client and the server.
 *
 * The client runs these in-process for streams and shm entities (no RPC); the server runs them for
 * targets only it can read (local images, variables) and for calc expressions.
 */

#ifndef MILK_WAITFOR_H
#define MILK_WAITFOR_H

#include <stddef.h>
#include <stdint.h>

/** Return codes beyond 0 (condition met): */
#define MILK_WAIT_TIMEOUT 124
#define MILK_WAIT_CANCEL 130
#define MILK_WAIT_ABSENT (-1) /**< stream does not exist and the caller asked not to wait for it */

typedef enum
{
    MILK_WAIT_BAD = 0, /**< malformed target */
    MILK_WAIT_EXPR,    /**< boolean calc expression (`<`, `>`, `==`, `!=`, `&&`, `||`) */
    MILK_WAIT_EQ, /**< `entity=value`, or `p.<name> [state]` rewritten to `p.<name>.status=<STATE>` */
    MILK_WAIT_STREAM /**< frame count on a stream */
} milk_wait_kind_t;

typedef struct
{
    milk_wait_kind_t kind;
    char             lhs[256];    /**< EQ: entity to read */
    char             expect[256]; /**< EQ: value to wait for */
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

milk_wait_kind_t milk_waitfor_parse(const char *target, const char *state, milk_wait_target_t *t);

int milk_waitfor_value_matches(const char *val, const char *expect);

double milk_wait_now(void);

void milk_wait_backoff(int *iter);

int milk_waitfor_eq(const char            *lhs,
                    const char            *expect,
                    double                 t_start,
                    double                 timeout_sec,
                    const milk_wait_ops_t *ops);

int milk_waitfor_stream(const char            *sname,
                        long                   nframes,
                        int                    req_sem,
                        const uint64_t        *abs_cnt,
                        int                    wait_create,
                        double                 t_start,
                        double                 timeout_sec,
                        const milk_wait_ops_t *ops);

#endif /* MILK_WAITFOR_H */
