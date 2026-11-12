// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_client_internal.h
 * @brief What the client's files share: the connection, the packed shell variables, the reply
 *        loop.
 *
 * Purpose:     every client call reaches the session the same way: a kept connection, the shell
 *              variables a calculator expression names, one request function.
 * Does:        declares the connection calls of milk_client_conn.c (connect, or start the
 *              shell's own session; send a request and read its replies, milk_client_request();
 *              pack shell variables into milk_packed_vars_t) and the helpers of milk_client_ops.c
 *              that send a keyword as another command (milk_client_call_as()) or bind a call's
 *              output to a shell array (milk_client_capture_array()).
 * Used by:     milk_client_ops.c, milk_client_stream.c; milk_bash_hooks.c (an @on watcher drops
 *              the shell's owner pipe and kept connections).
 * Uses:        milk_client.h (the hooks).
 * See also:    milk_client_conn.c, milk_client_ops.c.
 */

#ifndef MILK_CLIENT_INTERNAL_H
#define MILK_CLIENT_INTERNAL_H

#include <stddef.h>

#include "milk_client.h"
#include "milk_proto.h"

/**
 * @brief Shell variables referenced by a calc/predicate expression, packed into RPC payloads.
 */
typedef struct
{
    char names[256][64];
    char vals[256][256];
    int  count;
} milk_packed_vars_t;

/** milk_client_connect_or_start() - the connection to the shell's session, started if needed; -1. */
int milk_client_connect_or_start(const milk_client_hooks_t *hooks);

/** milk_client_connect() - this process's connection to @sockpath, kept between requests; -1. */
int milk_client_connect(const char *sockpath);

void milk_client_close_owner_pipe(void);

void milk_client_collect_identifiers(milk_packed_vars_t        *pv,
                                     const milk_client_hooks_t *hooks,
                                     const char                *s);

/** milk_client_pack_digits() - pack the shell's MILK_DPDIGITS: the session prints with its digits. */
void milk_client_pack_digits(milk_packed_vars_t *pv, const milk_client_hooks_t *hooks);

/** milk_client_pack_vars() - append @pv to @p: the count, then each name and value. */
void milk_client_pack_vars(milk_payload_t *p, const milk_packed_vars_t *pv);

/**
 * milk_client_request() - send the request @type with payload @p (freed here) on the connection
 * @fd (milk_client_connect_or_start()), passing the @nfds descriptors @fds, wait for the reply
 * and end the request. A VALUE reply lands in @out, or on stdout if @out is NULL.
 * Return: the request's status; MILK_CLIENT_RC_TRANSPORT if @fd < 0, or if the request could not
 * be sent or the session did not answer (with a message).
 */
int milk_client_request(int                        fd,
                        const milk_client_hooks_t *hooks,
                        uint8_t                    type,
                        uint8_t                    flags,
                        milk_payload_t            *p,
                        const int                 *fds,
                        int                        nfds,
                        char                      *out,
                        size_t                     outsz);

int milk_client_has_arg(int argc, char **argv, const char *opt);

int milk_client_call_as(const char                *as,
                        int                        argc,
                        char                     **argv,
                        const char                *opt,
                        const char                *last,
                        int                        async_flag,
                        const milk_client_hooks_t *hooks);

int milk_client_capture_array(int (*fn)(int, char **, const milk_client_hooks_t *),
                              int                        argc,
                              char                     **argv,
                              const char                *extra,
                              const milk_client_hooks_t *hooks);

#endif /* MILK_CLIENT_INTERNAL_H */
