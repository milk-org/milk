// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_client_internal.h
 * @brief What the client's files share: the connection, the packed shell variables, the reply
 *        loop.
 *
 * Purpose:     every client call reaches the session the same way: a kept connection, the shell
 *              variables a calculator expression names, one reply loop.
 * Does:        declares the connection calls of milk_client_conn.c (connect, or start the
 *              shell's own session; end a request; read the replies; pack shell variables into
 *              milk_packed_vars_t) and the helpers of milk_client_ops.c that send a keyword as
 *              another command (milk_client_call_as()) or bind a call's output to a shell array
 *              (milk_client_capture_array()).
 * Used by:     milk_client_ops.c, milk_client_stream.c; milk_bash_hooks.c (an @on watcher drops
 *              the shell's owner pipe and kept connections).
 * Uses:        milk_client.h (the hooks).
 * See also:    milk_client_conn.c, milk_client_ops.c.
 */

#ifndef MILK_CLIENT_INTERNAL_H
#define MILK_CLIENT_INTERNAL_H

#include <stddef.h>

#include "milk_client.h"

/**
 * @brief Shell variables referenced by a calc/predicate expression, packed into RPC payloads.
 */
typedef struct
{
    char names[256][64];
    char vals[256][256];
    int  count;
} milk_packed_vars_t;

int milk_client_connect_or_start(const milk_client_hooks_t *hooks,
                                 char                      *sockpath_out,
                                 size_t                     sockpath_sz);

/** milk_client_connect() - this process's connection to @sockpath, kept between requests; -1. */
int milk_client_connect(const char *sockpath);

/** milk_client_end() - end the request on @fd: keep the connection if its reply said so. */
void milk_client_end(int fd);

void milk_client_close_owner_pipe(void);

void milk_client_collect_identifiers(milk_packed_vars_t        *pv,
                                     const milk_client_hooks_t *hooks,
                                     const char                *s);

/** milk_client_pack_digits() - pack the shell's MILK_DPDIGITS: the session prints with its digits. */
void milk_client_pack_digits(milk_packed_vars_t *pv, const milk_client_hooks_t *hooks);

int milk_client_await_response(int                        fd,
                               const char                *sockpath,
                               const milk_client_hooks_t *hooks,
                               char                      *value_buf,
                               size_t                     value_sz,
                               int                       *got_value);

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
