// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_client_internal.h
 * @brief Internal declarations shared across `milk_client_conn.c`, `milk_client_ops.c`, and `milk_client_ext.c`.
 *
 * Role & Goal:
 *   Exposes the connection, lazy auto-start, variable-packing, response-loop,
 *   and array-capture helpers shared across the `ipc/` client implementation files.
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
    char vals[256][128];
    int  count;
} milk_packed_vars_t;

int milk_client_connect_or_start(const milk_client_hooks_t *hooks,
                                 char                      *sockpath_out,
                                 size_t                     sockpath_sz);

void milk_client_close_owner_pipe(void);

void milk_client_collect_identifiers(milk_packed_vars_t        *pv,
                                     const milk_client_hooks_t *hooks,
                                     const char                *s);

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

int milk_client_is_entity_spec_token(const char *s);

#endif /* MILK_CLIENT_INTERNAL_H */
