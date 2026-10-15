// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_client_internal.h
 * @brief Internal declarations shared between `milk_client_conn.c` and `milk_client_ops.c`.
 *
 * Role & Goal:
 *   Exposes the connection, lazy auto-start, variable-packing, and response
 *   loop helpers implemented in `milk_client_conn.c` to the public client
 *   operation entry points in `milk_client_ops.c`.
 */

#ifndef MILK_CLIENT_INTERNAL_H
#define MILK_CLIENT_INTERNAL_H

#include <stddef.h>

#include "milk_client.h"

/**
 * @brief Packed scalar shell variables forwarded with CALL / CALC / TEST requests.
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

void milk_client_collect_synced_vars(milk_packed_vars_t *pv, const milk_client_hooks_t *hooks);

int milk_client_await_response(int                        fd,
                               const char                *sockpath,
                               const milk_client_hooks_t *hooks,
                               char                      *value_buf,
                               size_t                     value_sz,
                               int                       *got_value);

#endif /* MILK_CLIENT_INTERNAL_H */
