// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_script_expand_internal.h
 *
 * @brief Internal shared types for the expand
 *        sub-modules.
 *
 * Not installed — library-private only.
 *
 * Exposes ArithParser and arith_expr() so that
 * CLIcore_script_expand_fps.c can call them
 * from expand_fpsvar_write() without duplicating
 * the definition.
 */

#ifndef CLICORE_SCRIPT_EXPAND_INTERNAL_H
#define CLICORE_SCRIPT_EXPAND_INTERNAL_H

typedef struct
{
    const char *s;
    int         pos;
} ArithParser;

double arith_expr(ArithParser *p);

#endif /* CLICORE_SCRIPT_EXPAND_INTERNAL_H */
