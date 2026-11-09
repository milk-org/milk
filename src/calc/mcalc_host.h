// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mcalc_host.h
 * @brief Host-environment bridge for string variables in `libmilkcalc`.
 *
 * Purpose:     the calculator keeps numeric variables itself (milk_data.variable[]) but string
 *              variables belong to the program that hosts it. libmilkcalc (Calculator tier)
 *              cannot depend on libmilkscript (CLI tier), which links it, nor include a header
 *              from src/cli (cmake/milk_layer_guards.cmake): the host registers callbacks.
 * Does:        mcalc_set_host() registers the host's get and set callbacks; the calculator
 *              reads, assigns and removes (NULL value) string variables through
 *              mc_host_str_get() and mc_host_str_set(). With no host registered
 *              (milk-fpsexec-milk-calc), they use a process-local table of 128 entries.
 * Used by:     CLIcore_vars.c, which registers cli_var_get() and cli_var_set() when
 *              libmilkscript loads; front/mc_resolve.c (name lookup) and run/mc_scalar_eval.c
 *              (reads and assignments).
 * See also:    mc_eval.c (implementation and fallback table); CLIcore_vars.c (the host's
 *              table); docs/guide/calculator.md, "Calculator (`@calc`, `@test`, `@pipe`)".
 */

#ifndef MILK_MCALC_HOST_H
#define MILK_MCALC_HOST_H

/* The host's string variables; libmilkscript registers its implementation (CLIcore_vars.c). */
typedef struct
{
    const char *(*str_get)(const char *name);
    void (*str_set)(const char *name, const char *val);
} mcalc_host_ops;

/** mcalc_set_host() - register the host's string-variable callbacks (NULL @ops: none). */
void mcalc_set_host(const mcalc_host_ops *ops);

/** mc_host_str_get() - look up string variable @name via the host or internal fallback table. */
const char *mc_host_str_get(const char *name);

/** mc_host_str_set() - store string variable @name via the host or internal fallback table;
 * a NULL @val removes it. */
void mc_host_str_set(const char *name, const char *val);

/** mc_host_str_clear() - clear the internal fallback string variable table and the statement
 * cache. */
void mc_host_str_clear(void);

#endif /* MILK_MCALC_HOST_H */
