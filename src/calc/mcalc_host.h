// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mcalc_host.h
 * @brief Host-environment bridge for string variables in `libmilkcalc`.
 *
 * Allows the upper CLI/scripting tier (`libmilkscript`) to register its variable
 * table callbacks (`cli_var_get` / `cli_var_set`) without introducing an upward link
 * dependency from `libmilkcalc` (Calculator tier) to `libmilkscript` (CLI tier). When no
 * host is registered (e.g. in standalone `milk-fpsexec-milk-calc`), `libmilkcalc`
 * falls back to its own process-local string variable table.
 */

#ifndef MILK_MCALC_HOST_H
#define MILK_MCALC_HOST_H

/* String variables and other host-owned names; the script layer registers its implementation. */
typedef struct
{
    const char *(*str_get)(const char *name);
    void (*str_set)(const char *name, const char *val);
} mcalc_host_ops;

/** mcalc_set_host() - register host callbacks for string variables and host-owned names. */
void mcalc_set_host(const mcalc_host_ops *ops);

/** mc_host_str_get() - look up string variable @name via the host or internal fallback table. */
const char *mc_host_str_get(const char *name);

/** mc_host_str_set() - store string variable @name via the host or internal fallback table;
 * a NULL @val removes it. */
void mc_host_str_set(const char *name, const char *val);

/** mc_host_str_clear() - clear the internal fallback string variable table. */
void mc_host_str_clear(void);

#endif /* MILK_MCALC_HOST_H */
