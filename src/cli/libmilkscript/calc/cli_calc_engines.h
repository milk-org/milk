// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_engines.h
 * @brief Catalog and listing of pre-compiled calculator bytecode engines, tokens, and C extensions.
 */

#ifndef CLI_CALC_ENGINES_H
#define CLI_CALC_ENGINES_H

#include "milk_calc_ext.h"

/**
 * cli_calc_ext_getter_fn - Callback to retrieve a loaded user C extension by index
 */
typedef const milk_calc_ext_entry_t *(*cli_calc_ext_getter_fn)(int idx);

/**
 * cli_calc_print_engines - Print the catalog of pre-compiled calculator bytecode engines and tokens
 * @pattern:   Optional case-insensitive substring filter (NULL or "" for all)
 * @json_mode: Non-zero to emit JSON output instead of a formatted table
 * @ext_get:   Optional callback returning loaded user C extension entry `i`
 * @ext_count: Number of currently loaded user C extensions (`0` if none)
 *
 * Return: 0 on success, 1 if @pattern matched no entries.
 */
int cli_calc_print_engines(const char            *pattern,
                           int                    json_mode,
                           cli_calc_ext_getter_fn ext_get,
                           int                    ext_count);

#endif /* CLI_CALC_ENGINES_H */
