// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_cli_function_registry.c
 * @brief   Global function pointer definitions
 *
 * The pointer is NULL by default and is set by an explicit
 * call to fps_cli_register() from the CLI startup code.
 *
 * Built as part of milkfps (no CLI dependency).
 */

#include <stddef.h>


void *fps_generic_CLIfunction_ptr = NULL;

char fps_last_used_name[200]   = "";
char fps_last_used_cmdkey[200] = "";
