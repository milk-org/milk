// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_cli_function_registry.c
 * @brief   Global function pointer definitions
 *
 * These pointers are NULL by default and get set by
 * milkfpsCLI at load time via a constructor function.
 *
 * Built as part of milkfps (no CLI dependency).
 */

#include <stddef.h>


void *fps_generic_CLIfunction_ptr = NULL;
void *fps_fill_farg_examples_ptr  = NULL;

char fps_last_used_name[200]   = "";
char fps_last_used_cmdkey[200] = "";
