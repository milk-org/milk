// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    runnabletestlegacy.c
 * @brief   Legacy (macro-injected) counterpart of the runnabletest module
 *
 * Same units as plugins/test-src/runnabletest, written with the pre-MILK_RUNNABLE implementation:
 * FPS_MAIN_STANDALONE_V2, INSERT_STD_* macros, one fpsexec per file, sources compiled once for
 * the library and once per executable. Baseline for behaviour and performance comparisons.
 *
 * | unit     | CLI                   | fpsexec                                  |
 * | -------- | --------------------- | ---------------------------------------- |
 * | scale    | runtestlegacy.scale   | milk-fpsexec-runnabletest-legacy-scale   |
 * | offset   | runtestlegacy.offset  | milk-fpsexec-runnabletest-legacy-offset  |
 * | negate   | runtestlegacy.negate  | -                                        |
 * | passthru | -                     | milk-fpsexec-runnabletest-legacy-passthru|
 * | total    | runtestlegacy.total   | -                                        |
 */

#define MODULE_SHORTNAME_DEFAULT "runtestlegacy"
#define MODULE_DESCRIPTION \
    "Test module: legacy macro-based FPS V2 units (baseline for runnabletest)"

#include "CLIcore.h"
#include "rt_legacy.h"

static errno_t init_module_CLI()
{
    CLIADDCMD_runnabletestlegacy__scale();
    CLIADDCMD_runnabletestlegacy__offset();
    CLIADDCMD_runnabletestlegacy__negate();
    CLIADDCMD_runnabletestlegacy__total();

    return RETURN_SUCCESS;
}

MILK_MODULE(runnabletestlegacy, init_module_CLI, NULL);
