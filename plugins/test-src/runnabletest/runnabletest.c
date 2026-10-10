// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    runnabletest.c
 * @brief   Test module for MILK_RUNNABLE descriptors
 *
 * Commands (CLI: runtest.<cmd>) and executables (milk-fpsexec-<id>):
 *
 * | unit     | CLI            | fpsexec                      | file       |
 * | -------- | -------------- | ---------------------------- | ---------- |
 * | scale    | runtest.scale  | milk-fpsexec-runnabletest-scale  | rt_affine.c |
 * | offset   | runtest.offset | milk-fpsexec-runnabletest-offset | rt_affine.c |
 * | negate   | runtest.negate | -                            | rt_affine.c |
 * | passthru | -              | milk-fpsexec-runnabletest-passthru | rt_affine.c |
 * | total    | runtest.total  | - (CLI-only command)         | rt_total.c |
 *
 * Counterpart on the legacy macro-based implementation: plugins/test-src/runnabletest_legacy.
 */

#define MODULE_SHORTNAME_DEFAULT "runtest"
#define MODULE_DESCRIPTION "Test module: MILK_RUNNABLE descriptors, several units per file"

#include "milk_runnable.h"
#include "runnabletest.h"

/* CLI exposure = membership here. passthru is deliberately absent (fpsexec only). */
static const MILK_RUNNABLE *const runnables[] = {
    &runnable_rt_scale,
    &runnable_rt_offset,
    &runnable_rt_negate,
    &runnable_rt_total,
};

MILK_MODULE_RUNNABLES(runnabletest, NULL, runnables);
