// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file example03fps_module.c
 * @brief Combined Milk CLI module for writer03 and processor03.
 *
 * Main entry point of the shared object 'example03fps.so': lists the runnables it exposes.
 */

/** @brief Default shortname for the module (used as command prefix in milk CLI). */
#define MODULE_SHORTNAME_DEFAULT "ex03"

/** @brief Human-readable description of the module. */
#define MODULE_DESCRIPTION "Example 03 FPS Writer and Processor"

#include "milk_runnable.h"

extern const MILK_RUNNABLE runnable_example03_writer03;
extern const MILK_RUNNABLE runnable_example03_processor03;

static const MILK_RUNNABLE *const runnables[] = {
    &runnable_example03_writer03,
    &runnable_example03_processor03,
};

MILK_MODULE_RUNNABLES(example03fps, NULL, runnables);
