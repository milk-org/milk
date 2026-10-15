// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file shmim_setowner.h
 * @brief Set stream owner PID
 */

#include "milk_runnable.h"

extern const MILK_RUNNABLE runnable_COREMOD_memory_shmimsetowncreator;
extern const MILK_RUNNABLE runnable_COREMOD_memory_shmimsetowncurrent;
extern const MILK_RUNNABLE runnable_COREMOD_memory_shmimsetowninit;

imageID shmim_setowner_creator(const char *name);

imageID shmim_setowner_current(const char *name);

imageID shmim_setowner_init(const char *name);
