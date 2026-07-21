// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file shmim_setowner.h
 * @brief Set stream owner PID
 */

errno_t CLIADDCMD_COREMOD_memory__shmim_setowner();

imageID shmim_setowner_creator(const char *name);

imageID shmim_setowner_current(const char *name);

imageID shmim_setowner_init(const char *name);
