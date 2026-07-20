// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    shmim_setowner.h
 */

imageID shmim_setowner_creator(const char *name);

imageID shmim_setowner_current(const char *name);

imageID shmim_setowner_init(const char *name);

errno_t shmim_setowner_addCLIcmd();
