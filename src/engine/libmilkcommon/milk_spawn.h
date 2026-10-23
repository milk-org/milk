// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_spawn.h
 * @brief Run external tools from an argv array, without a shell
 */

#ifndef MILK_SPAWN_H
#define MILK_SPAWN_H

#include <sys/types.h>

pid_t milkspawn_start(const char *const argv[], int out_fd, int err_fd);
int   milkspawn_wait(pid_t pid);
int   milkspawn_run(const char *const argv[], int out_fd, int err_fd);

#endif
