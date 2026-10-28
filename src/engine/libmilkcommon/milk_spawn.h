// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_spawn.h
 * @brief Run external tools from an argv array, without a shell
 */

#ifndef MILK_SPAWN_H
#define MILK_SPAWN_H

#include <stddef.h>
#include <sys/types.h>

pid_t milkspawn_start(const char *const argv[], int out_fd, int err_fd);
/** milkspawn_start_env - milkspawn_start() with environment @envp instead of environ. */
pid_t milkspawn_start_env(const char *const argv[], char *const envp[], int out_fd, int err_fd);
int   milkspawn_wait(pid_t pid);
int   milkspawn_run(const char *const argv[], int out_fd, int err_fd);

/** milkspawn_poll - reap @pid if it ended, without waiting. Return: 1 (*@status as milkspawn_wait()), 0 running, -1. */
int milkspawn_poll(pid_t pid, int *status);

/** milkspawn_tool_path - milk tool @tool: in $MILK_INSTALLDIR/bin, next to the running executable, else on PATH. */
void milkspawn_tool_path(const char *tool, char *buf, size_t size);

#endif
