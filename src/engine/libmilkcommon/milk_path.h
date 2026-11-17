// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef MILK_PATH_H
#define MILK_PATH_H

#include <stddef.h>
#include <sys/types.h>

void milkpath_resolve_realpath(char *dest, size_t destsize, const char *path);

int milkpath_shell_quote(char *dest, size_t destsize, const char *src);

/** milkpath_mkdir_p - create directory @path and its missing parents with @mode. Return: 0, or -1 with errno. */
int milkpath_mkdir_p(const char *path, mode_t mode);

#endif
