// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef MILK_PATH_H
#define MILK_PATH_H

#include <stddef.h>

void milkpath_resolve_realpath(char *dest, size_t destsize, const char *path);

#endif
