// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "milk_path.h"

#include <stdlib.h>
#include <string.h>

/**
 * milkpath_resolve_realpath - copy the canonical (symlink-resolved) form
 * of @path into @dest, falling back to @path verbatim if realpath()
 * fails (e.g. path does not exist yet). Works for files, directories,
 * or any other filesystem entry type realpath() accepts.
 */
void milkpath_resolve_realpath(char *dest, size_t destsize, const char *path)
{
    char *resolved = realpath(path, NULL);
    if (resolved != NULL)
    {
        strncpy(dest, resolved, destsize - 1);
        dest[destsize - 1] = '\0';
        free(resolved);
    }
    else
    {
        strncpy(dest, path, destsize - 1);
        dest[destsize - 1] = '\0';
    }
}
