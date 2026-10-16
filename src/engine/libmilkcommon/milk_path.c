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

/**
 * milkpath_shell_quote - write @src into @dest as one single-quoted sh word
 * (' becomes '\''). Returns 0, or -1 (and dest = "''") if @destsize is too small.
 */
int milkpath_shell_quote(char *dest, size_t destsize, const char *src)
{
    size_t n = 0;
    if (destsize < 3)
    {
        return -1;
    }
    dest[n++] = '\'';
    for (const char *c = src; *c; c++)
    {
        size_t need = (*c == '\'') ? 4 : 1;
        if (n + need + 2 > destsize)
        {
            strcpy(dest, "''");
            return -1;
        }
        if (*c == '\'')
        {
            memcpy(dest + n, "'\\''", 4);
        }
        else
        {
            dest[n] = *c;
        }
        n += need;
    }
    dest[n++] = '\'';
    dest[n]   = '\0';
    return 0;
}
