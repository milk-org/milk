// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "milk_path.h"

#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

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

int milkpath_mkdir_p(const char *path, mode_t mode)
{
    char        buf[PATH_MAX];
    struct stat st;
    size_t      len = strlen(path);

    if (len == 0 || len >= sizeof(buf))
    {
        errno = (len == 0) ? ENOENT : ENAMETOOLONG;
        return -1;
    }
    memcpy(buf, path, len + 1);
    for (char *p = buf + 1;; p++)
    {
        if (*p != '/' && *p != '\0')
        {
            continue;
        }
        char c = *p;
        *p     = '\0';
        if (mkdir(buf, mode) != 0 &&
            (errno != EEXIST || stat(buf, &st) != 0 || !S_ISDIR(st.st_mode)))
        {
            if (errno == EEXIST)
            {
                errno = ENOTDIR;
            }
            return -1;
        }
        *p = c;
        if (c == '\0')
        {
            return 0;
        }
    }
}
