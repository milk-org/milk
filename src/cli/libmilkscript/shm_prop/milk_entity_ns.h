// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_entity_ns.h
 * @brief Shared entity namespace and stream-specifier normalization helpers.
 *
 * Role & Goal:
 *   Provides a single source of truth for recognizing milk entity namespaces
 *   (`s`/`stream`, `l`/`local`, `f`/`fps`, `p`/`proc`/`procinfo`,
 *   `q`/`seq`/`fpsseq`) and normalizing `@S:<name>` / `@L:<name>` modifiers
 *   into canonical `s.<name>` / `l.<name>` entity paths.
 */

#ifndef MILK_ENTITY_NS_H
#define MILK_ENTITY_NS_H

#include <ctype.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

/**
 * @brief Return 1 if @s of length @nslen is a known entity namespace prefix.
 */
static inline int milk_is_entity_ns_prefix(const char *s, size_t nslen)
{
    if (s == NULL || nslen == 0)
    {
        return 0;
    }
    return (
        (nslen == 1 && (s[0] == 's' || s[0] == 'l' || s[0] == 'f' || s[0] == 'p' || s[0] == 'q')) ||
        (nslen == 3 && (strncmp(s, "fps", 3) == 0 || strncmp(s, "seq", 3) == 0)) ||
        (nslen == 4 && strncmp(s, "proc", 4) == 0) || (nslen == 5 && strncmp(s, "local", 5) == 0) ||
        (nslen == 6 && (strncmp(s, "stream", 6) == 0 || strncmp(s, "fpsseq", 6) == 0)) ||
        (nslen == 8 && strncmp(s, "procinfo", 8) == 0));
}

/**
 * @brief Return 1 if @ns (NUL-terminated) is a known entity namespace.
 */
static inline int milk_is_known_namespace(const char *ns)
{
    return (ns != NULL) ? milk_is_entity_ns_prefix(ns, strlen(ns)) : 0;
}

/**
 * @brief Normalize an entity specifier into @out (stripping `@` and converting `S:`/`L:`).
 *
 * Examples:
 *   `@S:im1.cnt0` -> `s.im1.cnt0`
 *   `@L:loc1`     -> `l.loc1`
 *   `@s.im1.cnt0` -> `s.im1.cnt0`
 */
static inline void milk_normalize_entity_spec(const char *in, char *out, size_t outsz)
{
    if (out == NULL || outsz == 0)
    {
        return;
    }
    if (in == NULL)
    {
        out[0] = '\0';
        return;
    }
    while (*in == '@')
    {
        in++;
    }
    if (strncmp(in, "S:", 2) == 0 && in[2] != '\0')
    {
        snprintf(out, outsz, "s.%s", in + 2);
        return;
    }
    if (strncmp(in, "L:", 2) == 0 && in[2] != '\0')
    {
        snprintf(out, outsz, "l.%s", in + 2);
        return;
    }
    const char *colon = strchr(in, ':');
    const char *dot   = strchr(in, '.');
    if (colon != NULL && colon > in && colon[1] != '\0' && (dot == NULL || colon < dot))
    {
        int is_l = 0;
        int is_s = 0;
        for (const char *p = in; p < colon; p++)
        {
            if (*p == 'L' && (p == in || p[-1] == ','))
            {
                is_l = 1;
            }
            else if (*p == 'S' && (p == in || p[-1] == ','))
            {
                is_s = 1;
            }
        }
        if (is_l)
        {
            snprintf(out, outsz, "l.%s", colon + 1);
            return;
        }
        if (is_s)
        {
            snprintf(out, outsz, "s.%s", colon + 1);
            return;
        }
    }
    snprintf(out, outsz, "%s", in);
}

#endif /* MILK_ENTITY_NS_H */
