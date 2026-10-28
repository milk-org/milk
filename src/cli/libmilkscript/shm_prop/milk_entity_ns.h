// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_entity_ns.h
 * @brief Shared entity namespace and stream-specifier normalization helpers.
 *
 * Role & Goal:
 *   Provides a single source of truth for recognizing milk entity namespaces
 *   (`s`/`stream`, `l`/`local`, `f`/`fps`, `p`/`proc`) and normalizing `@S:<name>` / `@L:<name>` modifiers
 *   into canonical `s.<name>` / `l.<name>` entity paths.
 */

#ifndef MILK_ENTITY_NS_H
#define MILK_ENTITY_NS_H

#include <ctype.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

/** Entity namespaces: what the prefix before the first '.' of an entity path refers to. */
typedef enum
{
    MILK_NS_NONE = 0,
    MILK_NS_STREAM, /* s.  stream.            shared-memory stream */
    MILK_NS_LOCAL,  /* l.  local.             image local to the milk-cli process */
    MILK_NS_FPS,    /* f.  fps.               function parameter structure */
    MILK_NS_PROC    /* p.  proc.              processinfo loop */
} milk_ns_kind_t;

/** The single list of namespace names (milk.bash gets it from `milkls --namespaces`). */
static const struct
{
    const char    *name;
    milk_ns_kind_t kind;
} milk_entity_ns_table[] = { { "s", MILK_NS_STREAM }, { "stream", MILK_NS_STREAM },
                             { "l", MILK_NS_LOCAL },  { "local", MILK_NS_LOCAL },
                             { "f", MILK_NS_FPS },    { "fps", MILK_NS_FPS },
                             { "p", MILK_NS_PROC },   { "proc", MILK_NS_PROC },
                             { NULL, MILK_NS_NONE } };

/** Kind names, indexed by milk_ns_kind_t (as printed by `milkls --namespaces`). */
static const char *const milk_ns_kind_names[] = { "none", "stream", "local", "fps", "proc" };

/** milk_entity_ns_removed() - hint when entity path @spec starts with a removed namespace, else NULL. */
static inline const char *milk_entity_ns_removed(const char *spec)
{
    while (spec != NULL && *spec == '@')
    {
        spec++;
    }
    if (spec != NULL && (strncmp(spec, "q.", 2) == 0 || strncmp(spec, "fpsseq.", 7) == 0 ||
                         strncmp(spec, "seq.", 4) == 0))
    {
        return "the sequencer (milk-seq) was removed: write the workflow as a milk script "
               "(@help scripting; docs/guide/scripting.md, Migrating Old Scripts)";
    }
    return NULL;
}

/** milk_entity_ns_lookup() - namespace kind of the @nslen-byte name @s (MILK_NS_NONE if unknown). */
static inline milk_ns_kind_t milk_entity_ns_lookup(const char *s, size_t nslen)
{
    if (s == NULL || nslen == 0)
    {
        return MILK_NS_NONE;
    }
    for (int i = 0; milk_entity_ns_table[i].name != NULL; i++)
    {
        if (strlen(milk_entity_ns_table[i].name) == nslen &&
            strncmp(s, milk_entity_ns_table[i].name, nslen) == 0)
        {
            return milk_entity_ns_table[i].kind;
        }
    }
    return MILK_NS_NONE;
}

/**
 * milk_entity_ns_kind() - namespace of entity path @spec ("s.cam.xsize"), skipping the namespace
 * and its '.' in *@rest if given; MILK_NS_NONE (and *@rest = @spec) when @spec has no known prefix.
 */
static inline milk_ns_kind_t milk_entity_ns_kind(const char *spec, const char **rest)
{
    const char    *dot = (spec != NULL) ? strchr(spec, '.') : NULL;
    milk_ns_kind_t kind =
        (dot != NULL) ? milk_entity_ns_lookup(spec, (size_t) (dot - spec)) : MILK_NS_NONE;
    if (rest != NULL)
    {
        *rest = (kind != MILK_NS_NONE) ? dot + 1 : spec;
    }
    return kind;
}

/**
 * @brief Return 1 if @s of length @nslen is a known entity namespace prefix.
 */
static inline int milk_is_entity_ns_prefix(const char *s, size_t nslen)
{
    return milk_entity_ns_lookup(s, nslen) != MILK_NS_NONE;
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
