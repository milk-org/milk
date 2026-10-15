// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_shm_prop.c
 * @brief Top-level dispatcher for milk shared-memory entity operations.
 *
 * Role & Goal:
 *   Parses entity path strings (`[@]<namespace>.<name>.<prop>` or legacy
 *   `[@]<name>.<prop>`) and routes read (`milk_shm_prop_get`), write
 *   (`milk_shm_prop_set`), and existence (`milk_shm_prop_exists`) queries to the
 *   appropriate namespace handler (`stream`, `fps`, `proc`, or `seq`).
 */

#include <stdio.h>
#include <string.h>

#include "milk_shm_prop.h"
#include "milk_shm_prop_internal.h"

/**
 * @brief Check whether a prefix token is a known entity namespace.
 *
 * @param prefix Token before the first dot
 * @return 1 if @prefix is a recognized namespace, 0 otherwise
 */
static int is_known_namespace(const char *prefix)
{
    return (strcmp(prefix, "s") == 0 || strcmp(prefix, "stream") == 0 || strcmp(prefix, "f") == 0 ||
            strcmp(prefix, "fps") == 0 || strcmp(prefix, "p") == 0 || strcmp(prefix, "proc") == 0 ||
            strcmp(prefix, "procinfo") == 0 || strcmp(prefix, "q") == 0 ||
            strcmp(prefix, "seq") == 0 || strcmp(prefix, "fpsseq") == 0);
}

/**
 * @brief Read a live milk entity property or FITS keyword into @out.
 *
 * @param spec   Entity path string (leading '@' optional)
 * @param out    Output buffer
 * @param outsz  Size of @out in bytes
 * @return 0 on success, -1 if not found or invalid
 */
int milk_shm_prop_get(const char *spec, char *out, size_t outsz)
{
    if (spec == NULL || out == NULL || outsz == 0)
    {
        return -1;
    }
    out[0] = '\0';

    while (*spec == '@')
    {
        spec++;
    }

    char buf[512];
    snprintf(buf, sizeof(buf), "%s", spec);

    char *dot1 = strchr(buf, '.');
    if (dot1 == NULL)
    {
        return -1;
    }
    *dot1      = '\0';
    char *rest = dot1 + 1;
    char *dot2 = strchr(rest, '.');

    if (dot2 != NULL && is_known_namespace(buf))
    {
        *dot2            = '\0';
        const char *nsp  = buf;
        const char *name = rest;
        const char *prop = dot2 + 1;

        if (strcmp(nsp, "s") == 0 || strcmp(nsp, "stream") == 0)
        {
            if (milk_shm_prop_get_stream(name, prop, out, outsz) == 0)
            {
                return 0;
            }
            if (strcmp(nsp, "s") == 0)
            {
                return milk_shm_prop_get_seq(name, prop, out, outsz);
            }
            return -1;
        }
        if (strcmp(nsp, "f") == 0 || strcmp(nsp, "fps") == 0)
        {
            return milk_shm_prop_get_fps(name, prop, out, outsz);
        }
        if (strcmp(nsp, "p") == 0 || strcmp(nsp, "proc") == 0 || strcmp(nsp, "procinfo") == 0)
        {
            return milk_shm_prop_get_proc(name, prop, out, outsz);
        }
        if (strcmp(nsp, "q") == 0 || strcmp(nsp, "seq") == 0 || strcmp(nsp, "fpsseq") == 0)
        {
            return milk_shm_prop_get_seq(name, prop, out, outsz);
        }
        return -1;
    }

    /* Legacy 1-dot fallback (@<name>.<prop>): try FPS first, then procinfo */
    if (milk_shm_prop_get_fps(buf, rest, out, outsz) == 0)
    {
        return 0;
    }
    if (milk_shm_prop_exists_fps(buf, 0))
    {
        return -1;
    }
    return milk_shm_prop_get_proc(buf, rest, out, outsz);
}

/**
 * @brief Write a live milk entity property or FITS keyword.
 *
 * @param spec     Entity path string (may also contain "=val")
 * @param val      Value string (or NULL if embedded in @spec, or "-" for stdin)
 * @param comment  Optional FITS keyword comment (may be NULL)
 * @return 0 on success, -1 on error
 */
int milk_shm_prop_set(const char *spec, const char *val, const char *comment)
{
    if (spec == NULL)
    {
        return -1;
    }
    while (*spec == '@')
    {
        spec++;
    }

    char specbuf[512];
    snprintf(specbuf, sizeof(specbuf), "%s", spec);

    char stdin_buf[512];
    if (val == NULL)
    {
        char *eq = strchr(specbuf, '=');
        if (eq != NULL)
        {
            *eq = '\0';
            val = eq + 1;
        }
    }
    if (val == NULL)
    {
        const char *last_dot = strrchr(specbuf, '.');
        if (last_dot != NULL &&
            (strcmp(last_dot + 1, "confstart") == 0 || strcmp(last_dot + 1, "confstop") == 0 ||
             strcmp(last_dot + 1, "confstep") == 0 || strcmp(last_dot + 1, "runstart") == 0 ||
             strcmp(last_dot + 1, "runstop") == 0))
        {
            val = "1";
        }
        else
        {
            return -1;
        }
    }
    if (strcmp(val, "-") == 0)
    {
        if (fgets(stdin_buf, sizeof(stdin_buf), stdin) == NULL)
        {
            return -1;
        }
        stdin_buf[strcspn(stdin_buf, "\r\n")] = '\0';
        val                                   = stdin_buf;
    }

    char *dot1 = strchr(specbuf, '.');
    if (dot1 == NULL)
    {
        return -1;
    }
    *dot1      = '\0';
    char *rest = dot1 + 1;
    char *dot2 = strchr(rest, '.');

    if (dot2 != NULL && is_known_namespace(specbuf))
    {
        *dot2            = '\0';
        const char *nsp  = specbuf;
        const char *name = rest;
        const char *prop = dot2 + 1;

        if (strcmp(nsp, "f") == 0 || strcmp(nsp, "fps") == 0)
        {
            return milk_shm_prop_set_fps(name, prop, val);
        }
        if (strcmp(nsp, "s") == 0 || strcmp(nsp, "stream") == 0)
        {
            return milk_shm_prop_set_stream(name, prop, val, comment);
        }
        if (strcmp(nsp, "p") == 0 || strcmp(nsp, "proc") == 0 || strcmp(nsp, "procinfo") == 0)
        {
            return milk_shm_prop_set_proc(name, prop, val);
        }
        return -1;
    }

    /* Legacy fallback: @<fpsname>.<param>=<val> */
    return milk_shm_prop_set_fps(specbuf, rest, val);
}

/**
 * @brief Check whether a milk entity exists (or is actively running).
 *
 * @param spec          Entity specification (e.g. "s.cam", "f.dm", "p.loop")
 * @param check_running If non-zero, verify that the entity is actively running
 * @return 1 if exists/running, 0 otherwise
 */
int milk_shm_prop_exists(const char *spec, int check_running)
{
    if (spec == NULL)
    {
        return 0;
    }
    while (*spec == '@')
    {
        spec++;
    }

    char buf[256];
    snprintf(buf, sizeof(buf), "%s", spec);
    char *dot = strchr(buf, '.');
    if (dot == NULL)
    {
        return 0;
    }
    *dot             = '\0';
    const char *nsp  = buf;
    const char *name = dot + 1;

    if (strcmp(nsp, "s") == 0 || strcmp(nsp, "stream") == 0)
    {
        return milk_shm_prop_exists_stream(name);
    }
    if (strcmp(nsp, "f") == 0 || strcmp(nsp, "fps") == 0)
    {
        return milk_shm_prop_exists_fps(name, check_running);
    }
    if (strcmp(nsp, "p") == 0 || strcmp(nsp, "proc") == 0 || strcmp(nsp, "procinfo") == 0)
    {
        return milk_shm_prop_exists_proc(name, check_running);
    }
    if (strcmp(nsp, "q") == 0 || strcmp(nsp, "seq") == 0 || strcmp(nsp, "fpsseq") == 0)
    {
        return milk_shm_prop_exists_seq(name, check_running);
    }
    return 0;
}
