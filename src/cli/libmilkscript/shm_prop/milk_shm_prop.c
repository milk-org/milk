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

#include <ctype.h>
#include <dirent.h>
#include <fnmatch.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
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

static int has_glob_chars(const char *s)
{
    return (s != NULL && (strchr(s, '*') != NULL || strchr(s, '?') != NULL));
}

static int token_in_space_list(const char *list, const char *tok)
{
    size_t tlen = strlen(tok);
    for (const char *p = list; *p != '\0';)
    {
        while (*p == ' ')
        {
            p++;
        }
        const char *e = p;
        while (*e != '\0' && *e != ' ')
        {
            e++;
        }
        if ((size_t) (e - p) == tlen && strncmp(p, tok, tlen) == 0)
        {
            return 1;
        }
        p = e;
    }
    return 0;
}

static int list_entities_in_namespace(const char *nsp, const char *pattern, char *out, size_t outsz)
{
    const char *pat = (pattern == NULL || strcmp(pattern, "list") == 0) ? "*" : pattern;
    size_t      pos = 0;
    out[0]          = '\0';

    if ((strcmp(nsp, "s") == 0 || strcmp(nsp, "stream") == 0) && data.core.image != NULL)
    {
        for (long i = 0; i < data.core.NB_MAX_IMAGE; i++)
        {
            if (data.core.image[i].used == 1 && data.core.image[i].name[0] != '\0')
            {
                const char *iname = data.core.image[i].name;
                if (fnmatch(pat, iname, 0) == 0 && !token_in_space_list(out, iname))
                {
                    int n = snprintf(out + pos, outsz - pos, "%s%s", (pos > 0) ? " " : "", iname);
                    if (n > 0 && pos + (size_t) n < outsz)
                    {
                        pos += (size_t) n;
                    }
                }
            }
        }
    }

    const char *shmdir = getenv("MILK_SHM_DIR");
    if (shmdir == NULL || shmdir[0] == '\0')
    {
        shmdir = (dcshmdir[0] != '\0') ? dcshmdir : "/dev/shm";
    }
    DIR *dp = opendir(shmdir);
    if (dp == NULL)
    {
        return 0;
    }

    struct dirent *de;
    while ((de = readdir(dp)) != NULL)
    {
        size_t dlen = strlen(de->d_name);
        char   ename[256];
        ename[0] = '\0';

        if ((strcmp(nsp, "s") == 0 || strcmp(nsp, "stream") == 0) && dlen > 7 &&
            strcmp(de->d_name + dlen - 7, ".im.shm") == 0 && (dlen - 7) < sizeof(ename))
        {
            memcpy(ename, de->d_name, dlen - 7);
            ename[dlen - 7] = '\0';
        }
        else if ((strcmp(nsp, "f") == 0 || strcmp(nsp, "fps") == 0) && dlen > 8 &&
                 strcmp(de->d_name + dlen - 8, ".fps.shm") == 0 && (dlen - 8) < sizeof(ename))
        {
            memcpy(ename, de->d_name, dlen - 8);
            ename[dlen - 8] = '\0';
        }
        else if ((strcmp(nsp, "p") == 0 || strcmp(nsp, "proc") == 0 ||
                  strcmp(nsp, "procinfo") == 0) &&
                 dlen > 9 && strcmp(de->d_name + dlen - 9, ".procinfo") == 0 &&
                 (dlen - 9) < sizeof(ename))
        {
            memcpy(ename, de->d_name, dlen - 9);
            ename[dlen - 9] = '\0';
            char *dot       = strrchr(ename, '.');
            if (dot != NULL)
            {
                *dot = '\0';
            }
            if (!milk_shm_prop_exists_proc(ename, 0))
            {
                ename[0] = '\0';
            }
        }
        else if ((strcmp(nsp, "q") == 0 || strcmp(nsp, "seq") == 0 || strcmp(nsp, "fpsseq") == 0) &&
                 dlen > 12 && strncmp(de->d_name, "milkseq.", 8) == 0 &&
                 strcmp(de->d_name + dlen - 4, ".shm") == 0 && (dlen - 12) < sizeof(ename))
        {
            memcpy(ename, de->d_name + 8, dlen - 12);
            ename[dlen - 12] = '\0';
        }

        if (ename[0] != '\0' && fnmatch(pat, ename, 0) == 0 && !token_in_space_list(out, ename))
        {
            int n = snprintf(out + pos, outsz - pos, "%s%s", (pos > 0) ? " " : "", ename);
            if (n > 0 && pos + (size_t) n < outsz)
            {
                pos += (size_t) n;
            }
        }
    }
    closedir(dp);
    return 0;
}

static int is_json_number(const char *s)
{
    if (s == NULL || *s == '\0')
    {
        return 0;
    }
    const char *p = s;
    if (*p == '-')
    {
        p++;
    }
    if (!isdigit((unsigned char) *p))
    {
        return 0;
    }
    if (*p == '0' && isdigit((unsigned char) p[1]))
    {
        return 0;
    }
    while (isdigit((unsigned char) *p))
    {
        p++;
    }
    if (*p == '.')
    {
        p++;
        if (!isdigit((unsigned char) *p))
        {
            return 0;
        }
        while (isdigit((unsigned char) *p))
        {
            p++;
        }
    }
    if (*p == 'e' || *p == 'E')
    {
        p++;
        if (*p == '+' || *p == '-')
        {
            p++;
        }
        if (!isdigit((unsigned char) *p))
        {
            return 0;
        }
        while (isdigit((unsigned char) *p))
        {
            p++;
        }
    }
    return (*p == '\0');
}

static void append_json_escaped_str(char *out, size_t outsz, size_t *pos, const char *s)
{
    size_t p = *pos;
    if (p + 2 >= outsz)
    {
        return;
    }
    out[p++] = '"';
    for (const char *c = s; *c != '\0' && p + 6 < outsz; c++)
    {
        if (*c == '"' || *c == '\\')
        {
            out[p++] = '\\';
            out[p++] = *c;
        }
        else if (*c == '\n')
        {
            out[p++] = '\\';
            out[p++] = 'n';
        }
        else if (*c == '\r')
        {
            out[p++] = '\\';
            out[p++] = 'r';
        }
        else if (*c == '\t')
        {
            out[p++] = '\\';
            out[p++] = 't';
        }
        else
        {
            out[p++] = *c;
        }
    }
    if (p + 1 < outsz)
    {
        out[p++] = '"';
    }
    out[p] = '\0';
    *pos   = p;
}

static void append_kv_entries_json(const char *kv, char *out, size_t outsz, size_t *pos, int *first)
{
    char *dup = strdup(kv ? kv : "");
    if (dup == NULL)
    {
        return;
    }
    char *saveptr = NULL;
    for (char *line = strtok_r(dup, "\r\n", &saveptr); line != NULL;
         line       = strtok_r(NULL, "\r\n", &saveptr))
    {
        char *sep = strchr(line, '\t');
        if (sep == NULL)
        {
            sep = strchr(line, '=');
        }
        if (sep == NULL)
        {
            continue;
        }
        *sep            = '\0';
        const char *key = line;
        const char *val = sep + 1;
        if (key[0] == '\0')
        {
            continue;
        }
        if (!(*first) && *pos + 1 < outsz)
        {
            out[(*pos)++] = ',';
            out[*pos]     = '\0';
        }
        *first = 0;
        append_json_escaped_str(out, outsz, pos, key);
        if (*pos + 1 < outsz)
        {
            out[(*pos)++] = ':';
            out[*pos]     = '\0';
        }
        if (is_json_number(val))
        {
            int n = snprintf(out + *pos, outsz - *pos, "%s", val);
            if (n > 0 && *pos + (size_t) n < outsz)
            {
                *pos += (size_t) n;
            }
        }
        else
        {
            append_json_escaped_str(out, outsz, pos, val);
        }
    }
    free(dup);
}

static void kv_to_json_obj(const char *kv, const char *extra_kw_kv, char *out, size_t outsz)
{
    size_t pos = 0;
    int    fst = 1;
    if (outsz < 3)
    {
        return;
    }
    out[pos++] = '{';
    out[pos]   = '\0';
    append_kv_entries_json(kv, out, outsz, &pos, &fst);
    if (extra_kw_kv != NULL && extra_kw_kv[0] != '\0')
    {
        if (!fst && pos + 1 < outsz)
        {
            out[pos++] = ',';
            out[pos]   = '\0';
        }
        int n = snprintf(out + pos, outsz - pos, "\"kw\":{");
        if (n > 0 && pos + (size_t) n < outsz)
        {
            pos += (size_t) n;
        }
        int kw_fst = 1;
        append_kv_entries_json(extra_kw_kv, out, outsz, &pos, &kw_fst);
        if (pos + 1 < outsz)
        {
            out[pos++] = '}';
            out[pos]   = '\0';
        }
    }
    if (pos + 1 < outsz)
    {
        out[pos++] = '}';
        out[pos]   = '\0';
    }
}

static void list_to_json_arr(const char *list, char *out, size_t outsz)
{
    size_t pos = 0;
    if (outsz < 3)
    {
        return;
    }
    out[pos++] = '[';
    out[pos]   = '\0';
    char *dup  = strdup(list ? list : "");
    if (dup != NULL)
    {
        int   fst     = 1;
        char *saveptr = NULL;
        for (char *tok = strtok_r(dup, " \t\r\n", &saveptr); tok != NULL;
             tok       = strtok_r(NULL, " \t\r\n", &saveptr))
        {
            if (!fst && pos + 1 < outsz)
            {
                out[pos++] = ',';
                out[pos]   = '\0';
            }
            fst = 0;
            append_json_escaped_str(out, outsz, &pos, tok);
        }
        free(dup);
    }
    if (pos + 1 < outsz)
    {
        out[pos++] = ']';
        out[pos]   = '\0';
    }
}

static int dispatch_ns_get(const char *nsp,
                           const char *name,
                           const char *prop,
                           char       *out,
                           size_t      outsz)
{
    if (strcmp(prop, "json") == 0)
    {
        char kv[16384];
        kv[0] = '\0';
        if (dispatch_ns_get(nsp, name, "kv", kv, sizeof(kv)) != 0)
        {
            return -1;
        }
        char kw_kv[4096];
        kw_kv[0] = '\0';
        if (strcmp(nsp, "s") == 0 || strcmp(nsp, "stream") == 0)
        {
            dispatch_ns_get(nsp, name, "kw.kv", kw_kv, sizeof(kw_kv));
        }
        kv_to_json_obj(kv, kw_kv[0] ? kw_kv : NULL, out, outsz);
        return 0;
    }
    if (strcmp(prop, "kw.json") == 0 && (strcmp(nsp, "s") == 0 || strcmp(nsp, "stream") == 0))
    {
        char kw_kv[8192];
        kw_kv[0] = '\0';
        if (milk_shm_prop_get_stream(name, "kw.kv", kw_kv, sizeof(kw_kv)) != 0)
        {
            return -1;
        }
        kv_to_json_obj(kw_kv, NULL, out, outsz);
        return 0;
    }

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

/**
 * @brief Read a live milk entity property or FITS keyword into @out.
 *
 * @param spec   Entity path string (leading '@' or 'json:' optional)
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

    int want_json = 0;
    if (strncmp(spec, "json:", 5) == 0)
    {
        want_json = 1;
        spec += 5;
    }
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

    /* 1-dot namespace queries: s.*, f.*, p.*, q.*, s.dm*, or --json s.im1 */
    if (dot2 == NULL && is_known_namespace(buf))
    {
        if (strcmp(rest, "*") == 0 || strcmp(rest, "list") == 0 || has_glob_chars(rest))
        {
            char lbuf[8192];
            list_entities_in_namespace(buf, rest, lbuf, sizeof(lbuf));
            if (want_json)
            {
                list_to_json_arr(lbuf, out, outsz);
            }
            else
            {
                snprintf(out, outsz, "%s", lbuf);
            }
            return 0;
        }
        if (want_json)
        {
            return dispatch_ns_get(buf, rest, "json", out, outsz);
        }
    }

    if (dot2 != NULL && is_known_namespace(buf))
    {
        *dot2            = '\0';
        const char *nsp  = buf;
        const char *name = rest;
        const char *prop = dot2 + 1;

        if (has_glob_chars(name))
        {
            char lbuf[8192];
            list_entities_in_namespace(nsp, name, lbuf, sizeof(lbuf));
            size_t pos     = 0;
            int    matched = 0;
            char  *saveptr = NULL;
            for (char *ent = strtok_r(lbuf, " \t\r\n", &saveptr); ent != NULL;
                 ent       = strtok_r(NULL, " \t\r\n", &saveptr))
            {
                char vbuf[1024];
                if (dispatch_ns_get(nsp, ent, prop, vbuf, sizeof(vbuf)) == 0)
                {
                    int n = snprintf(out + pos, outsz - pos, "%s%s=%s", (pos > 0) ? "\n" : "", ent,
                                     vbuf);
                    if (n > 0 && pos + (size_t) n < outsz)
                    {
                        pos += (size_t) n;
                    }
                    matched = 1;
                }
            }
            if (!matched)
            {
                return -1;
            }
            if (want_json)
            {
                char *kv_copy = strdup(out);
                if (kv_copy != NULL)
                {
                    kv_to_json_obj(kv_copy, NULL, out, outsz);
                    free(kv_copy);
                }
            }
            return 0;
        }

        if (want_json)
        {
            if (strcmp(prop, "*") == 0 || strcmp(prop, "list") == 0)
            {
                return dispatch_ns_get(nsp, name, "json", out, outsz);
            }
            if (strcmp(prop, "kw") == 0 || strcmp(prop, "kw.*") == 0)
            {
                return dispatch_ns_get(nsp, name, "kw.json", out, outsz);
            }
        }

        int rc = dispatch_ns_get(nsp, name, prop, out, outsz);
        if (rc == 0 && want_json && strcmp(prop, "json") != 0 && strcmp(prop, "kw.json") != 0)
        {
            char *raw = strdup(out);
            if (raw != NULL)
            {
                size_t p = 0;
                out[0]   = '\0';
                if (is_json_number(raw))
                {
                    snprintf(out, outsz, "%s", raw);
                }
                else
                {
                    append_json_escaped_str(out, outsz, &p, raw);
                }
                free(raw);
            }
        }
        return rc;
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
             strcmp(last_dot + 1, "runstop") == 0 || strcmp(last_dot + 1, "reset") == 0))
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
        if (strcmp(nsp, "q") == 0 || strcmp(nsp, "seq") == 0 || strcmp(nsp, "fpsseq") == 0)
        {
            return milk_shm_prop_set_seq(name, prop, val);
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
