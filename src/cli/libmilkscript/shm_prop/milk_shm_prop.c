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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "milk_entity_ns.h"
#include "milk_shm_prop.h"
#include "milk_shm_prop_internal.h"

/**
 * @brief Check whether a prefix token is a known entity namespace.
 *
 * @param prefix Token before the first dot
 * @return 1 if @prefix is a recognized namespace, 0 otherwise
 */
/** fmt_double_roundtrip() - shortest %g form of @v that strtod() maps back to @v. */
void milk_shm_prop_fmt_double(char *out, size_t outsz, double v)
{
    for (int prec = 15; prec <= 17; prec++)
    {
        snprintf(out, outsz, "%.*g", prec, v);
        if (strtod(out, NULL) == v)
        {
            return;
        }
    }
}

/** milk_shm_prop_shmdir() - stream/FPS directory, resolved like ImageStreamIO (MILK_SHM_DIR first). */
const char *milk_shm_prop_shmdir(void)
{
    const char *d = getenv("MILK_SHM_DIR");
    if (d != NULL && d[0] != '\0')
    {
        return d;
    }
    return (dcshmdir[0] != '\0') ? dcshmdir : "/dev/shm";
}

static int is_known_namespace(const char *prefix)
{
    return milk_is_known_namespace(prefix);
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

/** append_word() - append " @w" (no leading space at pos 0); returns 1, leaving @out intact, if it does not fit. */
static int append_word(char *out, size_t outsz, size_t *pos, const char *w)
{
    int n = snprintf(out + *pos, outsz - *pos, "%s%s", (*pos > 0) ? " " : "", w);
    if (n < 0 || *pos + (size_t) n >= outsz)
    {
        out[*pos] = '\0';
        return 1;
    }
    *pos += (size_t) n;
    return 0;
}

static int list_entities_in_namespace(const char *nsp, const char *pattern, char *out, size_t outsz)
{
    const char *pat  = (pattern == NULL || strcmp(pattern, "list") == 0) ? "*" : pattern;
    size_t      pos  = 0;
    int         full = 0;
    out[0]           = '\0';

    if ((strcmp(nsp, "l") == 0 || strcmp(nsp, "local") == 0))
    {
        if (data.core.image == NULL)
        {
            return -1;
        }
        for (long i = 0; i < data.core.NB_MAX_IMAGE; i++)
        {
            if (data.core.image[i].used == 1 && data.core.image[i].name[0] != '\0' &&
                strncmp(data.core.image[i].name, "_tmpcalc", 8) != 0 &&
                (data.core.image[i].md == NULL || !data.core.image[i].md->shared))
            {
                const char *iname = data.core.image[i].name;
                if (fnmatch(pat, iname, 0) == 0 && !token_in_space_list(out, iname))
                {
                    full |= append_word(out, outsz, &pos, iname);
                }
            }
        }
        return full ? -2 : 0;
    }

    if ((strcmp(nsp, "s") == 0 || strcmp(nsp, "stream") == 0) && data.core.image != NULL)
    {
        for (long i = 0; i < data.core.NB_MAX_IMAGE; i++)
        {
            if (data.core.image[i].used == 1 && data.core.image[i].name[0] != '\0' &&
                strncmp(data.core.image[i].name, "_tmpcalc", 8) != 0)
            {
                const char *iname = data.core.image[i].name;
                if (fnmatch(pat, iname, 0) == 0 && !token_in_space_list(out, iname))
                {
                    full |= append_word(out, outsz, &pos, iname);
                }
            }
        }
    }

    DIR *dp = opendir(milk_shm_prop_shmdir());
    if (dp == NULL)
    {
        return full ? -2 : 0;
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
                  strcmp(nsp, "procinfo") == 0))
        {
            if (dlen > 9 && strcmp(de->d_name + dlen - 9, ".procinfo") == 0 &&
                (dlen - 9) < sizeof(ename))
            {
                memcpy(ename, de->d_name, dlen - 9);
                ename[dlen - 9] = '\0';
                char *dot       = strrchr(ename, '.');
                if (dot != NULL)
                {
                    *dot = '\0';
                }
            }
            else if (dlen > 9 && strncmp(de->d_name, "proc.", 5) == 0 &&
                     strcmp(de->d_name + dlen - 4, ".shm") == 0 && (dlen - 9) < sizeof(ename))
            {
                memcpy(ename, de->d_name + 5, dlen - 9);
                ename[dlen - 9] = '\0';
                char *dot       = strrchr(ename, '.');
                if (dot != NULL)
                {
                    *dot = '\0';
                }
                else
                {
                    ename[0] = '\0';
                }
            }
            if (ename[0] != '\0' && !milk_shm_prop_exists_proc(ename, 0))
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
            full |= append_word(out, outsz, &pos, ename);
        }
    }
    closedir(dp);

    if (strcmp(nsp, "p") == 0 || strcmp(nsp, "proc") == 0 || strcmp(nsp, "procinfo") == 0)
    {
        (void) milk_shm_prop_exists_proc("", 0);
        if (pinfolist != NULL && pinfolist != (void *) -1)
        {
            for (int pi = 0; pi < PROCESSINFOLISTSIZE; pi++)
            {
                if (pinfolist->active[pi] && pinfolist->pnamearray[pi][0] != '\0')
                {
                    const char *pname = pinfolist->pnamearray[pi];
                    if (fnmatch(pat, pname, 0) == 0 && milk_shm_prop_exists_proc(pname, 0) &&
                        !token_in_space_list(out, pname))
                    {
                        full |= append_word(out, outsz, &pos, pname);
                    }
                }
            }
        }
    }
    return full ? -2 : 0; // -2: some entries did not fit
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
        if (strcmp(nsp, "s") == 0 || strcmp(nsp, "stream") == 0 || strcmp(nsp, "l") == 0 ||
            strcmp(nsp, "local") == 0)
        {
            dispatch_ns_get(nsp, name, "kw.kv", kw_kv, sizeof(kw_kv));
        }
        kv_to_json_obj(kv, kw_kv[0] ? kw_kv : NULL, out, outsz);
        return 0;
    }
    if (strcmp(prop, "kw.json") == 0 && (strcmp(nsp, "s") == 0 || strcmp(nsp, "stream") == 0 ||
                                         strcmp(nsp, "l") == 0 || strcmp(nsp, "local") == 0))
    {
        char kw_kv[8192];
        kw_kv[0] = '\0';
        if (dispatch_ns_get(nsp, name, "kw.kv", kw_kv, sizeof(kw_kv)) != 0)
        {
            return -1;
        }
        kv_to_json_obj(kw_kv, NULL, out, outsz);
        return 0;
    }

    if (strcmp(nsp, "s") == 0 || strcmp(nsp, "stream") == 0)
    {
        return milk_shm_prop_get_stream(name, prop, out, outsz);
    }
    if (strcmp(nsp, "l") == 0 || strcmp(nsp, "local") == 0)
    {
        return milk_shm_prop_get_local_stream(name, prop, out, outsz);
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

    char buf[512];
    milk_normalize_entity_spec(spec, buf, sizeof(buf));

    char *dot1 = strchr(buf, '.');
    if (dot1 == NULL)
    {
        return -1;
    }
    *dot1      = '\0';
    char *rest = dot1 + 1;
    char *dot2 = strchr(rest, '.');

    /* 1-dot namespace queries: s.*, l.*, f.*, p.*, q.*, s.dm*, or --json s.im1 */
    if (dot2 == NULL && is_known_namespace(buf))
    {
        if (strcmp(rest, "*") == 0 || strcmp(rest, "list") == 0 || has_glob_chars(rest))
        {
            char lbuf[8192];
            if (list_entities_in_namespace(buf, rest, lbuf, sizeof(lbuf)) != 0)
            {
                return -1;
            }
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
            if (list_entities_in_namespace(nsp, name, lbuf, sizeof(lbuf)) != 0)
            {
                return -1;
            }
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

static int dispatch_ns_set(const char *nsp,
                           const char *name,
                           const char *prop,
                           const char *val,
                           const char *comment)
{
    if (strcmp(nsp, "f") == 0 || strcmp(nsp, "fps") == 0)
    {
        return milk_shm_prop_set_fps(name, prop, val);
    }
    if (strcmp(nsp, "s") == 0 || strcmp(nsp, "stream") == 0)
    {
        return milk_shm_prop_set_stream(name, prop, val, comment);
    }
    if (strcmp(nsp, "l") == 0 || strcmp(nsp, "local") == 0)
    {
        return milk_shm_prop_set_local_stream(name, prop, val, comment);
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

/**
 * @brief Write a live milk entity property or FITS keyword.
 *
 * @param spec     Entity path string (may also contain "=val")
 * @param val      Value string (or NULL if embedded in @spec, or "-" for stdin)
 * @param comment  Optional FITS keyword comment (may be NULL)
 * @return 0 on success, -1 on error
 */
static int shm_prop_set_impl(const char *spec,
                             const char *val,
                             const char *comment,
                             int         allow_stdin)
{
    if (spec == NULL)
    {
        return -1;
    }

    char specbuf[512];
    milk_normalize_entity_spec(spec, specbuf, sizeof(specbuf));

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
        int is_proc = (strncmp(specbuf, "p.", 2) == 0 || strncmp(specbuf, "proc.", 5) == 0 ||
                       strncmp(specbuf, "procinfo.", 9) == 0);
        if (last_dot != NULL &&
            (strcmp(last_dot + 1, "confstart") == 0 || strcmp(last_dot + 1, "confstop") == 0 ||
             strcmp(last_dot + 1, "confstep") == 0 || strcmp(last_dot + 1, "runstart") == 0 ||
             strcmp(last_dot + 1, "runstop") == 0 || strcmp(last_dot + 1, "reset") == 0 ||
             strcmp(last_dot + 1, "pause") == 0 || strcmp(last_dot + 1, "resume") == 0 ||
             strcmp(last_dot + 1, "step") == 0 || strcmp(last_dot + 1, "incr") == 0 ||
             strcmp(last_dot + 1, "stop") == 0 || strcmp(last_dot + 1, "exit") == 0 ||
             strcmp(last_dot + 1, "kill") == 0 || (is_proc && strcmp(last_dot + 1, "run") == 0)))
        {
            val = "1";
        }
        else
        {
            return -1;
        }
    }
    if (allow_stdin && strcmp(val, "-") == 0)
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

        if (has_glob_chars(name))
        {
            char lbuf[8192];
            if (list_entities_in_namespace(nsp, name, lbuf, sizeof(lbuf)) != 0)
            {
                return -1;
            }
            int   matched = 0;
            char *saveptr = NULL;
            for (char *ent = strtok_r(lbuf, " \t\r\n", &saveptr); ent != NULL;
                 ent       = strtok_r(NULL, " \t\r\n", &saveptr))
            {
                if (dispatch_ns_set(nsp, ent, prop, val, comment) == 0)
                {
                    matched = 1;
                }
            }
            return matched ? 0 : -1;
        }

        return dispatch_ns_set(nsp, name, prop, val, comment);
    }

    /* Legacy fallback: @<fpsname>.<param>=<val> */
    return milk_shm_prop_set_fps(specbuf, rest, val);
}

int milk_shm_prop_set(const char *spec, const char *val, const char *comment)
{
    return shm_prop_set_impl(spec, val, comment, 1);
}

/**
 * @brief Check whether a milk entity exists (or is actively running).
 *
 * @param spec          Entity specification (e.g. "s.cam", "l.loc", "f.dm", "p.loop")
 * @param check_running If non-zero, verify that the entity is actively running
 * @return 1 if exists/running, 0 otherwise
 */
int milk_shm_prop_exists(const char *spec, int check_running)
{
    if (spec == NULL)
    {
        return 0;
    }

    char buf[256];
    milk_normalize_entity_spec(spec, buf, sizeof(buf));
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
    if (strcmp(nsp, "l") == 0 || strcmp(nsp, "local") == 0)
    {
        return milk_shm_prop_exists_local_stream(name);
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

static int is_fps_readonly_meta_key(const char *k)
{
    static const char *const ro[] = { "status",     "conf",        "run",     "confrunning",
                                      "runrunning", "pid",         "confpid", "runpid",
                                      "nbparam",    "description", NULL };
    for (int i = 0; ro[i] != NULL; i++)
    {
        if (strcmp(k, ro[i]) == 0)
        {
            return 1;
        }
    }
    return 0;
}

static int is_snapshot_filter_arg(const char *arg)
{
    if (arg == NULL || arg[0] == '\0' || strchr(arg, '/') != NULL)
    {
        return 0;
    }
    size_t alen = strlen(arg);
    if (alen > 5 && strcmp(arg + alen - 5, ".json") == 0)
    {
        return 0;
    }
    char norm[256];
    milk_normalize_entity_spec(arg, norm, sizeof(norm));
    return (strncmp(norm, "s.", 2) == 0 || strncmp(norm, "stream.", 7) == 0 ||
            strncmp(norm, "l.", 2) == 0 || strncmp(norm, "local.", 6) == 0 ||
            strncmp(norm, "f.", 2) == 0 || strncmp(norm, "fps.", 4) == 0 || strcmp(norm, "*") == 0);
}

static int snapshot_key_matches(const char *key, int nfilters, char **filters)
{
    if (nfilters <= 0)
    {
        return 1;
    }
    for (int i = 0; i < nfilters; i++)
    {
        char pat[256];
        milk_normalize_entity_spec(filters[i], pat, sizeof(pat));
        char norm[256];
        if (strncmp(pat, "stream.", 7) == 0)
        {
            snprintf(norm, sizeof(norm), "s.%s", pat + 7);
        }
        else if (strncmp(pat, "local.", 6) == 0)
        {
            snprintf(norm, sizeof(norm), "s.%s", pat + 6);
        }
        else if (strncmp(pat, "l.", 2) == 0)
        {
            snprintf(norm, sizeof(norm), "s.%s", pat + 2);
        }
        else if (strncmp(pat, "fps.", 4) == 0)
        {
            snprintf(norm, sizeof(norm), "f.%s", pat + 4);
        }
        else
        {
            snprintf(norm, sizeof(norm), "%s", pat);
        }

        if (fnmatch(norm, key, 0) == 0)
        {
            return 1;
        }
        char sub[280];
        snprintf(sub, sizeof(sub), "%s.*", norm);
        if (fnmatch(sub, key, 0) == 0)
        {
            return 1;
        }
    }
    return 0;
}

static void emit_snapshot_json_entry(FILE *fp, int *first, const char *key, const char *val)
{
    if (!(*first))
    {
        fprintf(fp, ",\n");
    }
    *first = 0;
    char   kbuf[1100];
    char   ebuf[2048];
    size_t kpos = 0;
    size_t pos  = 0;
    kbuf[0]     = '\0';
    ebuf[0]     = '\0';
    append_json_escaped_str(kbuf, sizeof(kbuf), &kpos, key);
    if (is_json_number(val))
    {
        snprintf(ebuf, sizeof(ebuf), "%s", val);
    }
    else
    {
        append_json_escaped_str(ebuf, sizeof(ebuf), &pos, val);
    }
    fprintf(fp, "  %s: %s", kbuf, ebuf);
}

/**
 * get_alloc() - run getter @fn(@a, @b, buf, size) with a growing heap buffer until nothing
 * was dropped (-2). Returns the buffer (caller frees) or NULL on error.
 */
static char *get_alloc(int (*fn)(const char *, const char *, char *, size_t),
                       const char *a,
                       const char *b)
{
    for (size_t sz = 16384; sz <= ((size_t) 64 << 20); sz *= 4)
    {
        char *buf = (char *) malloc(sz);
        if (buf == NULL)
        {
            return NULL;
        }
        buf[0] = '\0';
        int rc = fn(a, b, buf, sz);
        if (rc == 0)
        {
            return buf;
        }
        free(buf);
        if (rc != -2)
        {
            return NULL;
        }
    }
    return NULL;
}

/**
 * @brief Export FPS parameters and stream FITS keywords to JSON (`@snapshot`).
 */
int milk_shm_prop_snapshot(int argc, char **argv)
{
    const char *out_file = NULL;
    char       *filters[64];
    int         nfilters = 0;

    for (int i = 1; i < argc; i++)
    {
        if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc)
        {
            out_file = argv[++i];
        }
        else if (is_snapshot_filter_arg(argv[i]))
        {
            if (nfilters < 64)
            {
                filters[nfilters++] = argv[i];
            }
        }
        else if (out_file == NULL && argv[i][0] != '-')
        {
            out_file = argv[i];
        }
    }

    if (out_file != NULL && strcmp(out_file, "-") == 0)
    {
        out_file = NULL;
    }

    FILE *fp = (out_file != NULL) ? fopen(out_file, "w") : stdout;
    if (fp == NULL)
    {
        fprintf(stderr, "Error: @snapshot: cannot open '%s' for writing\n", out_file);
        return 1;
    }

    fprintf(fp, "{\n");
    int   first = 1;
    int   rc    = 0;
    char *lbuf  = get_alloc(list_entities_in_namespace, "f", "*");
    if (lbuf == NULL)
    {
        fprintf(stderr, "Error: @snapshot: cannot list FPS instances\n");
        rc = 1;
    }

    /* 1. Active FPS parameters */
    char *save_ent = NULL;
    for (char *ent = (lbuf != NULL) ? strtok_r(lbuf, " \t\r\n", &save_ent) : NULL; ent != NULL;
         ent       = strtok_r(NULL, " \t\r\n", &save_ent))
    {
        char *kv = get_alloc(milk_shm_prop_get_fps, ent, "kv");
        if (kv == NULL)
        {
            fprintf(stderr, "Error: @snapshot: cannot read FPS '%s'\n", ent);
            rc = 1;
        }
        else if (kv[0] != '\0')
        {
            char *save_ln = NULL;
            for (char *ln = strtok_r(kv, "\r\n", &save_ln); ln != NULL;
                 ln       = strtok_r(NULL, "\r\n", &save_ln))
            {
                char *tab = strchr(ln, '\t');
                if (tab == NULL)
                {
                    continue;
                }
                *tab          = '\0';
                const char *k = ln;
                const char *v = tab + 1;
                if (k[0] == '\0' || is_fps_readonly_meta_key(k))
                {
                    continue;
                }
                char fullkey[512];
                snprintf(fullkey, sizeof(fullkey), "f.%s.%s", ent, k);
                if (snapshot_key_matches(fullkey, nfilters, filters))
                {
                    emit_snapshot_json_entry(fp, &first, fullkey, v);
                }
            }
        }
        free(kv);
    }
    free(lbuf);

    /* 2. Active Stream FITS keywords */
    lbuf = get_alloc(list_entities_in_namespace, "s", "*");
    if (lbuf == NULL)
    {
        fprintf(stderr, "Error: @snapshot: cannot list streams\n");
        rc = 1;
    }
    save_ent = NULL;
    for (char *ent = (lbuf != NULL) ? strtok_r(lbuf, " \t\r\n", &save_ent) : NULL; ent != NULL;
         ent       = strtok_r(NULL, " \t\r\n", &save_ent))
    {
        char kv[8192];
        kv[0] = '\0';
        if (milk_shm_prop_get_stream(ent, "kw.kv", kv, sizeof(kv)) == 0 && kv[0] != '\0')
        {
            char *save_ln = NULL;
            for (char *ln = strtok_r(kv, "\r\n", &save_ln); ln != NULL;
                 ln       = strtok_r(NULL, "\r\n", &save_ln))
            {
                char *tab = strchr(ln, '\t');
                if (tab == NULL)
                {
                    continue;
                }
                *tab          = '\0';
                const char *k = ln;
                const char *v = tab + 1;
                if (k[0] == '\0')
                {
                    continue;
                }
                char fullkey[512];
                snprintf(fullkey, sizeof(fullkey), "s.%s.kw.%s", ent, k);
                if (snapshot_key_matches(fullkey, nfilters, filters))
                {
                    emit_snapshot_json_entry(fp, &first, fullkey, v);
                }
            }
        }
    }

    free(lbuf);

    fprintf(fp, "\n}\n");
    if (fp != stdout)
    {
        fclose(fp);
    }
    return rc;
}

static char *read_file_all(const char *path)
{
    FILE *fp = (strcmp(path, "-") == 0) ? stdin : fopen(path, "r");
    if (fp == NULL)
    {
        return NULL;
    }
    size_t cap = 8192;
    size_t len = 0;
    char  *buf = (char *) malloc(cap);
    if (buf == NULL)
    {
        if (fp != stdin)
        {
            fclose(fp);
        }
        return NULL;
    }
    size_t nr;
    while ((nr = fread(buf + len, 1, cap - len - 1, fp)) > 0)
    {
        len += nr;
        if (len + 1024 >= cap)
        {
            cap *= 2;
            char *nb = (char *) realloc(buf, cap);
            if (nb == NULL)
            {
                free(buf);
                if (fp != stdin)
                {
                    fclose(fp);
                }
                return NULL;
            }
            buf = nb;
        }
    }
    buf[len] = '\0';
    if (fp != stdin)
    {
        fclose(fp);
    }
    return buf;
}

static int scan_next_json_kv(const char **cursor, char *key, size_t keysz, char *val, size_t valsz)
{
    const char *p = *cursor;
    while (*p != '\0')
    {
        while (*p != '\0' && *p != '"')
        {
            p++;
        }
        if (*p != '"')
        {
            *cursor = p;
            return 0;
        }
        p++;
        size_t ki = 0;
        while (*p != '\0' && *p != '"')
        {
            if (*p == '\\' && p[1] != '\0')
            {
                p++;
            }
            if (ki + 1 < keysz)
            {
                key[ki++] = *p;
            }
            p++;
        }
        key[ki] = '\0';
        if (*p == '"')
        {
            p++;
        }
        while (*p != '\0' && isspace((unsigned char) *p))
        {
            p++;
        }
        if (*p != ':')
        {
            continue;
        }
        p++;
        while (*p != '\0' && isspace((unsigned char) *p))
        {
            p++;
        }

        size_t vi = 0;
        if (*p == '"')
        {
            p++;
            while (*p != '\0' && *p != '"')
            {
                if (*p == '\\' && p[1] != '\0')
                {
                    p++;
                    char c = *p;
                    if (c == 'n')
                    {
                        c = '\n';
                    }
                    else if (c == 'r')
                    {
                        c = '\r';
                    }
                    else if (c == 't')
                    {
                        c = '\t';
                    }
                    if (vi + 1 < valsz)
                    {
                        val[vi++] = c;
                    }
                    p++;
                    continue;
                }
                if (vi + 1 < valsz)
                {
                    val[vi++] = *p;
                }
                p++;
            }
            if (*p == '"')
            {
                p++;
            }
        }
        else
        {
            while (*p != '\0' && *p != ',' && *p != '}' && !isspace((unsigned char) *p))
            {
                if (vi + 1 < valsz)
                {
                    val[vi++] = *p;
                }
                p++;
            }
        }
        val[vi] = '\0';
        *cursor = p;
        return 1;
    }
    *cursor = p;
    return 0;
}

static int snapshot_values_equal(const char *a, const char *b)
{
    if (strcmp(a, b) == 0)
    {
        return 1;
    }
    char  *ea = NULL;
    char  *eb = NULL;
    double da = strtod(a, &ea);
    double db = strtod(b, &eb);
    if (a[0] != '\0' && b[0] != '\0' && ea != a && *ea == '\0' && eb != b && *eb == '\0')
    {
        double scale = fmax(1.0, fmax(fabs(da), fabs(db)));
        return (fabs(da - db) <= 1.0e-9 * scale) ? 1 : 0;
    }
    return 0;
}

/**
 * @brief Compare live FPS/stream state against a saved JSON snapshot (`@diff`).
 */
int milk_shm_prop_diff(int argc, char **argv)
{
    const char *in_file = NULL;
    char       *filters[64];
    int         nfilters = 0;

    for (int i = 1; i < argc; i++)
    {
        if (is_snapshot_filter_arg(argv[i]))
        {
            if (nfilters < 64)
            {
                filters[nfilters++] = argv[i];
            }
        }
        else if (in_file == NULL && argv[i][0] != '-')
        {
            in_file = argv[i];
        }
    }

    if (in_file == NULL)
    {
        fprintf(stderr, "Usage: @diff <snapshot.json> [glob_filter...]\n");
        return 2;
    }

    char *content = read_file_all(in_file);
    if (content == NULL)
    {
        fprintf(stderr, "Error: @diff: cannot read '%s'\n", in_file);
        return 2;
    }

    const char *cur = content;
    char        key[512];
    char        sval[1024];
    int         diff_count = 0;

    while (scan_next_json_kv(&cur, key, sizeof(key), sval, sizeof(sval)))
    {
        if (!snapshot_key_matches(key, nfilters, filters))
        {
            continue;
        }
        char lval[1024];
        if (milk_shm_prop_get(key, lval, sizeof(lval)) != 0)
        {
            printf("%s: %s -> <MISSING>\n", key, sval);
            diff_count++;
        }
        else if (!snapshot_values_equal(sval, lval))
        {
            printf("%s: %s -> %s\n", key, sval, lval);
            diff_count++;
        }
    }

    free(content);
    return (diff_count == 0) ? 0 : 1;
}

/**
 * @brief Restore FPS parameters and stream FITS keywords from a JSON snapshot (`@restore`).
 */
int milk_shm_prop_restore(int argc, char **argv)
{
    const char *in_file = NULL;
    char       *filters[64];
    int         nfilters = 0;
    int         dry_run  = 0;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--dry-run") == 0 || strcmp(argv[i], "-n") == 0)
        {
            dry_run = 1;
        }
        else if (is_snapshot_filter_arg(argv[i]))
        {
            if (nfilters < 64)
            {
                filters[nfilters++] = argv[i];
            }
        }
        else if (in_file == NULL && argv[i][0] != '-')
        {
            in_file = argv[i];
        }
    }

    if (in_file == NULL)
    {
        fprintf(stderr, "Usage: @restore [--dry-run|-n] <snapshot.json> [glob_filter...]\n");
        return 1;
    }

    char *content = read_file_all(in_file);
    if (content == NULL)
    {
        fprintf(stderr, "Error: @restore: cannot read '%s'\n", in_file);
        return 1;
    }

    const char *cur = content;
    char        key[512];
    char        sval[1024];
    int         rc = 0;

    while (scan_next_json_kv(&cur, key, sizeof(key), sval, sizeof(sval)))
    {
        if (!snapshot_key_matches(key, nfilters, filters))
        {
            continue;
        }
        if (dry_run)
        {
            printf("@set %s = %s\n", key, sval);
        }
        else if (shm_prop_set_impl(key, sval, NULL, 0) != 0) // stored "-" is a value, not stdin
        {
            rc = 1;
        }
    }

    free(content);
    return rc;
}
