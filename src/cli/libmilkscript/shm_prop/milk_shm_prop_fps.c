// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_shm_prop_fps.c
 * @brief Function Parameter Structure (`@f.` / `@fps.`) handler for entity ops.
 *
 * Role & Goal:
 *   Implements shared-memory parameter reading, wildcard (`*`) parameter
 *   enumeration, typed parameter writing (`cli_fps_set_param` /
 *   `milk_shm_prop_set_fps`), and existence / running-state checks for FPS
 *   structures in `/dev/shm`.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE // RTLD_DEFAULT
#endif

#include <dlfcn.h>
#include <errno.h>
#include <fnmatch.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/types.h>
#include <unistd.h>

#include "CLIcore.h"
#include "computeunit.h"
#include "fps.h"
#include "fps_GetParamIndex.h"
#include "fps_connect.h"
#include "fps_paramvalue.h"
#include "fps_printparameter_valuestring.h"
#include "milk_shm_cache.h"
#include "milk_shm_prop.h"
#include "milk_shm_prop_internal.h"
#include "milk_spawn.h"

extern char **environ;

/* Lines of a unit's output that f.<name>.run.out and conf.out return */
#define FPS_OUT_TAIL_LINES 20

/** fmt_float_roundtrip() - shortest %g form of @v that strtof() maps back to @v. */
static void fmt_float_roundtrip(char *out, size_t outsz, float v)
{
    for (int prec = 6; prec <= 9; prec++)
    {
        snprintf(out, outsz, "%.*g", prec, (double) v);
        if (strtof(out, NULL) == v)
        {
            return;
        }
    }
}

/** fps_is_trigmode() - 1 if @p is a unit's procinfo.triggermode, which scripts read and write by name. */
static int fps_is_trigmode(const FPS_PARAM *p)
{
    static const char suf[] = "procinfo.triggermode";
    size_t            kl    = strlen(p->keywordfull);
    size_t            sl    = sizeof(suf) - 1;

    return (p->type & FPTYPE_INT64) && kl >= sl && strcmp(p->keywordfull + kl - sl, suf) == 0 &&
           (kl == sl || p->keywordfull[kl - sl - 1] == '.');
}

/**
 * fps_value_in() - the value to write into @p for @valstr: a trigger mode name becomes its number in @buf.
 * Return: @valstr, @buf, or NULL (said on stderr) for a trigger mode that is neither.
 */
static const char *fps_value_in(const FPS_PARAM *p, const char *valstr, char *buf, size_t bufsz)
{
    if (!fps_is_trigmode(p))
    {
        return valstr;
    }
    int m = milk_trigmode_parse(valstr);
    if (m < 0)
    {
        char names[128] = "";
        for (long i = 0; milk_trigmode_name(i) != NULL; i++)
        {
            size_t l = strlen(names);
            snprintf(names + l, sizeof(names) - l, "%s ", milk_trigmode_name(i));
        }
        fprintf(stderr, "Error: trigger mode '%s': one of %sor its number\n", valstr, names);
        return NULL;
    }
    snprintf(buf, bufsz, "%d", m);
    return buf;
}

/**
 * fps_param_value_str() - parameter value for scripts: exact floats, bare stream names and trigger mode
 * names (functionparameter_GetParamValueString() is the decorated, rounded display form).
 */
static void fps_param_value_str(FPS_PARAM *p, char *out, size_t outsz)
{
    switch (p->type)
    {
    case FPTYPE_FLOAT64:
        milk_shm_prop_fmt_double(out, outsz, p->val.f64[0]);
        break;
    case FPTYPE_FLOAT32:
        fmt_float_roundtrip(out, outsz, p->val.f32[0]);
        break;
    case FPTYPE_STREAMNAME:
        snprintf(out, outsz, "%s", p->val.string[0]);
        break;
    default:
        if (fps_is_trigmode(p) && milk_trigmode_name(p->val.i64[0]) != NULL)
        {
            snprintf(out, outsz, "%s", milk_trigmode_name(p->val.i64[0]));
            break;
        }
        functionparameter_GetParamValueString(p, out, (int) outsz);
        break;
    }
}

/**
 * fps_check_msgs() - the messages of the FPS's last parameter check, one per line as
 * `<parameter>: <message>`, into @out (empty when it passed or never ran).
 */
static void fps_check_msgs(const FPS *fps, char *out, size_t outsz)
{
    const char *fname = fps->md->name;
    size_t      flen  = strlen(fname);
    size_t      pos   = 0;
    long        nmsg  = (fps->md->msgcnt < FPS_NB_MSG) ? fps->md->msgcnt : FPS_NB_MSG;

    out[0] = '\0';
    for (long i = 0; i < nmsg && pos < outsz; i++)
    {
        int         pi = fps->md->msgpindex[i];
        const char *kw = (pi >= 0 && pi < fps->md->NBparamMAX) ? fps->parray[pi].keywordfull : "";
        while (*kw == '.')
        {
            kw++;
        }
        if (strncmp(kw, fname, flen) == 0 && kw[flen] == '.')
        {
            kw += flen + 1;
        }
        int n = snprintf(out + pos, outsz - pos, "%s%s: %.*s", (pos > 0) ? "\n" : "", kw,
                         FUNCTION_PARAMETER_STRUCT_MSG_LEN, fps->md->message[i]);
        if (n < 0)
        {
            break;
        }
        pos += (size_t) n;
    }
}

/**
 * fps_param_exact() - index of the parameter named exactly @pname (with or without the FPS name and the
 * leading dot), or -1. functionparameter_GetParamIndex() also matches substrings: `conf` would find
 * `conf.timestring` and hide the unit's properties and actions.
 */
static int fps_param_exact(FPS *fps, const char *pname)
{
    const char *fname = fps->md->name;
    size_t      flen  = strlen(fname);
    const char *rel   = pname + (pname[0] == '.');

    for (int pi = 0; pi < fps->md->NBparamMAX; pi++)
    {
        const char *kw = fps->parray[pi].keywordfull;
        if ((fps->parray[pi].fpflag & FPFLAG_ACTIVE) &&
            (strcmp(kw, pname) == 0 || (kw[0] == '.' && strcmp(kw + 1, rel) == 0) ||
             (strncmp(kw, fname, flen) == 0 && kw[flen] == '.' && strcmp(kw + flen + 1, rel) == 0)))
        {
            return pi;
        }
    }
    return -1;
}

/** fps_meta_prop() - 1 if @prop is a property of the FPS itself rather than a parameter. */
static int fps_meta_prop(const char *prop)
{
    static const char *const meta[] = { "status",     "state",       "run",      "conf",
                                        "session",    "run.out",     "conf.out", "confrunning",
                                        "runrunning", "pid",         "confpid",  "runpid",
                                        "nbparam",    "description", "ok",       "errors",
                                        "msg" };

    for (size_t i = 0; i < sizeof(meta) / sizeof(meta[0]); i++)
    {
        if (strcmp(prop, meta[i]) == 0)
        {
            return 1;
        }
    }
    return 0;
}

/**
 * @brief Read an FPS parameter, metadata attribute, or parameter list (`*` / `list`).
 *
 * @param name  FPS shared-memory name
 * @param prop  Parameter key, metadata key (`status`, `run`, `conf`, `session`, `run.out`, `pid`, ...), or `*`/`list`
 * @param out   Output buffer
 * @param outsz Size of @out in bytes
 * @return 0 on success, -1 if FPS or parameter not found
 */
int milk_shm_prop_get_fps(const char *name, const char *prop, char *out, size_t outsz)
{
    FPS *cfps = milk_shm_cache_fps(name); /* cached connection: never disconnected here */
    if (cfps == NULL)
    {
        return -1;
    }
    FPS fps = *cfps; /* md/parray still point into the shared mapping */

    if (strcmp(prop, "*") == 0 || strcmp(prop, "list") == 0)
    {
        size_t pos  = 0;
        size_t nlen = strlen(name);
        for (int pi = 0; pi < fps.md->NBparamMAX; pi++)
        {
            if (fps.parray[pi].fpflag & FPFLAG_ACTIVE)
            {
                const char *kw = (fps.parray[pi].keywordfull[0] != '\0')
                                     ? fps.parray[pi].keywordfull
                                     : fps.parray[pi].keyword[0];
                while (*kw == '.')
                {
                    kw++;
                }
                if (strncmp(kw, name, nlen) == 0 && kw[nlen] == '.')
                {
                    kw += nlen + 1;
                }
                int n = snprintf(out + pos, outsz - pos, "%s%s", (pos > 0) ? " " : "", kw);
                if (n > 0 && pos + (size_t) n < outsz)
                {
                    pos += (size_t) n;
                }
            }
        }
        return 0;
    }

    if (strcmp(prop, "kv") == 0 && fps.md != NULL)
    {
        FPS_UNIT_STATUS rs;
        FPS_UNIT_STATUS cs;
        fps_unit_status_fps(&fps, FPS_UNIT_RUN, &rs, FPS_UNIT_REPAIR);
        fps_unit_status_fps(&fps, FPS_UNIT_CONF, &cs, FPS_UNIT_REPAIR);
        int cnt = 0;
        for (int pi = 0; pi < fps.md->NBparamMAX; pi++)
        {
            if (fps.parray[pi].fpflag & FPFLAG_ACTIVE)
            {
                cnt++;
            }
        }
        pid_t  pid  = (fps.md->runpid > 0) ? fps.md->runpid : fps.md->confpid;
        size_t pos  = 0;
        int    full = 0;
        size_t nlen = strlen(name);
        int    n0 =
            snprintf(out, outsz,
                     "status\t%s\nconf\t%s\nrun\t%s\nconfrunning\t%d\nrunrunning\t%d\nsession\t%s\n"
                     "pid\t%ld\nconfpid\t%ld\nrunpid\t%ld\nnbparam\t%d\ndescription\t%s\nok\t%d\n"
                     "errors\t%u",
                     fps_unit_state_name(rs.state), fps_unit_state_name(cs.state),
                     fps_unit_state_name(rs.state), fps_unit_state_alive(cs.state),
                     fps_unit_state_alive(rs.state), rs.session, (long) pid, (long) fps.md->confpid,
                     (long) fps.md->runpid, cnt, fps.md->description,
                     (fps.md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CHECKOK) ? 1 : 0,
                     fps.md->conferrcnt);
        if (n0 > 0 && (size_t) n0 < outsz)
        {
            pos = (size_t) n0;
        }
        for (int pi = 0; pi < fps.md->NBparamMAX; pi++)
        {
            if (fps.parray[pi].fpflag & FPFLAG_ACTIVE)
            {
                const char *kw = (fps.parray[pi].keywordfull[0] != '\0')
                                     ? fps.parray[pi].keywordfull
                                     : fps.parray[pi].keyword[0];
                while (*kw == '.')
                {
                    kw++;
                }
                if (strncmp(kw, name, nlen) == 0 && kw[nlen] == '.')
                {
                    kw += nlen + 1;
                }
                char vbuf[256];
                vbuf[0] = '\0';
                fps_param_value_str(&fps.parray[pi], vbuf, sizeof(vbuf));
                int n = snprintf(out + pos, outsz - pos, "\n%s\t%s", kw, vbuf);
                if (n > 0 && pos + (size_t) n < outsz)
                {
                    pos += (size_t) n;
                }
                else
                {
                    out[pos] = '\0';
                    full     = 1;
                }
            }
        }
        return full ? -2 : 0; // -2: buffer too small
    }

    // an exact parameter name, then the FPS's own properties, then the lookup's other matches
    int pindex = fps_param_exact(&fps, prop);
    if (pindex < 0 && !fps_meta_prop(prop))
    {
        pindex = functionparameter_GetParamIndex(&fps, prop);
        if (pindex < 0)
        {
            char dname[256];
            snprintf(dname, sizeof(dname), ".%s", prop);
            pindex = functionparameter_GetParamIndex(&fps, dname);
        }
    }

    int rc = -1;
    if (pindex >= 0)
    {
        fps_param_value_str(&fps.parray[pindex], out, outsz);
        rc = 0;
    }
    else if (fps.md != NULL)
    {
        // the unit's state (U10), "run.out"/"conf.out" its output; the conf role for conf*
        FPS_UNIT_ROLE   role = (strncmp(prop, "conf", 4) == 0) ? FPS_UNIT_CONF : FPS_UNIT_RUN;
        FPS_UNIT_STATUS st;
        if (strcmp(prop, "status") == 0 || strcmp(prop, "state") == 0 || strcmp(prop, "run") == 0 ||
            strcmp(prop, "conf") == 0)
        {
            fps_unit_status_fps(&fps, role, &st, FPS_UNIT_REPAIR);
            snprintf(out, outsz, "%s", fps_unit_state_name(st.state));
            rc = 0;
        }
        else if (strcmp(prop, "runrunning") == 0 || strcmp(prop, "confrunning") == 0)
        {
            fps_unit_status_fps(&fps, role, &st, FPS_UNIT_REPAIR);
            snprintf(out, outsz, "%d", fps_unit_state_alive(st.state));
            rc = 0;
        }
        else if (strcmp(prop, "session") == 0)
        {
            fps_unit_status_fps(&fps, FPS_UNIT_RUN, &st, FPS_UNIT_REPAIR);
            snprintf(out, outsz, "%s", st.session);
            rc = 0;
        }
        else if (strcmp(prop, "run.out") == 0 || strcmp(prop, "conf.out") == 0)
        {
            int n = fps_unit_out_tail(name, role, FPS_OUT_TAIL_LINES, out, outsz);
            if (n > 0 && out[n - 1] == '\n')
            {
                out[n - 1] = '\0';
            }
            else if (n < 0)
            {
                out[0] = '\0'; // never started: no output yet
            }
            rc = 0;
        }
        else if (strcmp(prop, "pid") == 0)
        {
            pid_t pid = (fps.md->runpid > 0) ? fps.md->runpid : fps.md->confpid;
            snprintf(out, outsz, "%ld", (long) pid);
            rc = 0;
        }
        else if (strcmp(prop, "confpid") == 0)
        {
            snprintf(out, outsz, "%ld", (long) fps.md->confpid);
            rc = 0;
        }
        else if (strcmp(prop, "runpid") == 0)
        {
            snprintf(out, outsz, "%ld", (long) fps.md->runpid);
            rc = 0;
        }
        else if (strcmp(prop, "description") == 0)
        {
            snprintf(out, outsz, "%s", fps.md->description);
            rc = 0;
        }
        else if (strcmp(prop, "nbparam") == 0)
        {
            int cnt = 0;
            for (int pi = 0; pi < fps.md->NBparamMAX; pi++)
            {
                if (fps.parray[pi].fpflag & FPFLAG_ACTIVE)
                {
                    cnt++;
                }
            }
            snprintf(out, outsz, "%d", cnt);
            rc = 0;
        }
        else if (strcmp(prop, "ok") ==
                 0) // the last parameter check: confstep, confstart or runstart
        {
            snprintf(out, outsz, "%d",
                     (fps.md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CHECKOK) ? 1 : 0);
            rc = 0;
        }
        else if (strcmp(prop, "errors") == 0)
        {
            snprintf(out, outsz, "%u", fps.md->conferrcnt);
            rc = 0;
        }
        else if (strcmp(prop, "msg") == 0)
        {
            fps_check_msgs(&fps, out, outsz);
            rc = 0;
        }
    }
    return rc;
}

/*
 * The environment for the programs @set starts: inside bash (the milk builtins) the shell's exported
 * variables, which environ does not follow; environ elsewhere.
 */
static char **child_environ(void)
{
    void (*mk)(void) = (void (*)(void)) dlsym(RTLD_DEFAULT, "maybe_make_export_env");
    char ***ee       = (char ***) dlsym(RTLD_DEFAULT, "export_env");

    if (mk != NULL && ee != NULL)
    {
        mk();
        if (*ee != NULL)
        {
            return *ee;
        }
    }
    return environ;
}

/**
 * try_fps_lifecycle_action() - `@set f.<name>.conf|run=<value>` and the action properties (confstart,
 * confstop, confstep, runstart, runstop, reset): the milk-fps-<op> tool, waited for, its messages to the
 * caller (U11). The state names that `@get` returns are values too, so `@set --restore` puts a unit back.
 * Return: 1 if @pname is one of them, *@out_rc 0 or MILK_SHM_PROP_ACTION_FAILED; 0 otherwise.
 */
static int try_fps_lifecycle_action(FPS        *fps,
                                    const char *fpsname,
                                    const char *pname,
                                    const char *valstr,
                                    int        *out_rc)
{
    static const struct
    {
        const char *word;
        const char *conf;
        const char *run;
    } words[]        = { { "start", "confstart", "runstart" },
                         { "on", "confstart", "runstart" },
                         { "1", "confstart", "runstart" },
                         { "running", "confstart", "runstart" },
                         { "starting", "confstart", "runstart" },
                         { "stop", "confstop", "runstop" },
                         { "off", "confstop", "runstop" },
                         { "0", "confstop", "runstop" },
                         { "stopped", "confstop", "runstop" },
                         { "stopping", "confstop", "runstop" },
                         { "exited", "confstop", "runstop" },
                         { "failed", "confstop", "runstop" },
                         { "step", "confstep", NULL },
                         { "reset", "reset", NULL } };
    const char *op   = NULL;
    int         conf = (strcmp(pname, "conf") == 0);

    if (fps->md == NULL || valstr == NULL)
    {
        return 0;
    }
    *out_rc = MILK_SHM_PROP_ACTION_FAILED; // done here, never sent again to a server (D29)
    if (conf || strcmp(pname, "run") == 0)
    {
        if (strcasestr(valstr, "tmux") != NULL)
        {
            fprintf(stderr,
                    "f.%s.%s=%s: tmux was removed: units start in the background (their output: "
                    "milk-fps-log %s, or @get f.%s.%s.out)\n",
                    fpsname, pname, valstr, fpsname, fpsname, pname);
            return 1;
        }
        for (size_t i = 0; op == NULL && i < sizeof(words) / sizeof(words[0]); i++)
        {
            if (strcasecmp(valstr, words[i].word) == 0)
            {
                op = conf ? words[i].conf : words[i].run;
            }
        }
        if (op == NULL)
        {
            fprintf(stderr, "Error: invalid value '%s' for f.%s.%s (expected start|stop%s)\n",
                    valstr, fpsname, pname, conf ? "|step|reset" : "");
            return 1;
        }
    }
    else if (strcmp(pname, "confstart") == 0 || strcmp(pname, "confstop") == 0 ||
             strcmp(pname, "confstep") == 0 || strcmp(pname, "runstart") == 0 ||
             strcmp(pname, "runstop") == 0 || strcmp(pname, "reset") == 0)
    {
        op = pname;
    }
    else
    {
        return 0;
    }

    int   reset = (strcmp(op, "reset") == 0);
    pid_t pid   = fps_unit_spawn(fpsname, reset ? "confstop" : op, child_environ(), STDOUT_FILENO,
                                 STDERR_FILENO);
    if (pid < 0)
    {
        fprintf(stderr, "f.%s.%s: cannot run milk-fps-%s: %s\n", fpsname, pname,
                reset ? "confstop" : op, strerror(errno));
        return 1;
    }
    if (milkspawn_wait(pid) != 0)
    {
        return 1;
    }
    if (reset)
    {
        fps->md->conferrcnt = 0;
        fps->md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;
    }
    *out_rc = 0;
    return 1;
}

/**
 * @brief Write a value string to a named FPS parameter or execute a lifecycle action.
 *
 * Connects to the named FPS, locates the parameter (or lifecycle property),
 * writes the parsed value via `functionparameter_SetParamValue_fromString()`,
 * and disconnects.
 *
 * @param fpsname  FPS shared-memory name
 * @param pname    Parameter key or lifecycle property inside FPS
 * @param valstr   Value to write (as string)
 * @return 0 on success, -1 on error
 */
int cli_fps_set_param(const char *fpsname, const char *pname, const char *valstr)
{
    FPS *cfps = milk_shm_cache_fps(fpsname); /* cached connection: never disconnected here */
    if (cfps == NULL)
    {
        printf("Error: cannot connect to FPS '%s'\n", fpsname);
        return -1;
    }
    FPS fps = *cfps; /* md/parray still point into the shared mapping */

    if (strchr(pname, '*') != NULL || strchr(pname, '?') != NULL)
    {
        int    matched = 0;
        size_t nlen    = strlen(fpsname);
        for (int pi = 0; pi < fps.md->NBparamMAX; pi++)
        {
            if (fps.parray[pi].fpflag & FPFLAG_ACTIVE)
            {
                const char *kw = (fps.parray[pi].keywordfull[0] != '\0')
                                     ? fps.parray[pi].keywordfull
                                     : fps.parray[pi].keyword[0];
                while (*kw == '.')
                {
                    kw++;
                }
                if (strncmp(kw, fpsname, nlen) == 0 && kw[nlen] == '.')
                {
                    kw += nlen + 1;
                }
                if (fnmatch(pname, kw, 0) == 0 ||
                    fnmatch(pname, fps.parray[pi].keywordfull, 0) == 0)
                {
                    char        vbuf[16];
                    const char *v = fps_value_in(&fps.parray[pi], valstr, vbuf, sizeof(vbuf));
                    if (v != NULL && functionparameter_SetParamValue_fromString(&fps, pi, v) == 0)
                    {
                        matched = 1;
                    }
                }
            }
        }
        return matched ? 0 : -1;
    }

    // an exact parameter name, then the unit actions, then the lookup's other matches
    int pindex = fps_param_exact(&fps, pname);
    int lc_rc  = -1;
    if (pindex < 0 && try_fps_lifecycle_action(&fps, fpsname, pname, valstr, &lc_rc))
    {
        return lc_rc;
    }
    if (pindex < 0)
    {
        pindex = functionparameter_GetParamIndex(&fps, pname);
    }
    if (pindex < 0)
    {
        char dotname[512];
        snprintf(dotname, sizeof(dotname), ".%s", pname);
        pindex = functionparameter_GetParamIndex(&fps, dotname);
    }

    if (pindex < 0)
    {
        printf("Error: parameter '%s' not found in FPS '%s'\n", pname, fpsname);
        return -1;
    }

    if (fps.parray[pindex].keywordfull[0] == '\0')
    {
        snprintf(fps.parray[pindex].keywordfull, FUNCTION_PARAMETER_KEYWORD_STRMAXLEN, ".%s",
                 pname);
    }

    char vbuf[16];
    valstr = fps_value_in(&fps.parray[pindex], valstr, vbuf, sizeof(vbuf));
    if (valstr == NULL)
    {
        return -1;
    }
    if (functionparameter_SetParamValue_fromString(&fps, pindex, valstr) == 0)
    {
        return 0;
    }

    /* Fallback for hex/octal integer literals not accepted by the base-10 setter:
     * the whole string must parse and fit, otherwise the parameter is left unchanged. */
    uint32_t    ptype = fps.parray[pindex].type;
    const char *kw    = fps.parray[pindex].keywordfull;
    char       *endp  = NULL;
    int         rc    = -1;
    errno             = 0;
    if (ptype & (FPTYPE_INT32 | FPTYPE_INT64))
    {
        long long v  = strtoll(valstr, &endp, 0);
        int       ok = (endp != valstr && *endp == '\0' && errno == 0);
        if (ok && (ptype & FPTYPE_INT32) && v >= INT32_MIN && v <= INT32_MAX)
        {
            rc = functionparameter_SetParamValue_INT32(&fps, kw, (int32_t) v) == EXIT_SUCCESS ? 0
                                                                                              : -1;
        }
        else if (ok && (ptype & FPTYPE_INT64))
        {
            rc = functionparameter_SetParamValue_INT64(&fps, kw, (int64_t) v) == EXIT_SUCCESS ? 0
                                                                                              : -1;
        }
    }
    else if ((ptype & (FPTYPE_UINT32 | FPTYPE_UINT64)) && valstr[0] != '-')
    {
        unsigned long long v  = strtoull(valstr, &endp, 0);
        int                ok = (endp != valstr && *endp == '\0' && errno == 0);
        if (ok && (ptype & FPTYPE_UINT32) && v <= UINT32_MAX)
        {
            rc = functionparameter_SetParamValue_UINT32(&fps, kw, (uint32_t) v) == EXIT_SUCCESS
                     ? 0
                     : -1;
        }
        else if (ok && (ptype & FPTYPE_UINT64))
        {
            rc = functionparameter_SetParamValue_UINT64(&fps, kw, (uint64_t) v) == EXIT_SUCCESS
                     ? 0
                     : -1;
        }
    }
    if (rc != 0)
    {
        fprintf(stderr, "Error: invalid value '%s' for '%s' (type 0x%x) in FPS '%s'\n", valstr,
                pname, ptype, fpsname);
    }
    return rc;
}

/**
 * @brief Write a value string to a named FPS parameter (`@f.<name>.<prop>`).
 *
 * @param name FPS shared-memory name
 * @param prop Parameter key inside FPS
 * @param val  Value string
 * @return 0 on success, -1 on error
 */
int milk_shm_prop_set_fps(const char *name, const char *prop, const char *val)
{
    int rc = cli_fps_set_param(name, prop, val);
    return (rc == 0 || rc == MILK_SHM_PROP_ACTION_FAILED) ? rc : -1;
}

/**
 * @brief Check whether a named FPS exists (and optionally is running).
 *
 * @param name          FPS shared-memory name
 * @param check_running If non-zero, require the run process of the unit to be alive (U10)
 * @return 1 if exists/running, 0 otherwise
 */
int milk_shm_prop_exists_fps(const char *name, int check_running)
{
    char shmpath[STRINGMAXLEN_DIRNAME + 256];
    snprintf(shmpath, sizeof(shmpath), "%s/%s.fps.shm", milk_shm_prop_shmdir(), name);
    if (access(shmpath, F_OK) != 0)
    {
        return 0;
    }
    if (!check_running)
    {
        return 1;
    }
    FPS            *cfps = milk_shm_cache_fps(name);
    FPS_UNIT_STATUS st;
    return (cfps != NULL && fps_unit_status_fps(cfps, FPS_UNIT_RUN, &st, FPS_UNIT_REPAIR) == 0 &&
            fps_unit_state_alive(st.state));
}
