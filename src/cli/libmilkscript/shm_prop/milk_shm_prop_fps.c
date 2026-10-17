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

#include <errno.h>
#include <fnmatch.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "CLIcore.h"
#include "fps.h"
#include "fps_CONFstop.h"
#include "fps_GetParamIndex.h"
#include "fps_connect.h"
#include "fps_paramvalue.h"
#include "fps_printparameter_valuestring.h"
#include "fps_tmux.h"
#include "milk_shm_cache.h"
#include "milk_shm_prop.h"
#include "milk_shm_prop_internal.h"
#include "milk_path.h"

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

/**
 * fps_param_value_str() - parameter value for scripts: exact floats and bare stream names
 * (functionparameter_GetParamValueString() is the decorated, rounded display form).
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
        functionparameter_GetParamValueString(p, out, (int) outsz);
        break;
    }
}

/**
 * @brief Read an FPS parameter, metadata attribute, or parameter list (`*` / `list`).
 *
 * @param name  FPS shared-memory name
 * @param prop  Parameter key, metadata key (`status`, `conf`, `run`, `pid`, `nbparam`), or `*`/`list`
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
        const char *st = "IDLE";
        if (fps.md->status &
            (FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN | FUNCTION_PARAMETER_STRUCT_STATUS_RUN |
             FUNCTION_PARAMETER_STRUCT_STATUS_RUNLOOP))
        {
            st = "RUN";
        }
        else if (fps.md->status &
                 (FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF | FUNCTION_PARAMETER_STRUCT_STATUS_CONF))
        {
            st = "CONF";
        }
        int confrun = (fps.md->status & (FUNCTION_PARAMETER_STRUCT_STATUS_CONF |
                                         FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF))
                          ? 1
                          : 0;
        int runrun  = (fps.md->status & (FUNCTION_PARAMETER_STRUCT_STATUS_RUN |
                                         FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN |
                                         FUNCTION_PARAMETER_STRUCT_STATUS_RUNLOOP))
                          ? 1
                          : 0;
        int cnt     = 0;
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
                     "status\t%s\nconf\t%s\nrun\t%s\nconfrunning\t%d\nrunrunning\t%d\n"
                     "pid\t%ld\nconfpid\t%ld\nrunpid\t%ld\nnbparam\t%d\ndescription\t%s",
                     st, confrun ? "ON" : "OFF", runrun ? "ON" : "OFF", confrun, runrun, (long) pid,
                     (long) fps.md->confpid, (long) fps.md->runpid, cnt, fps.md->description);
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

    int pindex = functionparameter_GetParamIndex(&fps, prop);
    if (pindex < 0)
    {
        char dname[256];
        snprintf(dname, sizeof(dname), ".%s", prop);
        pindex = functionparameter_GetParamIndex(&fps, dname);
    }

    int rc = -1;
    if (pindex >= 0)
    {
        fps_param_value_str(&fps.parray[pindex], out, outsz);
        rc = 0;
    }
    else if (fps.md != NULL)
    {
        if (strcmp(prop, "status") == 0 || strcmp(prop, "state") == 0)
        {
            if (fps.md->status &
                (FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN | FUNCTION_PARAMETER_STRUCT_STATUS_RUN |
                 FUNCTION_PARAMETER_STRUCT_STATUS_RUNLOOP))
            {
                snprintf(out, outsz, "RUN");
            }
            else if (fps.md->status & (FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF |
                                       FUNCTION_PARAMETER_STRUCT_STATUS_CONF))
            {
                snprintf(out, outsz, "CONF");
            }
            else
            {
                snprintf(out, outsz, "IDLE");
            }
            rc = 0;
        }
        else if (strcmp(prop, "conf") == 0)
        {
            snprintf(out, outsz, "%s",
                     (fps.md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF) ? "ON" : "OFF");
            rc = 0;
        }
        else if (strcmp(prop, "run") == 0)
        {
            snprintf(out, outsz, "%s",
                     (fps.md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN) ? "ON" : "OFF");
            rc = 0;
        }
        else if (strcmp(prop, "confrunning") == 0)
        {
            snprintf(out, outsz, "%d",
                     (fps.md->status & (FUNCTION_PARAMETER_STRUCT_STATUS_CONF |
                                        FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF))
                         ? 1
                         : 0);
            rc = 0;
        }
        else if (strcmp(prop, "runrunning") == 0)
        {
            snprintf(out, outsz, "%d",
                     (fps.md->status & (FUNCTION_PARAMETER_STRUCT_STATUS_RUN |
                                        FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN))
                         ? 1
                         : 0);
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
    }
    return rc;
}

/**
 * @brief Parse and execute an FPS lifecycle command (`confstart`, `confstop`,
 *        `confstep`, `runstart`, `runstop`, `reset`) when addressed via `@set f.<name>.<prop>`.
 *
 * @param fps      Connected FPS handle
 * @param fpsname  FPS shared-memory name
 * @param pname    Property name (`conf`, `run`, `confstart`, `confstop`, `confstep`, `runstart`, `runstop`, `reset`)
 * @param valstr   Value string (`start`, `stop`, `step`, `reset`, `1`, `0`, `ON`, `OFF`, optionally with `:tmux` or `tmux`)
 * @param out_rc   Receives 0 on success, -1 on error
 * @return 1 if @pname matched a lifecycle action, 0 otherwise
 */
static int try_fps_lifecycle_action(FPS        *fps,
                                    const char *fpsname,
                                    const char *pname,
                                    const char *valstr,
                                    int        *out_rc)
{
    if (pname == NULL || valstr == NULL || fps->md == NULL)
    {
        return 0;
    }

    char vclean[128];
    snprintf(vclean, sizeof(vclean), "%s", valstr);
    int   use_tmux = 0;
    char *colon    = strchr(vclean, ':');
    if (colon != NULL &&
        (strcasecmp(colon + 1, "tmux") == 0 || strcasecmp(colon + 1, "-tmux") == 0))
    {
        *colon   = '\0';
        use_tmux = 1;
    }
    else if (strcasecmp(vclean, "tmux") == 0 || strcasecmp(vclean, "-tmux") == 0)
    {
        use_tmux = 1;
        snprintf(vclean, sizeof(vclean), "1");
    }

    const char *command = NULL;
    if (strcmp(pname, "conf") == 0)
    {
        if (strcasecmp(vclean, "start") == 0 || strcasecmp(vclean, "on") == 0 ||
            strcmp(vclean, "1") == 0)
        {
            command = "confstart";
        }
        else if (strcasecmp(vclean, "stop") == 0 || strcasecmp(vclean, "off") == 0 ||
                 strcmp(vclean, "0") == 0)
        {
            command = "confstop";
        }
        else if (strcasecmp(vclean, "step") == 0)
        {
            command = "confstep";
        }
        else if (strcasecmp(vclean, "reset") == 0)
        {
            command = "reset";
        }
        else
        {
            printf("Error: invalid value '%s' for f.%s.conf (expected start|stop|step|reset)\n",
                   valstr, fpsname);
            *out_rc = -1;
            return 1;
        }
    }
    else if (strcmp(pname, "run") == 0)
    {
        if (strcasecmp(vclean, "start") == 0 || strcasecmp(vclean, "on") == 0 ||
            strcmp(vclean, "1") == 0)
        {
            command = "runstart";
        }
        else if (strcasecmp(vclean, "stop") == 0 || strcasecmp(vclean, "off") == 0 ||
                 strcmp(vclean, "0") == 0)
        {
            command = "runstop";
        }
        else
        {
            printf("Error: invalid value '%s' for f.%s.run (expected start|stop)\n", valstr,
                   fpsname);
            *out_rc = -1;
            return 1;
        }
    }
    else if (strcmp(pname, "confstart") == 0 || strcmp(pname, "confstop") == 0 ||
             strcmp(pname, "confstep") == 0 || strcmp(pname, "runstart") == 0 ||
             strcmp(pname, "runstop") == 0 || strcmp(pname, "reset") == 0)
    {
        command = pname;
    }
    else
    {
        return 0;
    }

    if (strcmp(command, "reset") == 0)
    {
        functionparameter_CONFstop(fps);
        fps->md->status &=
            ~(FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF | FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN);
        fps->md->conferrcnt = 0;
        fps->md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;
        *out_rc = 0;
        return 1;
    }

    int has_exec =
        (fps->md->execfullpath[0] != '\0' && strcmp(fps->md->execfullpath, "unknown") != 0 &&
         access(fps->md->execfullpath, X_OK) == 0);

    if (use_tmux && has_exec)
    {
        functionparameter_FPS_tmux_ensure(fps);
        char extra_args[256];
        snprintf(extra_args, sizeof(extra_args), " %s:%s", fpsname, command);
        if (functionparameter_FPS_tmux_send_dispatch(fpsname, command, fps->md->execfullpath,
                                                     extra_args) == 0)
        {
            *out_rc = 0;
            return 1;
        }
    }

    if (has_exec)
    {
        char arg[512];
        char qexec[1024];
        char qarg[1024];
        char cmdline[2100];
        snprintf(arg, sizeof(arg), "%s:%s", fpsname, command);
        if (milkpath_shell_quote(qexec, sizeof(qexec), fps->md->execfullpath) != 0 ||
            milkpath_shell_quote(qarg, sizeof(qarg), arg) != 0)
        {
            *out_rc = -1;
            return 1;
        }
        snprintf(cmdline, sizeof(cmdline), "%s %s", qexec, qarg);
        int ret = system(cmdline);
        *out_rc = (ret != -1 && WIFEXITED(ret) && WEXITSTATUS(ret) == 0) ? 0 : -1;
        return 1;
    }

    /* Direct SHM state update when no external executable is bound */
    if (strcmp(command, "confstart") == 0)
    {
        fps->md->status |= FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF;
        fps->md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_CONFRUN;
    }
    else if (strcmp(command, "confstop") == 0)
    {
        functionparameter_CONFstop(fps);
        fps->md->status &= ~FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF;
    }
    else if (strcmp(command, "confstep") == 0)
    {
        fps->md->conferrcnt = 0;
        fps->md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;
    }
    else if (strcmp(command, "runstart") == 0)
    {
        fps->md->status |= FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN;
    }
    else if (strcmp(command, "runstop") == 0)
    {
        fps->md->status &= ~FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN;
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
                    if (functionparameter_SetParamValue_fromString(&fps, pi, valstr) == 0)
                    {
                        matched = 1;
                    }
                }
            }
        }
        return matched ? 0 : -1;
    }

    int pindex = functionparameter_GetParamIndex(&fps, pname);
    if (pindex < 0)
    {
        char dotname[512];
        snprintf(dotname, sizeof(dotname), ".%s", pname);
        pindex = functionparameter_GetParamIndex(&fps, dotname);
    }

    if (pindex < 0)
    {
        int lc_rc = -1;
        if (try_fps_lifecycle_action(&fps, fpsname, pname, valstr, &lc_rc))
        {
            return lc_rc;
        }
        printf("Error: parameter '%s' not found in FPS '%s'\n", pname, fpsname);
        return -1;
    }

    if (fps.parray[pindex].keywordfull[0] == '\0')
    {
        snprintf(fps.parray[pindex].keywordfull, FUNCTION_PARAMETER_KEYWORD_STRMAXLEN, ".%s",
                 pname);
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
    return (cli_fps_set_param(name, prop, val) == 0) ? 0 : -1;
}

/**
 * @brief Check whether a named FPS exists (and optionally is running).
 *
 * @param name          FPS shared-memory name
 * @param check_running If non-zero, check `FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN`
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
    FPS *cfps = milk_shm_cache_fps(name);
    return (cfps != NULL && (cfps->md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN)) ? 1 : 0;
}
