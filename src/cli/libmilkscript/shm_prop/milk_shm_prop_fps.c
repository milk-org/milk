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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include "CLIcore.h"
#include "fps.h"
#include "fps_GetParamIndex.h"
#include "fps_connect.h"
#include "fps_printparameter_valuestring.h"
#include "milk_shm_prop.h"
#include "milk_shm_prop_internal.h"

/**
 * @brief Read an FPS parameter or enumerate all active parameters (`*`).
 *
 * @param name  FPS shared-memory name
 * @param prop  Parameter key (or "*" for space-separated parameter list)
 * @param out   Output buffer
 * @param outsz Size of @out in bytes
 * @return 0 on success, -1 if FPS or parameter not found
 */
int milk_shm_prop_get_fps(const char *name, const char *prop, char *out, size_t outsz)
{
    FPS fps;
    if (fps_connect(name, &fps, FPSCONNECT_SIMPLE) == -1 || fps.parray == NULL)
    {
        return -1;
    }

    if (strcmp(prop, "*") == 0)
    {
        size_t pos = 0;
        for (int pi = 0; pi < fps.md->NBparamMAX; pi++)
        {
            if (fps.parray[pi].fpflag & FPFLAG_ACTIVE)
            {
                const char *kw = (fps.parray[pi].keywordfull[0] != '\0')
                                     ? fps.parray[pi].keywordfull
                                     : fps.parray[pi].keyword[0];
                if (kw[0] == '.')
                {
                    kw++;
                }
                int n = snprintf(out + pos, outsz - pos, "%s%s", (pos > 0) ? " " : "", kw);
                if (n > 0 && pos + (size_t) n < outsz)
                {
                    pos += (size_t) n;
                }
            }
        }
        fps_disconnect(&fps);
        return 0;
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
        functionparameter_GetParamValueString(&fps.parray[pindex], out, (int) outsz);
        rc = 0;
    }
    fps_disconnect(&fps);
    return rc;
}

/**
 * @brief Write a value string to a named FPS parameter.
 *
 * Connects to the named FPS, locates the parameter, writes the parsed
 * value according to its `FPTYPE_*` bitmask, increments `cnt0`, and
 * disconnects.
 *
 * @param fpsname  FPS shared-memory name
 * @param pname    Parameter key inside FPS
 * @param valstr   Value to write (as string)
 * @return 0 on success, -1 on error
 */
int cli_fps_set_param(const char *fpsname, const char *pname, const char *valstr)
{
    FPS fps;
    int fpsconn = fps_connect(fpsname, &fps, FPSCONNECT_SIMPLE);
    if (fpsconn == -1 || fps.parray == NULL)
    {
        printf("Error: cannot connect to FPS '%s'\n", fpsname);
        return -1;
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
        printf("Error: parameter '%s' not found in FPS '%s'\n", pname, fpsname);
        fps_disconnect(&fps);
        return -1;
    }

    uint32_t ptype = fps.parray[pindex].type;
    if (ptype & FPTYPE_INT32)
    {
        fps.parray[pindex].val.i32[0] = (int32_t) strtol(valstr, NULL, 0);
        fps.parray[pindex].cnt0++;
    }
    else if (ptype & FPTYPE_UINT32)
    {
        fps.parray[pindex].val.ui32[0] = (uint32_t) strtoul(valstr, NULL, 0);
        fps.parray[pindex].cnt0++;
    }
    else if (ptype & FPTYPE_INT64)
    {
        fps.parray[pindex].val.i64[0] = (int64_t) strtoll(valstr, NULL, 0);
        fps.parray[pindex].cnt0++;
    }
    else if (ptype & FPTYPE_UINT64)
    {
        fps.parray[pindex].val.ui64[0] = (uint64_t) strtoull(valstr, NULL, 0);
        fps.parray[pindex].cnt0++;
    }
    else if (ptype & FPTYPE_FLOAT64)
    {
        fps.parray[pindex].val.f64[0] = strtod(valstr, NULL);
        fps.parray[pindex].cnt0++;
    }
    else if (ptype & FPTYPE_FLOAT32)
    {
        fps.parray[pindex].val.f32[0] = (float) strtod(valstr, NULL);
        fps.parray[pindex].cnt0++;
    }
    else if (ptype & FPTYPE_ONOFF)
    {
        if (strcmp(valstr, "ON") == 0 || strcmp(valstr, "on") == 0 || strcmp(valstr, "1") == 0)
        {
            fps.parray[pindex].val.i64[0] = 1;
        }
        else
        {
            fps.parray[pindex].val.i64[0] = 0;
        }
        fps.parray[pindex].cnt0++;
    }
    else if (ptype & FPTYPE_PID)
    {
        fps.parray[pindex].val.pid[0] = (pid_t) strtol(valstr, NULL, 0);
        fps.parray[pindex].cnt0++;
    }
    else if (FPTYPE_IS_STRING(ptype))
    {
        strncpy(fps.parray[pindex].val.string[0], valstr, FUNCTION_PARAMETER_STRMAXLEN - 1);
        fps.parray[pindex].val.string[0][FUNCTION_PARAMETER_STRMAXLEN - 1] = '\0';
        fps.parray[pindex].cnt0++;
    }
    else
    {
        printf("Error: unsupported param type 0x%x for '%s' in FPS '%s'\n", ptype, pname, fpsname);
        fps_disconnect(&fps);
        return -1;
    }

    fps_disconnect(&fps);
    return 0;
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
    const char *shmdir = (dcshmdir[0] != '\0') ? dcshmdir : getenv("MILK_SHM_DIR");
    if (shmdir == NULL || shmdir[0] == '\0')
    {
        shmdir = "/dev/shm";
    }
    char shmpath[STRINGMAXLEN_DIRNAME + 256];
    snprintf(shmpath, sizeof(shmpath), "%s/%s.fps.shm", shmdir, name);
    if (access(shmpath, F_OK) != 0)
    {
        return 0;
    }
    if (!check_running)
    {
        return 1;
    }
    FPS fps;
    if (fps_connect(name, &fps, FPSCONNECT_SIMPLE) == -1 || fps.md == NULL)
    {
        return 0;
    }
    int running = (fps.md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN) ? 1 : 0;
    fps_disconnect(&fps);
    return running;
}
