// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_server.c
 * @brief Milk UNIX domain socket command server and entity handlers.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <pthread.h>
#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#ifdef _OPENMP
#    include <omp.h>
#endif

#include "CLIcore.h"
#include "CLIcore_help.h"
#include "CLIcore_memory.h"
#include "CLIcore_modules.h"
#include "CLIcore_script.h"
#include "CLIcore_setSHMdir.h"
#include "CLIcore_signals.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "ImageStreamIO/ImageStreamIO.h"
#include "cli_calc_parser.h"
#include "fps.h"
#include "fps_GetParamIndex.h"
#include "fps_connect.h"
#include "fps_printparameter_valuestring.h"
#include "fpsseq.h"
#include "milk_proto.h"
#include "milk_server.h"
#include "processinfo.h"

extern PROCESSINFO *processinfo_shm_link(const char *pname, int *fd);
extern errno_t      processinfo_procdirname(char *procdname);
extern int          processinfo_shm_list_create(void);

#ifndef F_SETPIPE_SZ
#    define F_SETPIPE_SZ 1031
#endif

extern errno_t cli_fhelp(void);
extern int     cli_handle_shell_builtins(void);
extern void    handle_did_you_mean(const char *input_cmd);
extern int     cli_intercept_cmd_procctl(const char *p);
extern int     cli_intercept_cmd_procwait(const char *p);
extern int     cli_intercept_cmd_procstat(const char *p);
extern int     cli_intercept_cmd_assert(const char *p);
extern int     cli_intercept_cmd_assigncheck(const char *p);
extern int     cli_intercept_cmd_dpdigits(const char *p);
extern int     cli_intercept_cmd_watch(const char *p);

/* ============================================================
 *  OpenMP fork-safety hook
 * ============================================================ */

static void milk_child_reset_omp(void)
{
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
}

static pthread_once_t g_atfork_once = PTHREAD_ONCE_INIT;

static void register_atfork_handler(void)
{
    pthread_atfork(NULL, NULL, milk_child_reset_omp);
}

/* ============================================================
 *  dcvar <-> bash lossless synchronization cache
 * ============================================================ */

#define DCVAR_CACHE_MAX 4096
static char g_dcvar_cached_str[DCVAR_CACHE_MAX][128];

typedef struct
{
    int    used;
    int    type;
    double f;
    long   l;
    char   s[80];
    char   name[80];
} dcvar_snap_t;

static int is_builtin_const_var(const char *name)
{
    if (name == NULL || name[0] != '_')
    {
        return 0;
    }
    return (strcmp(name, "_PI") == 0 || strcmp(name, "_e") == 0 || strcmp(name, "_gamma") == 0 ||
            strcmp(name, "_c") == 0 || strcmp(name, "_h") == 0 || strcmp(name, "_k") == 0 ||
            strcmp(name, "_pc") == 0 || strcmp(name, "_ly") == 0 || strcmp(name, "_AU") == 0);
}

static void sync_bash_vars_into_dcvar(const uint8_t *buf,
                                      uint32_t       len,
                                      uint32_t       nvars,
                                      uint32_t      *offset_io)
{
    uint32_t off = *offset_io;
    for (uint32_t i = 0; i < nvars && off < len; i++)
    {
        const char *vname = (const char *) (buf + off);
        size_t      nlen  = strnlen(vname, len - off);
        off += (uint32_t) (nlen + 1);
        if (off > len)
        {
            break;
        }
        const char *vval = (const char *) (buf + off);
        size_t      vlen = strnlen(vval, len - off);
        off += (uint32_t) (vlen + 1);

        if (vname[0] == '\0' || is_builtin_const_var(vname))
        {
            continue;
        }
        /* Do not overwrite an existing image with a scalar variable */
        if (image_ID(vname, data.core.image, data.core.NB_MAX_IMAGE) != -1)
        {
            continue;
        }

        long vid = variable_ID(vname);
        if (vid >= 0 && vid < DCVAR_CACHE_MAX && g_dcvar_cached_str[vid][0] != '\0' &&
            strcmp(g_dcvar_cached_str[vid], vval) == 0)
        {
            /* Unchanged in bash since milk last wrote it: preserve exact 64-bit value */
            continue;
        }

        char *endp = NULL;
        long  lval = strtol(vval, &endp, 10);
        if (vval[0] != '\0' && endp != vval && *endp == '\0')
        {
            vid = create_variable_long_ID(vname, lval);
            if (vid >= 0 && vid < DCVAR_CACHE_MAX)
            {
                snprintf(g_dcvar_cached_str[vid], sizeof(g_dcvar_cached_str[vid]), "%s", vval);
            }
        }
        else
        {
            double dval = strtod(vval, &endp);
            if (vval[0] != '\0' && endp != vval && *endp == '\0')
            {
                vid = create_variable_ID(vname, dval);
                if (vid >= 0 && vid < DCVAR_CACHE_MAX)
                {
                    snprintf(g_dcvar_cached_str[vid], sizeof(g_dcvar_cached_str[vid]), "%s", vval);
                }
            }
        }
        cli_var_set(vname, vval);
    }
    *offset_io = off;
}

static dcvar_snap_t *snapshot_dcvar(long *out_nmax)
{
    long nmax = data.core.NB_MAX_VARIABLE;
    if (nmax < 0)
    {
        nmax = 0;
    }
    long          total = nmax + CLI_MAX_VARS;
    dcvar_snap_t *snap  = (dcvar_snap_t *) calloc((size_t) total, sizeof(dcvar_snap_t));
    if (snap == NULL)
    {
        *out_nmax = 0;
        return NULL;
    }
    if (data.core.variable != NULL)
    {
        for (long i = 0; i < nmax; i++)
        {
            snap[i].used = data.core.variable[i].used;
            if (snap[i].used)
            {
                snap[i].type = data.core.variable[i].type;
                snprintf(snap[i].name, sizeof(snap[i].name), "%s", data.core.variable[i].name);
                if (snap[i].type == 0)
                {
                    snap[i].f = data.core.variable[i].value.f;
                }
                else if (snap[i].type == 1)
                {
                    snap[i].l = data.core.variable[i].value.l;
                }
                else
                {
                    snprintf(snap[i].s, sizeof(snap[i].s), "%s", data.core.variable[i].value.s);
                }
            }
        }
    }
    for (int j = 0; j < CLI_MAX_VARS; j++)
    {
        long idx       = nmax + j;
        snap[idx].used = cli_vars[j].used;
        if (cli_vars[j].used)
        {
            snap[idx].type = 2;
            snprintf(snap[idx].name, sizeof(snap[idx].name), "%s", cli_vars[j].name);
            snprintf(snap[idx].s, sizeof(snap[idx].s), "%s", cli_vars[j].val);
        }
    }
    *out_nmax = nmax;
    return snap;
}

static void send_modified_dcvars(int client_fd, const dcvar_snap_t *snap, long snap_nmax)
{
    long nmax = data.core.NB_MAX_VARIABLE;
    if (data.core.variable != NULL)
    {
        for (long i = 0; i < nmax; i++)
        {
            if (!data.core.variable[i].used)
            {
                continue;
            }
            const char *vname = data.core.variable[i].name;
            if (vname[0] == '\0' || is_builtin_const_var(vname))
            {
                continue;
            }

            int changed = 0;
            if (snap == NULL || i >= snap_nmax || !snap[i].used ||
                strcmp(snap[i].name, vname) != 0 || snap[i].type != data.core.variable[i].type)
            {
                changed = 1;
            }
            else if (data.core.variable[i].type == 0 && snap[i].f != data.core.variable[i].value.f)
            {
                changed = 1;
            }
            else if (data.core.variable[i].type == 1 && snap[i].l != data.core.variable[i].value.l)
            {
                changed = 1;
            }
            else if (data.core.variable[i].type == 2 &&
                     strcmp(snap[i].s, data.core.variable[i].value.s) != 0)
            {
                changed = 1;
            }

            if (!changed)
            {
                continue;
            }

            char valstr[256];
            if (data.core.variable[i].type == 1)
            {
                snprintf(valstr, sizeof(valstr), "%ld", data.core.variable[i].value.l);
            }
            else if (data.core.variable[i].type == 0)
            {
                snprintf(valstr, sizeof(valstr), "%.*g", cli_float_digits,
                         data.core.variable[i].value.f);
            }
            else
            {
                snprintf(valstr, sizeof(valstr), "%s", data.core.variable[i].value.s);
            }

            if (i < DCVAR_CACHE_MAX)
            {
                snprintf(g_dcvar_cached_str[i], sizeof(g_dcvar_cached_str[i]), "%s", valstr);
            }

            char   pkt[512];
            size_t n1 = strlen(vname) + 1;
            size_t n2 = strlen(valstr) + 1;
            if (n1 + n2 <= sizeof(pkt))
            {
                memcpy(pkt, vname, n1);
                memcpy(pkt + n1, valstr, n2);
                milk_proto_send(client_fd, MILK_MSG_SETVAR, 0, pkt, (uint32_t) (n1 + n2));
            }
        }
    }

    /* Also send string variables assigned via cli_var_set (e.g. _su = "hello") */
    for (int j = 0; j < CLI_MAX_VARS; j++)
    {
        if (!cli_vars[j].used)
        {
            continue;
        }
        const char *vname = cli_vars[j].name;
        if (vname[0] == '\0' || is_builtin_const_var(vname) || strncmp(vname, "_tmpstr", 7) == 0 ||
            strncmp(vname, "_tmpcalc", 8) == 0 || variable_ID(vname) != -1)
        {
            continue;
        }
        int changed = 0;
        if (snap == NULL)
        {
            changed = 1;
        }
        else
        {
            long idx = snap_nmax + j;
            if (!snap[idx].used || strcmp(snap[idx].name, vname) != 0 ||
                strcmp(snap[idx].s, cli_vars[j].val) != 0)
            {
                changed = 1;
            }
        }
        if (!changed)
        {
            continue;
        }
        char   pkt[600];
        size_t n1 = strlen(vname) + 1;
        size_t n2 = strlen(cli_vars[j].val) + 1;
        if (n1 + n2 <= sizeof(pkt))
        {
            memcpy(pkt, vname, n1);
            memcpy(pkt + n1, cli_vars[j].val, n2);
            milk_proto_send(client_fd, MILK_MSG_SETVAR, 0, pkt, (uint32_t) (n1 + n2));
        }
    }
}

/* ============================================================
 *  Entity Get / Set / Exists (Streams, Keywords, FPS, Proc, Seq)
 * ============================================================ */

static IMAGE *find_or_open_stream_or_image(const char *sname, IMAGE *tmp_img, int *need_close)
{
    *need_close = 0;
    if (data.core.image != NULL)
    {
        for (long i = 0; i < data.core.NB_MAX_IMAGE; i++)
        {
            if (data.core.image[i].used == 1 && strcmp(data.core.image[i].name, sname) == 0)
            {
                return &data.core.image[i];
            }
        }
    }

    char        shmpath[STRINGMAXLEN_DIRNAME + 256];
    const char *shmdir = (dcshmdir[0] != '\0') ? dcshmdir : getenv("MILK_SHM_DIR");
    if (shmdir == NULL || shmdir[0] == '\0')
    {
        shmdir = "/dev/shm";
    }
    snprintf(shmpath, sizeof(shmpath), "%s/%s.im.shm", shmdir, sname);
    if (access(shmpath, F_OK) != 0)
    {
        return NULL;
    }

    memset(tmp_img, 0, sizeof(*tmp_img));
    if (ImageStreamIO_openIm(tmp_img, sname) == IMAGESTREAMIO_SUCCESS && tmp_img->md != NULL)
    {
        *need_close = 1;
        return tmp_img;
    }
    return NULL;
}

int milk_entity_get(const char *spec, char *out, size_t outsz)
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
    char *dot2 = strchr(dot1 + 1, '.');
    if (dot2 == NULL)
    {
        /* 1-dot fallback: fpsname.param */
        *dot1 = '\0';
        FPS fps;
        if (fps_connect(buf, &fps, FPSCONNECT_SIMPLE) == -1 || fps.parray == NULL)
        {
            return -1;
        }
        int pindex = functionparameter_GetParamIndex(&fps, dot1 + 1);
        if (pindex < 0)
        {
            char dname[256];
            snprintf(dname, sizeof(dname), ".%s", dot1 + 1);
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

    *dot1            = '\0';
    *dot2            = '\0';
    const char *nsp  = buf;
    const char *name = dot1 + 1;
    const char *prop = dot2 + 1;

    /* 1. Stream / Local Image */
    if (strcmp(nsp, "s") == 0 || strcmp(nsp, "stream") == 0)
    {
        IMAGE  tmp_img;
        int    need_close = 0;
        IMAGE *im         = find_or_open_stream_or_image(name, &tmp_img, &need_close);
        if (im == NULL || im->md == NULL)
        {
            return -1;
        }

        int found = 0;
        if (strcmp(prop, "xsize") == 0 || strcmp(prop, "size0") == 0)
        {
            snprintf(out, outsz, "%u", im->md->size[0]);
            found = 1;
        }
        else if (strcmp(prop, "ysize") == 0 || strcmp(prop, "size1") == 0)
        {
            snprintf(out, outsz, "%u",
                     (im->md->naxis > 1 && im->md->size[1] > 0) ? im->md->size[1] : 1U);
            found = 1;
        }
        else if (strcmp(prop, "zsize") == 0 || strcmp(prop, "size2") == 0)
        {
            snprintf(out, outsz, "%u",
                     (im->md->naxis > 2 && im->md->size[2] > 0) ? im->md->size[2] : 1U);
            found = 1;
        }
        else if (strcmp(prop, "naxis") == 0)
        {
            snprintf(out, outsz, "%u", (unsigned) im->md->naxis);
            found = 1;
        }
        else if (strcmp(prop, "type") == 0 || strcmp(prop, "typeid") == 0)
        {
            snprintf(out, outsz, "%u", (unsigned) im->md->datatype);
            found = 1;
        }
        else if (strcmp(prop, "typename") == 0 || strcmp(prop, "dtype") == 0)
        {
            const char *tn = ImageStreamIO_typename(im->md->datatype);
            snprintf(out, outsz, "%s", tn ? tn : "UNKNOWN");
            found = 1;
        }
        else if (strcmp(prop, "cnt0") == 0)
        {
            snprintf(out, outsz, "%lu", (unsigned long) im->md->cnt0);
            found = 1;
        }
        else if (strcmp(prop, "cnt1") == 0)
        {
            snprintf(out, outsz, "%lu", (unsigned long) im->md->cnt1);
            found = 1;
        }
        else if (strcmp(prop, "sem") == 0)
        {
            snprintf(out, outsz, "%u", (unsigned) im->md->sem);
            found = 1;
        }
        else if (strcmp(prop, "pid") == 0)
        {
            snprintf(out, outsz, "%d", (int) im->md->creatorPID);
            found = 1;
        }
        else if (strcmp(prop, "ownerPID") == 0)
        {
            snprintf(out, outsz, "%d", (int) im->md->ownerPID);
            found = 1;
        }
        else if (strcmp(prop, "nelement") == 0 || strcmp(prop, "nelem") == 0)
        {
            snprintf(out, outsz, "%lu", (unsigned long) im->md->nelement);
            found = 1;
        }
        else if (strcmp(prop, "shared") == 0)
        {
            snprintf(out, outsz, "%u", (unsigned) im->md->shared);
            found = 1;
        }
        else if (strcmp(prop, "NBkw") == 0)
        {
            snprintf(out, outsz, "%u", (unsigned) im->md->NBkw);
            found = 1;
        }
        else if (strncmp(prop, "kw.", 3) == 0 && im->kw != NULL)
        {
            const char *kname = prop + 3;
            if (strcmp(kname, "*") == 0)
            {
                size_t pos = 0;
                for (uint16_t k = 0; k < im->md->NBkw; k++)
                {
                    if (im->kw[k].type != 'N' && im->kw[k].type != '\0' &&
                        im->kw[k].name[0] != '\0')
                    {
                        int n = snprintf(out + pos, outsz - pos, "%s%s", (pos > 0) ? " " : "",
                                         im->kw[k].name);
                        if (n > 0 && pos + (size_t) n < outsz)
                        {
                            pos += (size_t) n;
                        }
                    }
                }
                found = 1;
            }
            else
            {
                for (uint16_t k = 0; k < im->md->NBkw; k++)
                {
                    if (im->kw[k].type != 'N' && im->kw[k].type != '\0' &&
                        strcmp(im->kw[k].name, kname) == 0)
                    {
                        if (im->kw[k].type == 'L')
                        {
                            snprintf(out, outsz, "%ld", (long) im->kw[k].value.numl);
                        }
                        else if (im->kw[k].type == 'D')
                        {
                            snprintf(out, outsz, "%.17g", im->kw[k].value.numf);
                        }
                        else
                        {
                            snprintf(out, outsz, "%s", im->kw[k].value.valstr);
                        }
                        found = 1;
                        break;
                    }
                }
            }
        }

        if (need_close)
        {
            ImageStreamIO_closeIm(&tmp_img);
        }
        return found ? 0 : -1;
    }

    /* 2. FPS */
    if (strcmp(nsp, "f") == 0 || strcmp(nsp, "fps") == 0)
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
                    int n = snprintf(out + pos, outsz - pos, "%s%s", (pos > 0) ? " " : "",
                                     fps.parray[pi].keyword[0]);
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

    /* 3. Processinfo */
    if (strcmp(nsp, "p") == 0 || strcmp(nsp, "proc") == 0 || strcmp(nsp, "procinfo") == 0)
    {
        if (pinfolist == NULL)
        {
            processinfo_shm_list_create();
        }
        if (pinfolist == NULL)
        {
            return -1;
        }
        pid_t found_pid = 0;
        for (int pi = 0; pi < PROCESSINFOLISTSIZE; pi++)
        {
            if (pinfolist->active[pi] && strcmp(pinfolist->pnamearray[pi], name) == 0)
            {
                found_pid = pinfolist->PIDarray[pi];
                break;
            }
        }
        if (found_pid <= 0)
        {
            return -1;
        }
        char pfname[STRINGMAXLEN_FULLFILENAME];
        char procdname[STRINGMAXLEN_DIRNAME];
        processinfo_procdirname(procdname);
        snprintf(pfname, sizeof(pfname), "%s/proc.%d.shm", procdname, (int) found_pid);
        int          pfd = -1;
        PROCESSINFO *pi  = processinfo_shm_link(pfname, &pfd);
        if (pi == MAP_FAILED || pi == NULL)
        {
            if (pfd >= 0)
            {
                close(pfd);
            }
            return -1;
        }
        int found = 1;
        if (strcmp(prop, "pid") == 0)
        {
            snprintf(out, outsz, "%d", (int) pi->PID);
        }
        else if (strcmp(prop, "loopstat") == 0)
        {
            snprintf(out, outsz, "%d", pi->loopstat);
        }
        else if (strcmp(prop, "loopcnt") == 0)
        {
            snprintf(out, outsz, "%ld", pi->loopcnt);
        }
        else if (strcmp(prop, "loopfreq") == 0)
        {
            double hz = (pi->dtmedian_iter_ns > 0) ? (1.0e9 / (double) pi->dtmedian_iter_ns) : 0.0;
            snprintf(out, outsz, "%.1f", hz);
        }
        else if (strcmp(prop, "exectime") == 0)
        {
            snprintf(out, outsz, "%.1f", (double) pi->dtmedian_exec_ns / 1000.0);
        }
        else if (strcmp(prop, "rtprio") == 0)
        {
            snprintf(out, outsz, "%d", pi->RT_priority);
        }
        else if (strcmp(prop, "ctrlval") == 0)
        {
            snprintf(out, outsz, "%d", pi->CTRLval);
        }
        else if (strcmp(prop, "trigmode") == 0)
        {
            snprintf(out, outsz, "%d", pi->triggermode);
        }
        else if (strcmp(prop, "statusmsg") == 0)
        {
            snprintf(out, outsz, "%s", pi->statusmsg);
        }
        else if (strcmp(prop, "tmux") == 0)
        {
            snprintf(out, outsz, "%s", pi->tmuxname);
        }
        else if (strcmp(prop, "description") == 0)
        {
            snprintf(out, outsz, "%s", pi->description);
        }
        else if (strcmp(prop, "missedframes") == 0)
        {
            snprintf(out, outsz, "%lu", (unsigned long) pi->triggermissedframe_cumul);
        }
        else
        {
            found = 0;
        }
        munmap(pi, sizeof(PROCESSINFO));
        close(pfd);
        return found ? 0 : -1;
    }

    /* 4. Sequencer */
    if (strcmp(nsp, "q") == 0 || strcmp(nsp, "seq") == 0 || strcmp(nsp, "fpsseq") == 0)
    {
        MILKSEQ_STATE *seqst = milkseq_connect(name);
        if (seqst == NULL)
        {
            return -1;
        }
        int found = 1;
        if (strcmp(prop, "status") == 0)
        {
            const char *s  = "IDLE";
            uint32_t    st = seqst->status;
            if (st & MILKSEQ_STATUS_ERROR)
            {
                s = "ERROR";
            }
            else if (st & MILKSEQ_STATUS_STOPPING)
            {
                s = "STOPPING";
            }
            else if (st & MILKSEQ_STATUS_RUNNING)
            {
                s = "RUNNING";
            }
            snprintf(out, outsz, "%s", s);
        }
        else if (strcmp(prop, "tasks") == 0)
        {
            snprintf(out, outsz, "%u", seqst->NBtasks_active);
        }
        else if (strcmp(prop, "errors") == 0)
        {
            snprintf(out, outsz, "%u", seqst->error_count);
        }
        else if (strcmp(prop, "pid") == 0)
        {
            snprintf(out, outsz, "%d", (int) seqst->pid);
        }
        else if (strcmp(prop, "completed") == 0)
        {
            snprintf(out, outsz, "%u", seqst->NBtasks_completed);
        }
        else
        {
            found = 0;
        }
        milkseq_disconnect(seqst);
        return found ? 0 : -1;
    }

    return -1;
}

int milk_entity_set(const char *spec, const char *val, const char *comment)
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
        return -1;
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
    char *dot2 = strchr(dot1 + 1, '.');
    if (dot2 == NULL)
    {
        *dot1 = '\0';
        return (cli_fps_set_param(specbuf, dot1 + 1, val) == 0) ? 0 : -1;
    }

    *dot1            = '\0';
    *dot2            = '\0';
    const char *nsp  = specbuf;
    const char *name = dot1 + 1;
    const char *prop = dot2 + 1;

    if (strcmp(nsp, "f") == 0 || strcmp(nsp, "fps") == 0)
    {
        return (cli_fps_set_param(name, prop, val) == 0) ? 0 : -1;
    }

    if (strcmp(nsp, "s") == 0 || strcmp(nsp, "stream") == 0)
    {
        if (strncmp(prop, "kw.", 3) != 0)
        {
            fprintf(stderr, "ERROR: only s.<name>.kw.<KEY> is writable on streams\n");
            return -1;
        }
        const char *kname = prop + 3;
        if (kname[0] == '\0')
        {
            return -1;
        }

        IMAGE  tmp_img;
        int    need_close = 0;
        IMAGE *im         = find_or_open_stream_or_image(name, &tmp_img, &need_close);
        if (im == NULL || im->md == NULL || im->kw == NULL)
        {
            fprintf(stderr, "ERROR: stream '%s' not found or has no keyword table\n", name);
            return -1;
        }

        int slot = -1;
        for (uint16_t k = 0; k < im->md->NBkw; k++)
        {
            if (im->kw[k].type != 'N' && im->kw[k].type != '\0' &&
                strcmp(im->kw[k].name, kname) == 0)
            {
                slot = (int) k;
                break;
            }
        }
        if (slot == -1)
        {
            for (uint16_t k = 0; k < im->md->NBkw; k++)
            {
                if (im->kw[k].type == 'N' || im->kw[k].type == '\0' || im->kw[k].name[0] == '\0')
                {
                    slot = (int) k;
                    break;
                }
            }
        }
        if (slot == -1)
        {
            if (need_close)
            {
                ImageStreamIO_closeIm(&tmp_img);
            }
            fprintf(stderr, "ERROR: keyword table full on stream '%s'\n", name);
            return -1;
        }

        snprintf(im->kw[slot].name, KEYWORD_MAX_STRING, "%s", kname);
        char     *endp = NULL;
        long long lval = strtoll(val, &endp, 10);
        if (val[0] != '\0' && endp != val && *endp == '\0')
        {
            im->kw[slot].type       = 'L';
            im->kw[slot].value.numl = (int64_t) lval;
        }
        else
        {
            double dval = strtod(val, &endp);
            if (val[0] != '\0' && endp != val && *endp == '\0')
            {
                im->kw[slot].type       = 'D';
                im->kw[slot].value.numf = dval;
            }
            else
            {
                im->kw[slot].type = 'S';
                snprintf(im->kw[slot].value.valstr, KEYWORD_MAX_STRING, "%s", val);
            }
        }
        if (comment != NULL && comment[0] != '\0')
        {
            snprintf(im->kw[slot].comment, KEYWORD_MAX_COMMENT, "%s", comment);
        }
        im->kw[slot].cnt++;

        if (need_close)
        {
            ImageStreamIO_closeIm(&tmp_img);
        }
        return 0;
    }

    if (strcmp(nsp, "p") == 0 || strcmp(nsp, "proc") == 0 || strcmp(nsp, "procinfo") == 0)
    {
        if (strcmp(prop, "ctrlval") != 0)
        {
            return -1;
        }
        if (pinfolist == NULL)
        {
            processinfo_shm_list_create();
        }
        if (pinfolist == NULL)
        {
            return -1;
        }
        int ctrlval_int = atoi(val);
        if (strcmp(val, "run") == 0)
        {
            ctrlval_int = PROCESSINFO_CTRLVAL_RUN;
        }
        else if (strcmp(val, "pause") == 0)
        {
            ctrlval_int = PROCESSINFO_CTRLVAL_PAUSE;
        }
        else if (strcmp(val, "step") == 0)
        {
            ctrlval_int = PROCESSINFO_CTRLVAL_INCR;
        }
        else if (strcmp(val, "stop") == 0 || strcmp(val, "exit") == 0)
        {
            ctrlval_int = PROCESSINFO_CTRLVAL_EXIT;
        }

        pid_t found_pid = 0;
        for (int pidx = 0; pidx < PROCESSINFOLISTSIZE; pidx++)
        {
            if (pinfolist->active[pidx] && strcmp(pinfolist->pnamearray[pidx], name) == 0)
            {
                found_pid = pinfolist->PIDarray[pidx];
                break;
            }
        }
        if (found_pid <= 0)
        {
            return -1;
        }
        char pfname[STRINGMAXLEN_FULLFILENAME];
        char procdname[STRINGMAXLEN_DIRNAME];
        processinfo_procdirname(procdname);
        snprintf(pfname, sizeof(pfname), "%s/proc.%d.shm", procdname, (int) found_pid);
        int          pfd    = -1;
        PROCESSINFO *pi_shm = processinfo_shm_link(pfname, &pfd);
        if (pi_shm != MAP_FAILED && pi_shm != NULL)
        {
            pi_shm->CTRLval = ctrlval_int;
            munmap(pi_shm, sizeof(PROCESSINFO));
            close(pfd);
            return 0;
        }
        if (pfd >= 0)
        {
            close(pfd);
        }
        return -1;
    }

    return -1;
}

int milk_entity_exists(const char *spec, int check_running)
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

    const char *shmdir = (dcshmdir[0] != '\0') ? dcshmdir : getenv("MILK_SHM_DIR");
    if (shmdir == NULL || shmdir[0] == '\0')
    {
        shmdir = "/dev/shm";
    }

    if (strcmp(nsp, "s") == 0 || strcmp(nsp, "stream") == 0)
    {
        if (data.core.image != NULL)
        {
            for (long i = 0; i < data.core.NB_MAX_IMAGE; i++)
            {
                if (data.core.image[i].used == 1 && strcmp(data.core.image[i].name, name) == 0)
                {
                    return 1;
                }
            }
        }
        char shmpath[STRINGMAXLEN_DIRNAME + 256];
        snprintf(shmpath, sizeof(shmpath), "%s/%s.im.shm", shmdir, name);
        return (access(shmpath, F_OK) == 0) ? 1 : 0;
    }

    if (strcmp(nsp, "f") == 0 || strcmp(nsp, "fps") == 0)
    {
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

    if (strcmp(nsp, "p") == 0 || strcmp(nsp, "proc") == 0 || strcmp(nsp, "procinfo") == 0)
    {
        if (pinfolist == NULL)
        {
            processinfo_shm_list_create();
        }
        if (pinfolist == NULL)
        {
            return 0;
        }
        pid_t found_pid = 0;
        for (int pidx = 0; pidx < PROCESSINFOLISTSIZE; pidx++)
        {
            if (pinfolist->active[pidx] && strcmp(pinfolist->pnamearray[pidx], name) == 0)
            {
                found_pid = pinfolist->PIDarray[pidx];
                break;
            }
        }
        if (found_pid <= 0 || kill(found_pid, 0) != 0)
        {
            return 0;
        }
        if (!check_running)
        {
            return 1;
        }
        char pfname[STRINGMAXLEN_FULLFILENAME];
        char procdname[STRINGMAXLEN_DIRNAME];
        processinfo_procdirname(procdname);
        snprintf(pfname, sizeof(pfname), "%s/proc.%d.shm", procdname, (int) found_pid);
        int          pfd    = -1;
        PROCESSINFO *pi_shm = processinfo_shm_link(pfname, &pfd);
        if (pi_shm == MAP_FAILED || pi_shm == NULL)
        {
            if (pfd >= 0)
            {
                close(pfd);
            }
            return 0;
        }
        int running = (pi_shm->loopstat == 1 && pi_shm->CTRLval == 0) ? 1 : 0;
        munmap(pi_shm, sizeof(PROCESSINFO));
        close(pfd);
        return running;
    }

    if (strcmp(nsp, "q") == 0 || strcmp(nsp, "seq") == 0 || strcmp(nsp, "fpsseq") == 0)
    {
        MILKSEQ_STATE *seqst = milkseq_connect(name);
        if (seqst == NULL)
        {
            return 0;
        }
        int running = check_running ? ((seqst->status & MILKSEQ_STATUS_RUNNING) ? 1 : 0) : 1;
        milkseq_disconnect(seqst);
        return running;
    }

    return 0;
}

/* ============================================================
 *  @waitfor and @grab implementations
 * ============================================================ */

static int check_client_cancelled(int client_fd)
{
    if (client_fd < 0)
    {
        return 0;
    }
    struct pollfd pfd;
    pfd.fd      = client_fd;
    pfd.events  = POLLIN | POLLHUP | POLLERR;
    pfd.revents = 0;
    if (poll(&pfd, 1, 0) > 0)
    {
        if (pfd.revents & (POLLHUP | POLLERR))
        {
            return 1;
        }
        if (pfd.revents & POLLIN)
        {
            milk_frame_hdr_t hdr;
            uint8_t         *pl = NULL;
            if (milk_proto_recv(client_fd, &hdr, &pl, NULL) == 0)
            {
                free(pl);
                if (hdr.type == MILK_MSG_CANCEL)
                {
                    return 1;
                }
            }
            else
            {
                return 1;
            }
        }
    }
    return 0;
}

static double monotonic_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1.0e-9 * (double) ts.tv_nsec;
}

static void expand_entities_in_expr(const char *in, char *out, size_t outsz)
{
    if (outsz == 0)
    {
        return;
    }
    char stage1[8192];
    snprintf(stage1, sizeof(stage1), "%s", (in != NULL) ? in : "");
    cli_expand_fpsvar(stage1, (int) sizeof(stage1));

    size_t ip    = 0;
    size_t op    = 0;
    size_t len   = strlen(stage1);
    int    in_sq = 0;
    int    in_dq = 0;

    while (ip < len && op + 1 < outsz)
    {
        char c = stage1[ip];
        if (c == '\'' && !in_dq)
        {
            in_sq     = !in_sq;
            out[op++] = stage1[ip++];
            continue;
        }
        if (c == '"' && !in_sq)
        {
            in_dq     = !in_dq;
            out[op++] = stage1[ip++];
            continue;
        }
        if (in_sq || in_dq)
        {
            out[op++] = stage1[ip++];
            continue;
        }

        if ((isalpha((unsigned char) c) || c == '_') &&
            (ip == 0 || (!isalnum((unsigned char) stage1[ip - 1]) && stage1[ip - 1] != '_' &&
                         stage1[ip - 1] != '.' && stage1[ip - 1] != '@')))
        {
            size_t end   = ip;
            int    ndots = 0;
            while (end < len && (isalnum((unsigned char) stage1[end]) || stage1[end] == '_' ||
                                 stage1[end] == '.' || stage1[end] == '-'))
            {
                if (stage1[end] == '.')
                {
                    ndots++;
                }
                end++;
            }
            size_t tlen = end - ip;
            if (ndots >= 2 && tlen < 256)
            {
                char tok[256];
                memcpy(tok, stage1 + ip, tlen);
                tok[tlen]         = '\0';
                const char *dot1  = strchr(tok, '.');
                size_t      nslen = (size_t) (dot1 - tok);
                if ((nslen == 1 &&
                     (tok[0] == 's' || tok[0] == 'f' || tok[0] == 'p' || tok[0] == 'q')) ||
                    (nslen == 6 && strncmp(tok, "stream", 6) == 0) ||
                    (nslen == 3 && (strncmp(tok, "fps", 3) == 0 || strncmp(tok, "seq", 3) == 0)) ||
                    (nslen == 4 && strncmp(tok, "proc", 4) == 0) ||
                    (nslen == 8 && strncmp(tok, "procinfo", 8) == 0))
                {
                    char val[512];
                    val[0] = '\0';
                    if (milk_entity_get(tok, val, sizeof(val)) == 0 && val[0] != '\0')
                    {
                        char  *endp = NULL;
                        double dv   = strtod(val, &endp);
                        (void) dv;
                        if (endp != val && *endp == '\0')
                        {
                            size_t vl = strlen(val);
                            if (op + vl < outsz)
                            {
                                memcpy(out + op, val, vl);
                                op += vl;
                            }
                        }
                        else
                        {
                            int n = snprintf(out + op, outsz - op, "\"%s\"", val);
                            if (n > 0 && op + (size_t) n < outsz)
                            {
                                op += (size_t) n;
                            }
                        }
                        ip = end;
                        continue;
                    }
                }
            }
        }
        out[op++] = stage1[ip++];
    }
    out[op] = '\0';
}

static int milk_server_waitfor(int         client_fd,
                               const char *target,
                               const char *arg2,
                               double      timeout_sec)
{
    while (*target == '@')
    {
        target++;
    }
    double t_start  = monotonic_sec();
    int    has_timo = (timeout_sec > 0.0);

    /* Case 1a: General boolean expression condition (contains <, >, ==, !=, &&, ||) */
    if (strchr(target, '<') != NULL || strchr(target, '>') != NULL ||
        strstr(target, "==") != NULL || strstr(target, "!=") != NULL ||
        strstr(target, "&&") != NULL || strstr(target, "||") != NULL)
    {
        while (1)
        {
            if (check_client_cancelled(client_fd))
            {
                return 130;
            }
            char expanded[8192];
            expand_entities_in_expr(target, expanded, sizeof(expanded));
            int eval_rc = 2;
            cli_calc_eval_bash(expanded, 1, &eval_rc);
            if (eval_rc == 0)
            {
                return 0;
            }
            if (has_timo && (monotonic_sec() - t_start) >= timeout_sec)
            {
                return 124;
            }
            struct timespec slp = { 0, 10000000L }; /* 10 ms */
            nanosleep(&slp, NULL);
        }
    }

    /* Case 1b: Condition wait f.name.param=val or s.name.prop=val */
    const char *eq = strchr(target, '=');
    if (eq != NULL)
    {
        char   lhs[256];
        size_t lhslen = (size_t) (eq - target);
        if (lhslen >= sizeof(lhs))
        {
            return 1;
        }
        memcpy(lhs, target, lhslen);
        lhs[lhslen]        = '\0';
        const char *expect = eq + 1;

        while (1)
        {
            if (check_client_cancelled(client_fd))
            {
                return 130;
            }
            char val[256];
            if (milk_entity_get(lhs, val, sizeof(val)) == 0)
            {
                if (strcmp(val, expect) == 0)
                {
                    return 0;
                }
                char  *e1 = NULL;
                char  *e2 = NULL;
                double d1 = strtod(val, &e1);
                double d2 = strtod(expect, &e2);
                if (val[0] != '\0' && expect[0] != '\0' && *e1 == '\0' && *e2 == '\0' &&
                    fabs(d1 - d2) <= 1.0e-9 * (1.0 + fabs(d2)))
                {
                    return 0;
                }
            }
            if (has_timo && (monotonic_sec() - t_start) >= timeout_sec)
            {
                return 124;
            }
            struct timespec slp = { 0, 10000000L }; /* 10 ms */
            nanosleep(&slp, NULL);
        }
    }

    /* Case 2: Stream frame wait: s.cam [nframes] */
    const char *sname = target;
    if (strncmp(sname, "s.", 2) == 0)
    {
        sname += 2;
    }
    else if (strncmp(sname, "stream.", 7) == 0)
    {
        sname += 7;
    }

    long nframes = (arg2 != NULL && arg2[0] != '\0') ? strtol(arg2, NULL, 10) : 1;
    if (nframes < 1)
    {
        nframes = 1;
    }

    IMAGE img;
    memset(&img, 0, sizeof(img));
    while (ImageStreamIO_openIm(&img, sname) != IMAGESTREAMIO_SUCCESS || img.md == NULL)
    {
        if (check_client_cancelled(client_fd))
        {
            return 130;
        }
        if (has_timo && (monotonic_sec() - t_start) >= timeout_sec)
        {
            return 124;
        }
        struct timespec slp = { 0, 10000000L };
        nanosleep(&slp, NULL);
    }

    uint64_t start_cnt0 = img.md->cnt0;
    uint64_t target_cnt = start_cnt0 + (uint64_t) nframes;
    int      semindex   = ImageStreamIO_getsemwaitindex(&img, -1);

    /* Drain stale semaphore posts */
    if (semindex >= 0 && semindex < img.md->sem)
    {
        while (sem_trywait(img.semptr[semindex]) == 0)
        {
        }
    }

    int rc = 0;
    while (img.md->cnt0 < target_cnt)
    {
        if (check_client_cancelled(client_fd))
        {
            rc = 130;
            break;
        }
        if (has_timo && (monotonic_sec() - t_start) >= timeout_sec)
        {
            rc = 124;
            break;
        }
        if (semindex >= 0 && semindex < img.md->sem)
        {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += 20000000L; /* 20 ms step */
            if (ts.tv_nsec >= 1000000000L)
            {
                ts.tv_sec += 1;
                ts.tv_nsec -= 1000000000L;
            }
            sem_timedwait(img.semptr[semindex], &ts);
        }
        else
        {
            struct timespec slp = { 0, 5000000L };
            nanosleep(&slp, NULL);
        }
    }

    if (semindex >= 0 && semindex < img.md->sem)
    {
        img.semReadPID[semindex] = 0;
    }
    ImageStreamIO_closeIm(&img);
    return rc;
}

static int milk_server_grab(int         client_fd,
                            const char *sname_raw,
                            long        nframes,
                            const char *outname,
                            double      timeout_sec)
{
    if (sname_raw == NULL || outname == NULL || nframes < 1)
    {
        return 1;
    }
    const char *sname = sname_raw;
    while (*sname == '@')
    {
        sname++;
    }
    if (strncmp(sname, "s.", 2) == 0)
    {
        sname += 2;
    }
    else if (strncmp(sname, "stream.", 7) == 0)
    {
        sname += 7;
    }

    IMAGE  tmp_in;
    int    need_close = 0;
    IMAGE *in_im      = find_or_open_stream_or_image(sname, &tmp_in, &need_close);
    if (in_im == NULL || in_im->md == NULL)
    {
        fprintf(stderr, "ERROR: @grab: stream '%s' not found\n", sname);
        return 1;
    }

    uint32_t nx      = in_im->md->size[0];
    uint32_t ny      = (in_im->md->naxis >= 2 && in_im->md->size[1] > 0) ? in_im->md->size[1] : 1U;
    uint64_t npix2d  = (uint64_t) nx * ny;
    uint8_t  in_dt   = in_im->md->datatype;
    size_t   in_elsz = ImageStreamIO_typesize(in_dt);

    IMGID out_img = imgid_make_from_name(outname);
    if (out_img.stream_parse_error)
    {
        imgid_free(&out_img);
        if (need_close)
        {
            ImageStreamIO_closeIm(&tmp_in);
        }
        return 1;
    }
    out_img.mdt->naxis    = 3;
    out_img.mdt->size[0]  = nx;
    out_img.mdt->size[1]  = ny;
    out_img.mdt->size[2]  = (uint32_t) nframes;
    out_img.mdt->datatype = (out_img.stream_datatype != 0) ? out_img.stream_datatype : in_dt;
    out_img.mdt->shared   = (out_img.stream_loc == 2) ? 0 : 1;
    out_img.mdt->NBkw     = 10;
    out_img.mdt->CBsize   = 0;

    if (create_image_ID_IMGID(&out_img) != RETURN_SUCCESS || out_img.ID == -1)
    {
        imgid_free(&out_img);
        if (need_close)
        {
            ImageStreamIO_closeIm(&tmp_in);
        }
        return 1;
    }

    IMAGE  *cube     = &data.core.image[out_img.ID];
    uint8_t out_dt   = cube->md->datatype;
    size_t  out_elsz = ImageStreamIO_typesize(out_dt);
    imgid_free(&out_img);

    int semindex = -1;
    if (in_im->md->shared && in_im->md->sem > 0)
    {
        semindex = ImageStreamIO_getsemwaitindex(in_im, -1);
        if (semindex >= 0 && semindex < in_im->md->sem)
        {
            while (sem_trywait(in_im->semptr[semindex]) == 0)
            {
            }
        }
    }

    double   t_start  = monotonic_sec();
    int      has_timo = (timeout_sec > 0.0);
    uint64_t last_cnt = in_im->md->cnt0;
    int      rc       = 0;

    cube->md->write = 1;
    for (long f = 0; f < nframes; f++)
    {
        /* Wait for next frame if shared stream */
        if (in_im->md->shared)
        {
            while (in_im->md->cnt0 == last_cnt)
            {
                if (check_client_cancelled(client_fd))
                {
                    rc = 130;
                    break;
                }
                if (has_timo && (monotonic_sec() - t_start) >= timeout_sec)
                {
                    rc = 124;
                    break;
                }
                if (semindex >= 0 && semindex < in_im->md->sem)
                {
                    struct timespec ts;
                    clock_gettime(CLOCK_REALTIME, &ts);
                    ts.tv_nsec += 20000000L;
                    if (ts.tv_nsec >= 1000000000L)
                    {
                        ts.tv_sec += 1;
                        ts.tv_nsec -= 1000000000L;
                    }
                    sem_timedwait(in_im->semptr[semindex], &ts);
                }
                else
                {
                    struct timespec slp = { 0, 2000000L };
                    nanosleep(&slp, NULL);
                }
            }
            if (rc != 0)
            {
                break;
            }
            last_cnt = in_im->md->cnt0;
        }

        /* Copy frame f into cube slice f */
        if (in_dt == out_dt)
        {
            memcpy(((uint8_t *) cube->array.raw) + (size_t) f * npix2d * out_elsz, in_im->array.raw,
                   npix2d * in_elsz);
        }
        else
        {
            uint64_t base = (uint64_t) f * npix2d;
            for (uint64_t p = 0; p < npix2d; p++)
            {
                double v = 0.0;
                if (in_dt == _DATATYPE_FLOAT)
                {
                    v = (double) in_im->array.F[p];
                }
                else if (in_dt == _DATATYPE_DOUBLE)
                {
                    v = in_im->array.D[p];
                }
                else if (in_dt == _DATATYPE_UINT16)
                {
                    v = (double) in_im->array.UI16[p];
                }
                else if (in_dt == _DATATYPE_INT16)
                {
                    v = (double) in_im->array.SI16[p];
                }
                else if (in_dt == _DATATYPE_UINT32)
                {
                    v = (double) in_im->array.UI32[p];
                }
                else if (in_dt == _DATATYPE_INT32)
                {
                    v = (double) in_im->array.SI32[p];
                }
                else if (in_dt == _DATATYPE_UINT8)
                {
                    v = (double) in_im->array.UI8[p];
                }

                if (out_dt == _DATATYPE_FLOAT)
                {
                    cube->array.F[base + p] = (float) v;
                }
                else if (out_dt == _DATATYPE_DOUBLE)
                {
                    cube->array.D[base + p] = v;
                }
                else if (out_dt == _DATATYPE_UINT16)
                {
                    cube->array.UI16[base + p] = (uint16_t) v;
                }
                else if (out_dt == _DATATYPE_INT16)
                {
                    cube->array.SI16[base + p] = (int16_t) v;
                }
            }
        }
    }

    processinfo_update_output_stream(NULL, cube, NULL);

    if (semindex >= 0 && semindex < in_im->md->sem)
    {
        in_im->semReadPID[semindex] = 0;
    }
    if (need_close)
    {
        ImageStreamIO_closeIm(&tmp_in);
    }
    return rc;
}

/* ============================================================
 *  Command lookup & dispatch (without wordexp / shell passes)
 * ============================================================ */

/**
 * @brief Resolve a command name (canonical "ns.cmd" or unambiguous short "cmd").
 *
 * @param name         Command name without leading sigil
 * @param out_ambig    Set to 1 if multiple modules define the same short name
 * @return Command index in data.cmd[], or -1 if not found / ambiguous
 */
static int resolve_milk_cmd_index(const char *name, int *out_ambig)
{
    *out_ambig = 0;
    if (name == NULL || name[0] == '\0')
    {
        return -1;
    }

    if (strcmp(name, "imgstat") == 0 || strcmp(name, "info.imgstat") == 0)
    {
        name = "info.imstats";
    }
    else if (strcmp(name, "mk2Drnd") == 0 || strcmp(name, "imgen.mk2Drnd") == 0)
    {
        name = "imgen.mkrndim";
    }
    else if (strcmp(name, "lm?") == 0)
    {
        name = "hm";
    }

    /* 1. Exact match on registered command key */
    for (unsigned int i = 0; i < data.NBcmd; i++)
    {
        if (strcmp(data.cmd[i].key, name) == 0)
        {
            return (int) i;
        }
    }

    /* 2. If no dot in name, search for unambiguous "<ns>.<name>" */
    if (strchr(name, '.') == NULL)
    {
        size_t nlen      = strlen(name);
        int    match_idx = -1;
        int    match_cnt = 0;
        for (unsigned int i = 0; i < data.NBcmd; i++)
        {
            const char *dot = strchr(data.cmd[i].key, '.');
            if (dot != NULL && strcmp(dot + 1, name) == 0 && (dot - data.cmd[i].key) > 0)
            {
                match_idx = (int) i;
                match_cnt++;
            }
            else if (data.cmd[i].moduleindex >= 0 &&
                     data.module[data.cmd[i].moduleindex].shortname[0] != '\0' &&
                     strcmp(data.cmd[i].key, name) == 0)
            {
                match_idx = (int) i;
                match_cnt++;
            }
            (void) nlen;
        }
        if (match_cnt == 1)
        {
            return match_idx;
        }
        if (match_cnt > 1)
        {
            *out_ambig = 1;
            fprintf(stderr, "\033[1;31mERROR:\033[0m ambiguous short command '@%s'; matches:\n",
                    name);
            for (unsigned int i = 0; i < data.NBcmd; i++)
            {
                const char *dot = strchr(data.cmd[i].key, '.');
                if (dot != NULL && strcmp(dot + 1, name) == 0)
                {
                    fprintf(stderr, "  @%s\n", data.cmd[i].key);
                }
            }
            return -1;
        }
    }

    /* 3. If name is "<shortmod>.<cmd>", check module shortname + key */
    const char *dot = strchr(name, '.');
    if (dot != NULL)
    {
        size_t      mlen = (size_t) (dot - name);
        const char *sub  = dot + 1;
        for (unsigned int i = 0; i < data.NBcmd; i++)
        {
            int midx = data.cmd[i].moduleindex;
            if (midx >= 0)
            {
                const char *ms = data.module[midx].shortname;
                const char *mn = data.module[midx].name;
                if (((strlen(ms) == mlen && strncmp(ms, name, mlen) == 0) ||
                     (strlen(mn) == mlen && strncmp(mn, name, mlen) == 0)) &&
                    strcmp(data.cmd[i].key, sub) == 0)
                {
                    return (int) i;
                }
            }
        }
    }

    return -1;
}

static int builtin_streamcat(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: @streamcat <stream>\n");
        return 1;
    }
    const char *sname = argv[1];
    while (*sname == '@')
    {
        sname++;
    }
    if (strncmp(sname, "s.", 2) == 0)
    {
        sname += 2;
    }
    IMAGE  tmp_im;
    int    need_close = 0;
    IMAGE *im         = find_or_open_stream_or_image(sname, &tmp_im, &need_close);
    if (im == NULL || im->md == NULL)
    {
        fprintf(stderr, "ERROR: @streamcat: stream '%s' not found\n", sname);
        return 1;
    }
    uint64_t nel = im->md->nelement;
    uint8_t  dt  = im->md->datatype;
    for (uint64_t i = 0; i < nel; i++)
    {
        if (dt == _DATATYPE_FLOAT)
        {
            printf("%.9g\n", (double) im->array.F[i]);
        }
        else if (dt == _DATATYPE_DOUBLE)
        {
            printf("%.17g\n", im->array.D[i]);
        }
        else if (dt == _DATATYPE_UINT16)
        {
            printf("%u\n", (unsigned) im->array.UI16[i]);
        }
        else if (dt == _DATATYPE_INT16)
        {
            printf("%d\n", (int) im->array.SI16[i]);
        }
        else if (dt == _DATATYPE_UINT32)
        {
            printf("%u\n", im->array.UI32[i]);
        }
        else if (dt == _DATATYPE_INT32)
        {
            printf("%d\n", im->array.SI32[i]);
        }
        else if (dt == _DATATYPE_INT64)
        {
            printf("%ld\n", (long) im->array.SI64[i]);
        }
        else if (dt == _DATATYPE_UINT8)
        {
            printf("%u\n", (unsigned) im->array.UI8[i]);
        }
    }
    fflush(stdout);
    if (need_close)
    {
        ImageStreamIO_closeIm(&tmp_im);
    }
    return 0;
}

static int builtin_streamwrite(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: @streamwrite <stream>\n");
        return 1;
    }
    const char *sname = argv[1];
    while (*sname == '@')
    {
        sname++;
    }
    if (strncmp(sname, "s.", 2) == 0)
    {
        sname += 2;
    }
    IMAGE  tmp_im;
    int    need_close = 0;
    IMAGE *im         = find_or_open_stream_or_image(sname, &tmp_im, &need_close);
    if (im == NULL || im->md == NULL)
    {
        fprintf(stderr, "ERROR: @streamwrite: stream '%s' not found\n", sname);
        return 1;
    }
    uint64_t nel  = im->md->nelement;
    uint8_t  dt   = im->md->datatype;
    im->md->write = 1;
    double val    = 0.0;
    for (uint64_t i = 0; i < nel; i++)
    {
        if (scanf("%lf", &val) != 1)
        {
            break;
        }
        if (dt == _DATATYPE_FLOAT)
        {
            im->array.F[i] = (float) val;
        }
        else if (dt == _DATATYPE_DOUBLE)
        {
            im->array.D[i] = val;
        }
        else if (dt == _DATATYPE_UINT16)
        {
            im->array.UI16[i] = (uint16_t) val;
        }
        else if (dt == _DATATYPE_INT16)
        {
            im->array.SI16[i] = (int16_t) val;
        }
        else if (dt == _DATATYPE_UINT32)
        {
            im->array.UI32[i] = (uint32_t) val;
        }
        else if (dt == _DATATYPE_INT32)
        {
            im->array.SI32[i] = (int32_t) val;
        }
        else if (dt == _DATATYPE_INT64)
        {
            im->array.SI64[i] = (int64_t) val;
        }
        else if (dt == _DATATYPE_UINT8)
        {
            im->array.UI8[i] = (uint8_t) val;
        }
    }
    processinfo_update_output_stream(NULL, im, NULL);
    if (need_close)
    {
        ImageStreamIO_closeIm(&tmp_im);
    }
    return 0;
}

static int dispatch_milk_argv(int argc, char **argv)
{
    if (argc <= 0 || argv[0] == NULL)
    {
        return 0;
    }

    const char *cmdname = argv[0];
    if (cmdname[0] == '@' || cmdname[0] == ',')
    {
        cmdname++;
    }

    /* Built-in stream I/O helpers */
    if (strcmp(cmdname, "streamcat") == 0)
    {
        return builtin_streamcat(argc, argv);
    }
    if (strcmp(cmdname, "streamwrite") == 0)
    {
        return builtin_streamwrite(argc, argv);
    }

    /* Reconstruct data.CLIcmdline for commands and interceptors that inspect it */
    {
        size_t      pos = 0;
        const char *k0  = (strcmp(cmdname, "mem.listim") == 0 && argc >= 2 &&
                           (strchr(argv[1], '*') != NULL || strchr(argv[1], '?') != NULL))
                              ? "listim"
                              : cmdname;
        pos += (size_t) snprintf(data.CLIcmdline + pos, sizeof(data.CLIcmdline) - pos, "%s", k0);
        for (int i = 1; i < argc && pos + 2 < sizeof(data.CLIcmdline); i++)
        {
            pos += (size_t) snprintf(data.CLIcmdline + pos, sizeof(data.CLIcmdline) - pos, " %s",
                                     argv[i]);
        }
    }

    /* Intercept commands handled by libmilkscript interceptors / shell builtins */
    if (strcmp(cmdname, "procctl") == 0)
    {
        cli_last_retval = 0;
        cli_intercept_cmd_procctl(data.CLIcmdline);
        return cli_last_retval;
    }
    if (strcmp(cmdname, "procwait") == 0)
    {
        cli_last_retval = 0;
        cli_intercept_cmd_procwait(data.CLIcmdline);
        return cli_last_retval;
    }
    if (strcmp(cmdname, "procstat") == 0)
    {
        cli_last_retval = 0;
        cli_intercept_cmd_procstat(data.CLIcmdline);
        return cli_last_retval;
    }
    if (strcmp(cmdname, "assert") == 0)
    {
        cli_last_retval = 0;
        if (!cli_intercept_cmd_assert(data.CLIcmdline))
        {
            printf("\033[1;31m[ASSERT] syntax: assert expr=expected [~tol] or expr<val\033[0m\n");
            return 1;
        }
        return cli_last_retval;
    }
    if (strcmp(cmdname, "assigncheck") == 0)
    {
        cli_last_retval = 0;
        if (!cli_intercept_cmd_assigncheck(data.CLIcmdline))
        {
            printf(
                "\033[1;31m[ASSIGNCHECK FAIL] Usage: assigncheck [-e] <var> <val> <tol>\033[0m\n");
            return 1;
        }
        return cli_last_retval;
    }
    if (strcmp(cmdname, "dpdigits") == 0)
    {
        cli_intercept_cmd_dpdigits(data.CLIcmdline);
        return 0;
    }
    if (strcmp(cmdname, "on_update") == 0 || strcmp(cmdname, "on_fpschange") == 0 ||
        ((strcmp(cmdname, "listim") == 0 || strcmp(cmdname, "mem.listim") == 0) && argc >= 2 &&
         argv[1][0] != '-' && (strchr(argv[1], '*') != NULL || strchr(argv[1], '?') != NULL)))
    {
        cli_last_retval = 0;
        if (strcmp(cmdname, "on_fpschange") == 0 && argc < 2)
        {
            printf("Usage: on_fpschange [-l] [-n N] fpsname.param { command }\n");
            return 1;
        }
        cli_handle_shell_builtins();
        if (cli_last_retval != 0 && strcmp(cmdname, "on_fpschange") == 0)
        {
            printf("Error: on_fpschange failed\n");
        }
        return cli_last_retval;
    }

    int ambig  = 0;
    int cmdidx = resolve_milk_cmd_index(cmdname, &ambig);
    if (cmdidx < 0)
    {
        if (ambig)
        {
            return 2;
        }
        fprintf(stderr, "\033[1;31mmilk:\033[0m unknown command '@%s'\n", cmdname);
        handle_did_you_mean(cmdname);
        return 127;
    }

    /* Ensure image & variable tables have headroom */
    memory_re_alloc();
    compute_image_memory();
    compute_nb_image();

    /* Reconstruct data.CLIcmdline for commands that inspect it directly
     * (e.g., fpslist, streamlist, proclist, milkquery, fpsdump, fpsset) */
    {
        size_t pos = 0;
        pos += (size_t) snprintf(data.CLIcmdline + pos, sizeof(data.CLIcmdline) - pos, "%s",
                                 data.cmd[cmdidx].key);
        for (int i = 1; i < argc && pos + 2 < sizeof(data.CLIcmdline); i++)
        {
            pos += (size_t) snprintf(data.CLIcmdline + pos, sizeof(data.CLIcmdline) - pos, " %s",
                                     argv[i]);
        }
    }

    /* Populate data.cmdargtoken[] directly from argv[] without wordexp() */
    data.parseerror      = 0;
    data.calctmp_imindex = 0;
    for (int i = 0; i < NB_ARG_MAX; i++)
    {
        data.cmdargtoken[i].type          = CMDARGTOKEN_TYPE_UNSOLVED;
        data.cmdargtoken[i].val.string[0] = '\0';
    }

    data.cmdindex = cmdidx;
    snprintf(data.cmdargtoken[0].val.string, STRINGMAXLEN_CMDARGTOKEN_VAL, "%s",
             data.cmd[cmdidx].key);
    data.cmdargtoken[0].type = CMDARGTOKEN_TYPE_COMMAND;
    data.cmdNBarg            = 1;

    for (int i = 1; i < argc && data.cmdNBarg < NB_ARG_MAX - 1; i++)
    {
        const char *arg    = argv[i];
        size_t      arglen = strlen(arg);
        int         is_slice =
            (arg[0] != '@' && strchr(arg, '[') != NULL && arglen > 0 && arg[arglen - 1] == ']');
        snprintf(data.cmdargtoken[data.cmdNBarg].val.string, STRINGMAXLEN_CMDARGTOKEN_VAL, "%s",
                 arg);

        if (!is_slice && (arg[0] == '-' || arg[0] == '/' || strchr(arg, ' ') != NULL ||
                          strchr(arg, '\t') != NULL || arg[0] == '@'))
        {
            data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_RAWSTRING;
            /* If it is a modifier on an existing image, check if bare image exists */
            if (arg[0] == '@')
            {
                IMGID tmpid = imgid_make_from_name(arg);
                if (!tmpid.stream_parse_error && tmpid.name[0] != '\0')
                {
                    if (image_ID(tmpid.name, data.core.image, data.core.NB_MAX_IMAGE) != -1)
                    {
                        data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_EXISTINGIMAGE;
                    }
                    else
                    {
                        data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_STRING;
                    }
                }
                imgid_free(&tmpid);
            }
        }
        else
        {
            char tokstr[512];
            snprintf(tokstr, sizeof(tokstr), "%s\n", arg);
            cli_parse(tokstr);
        }
        data.cmdNBarg++;
    }
    data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_UNSOLVED;

    data.cmd[cmdidx].callcount++;
    data.CMDerrstatus = RETURN_FAILURE;

    cli_fault_isolation_arm();
    if (sigsetjmp(*cli_get_repl_env(), 1) == 0)
    {
        data.CMDerrstatus = data.cmd[cmdidx].fp();
    }
    else
    {
        int was_sigint = dcsigINT;
        dcsigINT       = 0;
        dcsigSEGV      = 0;
        dcsigBUS       = 0;
        dcsigABRT      = 0;
        cli_fault_isolation_disarm();
        return was_sigint ? 130 : 1;
    }
    cli_fault_isolation_disarm();

    if (data.CMDerrstatus != RETURN_SUCCESS)
    {
        printf("\n%c[%d;%dm ERROR %c[%d;m CLI function %s returns %d\n", (char) 27, 1, 31,
               (char) 27, 0, data.cmd[cmdidx].key, data.CMDerrstatus);
    }

    /* Clean up temporary calc images */
    for (int i = 0; i < data.calctmp_imindex; i++)
    {
        char calctmpimname[STRINGMAXLEN_IMGNAME];
        CREATE_IMAGENAME(calctmpimname, "_tmpcalc%d", i);
        if (image_ID(calctmpimname, dcimg, dcnimg) != -1)
        {
            delete_image_ID(calctmpimname, DELETE_IMAGE_ERRMODE_WARNING);
        }
    }
    data.calctmp_imindex = 0;

    return (data.CMDerrstatus == RETURN_SUCCESS) ? 0 : 1;
}

/* ============================================================
 *  Command dumping & Completion
 * ============================================================ */

void milk_server_dump_commands(FILE *fp)
{
    if (fp == NULL)
    {
        return;
    }

    for (unsigned int i = 0; i < data.NBcmd; i++)
    {
        const char *key = data.cmd[i].key;
        if (key[0] == '\0' || strchr(key, '?') != NULL)
        {
            continue;
        }
        /* Skip shell builtins that belong to bash */
        if (strcmp(key, "cd") == 0 || strcmp(key, "echo") == 0 || strcmp(key, "printf") == 0 ||
            strcmp(key, "pwd") == 0 || strcmp(key, "read") == 0 || strcmp(key, "shift") == 0 ||
            strcmp(key, "export") == 0 || strcmp(key, "alias") == 0 ||
            strcmp(key, "unalias") == 0 || strcmp(key, "unset") == 0 || strcmp(key, "time") == 0 ||
            strcmp(key, "help") == 0 || strcmp(key, "exit") == 0 || strcmp(key, "quit") == 0)
        {
            continue;
        }

        fprintf(fp, "%s\n", key);

        /* Also emit module-qualified name if command has a module shortname and no dot */
        int midx = data.cmd[i].moduleindex;
        if (strchr(key, '.') == NULL && midx >= 0 && data.module[midx].shortname[0] != '\0')
        {
            fprintf(fp, "%s.%s\n", data.module[midx].shortname, key);
        }

        /* If key is "ns.short", also emit "short" if unambiguous across all commands */
        const char *dot = strchr(key, '.');
        if (dot != NULL && dot[1] != '\0')
        {
            const char *short_key = dot + 1;
            int         cnt       = 0;
            for (unsigned int j = 0; j < data.NBcmd; j++)
            {
                const char *d2 = strchr(data.cmd[j].key, '.');
                if ((d2 != NULL && strcmp(d2 + 1, short_key) == 0) ||
                    strcmp(data.cmd[j].key, short_key) == 0)
                {
                    cnt++;
                }
            }
            if (cnt == 1)
            {
                fprintf(fp, "%s\n", short_key);
            }
        }
    }

    fprintf(fp, "imgstat\ninfo.imgstat\nmk2Drnd\nimgen.mk2Drnd\n");
}

static void handle_complete_request(int client_fd, const char *prefix, const char *comp_type)
{
    char   out[32768];
    size_t pos = 0;
    out[0]     = '\0';

    const char *sigil = getenv("MILKCLI_SIGIL");
    if (sigil == NULL || sigil[0] == '\0')
    {
        sigil = "@";
    }
    size_t siglen = strlen(sigil);

    if (comp_type != NULL && strcmp(comp_type, "cmd") == 0)
    {
        const char *bare = prefix;
        if (strncmp(bare, sigil, siglen) == 0)
        {
            bare += siglen;
        }
        else if (bare[0] == '@')
        {
            bare++;
        }
        size_t blen = strlen(bare);

        static const char *const core_cmds[] = { "h",         "hc",          "hm",      "hf",
                                                 "get",       "set",         "calc",    "test",
                                                 "exists",    "running",     "waitfor", "waitany",
                                                 "grab",      "defer",       "on",      "source",
                                                 "streamcat", "streamwrite", "unload",  NULL };
        for (int i = 0; core_cmds[i] != NULL; i++)
        {
            if (strncmp(core_cmds[i], bare, blen) == 0)
            {
                int n = snprintf(out + pos, sizeof(out) - pos, "%s%s\n", sigil, core_cmds[i]);
                if (n > 0 && pos + (size_t) n < sizeof(out))
                {
                    pos += (size_t) n;
                }
            }
        }
        for (unsigned int i = 0; i < data.NBcmd; i++)
        {
            const char *k = data.cmd[i].key;
            if (k[0] == '\0' || strchr(k, '?') != NULL)
            {
                continue;
            }
            if (strncmp(k, bare, blen) == 0)
            {
                int n = snprintf(out + pos, sizeof(out) - pos, "%s%s\n", sigil, k);
                if (n > 0 && pos + (size_t) n < sizeof(out))
                {
                    pos += (size_t) n;
                }
            }
            const char *dot = strchr(k, '.');
            if (dot != NULL && strncmp(dot + 1, bare, blen) == 0)
            {
                int n = snprintf(out + pos, sizeof(out) - pos, "%s%s\n", sigil, dot + 1);
                if (n > 0 && pos + (size_t) n < sizeof(out))
                {
                    pos += (size_t) n;
                }
            }
        }
    }
    else
    {
        /* Argument completion: modifiers, entities, streams, local images, FPS */
        size_t plen = strlen(prefix);
        if (prefix[0] == '@')
        {
            static const char *const mods[] = {
                "@TF32:", "@TF64:", "@TF16:", "@TU8:",  "@TU16:",  "@TU32:",  "@TU64:",
                "@TI8:",  "@TI16:", "@TI32:", "@TI64:", "@TC64:",  "@TC128:", "@L:",
                "@S:",    "@N:",    "@E:",    "@F:",    "@LTF32:", "@STF32:", NULL
            };
            for (int i = 0; mods[i] != NULL; i++)
            {
                if (strncmp(mods[i], prefix, plen) == 0)
                {
                    int n = snprintf(out + pos, sizeof(out) - pos, "%s\n", mods[i]);
                    if (n > 0 && pos + (size_t) n < sizeof(out))
                    {
                        pos += (size_t) n;
                    }
                }
            }
        }

        /* Local images */
        if (data.core.image != NULL)
        {
            for (long i = 0; i < data.core.NB_MAX_IMAGE; i++)
            {
                if (data.core.image[i].used == 1 &&
                    strncmp(data.core.image[i].name, prefix, plen) == 0)
                {
                    int n = snprintf(out + pos, sizeof(out) - pos, "%s\n", data.core.image[i].name);
                    if (n > 0 && pos + (size_t) n < sizeof(out))
                    {
                        pos += (size_t) n;
                    }
                }
            }
        }

        /* SHM streams and FPS instances */
        const char *shmdir = (dcshmdir[0] != '\0') ? dcshmdir : "/dev/shm";
        DIR        *dp     = opendir(shmdir);
        if (dp != NULL)
        {
            struct dirent *de;
            while ((de = readdir(dp)) != NULL)
            {
                size_t dlen = strlen(de->d_name);
                if (dlen > 7 && strcmp(de->d_name + dlen - 7, ".im.shm") == 0)
                {
                    char   sname[256];
                    size_t slen = dlen - 7;
                    if (slen < sizeof(sname))
                    {
                        memcpy(sname, de->d_name, slen);
                        sname[slen] = '\0';
                        if (strncmp(sname, prefix, plen) == 0)
                        {
                            int n = snprintf(out + pos, sizeof(out) - pos, "%s\n", sname);
                            if (n > 0 && pos + (size_t) n < sizeof(out))
                            {
                                pos += (size_t) n;
                            }
                        }
                        char ent[280];
                        snprintf(ent, sizeof(ent), "s.%s", sname);
                        if (strncmp(ent, prefix, plen) == 0)
                        {
                            int n = snprintf(out + pos, sizeof(out) - pos, "%s.\n", ent);
                            if (n > 0 && pos + (size_t) n < sizeof(out))
                            {
                                pos += (size_t) n;
                            }
                        }
                    }
                }
                else if (dlen > 8 && strcmp(de->d_name + dlen - 8, ".fps.shm") == 0)
                {
                    char   fname[256];
                    size_t flen = dlen - 8;
                    if (flen < sizeof(fname))
                    {
                        memcpy(fname, de->d_name, flen);
                        fname[flen] = '\0';
                        char ent[280];
                        snprintf(ent, sizeof(ent), "f.%s", fname);
                        if (strncmp(ent, prefix, plen) == 0)
                        {
                            int n = snprintf(out + pos, sizeof(out) - pos, "%s.\n", ent);
                            if (n > 0 && pos + (size_t) n < sizeof(out))
                            {
                                pos += (size_t) n;
                            }
                        }
                    }
                }
            }
            closedir(dp);
        }
    }

    if (pos > 0)
    {
        milk_proto_send(client_fd, MILK_MSG_VALUE, 0, out, (uint32_t) (pos + 1));
    }
    int32_t rc = 0;
    milk_proto_send(client_fd, MILK_MSG_DONE, 0, &rc, sizeof(rc));
}

/* ============================================================
 *  Watcher thread for concurrent pipeline stages & Ctrl-C cancel
 * ============================================================ */

typedef struct
{
    int          client_fd;
    int          listen_fd;
    pthread_t    main_tid;
    volatile int active;
} call_watcher_ctx_t;

static void handle_client_connection(int client_fd, int listen_fd, int *shutdown_flag);

static void *call_watcher_thread(void *arg)
{
    call_watcher_ctx_t *ctx = (call_watcher_ctx_t *) arg;
    while (ctx->active)
    {
        struct pollfd pfds[2];
        pfds[0].fd      = ctx->client_fd;
        pfds[0].events  = POLLIN | POLLHUP | POLLERR;
        pfds[0].revents = 0;
        pfds[1].fd      = ctx->listen_fd;
        pfds[1].events  = POLLIN;
        pfds[1].revents = 0;

        int pr = poll(pfds, 2, 20);
        if (pr <= 0 || !ctx->active)
        {
            continue;
        }

        /* Check if active client sent CANCEL or WINCH or disconnected */
        if (pfds[0].revents & (POLLIN | POLLHUP | POLLERR))
        {
            if (pfds[0].revents & POLLIN)
            {
                milk_frame_hdr_t hdr;
                uint8_t         *pl = NULL;
                if (milk_proto_recv(ctx->client_fd, &hdr, &pl, NULL) == 0)
                {
                    if (hdr.type == MILK_MSG_CANCEL)
                    {
                        dcsigINT = 1;
                        pthread_kill(ctx->main_tid, SIGINT);
                    }
                    else if (hdr.type == MILK_MSG_WINCH)
                    {
                        pthread_kill(ctx->main_tid, SIGWINCH);
                    }
                    free(pl);
                }
                else
                {
                    dcsigINT = 1;
                    pthread_kill(ctx->main_tid, SIGINT);
                    break;
                }
            }
            else
            {
                dcsigINT = 1;
                pthread_kill(ctx->main_tid, SIGINT);
                break;
            }
        }

        /* Check if a concurrent pipeline stage connected while main thread is busy */
        if ((pfds[1].revents & POLLIN) && ctx->active)
        {
            int cfd = accept4(ctx->listen_fd, NULL, NULL, SOCK_CLOEXEC);
            if (cfd >= 0)
            {
                struct ucred cred;
                socklen_t    clen = sizeof(cred);
                if (getsockopt(cfd, SOL_SOCKET, SO_PEERCRED, &cred, &clen) == 0 &&
                    cred.uid == getuid())
                {
                    pid_t pid = fork();
                    if (pid == 0)
                    {
                        close(ctx->listen_fd);
                        int dummy_shutdown = 0;
                        handle_client_connection(cfd, -1, &dummy_shutdown);
                        close(cfd);
                        _exit(0);
                    }
                }
                close(cfd);
            }
        }
    }
    return NULL;
}

/* ============================================================
 *  Client Connection Handler
 * ============================================================ */

static void handle_client_connection(int client_fd, int listen_fd, int *shutdown_flag)
{
    milk_frame_hdr_t hdr;
    uint8_t         *payload     = NULL;
    int              recv_fds[4] = { -1, -1, -1, -1 };

    if (milk_proto_recv(client_fd, &hdr, &payload, recv_fds) != 0)
    {
        return;
    }

    /* If async flag is set, or if this is a blocking WAITFOR/GRAB request,
     * fork a worker in the parent server so the main server loop remains
     * non-blocking for concurrent clients. */
    int is_forked_worker = 0;
    if (((hdr.flags & MILK_FLAG_ASYNC) || hdr.type == MILK_MSG_WAITFOR ||
         hdr.type == MILK_MSG_GRAB) &&
        listen_fd >= 0)
    {
        pid_t pid = fork();
        if (pid > 0)
        {
            for (int i = 0; i < 4; i++)
            {
                if (recv_fds[i] >= 0)
                {
                    close(recv_fds[i]);
                }
            }
            free(payload);
            return;
        }
        else if (pid == 0)
        {
            is_forked_worker = 1;
            close(listen_fd);
            listen_fd = -1;
        }
    }

    int32_t rc = 0;

    if (hdr.type == MILK_MSG_PING)
    {
        milk_proto_send(client_fd, MILK_MSG_DONE, 0, &rc, sizeof(rc));
        goto cleanup;
    }

    if (hdr.type == MILK_MSG_SHUTDOWN)
    {
        *shutdown_flag = 1;
        milk_proto_send(client_fd, MILK_MSG_DONE, 0, &rc, sizeof(rc));
        goto cleanup;
    }

    if (hdr.type == MILK_MSG_LISTCMDS)
    {
        char  *cmdbuf = NULL;
        size_t cmdlen = 0;
        FILE  *mfp    = open_memstream(&cmdbuf, &cmdlen);
        if (mfp != NULL)
        {
            milk_server_dump_commands(mfp);
            fclose(mfp);
            if (cmdbuf != NULL)
            {
                milk_proto_send(client_fd, MILK_MSG_VALUE, 0, cmdbuf, (uint32_t) (cmdlen + 1));
                free(cmdbuf);
            }
        }
        milk_proto_send(client_fd, MILK_MSG_DONE, 0, &rc, sizeof(rc));
        goto cleanup;
    }

    if (hdr.type == MILK_MSG_COMPLETE)
    {
        const char *prefix = (payload != NULL) ? (const char *) payload : "";
        size_t      plen   = strlen(prefix) + 1;
        const char *comp_type =
            (payload != NULL && plen < hdr.len) ? (const char *) (payload + plen) : "arg";
        handle_complete_request(client_fd, prefix, comp_type);
        goto cleanup;
    }

    if (hdr.type == MILK_MSG_GET)
    {
        char val[1024];
        if (payload != NULL && milk_entity_get((const char *) payload, val, sizeof(val)) == 0)
        {
            milk_proto_send(client_fd, MILK_MSG_VALUE, 0, val, (uint32_t) (strlen(val) + 1));
            rc = 0;
        }
        else
        {
            rc = 1;
        }
        milk_proto_send(client_fd, MILK_MSG_DONE, 0, &rc, sizeof(rc));
        goto cleanup;
    }

    if (hdr.type == MILK_MSG_EXISTS)
    {
        if (payload != NULL && hdr.len >= 2)
        {
            int         check_run = (payload[0] == 'R') ? 1 : 0;
            const char *spec      = (const char *) (payload + 1);
            rc                    = milk_entity_exists(spec, check_run) ? 0 : 1;
        }
        else
        {
            rc = 1;
        }
        milk_proto_send(client_fd, MILK_MSG_DONE, 0, &rc, sizeof(rc));
        goto cleanup;
    }

    if (hdr.type == MILK_MSG_WAITFOR)
    {
        if (payload != NULL && hdr.len > 0)
        {
            const char *target = (const char *) payload;
            size_t      o1     = strlen(target) + 1;
            const char *arg2   = (o1 < hdr.len) ? (const char *) (payload + o1) : "";
            size_t      o2     = o1 + strlen(arg2) + 1;
            const char *tstr   = (o2 < hdr.len) ? (const char *) (payload + o2) : "0";
            double      tsec   = strtod(tstr, NULL);
            rc                 = milk_server_waitfor(client_fd, target, arg2, tsec);
        }
        else
        {
            rc = 1;
        }
        milk_proto_send(client_fd, MILK_MSG_DONE, 0, &rc, sizeof(rc));
        goto cleanup;
    }

    if (hdr.type == MILK_MSG_GRAB)
    {
        if (payload != NULL && hdr.len > 0)
        {
            const char *sname   = (const char *) payload;
            size_t      o1      = strlen(sname) + 1;
            const char *nfstr   = (o1 < hdr.len) ? (const char *) (payload + o1) : "1";
            size_t      o2      = o1 + strlen(nfstr) + 1;
            const char *outname = (o2 < hdr.len) ? (const char *) (payload + o2) : "";
            size_t      o3      = o2 + strlen(outname) + 1;
            const char *tstr    = (o3 < hdr.len) ? (const char *) (payload + o3) : "0";
            long        nf      = strtol(nfstr, NULL, 10);
            double      tsec    = strtod(tstr, NULL);
            rc                  = milk_server_grab(client_fd, sname, nf, outname, tsec);
        }
        else
        {
            rc = 1;
        }
        milk_proto_send(client_fd, MILK_MSG_DONE, 0, &rc, sizeof(rc));
        goto cleanup;
    }

    /* Commands that may read/write stdio or cwd: SET, CALC, TEST, CALL */
    {
        int saved_stdin  = -1;
        int saved_stdout = -1;
        int saved_stderr = -1;
        int saved_cwd    = -1;

        if (recv_fds[0] >= 0)
        {
            saved_stdin = fcntl(STDIN_FILENO, F_DUPFD_CLOEXEC, 3);
            dup2(recv_fds[0], STDIN_FILENO);
            close(recv_fds[0]);
            recv_fds[0] = -1;
        }
        if (recv_fds[1] >= 0)
        {
            struct stat st;
            if (fstat(recv_fds[1], &st) == 0 && S_ISFIFO(st.st_mode))
            {
                fcntl(recv_fds[1], F_SETPIPE_SZ, 1048576);
            }
            saved_stdout = fcntl(STDOUT_FILENO, F_DUPFD_CLOEXEC, 3);
            dup2(recv_fds[1], STDOUT_FILENO);
            close(recv_fds[1]);
            recv_fds[1] = -1;
        }
        if (recv_fds[2] >= 0)
        {
            saved_stderr = fcntl(STDERR_FILENO, F_DUPFD_CLOEXEC, 3);
            dup2(recv_fds[2], STDERR_FILENO);
            close(recv_fds[2]);
            recv_fds[2] = -1;
        }
        if (recv_fds[3] >= 0)
        {
            saved_cwd = open(".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
            if (fchdir(recv_fds[3]) != 0)
            {
            }
            close(recv_fds[3]);
            recv_fds[3] = -1;
        }

        if (hdr.type == MILK_MSG_SET)
        {
            if (payload != NULL && hdr.len > 0)
            {
                const char *spec    = (const char *) payload;
                size_t      o1      = strlen(spec) + 1;
                const char *val     = (o1 < hdr.len) ? (const char *) (payload + o1) : NULL;
                size_t      o2      = (val != NULL) ? (o1 + strlen(val) + 1) : hdr.len;
                const char *comment = (o2 < hdr.len) ? (const char *) (payload + o2) : NULL;
                rc                  = (milk_entity_set(spec, val, comment) == 0) ? 0 : 1;
            }
            else
            {
                rc = 1;
            }
        }
        else if (hdr.type == MILK_MSG_CALC || hdr.type == MILK_MSG_TEST)
        {
            int is_test = (hdr.type == MILK_MSG_TEST) ? 1 : 0;
            if (payload != NULL && hdr.len > 0)
            {
                const char *expr = (const char *) payload;
                uint32_t    off  = (uint32_t) (strlen(expr) + 1);
                if (off + sizeof(uint32_t) <= hdr.len)
                {
                    uint32_t nvars = 0;
                    memcpy(&nvars, payload + off, sizeof(uint32_t));
                    off += (uint32_t) sizeof(uint32_t);
                    sync_bash_vars_into_dcvar(payload, hdr.len, nvars, &off);
                }

                long          snap_nmax = 0;
                dcvar_snap_t *snap      = snapshot_dcvar(&snap_nmax);

                /* Expand inline @s./s., @f./f., @p./p., @q./q. entity references inside expr */
                char expanded[8192];
                expand_entities_in_expr(expr, expanded, sizeof(expanded));

                int eval_rc = 2;
                cli_calc_eval_bash(expanded, is_test, &eval_rc);
                rc = eval_rc;

                send_modified_dcvars(client_fd, snap, snap_nmax);
                free(snap);
            }
            else
            {
                rc = is_test ? 1 : 0;
            }
        }
        else if (hdr.type == MILK_MSG_CALL)
        {
            if (payload != NULL && hdr.len >= sizeof(uint32_t))
            {
                uint32_t off  = 0;
                uint32_t argc = 0;
                memcpy(&argc, payload + off, sizeof(uint32_t));
                off += (uint32_t) sizeof(uint32_t);

                char **argv = (char **) calloc(argc + 1, sizeof(char *));
                for (uint32_t i = 0; i < argc && off < hdr.len; i++)
                {
                    argv[i] = (char *) (payload + off);
                    off += (uint32_t) (strnlen(argv[i], hdr.len - off) + 1);
                }

                /* Unpack environment updates */
                if (off + sizeof(uint32_t) <= hdr.len)
                {
                    uint32_t nenv = 0;
                    memcpy(&nenv, payload + off, sizeof(uint32_t));
                    off += (uint32_t) sizeof(uint32_t);
                    for (uint32_t i = 0; i < nenv && off < hdr.len; i++)
                    {
                        const char *kv = (const char *) (payload + off);
                        size_t      kl = strnlen(kv, hdr.len - off);
                        off += (uint32_t) (kl + 1);
                        const char *eq = strchr(kv, '=');
                        if (eq != NULL)
                        {
                            char   kname[128];
                            size_t kn = (size_t) (eq - kv);
                            if (kn < sizeof(kname))
                            {
                                memcpy(kname, kv, kn);
                                kname[kn] = '\0';
                                setenv(kname, eq + 1, 1);
                                if (strcmp(kname, "MILK_SHM_DIR") == 0 &&
                                    strcmp(dcshmdir, eq + 1) != 0)
                                {
                                    int saved_q = dcquiet;
                                    dcquiet     = 1;
                                    setSHMdir();
                                    dcquiet = saved_q;
                                }
                            }
                        }
                    }
                }

                /* Unpack synced scalar variables */
                if (off + sizeof(uint32_t) <= hdr.len)
                {
                    uint32_t nvars = 0;
                    memcpy(&nvars, payload + off, sizeof(uint32_t));
                    off += (uint32_t) sizeof(uint32_t);
                    sync_bash_vars_into_dcvar(payload, hdr.len, nvars, &off);
                }

                long          snap_nmax    = 0;
                dcvar_snap_t *snap         = snapshot_dcvar(&snap_nmax);
                unsigned int  nbcmd_before = data.NBcmd;

                /* Start watcher thread for Ctrl-C / concurrent pipeline stage */
                call_watcher_ctx_t wctx;
                pthread_t          wtid;
                int                has_watcher = 0;
                if (listen_fd >= 0)
                {
                    wctx.client_fd = client_fd;
                    wctx.listen_fd = listen_fd;
                    wctx.main_tid  = pthread_self();
                    wctx.active    = 1;
                    if (pthread_create(&wtid, NULL, call_watcher_thread, &wctx) == 0)
                    {
                        has_watcher = 1;
                    }
                }

                rc = dispatch_milk_argv((int) argc, argv);

                if (has_watcher)
                {
                    wctx.active = 0;
                    pthread_join(wtid, NULL);
                }

                fflush(stdout);
                fflush(stderr);

                /* Send newly created/updated dcvar entries back to bash */
                send_modified_dcvars(client_fd, snap, snap_nmax);
                free(snap);

                /* If a module was dynamically loaded (@mload / @soload), send DEFCMD */
                for (unsigned int ci = nbcmd_before; ci < data.NBcmd; ci++)
                {
                    const char *ck = data.cmd[ci].key;
                    if (ck[0] != '\0')
                    {
                        milk_proto_send(client_fd, MILK_MSG_DEFCMD, 0, ck,
                                        (uint32_t) (strlen(ck) + 1));
                        int midx = data.cmd[ci].moduleindex;
                        if (strchr(ck, '.') == NULL && midx >= 0 &&
                            data.module[midx].shortname[0] != '\0')
                        {
                            char qk[256];
                            snprintf(qk, sizeof(qk), "%s.%s", data.module[midx].shortname, ck);
                            milk_proto_send(client_fd, MILK_MSG_DEFCMD, 0, qk,
                                            (uint32_t) (strlen(qk) + 1));
                        }
                    }
                }

                free(argv);
            }
            else
            {
                rc = 1;
            }
        }

        fflush(stdout);
        fflush(stderr);

        if (saved_stdin >= 0)
        {
            dup2(saved_stdin, STDIN_FILENO);
            close(saved_stdin);
        }
        if (saved_stdout >= 0)
        {
            dup2(saved_stdout, STDOUT_FILENO);
            close(saved_stdout);
        }
        if (saved_stderr >= 0)
        {
            dup2(saved_stderr, STDERR_FILENO);
            close(saved_stderr);
        }
        if (saved_cwd >= 0)
        {
            if (fchdir(saved_cwd) != 0)
            {
            }
            close(saved_cwd);
        }

        milk_proto_send(client_fd, MILK_MSG_DONE, 0, &rc, sizeof(rc));
    }

cleanup:
    for (int i = 0; i < 4; i++)
    {
        if (recv_fds[i] >= 0)
        {
            close(recv_fds[i]);
        }
    }
    free(payload);

    if (is_forked_worker)
    {
        close(client_fd);
        _exit(0);
    }
}

/* ============================================================
 *  Main Server Loop & Listener Helpers
 * ============================================================ */

int milk_server_create_listener(const char *sockpath)
{
    pthread_once(&g_atfork_once, register_atfork_handler);

    if (sockpath == NULL || sockpath[0] == '\0')
    {
        fprintf(stderr, "ERROR: milk-cli --server requires a socket path\n");
        return -1;
    }

    struct sockaddr_un addr;
    socklen_t          addrlen = 0;
    if (milk_sockaddr_init(sockpath, &addr, &addrlen) != 0)
    {
        fprintf(stderr, "ERROR: invalid socket path '%s'\n", sockpath);
        return -1;
    }

    if (sockpath[0] != '@')
    {
        unlink(sockpath);
    }

    int listen_fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (listen_fd < 0)
    {
        perror("milk-cli --server: socket");
        return -1;
    }

    if (bind(listen_fd, (struct sockaddr *) &addr, addrlen) != 0)
    {
        perror("milk-cli --server: bind");
        close(listen_fd);
        return -1;
    }

    if (sockpath[0] != '@')
    {
        chmod(sockpath, 0600);
    }

    if (listen(listen_fd, 64) != 0)
    {
        perror("milk-cli --server: listen");
        close(listen_fd);
        return -1;
    }

    setenv("MILKCLI_SOCK", sockpath, 1);

    /* Snapshot initial built-in dcvar entries into cache */
    if (data.core.variable != NULL)
    {
        for (long i = 0; i < data.core.NB_MAX_VARIABLE && i < DCVAR_CACHE_MAX; i++)
        {
            if (data.core.variable[i].used && data.core.variable[i].type == 0)
            {
                snprintf(g_dcvar_cached_str[i], sizeof(g_dcvar_cached_str[i]), "%.*g",
                         cli_float_digits, data.core.variable[i].value.f);
            }
        }
    }

    return listen_fd;
}

void milk_server_accept_and_handle(int listen_fd, int *shutdown_flag)
{
    while (waitpid(0, NULL, WNOHANG) > 0)
    {
    }

    int client_fd = accept4(listen_fd, NULL, NULL, SOCK_CLOEXEC);
    if (client_fd < 0)
    {
        return;
    }

    struct ucred cred;
    socklen_t    clen = sizeof(cred);
    if (getsockopt(client_fd, SOL_SOCKET, SO_PEERCRED, &cred, &clen) != 0 || cred.uid != getuid())
    {
        close(client_fd);
        return;
    }

    int dummy = 0;
    handle_client_connection(client_fd, listen_fd, shutdown_flag != NULL ? shutdown_flag : &dummy);
    close(client_fd);

    while (waitpid(0, NULL, WNOHANG) > 0)
    {
    }
}

int milk_server_run(const char *sockpath, int owner_fd, int ready_fd)
{
    int listen_fd = milk_server_create_listener(sockpath);
    if (listen_fd < 0)
    {
        return 1;
    }

    /* Notify caller that server is ready to accept connections */
    if (ready_fd >= 0)
    {
        if (write(ready_fd, "READY\n", 6) < 0)
        {
        }
        close(ready_fd);
    }

    int shutdown_flag = 0;
    while (!shutdown_flag)
    {
        /* Reap any finished async/concurrent worker children in our pgrp */
        while (waitpid(0, NULL, WNOHANG) > 0)
        {
        }

        struct pollfd pfds[2];
        int           npfds = 1;
        pfds[0].fd          = listen_fd;
        pfds[0].events      = POLLIN;
        pfds[0].revents     = 0;

        if (owner_fd >= 0)
        {
            pfds[1].fd      = owner_fd;
            pfds[1].events  = POLLIN | POLLHUP | POLLERR;
            pfds[1].revents = 0;
            npfds           = 2;
        }

        int pr = poll(pfds, (nfds_t) npfds, 1000);
        if (pr < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            break;
        }

        if (npfds == 2 && (pfds[1].revents & (POLLHUP | POLLERR | POLLIN)))
        {
            /* Owning shell exited: wait 50ms for any final subshell calls, then exit */
            struct pollfd drain_pfd = { listen_fd, POLLIN, 0 };
            if (poll(&drain_pfd, 1, 50) <= 0)
            {
                break;
            }
        }

        if (pfds[0].revents & POLLIN)
        {
            milk_server_accept_and_handle(listen_fd, &shutdown_flag);
        }
    }

    close(listen_fd);
    if (sockpath[0] != '@')
    {
        unlink(sockpath);
    }
    return 0;
}
