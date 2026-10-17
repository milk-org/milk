// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_inspect.c
 *
 * @brief CLI commands for system inspection, JSON introspection, and process control.
 */

#include <dirent.h>
#include <errno.h>
#include <fnmatch.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "milk_entity_ns.h"
#include "CLIcore.h"
#include "CLIcore_script.h"
#include "ImageStreamIO/ImageStreamIO.h"

#include "fps_connect.h"
#include "fps_disconnect.h"
#include "fps_printparameter_valuestring.h"
#include "fps_shmdirname.h"
#include "fpsseq.h"

extern PROCESSINFO *processinfo_shm_link(const char *pname, int *fd);
extern errno_t      processinfo_procdirname(char *procdname);

static const char *get_shmdir_local(void)
{
    const char *env = getenv("MILK_SHM_DIR");
    if (env != NULL && env[0] != '\0')
    {
        return env;
    }
    if (dcshmdir[0] != '\0')
    {
        return dcshmdir;
    }
    return "/dev/shm";
}

static void get_procdir_local(char *pdname, size_t pdsz)
{
    const char *pdir = getenv("MILK_PROC_DIR");
    const char *sdir = getenv("MILK_SHM_DIR");
    if (sdir != NULL && sdir[0] != '\0')
    {
        if (pdir == NULL || pdir[0] == '\0' || strncmp(pdir, "/milk/shm", 9) == 0 ||
            strcmp(pdir, "/dev/shm") == 0)
        {
            snprintf(pdname, pdsz, "%s", sdir);
            return;
        }
    }
    if (pdir != NULL && pdir[0] != '\0')
    {
        snprintf(pdname, pdsz, "%s", pdir);
        return;
    }
    if (dcshmdir[0] != '\0')
    {
        snprintf(pdname, pdsz, "%s", dcshmdir);
        return;
    }
    processinfo_procdirname(pdname);
}

static char s_pinfolist_dir[256] = "";
static int  s_pinfolist_fd       = -1;

static PROCESSINFOLIST *get_pinfolist_local(void)
{
    char pdname[256];
    get_procdir_local(pdname, sizeof(pdname));
    if (pinfolist != NULL && pinfolist != MAP_FAILED && strcmp(s_pinfolist_dir, pdname) == 0)
    {
        return pinfolist;
    }
    if (pinfolist != NULL && pinfolist != MAP_FAILED)
    {
        munmap(pinfolist, sizeof(PROCESSINFOLIST));
        pinfolist = NULL;
    }
    if (s_pinfolist_fd >= 0)
    {
        close(s_pinfolist_fd);
        s_pinfolist_fd = -1;
    }
    snprintf(s_pinfolist_dir, sizeof(s_pinfolist_dir), "%s", pdname);
    char lfn[512];
    snprintf(lfn, sizeof(lfn), "%s/processinfo.list.shm", pdname);
    if (access(lfn, R_OK) == 0)
    {
        int lfd   = -1;
        pinfolist = (PROCESSINFOLIST *) processinfo_shm_link(lfn, &lfd);
        if (pinfolist == MAP_FAILED)
        {
            if (lfd >= 0)
            {
                close(lfd);
            }
            pinfolist = NULL;
        }
        else
        {
            s_pinfolist_fd = lfd;
        }
    }
    return pinfolist;
}

static PROCESSINFO *open_proc_shm_local(const char *pdname,
                                        const char *pname,
                                        pid_t       fpid,
                                        int        *pfd_out)
{
    *pfd_out = -1;
    if (fpid <= 0)
    {
        return NULL;
    }
    char pfn[512];
    if (pname != NULL && pname[0] != '\0')
    {
        snprintf(pfn, sizeof(pfn), "%s/proc.%s.%06d.shm", pdname, pname, (int) fpid);
        if (access(pfn, R_OK) == 0)
        {
            PROCESSINFO *pi = processinfo_shm_link(pfn, pfd_out);
            if (pi != MAP_FAILED && pi != NULL)
            {
                return pi;
            }
            if (*pfd_out >= 0)
            {
                close(*pfd_out);
                *pfd_out = -1;
            }
        }
    }
    snprintf(pfn, sizeof(pfn), "%s/proc.%d.shm", pdname, (int) fpid);
    if (access(pfn, R_OK) == 0)
    {
        PROCESSINFO *pi = processinfo_shm_link(pfn, pfd_out);
        if (pi != MAP_FAILED && pi != NULL)
        {
            return pi;
        }
        if (*pfd_out >= 0)
        {
            close(*pfd_out);
            *pfd_out = -1;
        }
    }
    return NULL;
}

static const char *fps_status_str(uint32_t st)
{
    if (st & (FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN | FUNCTION_PARAMETER_STRUCT_STATUS_RUN |
              FUNCTION_PARAMETER_STRUCT_STATUS_RUNLOOP))
    {
        return "RUN";
    }
    if (st & (FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF | FUNCTION_PARAMETER_STRUCT_STATUS_CONF))
    {
        return "CONF";
    }
    return "IDLE";
}

static const char *proc_status_str(const PROCESSINFO *pi, pid_t pid)
{
    if (pi == NULL)
    {
        return (pid > 0 && kill(pid, 0) != 0 && errno == ESRCH) ? "CRASHED" : "UNKNOWN";
    }
    pid_t check_pid = (pi->PID > 0) ? pi->PID : pid;
    if (check_pid > 0 && kill(check_pid, 0) != 0 && errno == ESRCH)
    {
        if (pi->loopstat == PROCESSINFO_LOOPSTAT_STOP || pi->CTRLval == PROCESSINFO_CTRLVAL_EXIT)
        {
            return "STOPPED";
        }
        return "CRASHED";
    }
    if (pi->loopstat == PROCESSINFO_LOOPSTAT_STOP || pi->CTRLval == PROCESSINFO_CTRLVAL_EXIT)
    {
        return "STOPPED";
    }
    if (pi->loopstat == PROCESSINFO_LOOPSTAT_PAUSE || pi->CTRLval == PROCESSINFO_CTRLVAL_PAUSE)
    {
        return "PAUSED";
    }
    switch (pi->loopstat)
    {
    case PROCESSINFO_LOOPSTAT_INIT:
        return "INIT";
    case PROCESSINFO_LOOPSTAT_ACTIVE:
        return "ACTIVE";
    case PROCESSINFO_LOOPSTAT_PAUSE:
        return "PAUSED";
    case PROCESSINFO_LOOPSTAT_STOP:
        return "STOPPED";
    case PROCESSINFO_LOOPSTAT_ERROR:
        return "ERROR";
    case PROCESSINFO_LOOPSTAT_SPIN:
        return "SPINNING";
    case PROCESSINFO_LOOPSTAT_CRASHED:
        return "CRASHED";
    default:
        return "OTHER";
    }
}

static const char *strip_ws_local(const char *s)
{
    while (*s == ' ' || *s == '\t')
    {
        s++;
    }
    return s;
}

static int starts_with_local(const char *line, const char *prefix)
{
    return strncmp(line, prefix, strlen(prefix)) == 0;
}

/**
 * json_escape_str - write JSON-escaped string to buf
 * @buf:   destination buffer
 * @bufsz: size of @buf in bytes
 * @src:   NUL-terminated source string
 */
static void json_escape_str(char *buf, size_t bufsz, const char *src)
{
    size_t bi = 0;
    for (const char *p = src; *p != '\0' && bi + 6 < bufsz; p++)
    {
        unsigned char c = (unsigned char) *p;
        if (c == '"' || c == '\\')
        {
            buf[bi++] = '\\';
            buf[bi++] = (char) c;
        }
        else if (c == '\n')
        {
            buf[bi++] = '\\';
            buf[bi++] = 'n';
        }
        else if (c == '\r')
        {
            buf[bi++] = '\\';
            buf[bi++] = 'r';
        }
        else if (c == '\t')
        {
            buf[bi++] = '\\';
            buf[bi++] = 't';
        }
        else if (c < 0x20)
        {
            int n = snprintf(buf + bi, bufsz - bi, "\\u%04x", (unsigned int) c);
            if (n > 0)
            {
                bi += (size_t) n;
            }
        }
        else
        {
            buf[bi++] = (char) c;
        }
    }
    buf[bi] = '\0';
}

/* ============================================================
 *  fpslist
 * ============================================================
 */

static int emit_fps_json_body(const char *pat, int indent)
{
    char shmdname[STRINGMAXLEN_SHMDIRNAME];
    function_parameter_struct_shmdirname(shmdname);

    DIR *d = opendir(shmdname);
    if (d == NULL)
    {
        return 0;
    }

    char pad[32];
    {
        int n = indent;
        if (n >= (int) sizeof(pad))
        {
            n = (int) sizeof(pad) - 1;
        }
        memset(pad, ' ', (size_t) n);
        pad[n] = '\0';
    }

    int            count = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL)
    {
        char *sfx = strstr(de->d_name, ".fps.shm");
        if (sfx == NULL)
        {
            continue;
        }

        char   fpsname[STRINGMAXLEN_FPS_NAME];
        size_t nlen = (size_t) (sfx - de->d_name);
        if (nlen >= sizeof(fpsname))
        {
            continue;
        }
        strncpy(fpsname, de->d_name, nlen);
        fpsname[nlen] = '\0';

        if (pat != NULL && fnmatch(pat, fpsname, 0) != 0)
        {
            continue;
        }

        FPS fps;
        fps.SMfd = -1;
        int rc   = fps_connect(fpsname, &fps, FPSCONNECT_SIMPLE);
        if (rc == -1 || fps.md == NULL)
        {
            continue;
        }

        const char *ststr = fps_status_str(fps.md->status);

        char nesc[STRINGMAXLEN_FPS_NAME];
        char desc_esc[512];
        json_escape_str(nesc, sizeof(nesc), fpsname);
        json_escape_str(desc_esc, sizeof(desc_esc), fps.md->description);

        if (count > 0)
        {
            printf(",\n");
        }
        printf("%s{\n", pad);
        printf("%s  \"name\": \"%s\",\n", pad, nesc);
        printf("%s  \"status\": \"%s\",\n", pad, ststr);
        printf("%s  \"description\": \"%s\"\n", pad, desc_esc);
        printf("%s}", pad);

        count++;
        fps_disconnect(&fps);
    }
    closedir(d);
    return count;
}

/* ============================================================
 *  fpsdump
 * ============================================================
 */

#ifndef MILK_BASH_CLIENT /* an engine command: reads its arguments from the engine */
/**
 * @brief fpsdump command — dump all FPS parameters
 */
errno_t cli_cmd_fpsdump(void)
{
    int tabmode     = 0;
    int jsonmode    = 0;
    int arg_fpsname = -1;

    if (data.cmdNBarg < 2)
    {
        printf("Usage: fpsdump [-t] [--json] <fpsname>\n");
        return RETURN_FAILURE;
    }

    for (int a = 1; a < data.cmdNBarg; a++)
    {
        const char *tok = data.cmdargtoken[a].val.string;
        if (strcmp(tok, "-t") == 0)
        {
            tabmode = 1;
        }
        else if (strcmp(tok, "--json") == 0)
        {
            jsonmode = 1;
        }
        else
        {
            arg_fpsname = a;
        }
    }

    if (tabmode && jsonmode)
    {
        printf("fpsdump: -t and --json are mutually exclusive\n");
        return RETURN_FAILURE;
    }

    if (arg_fpsname < 0)
    {
        printf("Usage: fpsdump [-t] [--json] <fpsname>\n");
        return RETURN_FAILURE;
    }

    const char *fpsname = data.cmdargtoken[arg_fpsname].val.string;

    FPS fps;
    fps.SMfd = -1;
    int rc   = fps_connect(fpsname, &fps, FPSCONNECT_SIMPLE);
    if (rc == -1 || fps.md == NULL || fps.parray == NULL)
    {
        printf("fpsdump: cannot connect to FPS '%s'\n", fpsname);
        return RETURN_FAILURE;
    }

    if (jsonmode)
    {
        printf("{\n");
    }
    int first_json_item = 1;

    for (int pi = 0; pi < fps.md->NBparamMAX; pi++)
    {
        if (!(fps.parray[pi].fpflag & FPFLAG_ACTIVE))
        {
            continue;
        }
        char vstr[512];
        functionparameter_GetParamValueString(&fps.parray[pi], vstr, (int) sizeof(vstr));

        if (tabmode)
        {
            const char *tname = "UNKNOWN";
            switch (fps.parray[pi].type)
            {
            case FPTYPE_INT64:
                tname = "INT64";
                break;
            case FPTYPE_FLOAT64:
                tname = "FLOAT64";
                break;
            case FPTYPE_FLOAT32:
                tname = "FLOAT32";
                break;
            case FPTYPE_STRING:
                tname = "STRING";
                break;
            case FPTYPE_ONOFF:
                tname = "ONOFF";
                break;
            case FPTYPE_FILENAME:
                tname = "FILENAME";
                break;
            case FPTYPE_FITSFILENAME:
                tname = "FITSFILENAME";
                break;
            case FPTYPE_EXECFILENAME:
                tname = "EXECFILENAME";
                break;
            case FPTYPE_DIRNAME:
                tname = "DIRNAME";
                break;
            case FPTYPE_STREAMNAME:
                tname = "STREAMNAME";
                break;
            case FPTYPE_FPSNAME:
                tname = "FPSNAME";
                break;
            case FPTYPE_TIMESPEC:
                tname = "TIMESPEC";
                break;
            case FPTYPE_PID:
                tname = "PID";
                break;
            default:
                break;
            }
            printf("%s\t%s\t%s\n", fps.parray[pi].keyword[0], tname, vstr);
        }
        else if (jsonmode)
        {
            char kesc[128];
            char sesc[1024];
            json_escape_str(kesc, sizeof(kesc), fps.parray[pi].keyword[0]);
            if (!first_json_item)
            {
                printf(",\n");
            }
            first_json_item = 0;
            switch (fps.parray[pi].type)
            {
            case FPTYPE_INT64:
                printf("  \"%s\": %lld", kesc, (long long) fps.parray[pi].val.i64[0]);
                break;
            case FPTYPE_FLOAT64:
            {
                double v = fps.parray[pi].val.f64[0];
                if (isfinite(v))
                {
                    printf("  \"%s\": %g", kesc, v);
                }
                else
                {
                    printf("  \"%s\": null", kesc);
                }
                break;
            }
            case FPTYPE_FLOAT32:
            {
                float v = fps.parray[pi].val.f32[0];
                if (isfinite(v))
                {
                    printf("  \"%s\": %g", kesc, (double) v);
                }
                else
                {
                    printf("  \"%s\": null", kesc);
                }
                break;
            }
            case FPTYPE_ONOFF:
                printf("  \"%s\": %d", kesc, (int) fps.parray[pi].val.i64[0]);
                break;
            default:
                json_escape_str(sesc, sizeof(sesc), vstr);
                printf("  \"%s\": \"%s\"", kesc, sesc);
                break;
            }
        }
        else
        {
            printf("%s=%s\n", fps.parray[pi].keyword[0], vstr);
        }
    }

    if (jsonmode)
    {
        printf("\n}\n");
    }

    fps_disconnect(&fps);
    return RETURN_SUCCESS;
}
#endif

/* ============================================================
 *  streamlist
 * ============================================================
 */

static void format_img_dims(uint8_t naxis, const uint32_t *size, char *szstr, size_t szlen)
{
    if (naxis <= 1)
    {
        snprintf(szstr, szlen, "%u", (naxis == 1) ? size[0] : 0U);
    }
    else if (naxis == 2)
    {
        snprintf(szstr, szlen, "%ux%u", size[0], size[1]);
    }
    else
    {
        snprintf(szstr, szlen, "%ux%ux%u", size[0], size[1], size[2]);
    }
}

static int emit_streams_json_body(const char *pat, int indent)
{
    char pad[32];
    {
        int n = indent;
        if (n >= (int) sizeof(pad))
        {
            n = (int) sizeof(pad) - 1;
        }
        memset(pad, ' ', (size_t) n);
        pad[n] = '\0';
    }

    DIR *d = opendir(get_shmdir_local());
    if (d == NULL)
    {
        return 0;
    }

    int            count = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL)
    {
        char *sfx = strstr(de->d_name, ".im.shm");
        if (sfx == NULL || strstr(de->d_name, ".fps.shm") != NULL)
        {
            continue;
        }

        char   sname[256];
        size_t nlen = (size_t) (sfx - de->d_name);
        if (nlen >= sizeof(sname))
        {
            continue;
        }
        strncpy(sname, de->d_name, nlen);
        sname[nlen] = '\0';

        if (strncmp(sname, "_tmpcalc", 8) == 0)
        {
            continue;
        }
        if (pat != NULL && fnmatch(pat, sname, 0) != 0)
        {
            continue;
        }

        IMAGE img;
        memset(&img, 0, sizeof(IMAGE));
        errno_t sret = ImageStreamIO_openIm(&img, sname);
        if (sret != IMAGESTREAMIO_SUCCESS || img.md == NULL)
        {
            continue;
        }

        char        nesc[256];
        char        tesc[64];
        const char *dtype_name = ImageStreamIO_typename(img.md->datatype);
        if (dtype_name == NULL)
        {
            dtype_name = "?";
        }
        json_escape_str(nesc, sizeof(nesc), sname);
        json_escape_str(tesc, sizeof(tesc), dtype_name);

        if (count > 0)
        {
            printf(",\n");
        }
        printf("%s{\n", pad);
        printf("%s  \"name\": \"%s\",\n", pad, nesc);
        printf("%s  \"naxis\": %u,\n", pad, img.md->naxis);
        printf("%s  \"size\": [", pad);
        for (int ax = 0; ax < img.md->naxis; ax++)
        {
            if (ax > 0)
            {
                printf(", ");
            }
            printf("%u", img.md->size[ax]);
        }
        printf("],\n");
        printf("%s  \"type\": \"%s\",\n", pad, tesc);
        printf("%s  \"cnt0\": %lu\n", pad, (unsigned long) img.md->cnt0);
        printf("%s}", pad);

        count++;
        ImageStreamIO_closeIm(&img);
    }
    closedir(d);
    return count;
}

/* ============================================================
 *  proclist
 * ============================================================
 */

static int emit_procs_json_body(const char *pat, int indent)
{
    PROCESSINFOLIST *plist = get_pinfolist_local();
    if (plist == NULL)
    {
        return 0;
    }

    char pad[32];
    {
        int n = indent;
        if (n >= (int) sizeof(pad))
        {
            n = (int) sizeof(pad) - 1;
        }
        memset(pad, ' ', (size_t) n);
        pad[n] = '\0';
    }

    char pdname[256];
    get_procdir_local(pdname, sizeof(pdname));

    int count = 0;
    for (int pi = 0; pi < PROCESSINFOLISTSIZE; pi++)
    {
        if (!plist->active[pi])
        {
            continue;
        }
        const char *pname = plist->pnamearray[pi];
        if (pat != NULL && fnmatch(pat, pname, 0) != 0)
        {
            continue;
        }

        pid_t        fpid   = plist->PIDarray[pi];
        int          pfd    = -1;
        PROCESSINFO *pi_shm = (fpid > 0) ? open_proc_shm_local(pdname, pname, fpid, &pfd) : NULL;
        if (pi_shm == NULL && (fpid <= 0 || (kill(fpid, 0) != 0 && errno == ESRCH)))
        {
            continue;
        }
        const char *state = proc_status_str(pi_shm, fpid);
        double      freq  = 0.0;

        if (pi_shm != NULL)
        {
            if (pi_shm->dtmedian_iter_ns > 0)
            {
                freq = 1.0e9 / (double) pi_shm->dtmedian_iter_ns;
            }
            munmap(pi_shm, sizeof(PROCESSINFO));
            close(pfd);
        }

        char nesc[256];
        char sesc[64];
        json_escape_str(nesc, sizeof(nesc), pname);
        json_escape_str(sesc, sizeof(sesc), state);

        if (count > 0)
        {
            printf(",\n");
        }
        printf("%s{\n", pad);
        printf("%s  \"name\": \"%s\",\n", pad, nesc);
        printf("%s  \"state\": \"%s\",\n", pad, sesc);
        printf("%s  \"status\": \"%s\",\n", pad, sesc);
        printf("%s  \"pid\": %d,\n", pad, (int) fpid);
        printf("%s  \"freq_hz\": %g\n", pad, freq);
        printf("%s}", pad);
        count++;
    }
    return count;
}

/* ============================================================
 *  milkquery & @ls
 * ============================================================
 */

static int emit_seqs_json_body(const char *pat, int indent)
{
    char pad[32];
    {
        int n = indent;
        if (n >= (int) sizeof(pad))
        {
            n = (int) sizeof(pad) - 1;
        }
        memset(pad, ' ', (size_t) n);
        pad[n] = '\0';
    }

    DIR *d = opendir(get_shmdir_local());
    if (d == NULL)
    {
        return 0;
    }

    int            count = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL)
    {
        size_t dlen = strlen(de->d_name);
        if (dlen <= 12 || strncmp(de->d_name, "milkseq.", 8) != 0 ||
            strcmp(de->d_name + dlen - 4, ".shm") != 0 || (dlen - 12) >= 256)
        {
            continue;
        }
        char qname[256];
        memcpy(qname, de->d_name + 8, dlen - 12);
        qname[dlen - 12] = '\0';

        if (pat != NULL && fnmatch(pat, qname, 0) != 0)
        {
            continue;
        }

        MILKSEQ_STATE *seqst = milkseq_connect(qname);
        if (seqst == NULL)
        {
            continue;
        }

        const char *st = "IDLE";
        if (seqst->status & MILKSEQ_STATUS_ERROR)
        {
            st = "ERROR";
        }
        else if (seqst->status & MILKSEQ_STATUS_STOPPING)
        {
            st = "STOPPING";
        }
        else if (seqst->status & MILKSEQ_STATUS_RUNNING)
        {
            st = "RUNNING";
        }

        char nesc[256];
        json_escape_str(nesc, sizeof(nesc), qname);
        if (count > 0)
        {
            printf(",\n");
        }
        printf("%s{\n", pad);
        printf("%s  \"name\": \"%s\",\n", pad, nesc);
        printf("%s  \"status\": \"%s\",\n", pad, st);
        printf("%s  \"pid\": %d\n", pad, (int) seqst->pid);
        printf("%s}", pad);
        count++;
        milkseq_disconnect(seqst);
    }
    closedir(d);
    return count;
}

static int emit_local_json_body(const char *pat, int indent)
{
    if (MILK_LOCAL_IMAGES == NULL)
    {
        return 0;
    }

    char pad[32];
    {
        int n = indent;
        if (n >= (int) sizeof(pad))
        {
            n = (int) sizeof(pad) - 1;
        }
        memset(pad, ' ', (size_t) n);
        pad[n] = '\0';
    }

    int count = 0;
    for (long i = 0; i < MILK_NB_LOCAL_IMAGES; i++)
    {
        if (MILK_LOCAL_IMAGES[i].used != 1 || MILK_LOCAL_IMAGES[i].name[0] == '\0' ||
            strncmp(MILK_LOCAL_IMAGES[i].name, "_tmpcalc", 8) == 0 ||
            MILK_LOCAL_IMAGES[i].md == NULL || MILK_LOCAL_IMAGES[i].md->shared)
        {
            continue;
        }
        const char *iname = MILK_LOCAL_IMAGES[i].name;
        if (pat != NULL && fnmatch(pat, iname, 0) != 0)
        {
            continue;
        }

        IMAGE_METADATA *md         = MILK_LOCAL_IMAGES[i].md;
        const char     *dtype_name = ImageStreamIO_typename(md->datatype);
        if (dtype_name == NULL)
        {
            dtype_name = "?";
        }
        char nesc[256];
        char tesc[64];
        json_escape_str(nesc, sizeof(nesc), iname);
        json_escape_str(tesc, sizeof(tesc), dtype_name);

        if (count > 0)
        {
            printf(",\n");
        }
        printf("%s{\n", pad);
        printf("%s  \"name\": \"%s\",\n", pad, nesc);
        printf("%s  \"naxis\": %u,\n", pad, md->naxis);
        printf("%s  \"size\": [", pad);
        for (int ax = 0; ax < md->naxis; ax++)
        {
            if (ax > 0)
            {
                printf(", ");
            }
            printf("%u", md->size[ax]);
        }
        printf("],\n");
        printf("%s  \"type\": \"%s\",\n", pad, tesc);
        printf("%s  \"loc\": \"RAM\"\n", pad);
        printf("%s}", pad);
        count++;
    }
    return count;
}

/** ls_ns_arg() - namespace named by an `@ls` argument ("s", "streams", "s.cam*"); its pattern in *@pat. */
static milk_ns_kind_t ls_ns_arg(const char *p, const char **pat)
{
    *pat = NULL;
    if (strcmp(p, "streams") == 0)
    {
        return MILK_NS_STREAM;
    }
    if (strcmp(p, "procs") == 0)
    {
        return MILK_NS_PROC;
    }
    milk_ns_kind_t k = milk_entity_ns_lookup(p, strlen(p));
    if (k != MILK_NS_NONE)
    {
        return k;
    }
    const char *rest = p;
    k                = milk_entity_ns_kind(p, &rest);
    if (k != MILK_NS_NONE)
    {
        *pat = (*rest != '\0') ? rest : "*";
    }
    return k;
}

/**
 * @brief Unified entity discovery command (`@ls`).
 *
 * Supports:
 *   `@ls`                        (all active entities across s.*, f.*, p.*, q.*)
 *   `@ls s.*` / `@ls f.*` / `@ls p.*` / `@ls q.*` / `@ls l.*`
 *   `@ls [-l] [-q] [--json] [--local] [pattern]`
 */
int milk_ls_exec(int argc, char **argv)
{
    int         longmode   = 0;
    int         quietmode  = 0;
    int         jsonmode   = 0;
    int         want_local = 0;
    const char *raw_pat    = NULL;

    for (int i = 0; i < argc; i++)
    {
        const char *tok = argv[i];
        if (tok == NULL)
        {
            continue;
        }
        if (strcmp(tok, "-l") == 0 || strcmp(tok, "--long") == 0)
        {
            longmode = 1;
        }
        else if (strcmp(tok, "-1") == 0)
        {
            longmode = 0;
        }
        else if (strcmp(tok, "-q") == 0 || strcmp(tok, "--quiet") == 0)
        {
            quietmode = 1;
        }
        else if (strcmp(tok, "--json") == 0 || strcmp(tok, "-j") == 0)
        {
            jsonmode = 1;
        }
        else if (strcmp(tok, "--local") == 0)
        {
            want_local = 1;
        }
        else if (strcmp(tok, "-h") == 0 || strcmp(tok, "--help") == 0)
        {
            printf("Usage: @ls [-l] [-q] [--json] [--local] "
                   "[s.<pat>|f.<pat>|p.<pat>|q.<pat>|l.<pat>|<pat>]\n");
            return 0;
        }
        else if (raw_pat == NULL)
        {
            raw_pat = tok;
        }
    }

    int         do_streams = 0;
    int         do_fps     = 0;
    int         do_procs   = 0;
    int         do_seqs    = 0;
    int         do_local   = 0;
    int         multi_ns   = 0;
    const char *pat        = "*";

    if (raw_pat == NULL)
    {
        if (want_local)
        {
            do_local = 1;
        }
        else
        {
            do_streams = 1;
            do_fps     = 1;
            do_procs   = 1;
            do_seqs    = 1;
            multi_ns   = 1;
        }
    }
    else
    {
        const char    *p     = raw_pat;
        const char    *nspat = NULL;
        milk_ns_kind_t nsk   = MILK_NS_NONE;
        while (*p == '@')
        {
            p++;
        }
        if (strncmp(p, "S:", 2) == 0)
        {
            do_streams = 1;
            pat        = (p[2] != '\0') ? (p + 2) : "*";
        }
        else if (strncmp(p, "L:", 2) == 0)
        {
            do_local = 1;
            pat      = (p[2] != '\0') ? (p + 2) : "*";
        }
        else if ((nsk = ls_ns_arg(p, &nspat)) != MILK_NS_NONE)
        {
            do_streams |= (nsk == MILK_NS_STREAM);
            do_local |= (nsk == MILK_NS_LOCAL);
            do_fps |= (nsk == MILK_NS_FPS);
            do_procs |= (nsk == MILK_NS_PROC);
            do_seqs |= (nsk == MILK_NS_SEQ);
            if (nspat != NULL)
            {
                pat = nspat;
            }
        }
        else if (want_local)
        {
            do_local = 1;
            pat      = p;
        }
        else
        {
            do_streams = 1;
            do_fps     = 1;
            do_procs   = 1;
            do_seqs    = 1;
            multi_ns   = 1;
            pat        = p;
        }
    }

    if (jsonmode)
    {
        if (multi_ns)
        {
            printf("{\n");
            printf("  \"fps\": [\n");
            emit_fps_json_body(pat, 4);
            printf("\n  ],\n");
            printf("  \"streams\": [\n");
            emit_streams_json_body(pat, 4);
            printf("\n  ],\n");
            printf("  \"processes\": [\n");
            emit_procs_json_body(pat, 4);
            printf("\n  ]\n");
            printf("}\n");
        }
        else if (do_streams)
        {
            printf("[\n");
            emit_streams_json_body(pat, 2);
            printf("\n]\n");
        }
        else if (do_fps)
        {
            printf("[\n");
            emit_fps_json_body(pat, 2);
            printf("\n]\n");
        }
        else if (do_procs)
        {
            printf("[\n");
            emit_procs_json_body(pat, 2);
            printf("\n]\n");
        }
        else if (do_seqs)
        {
            printf("[\n");
            emit_seqs_json_body(pat, 2);
            printf("\n]\n");
        }
        else if (do_local)
        {
            printf("[\n");
            emit_local_json_body(pat, 2);
            printf("\n]\n");
        }
        fflush(stdout);
        return 0;
    }

    int show_prefix = (multi_ns && !quietmode);

    if (do_streams)
    {
        const char *shmdir = get_shmdir_local();
        DIR        *d      = opendir(shmdir);
        if (d != NULL)
        {
            struct dirent *de;
            while ((de = readdir(d)) != NULL)
            {
                char *sfx = strstr(de->d_name, ".im.shm");
                if (sfx == NULL || strstr(de->d_name, ".fps.shm") != NULL)
                {
                    continue;
                }
                char   sname[256];
                size_t nlen = (size_t) (sfx - de->d_name);
                if (nlen >= sizeof(sname))
                {
                    continue;
                }
                memcpy(sname, de->d_name, nlen);
                sname[nlen] = '\0';
                if (strncmp(sname, "_tmpcalc", 8) == 0 || fnmatch(pat, sname, 0) != 0)
                {
                    continue;
                }
                char dname[280];
                snprintf(dname, sizeof(dname), "%s%s", show_prefix ? "s." : "", sname);
                if (!longmode)
                {
                    printf("%s\n", dname);
                }
                else
                {
                    IMAGE img;
                    memset(&img, 0, sizeof(IMAGE));
                    if (ImageStreamIO_openIm(&img, sname) == IMAGESTREAMIO_SUCCESS &&
                        img.md != NULL)
                    {
                        char szstr[64];
                        format_img_dims(img.md->naxis, img.md->size, szstr, sizeof(szstr));
                        printf("%-24s %-12s %-7s cnt0=%lu\n", dname, szstr,
                               ImageStreamIO_typename(img.md->datatype),
                               (unsigned long) img.md->cnt0);
                        ImageStreamIO_closeIm(&img);
                    }
                    else
                    {
                        printf("%-24s UNAVAIL\n", dname);
                    }
                }
            }
            closedir(d);
        }
    }

    if (do_local && MILK_LOCAL_IMAGES != NULL)
    {
        for (long i = 0; i < MILK_NB_LOCAL_IMAGES; i++)
        {
            if (MILK_LOCAL_IMAGES[i].used != 1 || MILK_LOCAL_IMAGES[i].name[0] == '\0' ||
                strncmp(MILK_LOCAL_IMAGES[i].name, "_tmpcalc", 8) == 0 ||
                MILK_LOCAL_IMAGES[i].md == NULL || MILK_LOCAL_IMAGES[i].md->shared)
            {
                continue;
            }
            const char *iname = MILK_LOCAL_IMAGES[i].name;
            if (fnmatch(pat, iname, 0) != 0)
            {
                continue;
            }
            char dname[280];
            snprintf(dname, sizeof(dname), "%s%s", show_prefix ? "l." : "", iname);
            if (!longmode)
            {
                printf("%s\n", dname);
            }
            else
            {
                IMAGE_METADATA *md = MILK_LOCAL_IMAGES[i].md;
                char            szstr[64];
                format_img_dims(md->naxis, md->size, szstr, sizeof(szstr));
                printf("%-24s %-12s %-7s RAM\n", dname, szstr,
                       ImageStreamIO_typename(md->datatype));
            }
        }
    }

    if (do_fps)
    {
        char shmdname[STRINGMAXLEN_SHMDIRNAME];
        function_parameter_struct_shmdirname(shmdname);
        DIR *d = opendir(shmdname);
        if (d != NULL)
        {
            struct dirent *de;
            while ((de = readdir(d)) != NULL)
            {
                char *sfx = strstr(de->d_name, ".fps.shm");
                if (sfx == NULL)
                {
                    continue;
                }
                char   fpsname[STRINGMAXLEN_FPS_NAME];
                size_t nlen = (size_t) (sfx - de->d_name);
                if (nlen >= sizeof(fpsname))
                {
                    continue;
                }
                memcpy(fpsname, de->d_name, nlen);
                fpsname[nlen] = '\0';
                if (fnmatch(pat, fpsname, 0) != 0)
                {
                    continue;
                }
                char dname[280];
                snprintf(dname, sizeof(dname), "%s%s", show_prefix ? "f." : "", fpsname);
                if (!longmode)
                {
                    printf("%s\n", dname);
                }
                else
                {
                    FPS fps;
                    fps.SMfd = -1;
                    if (fps_connect(fpsname, &fps, FPSCONNECT_SIMPLE) != -1 && fps.md != NULL)
                    {
                        printf("%-24s %-12s %s\n", dname, fps_status_str(fps.md->status),
                               fps.md->description);
                        fps_disconnect(&fps);
                    }
                    else
                    {
                        printf("%-24s %-12s\n", dname, "UNAVAIL");
                    }
                }
            }
            closedir(d);
        }
    }

    if (do_procs)
    {
        PROCESSINFOLIST *plist = get_pinfolist_local();
        if (plist != NULL)
        {
            char pdname[256];
            get_procdir_local(pdname, sizeof(pdname));
            for (int pi = 0; pi < PROCESSINFOLISTSIZE; pi++)
            {
                if (!plist->active[pi])
                {
                    continue;
                }
                const char *pname = plist->pnamearray[pi];
                if (fnmatch(pat, pname, 0) != 0)
                {
                    continue;
                }
                pid_t        fpid = plist->PIDarray[pi];
                int          pfd  = -1;
                PROCESSINFO *pi_shm =
                    (fpid > 0) ? open_proc_shm_local(pdname, pname, fpid, &pfd) : NULL;
                if (pi_shm == NULL && (fpid <= 0 || (kill(fpid, 0) != 0 && errno == ESRCH)))
                {
                    continue;
                }
                char dname[280];
                snprintf(dname, sizeof(dname), "%s%s", show_prefix ? "p." : "", pname);
                if (!longmode)
                {
                    if (pi_shm != NULL)
                    {
                        munmap(pi_shm, sizeof(PROCESSINFO));
                        close(pfd);
                    }
                    printf("%s\n", dname);
                }
                else
                {
                    const char *state = proc_status_str(pi_shm, fpid);
                    double      freq  = 0.0;
                    if (pi_shm != NULL)
                    {
                        if (pi_shm->dtmedian_iter_ns > 0)
                        {
                            freq = 1.0e9 / (double) pi_shm->dtmedian_iter_ns;
                        }
                        munmap(pi_shm, sizeof(PROCESSINFO));
                        close(pfd);
                    }
                    printf("%-24s %-8s %8.1f Hz  pid=%d\n", dname, state, freq, (int) fpid);
                }
            }
        }
    }

    if (do_seqs)
    {
        DIR *d = opendir(get_shmdir_local());
        if (d != NULL)
        {
            struct dirent *de;
            while ((de = readdir(d)) != NULL)
            {
                size_t dlen = strlen(de->d_name);
                if (dlen <= 12 || strncmp(de->d_name, "milkseq.", 8) != 0 ||
                    strcmp(de->d_name + dlen - 4, ".shm") != 0 || (dlen - 12) >= 256)
                {
                    continue;
                }
                char qname[256];
                memcpy(qname, de->d_name + 8, dlen - 12);
                qname[dlen - 12] = '\0';
                if (fnmatch(pat, qname, 0) != 0)
                {
                    continue;
                }
                char dname[280];
                snprintf(dname, sizeof(dname), "%s%s", show_prefix ? "q." : "", qname);
                if (!longmode)
                {
                    printf("%s\n", dname);
                }
                else
                {
                    MILKSEQ_STATE *seqst = milkseq_connect(qname);
                    if (seqst != NULL)
                    {
                        const char *st = "IDLE";
                        if (seqst->status & MILKSEQ_STATUS_ERROR)
                        {
                            st = "ERROR";
                        }
                        else if (seqst->status & MILKSEQ_STATUS_STOPPING)
                        {
                            st = "STOPPING";
                        }
                        else if (seqst->status & MILKSEQ_STATUS_RUNNING)
                        {
                            st = "RUNNING";
                        }
                        uint32_t tot = seqst->NBtasks_active + seqst->NBtasks_completed;
                        printf("%-24s %-10s %u/%u  pid=%d\n", dname, st, seqst->NBtasks_completed,
                               tot, (int) seqst->pid);
                        milkseq_disconnect(seqst);
                    }
                    else
                    {
                        printf("%-24s UNAVAIL\n", dname);
                    }
                }
            }
            closedir(d);
        }
    }

    fflush(stdout);
    return 0;
}

/* ============================================================
 *  Process Control & Server Interceptors
 * ============================================================
 */

/**
 * @brief List SHM streams matching a wildcard pattern (`listim <pat>`)
 */
void cli_listim_glob(const char *pat)
{
    if (pat == NULL || pat[0] == '\0')
    {
        return;
    }
    char shmglob[512];
    snprintf(shmglob, sizeof(shmglob), "%s.im.shm", pat);
    DIR *dp = opendir(get_shmdir_local());
    if (dp != NULL)
    {
        struct dirent *de;
        int            count = 0;
        while ((de = readdir(dp)) != NULL)
        {
            if (fnmatch(shmglob, de->d_name, 0) == 0)
            {
                char nm[256];
                strncpy(nm, de->d_name, sizeof(nm) - 1);
                nm[sizeof(nm) - 1] = '\0';
                char *sfx          = strstr(nm, ".im.shm");
                if (sfx != NULL)
                {
                    *sfx = '\0';
                }
                if (strncmp(nm, "_tmpcalc", 8) == 0)
                {
                    continue;
                }
                printf("  %s\n", nm);
                count++;
            }
        }
        closedir(dp);
        printf("%d stream(s) matched\n", count);
    }
}
