// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_inspect.c
 *
 * @brief CLI commands for system inspection, JSON introspection, and process control.
 */

#include <dirent.h>
#include <fnmatch.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "CLIcore.h"
#include "CLIcore_script.h"
#include "ImageStreamIO/ImageStreamIO.h"

#include "fps_connect.h"
#include "fps_disconnect.h"
#include "fps_printparameter_valuestring.h"
#include "fps_shmdirname.h"

extern PROCESSINFO *processinfo_shm_link(const char *pname, int *fd);
extern errno_t      processinfo_procdirname(char *procdname);

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

        uint32_t    st    = fps.md->status;
        const char *ststr = "IDLE";
        if (st & FUNCTION_PARAMETER_STRUCT_STATUS_CONF)
        {
            ststr = "CONF_ON";
        }
        else if (st & FUNCTION_PARAMETER_STRUCT_STATUS_RUN)
        {
            ststr = "RUN";
        }

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

/**
 * @brief fpslist command — list live FPS instances
 */
errno_t cli_cmd_fpslist(void)
{
    int         quietmode = 0;
    int         jsonmode  = 0;
    const char *pat       = NULL;

    for (int a = 1; a < data.cmdNBarg; a++)
    {
        const char *tok = data.cmdargtoken[a].val.string;
        if (strcmp(tok, "-q") == 0 || strcmp(tok, "--quiet") == 0)
        {
            quietmode = 1;
        }
        else if (strcmp(tok, "--json") == 0)
        {
            jsonmode = 1;
        }
        else
        {
            pat = tok;
        }
    }

    if (jsonmode)
    {
        printf("[\n");
        emit_fps_json_body(pat, 2);
        printf("\n]\n");
        return RETURN_SUCCESS;
    }

    char shmdname[STRINGMAXLEN_SHMDIRNAME];
    function_parameter_struct_shmdirname(shmdname);

    int is_tty = isatty(STDOUT_FILENO);
    if (!quietmode && is_tty)
    {
        printf("%-24s %-12s %s\n", "FPS NAME", "STATUS", "DESCRIPTION");
        printf("%-24s %-12s %s\n", "------------------------", "------------",
               "----------------------------------------");
    }

    DIR           *d;
    struct dirent *de;
    d = opendir(shmdname);
    if (d == NULL)
    {
        fprintf(stderr, "Cannot open SHM dir: %s\n", shmdname);
        return RETURN_FAILURE;
    }

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

        if (quietmode)
        {
            printf("%s\n", fpsname);
            continue;
        }

        FPS fps;
        fps.SMfd = -1;
        int rc   = fps_connect(fpsname, &fps, FPSCONNECT_SIMPLE);
        if (rc == -1 || fps.md == NULL)
        {
            printf("%-24s %-12s %s\n", fpsname, "UNAVAIL", "");
            continue;
        }

        uint32_t st = fps.md->status;
        char     ststr[16];
        if (st & FUNCTION_PARAMETER_STRUCT_STATUS_RUN)
        {
            strncpy(ststr, "RUN", sizeof(ststr) - 1);
        }
        else if (st & FUNCTION_PARAMETER_STRUCT_STATUS_CONF)
        {
            strncpy(ststr, "CONF_ON", sizeof(ststr) - 1);
        }
        else
        {
            strncpy(ststr, "IDLE", sizeof(ststr) - 1);
        }
        ststr[sizeof(ststr) - 1] = '\0';

        printf("%-24s %-12s %s\n", fpsname, ststr, fps.md->description);

        fps_disconnect(&fps);
    }
    closedir(d);

    return RETURN_SUCCESS;
}

/* ============================================================
 *  fpsdump
 * ============================================================
 */

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

/* ============================================================
 *  streamlist
 * ============================================================
 */

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

    DIR *d = opendir(dcshmdir);
    if (d == NULL)
    {
        return 0;
    }

    int            count = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL)
    {
        char *sfx = strstr(de->d_name, ".im.shm");
        if (sfx == NULL)
        {
            continue;
        }
        if (strstr(de->d_name, ".fps.shm") != NULL)
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

/**
 * @brief streamlist command — enumerate SHM streams
 */
errno_t cli_cmd_streamlist(void)
{
    int         longmode = 0;
    int         jsonmode = 0;
    const char *pat      = NULL;
    int         argpos   = 1;

    for (int a = 1; a < data.cmdNBarg; a++)
    {
        const char *tok = data.cmdargtoken[a].val.string;
        if (strcmp(tok, "-l") == 0)
        {
            longmode = 1;
            argpos   = a + 1;
        }
        else if (strcmp(tok, "-q") == 0 || strcmp(tok, "--quiet") == 0)
        {
            longmode = 0;
            argpos   = a + 1;
        }
        else if (strcmp(tok, "--json") == 0)
        {
            jsonmode = 1;
            argpos   = a + 1;
        }
    }
    if (argpos < data.cmdNBarg)
    {
        pat = data.cmdargtoken[argpos].val.string;
    }

    DIR           *d;
    struct dirent *de;
    d = opendir(dcshmdir);
    if (d == NULL)
    {
        printf("Cannot open SHM dir: %s\n", dcshmdir);
        return RETURN_FAILURE;
    }

    if (jsonmode)
    {
        printf("[\n");
    }
    int first_json_item = 1;

    while ((de = readdir(d)) != NULL)
    {
        char *sfx = strstr(de->d_name, ".im.shm");
        if (sfx == NULL)
        {
            continue;
        }
        if (strstr(de->d_name, ".fps.shm") != NULL)
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

        if (pat != NULL && fnmatch(pat, sname, 0) != 0)
        {
            continue;
        }

        if (!longmode && !jsonmode)
        {
            printf("%s\n", sname);
        }
        else
        {
            IMAGE img;
            memset(&img, 0, sizeof(IMAGE));
            errno_t sret = ImageStreamIO_openIm(&img, sname);
            if (sret == IMAGESTREAMIO_SUCCESS && img.md != NULL)
            {
                if (jsonmode)
                {
                    char        nesc[256];
                    char        tesc[64];
                    const char *dtype_name = ImageStreamIO_typename(img.md->datatype);
                    if (dtype_name == NULL)
                    {
                        dtype_name = "?";
                    }
                    json_escape_str(nesc, sizeof(nesc), sname);
                    json_escape_str(tesc, sizeof(tesc), dtype_name);
                    if (!first_json_item)
                    {
                        printf(",\n");
                    }
                    first_json_item = 0;

                    printf("  {\n");
                    printf("    \"name\": \"%s\",\n", nesc);
                    printf("    \"naxis\": %u,\n", img.md->naxis);
                    printf("    \"size\": [");
                    for (int ax = 0; ax < img.md->naxis; ax++)
                    {
                        if (ax > 0)
                        {
                            printf(", ");
                        }
                        printf("%u", img.md->size[ax]);
                    }
                    printf("],\n");
                    printf("    \"type\": \"%s\",\n", tesc);
                    printf("    \"cnt0\": %lu\n", (unsigned long) img.md->cnt0);
                    printf("  }");
                }
                else
                {
                    char szstr[64];
                    if (img.md->naxis == 1)
                    {
                        snprintf(szstr, sizeof(szstr), "%u", img.md->size[0]);
                    }
                    else if (img.md->naxis == 2)
                    {
                        snprintf(szstr, sizeof(szstr), "%ux%u", img.md->size[0], img.md->size[1]);
                    }
                    else
                    {
                        snprintf(szstr, sizeof(szstr), "%ux%ux%u", img.md->size[0], img.md->size[1],
                                 img.md->size[2]);
                    }
                    printf("%-24s %-12s %-7s cnt0=%lu\n", sname, szstr,
                           ImageStreamIO_typename(img.md->datatype), (unsigned long) img.md->cnt0);
                }
                ImageStreamIO_closeIm(&img);
            }
            else if (!jsonmode)
            {
                printf("%-24s UNAVAIL\n", sname);
            }
        }
    }

    if (jsonmode)
    {
        printf("\n]\n");
    }
    closedir(d);

    return RETURN_SUCCESS;
}

/* ============================================================
 *  proclist
 * ============================================================
 */

static int emit_procs_json_body(int indent)
{
    if (pinfolist == NULL)
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
    for (int pi = 0; pi < PROCESSINFOLISTSIZE; pi++)
    {
        if (!pinfolist->active[pi])
        {
            continue;
        }

        pid_t       fpid  = pinfolist->PIDarray[pi];
        const char *state = "UNKNOWN";
        double      freq  = 0.0;

        if (fpid > 0)
        {
            char pfn[512];
            char pdname[256];
            processinfo_procdirname(pdname);
            snprintf(pfn, sizeof(pfn), "%s/proc.%d.shm", pdname, (int) fpid);
            int          pfd    = -1;
            PROCESSINFO *pi_shm = processinfo_shm_link(pfn, &pfd);
            if (pi_shm != MAP_FAILED && pi_shm != NULL)
            {
                switch (pi_shm->CTRLval)
                {
                case PROCESSINFO_CTRLVAL_RUN:
                    state = "ACTIVE";
                    break;
                case PROCESSINFO_CTRLVAL_PAUSE:
                    state = "PAUSED";
                    break;
                case PROCESSINFO_CTRLVAL_EXIT:
                    state = "STOPPED";
                    break;
                default:
                    state = "OTHER";
                    break;
                }
                if (pi_shm->dtmedian_iter_ns > 0)
                {
                    freq = 1.0e9 / (double) pi_shm->dtmedian_iter_ns;
                }
                munmap(pi_shm, sizeof(PROCESSINFO));
                close(pfd);
            }
            else if (pfd >= 0)
            {
                close(pfd);
            }
        }

        char nesc[256];
        char sesc[64];
        json_escape_str(nesc, sizeof(nesc), pinfolist->pnamearray[pi]);
        json_escape_str(sesc, sizeof(sesc), state);

        if (count > 0)
        {
            printf(",\n");
        }
        printf("%s{\n", pad);
        printf("%s  \"name\": \"%s\",\n", pad, nesc);
        printf("%s  \"state\": \"%s\",\n", pad, sesc);
        printf("%s  \"pid\": %d,\n", pad, (int) fpid);
        printf("%s  \"freq_hz\": %g\n", pad, freq);
        printf("%s}", pad);
        count++;
    }
    return count;
}

/**
 * @brief proclist command — list active processes
 */
errno_t cli_cmd_proclist(void)
{
    int longmode = 0;
    int jsonmode = 0;
    for (int a = 1; a < data.cmdNBarg; a++)
    {
        const char *tok = data.cmdargtoken[a].val.string;
        if (strcmp(tok, "-l") == 0)
        {
            longmode = 1;
        }
        else if (strcmp(tok, "--json") == 0)
        {
            jsonmode = 1;
        }
    }

    if (pinfolist == NULL)
    {
        printf("proclist: processinfo not available\n");
        return RETURN_FAILURE;
    }

    if (jsonmode)
    {
        printf("[\n");
    }
    int first_json_item = 1;

    for (int pi = 0; pi < PROCESSINFOLISTSIZE; pi++)
    {
        if (!pinfolist->active[pi])
        {
            continue;
        }

        if (!longmode && !jsonmode)
        {
            printf("%s\n", pinfolist->pnamearray[pi]);
        }
        else
        {
            pid_t       fpid  = pinfolist->PIDarray[pi];
            const char *state = "UNKNOWN";
            double      freq  = 0.0;

            if (fpid > 0)
            {
                char pfn[512];
                char pdname[256];
                processinfo_procdirname(pdname);
                snprintf(pfn, sizeof(pfn), "%s/proc.%d.shm", pdname, (int) fpid);
                int          pfd    = -1;
                PROCESSINFO *pi_shm = processinfo_shm_link(pfn, &pfd);
                if (pi_shm != MAP_FAILED && pi_shm != NULL)
                {
                    switch (pi_shm->CTRLval)
                    {
                    case PROCESSINFO_CTRLVAL_RUN:
                        state = "ACTIVE";
                        break;
                    case PROCESSINFO_CTRLVAL_PAUSE:
                        state = "PAUSED";
                        break;
                    case PROCESSINFO_CTRLVAL_EXIT:
                        state = "STOPPED";
                        break;
                    default:
                        state = "OTHER";
                        break;
                    }
                    if (pi_shm->dtmedian_iter_ns > 0)
                    {
                        freq = 1.0e9 / (double) pi_shm->dtmedian_iter_ns;
                    }
                    munmap(pi_shm, sizeof(PROCESSINFO));
                    close(pfd);
                }
                else if (pfd >= 0)
                {
                    close(pfd);
                }
            }

            if (jsonmode)
            {
                char nesc[256];
                char sesc[64];
                json_escape_str(nesc, sizeof(nesc), pinfolist->pnamearray[pi]);
                json_escape_str(sesc, sizeof(sesc), state);
                if (!first_json_item)
                {
                    printf(",\n");
                }
                first_json_item = 0;
                printf("  {\n");
                printf("    \"name\": \"%s\",\n", nesc);
                printf("    \"state\": \"%s\",\n", sesc);
                printf("    \"pid\": %d,\n", (int) fpid);
                printf("    \"freq_hz\": %g\n", freq);
                printf("  }");
            }
            else
            {
                printf("%-24s %-8s %8.1f Hz\n", pinfolist->pnamearray[pi], state, freq);
            }
        }
    }

    if (jsonmode)
    {
        printf("\n]\n");
    }
    return RETURN_SUCCESS;
}

/* ============================================================
 *  milkquery
 * ============================================================
 */

/**
 * @brief milkquery — unified JSON snapshot
 */
errno_t cli_cmd_milkquery(void)
{
    int         do_fps     = 0;
    int         do_streams = 0;
    int         do_procs   = 0;
    const char *fps_pat    = NULL;
    const char *stream_pat = NULL;

    for (int a = 1; a < data.cmdNBarg; a++)
    {
        const char *tok = data.cmdargtoken[a].val.string;
        if (strcmp(tok, "--fps") == 0)
        {
            do_fps = 1;
            if (a + 1 < data.cmdNBarg && data.cmdargtoken[a + 1].val.string[0] != '-')
            {
                fps_pat = data.cmdargtoken[a + 1].val.string;
                a++;
            }
        }
        else if (strcmp(tok, "--streams") == 0)
        {
            do_streams = 1;
            if (a + 1 < data.cmdNBarg && data.cmdargtoken[a + 1].val.string[0] != '-')
            {
                stream_pat = data.cmdargtoken[a + 1].val.string;
                a++;
            }
        }
        else if (strcmp(tok, "--procs") == 0)
        {
            do_procs = 1;
        }
        else
        {
            fprintf(stderr, "milkquery: unknown argument '%s'\n", tok);
            fprintf(stderr, "usage: milkquery [--fps [pattern]] [--streams [pattern]] [--procs]\n");
            return RETURN_FAILURE;
        }
    }

    if (!do_fps && !do_streams && !do_procs)
    {
        do_fps     = 1;
        do_streams = 1;
        do_procs   = 1;
    }

    printf("{\n");
    int need_comma = 0;

    if (do_fps)
    {
        if (need_comma)
        {
            printf(",\n");
        }
        printf("  \"fps\": [\n");
        emit_fps_json_body(fps_pat, 4);
        printf("\n  ]");
        need_comma = 1;
    }

    if (do_streams)
    {
        if (need_comma)
        {
            printf(",\n");
        }
        printf("  \"streams\": [\n");
        emit_streams_json_body(stream_pat, 4);
        printf("\n  ]");
        need_comma = 1;
    }

    if (do_procs)
    {
        if (need_comma)
        {
            printf(",\n");
        }
        printf("  \"processes\": [\n");
        emit_procs_json_body(4);
        printf("\n  ]");
        need_comma = 1;
    }

    (void) need_comma;
    printf("\n}\n");
    return RETURN_SUCCESS;
}

/* ============================================================
 *  Process Control & Server Interceptors
 * ============================================================
 */

int cli_intercept_cmd_procctl(const char *p)
{
    if (starts_with_local(p, "procctl ") || starts_with_local(p, "procctl\t"))
    {
        const char *ap = p + 7;
        while (*ap == ' ' || *ap == '\t')
        {
            ap++;
        }
        char pname[256];
        int  nlen = 0;
        while (*ap && *ap != ' ' && *ap != '\t' && nlen < 255)
        {
            pname[nlen++] = *ap++;
        }
        pname[nlen] = '\0';
        while (*ap == ' ' || *ap == '\t')
        {
            ap++;
        }
        int ctrlval = -1;
        if (strncmp(ap, "run", 3) == 0)
        {
            ctrlval = PROCESSINFO_CTRLVAL_RUN;
        }
        else if (strncmp(ap, "pause", 5) == 0)
        {
            ctrlval = PROCESSINFO_CTRLVAL_PAUSE;
        }
        else if (strncmp(ap, "step", 4) == 0)
        {
            ctrlval = PROCESSINFO_CTRLVAL_INCR;
        }
        else if (strncmp(ap, "stop", 4) == 0 || strncmp(ap, "exit", 4) == 0)
        {
            ctrlval = PROCESSINFO_CTRLVAL_EXIT;
        }
        if (ctrlval < 0)
        {
            printf("procctl: unknown action '%s' (use run|pause|step|stop)\n", ap);
            return 1;
        }
        if (pinfolist != NULL)
        {
            pid_t fpid = 0;
            for (int pi = 0; pi < PROCESSINFOLISTSIZE; pi++)
            {
                if (pinfolist->active[pi] && strcmp(pinfolist->pnamearray[pi], pname) == 0)
                {
                    fpid = pinfolist->PIDarray[pi];
                    break;
                }
            }
            if (fpid > 0)
            {
                char pfn[512];
                char pdname[256];
                processinfo_procdirname(pdname);
                snprintf(pfn, sizeof(pfn), "%s/proc.%d.shm", pdname, (int) fpid);
                int          pfd = -1;
                PROCESSINFO *pi  = processinfo_shm_link(pfn, &pfd);
                if (pi != MAP_FAILED && pi != NULL)
                {
                    pi->CTRLval = ctrlval;
                    munmap(pi, sizeof(PROCESSINFO));
                    close(pfd);
                }
                else if (pfd >= 0)
                {
                    close(pfd);
                }
            }
            else
            {
                printf("procctl: process '%s' not found\n", pname);
            }
        }
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_procwait(const char *p)
{
    if (starts_with_local(p, "procwait ") || starts_with_local(p, "procwait\t"))
    {
        const char *ap = p + 8;
        while (*ap == ' ' || *ap == '\t')
        {
            ap++;
        }
        char pname[256];
        int  nlen = 0;
        while (*ap && *ap != ' ' && *ap != '\t' && nlen < 255)
        {
            pname[nlen++] = *ap++;
        }
        pname[nlen] = '\0';
        while (*ap == ' ' || *ap == '\t')
        {
            ap++;
        }
        int tgt = -1;
        if (strncasecmp(ap, "INIT", 4) == 0)
        {
            tgt = PROCESSINFO_LOOPSTAT_INIT;
        }
        else if (strncasecmp(ap, "ACTIVE", 6) == 0)
        {
            tgt = PROCESSINFO_LOOPSTAT_ACTIVE;
        }
        else if (strncasecmp(ap, "PAUSE", 5) == 0)
        {
            tgt = PROCESSINFO_LOOPSTAT_PAUSE;
        }
        else if (strncasecmp(ap, "STOP", 4) == 0)
        {
            tgt = PROCESSINFO_LOOPSTAT_STOP;
        }
        else if (strncasecmp(ap, "ERROR", 5) == 0)
        {
            tgt = PROCESSINFO_LOOPSTAT_ERROR;
        }
        else
        {
            tgt = (int) strtol(ap, NULL, 0);
        }
        while (*ap && *ap != ' ' && *ap != '\t')
        {
            ap++;
        }
        while (*ap == ' ' || *ap == '\t')
        {
            ap++;
        }
        double timeout = 30.0;
        if (*ap != '\0')
        {
            timeout = strtod(ap, NULL);
        }
        struct timespec slp;
        slp.tv_sec      = 0;
        slp.tv_nsec     = 100000000;
        double elapsed  = 0.0;
        cli_last_retval = 1;
        while (elapsed < timeout)
        {
            if (pinfolist != NULL)
            {
                for (int pi = 0; pi < PROCESSINFOLISTSIZE; pi++)
                {
                    if (pinfolist->active[pi] && strcmp(pinfolist->pnamearray[pi], pname) == 0)
                    {
                        pid_t fpid = pinfolist->PIDarray[pi];
                        char  pfn[512];
                        char  pdname[256];
                        processinfo_procdirname(pdname);
                        snprintf(pfn, sizeof(pfn), "%s/proc.%d.shm", pdname, (int) fpid);
                        int          pfd = -1;
                        PROCESSINFO *pii = processinfo_shm_link(pfn, &pfd);
                        if (pii != MAP_FAILED && pii != NULL)
                        {
                            if (pii->loopstat == tgt)
                            {
                                cli_last_retval = 0;
                            }
                            munmap(pii, sizeof(PROCESSINFO));
                            close(pfd);
                        }
                        else if (pfd >= 0)
                        {
                            close(pfd);
                        }
                        break;
                    }
                }
            }
            if (cli_last_retval == 0)
            {
                break;
            }
            nanosleep(&slp, NULL);
            elapsed += 0.1;
        }
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_procstat(const char *p)
{
    if (strcmp(p, "procstat") == 0 || starts_with_local(p, "procstat ") ||
        starts_with_local(p, "procstat\t"))
    {
        const char *ap = p + 8;
        while (*ap == ' ' || *ap == '\t')
        {
            ap++;
        }
        char filter[256];
        filter[0] = '\0';
        if (*ap != '\0')
        {
            strncpy(filter, ap, sizeof(filter) - 1);
            filter[sizeof(filter) - 1] = '\0';
        }
        if (pinfolist != NULL)
        {
            char pdname[256];
            processinfo_procdirname(pdname);
            for (int pi = 0; pi < PROCESSINFOLISTSIZE; pi++)
            {
                if (!pinfolist->active[pi])
                {
                    continue;
                }
                if (filter[0] != '\0' && strcmp(pinfolist->pnamearray[pi], filter) != 0)
                {
                    continue;
                }
                pid_t fpid = pinfolist->PIDarray[pi];
                char  pfn[512];
                snprintf(pfn, sizeof(pfn), "%s/proc.%d.shm", pdname, (int) fpid);
                int          pfd = -1;
                PROCESSINFO *pii = processinfo_shm_link(pfn, &pfd);
                if (pii == MAP_FAILED || pii == NULL)
                {
                    if (pfd >= 0)
                    {
                        close(pfd);
                    }
                    continue;
                }
                const char *stname = "UNKNOWN";
                switch (pii->loopstat)
                {
                case 0:
                    stname = "INIT";
                    break;
                case 1:
                    stname = "ACTIVE";
                    break;
                case 2:
                    stname = "PAUSED";
                    break;
                case 3:
                    stname = "STOPPED";
                    break;
                case 4:
                    stname = "ERROR";
                    break;
                case 5:
                    stname = "SPINNING";
                    break;
                case 6:
                    stname = "CRASHED";
                    break;
                }
                double hz = 0.0;
                if (pii->dtmedian_iter_ns > 0)
                {
                    hz = 1.0e9 / (double) pii->dtmedian_iter_ns;
                }
                double us = (double) pii->dtmedian_exec_ns / 1000.0;
                printf("name=%s\n"
                       "pid=%d\n"
                       "loopstat=%s\n"
                       "loopcnt=%ld\n"
                       "loopfreq_hz=%.1f\n"
                       "exectime_us=%.1f\n"
                       "rtprio=%d\n"
                       "ctrlval=%d\n"
                       "missedframes=%lu\n"
                       "tmux=%s\n",
                       pii->name, (int) pii->PID, stname, pii->loopcnt, hz, us, pii->RT_priority,
                       pii->CTRLval, (unsigned long) pii->triggermissedframe_cumul, pii->tmuxname);
                munmap(pii, sizeof(PROCESSINFO));
                close(pfd);
                if (filter[0] != '\0')
                {
                    break;
                }
                printf("---\n");
            }
        }
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_dpdigits(const char *p)
{
    const char *sp = strip_ws_local(p);
    if (strcmp(sp, "dpdigits") == 0)
    {
        printf("dpdigits = %d\n", cli_float_digits);
        return 1;
    }
    if (starts_with_local(sp, "dpdigits ") || starts_with_local(sp, "dpdigits\t"))
    {
        const char *ap = sp + 9;
        while (*ap == ' ' || *ap == '\t')
        {
            ap++;
        }
        int val = (int) strtol(ap, NULL, 10);
        if (val < 1)
        {
            val = 1;
        }
        if (val > 17)
        {
            val = 17;
        }
        cli_float_digits = val;
        printf("dpdigits = %d\n", cli_float_digits);
        return 1;
    }
    return 0;
}

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
    DIR *dp = opendir(dcshmdir[0] != '\0' ? dcshmdir : "/dev/shm");
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
                printf("  %s\n", nm);
                count++;
            }
        }
        closedir(dp);
        printf("%d stream(s) matched\n", count);
    }
}
