/**
 * @file CLIcore_script_cmd_inspect.c
 *
 * @brief CLI commands for system inspection and
 *        JSON introspection.
 *
 * Implements commands that enumerate and expose the
 * milk runtime state as both human-readable tables
 * and machine-readable JSON:
 *
 *   fpslist    — list live FPS instances
 *   fpsdump    — dump all params of one FPS
 *   streamlist — enumerate SHM streams
 *   proclist   — enumerate active processes
 *   milkquery  — unified JSON snapshot
 *
 * JSON output helpers (json_escape_str,
 * emit_fps_json_body, emit_streams_json_body,
 * emit_procs_json_body) are file-static; the
 * command entry-points are declared in
 * CLIcore_script.h.
 */

#include <dirent.h>
#include <fnmatch.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "CLIcore.h"
#include "CLIcore_script.h"
#include "ImageStreamIO/ImageStreamIO.h"

#include "fps_connect.h"
#include "fps_disconnect.h"
#include "fps_shmdirname.h"
#include "fps_printparameter_valuestring.h"

/* processinfo functions — linked via milkprocessinfo */
extern PROCESSINFO *processinfo_shm_link(const char *pname, int *fd);
extern errno_t      processinfo_procdirname(char *procdname);


/* ============================================================
 *  JSON string helper
 * ============================================================
 */

/**
 * json_escape_str - write JSON-escaped string to buf
 * @buf:   destination buffer
 * @bufsz: size of @buf in bytes
 * @src:   NUL-terminated source string
 *
 * Escapes backslash, double-quote, and all ASCII
 * control characters so the result is safe to
 * embed inside a JSON string literal.
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

/**
 * emit_fps_json_body - emit FPS entries as JSON
 * @pat:    optional glob pattern (NULL = all)
 * @indent: number of spaces for indentation
 *
 * Scans the SHM directory for *.fps.shm files,
 * connecting to each to read status and description.
 * Emits the JSON array body (without outer [ ]) for
 * all matching FPS instances. Used by both fpslist
 * --json and milkquery.
 *
 * Returns the number of entries emitted.
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
 *
 * Scans the SHM directory for *.fps.shm files,
 * connects to each, and prints a summary table
 * showing name, status and description.
 *
 * Usage: fpslist [--json] [pattern]
 *   --json   JSON array of FPS metadata
 *   pattern  optional glob filter (e.g. "dm*")
 */
errno_t cli_cmd_fpslist(void)
{
    int         jsonmode = 0;
    const char *pat      = NULL;

    for (int a = 1; a < data.cmdNBarg; a++)
    {
        const char *tok = data.cmdargtoken[a].val.string;
        if (strcmp(tok, "--json") == 0)
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

    /* Table mode (default) */
    char shmdname[STRINGMAXLEN_SHMDIRNAME];
    function_parameter_struct_shmdirname(shmdname);

    printf("%-24s %-12s %s\n", "FPS NAME", "STATUS", "DESCRIPTION");
    printf("%-24s %-12s %s\n", "------------------------", "------------",
           "--------------------"
           "--------------------");

    DIR           *d;
    struct dirent *de;
    d = opendir(shmdname);
    if (d == NULL)
    {
        printf("Cannot open SHM dir: %s\n", shmdname);
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
 *
 * Connects to the named FPS and prints every active
 * parameter in plain key=value, tab-separated, or
 * JSON format.
 *
 * Usage: fpsdump [-t] [--json] <fpsname>
 *   -t      tab-separated: key\tTYPE\tvalue
 *   --json  JSON object with raw typed values
 */
errno_t cli_cmd_fpsdump(void)
{
    int tabmode     = 0;
    int jsonmode    = 0;
    int arg_fpsname = -1;

    if (data.cmdNBarg < 2)
    {
        printf("Usage: fpsdump [-t] [--json] "
               "<fpsname>\n");
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
        printf("fpsdump: -t and --json are "
               "mutually exclusive\n");
        return RETURN_FAILURE;
    }

    if (arg_fpsname < 0)
    {
        printf("Usage: fpsdump [-t] [--json] "
               "<fpsname>\n");
        return RETURN_FAILURE;
    }

    const char *fpsname = data.cmdargtoken[arg_fpsname].val.string;

    FPS fps;
    fps.SMfd = -1;
    int rc   = fps_connect(fpsname, &fps, FPSCONNECT_SIMPLE);
    if (rc == -1 || fps.md == NULL || fps.parray == NULL)
    {
        printf("fpsdump: cannot connect "
               "to FPS '%s'\n",
               fpsname);
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

/**
 * emit_streams_json_body - emit stream entries
 * @pat:    optional glob pattern (NULL = all)
 * @indent: indentation spaces
 *
 * Scans dcshmdir for *.im.shm files, opens each,
 * and emits JSON objects with name, naxis, size,
 * type, and cnt0 fields. Used by milkquery.
 *
 * Returns number of entries emitted.
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
 *
 * Scans SHM directory for *.im.shm files and lists
 * them in plain, long-form, or JSON format.
 *
 * Usage: streamlist [-l] [--json] [pattern]
 *   -l       long format: name size type cnt0
 *   --json   JSON array of stream metadata
 *   pattern  optional glob filter (e.g. "dm*")
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
                    printf("%-24s %-12s %-7s "
                           "cnt0=%lu\n",
                           sname, szstr, ImageStreamIO_typename(img.md->datatype),
                           (unsigned long) img.md->cnt0);
                }
                ImageStreamIO_closeIm(&img);
            }
            else
            {
                if (!jsonmode)
                {
                    printf("%-24s UNAVAIL\n", sname);
                }
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

/**
 * emit_procs_json_body - emit process entries
 * @indent: indentation spaces
 *
 * Iterates the processinfo list and emits JSON objects
 * with name, state, pid, and freq_hz fields for each
 * active process. Used by milkquery.
 *
 * Returns number of entries emitted.
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
 *
 * Iterates the processinfo list and prints active
 * process names in plain, long-form, or JSON format.
 *
 * Usage: proclist [-l] [--json]
 *   -l      long format: name  state  freq
 *   --json  JSON array of process metadata
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
        printf("proclist: processinfo "
               "not available\n");
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
 *
 * Emits a single JSON object containing selected
 * subsections (fps, streams, processes). With no
 * flags, all three sections are emitted.
 *
 * Usage: milkquery [--fps [pat]]
 *                  [--streams [pat]]
 *                  [--procs]
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
            fprintf(stderr,
                    "milkquery: unknown argument "
                    "'%s'\n",
                    tok);
            fprintf(stderr, "usage: milkquery "
                            "[--fps [pattern]] "
                            "[--streams [pattern]] "
                            "[--procs]\n");
            return RETURN_FAILURE;
        }
    }

    /* Default: emit all sections */
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
