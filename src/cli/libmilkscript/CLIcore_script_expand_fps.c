// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_script_expand_fps.c
 *
 * @brief FPS, stream, procinfo, and sequencer
 *        @-property expansion.
 *
 * Implements the @token expansion system that
 * maps namespace-prefixed tokens in command-line
 * strings to real-time values from shared memory:
 *
 *   @fps.name.param    — FPS parameter read
 *   @fps.name.param=v  — FPS parameter write
 *   @s.name.prop       — ImageStreamIO metadata
 *   @proc.name.prop    — processinfo telemetry
 *   @seq.name.prop     — sequencer state
 *   @fpsname.param     — legacy FPS lookup
 *
 * Public API (declared in CLIcore_script.h):
 *   cli_expand_fpsvar()
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "CLIcore.h"
#include "CLIcore_script.h"
#include "CLIcore/cli_calc_parser.h"
#include "CLIcore_script_expand_internal.h"
#include "ImageStreamIO/ImageStreamIO.h"
#include "fpsseq.h"

#include "fps.h"
#include "fps_GetParamIndex.h"
#include "fps_connect.h"
#include "fps_printparameter_valuestring.h"

/* processinfo functions — linked via milkprocessinfo */
extern PROCESSINFO *processinfo_shm_link(const char *pname, int *fd);
extern errno_t      processinfo_procdirname(char *procdname);


/* ============================================================
 *  expand_fpsvar_write — write path helper
 * ============================================================
 */

/**
 * expand_fpsvar_write - handle @fps.param=value write
 * @line:   Full command line buffer
 * @i:      Current parse position (after '='), updated
 * @token:  "namespace.name.param" token already collected
 * @out:    Output buffer
 * @opos:   Output position (updated)
 * @maxlen: Output buffer size
 *
 * Parses the value string (quote/escape/paren aware),
 * recursively expands it, then dispatches to the
 * appropriate write handler:
 *   - @fps.name.param=v  → cli_fps_set_param()
 *   - @proc.name.ctrlval=v → processinfo CTRLval write
 */
static void expand_fpsvar_write(char       *line,
                                int        *i,
                                const char *token,
                                char       *out,
                                int        *opos,
                                int         maxlen)
{
    char valstr[512];
    int  vlen  = 0;
    int  depth = 0;
    int  in_sq = 0;
    int  in_dq = 0;
    int  esc   = 0;
    int  pos   = *i;

    while (line[pos] == ' ' || line[pos] == '\t')
    {
        pos++;
    }

    while (line[pos] != '\0' && vlen < 511)
    {
        char ch = line[pos];

        if (!esc)
        {
            if (ch == '\\')
            {
                esc            = 1;
                valstr[vlen++] = ch;
                pos++;
                continue;
            }
            if (!in_sq && !in_dq)
            {
                if (ch == '(')
                {
                    depth++;
                }
                else if (ch == ')' && depth > 0)
                {
                    depth--;
                }
                if (depth == 0 && (ch == ';' || ch == '\n'))
                {
                    break;
                }
            }
            if (!in_dq && ch == '\'')
            {
                in_sq = !in_sq;
            }
            else if (!in_sq && ch == '"')
            {
                in_dq = !in_dq;
            }
        }
        else
        {
            esc = 0;
        }
        valstr[vlen++] = ch;
        pos++;
    }

    while (vlen > 0 && (valstr[vlen - 1] == ' ' || valstr[vlen - 1] == '\t'))
    {
        vlen--;
    }
    valstr[vlen] = '\0';

    cli_expand_fpsvar(valstr, (int) sizeof(valstr));
    cli_expand_env(valstr, (int) sizeof(valstr));
    cli_expand_arith(valstr, (int) sizeof(valstr));

    char tcopy[512];
    strncpy(tcopy, token, sizeof(tcopy) - 1);
    tcopy[sizeof(tcopy) - 1] = '\0';

    char *dot1 = strchr(tcopy, '.');
    char *dot2 = dot1 ? strchr(dot1 + 1, '.') : NULL;

    if (dot1 && dot2)
    {
        *dot1            = '\0';
        *dot2            = '\0';
        const char *nsp  = tcopy;
        const char *name = dot1 + 1;
        const char *prop = dot2 + 1;

        if (strcmp(nsp, "fps") == 0)
        {
            cli_fps_set_param(name, prop, valstr);
        }
        else if (strcmp(nsp, "proc") == 0)
        {
            if (strcmp(prop, "ctrlval") == 0)
            {
                if (pinfolist == NULL)
                {
                    printf("@proc write: "
                           "process list "
                           "unavailable\n");
                    return;
                }
                int ctrlval_int = -1;
                if (strcmp(valstr, "run") == 0)
                {
                    ctrlval_int = PROCESSINFO_CTRLVAL_RUN;
                }
                else if (strcmp(valstr, "pause") == 0)
                {
                    ctrlval_int = PROCESSINFO_CTRLVAL_PAUSE;
                }
                else if (strcmp(valstr, "step") == 0)
                {
                    ctrlval_int = PROCESSINFO_CTRLVAL_INCR;
                }
                else if (strcmp(valstr, "stop") == 0 || strcmp(valstr, "exit") == 0)
                {
                    ctrlval_int = PROCESSINFO_CTRLVAL_EXIT;
                }
                else
                {
                    ctrlval_int = atoi(valstr);
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
                if (found_pid > 0)
                {
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
                    }
                    else if (pfd >= 0)
                    {
                        close(pfd);
                    }
                    else
                    {
                        printf("@proc write: "
                               "cannot map SHM "
                               "for '%s'\n",
                               name);
                    }
                }
                else
                {
                    printf("@proc write: "
                           "process '%s' "
                           "not found\n",
                           name);
                }
            }
        }
    }
    else if (dot1)
    {
        /* Legacy: @fpsname.param=val */
        *dot1 = '\0';
        cli_fps_set_param(tcopy, dot1 + 1, valstr);
    }

    /* Insert no-op ":" if this is the full cmd */
    if (*opos == 0)
    {
        if (*opos < maxlen - 1)
        {
            out[(*opos)++] = ':';
        }
        if (*opos < maxlen - 1)
        {
            out[(*opos)++] = ' ';
        }
    }
    *i = pos;
}


/* ============================================================
 *  Typed namespace expanders
 * ============================================================
 */

/**
 * expand_fpsvar_seq - expand @seq.NAME.prop tokens
 * @pname:  "NAME.prop" (after the "seq." prefix)
 * @out:    Output buffer
 * @opos:   Output position (updated)
 * @maxlen: Output buffer size
 *
 * Connects to the sequencer SHM for the given name
 * and reads status, tasks, errors, pid, or completed.
 * Returns 1 if expanded, 0 if no second dot found.
 */
static int expand_fpsvar_seq(char *pname, char *out, int *opos, int maxlen)
{
    char *dot2 = strchr(pname, '.');
    if (dot2 == NULL)
    {
        return 0;
    }
    *dot2               = '\0';
    const char *seqname = pname;
    const char *seqprop = dot2 + 1;

    MILKSEQ_STATE *seqst = milkseq_connect(seqname);
    if (seqst == NULL)
    {
        return 1;
    }

    char vstr[512];
    vstr[0] = '\0';

    if (strcmp(seqprop, "status") == 0)
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
        strncpy(vstr, s, sizeof(vstr) - 1);
        vstr[sizeof(vstr) - 1] = '\0';
    }
    else if (strcmp(seqprop, "tasks") == 0)
    {
        snprintf(vstr, sizeof(vstr), "%u", seqst->NBtasks_active);
    }
    else if (strcmp(seqprop, "errors") == 0)
    {
        snprintf(vstr, sizeof(vstr), "%u", seqst->error_count);
    }
    else if (strcmp(seqprop, "pid") == 0)
    {
        snprintf(vstr, sizeof(vstr), "%d", (int) seqst->pid);
    }
    else if (strcmp(seqprop, "completed") == 0)
    {
        snprintf(vstr, sizeof(vstr), "%u", seqst->NBtasks_completed);
    }

    milkseq_disconnect(seqst);

    int vlen  = (int) strlen(vstr);
    int avail = maxlen - 1 - *opos;
    int clen  = vlen < avail ? vlen : avail;
    memcpy(out + *opos, vstr, (size_t) clen);
    *opos += clen;
    return 1;
}

/**
 * expand_fpsvar_procinfo - expand process info props
 * @fpsname: Process name to look up in pinfolist
 * @pname:   Property key (pid, loopstat, loopfreq…)
 * @out:     Output buffer
 * @opos:    Output position (updated)
 * @maxlen:  Output buffer size
 *
 * Scans pinfolist for a matching process name, maps
 * its SHM, reads the requested property, and writes
 * the result into the output buffer.
 *
 * Returns 1 if expanded (even if property unknown),
 * 0 if process was not found.
 */
static int expand_fpsvar_procinfo(const char *fpsname,
                                  const char *pname,
                                  char       *out,
                                  int        *opos,
                                  int         maxlen)
{
    if (pinfolist == NULL)
    {
        return 0;
    }

    pid_t found_pid = 0;
    for (int pi = 0; pi < PROCESSINFOLISTSIZE; pi++)
    {
        if (pinfolist->active[pi] && strcmp(pinfolist->pnamearray[pi], fpsname) == 0)
        {
            found_pid = pinfolist->PIDarray[pi];
            break;
        }
    }
    if (found_pid <= 0)
    {
        return 0;
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
        return 0;
    }

    char vstr[512];
    vstr[0] = '\0';

    if (strcmp(pname, "pid") == 0)
    {
        snprintf(vstr, sizeof(vstr), "%d", (int) pi->PID);
    }
    else if (strcmp(pname, "loopstat") == 0)
    {
        snprintf(vstr, sizeof(vstr), "%d", pi->loopstat);
    }
    else if (strcmp(pname, "loopcnt") == 0)
    {
        snprintf(vstr, sizeof(vstr), "%ld", pi->loopcnt);
    }
    else if (strcmp(pname, "loopfreq") == 0)
    {
        double hz = 0.0;
        if (pi->dtmedian_iter_ns > 0)
        {
            hz = 1.0e9 / (double) pi->dtmedian_iter_ns;
        }
        snprintf(vstr, sizeof(vstr), "%.1f", hz);
    }
    else if (strcmp(pname, "exectime") == 0)
    {
        double us = (double) pi->dtmedian_exec_ns / 1000.0;
        snprintf(vstr, sizeof(vstr), "%.1f", us);
    }
    else if (strcmp(pname, "rtprio") == 0)
    {
        snprintf(vstr, sizeof(vstr), "%d", pi->RT_priority);
    }
    else if (strcmp(pname, "ctrlval") == 0)
    {
        snprintf(vstr, sizeof(vstr), "%d", pi->CTRLval);
    }
    else if (strcmp(pname, "trigmode") == 0)
    {
        snprintf(vstr, sizeof(vstr), "%d", pi->triggermode);
    }
    else if (strcmp(pname, "statusmsg") == 0)
    {
        strncpy(vstr, pi->statusmsg, sizeof(vstr) - 1);
        vstr[sizeof(vstr) - 1] = '\0';
    }
    else if (strcmp(pname, "tmux") == 0)
    {
        strncpy(vstr, pi->tmuxname, sizeof(vstr) - 1);
        vstr[sizeof(vstr) - 1] = '\0';
    }
    else if (strcmp(pname, "description") == 0)
    {
        strncpy(vstr, pi->description, sizeof(vstr) - 1);
        vstr[sizeof(vstr) - 1] = '\0';
    }
    else if (strcmp(pname, "missedframes") == 0)
    {
        snprintf(vstr, sizeof(vstr), "%lu", (unsigned long) pi->triggermissedframe_cumul);
    }

    int vlen  = (int) strlen(vstr);
    int avail = maxlen - 1 - *opos;
    int clen  = vlen < avail ? vlen : avail;
    memcpy(out + *opos, vstr, (size_t) clen);
    *opos += clen;
    munmap(pi, sizeof(PROCESSINFO));
    close(pfd);
    return 1;
}

/**
 * expand_fpsvar_procinfo_strict - @proc.name.prop
 * @pname:  "name.prop" (after "proc." prefix)
 *
 * Splits on the second dot and delegates to
 * expand_fpsvar_procinfo(). Returns 1 if handled.
 */
static int expand_fpsvar_procinfo_strict(char *pname, char *out, int *opos, int maxlen)
{
    char *dot2 = strchr(pname, '.');
    if (dot2 == NULL)
    {
        return 0;
    }
    *dot2 = '\0';
    return expand_fpsvar_procinfo(pname, dot2 + 1, out, opos, maxlen);
}

/**
 * expand_fpsvar_stream - expand @s.name.prop tokens
 * @pname:  "name.prop" (after "s." or "stream." prefix)
 * @out:    Output buffer
 * @opos:   Output position (updated)
 * @maxlen: Output buffer size
 *
 * Opens the ImageStreamIO SHM and reads the requested
 * metadata property (xsize, ysize, cnt0, type, etc.).
 * Returns 1 if expanded, 0 if stream not found.
 */
static int expand_fpsvar_stream(char *pname, char *out, int *opos, int maxlen)
{
    char *dot2 = strchr(pname, '.');
    if (dot2 == NULL)
    {
        return 0;
    }
    *dot2             = '\0';
    const char *sname = pname;
    const char *prop  = dot2 + 1;

    char retbuf[512];
    retbuf[0] = '\0';

    char shmchkpath[STRINGMAXLEN_DIRNAME + 128 + 16];
    snprintf(shmchkpath, sizeof(shmchkpath), "%s/%s.im.shm", dcshmdir, sname);
    if (access(shmchkpath, F_OK) != 0)
    {
        return 0;
    }

    IMAGE img;
    memset(&img, 0, sizeof(IMAGE));
    errno_t sret = ImageStreamIO_openIm(&img, sname);
    if (sret == IMAGESTREAMIO_SUCCESS && img.md != NULL)
    {
        int found = 0;
        if (strcmp(prop, "xsize") == 0)
        {
            snprintf(retbuf, sizeof(retbuf), "%u", img.md->size[0]);
            found = 1;
        }
        else if (strcmp(prop, "ysize") == 0)
        {
            snprintf(retbuf, sizeof(retbuf), "%u",
                     (img.md->naxis > 1 && img.md->size[1] > 0) ? img.md->size[1] : 1U);
            found = 1;
        }
        else if (strcmp(prop, "zsize") == 0)
        {
            snprintf(retbuf, sizeof(retbuf), "%u",
                     (img.md->naxis > 2 && img.md->size[2] > 0) ? img.md->size[2] : 1U);
            found = 1;
        }
        else if (strcmp(prop, "naxis") == 0)
        {
            snprintf(retbuf, sizeof(retbuf), "%u", (unsigned) img.md->naxis);
            found = 1;
        }
        else if (strcmp(prop, "type") == 0)
        {
            snprintf(retbuf, sizeof(retbuf), "%u", (unsigned) img.md->datatype);
            found = 1;
        }
        else if (strcmp(prop, "typename") == 0)
        {
            const char *tn = ImageStreamIO_typename(img.md->datatype);
            if (tn != NULL)
            {
                strncpy(retbuf, tn, sizeof(retbuf) - 1);
                retbuf[sizeof(retbuf) - 1] = '\0';
            }
            else
            {
                snprintf(retbuf, sizeof(retbuf), "%u", (unsigned) img.md->datatype);
            }
            found = 1;
        }
        else if (strcmp(prop, "typeid") == 0)
        {
            snprintf(retbuf, sizeof(retbuf), "%u", (unsigned) img.md->datatype);
            found = 1;
        }
        else if (strcmp(prop, "cnt0") == 0)
        {
            snprintf(retbuf, sizeof(retbuf), "%lu", (unsigned long) img.md->cnt0);
            found = 1;
        }
        else if (strcmp(prop, "cnt1") == 0)
        {
            snprintf(retbuf, sizeof(retbuf), "%lu", (unsigned long) img.md->cnt1);
            found = 1;
        }
        else if (strcmp(prop, "sem") == 0)
        {
            snprintf(retbuf, sizeof(retbuf), "%u", (unsigned) img.md->sem);
            found = 1;
        }
        else if (strcmp(prop, "pid") == 0)
        {
            snprintf(retbuf, sizeof(retbuf), "%d", (int) img.md->creatorPID);
            found = 1;
        }
        else if (strcmp(prop, "ownerPID") == 0)
        {
            snprintf(retbuf, sizeof(retbuf), "%d", (int) img.md->ownerPID);
            found = 1;
        }
        else if (strcmp(prop, "nelement") == 0)
        {
            snprintf(retbuf, sizeof(retbuf), "%lu", (unsigned long) img.md->nelement);
            found = 1;
        }

        ImageStreamIO_closeIm(&img);

        if (found)
        {
            int vlen  = (int) strlen(retbuf);
            int avail = maxlen - 1 - *opos;
            int clen  = vlen < avail ? vlen : avail;
            memcpy(out + *opos, retbuf, (size_t) clen);
            *opos += clen;
            return 1;
        }
    }
    return 0;
}

/**
 * expand_fpsvar_fps_strict - @fps.name.param
 * @pname:  "name.param" (after "fps." prefix)
 *
 * Connects to the named FPS and reads the parameter.
 * Supports wildcard param "*" (lists all keywords).
 * Returns 1 if connected (even if param not found).
 */
static int expand_fpsvar_fps_strict(char *pname, char *out, int *opos, int maxlen)
{
    char *dot2 = strchr(pname, '.');
    if (dot2 == NULL)
    {
        return 0;
    }
    *dot2               = '\0';
    const char *fpsname = pname;
    const char *fprop   = dot2 + 1;

    FPS fps;
    int fpsconn = fps_connect(fpsname, &fps, FPSCONNECT_SIMPLE);
    if (fpsconn == -1 || fps.parray == NULL)
    {
        return 0;
    }

    if (strcmp(fprop, "*") == 0)
    {
        int first = 1;
        for (int pi = 0; pi < fps.md->NBparamMAX; pi++)
        {
            if (fps.parray[pi].fpflag & FPFLAG_ACTIVE)
            {
                const char *kw    = fps.parray[pi].keyword[0];
                int         kwlen = (int) strlen(kw);
                int         avail = maxlen - 1 - *opos;

                if (!first && *opos < maxlen - 1)
                {
                    out[(*opos)++] = ' ';
                    avail--;
                }
                first = 0;

                if (kwlen > avail)
                {
                    kwlen = avail;
                }
                if (kwlen > 0)
                {
                    memcpy(out + *opos, kw, (size_t) kwlen);
                    *opos += kwlen;
                }
            }
        }
        fps_disconnect(&fps);
        return 1;
    }

    int pindex = functionparameter_GetParamIndex(&fps, fprop);
    if (pindex < 0)
    {
        char dotname[512];
        snprintf(dotname, sizeof(dotname), ".%s", fprop);
        pindex = functionparameter_GetParamIndex(&fps, dotname);
    }

    if (pindex >= 0)
    {
        char vstr[512];
        functionparameter_GetParamValueString(&fps.parray[pindex], vstr, (int) sizeof(vstr));

        int vlen  = (int) strlen(vstr);
        int avail = maxlen - 1 - *opos;
        int clen  = vlen < avail ? vlen : avail;
        memcpy(out + *opos, vstr, (size_t) clen);
        *opos += clen;
    }

    fps_disconnect(&fps);
    return 1;
}


/* ============================================================
 *  cli_expand_fpsvar — main @ expansion entry point
 * ============================================================
 */

/**
 * @brief Expand @namespace.name.prop tokens in place
 *
 * Scans the command line for @-tokens (outside quotes
 * and backslash escapes). Dispatches tokens to the
 * appropriate namespace handler:
 *   - @seq.NAME.prop  → sequencer SHM
 *   - @proc.NAME.prop → processinfo SHM
 *   - @s.NAME.prop    → ImageStreamIO SHM
 *   - @fps.NAME.prop  → FPS SHM (strict)
 *   - @NAME.prop      → FPS legacy fallback,
 *                        then procinfo fallback
 *
 * Write path (@token=val) is handled via
 * expand_fpsvar_write() before the read dispatch.
 *
 * Expansion is suppressed inside single and double
 * quotes and after backslash escapes.
 *
 * @param line    Command line buffer (modified in-place)
 * @param maxlen  Buffer size
 */
void cli_expand_fpsvar(char *line, int maxlen)
{
    char out[STRINGMAXLEN_CLICMDLINE];
    int  opos      = 0;
    int  i         = 0;
    int  in_single = 0;
    int  in_double = 0;
    int  out_esc   = 0;

    while (line[i] != '\0' && opos < maxlen - 1)
    {
        char c = line[i];

        /* ---- Quote / escape tracking ---- */
        if (out_esc)
        {
            out_esc     = 0;
            out[opos++] = c;
            i++;
            continue;
        }

        if (c == '\\' && !in_single)
        {
            out_esc     = 1;
            out[opos++] = c;
            i++;
            continue;
        }

        if (c == '\'' && !in_double)
        {
            in_single   = !in_single;
            out[opos++] = c;
            i++;
            continue;
        }

        if (c == '"' && !in_single)
        {
            in_double   = !in_double;
            out[opos++] = c;
            i++;
            continue;
        }

        /* @ expansion only outside quotes */
        if (c == '@' && !in_single && !in_double)
        {
            i++; /* skip @ */
            char token[512];
            int  tlen = 0;
            while (line[i] != '\0' && tlen < 510)
            {
                char tc = line[i];
                if (isalnum((unsigned char) tc) || tc == '_' || tc == '.' || tc == '-' || tc == '*')
                {
                    token[tlen++] = line[i++];
                }
                else if (tc == '$')
                {
                    /* Absorb $id or ${...}
                     * verbatim for later */
                    token[tlen++] = line[i++];
                    if (line[i] == '{')
                    {
                        token[tlen++] = line[i++];
                        while (line[i] != '\0' && line[i] != '}' && tlen < 510)
                        {
                            token[tlen++] = line[i++];
                        }
                        if (line[i] == '}' && tlen < 511)
                        {
                            token[tlen++] = line[i++];
                        }
                    }
                    else
                    {
                        while (line[i] != '\0' && tlen < 510 &&
                               (isalnum((unsigned char) line[i]) || line[i] == '_'))
                        {
                            token[tlen++] = line[i++];
                        }
                    }
                }
                else
                {
                    break;
                }
            }
            token[tlen] = '\0';

            /* Expand $VAR embedded in token */
            if (strchr(token, '$') != NULL)
            {
                cli_expand_env(token, (int) sizeof(token));
                tlen = (int) strlen(token);
            }

            /* Write path: @token=val */
            if (line[i] == '=' && strchr(token, '.') != NULL)
            {
                i++;
                expand_fpsvar_write(line, &i, token, out, &opos, maxlen);
                continue;
            }

            char *dot = strchr(token, '.');
            if (dot == NULL)
            {
                /* No dot: pass @ through */
                if (opos < maxlen - 1)
                {
                    out[opos++] = '@';
                }
                int clen = tlen;
                if (opos + clen > maxlen - 1)
                {
                    clen = maxlen - 1 - opos;
                }
                memcpy(out + opos, token, (size_t) clen);
                opos += clen;
                continue;
            }

            *dot                = '\0';
            const char *fpsname = token;
            char       *pname   = dot + 1;

            /* ---- Strict namespaces ---- */
            if (strcmp(fpsname, "seq") == 0)
            {
                if (expand_fpsvar_seq(pname, out, &opos, maxlen) != 0)
                {
                    continue;
                }
            }
            else if (strcmp(fpsname, "proc") == 0)
            {
                if (expand_fpsvar_procinfo_strict(pname, out, &opos, maxlen) != 0)
                {
                    continue;
                }
            }
            else if (strcmp(fpsname, "s") == 0 || strcmp(fpsname, "stream") == 0)
            {
                if (expand_fpsvar_stream(pname, out, &opos, maxlen) != 0)
                {
                    continue;
                }
            }
            else if (strcmp(fpsname, "fps") == 0)
            {
                if (expand_fpsvar_fps_strict(pname, out, &opos, maxlen) != 0)
                {
                    continue;
                }
            }

            /* ---- Legacy fallback ---- */
            FPS fps;
            int fpsconn = fps_connect(fpsname, &fps, FPSCONNECT_SIMPLE);

            if (fpsconn == -1 || fps.parray == NULL)
            {
                /* Not an FPS — try procinfo */
                expand_fpsvar_procinfo(fpsname, pname, out, &opos, maxlen);
                continue;
            }

            /* @fps.* — enumerate params */
            if (strcmp(pname, "*") == 0)
            {
                int first = 1;
                for (int pi = 0; pi < fps.md->NBparamMAX; pi++)
                {
                    if (fps.parray[pi].fpflag & FPFLAG_ACTIVE)
                    {
                        const char *kw    = fps.parray[pi].keyword[0];
                        int         kwlen = (int) strlen(kw);
                        int         avail = maxlen - 1 - opos;

                        if (!first && opos < maxlen - 1)
                        {
                            out[opos++] = ' ';
                            avail--;
                        }
                        first = 0;

                        if (kwlen > avail)
                        {
                            kwlen = avail;
                        }
                        if (kwlen > 0)
                        {
                            memcpy(out + opos, kw, (size_t) kwlen);
                            opos += kwlen;
                        }
                    }
                }
                fps_disconnect(&fps);
                continue;
            }

            int pindex = functionparameter_GetParamIndex(&fps, pname);

            if (pindex < 0)
            {
                char dotname[512];
                snprintf(dotname, sizeof(dotname), ".%s", pname);
                pindex = functionparameter_GetParamIndex(&fps, dotname);
            }

            if (pindex >= 0)
            {
                char vstr[512];
                functionparameter_GetParamValueString(&fps.parray[pindex], vstr,
                                                      (int) sizeof(vstr));

                int vlen  = (int) strlen(vstr);
                int avail = maxlen - 1 - opos;
                int clen  = vlen < avail ? vlen : avail;
                memcpy(out + opos, vstr, (size_t) clen);
                opos += clen;
            }

            fps_disconnect(&fps);
        }
        else
        {
            out[opos++] = line[i++];
        }
    }
    out[opos] = '\0';
    strncpy(line, out, (size_t) maxlen);
    line[maxlen - 1] = '\0';
}
