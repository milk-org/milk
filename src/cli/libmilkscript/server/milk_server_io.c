// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_server_io.c
 * @brief Tab-completion handler for the milk-cli server.
 *
 * Role & Goal:
 *   `milk_server_handle_complete()` services `MILK_MSG_COMPLETE` requests for command names
 *   (`comp_type == "cmd"`: `@keywords` and `module.command`s) and command arguments (modifiers
 *   `@T*:`, local images, SHM streams, `s.<stream>.`, `f.<fps>.`).
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "milk_lexicon.h"
#include "milk_cmd_index.h"
#include "CLIcore.h"
#include "ImageStreamIO/ImageStreamIO.h"
#include "fps.h"
#include "milk_entity_ns.h"
#include "milk_proto.h"
#include "milk_server_internal.h"
#include "milk_shm_prop.h"

/**
 * @brief Append command-name completions of @prefix into @out: `@keywords`, or `module.command`s.
 */
static size_t complete_commands(const char *prefix, char *out, size_t outsz)
{
    size_t pos  = 0;
    int    kw   = (prefix[0] == '@');
    size_t blen = strlen(prefix + kw);

    for (int i = 0; kw && milk_lexicon_keywords()[i].name != NULL; i++) /* milk.bash keywords */
    {
        const milk_keyword_t *k = &milk_lexicon_keywords()[i];
        int                   n;

        if ((k->avail & MILK_KW_BASH) && strncmp(k->name, prefix + 1, blen) == 0 &&
            (n = snprintf(out + pos, outsz - pos, "@%s\n", k->name)) > 0 &&
            pos + (size_t) n < outsz)
        {
            pos += (size_t) n;
        }
    }
    for (unsigned int i = 0; !kw && i < data.NBcmd; i++) /* by key or by command name ("mk2D") */
    {
        const char *k   = data.cmd[i].key;
        const char *dot = strchr(k, '.');
        int         n;

        if (dot != NULL && (strncmp(k, prefix, blen) == 0 || strncmp(dot + 1, prefix, blen) == 0) &&
            (n = snprintf(out + pos, outsz - pos, "%s\n", k)) > 0 && pos + (size_t) n < outsz)
        {
            pos += (size_t) n;
        }
    }
    return pos;
}

/**
 * @brief Complete stream attributes (`<ns>.<sname>.<attr>`) and FITS keywords.
 */
static void complete_stream_attrs_ns(const char *lead,
                                     const char *ns,
                                     const char *sname,
                                     const char *sub,
                                     size_t      slen,
                                     char       *out,
                                     size_t      outsz,
                                     size_t     *pos_io)
{
    static const char *const sattrs[] = { "xsize", "ysize",     "zsize",    "naxis", "nelement",
                                          "dtype", "typename",  "cnt0",     "cnt1",  "shared",
                                          "sem",   "owner_pid", "location", "inode", "age",
                                          "fps",   "summary",   "list",     "kw.",   NULL };
    size_t                   pos      = *pos_io;

    if (strncmp(sub, "kw.", 3) == 0)
    {
        const char *ksub = sub + 3;
        size_t      klen = slen - 3;
        IMAGE       tmp_im;
        int         need_close = 0;
        IMAGE      *im         = milk_shm_prop_find_or_open_stream(sname, &tmp_im, &need_close);
        if (im != NULL && im->md != NULL && im->kw != NULL)
        {
            uint16_t nkw = im->md->NBkw;
            for (uint16_t k = 0; k < nkw; k++)
            {
                if (im->kw[k].type != 'N' && im->kw[k].name[0] != '\0' &&
                    strncmp(im->kw[k].name, ksub, klen) == 0)
                {
                    int n = snprintf(out + pos, outsz - pos, "%s%s.%s.kw.%s\n", lead, ns, sname,
                                     im->kw[k].name);
                    if (n > 0 && pos + (size_t) n < outsz)
                    {
                        pos += (size_t) n;
                    }
                }
            }
        }
        if (need_close)
        {
            ImageStreamIO_closeIm(&tmp_im);
        }
        *pos_io = pos;
        return;
    }

    for (int a = 0; sattrs[a] != NULL; a++)
    {
        if (strncmp(sattrs[a], sub, slen) == 0)
        {
            int n = snprintf(out + pos, outsz - pos, "%s%s.%s.%s\n", lead, ns, sname, sattrs[a]);
            if (n > 0 && pos + (size_t) n < outsz)
            {
                pos += (size_t) n;
            }
        }
    }
    *pos_io = pos;
}

/**
 * @brief Append argument completions (modifiers, local images, SHM streams/FPS/proc/seq) matching @prefix.
 */
static size_t complete_arguments(const char *prefix, char *out, size_t outsz)
{
    size_t pos  = 0;
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
                int n = snprintf(out + pos, outsz - pos, "%s\n", mods[i]);
                if (n > 0 && pos + (size_t) n < outsz)
                {
                    pos += (size_t) n;
                }
            }
        }
    }

    const char *lead  = (prefix[0] == '@') ? "@" : "";
    const char *bpref = (prefix[0] == '@') ? (prefix + 1) : prefix;
    size_t      blen  = strlen(bpref);

    if (data.core.image != NULL)
    {
        for (long i = 0; i < data.core.NB_MAX_IMAGE; i++)
        {
            if (data.core.image[i].used != 1)
            {
                continue;
            }
            const char *iname = data.core.image[i].name;
            if (prefix[0] != '@' && strncmp(iname, prefix, plen) == 0)
            {
                int n = snprintf(out + pos, outsz - pos, "%s\n", iname);
                if (n > 0 && pos + (size_t) n < outsz)
                {
                    pos += (size_t) n;
                }
            }
            const char *ns_pair[2] = { "l", "s" };
            for (int np = 0; np < 2; np++)
            {
                if (np == 1 && data.core.image[i].shmfd < 0)
                {
                    continue;
                }
                char ent[280];
                snprintf(ent, sizeof(ent), "%s.%s", ns_pair[np], iname);
                size_t elen = strlen(ent);
                if (strncmp(ent, bpref, blen) == 0)
                {
                    int n = snprintf(out + pos, outsz - pos, "%s%s.\n", lead, ent);
                    if (n > 0 && pos + (size_t) n < outsz)
                    {
                        pos += (size_t) n;
                    }
                }
                else if (blen > elen && strncmp(ent, bpref, elen) == 0 && bpref[elen] == '.')
                {
                    complete_stream_attrs_ns(lead, ns_pair[np], iname, bpref + elen + 1,
                                             blen - elen - 1, out, outsz, &pos);
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
                    if (prefix[0] != '@' && strncmp(sname, prefix, plen) == 0)
                    {
                        int n = snprintf(out + pos, outsz - pos, "%s\n", sname);
                        if (n > 0 && pos + (size_t) n < outsz)
                        {
                            pos += (size_t) n;
                        }
                    }
                    char ent[280];
                    snprintf(ent, sizeof(ent), "s.%s", sname);
                    size_t elen = strlen(ent);
                    if (strncmp(ent, bpref, blen) == 0)
                    {
                        int n = snprintf(out + pos, outsz - pos, "%s%s.\n", lead, ent);
                        if (n > 0 && pos + (size_t) n < outsz)
                        {
                            pos += (size_t) n;
                        }
                    }
                    else if (blen > elen && strncmp(ent, bpref, elen) == 0 && bpref[elen] == '.')
                    {
                        complete_stream_attrs_ns(lead, "s", sname, bpref + elen + 1,
                                                 blen - elen - 1, out, outsz, &pos);
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
                    size_t elen = strlen(ent);
                    if (strncmp(ent, bpref, blen) == 0)
                    {
                        int n = snprintf(out + pos, outsz - pos, "%s%s.\n", lead, ent);
                        if (n > 0 && pos + (size_t) n < outsz)
                        {
                            pos += (size_t) n;
                        }
                    }
                    else if (blen > elen && strncmp(ent, bpref, elen) == 0 && bpref[elen] == '.')
                    {
                        static const char *const fattrs[] = {
                            "status",      "conf",     "run",      "confrunning", "runrunning",
                            "confstart",   "confstop", "confstep", "runstart",    "runstop",
                            "reset",       "pid",      "confpid",  "runpid",      "nbparam",
                            "description", "list",     NULL
                        };
                        const char *sub  = bpref + elen + 1;
                        size_t      slen = blen - elen - 1;
                        for (int a = 0; fattrs[a] != NULL; a++)
                        {
                            if (strncmp(fattrs[a], sub, slen) == 0)
                            {
                                int n = snprintf(out + pos, outsz - pos, "%s%s.%s\n", lead, ent,
                                                 fattrs[a]);
                                if (n > 0 && pos + (size_t) n < outsz)
                                {
                                    pos += (size_t) n;
                                }
                            }
                        }

                        char lspec[300];
                        char pbuf[4096];
                        snprintf(lspec, sizeof(lspec), "f.%s.*", fname);
                        if (milk_shm_prop_get(lspec, pbuf, sizeof(pbuf)) == 0)
                        {
                            char *saveptr = NULL;
                            for (char *tok = strtok_r(pbuf, " \t\n", &saveptr); tok != NULL;
                                 tok       = strtok_r(NULL, " \t\n", &saveptr))
                            {
                                const char *pname = tok;
                                while (*pname == '.')
                                {
                                    pname++;
                                }
                                if (strncmp(pname, fname, flen) == 0 && pname[flen] == '.')
                                {
                                    pname += flen + 1;
                                }
                                if (pname[0] != '\0' && strncmp(pname, sub, slen) == 0)
                                {
                                    int n = snprintf(out + pos, outsz - pos, "%s%s.%s\n", lead, ent,
                                                     pname);
                                    if (n > 0 && pos + (size_t) n < outsz)
                                    {
                                        pos += (size_t) n;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else if ((dlen > 9 && strcmp(de->d_name + dlen - 9, ".procinfo") == 0) ||
                     (dlen > 10 && strncmp(de->d_name, "proc.", 5) == 0 &&
                      strcmp(de->d_name + dlen - 4, ".shm") == 0))
            {
                char pname[256];
                pname[0] = '\0';
                if (dlen > 9 && strcmp(de->d_name + dlen - 9, ".procinfo") == 0)
                {
                    size_t nlen = dlen - 9;
                    if (nlen < sizeof(pname))
                    {
                        memcpy(pname, de->d_name, nlen);
                        pname[nlen] = '\0';
                        char *dot   = strrchr(pname, '.');
                        if (dot != NULL)
                        {
                            *dot = '\0';
                        }
                    }
                }
                else
                {
                    size_t bl = dlen - 9;
                    if (bl < sizeof(pname))
                    {
                        memcpy(pname, de->d_name + 5, bl);
                        pname[bl] = '\0';
                        char *dot = strrchr(pname, '.');
                        if (dot != NULL)
                        {
                            *dot = '\0';
                        }
                    }
                }
                if (pname[0] != '\0')
                {
                    char ent[280];
                    snprintf(ent, sizeof(ent), "p.%s", pname);
                    size_t elen = strlen(ent);
                    if (strncmp(ent, bpref, blen) == 0)
                    {
                        int n = snprintf(out + pos, outsz - pos, "%s%s.\n", lead, ent);
                        if (n > 0 && pos + (size_t) n < outsz)
                        {
                            pos += (size_t) n;
                        }
                    }
                    else if (blen > elen && strncmp(ent, bpref, elen) == 0 && bpref[elen] == '.')
                    {
                        static const char *const pattrs[] = {
                            "PID",          "status", "state",      "ctrl",          "ctrlval",
                            "loopcnt",      "cnt0",   "loopcntMax", "loopstat",      "dt_ns",
                            "dt_median_ns", "fps_hz", "tmuxname",   "triggerstream", "description",
                            "msg",          "list",   NULL
                        };
                        const char *sub  = bpref + elen + 1;
                        size_t      slen = blen - elen - 1;
                        for (int a = 0; pattrs[a] != NULL; a++)
                        {
                            if (strncmp(pattrs[a], sub, slen) == 0)
                            {
                                int n = snprintf(out + pos, outsz - pos, "%s%s.%s\n", lead, ent,
                                                 pattrs[a]);
                                if (n > 0 && pos + (size_t) n < outsz)
                                {
                                    pos += (size_t) n;
                                }
                            }
                        }
                    }
                }
            }
            else if (dlen > 12 && strncmp(de->d_name, "milkseq.", 8) == 0 &&
                     strcmp(de->d_name + dlen - 4, ".shm") == 0)
            {
                char   qname[256];
                size_t qlen = dlen - 12;
                if (qlen < sizeof(qname))
                {
                    memcpy(qname, de->d_name + 8, qlen);
                    qname[qlen] = '\0';
                    char ent[280];
                    snprintf(ent, sizeof(ent), "q.%s", qname);
                    size_t elen = strlen(ent);
                    if (strncmp(ent, bpref, blen) == 0)
                    {
                        int n = snprintf(out + pos, outsz - pos, "%s%s.\n", lead, ent);
                        if (n > 0 && pos + (size_t) n < outsz)
                        {
                            pos += (size_t) n;
                        }
                    }
                    else if (blen > elen && strncmp(ent, bpref, elen) == 0 && bpref[elen] == '.')
                    {
                        static const char *const qattrs[] = {
                            "state",      "active_step", "progress", "step_count", "PID",  "name",
                            "last_error", "script",      "action",   "submit",     "list", NULL
                        };
                        const char *sub  = bpref + elen + 1;
                        size_t      slen = blen - elen - 1;
                        for (int a = 0; qattrs[a] != NULL; a++)
                        {
                            if (strncmp(qattrs[a], sub, slen) == 0)
                            {
                                int n = snprintf(out + pos, outsz - pos, "%s%s.%s\n", lead, ent,
                                                 qattrs[a]);
                                if (n > 0 && pos + (size_t) n < outsz)
                                {
                                    pos += (size_t) n;
                                }
                            }
                        }
                    }
                }
            }
        }
        closedir(dp);
    }

    return pos;
}

/**
 * @brief Resolve a command name (ignoring a leading @) to its index in `data.cmd[]`.
 */
static int find_server_cmd(const char *cmdname)
{
    if (cmdname == NULL || cmdname[0] == '\0')
    {
        return -1;
    }
    cmdname += (cmdname[0] == '@');
    if (cmdname[0] == '\0')
    {
        return -1;
    }

    int exact = milk_cmd_lookup(cmdname);
    if (exact >= 0)
    {
        return exact;
    }

    int match = -1;
    int count = 0;
    for (unsigned int i = 0; i < data.NBcmd; i++)
    {
        const char *dot = strchr(data.cmd[i].key, '.');
        if (dot != NULL && strcmp(dot + 1, cmdname) == 0)
        {
            match = (int) i;
            count++;
        }
    }
    return (count == 1) ? match : -1;
}

/**
 * argtype_label - Map a `CLIARG_*` parameter type constant to a short display label
 * @type: `CLICMDARGDATA` argument type code (`CLIARG_*`)
 *
 * Returns a concise uppercase label (`FITS`, `FILE`, `IMG`, `FPS`, `ONOFF`, `FLOAT`,
 * `INT`, `UINT`, `STR`, or `ARG`) used when formatting command signatures during
 * interactive tab-completion.
 *
 * Return: Static short label string for @type
 */
static const char *argtype_label(uint64_t type)
{
    if (type == CLIARG_FITSFILENAME)
    {
        return "FITS";
    }
    if (type == CLIARG_FILENAME)
    {
        return "FILE";
    }
    if (type == CLIARG_IMG || type == CLIARG_STREAM)
    {
        return "IMG";
    }
    if (type == CLIARG_FPSNAME)
    {
        return "FPS";
    }
    if (type == CLIARG_ONOFF)
    {
        return "ONOFF";
    }
    if (type == CLIARG_FLOAT || type == CLIARG_FLOAT32)
    {
        return "FLOAT";
    }
    if (type == CLIARG_LONG || type == CLIARG_INT32 || type == CLIARG_INT64)
    {
        return "INT";
    }
    if (type == CLIARG_UINT32 || type == CLIARG_UINT64)
    {
        return "UINT";
    }
    if (type == CLIARG_STR || type == CLIARG_STR_NOT_IMG)
    {
        return "STR";
    }
    return "ARG";
}

/**
 * find_cmd_cli_arg - Locate the metadata descriptor for the N-th positional CLI argument
 * @cmdidx: Index of the command in `data.cmd[]`
 * @argidx: Zero-based positional argument index on the command line
 *
 * Walks the command's `argdata` array, counting entries marked with
 * `FPFLAG_PRIMARY_CLI_INPUT` (or falling back to raw index order if none are
 * marked), so context-aware tab-completion knows the expected type of the active
 * argument.
 *
 * Return: Pointer to the matching `CLICMDARGDATA` entry, or NULL if out of range
 */
static const CLICMDARGDATA *find_cmd_cli_arg(int cmdidx, int argidx)
{
    if (cmdidx < 0 || argidx < 0 || data.cmd[cmdidx].argdata == NULL)
    {
        return NULL;
    }
    int cli_ai      = 0;
    int any_primary = 0;
    for (int ai = 0; ai < data.cmd[cmdidx].nbparam; ai++)
    {
        if (data.cmd[cmdidx].argdata[ai].fpflag & FPFLAG_PRIMARY_CLI_INPUT)
        {
            any_primary = 1;
            if (cli_ai == argidx)
            {
                return &data.cmd[cmdidx].argdata[ai];
            }
            cli_ai++;
        }
    }
    if (!any_primary && argidx < data.cmd[cmdidx].nbarg && argidx < data.cmd[cmdidx].nbparam)
    {
        return &data.cmd[cmdidx].argdata[argidx];
    }
    return NULL;
}

/**
 * format_cmd_signature - Format a command's argument signature and active-argument hint
 * @cmdname:       Command name to look up
 * @active_argidx: Zero-based index of the argument currently being typed (-1 if none)
 * @out:           Output buffer receiving the formatted signature text
 * @outsz:         Capacity of @out in bytes
 *
 * Builds a two-line help snippet showing the command's primary CLI arguments
 * (highlighting @active_argidx with `*...*`) and the description/default of the
 * active argument.
 *
 * Return: Number of bytes written to @out (0 if @cmdname is not found)
 */
static size_t format_cmd_signature(const char *cmdname, int active_argidx, char *out, size_t outsz)
{
    int cmdidx = find_server_cmd(cmdname);
    if (cmdidx < 0)
    {
        return 0;
    }

    size_t               pos         = 0;
    const CMD           *cmd         = &data.cmd[cmdidx];
    const CLICMDARGDATA *active_argd = NULL;
    int                  nprinted    = 0;

    int n = snprintf(out + pos, outsz - pos, "%s", cmd->key);
    if (n > 0 && pos + (size_t) n < outsz)
    {
        pos += (size_t) n;
    }

    if (cmd->argdata != NULL && cmd->nbparam > 0)
    {
        int cli_ai = 0;
        for (int ai = 0; ai < cmd->nbparam; ai++)
        {
            if (!(cmd->argdata[ai].fpflag & FPFLAG_PRIMARY_CLI_INPUT))
            {
                continue;
            }
            const CLICMDARGDATA *ad  = &cmd->argdata[ai];
            const char          *tag = ad->fpstag;
            while (*tag == '.')
            {
                tag++;
            }
            int is_active = (cli_ai == active_argidx);
            if (is_active)
            {
                active_argd = ad;
            }
            if (ad->example[0] != '\0')
            {
                n = snprintf(out + pos, outsz - pos, " %s<%s:%s=%s>%s", is_active ? "*" : "", tag,
                             argtype_label(ad->type), ad->example, is_active ? "*" : "");
            }
            else
            {
                n = snprintf(out + pos, outsz - pos, " %s<%s:%s>%s", is_active ? "*" : "", tag,
                             argtype_label(ad->type), is_active ? "*" : "");
            }
            if (n > 0 && pos + (size_t) n < outsz)
            {
                pos += (size_t) n;
            }
            cli_ai++;
            nprinted++;
        }
    }

    if (nprinted == 0 && cmd->syntax[0] != '\0')
    {
        n = snprintf(out + pos, outsz - pos, " %s", cmd->syntax);
        if (n > 0 && pos + (size_t) n < outsz)
        {
            pos += (size_t) n;
        }
    }

    if (cmd->info[0] != '\0')
    {
        n = snprintf(out + pos, outsz - pos, " -- %s", cmd->info);
        if (n > 0 && pos + (size_t) n < outsz)
        {
            pos += (size_t) n;
        }
    }

    if (active_argd != NULL)
    {
        const char *tag = active_argd->fpstag;
        while (*tag == '.')
        {
            tag++;
        }
        n = snprintf(out + pos, outsz - pos, "\n[arg %d] %s (%s%s%s): %s", active_argidx + 1, tag,
                     argtype_label(active_argd->type), active_argd->example[0] ? ", default=" : "",
                     active_argd->example[0] ? active_argd->example : "", active_argd->descr);
        if (n > 0 && pos + (size_t) n < outsz)
        {
            pos += (size_t) n;
        }
    }

    if (pos + 1 < outsz)
    {
        out[pos++] = '\n';
        out[pos]   = '\0';
    }
    return pos;
}

/**
 * complete_cmd_dot_tags - Complete named `.tag` parameter overrides for a command
 * @cmdidx: Index of the command in `data.cmd[]`
 * @prefix: Current word prefix starting with `.`
 * @out:    Output buffer receiving newline-delimited `.tag` completions
 * @outsz:  Capacity of @out in bytes
 *
 * Scans all parameters declared in `data.cmd[cmdidx].argdata` and emits matching
 * `.fpstag` names so users can tab-complete named argument overrides.
 *
 * Return: Number of bytes written to @out
 */
static size_t complete_cmd_dot_tags(int cmdidx, const char *prefix, char *out, size_t outsz)
{
    size_t pos  = 0;
    size_t plen = strlen(prefix);
    if (cmdidx < 0 || data.cmd[cmdidx].argdata == NULL)
    {
        return 0;
    }
    for (int ai = 0; ai < data.cmd[cmdidx].nbparam; ai++)
    {
        const char *tag = data.cmd[cmdidx].argdata[ai].fpstag;
        if (tag[0] == '\0')
        {
            continue;
        }
        char dtag[128];
        if (tag[0] == '.')
        {
            snprintf(dtag, sizeof(dtag), "%s", tag);
        }
        else
        {
            snprintf(dtag, sizeof(dtag), ".%s", tag);
        }
        if (strncmp(dtag, prefix, plen) == 0)
        {
            int n = snprintf(out + pos, outsz - pos, "%s\n", dtag);
            if (n > 0 && pos + (size_t) n < outsz)
            {
                pos += (size_t) n;
            }
        }
    }
    return pos;
}

/**
 * complete_fps_only - Complete active FPS shared-memory names matching a prefix
 * @prefix: Prefix to match against `<name>.fps.shm` entries
 * @out:    Output buffer receiving newline-delimited FPS names
 * @outsz:  Capacity of @out in bytes
 *
 * Scans the shared-memory directory for `.fps.shm` files and emits bare FPS names
 * for commands expecting a `CLIARG_FPSNAME` argument.
 *
 * Return: Number of bytes written to @out
 */
static size_t complete_fps_only(const char *prefix, char *out, size_t outsz)
{
    size_t      pos    = 0;
    size_t      plen   = strlen(prefix);
    const char *shmdir = getenv("MILK_SHM_DIR");
    if (shmdir == NULL || shmdir[0] == '\0')
    {
        shmdir = (dcshmdir[0] != '\0') ? dcshmdir : "/dev/shm";
    }
    DIR *dp = opendir(shmdir);
    if (dp != NULL)
    {
        struct dirent *de;
        while ((de = readdir(dp)) != NULL)
        {
            size_t dlen = strlen(de->d_name);
            if (dlen > 8 && strcmp(de->d_name + dlen - 8, ".fps.shm") == 0)
            {
                char   fname[256];
                size_t flen = dlen - 8;
                if (flen < sizeof(fname))
                {
                    memcpy(fname, de->d_name, flen);
                    fname[flen] = '\0';
                    if (strncmp(fname, prefix, plen) == 0)
                    {
                        int n = snprintf(out + pos, outsz - pos, "%s\n", fname);
                        if (n > 0 && pos + (size_t) n < outsz)
                        {
                            pos += (size_t) n;
                        }
                    }
                }
            }
        }
        closedir(dp);
    }
    return pos;
}

/**
 * complete_images_only - Complete stream modifiers, local images, and SHM streams
 * @prefix: Prefix to match against `@*:` modifiers, local image names, and `.im.shm` files
 * @out:    Output buffer receiving newline-delimited completion candidates
 * @outsz:  Capacity of @out in bytes
 *
 * Provides targeted completion candidates when a command parameter has type
 * `CLIARG_IMG` or `CLIARG_STREAM`.
 *
 * Return: Number of bytes written to @out
 */
static size_t complete_images_only(const char *prefix, char *out, size_t outsz)
{
    size_t pos  = 0;
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
                int n = snprintf(out + pos, outsz - pos, "%s\n", mods[i]);
                if (n > 0 && pos + (size_t) n < outsz)
                {
                    pos += (size_t) n;
                }
            }
        }
        return pos;
    }

    if (data.core.image != NULL)
    {
        for (long i = 0; i < data.core.NB_MAX_IMAGE; i++)
        {
            if (data.core.image[i].used != 1)
            {
                continue;
            }
            const char *iname = data.core.image[i].name;
            if (strncmp(iname, prefix, plen) == 0)
            {
                int n = snprintf(out + pos, outsz - pos, "%s\n", iname);
                if (n > 0 && pos + (size_t) n < outsz)
                {
                    pos += (size_t) n;
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
                        int n = snprintf(out + pos, outsz - pos, "%s\n", sname);
                        if (n > 0 && pos + (size_t) n < outsz)
                        {
                            pos += (size_t) n;
                        }
                    }
                }
            }
        }
        closedir(dp);
    }
    return pos;
}

/**
 * @brief Handle a `MILK_MSG_COMPLETE` request and send candidates back to @client_fd.
 *
 * @param client_fd Connected client socket fd
 * @param prefix    Token prefix to complete
 * @param comp_type `"cmd"`, `"sig[:<cmd>[:<argidx>]]"`, or `"arg[:<cmd>:<cword>]"`
 */
void milk_server_handle_complete(int client_fd, const char *prefix, const char *comp_type)
{
    char   out[32768];
    size_t pos = 0;
    out[0]     = '\0';

    if (comp_type != NULL && strcmp(comp_type, "cmd") == 0)
    {
        pos = complete_commands(prefix, out, sizeof(out));
    }
    else if (comp_type != NULL && strcmp(comp_type, "sig") == 0)
    {
        pos = format_cmd_signature(prefix, -1, out, sizeof(out));
    }
    else if (comp_type != NULL && strncmp(comp_type, "sig:", 4) == 0)
    {
        char cbuf[256];
        snprintf(cbuf, sizeof(cbuf), "%s", comp_type + 4);
        int   argidx = -1;
        char *colon  = strrchr(cbuf, ':');
        if (colon != NULL && colon[1] >= '0' && colon[1] <= '9')
        {
            *colon = '\0';
            argidx = atoi(colon + 1) - 1;
        }
        else if (prefix != NULL && prefix[0] >= '0' && prefix[0] <= '9')
        {
            argidx = atoi(prefix) - 1;
        }
        pos = format_cmd_signature(cbuf, argidx, out, sizeof(out));
    }
    else if (comp_type != NULL && strncmp(comp_type, "arg:", 4) == 0)
    {
        char cbuf[256];
        snprintf(cbuf, sizeof(cbuf), "%s", comp_type + 4);
        int   cword = 1;
        char *colon = strrchr(cbuf, ':');
        if (colon != NULL)
        {
            *colon = '\0';
            cword  = atoi(colon + 1);
            if (cword < 1)
            {
                cword = 1;
            }
        }
        int         argidx = cword - 1;
        const char *bcmd   = cbuf + (cbuf[0] == '@');

        const char *bpref          = (prefix[0] == '@') ? (prefix + 1) : prefix;
        int         is_entity_pref = (milk_entity_ns_kind(bpref, NULL) != MILK_NS_NONE);

        int cmdidx = find_server_cmd(bcmd);

        if (strcmp(bcmd, "help") == 0 && argidx == 0)
        {
            pos = complete_commands(prefix, out, sizeof(out));
        }
        else if (prefix[0] == '.' && prefix[1] != '/' && prefix[1] != '.' && cmdidx >= 0)
        {
            pos = complete_cmd_dot_tags(cmdidx, prefix, out, sizeof(out));
        }
        else if (is_entity_pref)
        {
            pos = complete_arguments(prefix, out, sizeof(out));
        }
        else if (((strcmp(bcmd, "loadfits") == 0 || strcmp(bcmd, "iofits.loadfits") == 0) &&
                  argidx == 0) ||
                 ((strcmp(bcmd, "saveFITS") == 0 || strcmp(bcmd, "iofits.saveFITS") == 0) &&
                  argidx == 1))
        {
            pos = (size_t) snprintf(out, sizeof(out), "__MILK_COMPLETE_FITS__\n");
        }
        else if ((strcmp(bcmd, "loadfits") == 0 || strcmp(bcmd, "iofits.loadfits") == 0) &&
                 argidx == 1)
        {
            pos = complete_images_only(prefix, out, sizeof(out));
        }
        else if ((strcmp(bcmd, "saveFITS") == 0 || strcmp(bcmd, "iofits.saveFITS") == 0) &&
                 argidx == 0)
        {
            pos = complete_images_only(prefix, out, sizeof(out));
        }
        else if (strcmp(bcmd, "source") == 0)
        {
            pos = (size_t) snprintf(out, sizeof(out), "__MILK_COMPLETE_FILES__\n");
        }
        else if (strcmp(bcmd, "fpsCTRL") == 0 || strcmp(bcmd, "fpsconf") == 0 ||
                 strcmp(bcmd, "fpsrun") == 0 || strcmp(bcmd, "fpsstop") == 0 ||
                 strcmp(bcmd, "waitfor_fps") == 0)
        {
            pos = complete_fps_only(prefix, out, sizeof(out));
        }
        else if (cmdidx >= 0)
        {
            const CLICMDARGDATA *argd = find_cmd_cli_arg(cmdidx, argidx);
            if (argd != NULL)
            {
                if (argd->type == CLIARG_FITSFILENAME)
                {
                    pos = (size_t) snprintf(out, sizeof(out), "__MILK_COMPLETE_FITS__\n");
                }
                else if (argd->type == CLIARG_FILENAME)
                {
                    pos = (size_t) snprintf(out, sizeof(out), "__MILK_COMPLETE_FILES__\n");
                }
                else if (argd->type == CLIARG_FPSNAME)
                {
                    pos = complete_fps_only(prefix, out, sizeof(out));
                }
                else if (argd->type == CLIARG_IMG || argd->type == CLIARG_STREAM)
                {
                    pos = complete_images_only(prefix, out, sizeof(out));
                }
                else if (argd->type == CLIARG_ONOFF)
                {
                    static const char *const onoff[] = { "ON", "OFF", "1", "0", NULL };
                    for (int i = 0; onoff[i] != NULL; i++)
                    {
                        if (strncmp(onoff[i], prefix, strlen(prefix)) == 0)
                        {
                            int n = snprintf(out + pos, sizeof(out) - pos, "%s\n", onoff[i]);
                            if (n > 0 && pos + (size_t) n < sizeof(out))
                            {
                                pos += (size_t) n;
                            }
                        }
                    }
                }
                else
                {
                    if (prefix[0] == '\0' && argd->example[0] != '\0')
                    {
                        int n = snprintf(out + pos, sizeof(out) - pos, "%s\n", argd->example);
                        if (n > 0 && pos + (size_t) n < sizeof(out))
                        {
                            pos += (size_t) n;
                        }
                    }
                    else
                    {
                        pos = complete_arguments(prefix, out, sizeof(out));
                    }
                }
            }
            else
            {
                pos = complete_arguments(prefix, out, sizeof(out));
            }
        }
        else
        {
            pos = complete_arguments(prefix, out, sizeof(out));
        }
    }
    else
    {
        pos = complete_arguments(prefix, out, sizeof(out));
    }

    if (pos > 0)
    {
        milk_proto_send(client_fd, MILK_MSG_VALUE, 0, out, (uint32_t) (pos + 1));
    }
    int32_t rc = 0;
    milk_proto_send(client_fd, MILK_MSG_DONE, 0, &rc, sizeof(rc));
}
