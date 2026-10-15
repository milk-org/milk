// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_server_io.c
 * @brief Command catalog dumping and tab-completion handler for milk-cli server.
 *
 * Role & Goal:
 *   1. `milk_server_dump_commands()`: Emits all canonical and unambiguous short
 *      command names registered in `data.cmd[]` (excluding shell builtins owned
 *      by bash) so `milk-bash-init` can define `@<cmd>` shell wrapper functions.
 *   2. `milk_server_handle_complete()`: Services `MILK_MSG_COMPLETE` requests
 *      for both command names (`comp_type == "cmd"`) and command arguments
 *      (modifiers `@T*:`, local images, SHM streams, `s.<stream>.`, `f.<fps>.`).
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "ImageStreamIO/ImageStreamIO.h"
#include "fps.h"
#include "milk_proto.h"
#include "milk_server_internal.h"
#include "milk_shm_prop.h"

/**
 * @brief Return 1 if @key is a legacy shell builtin, REPL-only command,
 *        internal help topic, or deprecated alias that should not be
 *        exported as an `@<cmd>` in Bash.
 *
 * @param key Command key registered in `data.cmd[]`
 * @return 1 if excluded from Bash export/completion, 0 otherwise
 */
static int is_bash_excluded_cmd(const char *key)
{
    static const char *const excluded[] = {
        /* Shell builtins owned by Bash */
        "cd", "pwd", "echo", "printf", "read", "shift", "export", "unset", "vars", "alias",
        "unalias", "aliases", "time", "watch", "exit", "quit", "exitCLI",
        /* Interactive milk-cli REPL / readline commands */
        "history", "searchhist", "fhist", "ghistory", "lhistory", "savehistory", "savescript",
        "setprompt", "bookmark", "sessionlog", "synhl", "syndiag", "cliindent", "showmatch",
        "helprl", "fhelp", "fparam",
        /* Internal help-topic keys (use @h <topic>, @hb, @hc, @hm, @hf) */
        "help", "help-bash", "help-cmdopts", "help-syntax", "help-commands", "help-variables",
        "help-flowcontrol", "help-scripting", "help-milk",
        /* Deprecated / duplicate aliases */
        "list-streams", "list-fps", "fpsset", NULL
    };

    if (key == NULL || key[0] == '\0')
    {
        return 1;
    }

    for (int i = 0; excluded[i] != NULL; i++)
    {
        if (strcmp(key, excluded[i]) == 0)
        {
            return 1;
        }
    }
    return 0;
}

/**
 * @brief Dump all registered canonical and unambiguous short command names.
 *
 * @param fp Output stream
 */
void milk_server_dump_commands(FILE *fp)
{
    if (fp == NULL)
    {
        return;
    }

    for (unsigned int i = 0; i < data.NBcmd; i++)
    {
        const char *key = data.cmd[i].key;
        if (key[0] == '\0' || strchr(key, '?') != NULL || is_bash_excluded_cmd(key))
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
        if (dot != NULL && dot[1] != '\0' && !is_bash_excluded_cmd(dot + 1))
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

    fprintf(fp, "imgstat\ninfo.imgstat\nmk2Drnd\nimgen.mk2Drnd\nprocstat\nprocctl\nprocwait\n");
}

/**
 * @brief Append command-name completions matching @prefix into @out.
 */
static size_t complete_commands(const char *prefix, const char *sigil, char *out, size_t outsz)
{
    size_t      pos    = 0;
    size_t      siglen = strlen(sigil);
    const char *bare   = prefix;
    if (strncmp(bare, sigil, siglen) == 0)
    {
        bare += siglen;
    }
    else if (bare[0] == '@')
    {
        bare++;
    }
    size_t blen = strlen(bare);

    static const char *const core_cmds[] = {
        "h",      "hb",       "hc",      "hm",       "hf",      "get",       "set",
        "calc",   "test",     "exists",  "running",  "waitfor", "waitany",   "grab",
        "defer",  "on",       "source",  "server",   "log",     "streamcat", "streamwrite",
        "unload", "procstat", "procctl", "procwait", NULL
    };
    for (int i = 0; core_cmds[i] != NULL; i++)
    {
        if (strncmp(core_cmds[i], bare, blen) == 0)
        {
            int n = snprintf(out + pos, outsz - pos, "%s%s\n", sigil, core_cmds[i]);
            if (n > 0 && pos + (size_t) n < outsz)
            {
                pos += (size_t) n;
            }
        }
    }
    for (unsigned int i = 0; i < data.NBcmd; i++)
    {
        const char *k = data.cmd[i].key;
        if (k[0] == '\0' || strchr(k, '?') != NULL || is_bash_excluded_cmd(k))
        {
            continue;
        }
        if (strncmp(k, bare, blen) == 0)
        {
            int n = snprintf(out + pos, outsz - pos, "%s%s\n", sigil, k);
            if (n > 0 && pos + (size_t) n < outsz)
            {
                pos += (size_t) n;
            }
        }
        const char *dot = strchr(k, '.');
        if (dot != NULL && !is_bash_excluded_cmd(dot + 1) && strncmp(dot + 1, bare, blen) == 0)
        {
            int n = snprintf(out + pos, outsz - pos, "%s%s\n", sigil, dot + 1);
            if (n > 0 && pos + (size_t) n < outsz)
            {
                pos += (size_t) n;
            }
        }
    }
    return pos;
}

/**
 * @brief Complete stream attributes (`s.<sname>.<attr>`) and FITS keywords (`s.<sname>.kw.<KEY>`).
 */
static void complete_stream_attrs(const char *lead,
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
                    int n = snprintf(out + pos, outsz - pos, "%ss.%s.kw.%s\n", lead, sname,
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
            int n = snprintf(out + pos, outsz - pos, "%ss.%s.%s\n", lead, sname, sattrs[a]);
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
            char ent[280];
            snprintf(ent, sizeof(ent), "s.%s", iname);
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
                complete_stream_attrs(lead, iname, bpref + elen + 1, blen - elen - 1, out, outsz,
                                      &pos);
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
                        complete_stream_attrs(lead, sname, bpref + elen + 1, blen - elen - 1, out,
                                              outsz, &pos);
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
            else if (dlen > 9 && strcmp(de->d_name + dlen - 9, ".procinfo") == 0)
            {
                char   pname[256];
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
                            "PID",        "status",   "loopcnt",       "cnt0",
                            "loopcntMax", "loopstat", "dt_ns",         "dt_median_ns",
                            "fps_hz",     "tmuxname", "triggerstream", "description",
                            "msg",        "ctrlval",  "list",          NULL
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
 * @brief Resolve a command name (ignoring leading sigil) to its index in `data.cmd[]`.
 */
static int find_server_cmd(const char *cmdname)
{
    if (cmdname == NULL || cmdname[0] == '\0')
    {
        return -1;
    }
    const char *sigil = getenv("MILKCLI_SIGIL");
    if (sigil != NULL && sigil[0] != '\0' && strncmp(cmdname, sigil, strlen(sigil)) == 0)
    {
        cmdname += strlen(sigil);
    }
    else if (cmdname[0] == '@' || cmdname[0] == ',')
    {
        cmdname++;
    }
    if (cmdname[0] == '\0')
    {
        return -1;
    }

    /* Common built-in aliases */
    if (strcmp(cmdname, "imstats") == 0 || strcmp(cmdname, "imgstat") == 0)
    {
        cmdname = "info.imgstat";
    }
    else if (strcmp(cmdname, "mk2Drnd") == 0)
    {
        cmdname = "imgen.mk2Drnd";
    }

    for (unsigned int i = 0; i < data.NBcmd; i++)
    {
        if (strcmp(data.cmd[i].key, cmdname) == 0)
        {
            return (int) i;
        }
        int midx = data.cmd[i].moduleindex;
        if (midx >= 0 && data.module[midx].shortname[0] != '\0')
        {
            char qname[256];
            snprintf(qname, sizeof(qname), "%s.%s", data.module[midx].shortname, data.cmd[i].key);
            if (strcmp(qname, cmdname) == 0)
            {
                return (int) i;
            }
        }
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

    const char *sigil = getenv("MILKCLI_SIGIL");
    if (sigil == NULL || sigil[0] == '\0')
    {
        sigil = "@";
    }

    if (comp_type != NULL && strcmp(comp_type, "cmd") == 0)
    {
        pos = complete_commands(prefix, sigil, out, sizeof(out));
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
        const char *bcmd   = cbuf;
        if (strncmp(bcmd, sigil, strlen(sigil)) == 0)
        {
            bcmd += strlen(sigil);
        }
        else if (bcmd[0] == '@' || bcmd[0] == ',')
        {
            bcmd++;
        }

        const char *bpref  = (prefix[0] == '@') ? (prefix + 1) : prefix;
        int is_entity_pref = (strncmp(bpref, "s.", 2) == 0 || strncmp(bpref, "f.", 2) == 0 ||
                              strncmp(bpref, "p.", 2) == 0 || strncmp(bpref, "q.", 2) == 0 ||
                              strncmp(bpref, "stream.", 7) == 0 || strncmp(bpref, "fps.", 4) == 0 ||
                              strncmp(bpref, "proc.", 5) == 0 || strncmp(bpref, "seq.", 4) == 0);

        int cmdidx = find_server_cmd(bcmd);

        if (strcmp(bcmd, "hc") == 0 && argidx == 0)
        {
            pos = complete_commands(prefix, "", out, sizeof(out));
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
                 ((strcmp(bcmd, "savefits") == 0 || strcmp(bcmd, "saveFITS") == 0 ||
                   strcmp(bcmd, "iofits.savefits") == 0) &&
                  argidx == 1))
        {
            pos = (size_t) snprintf(out, sizeof(out), "__MILK_COMPLETE_FITS__\n");
        }
        else if ((strcmp(bcmd, "loadfits") == 0 || strcmp(bcmd, "iofits.loadfits") == 0) &&
                 argidx == 1)
        {
            pos = complete_images_only(prefix, out, sizeof(out));
        }
        else if ((strcmp(bcmd, "savefits") == 0 || strcmp(bcmd, "saveFITS") == 0 ||
                  strcmp(bcmd, "iofits.savefits") == 0) &&
                 argidx == 0)
        {
            pos = complete_images_only(prefix, out, sizeof(out));
        }
        else if (strcmp(bcmd, "source") == 0 || strcmp(bcmd, "include_once") == 0)
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
