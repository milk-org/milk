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
#include "milk_proto.h"
#include "milk_server_internal.h"

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
 * @brief Append argument completions (modifiers, local images, SHM streams/FPS) matching @prefix.
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

    if (data.core.image != NULL)
    {
        for (long i = 0; i < data.core.NB_MAX_IMAGE; i++)
        {
            if (data.core.image[i].used == 1 && strncmp(data.core.image[i].name, prefix, plen) == 0)
            {
                int n = snprintf(out + pos, outsz - pos, "%s\n", data.core.image[i].name);
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
                    char ent[280];
                    snprintf(ent, sizeof(ent), "s.%s", sname);
                    if (strncmp(ent, prefix, plen) == 0)
                    {
                        int n = snprintf(out + pos, outsz - pos, "%s.\n", ent);
                        if (n > 0 && pos + (size_t) n < outsz)
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
                    size_t elen = strlen(ent);
                    if (strncmp(ent, prefix, plen) == 0)
                    {
                        int n = snprintf(out + pos, outsz - pos, "%s.\n", ent);
                        if (n > 0 && pos + (size_t) n < outsz)
                        {
                            pos += (size_t) n;
                        }
                    }
                    else if (plen > elen && strncmp(ent, prefix, elen) == 0 && prefix[elen] == '.')
                    {
                        static const char *const fattrs[] = { "status",    "conf",     "run",
                                                              "confstart", "confstop", "confstep",
                                                              "runstart",  "runstop",  "pid",
                                                              "nbparam",   "list",     NULL };
                        const char              *sub      = prefix + elen + 1;
                        size_t                   slen     = plen - elen - 1;
                        for (int a = 0; fattrs[a] != NULL; a++)
                        {
                            if (strncmp(fattrs[a], sub, slen) == 0)
                            {
                                int n = snprintf(out + pos, outsz - pos, "%s.%s\n", ent, fattrs[a]);
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
 * @brief Handle a `MILK_MSG_COMPLETE` request and send candidates back to @client_fd.
 *
 * @param client_fd Connected client socket fd
 * @param prefix    Token prefix to complete
 * @param comp_type `"cmd"` for command completion, `"arg"` for argument completion
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
