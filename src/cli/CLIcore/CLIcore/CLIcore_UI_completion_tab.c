// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_UI_completion_tab.c
 *
 * @brief Tab-completion generator and dispatcher
 *
 * Contains CLI_generator() and CLI_completion()
 * which implement the readline tab-completion
 * engine for commands, streams, FPS, files, and
 * argument types.
 *
 * @see CLIcore_UI_completion.c for prompt, input
 *      handling, and Levenshtein distance.
 */

#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>
#include <string.h>
#include <unistd.h>

#ifdef USE_READLINE
#    include <readline/history.h>
#    include <readline/readline.h>
#endif

#include <ctype.h>

#include "CLIcore.h"
#include "CLIcore_script.h"
#include "CLIcore_UI_execute.h"
#include "fps_connect.h"
#include "treesitter/cli_treesitter.h"

extern char **environ;



#ifdef USE_READLINE

/* ---- Tab-completion generator ---- */

/**
 * @brief State for fuzzy fallback pass
 *
 * After a normal prefix-match pass, if nothing
 * matched and fuzzy is enabled, we restart with
 * substring match.
 */
int generator_fuzzy_pass = 0;

/**
 * @brief Generate tab-completion candidates
 *
 * Called repeatedly by readline to produce
 * matching candidates. The match mode
 * (commands, images, args, files, FPS)
 * determines the search space.
 *
 * On first call (state == 0), initializes
 * the search. Returns one match at a time,
 * or NULL when exhausted.
 */
char *CLI_generator(const char *text, int state)
{
    static unsigned int list_index;
    static unsigned int len;
    static int          matches_found_in_pass = 0;
    char               *name;

#define GEN_DUPSTR(s) (matches_found_in_pass++, dupstr(s))

    if (!state)
    {
        list_index            = 0;
        len                   = strlen(text);
        generator_fuzzy_pass  = 0;
        matches_found_in_pass = 0;
    }

retry_fuzzy:

    if (data.CLImatchMode == CLICOMPLETIONMODE_COMMANDS)
    {
        /* Built-in keywords not in data.cmd[] */
        static const char        *builtins[] = { "if",
                                                 "elif",
                                                 "else",
                                                 "fi",
                                                 "for",
                                                 "while",
                                                 "until",
                                                 "do",
                                                 "done",
                                                 "case",
                                                 "esac",
                                                 "select",
                                                 "function",
                                                 ".",
                                                 "source",
                                                 "break",
                                                 "continue",
                                                 "return",
                                                 "true",
                                                 "false",
                                                 "exit",
                                                 "shift",
                                                 "assert",
                                                 "assigncheck",
                                                 "dpdigits",
                                                 "set",
                                                 "export",
                                                 "readonly",
                                                 "local",
                                                 "declare",
                                                 "let",
                                                 "eval",
                                                 "type",
                                                 "command",
                                                 "trap",
                                                 "watch",
                                                 "time",
                                                 "timeout",
                                                 "wait",
                                                 "wait_any",
                                                 "printf",
                                                 "echo",
                                                 "getopts",
                                                 "mapfile",
                                                 "alias",
                                                 "unalias",
                                                 "basename",
                                                 "dirname",
                                                 "pushd",
                                                 "popd",
                                                 "dirs",
                                                 "seq",
                                                 "[[",
                                                 "procctl",
                                                 "procwait",
                                                 "procstat",
                                                 "waitfor_stream",
                                                 "waitfor_fps",
                                                 "on_update",
                                                 "on_fpschange",
                                                 "include_once",
                                                 "savescript",
                                                 "savehistory",
                                                 NULL };
        static const unsigned int nbuiltins =
            sizeof(builtins) / sizeof(builtins[0]) - 1; /* exclude NULL */

        /* Phase 1: registered commands */
        while (list_index < data.NBcmd)
        {
            name = data.cmd[list_index].key;
            list_index++;
            if (generator_fuzzy_pass == 0)
            {
                if (strncmp(name, text, len) == 0)
                {
                    return (GEN_DUPSTR(name));
                }
            }
            else
            {
                /* Fuzzy: substring match */
                if (strstr(name, text) != NULL)
                {
                    return (GEN_DUPSTR(name));
                }
            }
        }

        /* Phase 2: built-in keywords */
        unsigned int bi = list_index - data.NBcmd;
        while (bi < nbuiltins)
        {
            name = (char *) builtins[bi];
            list_index++;
            bi++;
            if (generator_fuzzy_pass == 0)
            {
                if (strncmp(name, text, len) == 0)
                {
                    return (GEN_DUPSTR(name));
                }
            }
            else
            {
                if (strstr(name, text) != NULL)
                {
                    return (GEN_DUPSTR(name));
                }
            }
        }
    }

    if (data.CLImatchMode == CLICOMPLETIONMODE_IMAGES)
    {
        static DIR *img_dirp = NULL;

        if (!state)
        {
            if (img_dirp != NULL)
            {
                closedir(img_dirp);
                img_dirp = NULL;
            }
            img_dirp = opendir(dcshmdir);
        }

        if (img_dirp != NULL)
        {
            struct dirent *ent;
            while ((ent = readdir(img_dirp)) != NULL)
            {
                char *ext = strstr(ent->d_name, ".im.shm");
                if (ext != NULL && strcmp(ext, ".im.shm") == 0)
                {
                    char imgname[256];
                    int  namelen = ext - ent->d_name;
                    if (namelen > 255)
                    {
                        namelen = 255;
                    }
                    strncpy(imgname, ent->d_name, namelen);
                    imgname[namelen] = '\0';

                    if (generator_fuzzy_pass == 0)
                    {
                        if (strncmp(imgname, text, len) == 0)
                        {
                            return (GEN_DUPSTR(imgname));
                        }
                    }
                    else
                    {
                        if (strstr(imgname, text) != NULL)
                        {
                            return (GEN_DUPSTR(imgname));
                        }
                    }
                }
            }
            closedir(img_dirp);
            img_dirp = NULL;
        }
    }

    if (data.CLImatchMode == CLICOMPLETIONMODE_CMDARGS)
    {
        if (data.cmd[data.cmdindex].argdata != NULL)
        {
            while ((int) list_index < data.cmd[data.cmdindex].nbarg)
            {
                name = data.cmd[data.cmdindex].argdata[list_index].fpstag;
                list_index++;
                if (name != NULL && strncmp(name, text, len) == 0)
                {
                    return (GEN_DUPSTR(name));
                }
            }
        }
    }

    if (data.CLImatchMode == CLICOMPLETIONMODE_FILES)
    {
        static DIR         *dirp = NULL;
        static char         dirpart[512];
        static char         prefix[256];
        static unsigned int preflen;

        if (!state)
        {
            if (dirp != NULL)
            {
                closedir(dirp);
                dirp = NULL;
            }

            const char *slash = strrchr(text, '/');
            if (slash != NULL)
            {
                int dlen = (int) (slash - text) + 1;
                if (dlen > (int) sizeof(dirpart) - 1)
                {
                    dlen = (int) sizeof(dirpart) - 1;
                }
                memcpy(dirpart, text, dlen);
                dirpart[dlen] = '\0';
                strncpy(prefix, slash + 1, sizeof(prefix) - 1);
                prefix[sizeof(prefix) - 1] = '\0';
            }
            else
            {
                snprintf(dirpart, sizeof(dirpart), ".");
                strncpy(prefix, text, sizeof(prefix) - 1);
                prefix[sizeof(prefix) - 1] = '\0';
            }
            preflen = strlen(prefix);

            dirp = opendir(dirpart);
        }

        if (dirp != NULL)
        {
            struct dirent *ent;
            while ((ent = readdir(dirp)) != NULL)
            {
                if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
                {
                    continue;
                }

                if (strncmp(ent->d_name, prefix, preflen) == 0)
                {
                    char fullpath[1024];
                    snprintf(fullpath, sizeof(fullpath), "%s/%s", dirpart, ent->d_name);

                    char result[1024];
                    if (strcmp(dirpart, ".") == 0)
                    {
                        snprintf(result, sizeof(result), "%s", ent->d_name);
                    }
                    else
                    {
                        snprintf(result, sizeof(result), "%s%s", dirpart, ent->d_name);
                    }

                    struct stat st;
                    if (stat(fullpath, &st) == 0 && S_ISDIR(st.st_mode))
                    {
                        strncat(result, "/", sizeof(result) - strlen(result) - 1);
                    }

                    return GEN_DUPSTR(result);
                }
            }
            closedir(dirp);
            dirp = NULL;
        }
    }

    if (data.CLImatchMode == CLICOMPLETIONMODE_FPSPARAMS)
    {
        static DIR *fps_dirp = NULL;

        if (!state)
        {
            if (fps_dirp != NULL)
            {
                closedir(fps_dirp);
                fps_dirp = NULL;
            }
            fps_dirp = opendir(dcshmdir);
        }

        if (fps_dirp != NULL)
        {
            struct dirent *ent;
            while ((ent = readdir(fps_dirp)) != NULL)
            {
                if (strncmp(ent->d_name, "fps.", 4) != 0)
                {
                    continue;
                }
                char *ext = strstr(ent->d_name, ".datadir");
                if (ext != NULL && strcmp(ext, ".datadir") == 0)
                {
                    char fpsname[256];
                    int  namelen = ext - (ent->d_name + 4);
                    if (namelen > 255)
                    {
                        namelen = 255;
                    }
                    strncpy(fpsname, ent->d_name + 4, namelen);
                    fpsname[namelen] = '\0';

                    if (generator_fuzzy_pass == 0)
                    {
                        if (strncmp(fpsname, text, len) == 0)
                        {
                            return GEN_DUPSTR(fpsname);
                        }
                    }
                    else
                    {
                        if (strstr(fpsname, text) != NULL)
                        {
                            return GEN_DUPSTR(fpsname);
                        }
                    }
                }
            }
            closedir(fps_dirp);
            fps_dirp = NULL;
        }
    }

    if (data.CLImatchMode == CLICOMPLETIONMODE_VARS_FPS)
    {
        const char *dot1 = strchr(text, '.');
        const char *dot2 = (dot1 != NULL) ? strchr(dot1 + 1, '.') : NULL;

        if (dot2 != NULL)
        {
            static FPS  fps;
            static int  fps_connected = 0;
            static int  param_idx     = 0;
            static char fps_target[128];
            static char param_prefix[128];

            if (!state)
            {
                param_idx = 0;
                size_t fnlen = (size_t) (dot2 - (dot1 + 1));
                if (fnlen >= sizeof(fps_target))
                {
                    fnlen = sizeof(fps_target) - 1;
                }
                strncpy(fps_target, dot1 + 1, fnlen);
                fps_target[fnlen] = '\0';

                strncpy(param_prefix, dot2 + 1, sizeof(param_prefix) - 1);
                param_prefix[sizeof(param_prefix) - 1] = '\0';

                memset(&fps, 0, sizeof(FPS));
                fps_connected = (fps_connect(fps_target, &fps, FPSCONNECT_SIMPLE) == 0 &&
                                 fps.parray != NULL);
            }

            if (fps_connected)
            {
                while (param_idx < fps.md->NBparamMAX)
                {
                    int pi = param_idx++;
                    if (!(fps.parray[pi].fpflag & FPFLAG_ACTIVE))
                    {
                        continue;
                    }

                    const char *pname = fps.parray[pi].keyword[0];
                    if (pname == NULL || pname[0] == '\0')
                    {
                        continue;
                    }

                    int match = 0;
                    if (generator_fuzzy_pass == 0)
                    {
                        if (strncmp(pname, param_prefix, strlen(param_prefix)) == 0)
                        {
                            match = 1;
                        }
                    }
                    else if (strstr(pname, param_prefix) != NULL)
                    {
                        match = 1;
                    }

                    if (match)
                    {
                        char buf[256];
                        snprintf(buf, sizeof(buf), "@fps.%s.%s", fps_target, pname);
                        return GEN_DUPSTR(buf);
                    }
                }
            }
        }
        else
        {
            static DIR *vfps_dirp = NULL;
            if (!state)
            {
                if (vfps_dirp != NULL)
                {
                    closedir(vfps_dirp);
                    vfps_dirp = NULL;
                }
                vfps_dirp = opendir(dcshmdir);
            }
            if (vfps_dirp != NULL)
            {
                struct dirent *ent;
                while ((ent = readdir(vfps_dirp)) != NULL)
                {
                    if (strncmp(ent->d_name, "fps.", 4) == 0)
                    {
                        char *ext = strstr(ent->d_name, ".datadir");
                        if (ext != NULL && strcmp(ext, ".datadir") == 0)
                        {
                            char fpsname[256];
                            int  namelen = ext - (ent->d_name + 4);
                            if (namelen > 240)
                            {
                                namelen = 240;
                            }
                            snprintf(fpsname, sizeof(fpsname), "@fps.%.*s.",
                                     namelen, ent->d_name + 4);

                            if (generator_fuzzy_pass == 0)
                            {
                                if (strncmp(fpsname, text, len) == 0)
                                {
                                    return GEN_DUPSTR(fpsname);
                                }
                            }
                            else if (strstr(fpsname, text) != NULL)
                            {
                                return GEN_DUPSTR(fpsname);
                            }
                        }
                    }
                }
                closedir(vfps_dirp);
                vfps_dirp = NULL;
            }
        }
    }

    if (data.CLImatchMode == CLICOMPLETIONMODE_VARS_SEQ)
    {
        static DIR *vseq_dirp = NULL;
        if (!state)
        {
            if (vseq_dirp != NULL)
            {
                closedir(vseq_dirp);
                vseq_dirp = NULL;
            }
            vseq_dirp = opendir(dcshmdir);
        }
        if (vseq_dirp != NULL)
        {
            struct dirent *ent;
            while ((ent = readdir(vseq_dirp)) != NULL)
            {
                if (strncmp(ent->d_name, "seq.", 4) == 0)
                {
                    char *ext = strstr(ent->d_name, ".shm");
                    if (ext != NULL && strcmp(ext, ".shm") == 0)
                    {
                        char seqname[256];
                        int  namelen = ext - (ent->d_name + 4);
                        if (namelen > 240)
                        {
                            namelen = 240;
                        }
                        snprintf(seqname, sizeof(seqname), "@seq.%.*s.",
                                 namelen, ent->d_name + 4);

                        if (generator_fuzzy_pass == 0)
                        {
                            if (strncmp(seqname, text, len) == 0)
                            {
                                return GEN_DUPSTR(seqname);
                            }
                        }
                        else if (strstr(seqname, text) != NULL)
                        {
                            return GEN_DUPSTR(seqname);
                        }
                    }
                }
            }
            closedir(vseq_dirp);
            vseq_dirp = NULL;
        }
    }

    if (data.CLImatchMode == CLICOMPLETIONMODE_VARS_STREAM)
    {
        const char *dot1 = strchr(text, '.');
        const char *dot2 = (dot1 != NULL) ? strchr(dot1 + 1, '.') : NULL;
        int has_brace = (text[0] == '$' && text[1] == '{');

        if (dot2 != NULL)
        {
            static int prop_idx = 0;
            static const char *stream_props[] = {
                "xsize", "ysize", "zsize", "naxis", "type", "typename",
                "nelem", "cnt0", "cnt1", "sem", NULL
            };
            static char stream_target[128];
            static char prop_prefix[64];

            if (!state)
            {
                prop_idx = 0;
                size_t snlen = (size_t) (dot2 - (dot1 + 1));
                if (snlen >= sizeof(stream_target))
                {
                    snlen = sizeof(stream_target) - 1;
                }
                strncpy(stream_target, dot1 + 1, snlen);
                stream_target[snlen] = '\0';

                strncpy(prop_prefix, dot2 + 1, sizeof(prop_prefix) - 1);
                prop_prefix[sizeof(prop_prefix) - 1] = '\0';
            }

            while (stream_props[prop_idx] != NULL)
            {
                const char *sprop = stream_props[prop_idx++];
                int match = 0;
                if (generator_fuzzy_pass == 0)
                {
                    if (strncmp(sprop, prop_prefix, strlen(prop_prefix)) == 0)
                    {
                        match = 1;
                    }
                }
                else if (strstr(sprop, prop_prefix) != NULL)
                {
                    match = 1;
                }

                if (match)
                {
                    char buf[256];
                    if (has_brace)
                    {
                        snprintf(buf, sizeof(buf), "${s.%s.%s}", stream_target, sprop);
                    }
                    else
                    {
                        snprintf(buf, sizeof(buf), "@s.%s.%s", stream_target, sprop);
                    }
                    return GEN_DUPSTR(buf);
                }
            }
        }
        else
        {
            static DIR *vstream_dirp = NULL;
            if (!state)
            {
                if (vstream_dirp != NULL)
                {
                    closedir(vstream_dirp);
                    vstream_dirp = NULL;
                }
                vstream_dirp = opendir(dcshmdir);
            }
            if (vstream_dirp != NULL)
            {
                struct dirent *ent;
                while ((ent = readdir(vstream_dirp)) != NULL)
                {
                    char *ext = strstr(ent->d_name, ".im.shm");
                    if (ext != NULL && strcmp(ext, ".im.shm") == 0)
                    {
                        char sname[256];
                        int  namelen = ext - ent->d_name;
                        if (namelen > 240)
                        {
                            namelen = 240;
                        }
                        if (has_brace)
                        {
                            snprintf(sname, sizeof(sname), "${s.%.*s.",
                                     namelen, ent->d_name);
                        }
                        else
                        {
                            snprintf(sname, sizeof(sname), "@s.%.*s.",
                                     namelen, ent->d_name);
                        }

                        if (generator_fuzzy_pass == 0)
                        {
                            if (strncmp(sname, text, len) == 0)
                            {
                                return GEN_DUPSTR(sname);
                            }
                        }
                        else if (strstr(sname, text) != NULL)
                        {
                            return GEN_DUPSTR(sname);
                        }
                    }
                }
                closedir(vstream_dirp);
                vstream_dirp = NULL;
            }
        }
    }

    if (data.CLImatchMode == CLICOMPLETIONMODE_VARS_ENV)
    {
        static int  var_phase;
        static int  var_idx;
        static int  has_dollar;
        static int  has_brace;
        static char vprefix[256];
        static int  vpreflen;

        if (!state)
        {
            var_phase  = 0;
            var_idx    = 0;
            has_dollar = (text[0] == '$');
            has_brace  = (has_dollar && text[1] == '{');

            const char *vp = text;
            if (has_brace)
            {
                vp = text + 2;
            }
            else if (has_dollar)
            {
                vp = text + 1;
            }
            strncpy(vprefix, vp, sizeof(vprefix) - 1);
            vprefix[sizeof(vprefix) - 1] = '\0';
            vpreflen = (int) strlen(vprefix);
        }

        /* Phase 0: special shell variables */
        static const char *special_vars[] = {
            "?", "$", "!", "#", "0", "MCLIFIFO", "PROCINFO_NCPU", "PROCINFO_NPROC", NULL
        };

        while (var_phase == 0)
        {
            const char *sname = special_vars[var_idx++];
            if (sname == NULL)
            {
                var_phase = 1;
                var_idx   = 0;
                break;
            }

            int match = 0;
            if (generator_fuzzy_pass == 0)
            {
                if (strncmp(sname, vprefix, vpreflen) == 0)
                {
                    match = 1;
                }
            }
            else if (strstr(sname, vprefix) != NULL)
            {
                match = 1;
            }

            if (match)
            {
                char buf[512];
                if (has_brace)
                {
                    snprintf(buf, sizeof(buf), "${%s}", sname);
                }
                else if (has_dollar)
                {
                    snprintf(buf, sizeof(buf), "$%s", sname);
                }
                else
                {
                    snprintf(buf, sizeof(buf), "%s", sname);
                }
                return GEN_DUPSTR(buf);
            }
        }

        /* Phase 1: CLI script variables */
        while (var_phase == 1)
        {
            if (var_idx >= CLI_MAX_VARS)
            {
                var_phase = 2;
                var_idx   = 0;
                break;
            }

            int i = var_idx++;
            if (!cli_vars[i].used)
            {
                continue;
            }

            const char *vname = cli_vars[i].name;
            int match = 0;
            if (generator_fuzzy_pass == 0)
            {
                if (strncmp(vname, vprefix, vpreflen) == 0)
                {
                    match = 1;
                }
            }
            else if (strstr(vname, vprefix) != NULL)
            {
                match = 1;
            }

            if (match)
            {
                char buf[512];
                if (has_brace)
                {
                    snprintf(buf, sizeof(buf), "${%s}", vname);
                }
                else if (has_dollar)
                {
                    snprintf(buf, sizeof(buf), "$%s", vname);
                }
                else
                {
                    snprintf(buf, sizeof(buf), "%s", vname);
                }
                return GEN_DUPSTR(buf);
            }
        }

        /* Phase 2: CLI script arrays */
        while (var_phase == 2)
        {
            if (var_idx >= CLI_MAX_ARRAYS)
            {
                var_phase = 3;
                var_idx   = 0;
                break;
            }

            int i = var_idx++;
            if (!cli_arrays[i].used)
            {
                continue;
            }

            const char *aname = cli_arrays[i].name;
            int match = 0;
            if (generator_fuzzy_pass == 0)
            {
                if (strncmp(aname, vprefix, vpreflen) == 0)
                {
                    match = 1;
                }
            }
            else if (strstr(aname, vprefix) != NULL)
            {
                match = 1;
            }

            if (match)
            {
                char buf[512];
                if (has_brace)
                {
                    snprintf(buf, sizeof(buf), "${%s[@]}", aname);
                }
                else if (has_dollar)
                {
                    snprintf(buf, sizeof(buf), "$%s", aname);
                }
                else
                {
                    snprintf(buf, sizeof(buf), "%s", aname);
                }
                return GEN_DUPSTR(buf);
            }
        }

        /* Phase 3: Environment variables */
        while (var_phase == 3)
        {
            if (environ == NULL || environ[var_idx] == NULL)
            {
                var_phase = 4;
                var_idx   = 0;
                break;
            }

            const char *entry = environ[var_idx++];
            const char *eq = strchr(entry, '=');
            if (eq == NULL)
            {
                continue;
            }

            char ename[256];
            size_t nlen = (size_t) (eq - entry);
            if (nlen >= sizeof(ename))
            {
                nlen = sizeof(ename) - 1;
            }
            strncpy(ename, entry, nlen);
            ename[nlen] = '\0';

            /* Avoid duplicating variables already in cli_vars */
            int already_in_cli = 0;
            for (int k = 0; k < CLI_MAX_VARS; k++)
            {
                if (cli_vars[k].used && strcmp(cli_vars[k].name, ename) == 0)
                {
                    already_in_cli = 1;
                    break;
                }
            }
            if (already_in_cli)
            {
                continue;
            }

            int match = 0;
            if (generator_fuzzy_pass == 0)
            {
                if (strncmp(ename, vprefix, vpreflen) == 0)
                {
                    match = 1;
                }
            }
            else if (strstr(ename, vprefix) != NULL)
            {
                match = 1;
            }

            if (match)
            {
                char buf[512];
                if (has_brace)
                {
                    snprintf(buf, sizeof(buf), "${%s}", ename);
                }
                else if (has_dollar)
                {
                    snprintf(buf, sizeof(buf), "$%s", ename);
                }
                else
                {
                    snprintf(buf, sizeof(buf), "%s", ename);
                }
                return GEN_DUPSTR(buf);
            }
        }
    }

    /* Fuzzy fallback: if prefix pass found
     * nothing, restart with substring */
    if (generator_fuzzy_pass == 0 && matches_found_in_pass == 0 && data.autocomplete_fuzzy)
    {
        generator_fuzzy_pass = 1;
        list_index           = 0;
        state                = 0;
        goto retry_fuzzy;
    }

#undef GEN_DUPSTR

    return ((char *) NULL);
}


/* ---- TAB completion dispatcher ---- */

/**
 * @brief Readline custom completion dispatcher
 *
 * Invoked on TAB. Uses Tree-sitter AST and statement-boundary
 * classification to determine completion mode based on cursor position
 * and the command/argument being typed.
 */
char **CLI_completion(const char *text, int start, int __attribute__((unused)) end)
{
    char **matches = NULL;
    char   cmdname[128] = "";
    int    argidx = 0;

    int mode = cli_ts_determine_completion_mode(
        rl_line_buffer, start, text, cmdname, sizeof(cmdname), &argidx);

    if (mode >= 0)
    {
        data.CLImatchMode = mode;
    }
    else
    {
        /* A command was found (cmdname), and cursor is on argument argidx */
        if (text[0] == '.' && text[1] != '/' && text[1] != '.')
        {
            data.CLImatchMode = CLICOMPLETIONMODE_CMDARGS;
        }
        else if (strcmp(cmdname, "loadfits") == 0 ||
                 strcmp(cmdname, "savefits") == 0 ||
                 strcmp(cmdname, "saveFITS") == 0 ||
                 strcmp(cmdname, "source") == 0 ||
                 strcmp(cmdname, ".") == 0 ||
                 strcmp(cmdname, "cat") == 0 ||
                 strcmp(cmdname, "cd") == 0 ||
                 strcmp(cmdname, "ls") == 0 ||
                 strcmp(cmdname, "vi") == 0 ||
                 strcmp(cmdname, "vim") == 0 ||
                 strcmp(cmdname, "nano") == 0 ||
                 strcmp(cmdname, "head") == 0 ||
                 strcmp(cmdname, "tail") == 0 ||
                 strcmp(cmdname, "cp") == 0 ||
                 strcmp(cmdname, "mv") == 0 ||
                 strcmp(cmdname, "rm") == 0 ||
                 strcmp(cmdname, "less") == 0 ||
                 strcmp(cmdname, "more") == 0 ||
                 strcmp(cmdname, "include_once") == 0 ||
                 strcmp(cmdname, "savescript") == 0 ||
                 strcmp(cmdname, "savehistory") == 0 ||
                 strcmp(cmdname, "run") == 0)
        {
            data.CLImatchMode = CLICOMPLETIONMODE_FILES;
        }
        else if (strcmp(cmdname, "fpsCTRL") == 0 ||
                 strcmp(cmdname, "fparam") == 0 ||
                 strcmp(cmdname, "fpsload") == 0 ||
                 strcmp(cmdname, "dpsingle") == 0 ||
                 strcmp(cmdname, "fpsconf") == 0 ||
                 strcmp(cmdname, "fpsrun") == 0 ||
                 strcmp(cmdname, "fpsstop") == 0 ||
                 strcmp(cmdname, "waitfor_fps") == 0)
        {
            data.CLImatchMode = CLICOMPLETIONMODE_FPSPARAMS;
        }
        else if (strcmp(cmdname, "export") == 0 ||
                 strcmp(cmdname, "readonly") == 0 ||
                 strcmp(cmdname, "unset") == 0 ||
                 strcmp(cmdname, "local") == 0 ||
                 strcmp(cmdname, "declare") == 0)
        {
            data.CLImatchMode = CLICOMPLETIONMODE_VARS_ENV;
        }
        else
        {
            /* Lookup registered milk command in data.cmd */
            int cmdimatch = find_command_match(cmdname);
            int matched_mode = -1;

            if (cmdimatch >= 0 && data.cmd[cmdimatch].argdata != NULL)
            {
                int cli_ai = 0;
                for (int ai = 0; ai < data.cmd[cmdimatch].nbparam; ai++)
                {
                    if (data.cmd[cmdimatch].argdata[ai].fpflag & FPFLAG_PRIMARY_CLI_INPUT)
                    {
                        if (cli_ai == argidx)
                        {
                            uint64_t atype = data.cmd[cmdimatch].argdata[ai].type;
                            if (atype == CLIARG_FILENAME || atype == CLIARG_FITSFILENAME)
                            {
                                matched_mode = CLICOMPLETIONMODE_FILES;
                            }
                            else if (atype == CLIARG_FPSNAME)
                            {
                                matched_mode = CLICOMPLETIONMODE_FPSPARAMS;
                            }
                            else if (atype == CLIARG_IMG || atype == CLIARG_STREAM)
                            {
                                matched_mode = CLICOMPLETIONMODE_IMAGES;
                            }
                            break;
                        }
                        cli_ai++;
                    }
                }
            }

            if (matched_mode >= 0)
            {
                data.CLImatchMode = matched_mode;
            }
            else
            {
                data.CLImatchMode = CLICOMPLETIONMODE_IMAGES;
            }
        }
    }

    if (data.CLImatchMode == CLICOMPLETIONMODE_FILES)
    {
        /* Use standard readline filename completion */
        matches = rl_completion_matches(
            (char *) text, (rl_compentry_func_t *) rl_filename_completion_function);
    }
    else
    {
        /* Use custom generator for commands, images, fps parameters, etc. */
        matches = rl_completion_matches((char *) text, &CLI_generator);
    }

    /* Prevent readline from falling back to default filename completion
     * when our custom generators return NULL. */
    rl_attempted_completion_over = 1;

    /* Reset append char based on completion mode */
    if (data.CLImatchMode == CLICOMPLETIONMODE_FILES)
    {
        rl_completion_append_character = '\0';
    }
    else if (data.CLImatchMode == CLICOMPLETIONMODE_VARS_FPS)
    {
        const char *d1 = strchr(text, '.');
        const char *d2 = (d1 != NULL) ? strchr(d1 + 1, '.') : NULL;
        rl_completion_append_character = (d2 != NULL) ? ' ' : '\0';
    }
    else if (data.CLImatchMode == CLICOMPLETIONMODE_VARS_STREAM)
    {
        const char *d1 = strchr(text, '.');
        const char *d2 = (d1 != NULL) ? strchr(d1 + 1, '.') : NULL;
        rl_completion_append_character = (d2 != NULL) ? ' ' : '\0';
    }
    else if (data.CLImatchMode == CLICOMPLETIONMODE_VARS_SEQ)
    {
        rl_completion_append_character = '\0';
    }
    else
    {
        rl_completion_append_character = ' ';
    }

    return matches;
}
#endif
