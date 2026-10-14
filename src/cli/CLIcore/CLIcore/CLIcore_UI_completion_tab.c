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
#include <termios.h>
#include <sys/ioctl.h>
#include <sys/select.h>

#ifdef USE_READLINE
#    include <readline/history.h>
#    include <readline/readline.h>
#endif

#include <ctype.h>

#include "CLIcore.h"
#include "CLIcore_script.h"
#include "CLIcore_UI_execute.h"
#include "fps_connect.h"
#include "processinfo.h"
#include "processinfo_shm_list_create.h"
#include "treesitter/cli_treesitter.h"

extern char **environ;

#ifdef USE_READLINE

static int g_comp_start         = 0;
static int g_comp_end           = 0;
static int g_menu_just_selected = 0;

/**
 * @brief Custom readline getc function to reset menu-completion state
 *
 * When the interactive menu finishes, g_menu_just_selected is set.
 * On the next key read by readline, this hook resets rl_last_func to NULL,
 * preventing rl_menu_complete from continuing to cycle previous matches.
 */
int cli_custom_getc(FILE *stream)
{
    if (g_menu_just_selected)
    {
        rl_last_func         = NULL;
        g_menu_just_selected = 0;
    }
    return rl_getc(stream);
}

static int menu_readchar(void)
{
    unsigned char ch;
    ssize_t       n = read(STDIN_FILENO, &ch, 1);
    return (n == 1) ? (int) ch : -1;
}

static int menu_stdin_wait_ms(int ms)
{
    fd_set         fds;
    struct timeval tv;

    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    tv.tv_sec  = 0;
    tv.tv_usec = ms * 1000;
    return select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv) > 0;
}

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

#    define GEN_DUPSTR(s) (matches_found_in_pass++, dupstr(s))

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
                                                 "@h",
                                                 "@hc",
                                                 "@hm",
                                                 "@hf",
                                                 "@calc",
                                                 "@get",
                                                 "@set",
                                                 "@test",
                                                 "@exists",
                                                 "@running",
                                                 "@waitfor",
                                                 "@waitany",
                                                 "@grab",
                                                 "@on",
                                                 "@defer",
                                                 "@source",
                                                 "@streamcat",
                                                 "@streamwrite",
                                                 NULL };
        static const unsigned int nbuiltins =
            sizeof(builtins) / sizeof(builtins[0]) - 1; /* exclude NULL */

        int         has_at = (text[0] == '@');
        const char *qtext  = has_at ? (text + 1) : text;
        size_t      qlen   = has_at ? (len - 1) : len;

        /* Phase 1: registered commands */
        while (list_index < data.NBcmd)
        {
            name = data.cmd[list_index].key;
            list_index++;
            int match = 0;
            if (generator_fuzzy_pass == 0)
            {
                if (strncmp(name, qtext, qlen) == 0)
                {
                    match = 1;
                }
            }
            else if (strstr(name, qtext) != NULL)
            {
                match = 1;
            }
            if (match)
            {
                if (has_at)
                {
                    char abuf[256];
                    snprintf(abuf, sizeof(abuf), "@%s", name);
                    return (GEN_DUPSTR(abuf));
                }
                return (GEN_DUPSTR(name));
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
        /* Support @T<type>:<stream> modifier completion */
        if (strncmp(text, "@T", 2) == 0 && strchr(text, ':') == NULL)
        {
            static const char *const tmods[] = { "@TU8:",  "@TI8:",   "@TU16:", "@TI16:", "@TU32:",
                                                 "@TI32:", "@TU64:",  "@TI64:", "@TF32:", "@TF64:",
                                                 "@TC64:", "@TC128:", NULL };
            while (tmods[list_index] != NULL)
            {
                const char *tm = tmods[list_index++];
                if (strncmp(tm, text, len) == 0)
                {
                    return GEN_DUPSTR((char *) tm);
                }
            }
            return NULL;
        }

        const char *colon    = (strncmp(text, "@T", 2) == 0) ? strchr(text, ':') : NULL;
        const char *stext    = (colon != NULL) ? (colon + 1) : text;
        size_t      slen     = strlen(stext);
        int         tpreflen = (colon != NULL) ? (int) (colon - text + 1) : 0;

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
                    if (namelen > 200)
                    {
                        namelen = 200;
                    }
                    strncpy(imgname, ent->d_name, namelen);
                    imgname[namelen] = '\0';

                    int match = 0;
                    if (generator_fuzzy_pass == 0)
                    {
                        if (strncmp(imgname, stext, slen) == 0)
                        {
                            match = 1;
                        }
                    }
                    else if (strstr(imgname, stext) != NULL)
                    {
                        match = 1;
                    }
                    if (match)
                    {
                        if (tpreflen > 0)
                        {
                            char full[300];
                            snprintf(full, sizeof(full), "%.*s%s", tpreflen, text, imgname);
                            return (GEN_DUPSTR(full));
                        }
                        return (GEN_DUPSTR(imgname));
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
                char *ext = strstr(ent->d_name, ".fps.shm");
                if (ext != NULL && strcmp(ext, ".fps.shm") == 0)
                {
                    char fpsname[256];
                    int  namelen = ext - ent->d_name;
                    if (namelen > 255)
                    {
                        namelen = 255;
                    }
                    strncpy(fpsname, ent->d_name, namelen);
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
        const char *dot1      = strchr(text, '.');
        const char *dot2      = (dot1 != NULL) ? strchr(dot1 + 1, '.') : NULL;
        int         has_brace = (text[0] == '$' && text[1] == '{');

        if (dot2 != NULL)
        {
            static FPS  fps;
            static int  fps_connected = 0;
            static int  param_idx     = 0;
            static char fps_target[128];
            static char param_prefix[128];

            if (!state)
            {
                param_idx    = 0;
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
                fps_connected =
                    (fps_connect(fps_target, &fps, FPSCONNECT_SIMPLE) != -1 && fps.parray != NULL);
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

                    const char *pname      = fps.parray[pi].keywordfull;
                    size_t      prefix_len = strlen(fps.md->name);
                    if (strncmp(pname, fps.md->name, prefix_len) == 0 && pname[prefix_len] == '.')
                    {
                        pname += prefix_len + 1;
                    }
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
                        int  nslen = (dot1 != NULL) ? (int) (dot1 - text) : 4;
                        if (has_brace)
                        {
                            snprintf(buf, sizeof(buf), "%.*s.%s.%s}", nslen, text, fps_target,
                                     pname);
                        }
                        else
                        {
                            snprintf(buf, sizeof(buf), "%.*s.%s.%s", nslen, text, fps_target,
                                     pname);
                        }
                        return GEN_DUPSTR(buf);
                    }
                }
            }
        }
        else
        {
            static DIR *vfps_dirp = NULL;
            static char prev_fps[256][64];
            static int  nb_prev_fps = 0;

            if (!state)
            {
                if (vfps_dirp != NULL)
                {
                    closedir(vfps_dirp);
                    vfps_dirp = NULL;
                }
                vfps_dirp   = opendir(dcshmdir);
                nb_prev_fps = 0;
            }
            if (vfps_dirp != NULL)
            {
                struct dirent *ent;
                while ((ent = readdir(vfps_dirp)) != NULL)
                {
                    char *ext = strstr(ent->d_name, ".fps.shm");
                    if (ext != NULL && strcmp(ext, ".fps.shm") == 0)
                    {
                        char fpsname[256];
                        int  namelen = ext - ent->d_name;
                        if (namelen > 200)
                        {
                            namelen = 200;
                        }
                        int nslen = (dot1 != NULL) ? (int) (dot1 - text) : 4;
                        snprintf(fpsname, sizeof(fpsname), "%.*s.%.*s.", nslen, text, namelen,
                                 ent->d_name);

                        int is_dup = 0;
                        for (int pi = 0; pi < nb_prev_fps; pi++)
                        {
                            if (strcmp(prev_fps[pi], fpsname) == 0)
                            {
                                is_dup = 1;
                                break;
                            }
                        }
                        if (is_dup)
                        {
                            continue;
                        }
                        if (nb_prev_fps < 256)
                        {
                            strncpy(prev_fps[nb_prev_fps], fpsname, sizeof(prev_fps[0]) - 1);
                            prev_fps[nb_prev_fps][sizeof(prev_fps[0]) - 1] = '\0';
                            nb_prev_fps++;
                        }

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
                closedir(vfps_dirp);
                vfps_dirp = NULL;
            }
        }
    }

    if (data.CLImatchMode == CLICOMPLETIONMODE_VARS_SEQ)
    {
        const char *dot1      = strchr(text, '.');
        const char *dot2      = (dot1 != NULL) ? strchr(dot1 + 1, '.') : NULL;
        int         has_brace = (text[0] == '$' && text[1] == '{');

        if (dot2 != NULL)
        {
            static int         seq_prop_idx = 0;
            static const char *seq_props[]  = {
                "status", "tasks", "errors", "pid", "completed", NULL
            };
            static char seq_target[128];
            static char seq_prop_prefix[64];

            if (!state)
            {
                seq_prop_idx = 0;
                size_t snlen = (size_t) (dot2 - (dot1 + 1));
                if (snlen >= sizeof(seq_target))
                {
                    snlen = sizeof(seq_target) - 1;
                }
                strncpy(seq_target, dot1 + 1, snlen);
                seq_target[snlen] = '\0';

                strncpy(seq_prop_prefix, dot2 + 1, sizeof(seq_prop_prefix) - 1);
                seq_prop_prefix[sizeof(seq_prop_prefix) - 1] = '\0';
            }

            while (seq_props[seq_prop_idx] != NULL)
            {
                const char *sprop = seq_props[seq_prop_idx++];
                if (strncmp(sprop, seq_prop_prefix, strlen(seq_prop_prefix)) == 0)
                {
                    char buf[256];
                    int  nslen = (dot1 != NULL) ? (int) (dot1 - text) : 2;
                    if (has_brace)
                    {
                        snprintf(buf, sizeof(buf), "%.*s.%s.%s}", nslen, text, seq_target, sprop);
                    }
                    else
                    {
                        snprintf(buf, sizeof(buf), "%.*s.%s.%s", nslen, text, seq_target, sprop);
                    }
                    return GEN_DUPSTR(buf);
                }
            }
        }
        else
        {
            static DIR *vseq_dirp = NULL;
            static char prev_seq[256][64];
            static int  nb_prev_seq = 0;

            if (!state)
            {
                if (vseq_dirp != NULL)
                {
                    closedir(vseq_dirp);
                    vseq_dirp = NULL;
                }
                vseq_dirp   = opendir(dcshmdir);
                nb_prev_seq = 0;
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
                            if (namelen > 200)
                            {
                                namelen = 200;
                            }
                            int nslen = (dot1 != NULL) ? (int) (dot1 - text) : 4;
                            snprintf(seqname, sizeof(seqname), "%.*s.%.*s.", nslen, text, namelen,
                                     ent->d_name + 4);

                            int is_dup = 0;
                            for (int pi = 0; pi < nb_prev_seq; pi++)
                            {
                                if (strcmp(prev_seq[pi], seqname) == 0)
                                {
                                    is_dup = 1;
                                    break;
                                }
                            }
                            if (is_dup)
                            {
                                continue;
                            }
                            if (nb_prev_seq < 256)
                            {
                                strncpy(prev_seq[nb_prev_seq], seqname, sizeof(prev_seq[0]) - 1);
                                prev_seq[nb_prev_seq][sizeof(prev_seq[0]) - 1] = '\0';
                                nb_prev_seq++;
                            }

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
    }

    if (data.CLImatchMode == CLICOMPLETIONMODE_VARS_STREAM)
    {
        const char *dot1      = strchr(text, '.');
        const char *dot2      = (dot1 != NULL) ? strchr(dot1 + 1, '.') : NULL;
        int         has_brace = (text[0] == '$' && text[1] == '{');

        if (dot2 != NULL)
        {
            static int         prop_idx       = 0;
            static const char *stream_props[] = { "xsize", "ysize",    "zsize", "naxis",
                                                  "type",  "typename", "nelem", "cnt0",
                                                  "cnt1",  "sem",      "kw.",   NULL };
            static char        stream_target[128];
            static char        prop_prefix[64];

            if (!state)
            {
                prop_idx     = 0;
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
                int         match = 0;
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
                    int  nslen = (dot1 != NULL) ? (int) (dot1 - text) : 2;
                    if (has_brace)
                    {
                        snprintf(buf, sizeof(buf), "%.*s.%s.%s}", nslen, text, stream_target,
                                 sprop);
                    }
                    else
                    {
                        snprintf(buf, sizeof(buf), "%.*s.%s.%s", nslen, text, stream_target, sprop);
                    }
                    return GEN_DUPSTR(buf);
                }
            }
        }
        else
        {
            static DIR *vstream_dirp = NULL;
            static char prev_streams[256][64];
            static int  nb_prev_streams = 0;

            if (!state)
            {
                if (vstream_dirp != NULL)
                {
                    closedir(vstream_dirp);
                    vstream_dirp = NULL;
                }
                vstream_dirp    = opendir(dcshmdir);
                nb_prev_streams = 0;
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
                        if (namelen > 200)
                        {
                            namelen = 200;
                        }
                        int nslen = (dot1 != NULL) ? (int) (dot1 - text) : 2;
                        snprintf(sname, sizeof(sname), "%.*s.%.*s.", nslen, text, namelen,
                                 ent->d_name);

                        int is_dup = 0;
                        for (int pi = 0; pi < nb_prev_streams; pi++)
                        {
                            if (strcmp(prev_streams[pi], sname) == 0)
                            {
                                is_dup = 1;
                                break;
                            }
                        }
                        if (is_dup)
                        {
                            continue;
                        }
                        if (nb_prev_streams < 256)
                        {
                            strncpy(prev_streams[nb_prev_streams], sname,
                                    sizeof(prev_streams[0]) - 1);
                            prev_streams[nb_prev_streams][sizeof(prev_streams[0]) - 1] = '\0';
                            nb_prev_streams++;
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

    if (data.CLImatchMode == CLICOMPLETIONMODE_VARS_PROC)
    {
        const char *dot1      = strchr(text, '.');
        const char *dot2      = (dot1 != NULL) ? strchr(dot1 + 1, '.') : NULL;
        int         has_brace = (text[0] == '$' && text[1] == '{');

        if (dot2 != NULL)
        {
            static int         proc_prop_idx = 0;
            static const char *proc_props[]  = { "pid",      "loopstat",    "loopcnt",
                                                 "loopfreq", "exectime",    "rtprio",
                                                 "ctrlval",  "trigmode",    "statusmsg",
                                                 "tmux",     "description", "missedframes",
                                                 NULL };
            static char        proc_target[128];
            static char        proc_prop_prefix[64];

            if (!state)
            {
                proc_prop_idx = 0;
                size_t pnlen  = (size_t) (dot2 - (dot1 + 1));
                if (pnlen >= sizeof(proc_target))
                {
                    pnlen = sizeof(proc_target) - 1;
                }
                strncpy(proc_target, dot1 + 1, pnlen);
                proc_target[pnlen] = '\0';

                strncpy(proc_prop_prefix, dot2 + 1, sizeof(proc_prop_prefix) - 1);
                proc_prop_prefix[sizeof(proc_prop_prefix) - 1] = '\0';
            }

            while (proc_props[proc_prop_idx] != NULL)
            {
                const char *pprop = proc_props[proc_prop_idx++];
                int         match = 0;
                if (generator_fuzzy_pass == 0)
                {
                    if (strncmp(pprop, proc_prop_prefix, strlen(proc_prop_prefix)) == 0)
                    {
                        match = 1;
                    }
                }
                else if (strstr(pprop, proc_prop_prefix) != NULL)
                {
                    match = 1;
                }

                if (match)
                {
                    char buf[256];
                    int  nslen = (dot1 != NULL) ? (int) (dot1 - text) : 4;
                    if (has_brace)
                    {
                        snprintf(buf, sizeof(buf), "%.*s.%s.%s}", nslen, text, proc_target, pprop);
                    }
                    else
                    {
                        snprintf(buf, sizeof(buf), "%.*s.%s.%s", nslen, text, proc_target, pprop);
                    }
                    return GEN_DUPSTR(buf);
                }
            }
        }
        else
        {
            static int pinfo_idx = 0;
            if (!state)
            {
                pinfo_idx = 0;
                if (pinfolist == NULL)
                {
                    long pidx_unused = -1;
                    processinfo_shm_list_create(&pidx_unused);
                }
            }

            if (pinfolist != NULL)
            {
                while (pinfo_idx < PROCESSINFOLISTSIZE)
                {
                    int pi = pinfo_idx++;
                    if (!pinfolist->active[pi])
                    {
                        continue;
                    }

                    const char *pname = pinfolist->pnamearray[pi];
                    if (pname == NULL || pname[0] == '\0')
                    {
                        continue;
                    }

                    int duplicate = 0;
                    for (int prev = 0; prev < pi; prev++)
                    {
                        if (pinfolist->active[prev] &&
                            strcmp(pinfolist->pnamearray[prev], pname) == 0)
                        {
                            duplicate = 1;
                            break;
                        }
                    }
                    if (duplicate)
                    {
                        continue;
                    }

                    char procname[256];
                    int  nslen = (dot1 != NULL) ? (int) (dot1 - text) : 4;
                    snprintf(procname, sizeof(procname), "%.*s.%s.", nslen, text, pname);

                    int match = 0;
                    if (generator_fuzzy_pass == 0)
                    {
                        if (strncmp(procname, text, len) == 0)
                        {
                            match = 1;
                        }
                    }
                    else if (strstr(procname, text) != NULL)
                    {
                        match = 1;
                    }

                    if (match)
                    {
                        return GEN_DUPSTR(procname);
                    }
                }
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
            vpreflen                     = (int) strlen(vprefix);
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
            int         match = 0;
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
            int         match = 0;
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
            const char *eq    = strchr(entry, '=');
            if (eq == NULL)
            {
                continue;
            }

            char   ename[256];
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

#    undef GEN_DUPSTR

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
char **CLI_completion(const char *text, int start, int end)
{
    g_comp_start = start;
    g_comp_end   = end;

    char **matches      = NULL;
    char   cmdname[128] = "";
    int    argidx       = 0;

    int mode = cli_ts_determine_completion_mode(rl_line_buffer, start, text, cmdname,
                                                sizeof(cmdname), &argidx);

    if (mode >= 0)
    {
        data.CLImatchMode = mode;
    }
    else
    {
        const char *barecmd = (cmdname[0] == '@') ? (cmdname + 1) : cmdname;
        /* A command was found (cmdname), and cursor is on argument argidx */
        if (text[0] == '.' && text[1] != '/' && text[1] != '.')
        {
            data.CLImatchMode = CLICOMPLETIONMODE_CMDARGS;
        }
        else if (strcmp(barecmd, "loadfits") == 0 || strcmp(barecmd, "savefits") == 0 ||
                 strcmp(barecmd, "saveFITS") == 0 || strcmp(barecmd, "source") == 0 ||
                 strcmp(barecmd, ".") == 0 || strcmp(barecmd, "cat") == 0 ||
                 strcmp(barecmd, "cd") == 0 || strcmp(barecmd, "ls") == 0 ||
                 strcmp(barecmd, "vi") == 0 || strcmp(barecmd, "vim") == 0 ||
                 strcmp(barecmd, "nano") == 0 || strcmp(barecmd, "head") == 0 ||
                 strcmp(barecmd, "tail") == 0 || strcmp(barecmd, "cp") == 0 ||
                 strcmp(barecmd, "mv") == 0 || strcmp(barecmd, "rm") == 0 ||
                 strcmp(barecmd, "less") == 0 || strcmp(barecmd, "more") == 0 ||
                 strcmp(barecmd, "include_once") == 0 || strcmp(barecmd, "savescript") == 0 ||
                 strcmp(barecmd, "savehistory") == 0 || strcmp(barecmd, "run") == 0)
        {
            data.CLImatchMode = CLICOMPLETIONMODE_FILES;
        }
        else if (strcmp(barecmd, "fpsCTRL") == 0 || strcmp(barecmd, "fparam") == 0 ||
                 strcmp(barecmd, "fpsload") == 0 || strcmp(barecmd, "dpsingle") == 0 ||
                 strcmp(barecmd, "fpsconf") == 0 || strcmp(barecmd, "fpsrun") == 0 ||
                 strcmp(barecmd, "fpsstop") == 0 || strcmp(barecmd, "waitfor_fps") == 0)
        {
            data.CLImatchMode = CLICOMPLETIONMODE_FPSPARAMS;
        }
        else if (strcmp(barecmd, "export") == 0 || strcmp(barecmd, "readonly") == 0 ||
                 strcmp(barecmd, "unset") == 0 || strcmp(barecmd, "local") == 0 ||
                 strcmp(barecmd, "declare") == 0)
        {
            data.CLImatchMode = CLICOMPLETIONMODE_VARS_ENV;
        }
        else if (strcmp(barecmd, "grab") == 0 || strcmp(barecmd, "streamcat") == 0 ||
                 strcmp(barecmd, "streamwrite") == 0)
        {
            data.CLImatchMode = CLICOMPLETIONMODE_IMAGES;
        }
        else
        {
            /* Lookup registered milk command in data.cmd */
            int cmdimatch    = find_command_match(cmdname);
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
        matches = rl_completion_matches((char *) text,
                                        (rl_compentry_func_t *) rl_filename_completion_function);
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
    else if (data.CLImatchMode == CLICOMPLETIONMODE_VARS_FPS ||
             data.CLImatchMode == CLICOMPLETIONMODE_VARS_STREAM ||
             data.CLImatchMode == CLICOMPLETIONMODE_VARS_PROC)
    {
        const char *d1                 = strchr(text, '.');
        const char *d2                 = (d1 != NULL) ? strchr(d1 + 1, '.') : NULL;
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

/**
 * @brief Interactive Zsh-style menu completion display hook
 *
 * Intercepts terminal when multiple completion matches are available,
 * displaying a live-highlighted interactive grid below the prompt.
 *
 * Navigation:
 *  - Left / Right / Up / Down / Tab / Shift-Tab
 *  - Enter: selects candidate and drops it into rl_line_buffer
 *  - Esc / Ctrl-C: cancels menu without modifying line buffer
 *
 * @param matches     Array of matches (matches[0] = common prefix, matches[1..n] = candidates)
 * @param num_matches Number of candidate matches
 * @param max_length  Length of the longest match string
 */
void cli_menu_display_matches(char **matches, int num_matches, int max_length)
{
    if (matches == NULL || num_matches <= 0)
    {
        return;
    }

    /* Deduplicate candidate list in place */
    int uniq_count = 0;
    for (int i = 1; i <= num_matches; i++)
    {
        int is_dup = 0;
        for (int j = 1; j <= uniq_count; j++)
        {
            if (strcmp(matches[i], matches[j]) == 0)
            {
                is_dup = 1;
                break;
            }
        }
        if (!is_dup)
        {
            uniq_count++;
            if (uniq_count != i)
            {
                matches[uniq_count] = matches[i];
            }
        }
        else
        {
            free(matches[i]);
            matches[i] = NULL;
        }
    }
    matches[uniq_count + 1] = NULL;
    num_matches             = uniq_count;

    if (num_matches <= 0)
    {
        return;
    }

    /* If only one unique candidate remains, accept it directly without menu */
    if (num_matches == 1)
    {
        const char *cand     = matches[1];
        int         del_from = g_comp_start;
        int         del_to   = rl_point;
        if (del_to < del_from)
        {
            del_to = del_from;
        }
        rl_delete_text(del_from, del_to);
        rl_point = del_from;
        rl_insert_text(cand);

        size_t clen = strlen(cand);
        if (clen > 0 && cand[clen - 1] != '.' && cand[clen - 1] != '/')
        {
            rl_insert_text(" ");
        }

        if (ghost_chars_on_line > 0)
        {
            printf("\033[K");
            ghost_chars_on_line = 0;
        }
        set_pending_suggestion(NULL, 0);

        g_menu_just_selected = 1;

        rl_on_new_line();
        rl_forced_update_display();
        return;
    }

    /* Erase any active ghost text on the prompt line before menu opens */
    if (ghost_chars_on_line > 0)
    {
        printf("\033[K");
        ghost_chars_on_line = 0;
    }
    set_pending_suggestion(NULL, 0);
    fflush(stdout);

    struct winsize ws;
    int            cols = 80;
    int            rows = 24;
    if (ioctl(STDIN_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0)
    {
        cols = ws.ws_col;
        rows = ws.ws_row;
    }

    if (cols < 20 || rows < 5)
    {
        return;
    }

    int col_width = max_length + 2;
    if (col_width > cols - 2)
    {
        col_width = cols - 2;
    }
    if (col_width < 4)
    {
        col_width = 4;
    }

    int ncols = cols / col_width;
    if (ncols < 1)
    {
        ncols = 1;
    }
    int total_rows = (num_matches + ncols - 1) / ncols;

    int max_visible_rows = 6;
    if (max_visible_rows > rows - 4)
    {
        max_visible_rows = rows - 4;
    }
    if (max_visible_rows < 1)
    {
        max_visible_rows = 1;
    }

    int visible_rows = total_rows < max_visible_rows ? total_rows : max_visible_rows;

    int selected          = 0;
    int first_visible_row = 0;
    int accepted          = 0;

    struct termios orig_termios, raw_termios;
    tcgetattr(STDIN_FILENO, &orig_termios);
    raw_termios = orig_termios;
    raw_termios.c_lflag &= ~(ICANON | ECHO | ISIG);
    raw_termios.c_cc[VMIN]  = 1;
    raw_termios.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &raw_termios);

    /* Hide cursor */
    printf("\033[?25l");
    fflush(stdout);

    /* Allocate menu lines below prompt */
    printf("\n");
    for (int r = 0; r < visible_rows; r++)
    {
        printf("\033[2K\n");
    }
    printf("\033[2K");
    /* Move back up to row 0 of the menu */
    printf("\033[%dA\r", visible_rows);
    fflush(stdout);

    int redraw = 1;
    while (1)
    {
        if (redraw)
        {
            int sel_row = selected / ncols;
            if (sel_row < first_visible_row)
            {
                first_visible_row = sel_row;
            }
            if (sel_row >= first_visible_row + visible_rows)
            {
                first_visible_row = sel_row - visible_rows + 1;
            }

            /* Draw candidate rows */
            for (int vr = 0; vr < visible_rows; vr++)
            {
                int r = first_visible_row + vr;
                printf("\033[2K\r");
                if (r < total_rows)
                {
                    for (int c = 0; c < ncols; c++)
                    {
                        int idx = r * ncols + c;
                        if (idx >= num_matches)
                        {
                            break;
                        }
                        const char *cand = matches[idx + 1];
                        if (idx == selected)
                        {
                            /* Live highlight: Reverse bold cyan */
                            printf("\033[7;1;36m %-*s\033[0m", col_width - 1, cand);
                        }
                        else
                        {
                            printf(" %-*s", col_width - 1, cand);
                        }
                    }
                }
                printf("\n");
            }

            /* Draw status line */
            printf("\033[2K\r");
            int has_more_above = (first_visible_row > 0);
            int has_more_below = (first_visible_row + visible_rows < total_rows);
            printf("\033[36m\xe2\x94\x80\xe2\x94\x80\033[0m \033[1m[%d/%d]\033[0m %s%s\033[2m"
                   "\xe2\x97\x84\xe2\x96\xb2\xe2\x96\xbc\xe2\x96\xba/Tab: navigate "
                   "\xe2\x80\xa2 Enter: select \xe2\x80\xa2 Esc: cancel\033[0m "
                   "\033[36m\xe2\x94\x80\xe2\x94\x80\033[0m",
                   selected + 1, num_matches, has_more_above ? "\xe2\x96\xb2 " : "",
                   has_more_below ? "\xe2\x96\xbc " : "");
            fflush(stdout);

            /* Move cursor back up to row 0 of menu for next cycle */
            printf("\033[%dA\r", visible_rows);
            fflush(stdout);
            redraw = 0;
        }

        int c = menu_readchar();
        if (c == 27)
        {
            if (menu_stdin_wait_ms(50))
            {
                int b1 = menu_readchar();
                if (b1 == '[' || b1 == 'O')
                {
                    int b2 = menu_readchar();
                    if (b2 == 'A') /* Up */
                    {
                        selected -= ncols;
                        if (selected < 0)
                        {
                            selected += total_rows * ncols;
                            if (selected >= num_matches)
                            {
                                selected -= ncols;
                            }
                        }
                        redraw = 1;
                    }
                    else if (b2 == 'B') /* Down */
                    {
                        selected += ncols;
                        if (selected >= num_matches)
                        {
                            selected %= ncols;
                        }
                        redraw = 1;
                    }
                    else if (b2 == 'C') /* Right */
                    {
                        selected = (selected + 1) % num_matches;
                        redraw   = 1;
                    }
                    else if (b2 == 'D') /* Left */
                    {
                        selected = (selected - 1 + num_matches) % num_matches;
                        redraw   = 1;
                    }
                    else if (b2 == 'Z') /* Shift-Tab */
                    {
                        selected = (selected - 1 + num_matches) % num_matches;
                        redraw   = 1;
                    }
                    else if (b2 == '5') /* PgUp */
                    {
                        menu_readchar(); /* consume '~' */
                        selected -= (ncols * visible_rows);
                        if (selected < 0)
                        {
                            selected = 0;
                        }
                        redraw = 1;
                    }
                    else if (b2 == '6') /* PgDn */
                    {
                        menu_readchar(); /* consume '~' */
                        selected += (ncols * visible_rows);
                        if (selected >= num_matches)
                        {
                            selected = num_matches - 1;
                        }
                        redraw = 1;
                    }
                }
            }
            else
            {
                /* Bare ESC: Cancel */
                accepted = 0;
                break;
            }
        }
        else if (c == '\t')
        {
            selected = (selected + 1) % num_matches;
            redraw   = 1;
        }
        else if (c == '\r' || c == '\n')
        {
            accepted = 1;
            break;
        }
        else if (c == 3 || c == 4 || c == 7) /* Ctrl+C, Ctrl+D, Ctrl+G */
        {
            accepted = 0;
            break;
        }
        else
        {
            /* Any other key cancels menu and passes character back to readline */
            accepted = 0;
            if (c > 0 && c != 27)
            {
                rl_stuff_char(c);
            }
            break;
        }
    }

    /* Clean up menu display */
    for (int vr = 0; vr < visible_rows; vr++)
    {
        printf("\033[2K\n");
    }
    printf("\033[2K");
    /* Move back up to prompt line */
    printf("\033[%dA\r", visible_rows + 1);
    /* Clear prompt line cleanly so readline can redraw from column 0 */
    printf("\033[2K\r");
    fflush(stdout);

    /* Restore terminal */
    tcsetattr(STDIN_FILENO, TCSANOW, &orig_termios);
    printf("\033[?25h"); /* Show cursor */
    fflush(stdout);

    if (accepted && selected >= 0 && selected < num_matches)
    {
        const char *cand     = matches[selected + 1];
        int         del_from = g_comp_start;
        int         del_to   = rl_point;
        if (del_to < del_from)
        {
            del_to = del_from;
        }
        rl_delete_text(del_from, del_to);
        rl_point = del_from;
        rl_insert_text(cand);

        size_t clen = strlen(cand);
        if (clen > 0 && cand[clen - 1] != '.' && cand[clen - 1] != '/')
        {
            rl_insert_text(" ");
        }
    }

    if (ghost_chars_on_line > 0)
    {
        ghost_chars_on_line = 0;
    }
    set_pending_suggestion(NULL, 0);

    g_menu_just_selected = 1;

    rl_on_new_line();
    rl_forced_update_display();
}
#endif

#ifndef USE_READLINE
void cli_menu_display_matches(char **matches, int num_matches, int max_length)
{
    (void) matches;
    (void) num_matches;
    (void) max_length;
}

int cli_custom_getc(FILE *stream)
{
    return fgetc(stream);
}
#endif
