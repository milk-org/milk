// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_UI_completion_tab.c
 * @brief Readline tab-completion candidate generator and context dispatcher.
 *
 * Role & Goal:
 *   Implements `CLI_generator()` and `CLI_completion()` for the `milk-cli`
 *   interactive prompt. Generates prefix and fuzzy-fallback completion candidates
 *   across registered commands, shared-memory streams, FPS instances/parameters,
 *   sequencer/processinfo properties, shell/environment variables, and files.
 *
 * @see CLIcore_UI_completion_menu.c for the interactive Zsh-style match grid.
 * @see CLIcore_UI_completion.c for prompt rendering and Levenshtein distance.
 */

#include <ctype.h>
#include <dirent.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef USE_READLINE
#    include <readline/history.h>
#    include <readline/readline.h>
#endif

#include "CLIcore.h"
#include "CLIcore_UI_execute.h"
#include "CLIcore_script.h"
#include "fps_connect.h"
#include "processinfo.h"
#include "processinfo_shm_list_create.h"
#include "treesitter/cli_treesitter.h"

extern char **environ;

#ifdef USE_READLINE

int cli_comp_start       = 0;
int cli_comp_end         = 0;
int generator_fuzzy_pass = 0;

/**
 * @brief Return 1 if @candidate matches @query under the current prefix/fuzzy pass.
 */
static inline int match_prefix_or_fuzzy(const char *candidate, const char *query, size_t qlen)
{
    if (generator_fuzzy_pass == 0)
    {
        return (strncmp(candidate, query, qlen) == 0) ? 1 : 0;
    }
    return (strstr(candidate, query) != NULL) ? 1 : 0;
}

/**
 * @brief Generate command and built-in keyword candidates (`CLICOMPLETIONMODE_COMMANDS`).
 */
static char *gen_commands_match(const char *text, size_t len, unsigned int *list_index)
{
    static const char *const builtins[] = {
        "if",           "elif",       "else",           "fi",           "for",       "while",
        "until",        "do",         "done",           "case",         "esac",      "select",
        "function",     ".",          "source",         "break",        "continue",  "return",
        "true",         "false",      "exit",           "shift",        "assert",    "assigncheck",
        "dpdigits",     "set",        "export",         "readonly",     "local",     "declare",
        "let",          "eval",       "type",           "command",      "trap",      "watch",
        "time",         "timeout",    "wait",           "wait_any",     "printf",    "echo",
        "getopts",      "mapfile",    "alias",          "unalias",      "basename",  "dirname",
        "pushd",        "popd",       "dirs",           "seq",          "[[",        "procctl",
        "procwait",     "procstat",   "waitfor_stream", "waitfor_fps",  "on_update", "on_fpschange",
        "include_once", "savescript", "savehistory",    "@h",           "@hb",       "@hc",
        "@hm",          "@hf",        "@calc",          "@get",         "@set",      "@test",
        "@exists",      "@running",   "@waitfor",       "@waitany",     "@grab",     "@on",
        "@defer",       "@source",    "@streamcat",     "@streamwrite", NULL
    };
    static const unsigned int nbuiltins = sizeof(builtins) / sizeof(builtins[0]) - 1;

    int         has_at = (text[0] == '@');
    const char *qtext  = has_at ? (text + 1) : text;
    size_t      qlen   = has_at ? (len - 1) : len;

    while (*list_index < data.NBcmd)
    {
        const char *name = data.cmd[*list_index].key;
        (*list_index)++;
        if (match_prefix_or_fuzzy(name, qtext, qlen))
        {
            if (has_at)
            {
                char abuf[256];
                snprintf(abuf, sizeof(abuf), "@%s", name);
                return dupstr(abuf);
            }
            return dupstr((char *) name);
        }
    }

    unsigned int bi = *list_index - data.NBcmd;
    while (bi < nbuiltins)
    {
        const char *name = builtins[bi];
        (*list_index)++;
        bi++;
        if (match_prefix_or_fuzzy(name, text, len))
        {
            return dupstr((char *) name);
        }
    }
    return NULL;
}

/**
 * @brief Generate shared-memory stream and `@T<type>:` candidates (`CLICOMPLETIONMODE_IMAGES`).
 */
static char *gen_images_match(const char *text, size_t len, int state, unsigned int *list_index)
{
    if (strncmp(text, "@T", 2) == 0 && strchr(text, ':') == NULL)
    {
        static const char *const tmods[] = { "@TU8:",  "@TI8:",   "@TU16:", "@TI16:", "@TU32:",
                                             "@TI32:", "@TU64:",  "@TI64:", "@TF32:", "@TF64:",
                                             "@TC64:", "@TC128:", NULL };
        while (tmods[*list_index] != NULL)
        {
            const char *tm = tmods[(*list_index)++];
            if (strncmp(tm, text, len) == 0)
            {
                return dupstr((char *) tm);
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
        }
        img_dirp = opendir(dcshmdir);
    }
    if (img_dirp == NULL)
    {
        return NULL;
    }

    struct dirent *ent;
    while ((ent = readdir(img_dirp)) != NULL)
    {
        char *ext = strstr(ent->d_name, ".im.shm");
        if (ext != NULL && strcmp(ext, ".im.shm") == 0)
        {
            char imgname[256];
            int  namelen = (int) (ext - ent->d_name);
            if (namelen > 200)
            {
                namelen = 200;
            }
            strncpy(imgname, ent->d_name, (size_t) namelen);
            imgname[namelen] = '\0';

            if (match_prefix_or_fuzzy(imgname, stext, slen))
            {
                if (tpreflen > 0)
                {
                    char full[300];
                    snprintf(full, sizeof(full), "%.*s%s", tpreflen, text, imgname);
                    return dupstr(full);
                }
                return dupstr(imgname);
            }
        }
    }
    closedir(img_dirp);
    img_dirp = NULL;
    return NULL;
}

/**
 * @brief Generate command argument tag candidates (`CLICOMPLETIONMODE_CMDARGS`).
 */
static char *gen_cmdargs_match(const char *text, size_t len, unsigned int *list_index)
{
    if (data.cmd[data.cmdindex].argdata == NULL)
    {
        return NULL;
    }
    while ((int) *list_index < data.cmd[data.cmdindex].nbarg)
    {
        char *name = data.cmd[data.cmdindex].argdata[*list_index].fpstag;
        (*list_index)++;
        if (name != NULL && strncmp(name, text, len) == 0)
        {
            return dupstr(name);
        }
    }
    return NULL;
}

/**
 * @brief Generate filesystem path candidates (`CLICOMPLETIONMODE_FILES`).
 */
static char *gen_files_match(const char *text, int state)
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
            memcpy(dirpart, text, (size_t) dlen);
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
        preflen = (unsigned int) strlen(prefix);
        dirp    = opendir(dirpart);
    }

    if (dirp == NULL)
    {
        return NULL;
    }

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
            return dupstr(result);
        }
    }
    closedir(dirp);
    dirp = NULL;
    return NULL;
}

/**
 * @brief Generate FPS instance name candidates (`CLICOMPLETIONMODE_FPSPARAMS`).
 */
static char *gen_fpsparams_match(const char *text, size_t len, int state)
{
    static DIR *fps_dirp = NULL;
    if (!state)
    {
        if (fps_dirp != NULL)
        {
            closedir(fps_dirp);
        }
        fps_dirp = opendir(dcshmdir);
    }
    if (fps_dirp == NULL)
    {
        return NULL;
    }

    struct dirent *ent;
    while ((ent = readdir(fps_dirp)) != NULL)
    {
        char *ext = strstr(ent->d_name, ".fps.shm");
        if (ext != NULL && strcmp(ext, ".fps.shm") == 0)
        {
            char fpsname[256];
            int  namelen = (int) (ext - ent->d_name);
            if (namelen > 255)
            {
                namelen = 255;
            }
            strncpy(fpsname, ent->d_name, (size_t) namelen);
            fpsname[namelen] = '\0';
            if (match_prefix_or_fuzzy(fpsname, text, len))
            {
                return dupstr(fpsname);
            }
        }
    }
    closedir(fps_dirp);
    fps_dirp = NULL;
    return NULL;
}

/**
 * @brief Format a 3-segment dotted property candidate (`$ns.target.prop` or `${ns.target.prop}`).
 */
static char *format_dotted_prop_candidate(const char *text,
                                          const char *dot1,
                                          int         default_nslen,
                                          int         has_brace,
                                          const char *target,
                                          const char *prop)
{
    char buf[256];
    int  nslen = (dot1 != NULL) ? (int) (dot1 - text) : default_nslen;
    snprintf(buf, sizeof(buf), has_brace ? "%.*s.%s.%s}" : "%.*s.%s.%s", nslen, text, target, prop);
    return dupstr(buf);
}

/**
 * @brief Shared helper to iterate static property names after the second dot (`dot2 != NULL`).
 */
static char *gen_static_props_after_dot2(const char        *text,
                                         const char        *dot1,
                                         const char        *dot2,
                                         int                default_nslen,
                                         int                has_brace,
                                         int                state,
                                         const char *const *props,
                                         int               *prop_idx,
                                         char              *target_buf,
                                         size_t             target_sz,
                                         char              *prefix_buf,
                                         size_t             prefix_sz)
{
    if (!state)
    {
        *prop_idx    = 0;
        size_t tnlen = (size_t) (dot2 - (dot1 + 1));
        if (tnlen >= target_sz)
        {
            tnlen = target_sz - 1;
        }
        strncpy(target_buf, dot1 + 1, tnlen);
        target_buf[tnlen] = '\0';
        strncpy(prefix_buf, dot2 + 1, prefix_sz - 1);
        prefix_buf[prefix_sz - 1] = '\0';
    }

    size_t preflen = strlen(prefix_buf);
    while (props[*prop_idx] != NULL)
    {
        const char *prop = props[(*prop_idx)++];
        if (match_prefix_or_fuzzy(prop, prefix_buf, preflen))
        {
            return format_dotted_prop_candidate(text, dot1, default_nslen, has_brace, target_buf,
                                                prop);
        }
    }
    return NULL;
}

/**
 * @brief Shared helper to scan `dcshmdir` for `<prefix><name><suffix>` entities before the second dot.
 */
static char *gen_shm_entities_before_dot2(const char *text,
                                          size_t      len,
                                          const char *dot1,
                                          int         default_nslen,
                                          int         state,
                                          const char *file_prefix,
                                          const char *file_suffix,
                                          DIR       **dirp,
                                          char        prev_names[256][64],
                                          int        *nb_prev)
{
    if (!state)
    {
        if (*dirp != NULL)
        {
            closedir(*dirp);
        }
        *dirp    = opendir(dcshmdir);
        *nb_prev = 0;
    }
    if (*dirp == NULL)
    {
        return NULL;
    }

    size_t         fplen = (file_prefix != NULL) ? strlen(file_prefix) : 0;
    struct dirent *ent;
    while ((ent = readdir(*dirp)) != NULL)
    {
        if (fplen > 0 && strncmp(ent->d_name, file_prefix, fplen) != 0)
        {
            continue;
        }
        char *ext = strstr(ent->d_name + fplen, file_suffix);
        if (ext == NULL || strcmp(ext, file_suffix) != 0)
        {
            continue;
        }

        char cand[256];
        int  namelen = (int) (ext - (ent->d_name + fplen));
        if (namelen > 200)
        {
            namelen = 200;
        }
        int nslen = (dot1 != NULL) ? (int) (dot1 - text) : default_nslen;
        snprintf(cand, sizeof(cand), "%.*s.%.*s.", nslen, text, namelen, ent->d_name + fplen);

        int is_dup = 0;
        for (int pi = 0; pi < *nb_prev; pi++)
        {
            if (strcmp(prev_names[pi], cand) == 0)
            {
                is_dup = 1;
                break;
            }
        }
        if (is_dup)
        {
            continue;
        }
        if (*nb_prev < 256)
        {
            strncpy(prev_names[*nb_prev], cand, sizeof(prev_names[0]) - 1);
            prev_names[*nb_prev][sizeof(prev_names[0]) - 1] = '\0';
            (*nb_prev)++;
        }
        if (match_prefix_or_fuzzy(cand, text, len))
        {
            return dupstr(cand);
        }
    }
    closedir(*dirp);
    *dirp = NULL;
    return NULL;
}

/**
 * @brief Generate `$fps.<name>.<param>` candidates (`CLICOMPLETIONMODE_VARS_FPS`).
 */
static char *gen_vars_fps_match(const char *text, size_t len, int state)
{
    const char *dot1      = strchr(text, '.');
    const char *dot2      = (dot1 != NULL) ? strchr(dot1 + 1, '.') : NULL;
    int         has_brace = (text[0] == '$' && text[1] == '{');

    if (dot2 == NULL)
    {
        static DIR *vfps_dirp = NULL;
        static char prev_fps[256][64];
        static int  nb_prev_fps = 0;
        return gen_shm_entities_before_dot2(text, len, dot1, 4, state, NULL, ".fps.shm", &vfps_dirp,
                                            prev_fps, &nb_prev_fps);
    }

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

    if (!fps_connected)
    {
        return NULL;
    }

    size_t preflen = strlen(param_prefix);
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
        if (match_prefix_or_fuzzy(pname, param_prefix, preflen))
        {
            return format_dotted_prop_candidate(text, dot1, 4, has_brace, fps_target, pname);
        }
    }
    return NULL;
}

/**
 * @brief Generate `$seq.<name>.<prop>` candidates (`CLICOMPLETIONMODE_VARS_SEQ`).
 */
static char *gen_vars_seq_match(const char *text, size_t len, int state)
{
    const char *dot1      = strchr(text, '.');
    const char *dot2      = (dot1 != NULL) ? strchr(dot1 + 1, '.') : NULL;
    int         has_brace = (text[0] == '$' && text[1] == '{');

    if (dot2 == NULL)
    {
        static DIR *vseq_dirp = NULL;
        static char prev_seq[256][64];
        static int  nb_prev_seq = 0;
        return gen_shm_entities_before_dot2(text, len, dot1, 4, state, "seq.", ".shm", &vseq_dirp,
                                            prev_seq, &nb_prev_seq);
    }

    static const char *const seq_props[] = {
        "status", "tasks", "errors", "pid", "completed", NULL
    };
    static int  seq_prop_idx = 0;
    static char seq_target[128];
    static char seq_prop_prefix[64];
    return gen_static_props_after_dot2(text, dot1, dot2, 2, has_brace, state, seq_props,
                                       &seq_prop_idx, seq_target, sizeof(seq_target),
                                       seq_prop_prefix, sizeof(seq_prop_prefix));
}

/**
 * @brief Generate `$s.<name>.<prop>` candidates (`CLICOMPLETIONMODE_VARS_STREAM`).
 */
static char *gen_vars_stream_match(const char *text, size_t len, int state)
{
    const char *dot1      = strchr(text, '.');
    const char *dot2      = (dot1 != NULL) ? strchr(dot1 + 1, '.') : NULL;
    int         has_brace = (text[0] == '$' && text[1] == '{');

    if (dot2 == NULL)
    {
        static DIR *vstream_dirp = NULL;
        static char prev_streams[256][64];
        static int  nb_prev_streams = 0;
        return gen_shm_entities_before_dot2(text, len, dot1, 2, state, NULL, ".im.shm",
                                            &vstream_dirp, prev_streams, &nb_prev_streams);
    }

    static const char *const stream_props[] = { "xsize", "ysize",    "zsize", "naxis",
                                                "type",  "typename", "nelem", "cnt0",
                                                "cnt1",  "sem",      "kw.",   NULL };
    static int               prop_idx       = 0;
    static char              stream_target[128];
    static char              prop_prefix[64];
    return gen_static_props_after_dot2(text, dot1, dot2, 2, has_brace, state, stream_props,
                                       &prop_idx, stream_target, sizeof(stream_target), prop_prefix,
                                       sizeof(prop_prefix));
}

/**
 * @brief Generate `$proc.<name>.<prop>` candidates (`CLICOMPLETIONMODE_VARS_PROC`).
 */
static char *gen_vars_proc_match(const char *text, size_t len, int state)
{
    const char *dot1      = strchr(text, '.');
    const char *dot2      = (dot1 != NULL) ? strchr(dot1 + 1, '.') : NULL;
    int         has_brace = (text[0] == '$' && text[1] == '{');

    if (dot2 != NULL)
    {
        static const char *const proc_props[] = {
            "pid",      "loopstat",  "loopcnt", "loopfreq",    "exectime",     "rtprio", "ctrlval",
            "trigmode", "statusmsg", "tmux",    "description", "missedframes", NULL
        };
        static int  proc_prop_idx = 0;
        static char proc_target[128];
        static char proc_prop_prefix[64];
        return gen_static_props_after_dot2(text, dot1, dot2, 4, has_brace, state, proc_props,
                                           &proc_prop_idx, proc_target, sizeof(proc_target),
                                           proc_prop_prefix, sizeof(proc_prop_prefix));
    }

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
    if (pinfolist == NULL)
    {
        return NULL;
    }

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
            if (pinfolist->active[prev] && strcmp(pinfolist->pnamearray[prev], pname) == 0)
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
        if (match_prefix_or_fuzzy(procname, text, len))
        {
            return dupstr(procname);
        }
    }
    return NULL;
}

static char *format_env_var_candidate(const char *vname,
                                      int         has_dollar,
                                      int         has_brace,
                                      int         is_array)
{
    char buf[512];
    if (has_brace)
    {
        snprintf(buf, sizeof(buf), is_array ? "${%s[@]}" : "${%s}", vname);
    }
    else if (has_dollar)
    {
        snprintf(buf, sizeof(buf), "$%s", vname);
    }
    else
    {
        snprintf(buf, sizeof(buf), "%s", vname);
    }
    return dupstr(buf);
}

/**
 * @brief Generate shell, script, array, and environment variable candidates (`CLICOMPLETIONMODE_VARS_ENV`).
 */
static char *gen_vars_env_match(const char *text, int state)
{
    static int  var_phase;
    static int  var_idx;
    static int  has_dollar;
    static int  has_brace;
    static char vprefix[256];
    static int  vpreflen;

    if (!state)
    {
        var_phase      = 0;
        var_idx        = 0;
        has_dollar     = (text[0] == '$');
        has_brace      = (has_dollar && text[1] == '{');
        const char *vp = has_brace ? (text + 2) : (has_dollar ? (text + 1) : text);
        strncpy(vprefix, vp, sizeof(vprefix) - 1);
        vprefix[sizeof(vprefix) - 1] = '\0';
        vpreflen                     = (int) strlen(vprefix);
    }

    static const char *const special_vars[] = {
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
        if (match_prefix_or_fuzzy(sname, vprefix, (size_t) vpreflen))
        {
            return format_env_var_candidate(sname, has_dollar, has_brace, 0);
        }
    }

    while (var_phase == 1)
    {
        if (var_idx >= CLI_MAX_VARS)
        {
            var_phase = 2;
            var_idx   = 0;
            break;
        }
        int i = var_idx++;
        if (cli_vars[i].used && match_prefix_or_fuzzy(cli_vars[i].name, vprefix, (size_t) vpreflen))
        {
            return format_env_var_candidate(cli_vars[i].name, has_dollar, has_brace, 0);
        }
    }

    while (var_phase == 2)
    {
        if (var_idx >= CLI_MAX_ARRAYS)
        {
            var_phase = 3;
            var_idx   = 0;
            break;
        }
        int i = var_idx++;
        if (cli_arrays[i].used &&
            match_prefix_or_fuzzy(cli_arrays[i].name, vprefix, (size_t) vpreflen))
        {
            return format_env_var_candidate(cli_arrays[i].name, has_dollar, has_brace, 1);
        }
    }

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

        int already_in_cli = 0;
        for (int k = 0; k < CLI_MAX_VARS; k++)
        {
            if (cli_vars[k].used && strcmp(cli_vars[k].name, ename) == 0)
            {
                already_in_cli = 1;
                break;
            }
        }
        if (!already_in_cli && match_prefix_or_fuzzy(ename, vprefix, (size_t) vpreflen))
        {
            return format_env_var_candidate(ename, has_dollar, has_brace, 0);
        }
    }
    return NULL;
}

/**
 * @brief Generate tab-completion candidates for readline.
 */
char *CLI_generator(const char *text, int state)
{
    static unsigned int list_index;
    static unsigned int len;
    static int          matches_found_in_pass = 0;

    if (!state)
    {
        list_index            = 0;
        len                   = (unsigned int) strlen(text);
        generator_fuzzy_pass  = 0;
        matches_found_in_pass = 0;
    }

retry_fuzzy:;
    char *match = NULL;
    switch (data.CLImatchMode)
    {
    case CLICOMPLETIONMODE_COMMANDS:
        match = gen_commands_match(text, len, &list_index);
        break;
    case CLICOMPLETIONMODE_IMAGES:
        match = gen_images_match(text, len, state, &list_index);
        break;
    case CLICOMPLETIONMODE_CMDARGS:
        match = gen_cmdargs_match(text, len, &list_index);
        break;
    case CLICOMPLETIONMODE_FILES:
        match = gen_files_match(text, state);
        break;
    case CLICOMPLETIONMODE_FPSPARAMS:
        match = gen_fpsparams_match(text, len, state);
        break;
    case CLICOMPLETIONMODE_VARS_FPS:
        match = gen_vars_fps_match(text, len, state);
        break;
    case CLICOMPLETIONMODE_VARS_SEQ:
        match = gen_vars_seq_match(text, len, state);
        break;
    case CLICOMPLETIONMODE_VARS_STREAM:
        match = gen_vars_stream_match(text, len, state);
        break;
    case CLICOMPLETIONMODE_VARS_PROC:
        match = gen_vars_proc_match(text, len, state);
        break;
    case CLICOMPLETIONMODE_VARS_ENV:
        match = gen_vars_env_match(text, state);
        break;
    default:
        break;
    }

    if (match != NULL)
    {
        matches_found_in_pass++;
        return match;
    }

    if (generator_fuzzy_pass == 0 && matches_found_in_pass == 0 && data.autocomplete_fuzzy)
    {
        generator_fuzzy_pass = 1;
        list_index           = 0;
        state                = 0;
        goto retry_fuzzy;
    }

    return NULL;
}

/**
 * @brief Infer the completion mode for argument @argidx of command @cmdname.
 */
static int resolve_cmd_arg_completion_mode(const char *text, const char *cmdname, int argidx)
{
    const char *barecmd = (cmdname[0] == '@') ? (cmdname + 1) : cmdname;
    if (text[0] == '.' && text[1] != '/' && text[1] != '.')
    {
        return CLICOMPLETIONMODE_CMDARGS;
    }

    static const char *const file_cmds[] = {
        "loadfits", "savefits", "saveFITS",     "source",     ".",           "cat", "cd", "ls",
        "vi",       "vim",      "nano",         "head",       "tail",        "cp",  "mv", "rm",
        "less",     "more",     "include_once", "savescript", "savehistory", "run", NULL
    };
    for (int i = 0; file_cmds[i] != NULL; i++)
    {
        if (strcmp(barecmd, file_cmds[i]) == 0)
        {
            return CLICOMPLETIONMODE_FILES;
        }
    }

    static const char *const fps_cmds[] = { "fpsCTRL",  "fparam",      "fpsload",
                                            "dpsingle", "fpsconf",     "fpsrun",
                                            "fpsstop",  "waitfor_fps", NULL };
    for (int i = 0; fps_cmds[i] != NULL; i++)
    {
        if (strcmp(barecmd, fps_cmds[i]) == 0)
        {
            return CLICOMPLETIONMODE_FPSPARAMS;
        }
    }

    static const char *const env_cmds[] = {
        "export", "readonly", "unset", "local", "declare", NULL
    };
    for (int i = 0; env_cmds[i] != NULL; i++)
    {
        if (strcmp(barecmd, env_cmds[i]) == 0)
        {
            return CLICOMPLETIONMODE_VARS_ENV;
        }
    }

    if (strcmp(barecmd, "grab") == 0 || strcmp(barecmd, "streamcat") == 0 ||
        strcmp(barecmd, "streamwrite") == 0)
    {
        return CLICOMPLETIONMODE_IMAGES;
    }

    int cmdimatch = find_command_match(cmdname);
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
                        return CLICOMPLETIONMODE_FILES;
                    }
                    if (atype == CLIARG_FPSNAME)
                    {
                        return CLICOMPLETIONMODE_FPSPARAMS;
                    }
                    if (atype == CLIARG_IMG || atype == CLIARG_STREAM)
                    {
                        return CLICOMPLETIONMODE_IMAGES;
                    }
                    break;
                }
                cli_ai++;
            }
        }
    }
    return CLICOMPLETIONMODE_IMAGES;
}

/**
 * @brief Readline custom completion dispatcher invoked on TAB.
 */
char **CLI_completion(const char *text, int start, int end)
{
    cli_comp_start = start;
    cli_comp_end   = end;

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
        data.CLImatchMode = resolve_cmd_arg_completion_mode(text, cmdname, argidx);
    }

    if (data.CLImatchMode == CLICOMPLETIONMODE_FILES)
    {
        matches = rl_completion_matches((char *) text,
                                        (rl_compentry_func_t *) rl_filename_completion_function);
    }
    else
    {
        matches = rl_completion_matches((char *) text, &CLI_generator);
    }

    rl_attempted_completion_over = 1;

    if (data.CLImatchMode == CLICOMPLETIONMODE_FILES ||
        data.CLImatchMode == CLICOMPLETIONMODE_VARS_SEQ)
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
    else
    {
        rl_completion_append_character = ' ';
    }

    return matches;
}

#endif /* USE_READLINE */
