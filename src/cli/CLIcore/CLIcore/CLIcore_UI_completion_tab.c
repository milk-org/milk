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

#include "CLIcore.h"

/* Completion mode constants — must match
 * CLIcore_UI_completion.c */
#define CLICOMPLETIONMODE_COMMANDS 0
#define CLICOMPLETIONMODE_IMAGES 1
#define CLICOMPLETIONMODE_CMDARGS 2
#define CLICOMPLETIONMODE_FILES 3
#define CLICOMPLETIONMODE_FPSPARAMS 4
#define CLICOMPLETIONMODE_VARS_FPS 5
#define CLICOMPLETIONMODE_VARS_SEQ 6
#define CLICOMPLETIONMODE_VARS_STREAM 7

/* From CLIcore_UI_completion.c */
extern void *xmalloc(int size);
extern char *dupstr(const char *s);


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
    char               *name;

    if (!state)
    {
        list_index           = 0;
        len                  = strlen(text);
        generator_fuzzy_pass = 0;
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
                    return (dupstr(name));
                }
            }
            else
            {
                /* Fuzzy: substring match */
                if (strstr(name, text) != NULL)
                {
                    return (dupstr(name));
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
                    return (dupstr(name));
                }
            }
            else
            {
                if (strstr(name, text) != NULL)
                {
                    return (dupstr(name));
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
                            return (dupstr(imgname));
                        }
                    }
                    else
                    {
                        if (strstr(imgname, text) != NULL)
                        {
                            return (dupstr(imgname));
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
        while ((int) list_index < data.cmd[data.cmdindex].nbarg)
        {
            name = data.cmd[data.cmdindex].argdata[list_index].fpstag;
            list_index++;
            if (strncmp(name, text, len) == 0)
            {
                return (dupstr(name));
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

                    return dupstr(result);
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
                            return dupstr(fpsname);
                        }
                    }
                    else
                    {
                        if (strstr(fpsname, text) != NULL)
                        {
                            return dupstr(fpsname);
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
                        snprintf(fpsname, sizeof(fpsname), "@fps.%.*s.", namelen, ent->d_name + 4);

                        if (generator_fuzzy_pass == 0)
                        {
                            if (strncmp(fpsname, text, len) == 0)
                            {
                                return dupstr(fpsname);
                            }
                        }
                        else
                        {
                            if (strstr(fpsname, text) != NULL)
                            {
                                return dupstr(fpsname);
                            }
                        }
                    }
                }
            }
            closedir(vfps_dirp);
            vfps_dirp = NULL;
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
                        snprintf(seqname, sizeof(seqname), "@seq.%.*s.", namelen, ent->d_name + 4);

                        if (generator_fuzzy_pass == 0)
                        {
                            if (strncmp(seqname, text, len) == 0)
                            {
                                return dupstr(seqname);
                            }
                        }
                        else
                        {
                            if (strstr(seqname, text) != NULL)
                            {
                                return dupstr(seqname);
                            }
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
                    snprintf(sname, sizeof(sname), "${s.%.*s.", namelen, ent->d_name);

                    if (generator_fuzzy_pass == 0)
                    {
                        if (strncmp(sname, text, len) == 0)
                        {
                            return dupstr(sname);
                        }
                    }
                    else
                    {
                        if (strstr(sname, text) != NULL)
                        {
                            return dupstr(sname);
                        }
                    }
                }
            }
            closedir(vstream_dirp);
            vstream_dirp = NULL;
        }
    }

    /* Fuzzy fallback: if prefix pass found
     * nothing, restart with substring */
    if (generator_fuzzy_pass == 0 && data.autocomplete_fuzzy)
    {
        generator_fuzzy_pass = 1;
        list_index           = 0;
        goto retry_fuzzy;
    }

    return ((char *) NULL);
}


/* ---- TAB completion dispatcher ---- */

/**
 * @brief Readline custom completion dispatcher
 *
 * Invoked on TAB. Determines completion mode
 * based on cursor position and the command
 * being typed.
 */
char **CLI_completion(const char *text, int start, int __attribute__((unused)) end)
{
    char **matches;

    matches = (char **) NULL;

    if ((start == 0) || (strncmp(rl_line_buffer, "cmd?", strlen("cmd?")) == 0))
    {
        data.CLImatchMode = CLICOMPLETIONMODE_COMMANDS;
    }
    else if (strncmp(text, "@fps.", 5) == 0)
    {
        data.CLImatchMode = CLICOMPLETIONMODE_VARS_FPS;
    }
    else if (strncmp(text, "@seq.", 5) == 0)
    {
        data.CLImatchMode = CLICOMPLETIONMODE_VARS_SEQ;
    }
    else if (strncmp(text, "${s.", 4) == 0)
    {
        data.CLImatchMode = CLICOMPLETIONMODE_VARS_STREAM;
    }
    else
    {
        char  str[200];
        char *firstword;
        strncpy(str, rl_line_buffer, sizeof(str) - 1);
        str[sizeof(str) - 1] = '\0';
        firstword            = strtok(str, " ");
        if (firstword == NULL)
        {
            return NULL;
        }
        int      cmdimatch = -1;
        uint32_t cmdi      = 0;
        while ((cmdimatch == -1) && (cmdi < data.NBcmd))
        {
            if (strcmp(firstword, data.cmd[cmdi].key) == 0)
            {
                cmdimatch     = cmdi;
                data.cmdindex = cmdi;
            }
            cmdi++;
        }

        if ((cmdimatch != -1) && (text[0] == '.'))
        {
            data.CLImatchMode = CLICOMPLETIONMODE_CMDARGS;
        }
        else if (cmdimatch != -1)
        {
            int argpos = 0;
            {
                const char *p = rl_line_buffer;
                while (*p && *p != ' ')
                {
                    p++;
                }
                int in_word = 0;
                while (*p)
                {
                    if (*p != ' ')
                    {
                        if (!in_word)
                        {
                            argpos++;
                            in_word = 1;
                        }
                    }
                    else
                    {
                        in_word = 0;
                    }
                    p++;
                }
                if (rl_end > 0 && rl_line_buffer[rl_end - 1] == ' ')
                {
                    /* argpos correct */
                }
                else if (argpos > 0)
                {
                    argpos--;
                }
            }

            int cli_ai       = 0;
            int matched_file = 0;
            for (int ai = 0; ai < data.cmd[cmdimatch].nbparam; ai++)
            {
                if (data.cmd[cmdimatch].argdata[ai].fpflag & FPFLAG_PRIMARY_CLI_INPUT)
                {
                    if (cli_ai == argpos)
                    {
                        uint64_t atype = data.cmd[cmdimatch].argdata[ai].type;
                        if (atype == CLIARG_FILENAME || atype == CLIARG_FITSFILENAME)
                        {
                            matched_file = 1;
                        }
                        if (atype == CLIARG_FPSNAME)
                        {
                            data.CLImatchMode = CLICOMPLETIONMODE_FPSPARAMS;
                        }
                        break;
                    }
                    cli_ai++;
                }
            }

            if (matched_file)
            {
                data.CLImatchMode              = CLICOMPLETIONMODE_FILES;
                rl_completion_append_character = '\0';
            }
            else if (data.CLImatchMode != CLICOMPLETIONMODE_FPSPARAMS)
            {
                if (strcmp(data.cmd[cmdimatch].key, "fparam") == 0 ||
                    strcmp(data.cmd[cmdimatch].key, "fpsCTRL") == 0 ||
                    strcmp(data.cmd[cmdimatch].key, "fpsload") == 0 ||
                    strcmp(data.cmd[cmdimatch].key, "dpsingle") == 0)
                {
                    data.CLImatchMode = CLICOMPLETIONMODE_FPSPARAMS;
                }
                else
                {
                    data.CLImatchMode = CLICOMPLETIONMODE_IMAGES;
                }
            }
        }
        else
        {
            data.CLImatchMode = CLICOMPLETIONMODE_IMAGES;
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

    /* Reset append char to default space */
    if (data.CLImatchMode == CLICOMPLETIONMODE_FILES ||
        data.CLImatchMode == CLICOMPLETIONMODE_VARS_FPS ||
        data.CLImatchMode == CLICOMPLETIONMODE_VARS_SEQ ||
        data.CLImatchMode == CLICOMPLETIONMODE_VARS_STREAM)
    {
        rl_completion_append_character = '\0';
    }
    else
    {
        rl_completion_append_character = ' ';
    }

    return (matches);
}
#endif
