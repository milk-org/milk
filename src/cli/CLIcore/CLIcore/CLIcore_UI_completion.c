/**
 * @file CLIcore_UI_completion.c
 *
 * @brief Readline tab-completion and prompt construction
 *
 * Provides the tab-completion engine for the milk CLI.
 * Completion matches against registered commands, shared-
 * memory image streams, FPS names, command arguments (dot-
 * prefixed FPS tags), and filesystem paths.
 *
 * Also provides the prompt builder (including PS1 support)
 * and the readline callback that hands accepted input to
 * the command execution pipeline.
 *
 * ## Key design choices
 *
 * - **Two-pass completion**: The generator first tries
 *   prefix matching, then falls back to substring (fuzzy)
 *   matching if nothing was found and fuzzy mode is on.
 *
 * - **Argument-type-aware completion**: When the cursor is
 *   on a positional argument of a known command, the
 *   completion mode switches to match the expected argument
 *   type (image stream, filename, FPS name, etc.).
 *
 * - **Levenshtein distance**: Used by the "did you mean?"
 *   suggestions when a command is not found.
 */

#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <termios.h>

#ifdef USE_READLINE
#include <readline/history.h>
#include <readline/readline.h>
#endif

#include "CLIcore.h"
#include "CLIcore/cli_calc_parser.h"
#include "CLIcore_script.h"
#include "CLIcore_UI.h"

#include <fnmatch.h>
#include <glob.h>
#include <sys/wait.h>

#include "COREMOD_memory/COREMOD_memory.h"
#include "timeutils.h"

#define CLICOMPLETIONMODE_COMMANDS  0
#define CLICOMPLETIONMODE_IMAGES   1
#define CLICOMPLETIONMODE_CMDARGS  2
#define CLICOMPLETIONMODE_FILES    3
#define CLICOMPLETIONMODE_FPSPARAMS 4

#define COLORRED       "\001\033[31m\002"
#define COLORHBOLDCYAN "\001\e[0;96m\002"
#define COLORDIMYELLOW "\033[2;33m"
#define COLORRST       "\033[0m"
#define RL_COLORRESET  "\001\033[0m\002"


/* ---- String utilities ---- */

void *xmalloc(int size)
{
    void *buf;

    buf = malloc(size);
    if(!buf)
    {
        fprintf(stderr,
                COLORRED
                "Error: Out of memory. Exiting.'n"
                COLORRESET);
        exit(1);
    }

    return buf;
}

/**
 * @brief Duplicate a string using xmalloc.
 *
 * Allocates memory for a copy of @s and returns
 * the copy. The caller must free() the result.
 *
 * @param s  String to duplicate
 * @return Newly allocated copy of @s
 */
char *dupstr(char *s)
{
    char *r;

    r = (char *) xmalloc((strlen(s) + 1));
    strcpy(r, s);
    return (r);
}


/* ---- Readline callback and prompt ---- */

#ifdef USE_READLINE

/**
 * Number of ghost chars rendered on current line.
 * Set by print_ghost(), read by cli_accept_line().
 */
int ghost_chars_on_line = 0;

/**
 * @brief Custom accept-line handler for readline
 *
 * Bound to Enter key. Overwrites ghost suggestion
 * text with spaces before accepting the line, so
 * the terminal scrollback entry is clean.
 */
int cli_accept_line(
    int count,
    int key
)
{
    if(ghost_chars_on_line > 0)
    {
        int n = ghost_chars_on_line;
        for(int i = 0; i < n; i++)
        {
            putchar(' ');
        }
        for(int i = 0; i < n; i++)
        {
            putchar('\b');
        }
        fflush(stdout);
        ghost_chars_on_line = 0;
    }

    return rl_newline(count, key);
}

/**
 * @brief Readline callback handler — processes a
 *        completed input line.
 *
 * Invoked by rl_callback_read_char() when the user
 * presses Enter. Copies the input into
 * data.CLIcmdline, handles backslash line
 * continuation (reading extra lines until no
 * trailing backslash), then dispatches the
 * assembled command via CLI_execute_line().
 *
 * If linein is NULL (Ctrl-D / EOF), sets
 * data.CLIloopON=0 to exit the main loop.
 *
 * @param linein  Line text from readline
 *                (caller-allocated, freed here)
 */
void rl_cb_linehandler(char *linein)
{
    if(NULL == linein)
    {
        data.CLIloopON = 0;
        return;
    }

    data.CLIexecuteCMDready = 1;

    // copy input into data.CLIcmdline
    strcpy(data.CLIcmdline, linein);

    /* Handle backslash line continuation:
     * temporarily switch to blocking readline
     * to read additional lines */
    {
        size_t len = strlen(data.CLIcmdline);
        while(len > 0
                && data.CLIcmdline[len - 1]
                == '\\')
        {
            data.CLIcmdline[len - 1] = ' ';
            /* Remove callback handler to avoid
             * interference, use direct readline
             * for continuation */
            rl_callback_handler_remove();
            char *cont = readline("> ");
            /* Re-install with dummy prompt;
             * the main loop will re-install
             * with the proper prompt after this
             * handler returns */
            rl_callback_handler_install(
                "",
                (rl_vcpfunc_t *)
                &rl_cb_linehandler);
            if(cont == NULL)
            {
                break;
            }
            int avail =
                STRINGMAXLEN_CLICMDLINE
                - (int) strlen(data.CLIcmdline)
                - 1;
            if(avail > 0)
            {
                strncat(data.CLIcmdline, cont,
                        (size_t) avail);
            }
            free(cont);
            len = strlen(data.CLIcmdline);
        }
    }

    CLI_execute_line();

    free(linein);
}
#endif

/**
 * @brief Build the prompt string for the CLI
 *
 * Checks for a PS1 variable in CLI vars or the
 * environment. Falls back to the default colored
 * prompt with the process name.
 */
errno_t runCLI_prompt(char *promptstring, char *prompt)
{
    // Try to get PS1 from CLI vars or environment
    const char *ps1_val = cli_var_lookup("PS1");
    if(ps1_val == NULL)
    {
        ps1_val = getenv("PS1");
    }

    if(ps1_val != NULL && strlen(ps1_val) > 0)
    {
        char expanded_ps1[FPS_DIR_STRLENMAX];
        strncpy(expanded_ps1, ps1_val,
                FPS_DIR_STRLENMAX - 1);
        expanded_ps1[FPS_DIR_STRLENMAX - 1] = '\0';
        cli_expand_env(expanded_ps1,
                       FPS_DIR_STRLENMAX);
        strncpy(prompt, expanded_ps1,
                FPS_DIR_STRLENMAX - 1);
        prompt[FPS_DIR_STRLENMAX - 1] = '\0';
        return RETURN_SUCCESS;
    }

    if(strlen(promptstring) > 0)
    {
        if(data.processnameflag == 0)
        {
            snprintf(prompt, FPS_DIR_STRLENMAX,
                     COLORHBOLDCYAN
                     "%s > " RL_COLORRESET,
                     promptstring);
        }
        else
        {
            snprintf(prompt,
                     FPS_DIR_STRLENMAX,
                     COLORHBOLDCYAN
                     "%s-%s > " RL_COLORRESET,
                     promptstring,
                     data.processname);
        }
    }
    else
    {
        snprintf(prompt, FPS_DIR_STRLENMAX,
                 COLORHBOLDCYAN
                 "%s > " RL_COLORRESET,
                 data.processname);
    }

    return RETURN_SUCCESS;
}


/* ---- Levenshtein distance (fuzzy matching) ---- */

#ifdef USE_READLINE

/**
 * @brief Compute Levenshtein edit distance
 *
 * Used to suggest similar commands when a typed
 * command is not found ("did you mean?").
 */
int levenshtein_distance(
    const char *s1,
    const char *s2
)
{
    unsigned int len1 = strlen(s1);
    unsigned int len2 = strlen(s2);
    unsigned int *d = (unsigned int *)
        xmalloc((len1 + 1) * (len2 + 1)
                * sizeof(unsigned int));

    for(unsigned int i = 0; i <= len1; i++)
    {
        d[i * (len2 + 1)] = i;
    }
    for(unsigned int j = 0; j <= len2; j++)
    {
        d[j] = j;
    }

    for(unsigned int i = 1; i <= len1; i++)
    {
        for(unsigned int j = 1; j <= len2; j++)
        {
            unsigned int cost =
                (s1[i - 1] == s2[j - 1])
                ? 0 : 1;
            unsigned int min1 =
                d[(i - 1) * (len2 + 1) + j] + 1;
            unsigned int min2 =
                d[i * (len2 + 1) + j - 1] + 1;
            unsigned int min3 =
                d[(i - 1) * (len2 + 1)
                  + j - 1] + cost;
            unsigned int m =
                (min1 < min2) ? min1 : min2;
            d[i * (len2 + 1) + j] =
                (m < min3) ? m : min3;
        }
    }
    int dist = d[len1 * (len2 + 1) + len2];
    free(d);
    return dist;
}


/* ---- Tab-completion generator ---- */

/**
 * @brief State for fuzzy fallback pass in generator
 *
 * After a normal prefix-match pass, if nothing
 * matched and fuzzy is enabled, we restart with
 * substring match.
 */
int generator_fuzzy_pass = 0;

/**
 * @brief Generate tab-completion candidates
 *
 * Called repeatedly by readline to produce matching
 * candidates. The match mode (commands, images,
 * args, files, FPS) determines the search space.
 *
 * On first call (state == 0), initializes the search.
 * Returns one match at a time, or NULL when exhausted.
 */
char *CLI_generator(const char *text, int state)
{
    static unsigned int list_index;
    static unsigned int len;
    char               *name;

    if(!state)
    {
        list_index  = 0;
        len         = strlen(text);
        generator_fuzzy_pass = 0;
    }

retry_fuzzy:

    if(data.CLImatchMode
       == CLICOMPLETIONMODE_COMMANDS)
    {
        while(list_index < data.NBcmd)
        {
            name = data.cmd[list_index].key;
            list_index++;
            if(generator_fuzzy_pass == 0)
            {
                if(strncmp(name, text, len)
                   == 0)
                {
                    return (dupstr(name));
                }
            }
            else
            {
                /* Fuzzy: substring match */
                if(strstr(name, text) != NULL)
                {
                    return (dupstr(name));
                }
            }
        }
    }

    if(data.CLImatchMode
       == CLICOMPLETIONMODE_IMAGES)
    {
        static DIR *img_dirp = NULL;

        if(!state)
        {
            if(img_dirp != NULL)
            {
                closedir(img_dirp);
                img_dirp = NULL;
            }
            img_dirp = opendir(dcshmdir);
        }

        if(img_dirp != NULL)
        {
            struct dirent *ent;
            while((ent = readdir(img_dirp))
                  != NULL)
            {
                char *ext = strstr(
                    ent->d_name, ".im.shm");
                if(ext != NULL
                   && strcmp(ext, ".im.shm")
                   == 0)
                {
                    char imgname[256];
                    int namelen =
                        ext - ent->d_name;
                    if(namelen > 255)
                    {
                        namelen = 255;
                    }
                    strncpy(imgname,
                            ent->d_name,
                            namelen);
                    imgname[namelen] = '\0';

                    if(generator_fuzzy_pass == 0)
                    {
                        if(strncmp(imgname, text,
                                   len) == 0)
                        {
                            return (dupstr(
                                imgname));
                        }
                    }
                    else
                    {
                        if(strstr(imgname, text)
                           != NULL)
                        {
                            return (dupstr(
                                imgname));
                        }
                    }
                }
            }
            closedir(img_dirp);
            img_dirp = NULL;
        }
    }

    if(data.CLImatchMode
       == CLICOMPLETIONMODE_CMDARGS)
    {
        while((int) list_index
              < data.cmd[data.cmdindex].nbarg)
        {
            name = data.cmd[data.cmdindex]
                       .argdata[list_index]
                       .fpstag;
            list_index++;
            if(strncmp(name, text, len) == 0)
            {
                return (dupstr(name));
            }
        }
    }

    if(data.CLImatchMode
       == CLICOMPLETIONMODE_FILES)
    {
        /* Filesystem path completion.
         * Split text into directory + prefix,
         * enumerate with opendir/readdir. */
        static DIR *dirp = NULL;
        static char dirpart[512];
        static char prefix[256];
        static unsigned int preflen;

        if(!state)
        {
            if(dirp != NULL)
            {
                closedir(dirp);
                dirp = NULL;
            }

            /* Split text into dir + prefix */
            const char *slash =
                strrchr(text, '/');
            if(slash != NULL)
            {
                int dlen =
                    (int)(slash - text) + 1;
                if(dlen
                   > (int) sizeof(dirpart) - 1)
                {
                    dlen =
                        (int) sizeof(dirpart)
                        - 1;
                }
                memcpy(dirpart, text, dlen);
                dirpart[dlen] = '\0';
                strncpy(prefix, slash + 1,
                        sizeof(prefix) - 1);
                prefix[sizeof(prefix) - 1] =
                    '\0';
            }
            else
            {
                strcpy(dirpart, ".");
                strncpy(prefix, text,
                        sizeof(prefix) - 1);
                prefix[sizeof(prefix) - 1] =
                    '\0';
            }
            preflen = strlen(prefix);

            dirp = opendir(dirpart);
        }

        if(dirp != NULL)
        {
            struct dirent *ent;
            while((ent = readdir(dirp)) != NULL)
            {
                /* Skip . and .. */
                if(strcmp(ent->d_name, ".") == 0
                    || strcmp(ent->d_name,
                             "..") == 0)
                {
                    continue;
                }

                if(strncmp(ent->d_name, prefix,
                           preflen) == 0)
                {
                    char fullpath[1024];
                    snprintf(fullpath,
                             sizeof(fullpath),
                             "%s/%s",
                             dirpart,
                             ent->d_name);

                    char result[1024];
                    if(strcmp(dirpart, ".")
                       == 0)
                    {
                        snprintf(
                            result,
                            sizeof(result),
                            "%s",
                            ent->d_name);
                    }
                    else
                    {
                        snprintf(
                            result,
                            sizeof(result),
                            "%s%s",
                            dirpart,
                            ent->d_name);
                    }

                    /* Append / for dirs */
                    struct stat st;
                    if(stat(fullpath, &st) == 0
                        && S_ISDIR(st.st_mode))
                    {
                        strncat(result, "/",
                                sizeof(result)
                                - strlen(result)
                                - 1);
                    }

                    return dupstr(result);
                }
            }
            closedir(dirp);
            dirp = NULL;
        }
    }

    if(data.CLImatchMode
       == CLICOMPLETIONMODE_FPSPARAMS)
    {
        /* FPS name completion.
         * Scan dcshmdir for fps.*.datadir,
         * strip "fps." prefix and ".datadir"
         * suffix. */
        static DIR *fps_dirp = NULL;

        if(!state)
        {
            if(fps_dirp != NULL)
            {
                closedir(fps_dirp);
                fps_dirp = NULL;
            }
            fps_dirp = opendir(dcshmdir);
        }

        if(fps_dirp != NULL)
        {
            struct dirent *ent;
            while((ent = readdir(fps_dirp))
                  != NULL)
            {
                if(strncmp(ent->d_name,
                           "fps.", 4) != 0)
                {
                    continue;
                }
                char *ext = strstr(
                    ent->d_name, ".datadir");
                if(ext != NULL
                   && strcmp(ext, ".datadir")
                   == 0)
                {
                    char fpsname[256];
                    int namelen =
                        ext
                        - (ent->d_name + 4);
                    if(namelen > 255)
                    {
                        namelen = 255;
                    }
                    strncpy(fpsname,
                            ent->d_name + 4,
                            namelen);
                    fpsname[namelen] = '\0';

                    if(generator_fuzzy_pass
                       == 0)
                    {
                        if(strncmp(
                               fpsname, text,
                               len) == 0)
                        {
                            return dupstr(
                                fpsname);
                        }
                    }
                    else
                    {
                        if(strstr(fpsname,
                                  text)
                           != NULL)
                        {
                            return dupstr(
                                fpsname);
                        }
                    }
                }
            }
            closedir(fps_dirp);
            fps_dirp = NULL;
        }
    }

    /* Fuzzy fallback: if prefix pass found
     * nothing, restart with substring matching */
    if(generator_fuzzy_pass == 0
       && data.autocomplete_fuzzy)
    {
        generator_fuzzy_pass = 1;
        list_index  = 0;
        goto retry_fuzzy;
    }

    return ((char *) NULL);
}


/* ---- TAB completion dispatcher ---- */

/**
 * @brief Readline custom completion dispatcher
 *
 * Invoked when pressing TAB. Determines the
 * completion mode based on cursor position and
 * the command being typed:
 *
 * - First word → match commands
 * - Known command + dot-prefix → FPS param tags
 * - Known command + FILENAME arg → filesystem
 * - Known command + FPSNAME arg → FPS names
 * - Otherwise → image stream names
 */
char **
CLI_completion(
    const char *text,
    int start,
    int __attribute__((unused)) end
)
{
    char **matches;

    matches = (char **) NULL;

    if((start == 0)
       || (strncmp(rl_line_buffer, "cmd?",
                   strlen("cmd?")) == 0))
    {
        data.CLImatchMode =
            CLICOMPLETIONMODE_COMMANDS;
    }
    else
    {
        char  str[200];
        char *firstword;
        firstword = strcpy(str, rl_line_buffer);
        strtok(str, " ");
        int      cmdimatch = -1;
        uint32_t cmdi      = 0;
        while((cmdimatch == -1)
              && (cmdi < data.NBcmd))
        {
            if(strcmp(firstword,
                     data.cmd[cmdi].key) == 0)
            {
                cmdimatch = cmdi;
                data.cmdindex = cmdi;
            }
            cmdi++;
        }

        if((cmdimatch != -1)
           && (text[0] == '.'))
        {
            data.CLImatchMode =
                CLICOMPLETIONMODE_CMDARGS;
        }
        else if(cmdimatch != -1)
        {
            /* Count which CLI argument position
             * the cursor is at */
            int argpos = 0;
            {
                const char *p = rl_line_buffer;
                while(*p && *p != ' ')
                {
                    p++;
                }
                int in_word = 0;
                while(*p)
                {
                    if(*p != ' ')
                    {
                        if(!in_word)
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
                if(rl_end > 0
                    && rl_line_buffer[
                        rl_end - 1] == ' ')
                {
                    /* argpos already correct */
                }
                else if(argpos > 0)
                {
                    argpos--;
                }
            }

            /* Check argument type at this
             * position (CLI-visible args) */
            int cli_ai = 0;
            int matched_file = 0;
            for(int ai = 0;
                ai < data.cmd[cmdimatch]
                         .nbparam;
                ai++)
            {
                if(data.cmd[cmdimatch]
                    .argdata[ai].fpflag
                    & FPFLAG_PRIMARY_CLI_INPUT)
                {
                    if(cli_ai == argpos)
                    {
                        uint64_t atype =
                            data.cmd[cmdimatch]
                            .argdata[ai].type;
                        if(atype
                               == CLIARG_FILENAME
                            || atype
                               == CLIARG_FITSFILENAME)
                        {
                            matched_file = 1;
                        }
                        if(atype
                           == CLIARG_FPSNAME)
                        {
                            data.CLImatchMode =
                                CLICOMPLETIONMODE_FPSPARAMS;
                        }
                        break;
                    }
                    cli_ai++;
                }
            }

            if(matched_file)
            {
                data.CLImatchMode =
                    CLICOMPLETIONMODE_FILES;
                rl_completion_append_character
                    = '\0';
            }
            else if(data.CLImatchMode
                    != CLICOMPLETIONMODE_FPSPARAMS)
            {
                if(strcmp(
                       data.cmd[cmdimatch].key,
                       "fparam") == 0
                   || strcmp(
                          data.cmd[cmdimatch]
                              .key,
                          "fpsCTRL") == 0
                   || strcmp(
                          data.cmd[cmdimatch]
                              .key,
                          "fpsload") == 0
                   || strcmp(
                          data.cmd[cmdimatch]
                              .key,
                          "dpsingle") == 0)
                {
                    data.CLImatchMode =
                        CLICOMPLETIONMODE_FPSPARAMS;
                }
                else
                {
                    data.CLImatchMode =
                        CLICOMPLETIONMODE_IMAGES;
                }
            }
        }
        else
        {
            data.CLImatchMode =
                CLICOMPLETIONMODE_IMAGES;
        }
    }

    matches = rl_completion_matches(
        (char *) text, &CLI_generator);

    /* Reset append char to default space */
    if(data.CLImatchMode
       != CLICOMPLETIONMODE_FILES)
    {
        rl_completion_append_character = ' ';
    }

    return (matches);
}
#endif
