/**
 * @file CLIcore_UI.c
 *
 * @brief User input (UI) functions
 *
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

#include <glob.h>
#include <sys/wait.h>

#include "COREMOD_memory/COREMOD_memory.h"
#include "timeutils.h"

#define CLICOMPLETIONMODE_COMMANDS 0
#define CLICOMPLETIONMODE_IMAGES   1
#define CLICOMPLETIONMODE_CMDARGS  2
#define CLICOMPLETIONMODE_FILES    3
#define CLICOMPLETIONMODE_FPSPARAMS 4

// COLORRESET removed to prevent redefinition with fps.h
#define COLORRED       "\001\033[31m\002" /* Red */
#define COLORHBOLDCYAN "\001\e[0;96m\002" /* High Intensity Bold Cyan */
#define RL_COLORRESET  "\001\033[0m\002"

extern void yy_scan_string(const char *);
extern int  yylex_destroy(void);

#ifdef USE_READLINE
/**
 * @brief Readline callback
 *
 **/
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


static const char *CLI_history_file(void)
{
    static char path[1024] = {0};
    if(path[0] == '\0')
    {
        const char *home = getenv("HOME");
        if(home)
        {
            snprintf(path, sizeof(path), "%s/.milk_history", home);
        }
        else
        {
            snprintf(path, sizeof(path), ".milk_history");
        }
    }
    return path;
}

errno_t runCLI_prompt(char *promptstring, char *prompt)
{
    //int color_cyan = 36;

    if(strlen(promptstring) > 0)
    {
        if(data.processnameflag == 0)
        {
            snprintf(prompt, FPS_DIR_STRLENMAX, COLORHBOLDCYAN "%s > " RL_COLORRESET,
                     promptstring);
        }
        else
        {
            snprintf(prompt,
                     FPS_DIR_STRLENMAX,
                     COLORHBOLDCYAN "%s-%s > " RL_COLORRESET,
                     promptstring,
                     data.processname);
        }
    }
    else
    {
        snprintf(prompt, FPS_DIR_STRLENMAX, COLORHBOLDCYAN "%s > " RL_COLORRESET,
                 data.processname);
    }

    return RETURN_SUCCESS;
}


static void *xmalloc(int size)
{
    void *buf;

    buf = malloc(size);
    if(!buf)
    {
        fprintf(stderr, COLORRED "Error: Out of memory. Exiting.'n" COLORRESET);
        exit(1);
    }

    return buf;
}

static char *dupstr(char *s)
{
    char *r;

    r = (char *) xmalloc((strlen(s) + 1));
    strcpy(r, s);
    return (r);
}

#ifdef USE_READLINE
static int levenshtein_distance(const char *s1, const char *s2)
{
    unsigned int len1 = strlen(s1);
    unsigned int len2 = strlen(s2);
    unsigned int *d = (unsigned int *)xmalloc((len1 + 1) * (len2 + 1) * sizeof(unsigned int));

    for(unsigned int i = 0; i <= len1; i++) d[i * (len2 + 1)] = i;
    for(unsigned int j = 0; j <= len2; j++) d[j] = j;

    for(unsigned int i = 1; i <= len1; i++) {
        for(unsigned int j = 1; j <= len2; j++) {
            unsigned int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
            unsigned int min1 = d[(i - 1) * (len2 + 1) + j] + 1;
            unsigned int min2 = d[i * (len2 + 1) + j - 1] + 1;
            unsigned int min3 = d[(i - 1) * (len2 + 1) + j - 1] + cost;
            unsigned int m = (min1 < min2) ? min1 : min2;
            d[i * (len2 + 1) + j] = (m < min3) ? m : min3;
        }
    }
    int dist = d[len1 * (len2 + 1) + len2];
    free(d);
    return dist;
}

/**
 * @brief State for fuzzy fallback pass in generator
 *
 * After a normal prefix-match pass, if nothing matched
 * and fuzzy is enabled, we restart with substring match.
 */
static int generator_fuzzy_pass = 0;

static char *CLI_generator(const char *text, int state)
{
    static unsigned int list_index;
    static unsigned int list_index1;
    static unsigned int len;
    char               *name;

    if(!state)
    {
        list_index  = 0;
        list_index1 = 0;
        len         = strlen(text);
        generator_fuzzy_pass = 0;
    }

retry_fuzzy:

    if(data.CLImatchMode == CLICOMPLETIONMODE_COMMANDS)
    {
        while(list_index < data.NBcmd)
        {
            name = data.cmd[list_index].key;
            list_index++;
            if(generator_fuzzy_pass == 0)
            {
                if(strncmp(name, text, len) == 0)
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

    if(data.CLImatchMode == CLICOMPLETIONMODE_IMAGES)
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
            while((ent = readdir(img_dirp)) != NULL)
            {
                char *ext = strstr(ent->d_name, ".im.shm");
                if(ext != NULL && strcmp(ext, ".im.shm") == 0)
                {
                    char imgname[256];
                    int namelen = ext - ent->d_name;
                    if(namelen > 255)
                    {
                        namelen = 255;
                    }
                    strncpy(imgname, ent->d_name, namelen);
                    imgname[namelen] = '\0';

                    if(generator_fuzzy_pass == 0)
                    {
                        if(strncmp(imgname, text, len) == 0)
                        {
                            return (dupstr(imgname));
                        }
                    }
                    else
                    {
                        if(strstr(imgname, text) != NULL)
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

    if(data.CLImatchMode == CLICOMPLETIONMODE_CMDARGS)
    {
        while((int) list_index <
                data.cmd[data.cmdindex].nbarg)
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

    if(data.CLImatchMode == CLICOMPLETIONMODE_FILES)
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

            /* Split text into dir + filename prefix */
            const char *slash =
                strrchr(text, '/');
            if(slash != NULL)
            {
                int dlen = (int)(slash - text) + 1;
                if(dlen > (int) sizeof(dirpart) - 1)
                {
                    dlen = (int) sizeof(dirpart) - 1;
                }
                memcpy(dirpart, text, dlen);
                dirpart[dlen] = '\0';
                strncpy(prefix, slash + 1,
                        sizeof(prefix) - 1);
                prefix[sizeof(prefix) - 1] = '\0';
            }
            else
            {
                strcpy(dirpart, ".");
                strncpy(prefix, text,
                        sizeof(prefix) - 1);
                prefix[sizeof(prefix) - 1] = '\0';
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
                    /* Build full path for
                     * stat check */
                    char fullpath[1024];
                    snprintf(fullpath,
                             sizeof(fullpath),
                             "%s/%s",
                             dirpart,
                             ent->d_name);

                    /* Build result: dirpart
                     * prefix + entry name */
                    char result[1024];
                    if(strcmp(dirpart, ".") == 0)
                    {
                        snprintf(result,
                                 sizeof(result),
                                 "%s",
                                 ent->d_name);
                    }
                    else
                    {
                        snprintf(result,
                                 sizeof(result),
                                 "%s%s",
                                 dirpart,
                                 ent->d_name);
                    }

                    /* Append / for directories */
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

    if(data.CLImatchMode == CLICOMPLETIONMODE_FPSPARAMS)
    {
        /*
         * FPS name completion.
         * Scan dcshmdir for fps.*.datadir entries,
         * strip "fps." prefix and ".datadir" suffix.
         */
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
            while((ent = readdir(fps_dirp)) != NULL)
            {
                /* Only fps.*.datadir entries */
                if(strncmp(ent->d_name, "fps.", 4) != 0)
                {
                    continue;
                }
                char *ext = strstr(ent->d_name, ".datadir");
                if(ext != NULL && strcmp(ext, ".datadir") == 0)
                {
                    char fpsname[256];
                    int namelen = ext - (ent->d_name + 4);
                    if(namelen > 255)
                    {
                        namelen = 255;
                    }
                    strncpy(fpsname, ent->d_name + 4, namelen);
                    fpsname[namelen] = '\0';

                    if(generator_fuzzy_pass == 0)
                    {
                        if(strncmp(fpsname, text, len) == 0)
                        {
                            return dupstr(fpsname);
                        }
                    }
                    else
                    {
                        if(strstr(fpsname, text) != NULL)
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

    /* Fuzzy fallback: if prefix pass found nothing,
     * restart with substring matching */
    if(generator_fuzzy_pass == 0 &&
            data.autocomplete_fuzzy)
    {
        generator_fuzzy_pass = 1;
        list_index  = 0;
        list_index1 = 0;
        goto retry_fuzzy;
    }

    return ((char *) NULL);
}

/** @brief readline custom completion
 *
 * Invoked when pressing TAB
 */

char **
CLI_completion(const char *text, int start, int __attribute__((unused)) end)
{
    char **matches;

    //printf("[%d | %s | %s]", start, rl_line_buffer, text);
    //rl_message("\n[%d %s]\n", start, rl_line_buffer);
    //rl_redisplay();
    //rl_forced_update_display();

    matches = (char **) NULL;

    if((start == 0) || (strncmp(rl_line_buffer, "cmd?", strlen("cmd?")) == 0))
    {
        // if first word, or second argument to cmd?, match string with commands
        data.CLImatchMode = CLICOMPLETIONMODE_COMMANDS;
    }
    else
    {
        // test if first word is a command
        char  str[200];
        char *firstword;
        firstword = strcpy(str, rl_line_buffer);
        strtok(str, " ");
        int      cmdimatch = -1;
        uint32_t cmdi      = 0;
        while((cmdimatch == -1) && (cmdi < data.NBcmd))
        {
            if(strcmp(firstword, data.cmd[cmdi].key) == 0)
            {
                cmdimatch = cmdi;
                //printf("COMMAND MATCH %s\n", data.cmd[cmdi].key);
                data.cmdindex = cmdi;
            }
            cmdi++;
        }

        if((cmdimatch != -1) && (text[0] == '.'))
        {
            data.CLImatchMode = CLICOMPLETIONMODE_CMDARGS;
        }
        else if(cmdimatch != -1)
        {
            /* Count which CLI argument position
             * the cursor is at */
            int argpos = 0;
            {
                const char *p = rl_line_buffer;
                /* Skip command word */
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
                /* If trailing space, next arg */
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
                ai < data.cmd[cmdimatch].nbparam;
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
                        if(atype == CLIARG_FILENAME
                            || atype
                            == CLIARG_FITSFILENAME)
                        {
                            matched_file = 1;
                        }
                        if(atype == CLIARG_FPSNAME)
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
                /* Don't append space after
                 * directory names */
                rl_completion_append_character
                    = '\0';
            }
            else if(data.CLImatchMode
                    != CLICOMPLETIONMODE_FPSPARAMS)
            {
                if(strcmp(data.cmd[cmdimatch].key, "fparam") == 0 ||
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
            // match string with images
            data.CLImatchMode = CLICOMPLETIONMODE_IMAGES;
        }
    }

    matches = rl_completion_matches((char *) text, &CLI_generator);

    /* Reset append char to default space */
    if(data.CLImatchMode != CLICOMPLETIONMODE_FILES)
    {
        rl_completion_append_character = ' ';
    }

    //    else
    //  rl_bind_key('\t',rl_abort);

    return (matches);
}
#endif

errno_t write_tracedebugfile()
{
    pid_t thisPID = getpid();

    char fname[STRINGMAXLEN_FILENAME];
    WRITE_FILENAME(fname, "milk-codetracepoint.%05d.log", thisPID);

    printf("Writing output trace to file %s\n", fname);
    printf("dctestptinit = %d\n", dctestptinit);

    FILE *fp = fopen(fname, "w");
    if(fp != NULL)
    {
        for(uint64_t i = 0; i < CODETESTPOINTARRAY_NBCNT; i++)
        {
            long j = (i + dctestptcnt) % CODETESTPOINTARRAY_NBCNT;

            uint64_t index =
                dctestptarr[j].loopcnt * CODETESTPOINTARRAY_NBCNT + j;

            if(dctestptarr[j].line != 0)
            {
                char timestring[TIMESTRINGLEN];
                mkUTtimestring_nanosec(timestring, dctestptarr[j].time);

                // extract last word
                char str[STRINGMAXLEN_FULLFILENAME];
                strcpy(str, dctestptarr[j].file);
                char *lastword = strrchr(str, '/') + 1;

                fprintf(fp,
                        "T %6ld %s %-20s %6d %-20s  %s\n",
                        index,
                        timestring,
                        lastword,
                        dctestptarr[j].line,
                        dctestptarr[j].func,
                        dctestptarr[j].msg);
                fprintf(fp,
                        "       FTRACE %d ",
                        dctestptarr[j].funclevel);
                for(int level = 0; level < dctestptarr[j].funclevel;
                        level++)
                {
                    fprintf(fp,
                            " (%d) >> %ld:%s",
                            dctestptarr[j].linestack[level],
                            dctestptarr[j].fcntstack[level],
                            dctestptarr[j].funcstack[level]);
                }
                fprintf(fp, "\n\n");

                //printf("%s\n", p + 1);
            }
        }
        fclose(fp);
    }

    return RETURN_SUCCESS;
}


/*
 * ============================================================
 *  Command Alias Subsystem
 * ============================================================
 *
 * Aliases are stored in data.alias[] and persisted
 * to ~/.milk_aliases (one "name=command" per line).
 */

/** @brief Path to alias persistence file */
static const char *CLI_alias_file(void)
{
    static char path[1024] = {0};
    if(path[0] == '\0')
    {
        const char *home = getenv("HOME");
        if(home)
        {
            snprintf(path, sizeof(path),
                     "%s/.milk_aliases", home);
        }
        else
        {
            snprintf(path, sizeof(path),
                     ".milk_aliases");
        }
    }
    return path;
}

/**
 * @brief Load aliases from ~/.milk_aliases
 */
void cli_alias_load(void)
{
    data.NBalias = 0;
    FILE *fp = fopen(CLI_alias_file(), "r");
    if(fp == NULL)
    {
        return;
    }

    char line[CLI_ALIAS_NAMELEN + CLI_ALIAS_CMDLEN + 4];
    while(fgets(line, (int) sizeof(line), fp)
            != NULL)
    {
        if(data.NBalias >= CLI_MAX_ALIASES)
        {
            break;
        }
        /* Strip trailing newline */
        line[strcspn(line, "\n")] = '\0';
        /* Skip empty / comment lines */
        if(line[0] == '\0' || line[0] == '#')
        {
            continue;
        }
        /* Split on first '=' */
        char *eq = strchr(line, '=');
        if(eq == NULL)
        {
            continue;
        }
        *eq = '\0';
        strncpy(data.alias[data.NBalias].name,
                line,
                CLI_ALIAS_NAMELEN - 1);
        data.alias[data.NBalias].name[
            CLI_ALIAS_NAMELEN - 1] = '\0';
        strncpy(data.alias[data.NBalias].cmd,
                eq + 1,
                CLI_ALIAS_CMDLEN - 1);
        data.alias[data.NBalias].cmd[
            CLI_ALIAS_CMDLEN - 1] = '\0';
        data.NBalias++;
    }
    fclose(fp);
}

/**
 * @brief Save aliases to ~/.milk_aliases
 */
static void cli_alias_save(void)
{
    FILE *fp = fopen(CLI_alias_file(), "w");
    if(fp == NULL)
    {
        printf("ERROR: cannot write %s\n",
               CLI_alias_file());
        return;
    }
    for(int i = 0; i < data.NBalias; i++)
    {
        fprintf(fp, "%s=%s\n",
                data.alias[i].name,
                data.alias[i].cmd);
    }
    fclose(fp);
}

/**
 * @brief CLI handler: alias <name> <command...>
 *
 * Creates or updates a command alias.
 */
errno_t cli_alias_add(void)
{
    if(data.cmdNBarg < 3)
    {
        printf("Usage: alias <name> <command...>\n");
        return RETURN_FAILURE;
    }

    const char *name =
        data.cmdargtoken[1].val.string;

    /* Build command from args 2..N */
    char cmd[CLI_ALIAS_CMDLEN];
    cmd[0] = '\0';
    for(long a = 2; a < data.cmdNBarg; a++)
    {
        if(a > 2)
        {
            strncat(cmd, " ",
                    CLI_ALIAS_CMDLEN
                    - strlen(cmd) - 1);
        }
        strncat(cmd,
                data.cmdargtoken[a].val.string,
                CLI_ALIAS_CMDLEN
                - strlen(cmd) - 1);
    }

    /* Check if alias already exists — update */
    for(int i = 0; i < data.NBalias; i++)
    {
        if(strcmp(data.alias[i].name, name) == 0)
        {
            strncpy(data.alias[i].cmd, cmd,
                    CLI_ALIAS_CMDLEN - 1);
            data.alias[i].cmd[
                CLI_ALIAS_CMDLEN - 1] = '\0';
            cli_alias_save();
            printf("Alias updated: %s = %s\n",
                   name, cmd);
            return RETURN_SUCCESS;
        }
    }

    /* Add new alias */
    if(data.NBalias >= CLI_MAX_ALIASES)
    {
        printf("ERROR: alias table full (%d)\n",
               CLI_MAX_ALIASES);
        return RETURN_FAILURE;
    }

    strncpy(data.alias[data.NBalias].name,
            name, CLI_ALIAS_NAMELEN - 1);
    data.alias[data.NBalias].name[
        CLI_ALIAS_NAMELEN - 1] = '\0';
    strncpy(data.alias[data.NBalias].cmd,
            cmd, CLI_ALIAS_CMDLEN - 1);
    data.alias[data.NBalias].cmd[
        CLI_ALIAS_CMDLEN - 1] = '\0';
    data.NBalias++;

    cli_alias_save();
    printf("Alias created: %s = %s\n",
           name, cmd);

    return RETURN_SUCCESS;
}

/**
 * @brief CLI handler: unalias <name>
 */
errno_t cli_alias_remove(void)
{
    if(data.cmdNBarg < 2)
    {
        printf("Usage: unalias <name>\n");
        return RETURN_FAILURE;
    }

    const char *name =
        data.cmdargtoken[1].val.string;

    for(int i = 0; i < data.NBalias; i++)
    {
        if(strcmp(data.alias[i].name, name) == 0)
        {
            /* Shift remaining entries */
            for(int j = i;
                    j < data.NBalias - 1; j++)
            {
                data.alias[j] = data.alias[j + 1];
            }
            data.NBalias--;
            cli_alias_save();
            printf("Alias removed: %s\n", name);
            return RETURN_SUCCESS;
        }
    }

    printf("Alias '%s' not found\n", name);
    return RETURN_FAILURE;
}

/**
 * @brief CLI handler: aliases
 */
errno_t cli_alias_list(void)
{
    if(data.NBalias == 0)
    {
        printf("No aliases defined.\n");
        return RETURN_SUCCESS;
    }
    printf("--- Aliases (%d) ---\n",
           data.NBalias);
    for(int i = 0; i < data.NBalias; i++)
    {
        printf("  %-16s = %s\n",
               data.alias[i].name,
               data.alias[i].cmd);
    }
    return RETURN_SUCCESS;
}

/**
 * @brief Expand alias in data.CLIcmdline
 *
 * If the first word matches an alias name,
 * substitute it with the alias command, keeping
 * any trailing arguments.
 */
static void cli_alias_expand(void)
{
    if(data.NBalias == 0)
    {
        return;
    }

    /* Extract first word */
    char firstword[CLI_ALIAS_NAMELEN];
    int  fwlen = 0;
    const char *p = data.CLIcmdline;

    /* Skip leading whitespace */
    while(*p == ' ' || *p == '\t')
    {
        p++;
    }
    while(*p != '\0' && *p != ' '
            && *p != '\t' && *p != '\n')
    {
        if(fwlen < CLI_ALIAS_NAMELEN - 1)
        {
            firstword[fwlen++] = *p;
        }
        p++;
    }
    firstword[fwlen] = '\0';

    if(fwlen == 0)
    {
        return;
    }

    /* Search aliases */
    for(int i = 0; i < data.NBalias; i++)
    {
        if(strcmp(data.alias[i].name,
                 firstword) == 0)
        {
            /* Build expanded line */
            char expanded[STRINGMAXLEN_CLICMDLINE];
            snprintf(expanded,
                     STRINGMAXLEN_CLICMDLINE,
                     "%s%s",
                     data.alias[i].cmd,
                     p); /* p points to rest */
            strncpy(data.CLIcmdline, expanded,
                    STRINGMAXLEN_CLICMDLINE - 1);
            data.CLIcmdline[
                STRINGMAXLEN_CLICMDLINE - 1]
                = '\0';
            return;
        }
    }
}


/*
 * ============================================================
 *  Watch Command
 * ============================================================
 */

/**
 * @brief CLI handler: watch <interval_ms> <command>
 *
 * Repeats a command at a fixed interval with
 * in-place terminal refresh. Press any key to stop.
 */
errno_t cli_watch(void)
{
    if(data.cmdNBarg < 3)
    {
        printf(
            "Usage: watch <interval_ms>"
            " <command...>\n");
        return RETURN_FAILURE;
    }

    long interval_ms =
        data.cmdargtoken[1].val.numl;
    if(interval_ms < 10)
    {
        interval_ms = 10;
    }

    /* Build command from remaining args */
    char watchcmd[STRINGMAXLEN_CLICMDLINE];
    watchcmd[0] = '\0';
    for(long a = 2; a < data.cmdNBarg; a++)
    {
        if(a > 2)
        {
            strncat(watchcmd, " ",
                    STRINGMAXLEN_CLICMDLINE
                    - strlen(watchcmd) - 1);
        }
        strncat(watchcmd,
                data.cmdargtoken[a].val.string,
                STRINGMAXLEN_CLICMDLINE
                - strlen(watchcmd) - 1);
    }

    /* Switch terminal to raw mode so we can
     * detect single keypresses without Enter.
     * Readline leaves the terminal in cooked
     * mode which buffers input. */
    struct termios orig_termios;
    struct termios raw_termios;
    tcgetattr(STDIN_FILENO, &orig_termios);
    raw_termios = orig_termios;
    raw_termios.c_lflag &=
        ~((tcflag_t) ICANON | (tcflag_t) ECHO);
    raw_termios.c_cc[VMIN]  = 0;
    raw_termios.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW,
              &raw_termios);

    /* Loop until keypress */
    for(;;)
    {
        /* Clear screen, move cursor to top */
        printf("\033[2J\033[H");

        /* Print header */
        {
            time_t now = time(NULL);
            struct tm *tm = localtime(&now);
            printf(
                "Every %ldms: %s   "
                "%02d:%02d:%02d"
                "  (press any key to stop)\n\n",
                interval_ms, watchcmd,
                tm->tm_hour, tm->tm_min,
                tm->tm_sec);
        }

        /* Execute the command */
        strncpy(data.CLIcmdline, watchcmd,
                STRINGMAXLEN_CLICMDLINE - 1);
        data.CLIcmdline[
            STRINGMAXLEN_CLICMDLINE - 1] = '\0';
        CLI_execute_line();

        fflush(stdout);

        /* Sleep in small increments, checking
         * for keypress */
        {
            long slept = 0;
            long step = 50000; /* 50 ms */
            while(slept < interval_ms * 1000)
            {
                struct timeval tv;
                fd_set fds;
                tv.tv_sec  = 0;
                tv.tv_usec = step;
                FD_ZERO(&fds);
                FD_SET(STDIN_FILENO, &fds);
                int r = select(
                    STDIN_FILENO + 1,
                    &fds, NULL, NULL, &tv);
                if(r > 0)
                {
                    /* Consume the keypress */
                    char discard;
                    if(read(STDIN_FILENO,
                            &discard, 1) > 0)
                    {
                        /* ignore value */
                    }
                    goto watch_done;
                }
                slept += step;
            }
        }
    }

watch_done:
    /* Restore original terminal settings */
    tcsetattr(STDIN_FILENO, TCSANOW,
              &orig_termios);
    printf("\nwatch stopped.\n");

    return RETURN_SUCCESS;
}


/*
 * ============================================================
 *  Startup Script (~/.milkrc)
 * ============================================================
 */

void cli_milkrc_load(void)
{
    char rcpath[STRINGMAXLEN_FULLFILENAME];
    snprintf(rcpath,
             STRINGMAXLEN_FULLFILENAME,
             "%s/.milkrc", getenv("HOME"));

    FILE *fp = fopen(rcpath, "r");
    if(fp == NULL)
    {
        return;
    }

    char line[STRINGMAXLEN_CLICMDLINE];
    while(fgets(line, STRINGMAXLEN_CLICMDLINE,
                fp) != NULL)
    {
        size_t len = strlen(line);
        if(len > 0 && line[len - 1] == '\n')
        {
            line[len - 1] = '\0';
        }
        const char *p = line;
        while(*p == ' ' || *p == '\t')
        {
            p++;
        }
        if(*p == '\0' || *p == '#')
        {
            continue;
        }
        strncpy(data.CLIcmdline, line,
                STRINGMAXLEN_CLICMDLINE - 1);
        data.CLIcmdline[
            STRINGMAXLEN_CLICMDLINE - 1] = '\0';
        CLI_execute_line();
    }
    fclose(fp);
}


/*
 * ============================================================
 *  Command Timing
 * ============================================================
 */

errno_t cli_time(void)
{
    if(data.cmdNBarg < 2)
    {
        printf("Usage: time <command...>\n");
        return RETURN_FAILURE;
    }
    char timecmd[STRINGMAXLEN_CLICMDLINE];
    timecmd[0] = '\0';
    for(long a = 1; a < data.cmdNBarg; a++)
    {
        if(a > 1)
        {
            strncat(timecmd, " ",
                    STRINGMAXLEN_CLICMDLINE
                    - strlen(timecmd) - 1);
        }
        strncat(timecmd,
                data.cmdargtoken[a].val.string,
                STRINGMAXLEN_CLICMDLINE
                - strlen(timecmd) - 1);
    }
    struct timespec t0;
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    strncpy(data.CLIcmdline, timecmd,
            STRINGMAXLEN_CLICMDLINE - 1);
    data.CLIcmdline[
        STRINGMAXLEN_CLICMDLINE - 1] = '\0';
    CLI_execute_line();

    clock_gettime(CLOCK_MONOTONIC, &t1);
    {
        double elapsed =
            (double)(t1.tv_sec - t0.tv_sec)
            + 1.0e-9
            * (double)(t1.tv_nsec - t0.tv_nsec);
        printf(
            "\n\033[33mElapsed: %.6f s\033[0m\n",
            elapsed);
    }
    return RETURN_SUCCESS;
}


/*
 * ============================================================
 *  Command Statistics
 * ============================================================
 */

errno_t cli_cmdstats(void)
{
    typedef struct
    {
        const char *key;
        uint32_t    count;
    } CmdStatEntry;

    CmdStatEntry entries[DATA_NB_MAX_COMMAND];
    int nused = 0;
    for(uint32_t i = 0; i < data.NBcmd; i++)
    {
        if(data.cmd[i].callcount > 0)
        {
            entries[nused].key =
                data.cmd[i].key;
            entries[nused].count =
                data.cmd[i].callcount;
            nused++;
        }
    }
    if(nused == 0)
    {
        printf("No commands executed yet.\n");
        return RETURN_SUCCESS;
    }
    for(int i = 1; i < nused; i++)
    {
        CmdStatEntry tmp = entries[i];
        int j = i - 1;
        while(j >= 0
                && entries[j].count < tmp.count)
        {
            entries[j + 1] = entries[j];
            j--;
        }
        entries[j + 1] = tmp;
    }
    int show = nused < 20 ? nused : 20;
    printf("\n\033[1mCommand usage "
           "(top %d):\033[0m\n", show);
    printf("  %-30s  %s\n", "COMMAND", "CALLS");
    printf("  %-30s  %s\n",
           "------------------------------",
           "-----");
    for(int i = 0; i < show; i++)
    {
        printf("  %-30s  %u\n",
               entries[i].key,
               entries[i].count);
    }
    printf("\n");
    return RETURN_SUCCESS;
}


/*
 * ============================================================
 *  Syntax Highlighting (optional, ON by default)
 * ============================================================
 */

#ifdef USE_READLINE

errno_t cli_timing_toggle(void)
{
    if(data.cmdNBarg >= 2)
    {
        const char *arg =
            data.cmdargtoken[1].val.string;
        if(strcmp(arg, "on") == 0
                || strcmp(arg, "1") == 0)
        {
            data.print_cmd_timing = 1;
            printf("Command execution timing ON\n");
        }
        else if(strcmp(arg, "off") == 0
                || strcmp(arg, "0") == 0)
        {
            data.print_cmd_timing = 0;
            printf("Command execution timing OFF\n");
        }
        else
        {
            printf("Usage: cli.timing [on|off]\n");
        }
    }
    else
    {
        data.print_cmd_timing =
            !data.print_cmd_timing;
        printf("Command execution timing %s\n",
               data.print_cmd_timing
               ? "ON" : "OFF");
    }
    return RETURN_SUCCESS;
}

errno_t cli_syntax_highlight_toggle(void)
{
    if(data.cmdNBarg >= 2)
    {
        const char *arg =
            data.cmdargtoken[1].val.string;
        if(strcmp(arg, "on") == 0
                || strcmp(arg, "1") == 0)
        {
            data.syntax_highlight = 1;
            printf("Syntax highlighting ON\n");
        }
        else if(strcmp(arg, "off") == 0
                || strcmp(arg, "0") == 0)
        {
            data.syntax_highlight = 0;
            printf("Syntax highlighting OFF\n");
        }
        else
        {
            printf("Usage: synhl [on|off]\n");
        }
    }
    else
    {
        data.syntax_highlight =
            !data.syntax_highlight;
        printf("Syntax highlighting %s\n",
               data.syntax_highlight
               ? "ON" : "OFF");
    }
    return RETURN_SUCCESS;
}

static int cli_is_command(const char *word)
{
    for(uint32_t i = 0; i < data.NBcmd; i++)
    {
        if(strcmp(data.cmd[i].key, word) == 0)
        {
            return 1;
        }
    }
    return 0;
}

static void cli_highlight_redisplay(void)
{
    if(!data.syntax_highlight)
    {
        rl_redisplay();
        return;
    }

    /*
     * Let readline draw normally first so its
     * internal cursor state stays consistent.
     * Then overwrite just the first word in color.
     */
    rl_redisplay();

    /* Find the first word boundaries */
    int ws = 0;
    while(rl_line_buffer[ws] == ' '
            || rl_line_buffer[ws] == '\t')
    {
        ws++;
    }
    int we = ws;
    while(rl_line_buffer[we] != '\0'
            && rl_line_buffer[we] != ' '
            && rl_line_buffer[we] != '\t')
    {
        we++;
    }
    if(we == ws)
    {
        fflush(stdout);
        return;
    }

    /* Extract first word */
    char firstword[200];
    int fwlen = we - ws;
    if(fwlen > 199)
    {
        fwlen = 199;
    }
    memcpy(firstword, rl_line_buffer + ws,
           (size_t) fwlen);
    firstword[fwlen] = '\0';

    /* Pick color */
    const char *col;
    if(cli_is_command(firstword))
    {
        col = "\033[32m"; /* green */
    }
    else
    {
        col = "\033[31m"; /* red */
    }

    /*
     * After rl_redisplay(), cursor is at rl_point.
     * Move back to the first word, overwrite with
     * color, then restore cursor position.
     * Use cursor-relative movement (not absolute
     * column) to avoid prompt-width errors from
     * invisible escape sequences in the prompt.
     */
    fprintf(rl_outstream, "\033[s");  /* ANSI save */
    {
        int back = rl_point - ws;
        if(back > 0)
        {
            fprintf(rl_outstream,
                    "\033[%dD", back);
        }
    }
    fprintf(rl_outstream, "%s%s\033[0m",
            col, firstword);
    fprintf(rl_outstream, "\033[u");  /* ANSI restore */
    fflush(rl_outstream);
}

#endif /* USE_READLINE */


/*
 * ============================================================
 *  Persistent History (~/.milk_history)
 * ============================================================
 */

#define MILK_HISTORY_MAXLINES 1000

void cli_history_load(void)
{
#ifdef USE_READLINE
    char hpath[STRINGMAXLEN_FULLFILENAME];
    snprintf(hpath,
             STRINGMAXLEN_FULLFILENAME,
             "%s/.milk_history",
             getenv("HOME"));
    read_history(hpath);
#endif
}

void cli_history_save(void)
{
#ifdef USE_READLINE
    char hpath[STRINGMAXLEN_FULLFILENAME];
    snprintf(hpath,
             STRINGMAXLEN_FULLFILENAME,
             "%s/.milk_history",
             getenv("HOME"));
    write_history(hpath);
    history_truncate_file(hpath,
                          MILK_HISTORY_MAXLINES);
#endif
}


/*
 * ============================================================
 *  History Expansion (!! and !$)
 * ============================================================
 *
 * Called at the very start of CLI_execute_line(),
 * before alias expansion.
 *
 * !!    → replace with last executed command
 * !$    → replace with last argument of previous cmd
 * !<prefix> → last command starting with <prefix>
 */

static void cli_history_expand(void)
{
#ifdef USE_READLINE
    char *line = data.CLIcmdline;

    /* Quick check: must start with '!' */
    if(line[0] != '!')
    {
        return;
    }

    /* !! — replay last command */
    if(line[1] == '!')
    {
        HIST_ENTRY *prev = history_get(
            history_length);
        if(prev != NULL)
        {
            char suffix[STRINGMAXLEN_CLICMDLINE];
            suffix[0] = '\0';
            if(line[2] != '\0')
            {
                strncpy(suffix, line + 2,
                        STRINGMAXLEN_CLICMDLINE
                        - 1);
                suffix[
                    STRINGMAXLEN_CLICMDLINE - 1]
                    = '\0';
            }
            snprintf(data.CLIcmdline,
                     STRINGMAXLEN_CLICMDLINE,
                     "%s%s",
                     prev->line, suffix);
            printf(">> %s\n", data.CLIcmdline);
        }
        return;
    }

    /* !$ — last argument of previous command */
    if(line[1] == '$')
    {
        if(data.last_argument[0] != '\0')
        {
            char rest[STRINGMAXLEN_CLICMDLINE];
            strncpy(rest, line + 2,
                    STRINGMAXLEN_CLICMDLINE - 1);
            rest[
                STRINGMAXLEN_CLICMDLINE - 1]
                = '\0';
            snprintf(data.CLIcmdline,
                     STRINGMAXLEN_CLICMDLINE,
                     "%s%s",
                     data.last_argument, rest);
            printf(">> %s\n", data.CLIcmdline);
        }
        return;
    }

    /* !<prefix> — last command starting with it */
    {
        const char *prefix = line + 1;
        size_t plen = strlen(prefix);
        /* Trim trailing spaces from prefix */
        while(plen > 0
                && prefix[plen - 1] == ' ')
        {
            plen--;
        }
        if(plen == 0)
        {
            return;
        }
        HIST_ENTRY **hist = history_list();
        if(hist == NULL)
        {
            return;
        }
        int hlen = history_length;
        for(int i = hlen - 1; i >= 0; i--)
        {
            if(strncmp(hist[i]->line,
                       prefix, plen) == 0)
            {
                strncpy(data.CLIcmdline,
                        hist[i]->line,
                        STRINGMAXLEN_CLICMDLINE
                        - 1);
                data.CLIcmdline[
                    STRINGMAXLEN_CLICMDLINE - 1]
                    = '\0';
                printf(">> %s\n",
                       data.CLIcmdline);
                return;
            }
        }
        printf("!%.*s: event not found\n",
               (int) plen, prefix);
        data.CLIcmdline[0] = '\0';
    }
#endif
}


/**
 * @brief Save last argument after command execution
 */
static void cli_save_last_argument(void)
{
    if(data.cmdNBarg > 1)
    {
        long last = data.cmdNBarg - 1;
        strncpy(data.last_argument,
                data.cmdargtoken[last].val.string,
                sizeof(data.last_argument) - 1);
        data.last_argument[
            sizeof(data.last_argument) - 1]
            = '\0';
    }
}


/*
 * ============================================================
 *  Source Command — execute a milk script file
 * ============================================================
 */

errno_t cli_source(void)
{
    if(data.cmdNBarg < 2)
    {
        printf("Usage: source <filename>\n");
        return RETURN_FAILURE;
    }
    const char *fname =
        data.cmdargtoken[1].val.string;
    FILE *fp = fopen(fname, "r");
    if(fp == NULL)
    {
        printf("source: cannot open '%s'\n",
               fname);
        return RETURN_FAILURE;
    }
    char line[STRINGMAXLEN_CLICMDLINE];
    int lineno = 0;
    while(fgets(line, STRINGMAXLEN_CLICMDLINE,
                fp) != NULL)
    {
        lineno++;
        {
            size_t len = strlen(line);
            if(len > 0
                    && line[len - 1] == '\n')
            {
                line[len - 1] = '\0';
            }
        }
        const char *p = line;
        while(*p == ' ' || *p == '\t')
        {
            p++;
        }
        if(*p == '\0' || *p == '#')
        {
            continue;
        }
        strncpy(data.CLIcmdline, line,
                STRINGMAXLEN_CLICMDLINE - 1);
        data.CLIcmdline[
            STRINGMAXLEN_CLICMDLINE - 1] = '\0';
        errno_t ret = CLI_execute_line();
        if(ret != RETURN_SUCCESS)
        {
            printf(
                "\033[31m[source:%s:%d] "
                "error\033[0m\n",
                fname, lineno);
        }
    }
    fclose(fp);
    return RETURN_SUCCESS;
}


/*
 * ============================================================
 *  Save Script — export variables and functions
 * ============================================================
 */

/**
 * @brief Write all CLI variables and user functions
 *        to a file that can be sourced later.
 *
 * Usage: savescript <filename>
 */
errno_t cli_savescript(void)
{
    if(data.cmdNBarg < 2)
    {
        printf("Usage: savescript <filename>\n");
        return RETURN_FAILURE;
    }
    const char *fname =
        data.cmdargtoken[1].val.string;
    FILE *fp = fopen(fname, "w");
    if(fp == NULL)
    {
        printf("savescript: cannot open "
               "'%s' for writing\n", fname);
        return RETURN_FAILURE;
    }

    fprintf(fp, "# milk-cli script\n");
    fprintf(fp,
            "# saved by savescript command\n\n");

    /* Export variables */
    int nv = 0;
    for(int i = 0; i < CLI_MAX_VARS; i++)
    {
        if(cli_vars[i].used)
        {
            fprintf(fp, "%s=%s\n",
                    cli_vars[i].name,
                    cli_vars[i].val);
            nv++;
        }
    }
    if(nv > 0)
    {
        fprintf(fp, "\n");
    }

    /* Export user-defined functions */
    int nf = 0;
    for(int i = 0; i < CLI_MAX_FUNCS; i++)
    {
        if(cli_funcs[i].used)
        {
            fprintf(fp, "function %s {\n",
                    cli_funcs[i].name);
            for(int j = 0;
                j < cli_funcs[i].nbody; j++)
            {
                fprintf(fp, "%s\n",
                        cli_funcs[i].body[j]);
            }
            fprintf(fp, "}\n\n");
            nf++;
        }
    }

    fclose(fp);
    printf("Saved %d variables, %d functions "
           "to '%s'\n", nv, nf, fname);
    return RETURN_SUCCESS;
}


/*
 * ============================================================
 *  Save History — export readline history
 * ============================================================
 */

/**
 * @brief Write readline command history to a file.
 *
 * Usage: savehistory <filename>
 */
errno_t cli_savehistory(void)
{
    if(data.cmdNBarg < 2)
    {
        printf("Usage: savehistory "
               "<filename>\n");
        return RETURN_FAILURE;
    }
    const char *fname =
        data.cmdargtoken[1].val.string;

#ifdef USE_READLINE
    if(write_history(fname) != 0)
    {
        printf("savehistory: failed to write "
               "'%s'\n", fname);
        return RETURN_FAILURE;
    }
    printf("History saved to '%s'\n", fname);
    return RETURN_SUCCESS;
#else
    printf("savehistory: readline not "
           "available\n");
    (void) fname;
    return RETURN_FAILURE;
#endif
}


/*
 * ============================================================
 *  Configurable Prompt — setprompt command
 * ============================================================
 *
 * Format tokens:
 *   %h = hostname
 *   %u = username
 *   %d = cwd basename
 *   %t = HH:MM:SS
 *   %n = CLI process name (data.processname)
 */

/** prompt_format stored in data struct is TBD;
 *  for now use a file-scope buffer. */
static char cli_prompt_format[200] = "";

/**
 * @brief Build prompt string from format tokens
 */
void cli_build_prompt(
    const char *fmt,
    char       *out,
    int         maxlen
)
{
    int pos = 0;
    for(int i = 0; fmt[i] != '\0'
            && pos < maxlen - 1; i++)
    {
        if(fmt[i] == '%' && fmt[i + 1] != '\0')
        {
            i++;
            switch(fmt[i])
            {
            case 'h':
            {
                char hn[64];
                gethostname(hn, sizeof(hn));
                pos += snprintf(out + pos,
                    (size_t)(maxlen - pos),
                    "%s", hn);
                break;
            }
            case 'u':
            {
                const char *u = getenv("USER");
                pos += snprintf(out + pos,
                    (size_t)(maxlen - pos),
                    "%s", u ? u : "?");
                break;
            }
            case 'd':
            {
                char cwd[256];
                if(getcwd(cwd, sizeof(cwd)))
                {
                    char *base = strrchr(cwd,
                                         '/');
                    pos += snprintf(out + pos,
                        (size_t)(maxlen - pos),
                        "%s",
                        base ? base + 1 : cwd);
                }
                break;
            }
            case 't':
            {
                time_t now = time(NULL);
                struct tm *tm = localtime(&now);
                pos += (int) strftime(
                    out + pos,
                    (size_t)(maxlen - pos),
                    "%H:%M:%S", tm);
                break;
            }
            case 'n':
                pos += snprintf(out + pos,
                    (size_t)(maxlen - pos),
                    "%s", data.processname);
                break;
            default:
                if(pos < maxlen - 2)
                {
                    out[pos++] = '%';
                    out[pos++] = fmt[i];
                }
                break;
            }
        }
        else
        {
            out[pos++] = fmt[i];
        }
    }
    out[pos] = '\0';
}

errno_t cli_setprompt(void)
{
    if(data.cmdNBarg < 2)
    {
        if(cli_prompt_format[0] != '\0')
        {
            printf("Current prompt format: "
                   "'%s'\n",
                   cli_prompt_format);
        }
        else
        {
            printf("Using default prompt\n");
        }
        printf("Tokens: %%h=host %%u=user "
               "%%d=dir %%t=time %%n=name\n");
        return RETURN_SUCCESS;
    }
    strncpy(cli_prompt_format,
            data.cmdargtoken[1].val.string,
            sizeof(cli_prompt_format) - 1);
    cli_prompt_format[
        sizeof(cli_prompt_format) - 1] = '\0';
    printf("Prompt set to: '%s'\n",
           cli_prompt_format);
    return RETURN_SUCCESS;
}


/*
 * ============================================================
 *  Command Bookmarks
 * ============================================================
 */

#define BOOKMARK_MAX       64
#define BOOKMARK_NAMELEN   64
#define BOOKMARK_CMDLEN   512

struct bookmark_entry
{
    char name[BOOKMARK_NAMELEN];
    char cmd[BOOKMARK_CMDLEN];
};

static struct bookmark_entry
    bookmarks[BOOKMARK_MAX];
static int bookmark_count = 0;

/**
 * @brief Load bookmarks from ~/.milk_bookmarks
 */
void cli_bookmark_load(void)
{
    char path[STRINGMAXLEN_FULLFILENAME];
    snprintf(path,
             STRINGMAXLEN_FULLFILENAME,
             "%s/.milk_bookmarks",
             getenv("HOME"));
    FILE *fp = fopen(path, "r");
    if(fp == NULL)
    {
        return;
    }
    bookmark_count = 0;
    char line[1024];
    while(fgets(line, (int) sizeof(line), fp)
            && bookmark_count < BOOKMARK_MAX)
    {
        size_t len = strlen(line);
        if(len > 0 && line[len - 1] == '\n')
        {
            line[len - 1] = '\0';
        }
        char *tab = strchr(line, '\t');
        if(tab == NULL)
        {
            continue;
        }
        *tab = '\0';
        strncpy(
            bookmarks[bookmark_count].name,
            line,
            BOOKMARK_NAMELEN - 1);
        bookmarks[bookmark_count].name[
            BOOKMARK_NAMELEN - 1] = '\0';
        strncpy(
            bookmarks[bookmark_count].cmd,
            tab + 1,
            BOOKMARK_CMDLEN - 1);
        bookmarks[bookmark_count].cmd[
            BOOKMARK_CMDLEN - 1] = '\0';
        bookmark_count++;
    }
    fclose(fp);
}

/**
 * @brief Save bookmarks to ~/.milk_bookmarks
 */
static void cli_bookmark_save(void)
{
    char path[STRINGMAXLEN_FULLFILENAME];
    snprintf(path,
             STRINGMAXLEN_FULLFILENAME,
             "%s/.milk_bookmarks",
             getenv("HOME"));
    FILE *fp = fopen(path, "w");
    if(fp == NULL)
    {
        return;
    }
    for(int i = 0; i < bookmark_count; i++)
    {
        fprintf(fp, "%s\t%s\n",
                bookmarks[i].name,
                bookmarks[i].cmd);
    }
    fclose(fp);
}

errno_t cli_bookmark(void)
{
    if(data.cmdNBarg < 2)
    {
        printf("Usage:\n"
               "  bookmark save <name> "
               "\"cmd1 ; cmd2\"\n"
               "  bookmark run  <name>\n"
               "  bookmark list\n"
               "  bookmark rm   <name>\n");
        return RETURN_SUCCESS;
    }
    const char *action =
        data.cmdargtoken[1].val.string;

    if(strcmp(action, "list") == 0)
    {
        if(bookmark_count == 0)
        {
            printf("No bookmarks saved\n");
        }
        for(int i = 0; i < bookmark_count; i++)
        {
            printf("  \033[1m%-16s\033[0m %s\n",
                   bookmarks[i].name,
                   bookmarks[i].cmd);
        }
        return RETURN_SUCCESS;
    }

    if(strcmp(action, "save") == 0)
    {
        if(data.cmdNBarg < 4)
        {
            printf("Usage: bookmark save "
                   "<name> \"cmd\"\n");
            return RETURN_FAILURE;
        }
        if(bookmark_count >= BOOKMARK_MAX)
        {
            printf("Bookmark limit reached\n");
            return RETURN_FAILURE;
        }
        strncpy(
            bookmarks[bookmark_count].name,
            data.cmdargtoken[2].val.string,
            BOOKMARK_NAMELEN - 1);
        bookmarks[bookmark_count].name[
            BOOKMARK_NAMELEN - 1] = '\0';
        /* Join remaining args as command */
        {
            char cmd[BOOKMARK_CMDLEN] = "";
            for(long a = 3;
                    a < data.cmdNBarg; a++)
            {
                if(a > 3)
                {
                    strncat(cmd, " ",
                        BOOKMARK_CMDLEN
                        - strlen(cmd) - 1);
                }
                strncat(cmd,
                    data.cmdargtoken[a]
                        .val.string,
                    BOOKMARK_CMDLEN
                    - strlen(cmd) - 1);
            }
            strncpy(
                bookmarks[bookmark_count].cmd,
                cmd, BOOKMARK_CMDLEN - 1);
            bookmarks[bookmark_count].cmd[
                BOOKMARK_CMDLEN - 1] = '\0';
        }
        bookmark_count++;
        cli_bookmark_save();
        printf("Bookmark '%s' saved\n",
               data.cmdargtoken[2].val.string);
        return RETURN_SUCCESS;
    }

    if(strcmp(action, "run") == 0)
    {
        if(data.cmdNBarg < 3)
        {
            printf("Usage: bookmark run "
                   "<name>\n");
            return RETURN_FAILURE;
        }
        const char *name =
            data.cmdargtoken[2].val.string;
        for(int i = 0; i < bookmark_count; i++)
        {
            if(strcmp(bookmarks[i].name,
                     name) == 0)
            {
                strncpy(data.CLIcmdline,
                        bookmarks[i].cmd,
                        STRINGMAXLEN_CLICMDLINE
                        - 1);
                data.CLIcmdline[
                    STRINGMAXLEN_CLICMDLINE - 1]
                    = '\0';
                return CLI_execute_line();
            }
        }
        printf("Bookmark '%s' not found\n",
               name);
        return RETURN_FAILURE;
    }

    if(strcmp(action, "rm") == 0)
    {
        if(data.cmdNBarg < 3)
        {
            printf("Usage: bookmark rm "
                   "<name>\n");
            return RETURN_FAILURE;
        }
        const char *name =
            data.cmdargtoken[2].val.string;
        for(int i = 0; i < bookmark_count; i++)
        {
            if(strcmp(bookmarks[i].name,
                     name) == 0)
            {
                for(int j = i;
                        j < bookmark_count - 1;
                        j++)
                {
                    bookmarks[j] =
                        bookmarks[j + 1];
                }
                bookmark_count--;
                cli_bookmark_save();
                printf("Bookmark '%s' "
                       "removed\n", name);
                return RETURN_SUCCESS;
            }
        }
        printf("Bookmark '%s' not found\n",
               name);
        return RETURN_FAILURE;
    }

    printf("Unknown bookmark action '%s'\n",
           action);
    return RETURN_FAILURE;
}


/*
 * ============================================================
 *  Session Logging
 * ============================================================
 */

static FILE *session_log_fp = NULL;
static struct timespec session_log_t0;

errno_t cli_sessionlog(void)
{
    if(data.cmdNBarg < 2)
    {
        printf("Usage: sessionlog "
               "[on|off|<filename>]\n");
        printf("Status: %s\n",
               session_log_fp ? "ON" : "OFF");
        return RETURN_SUCCESS;
    }
    const char *arg =
        data.cmdargtoken[1].val.string;

    if(strcmp(arg, "off") == 0)
    {
        if(session_log_fp)
        {
            fclose(session_log_fp);
            session_log_fp = NULL;
            printf("Session logging stopped\n");
        }
        return RETURN_SUCCESS;
    }

    /* Close previous if open */
    if(session_log_fp)
    {
        fclose(session_log_fp);
        session_log_fp = NULL;
    }

    char logpath[STRINGMAXLEN_FULLFILENAME];
    if(strcmp(arg, "on") == 0)
    {
        snprintf(logpath,
                 STRINGMAXLEN_FULLFILENAME,
                 "%s/.milk_session.log",
                 getenv("HOME"));
    }
    else
    {
        strncpy(logpath, arg,
                STRINGMAXLEN_FULLFILENAME - 1);
        logpath[
            STRINGMAXLEN_FULLFILENAME - 1]
            = '\0';
    }

    session_log_fp = fopen(logpath, "a");
    if(session_log_fp == NULL)
    {
        printf("Cannot open '%s'\n", logpath);
        return RETURN_FAILURE;
    }
    clock_gettime(CLOCK_MONOTONIC,
                  &session_log_t0);
    printf("Session logging to '%s'\n", logpath);

    /* Write session start marker */
    {
        time_t now = time(NULL);
        char tbuf[64];
        strftime(tbuf, sizeof(tbuf),
                 "%Y-%m-%dT%H:%M:%S",
                 localtime(&now));
        fprintf(session_log_fp,
                "# Session started %s\n", tbuf);
        fflush(session_log_fp);
    }
    return RETURN_SUCCESS;
}

/**
 * @brief Log a command to session log if active
 */
static void cli_session_log_cmd(
    const char *cmd
)
{
    if(session_log_fp == NULL)
    {
        return;
    }
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double elapsed_ms =
        (double)(now.tv_sec
                 - session_log_t0.tv_sec)
        * 1000.0
        + (double)(now.tv_nsec
                   - session_log_t0.tv_nsec)
        / 1.0e6;
    {
        time_t t = time(NULL);
        char tbuf[64];
        strftime(tbuf, sizeof(tbuf),
                 "%Y-%m-%dT%H:%M:%S",
                 localtime(&t));
        fprintf(session_log_fp,
                "[%s] [%10.1f ms] %s\n",
                tbuf, elapsed_ms, cmd);
        fflush(session_log_fp);
    }
}


/*
 * ============================================================
 *  Brace Expansion
 * ============================================================
 *
 * Expand {N..M} into space-separated integers,
 * and {N..M..S} with step S.
 */

/**
 * @brief Expand {N..M} and {N..M..S} brace ranges
 *
 * Replaces tokens like {1..5} with "1 2 3 4 5"
 * and {0..10..2} with "0 2 4 6 8 10".
 */
static void emit_str(
    char       *out,
    int        *opos,
    int         maxlen,
    const char *s
);
static void cli_expand_braces(
    char *line,
    int   maxlen
)
{
    char out[STRINGMAXLEN_CLICMDLINE];
    int  opos = 0;
    int  i = 0;

    while(line[i] != '\0'
          && opos < maxlen - 1)
    {
        if(line[i] == '{')
        {
            /* Try {N..M} or {N..M..S} */
            char *endp = NULL;
            long sv =
                strtol(line + i + 1,
                       &endp, 10);
            if(endp != NULL
               && endp[0] == '.'
               && endp[1] == '.')
            {
                char *endp2 = NULL;
                long ev =
                    strtol(endp + 2,
                           &endp2, 10);
                long step = 1;
                if(endp2 != NULL
                   && endp2[0] == '.'
                   && endp2[1] == '.')
                {
                    char *endp3 = NULL;
                    step =
                        strtol(endp2 + 2,
                               &endp3,
                               10);
                    endp2 = endp3;
                }
                if(endp2 != NULL
                   && *endp2 == '}'
                   && step != 0)
                {
                    int first = 1;
                    if(sv <= ev)
                    {
                        if(step < 0)
                        {
                            step = -step;
                        }
                        for(long v = sv;
                            v <= ev;
                            v += step)
                        {
                            char nb[32];
                            snprintf(
                                nb,
                                sizeof(nb),
                                "%s%ld",
                                first
                                ? "" : " ",
                                v);
                            first = 0;
                            emit_str(
                                out, &opos,
                                maxlen, nb);
                        }
                    }
                    else
                    {
                        if(step > 0)
                        {
                            step = -step;
                        }
                        for(long v = sv;
                            v >= ev;
                            v += step)
                        {
                            char nb[32];
                            snprintf(
                                nb,
                                sizeof(nb),
                                "%s%ld",
                                first
                                ? "" : " ",
                                v);
                            first = 0;
                            emit_str(
                                out, &opos,
                                maxlen, nb);
                        }
                    }
                    i = (int)(endp2
                              - line) + 1;
                    continue;
                }
            }
            out[opos++] = line[i++];
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


/**
 * @brief Expand tilde (~) to $HOME
 *
 * Replaces ~ or ~/path at start of tokens
 * with the HOME environment variable value.
 */
static void cli_expand_tilde(
    char *line,
    int   maxlen
)
{
    const char *home = getenv("HOME");
    if(home == NULL)
    {
        return;
    }
    char out[STRINGMAXLEN_CLICMDLINE];
    int  opos = 0;
    int  i = 0;
    int  at_tok_start = 1;
    int  in_sq = 0;
    int  in_dq = 0;

    while(line[i] != '\0'
          && opos < maxlen - 1)
    {
        char c = line[i];
        if(c == '\'' && !in_dq)
        {
            in_sq = !in_sq;
            out[opos++] = line[i++];
            at_tok_start = 0;
            continue;
        }
        if(c == '"' && !in_sq)
        {
            in_dq = !in_dq;
            out[opos++] = line[i++];
            at_tok_start = 0;
            continue;
        }
        if(c == ' ' || c == '\t')
        {
            out[opos++] = line[i++];
            at_tok_start = 1;
            continue;
        }
        if(at_tok_start
           && !in_sq && !in_dq
           && c == '~'
           && (line[i + 1] == '/'
               || line[i + 1] == ' '
               || line[i + 1] == '\t'
               || line[i + 1] == '\0'))
        {
            /* Replace ~ with $HOME */
            const char *hp = home;
            while(*hp != '\0'
                  && opos < maxlen - 1)
            {
                out[opos++] = *hp++;
            }
            i++; /* skip ~ */
            at_tok_start = 0;
            continue;
        }
        out[opos++] = line[i++];
        at_tok_start = 0;
    }
    out[opos] = '\0';
    strncpy(line, out, (size_t) maxlen);
    line[maxlen - 1] = '\0';
}

/**
 * @brief Expand filename globs (* and ?)
 *
 * Tokens containing * or ? that are not inside
 * quotes are expanded using POSIX glob().
 * Example: *.fits → file1.fits file2.fits
 */
static void cli_expand_globs(
    char *line,
    int   maxlen
)
{
    char out[STRINGMAXLEN_CLICMDLINE];
    int  opos = 0;
    int  i = 0;
    int  in_sq = 0;
    int  in_dq = 0;

    while(line[i] != '\0'
          && opos < maxlen - 1)
    {
        char c = line[i];
        if(c == '\'' && !in_dq)
        {
            in_sq = !in_sq;
            out[opos++] = line[i++];
            continue;
        }
        if(c == '"' && !in_sq)
        {
            in_dq = !in_dq;
            out[opos++] = line[i++];
            continue;
        }
        if(in_sq || in_dq)
        {
            out[opos++] = line[i++];
            continue;
        }
        if(c == ' ' || c == '\t')
        {
            out[opos++] = line[i++];
            continue;
        }
        /* Extract token */
        int tstart = i;
        int has_glob = 0;
        while(line[i] != '\0'
              && line[i] != ' '
              && line[i] != '\t')
        {
            if(line[i] == '*'
               || line[i] == '?')
            {
                has_glob = 1;
            }
            i++;
        }
        int tlen = i - tstart;
        if(!has_glob || tlen <= 0)
        {
            for(int j = tstart;
                j < i
                && opos < maxlen - 1;
                j++)
            {
                out[opos++] = line[j];
            }
            continue;
        }
        /* Run glob */
        char pat[512];
        int plen = tlen;
        if(plen >= 512)
        {
            plen = 511;
        }
        memcpy(pat, line + tstart,
               (size_t) plen);
        pat[plen] = '\0';

        glob_t gl;
        int gret = glob(pat,
                        GLOB_NOCHECK,
                        NULL, &gl);
        if(gret == 0
           && gl.gl_pathc > 0)
        {
            for(size_t g = 0;
                g < gl.gl_pathc; g++)
            {
                if(g > 0
                   && opos < maxlen - 1)
                {
                    out[opos++] = ' ';
                }
                const char *gp =
                    gl.gl_pathv[g];
                while(*gp != '\0'
                      && opos
                      < maxlen - 1)
                {
                    out[opos++] = *gp++;
                }
            }
            globfree(&gl);
        }
        else
        {
            if(gret == 0)
            {
                globfree(&gl);
            }
            for(int j = tstart;
                j < tstart + tlen
                && opos < maxlen - 1;
                j++)
            {
                out[opos++] = line[j];
            }
        }
    }
    out[opos] = '\0';
    strncpy(line, out, (size_t) maxlen);
    line[maxlen - 1] = '\0';
}


/*
 * ============================================================
 *  Command Substitution
 * ============================================================
 *
 * Replace $(cmd) and `cmd` in the command line with
 * the standard output of the command execution.
 */

static void cli_expand_cmdsub(
    char *line,
    int   maxlen
)
{
    char out[STRINGMAXLEN_CLICMDLINE];
    int  opos = 0;
    int  i = 0;

    while(line[i] != '\0' && opos < maxlen - 1)
    {
        int is_dollar_paren =
            (line[i] == '$'
             && line[i + 1] == '('
             && line[i + 2] != '(');
        int is_backtick = (line[i] == '`');

        if(is_dollar_paren || is_backtick)
        {
            char cmd[512];
            int clen = 0;
            
            if (is_dollar_paren)
            {
                i += 2; /* Skip $( */
                while(line[i] != '\0' && line[i] != ')' && clen < 511)
                {
                    cmd[clen++] = line[i++];
                }
                if (line[i] == ')') i++; /* Skip ) */
            }
            else /* is_backtick */
            {
                i++; /* Skip ` */
                while(line[i] != '\0' && line[i] != '`' && clen < 511)
                {
                    cmd[clen++] = line[i++];
                }
                if (line[i] == '`') i++; /* Skip ` */
            }
            cmd[clen] = '\0';

            /* Execute command and read output */
            if (clen > 0)
            {
                FILE *fp = popen(cmd, "r");
                if (fp != NULL)
                {
                    char buf[1024];
                    size_t read_bytes = fread(buf, 1, sizeof(buf) - 1, fp);
                    buf[read_bytes] = '\0';
                    pclose(fp);

                    /* Strip trailing newlines */
                    while(read_bytes > 0 && (buf[read_bytes - 1] == '\n' || buf[read_bytes - 1] == '\r'))
                    {
                        buf[--read_bytes] = '\0';
                    }

                    /* Copy to output */
                    int vallen = (int) read_bytes;
                    int avail = maxlen - 1 - opos;
                    int copylen = vallen < avail ? vallen : avail;
                    memcpy(out + opos, buf, (size_t) copylen);
                    opos += copylen;
                }
            }
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


/*
 * ============================================================
 *  Environment Variable Expansion
 * ============================================================
 *
 * Replace $VAR and ${VAR} with variable values.
 * Supports string operations, arrays, and special
 * forms inside ${...}.
 */

/**
 * @brief Emit string into output buffer
 */
static void emit_str(
    char *out,
    int  *opos,
    int   maxlen,
    const char *s
)
{
    while(*s != '\0' && *opos < maxlen - 1)
    {
        out[(*opos)++] = *s++;
    }
}

/**
 * @brief Handle ${...} braced expansion
 *
 * Supports:
 *   ${var}          plain lookup
 *   ${#var}         string length
 *   ${var:-default} default if unset
 *   ${var:=default} assign if unset
 *   ${var:+alt}     alt if set
 *   ${var:?error}   error if unset
 *   ${var:off:len}  substring
 *   ${var%%pat}     strip longest suffix
 *   ${var##pat}     strip longest prefix
 *   ${var%pat}      strip shortest suffix
 *   ${var#pat}      strip shortest prefix
 *   ${var/find/rep} replace first
 *   ${var//find/rep} replace all
 *   ${arr[N]}       array element
 *   ${arr[@]}       all array elements
 *   ${#arr[@]}      array element count
 */
static void expand_braced(
    char *out,
    int  *opos,
    int   maxlen,
    const char *inner
)
{
    /* ${#...} — length or array count */
    if(inner[0] == '#')
    {
        const char *nm = inner + 1;
        /* ${#arr[@]} */
        const char *br = strchr(nm, '[');
        if(br != NULL)
        {
            char aname[CLI_VAR_NAMELEN];
            int alen = (int)(br - nm);
            if(alen >= CLI_VAR_NAMELEN)
            {
                alen = CLI_VAR_NAMELEN - 1;
            }
            memcpy(aname, nm,
                   (size_t) alen);
            aname[alen] = '\0';
            for(int k = 0;
                k < CLI_MAX_ARRAYS; k++)
            {
                if(cli_arrays[k].used
                   && strcmp(
                       cli_arrays[k].name,
                       aname) == 0)
                {
                    char nb[32];
                    snprintf(
                        nb, sizeof(nb),
                        "%d",
                        cli_arrays[k]
                        .nelem);
                    emit_str(out, opos,
                             maxlen, nb);
                    return;
                }
            }
            emit_str(out, opos,
                     maxlen, "0");
            return;
        }
        /* ${#var} — string length */
        const char *val =
            cli_var_lookup(nm);
        char lb[32];
        snprintf(lb, sizeof(lb), "%d",
                 val ? (int) strlen(val)
                 : 0);
        emit_str(out, opos, maxlen, lb);
        return;
    }

    /* ${!var} — indirect expansion */
    if(inner[0] == '!')
    {
        const char *iname =
            inner + 1;
        const char *iref =
            cli_var_lookup(iname);
        if(iref != NULL)
        {
            const char *ival =
                cli_var_lookup(iref);
            if(ival != NULL)
            {
                emit_str(out, opos,
                         maxlen,
                         ival);
            }
        }
        return;
    }

    /* ${arr[N]} or ${arr[@]} or
     * ${assoc[key]} */
    {
        const char *br =
            strchr(inner, '[');
        if(br != NULL)
        {
            char aname[CLI_VAR_NAMELEN];
            int alen = (int)(br - inner);
            if(alen >= CLI_VAR_NAMELEN)
            {
                alen =
                    CLI_VAR_NAMELEN - 1;
            }
            memcpy(aname, inner,
                   (size_t) alen);
            aname[alen] = '\0';
            const char *idx_s = br + 1;
            if(idx_s[0] == '@')
            {
                for(int k = 0;
                    k < CLI_MAX_ARRAYS;
                    k++)
                {
                    if(cli_arrays[k].used
                       && strcmp(
                           cli_arrays[k]
                           .name,
                           aname) == 0)
                    {
                        for(int e = 0;
                            e
                            < cli_arrays[k]
                            .nelem;
                            e++)
                        {
                            if(e > 0)
                            {
                                emit_str(
                                    out,
                                    opos,
                                    maxlen,
                                    " ");
                            }
                            emit_str(
                                out,
                                opos,
                                maxlen,
                                cli_arrays
                                [k]
                                .elem[e]);
                        }
                        return;
                    }
                }
                return;
            }
            int idx = (int) strtol(
                idx_s, NULL, 0);
            for(int k = 0;
                k < CLI_MAX_ARRAYS;
                k++)
            {
                if(cli_arrays[k].used
                   && strcmp(
                       cli_arrays[k]
                       .name,
                       aname) == 0)
                {
                    if(idx >= 0
                       && idx
                       < cli_arrays[k]
                       .nelem)
                    {
                        emit_str(
                            out, opos,
                            maxlen,
                            cli_arrays[k]
                            .elem[idx]);
                    }
                    return;
                }
            }
            return;
        }
    }

    /* ${assoc[key]} — associative
     * array lookup */
    {
        const char *br =
            strchr(inner, '[');
        if(br != NULL)
        {
            const char *brend =
                strchr(br, ']');
            if(brend != NULL)
            {
                char aname[
                    CLI_VAR_NAMELEN];
                int nl =
                    (int)(br
                          - inner);
                if(nl
                   >= CLI_VAR_NAMELEN)
                {
                    nl =
                        CLI_VAR_NAMELEN
                        - 1;
                }
                memcpy(aname,
                       inner,
                       (size_t) nl);
                aname[nl] = '\0';
                char key[
                    CLI_VAR_NAMELEN];
                int kl =
                    (int)(brend
                          - br - 1);
                if(kl
                   >= CLI_VAR_NAMELEN)
                {
                    kl =
                        CLI_VAR_NAMELEN
                        - 1;
                }
                memcpy(key, br + 1,
                       (size_t) kl);
                key[kl] = '\0';
                for(int k = 0;
                    k
                    < CLI_MAX_ASSOC;
                    k++)
                {
                    if(cli_assoc[k]
                        .used
                       && strcmp(
                           cli_assoc[
                               k]
                           .name,
                           aname)
                       == 0)
                    {
                        for(int e = 0;
                            e
                            < cli_assoc[
                                k]
                            .nelem;
                            e++)
                        {
                            if(strcmp(
                                cli_assoc[
                                    k]
                                .keys[e],
                                key)
                               == 0)
                            {
                                emit_str(
                                    out,
                                    opos,
                                    maxlen,
                                    cli_assoc[
                                        k]
                                    .vals[
                                        e]);
                                return;
                            }
                        }
                        return;
                    }
                }
            }
        }
    }

    /* ${var//find/rep} — replace all */
    {
        const char *ds =
            strstr(inner, "//");
        if(ds != NULL)
        {
            char vn[CLI_VAR_NAMELEN];
            int nlen = (int)(ds - inner);
            if(nlen >= CLI_VAR_NAMELEN)
            {
                nlen =
                    CLI_VAR_NAMELEN - 1;
            }
            memcpy(vn, inner,
                   (size_t) nlen);
            vn[nlen] = '\0';
            const char *find =
                ds + 2;
            const char *sl2 =
                strchr(find, '/');
            char fp[256] = "";
            char rp[256] = "";
            if(sl2 != NULL)
            {
                int fl2 =
                    (int)(sl2 - find);
                if(fl2 > 255)
                {
                    fl2 = 255;
                }
                memcpy(fp, find,
                       (size_t) fl2);
                fp[fl2] = '\0';
                strncpy(rp,
                        sl2 + 1, 255);
            }
            else
            {
                strncpy(fp, find, 255);
            }
            const char *val =
                cli_var_lookup(vn);
            if(val != NULL
               && fp[0] != '\0')
            {
                int fplen =
                    (int) strlen(fp);
                int rplen =
                    (int) strlen(rp);
                const char *s = val;
                while(*s != '\0'
                      && *opos
                      < maxlen - 1)
                {
                    if(strncmp(s, fp,
                               (size_t)
                               fplen)
                       == 0)
                    {
                        emit_str(
                            out, opos,
                            maxlen, rp);
                        s += fplen;
                    }
                    else
                    {
                        out[(*opos)++] =
                            *s++;
                    }
                }
            }
            else if(val != NULL)
            {
                emit_str(out, opos,
                         maxlen, val);
            }
            return;
        }
    }

    /* ${var/find/rep} — replace first */
    {
        const char *sl =
            strchr(inner, '/');
        if(sl != NULL)
        {
            char vn[CLI_VAR_NAMELEN];
            int nlen = (int)(sl - inner);
            if(nlen >= CLI_VAR_NAMELEN)
            {
                nlen =
                    CLI_VAR_NAMELEN - 1;
            }
            memcpy(vn, inner,
                   (size_t) nlen);
            vn[nlen] = '\0';
            const char *find = sl + 1;
            const char *sl2 =
                strchr(find, '/');
            char fp[256] = "";
            char rp[256] = "";
            if(sl2 != NULL)
            {
                int fl2 =
                    (int)(sl2 - find);
                if(fl2 > 255)
                {
                    fl2 = 255;
                }
                memcpy(fp, find,
                       (size_t) fl2);
                fp[fl2] = '\0';
                strncpy(rp,
                        sl2 + 1, 255);
            }
            else
            {
                strncpy(fp, find, 255);
            }
            const char *val =
                cli_var_lookup(vn);
            if(val != NULL
               && fp[0] != '\0')
            {
                const char *m =
                    strstr(val, fp);
                if(m != NULL)
                {
                    int pre =
                        (int)(m - val);
                    int avail =
                        maxlen - 1
                        - *opos;
                    if(pre > avail)
                    {
                        pre = avail;
                    }
                    memcpy(
                        out + *opos,
                        val,
                        (size_t) pre);
                    *opos += pre;
                    emit_str(
                        out, opos,
                        maxlen, rp);
                    emit_str(
                        out, opos,
                        maxlen,
                        m + strlen(fp));
                }
                else
                {
                    emit_str(
                        out, opos,
                        maxlen, val);
                }
            }
            return;
        }
    }

    /* ${var%%pattern} — strip suffix */
    {
        const char *pp =
            strstr(inner, "%%");
        if(pp != NULL)
        {
            char vn[CLI_VAR_NAMELEN];
            int nlen =
                (int)(pp - inner);
            if(nlen >= CLI_VAR_NAMELEN)
            {
                nlen =
                    CLI_VAR_NAMELEN - 1;
            }
            memcpy(vn, inner,
                   (size_t) nlen);
            vn[nlen] = '\0';
            const char *pat = pp + 2;
            const char *val =
                cli_var_lookup(vn);
            if(val != NULL)
            {
                const char *m =
                    strstr(val, pat);
                if(m != NULL)
                {
                    int clen =
                        (int)(m - val);
                    int avail =
                        maxlen - 1
                        - *opos;
                    if(clen > avail)
                    {
                        clen = avail;
                    }
                    memcpy(
                        out + *opos,
                        val,
                        (size_t) clen);
                    *opos += clen;
                }
                else
                {
                    emit_str(
                        out, opos,
                        maxlen, val);
                }
            }
            return;
        }
    }

    /* ${var##pattern} — strip prefix */
    {
        const char *pp =
            strstr(inner, "##");
        if(pp != NULL)
        {
            char vn[CLI_VAR_NAMELEN];
            int nlen =
                (int)(pp - inner);
            if(nlen >= CLI_VAR_NAMELEN)
            {
                nlen =
                    CLI_VAR_NAMELEN - 1;
            }
            memcpy(vn, inner,
                   (size_t) nlen);
            vn[nlen] = '\0';
            const char *pat = pp + 2;
            const char *val =
                cli_var_lookup(vn);
            if(val != NULL)
            {
                const char *m =
                    strstr(val, pat);
                if(m != NULL)
                {
                    emit_str(
                        out, opos,
                        maxlen,
                        m + strlen(pat));
                }
                else
                {
                    emit_str(
                        out, opos,
                        maxlen, val);
                }
            }
            return;
        }
    }

    /* ${var%pat} — strip shortest suffix */
    {
        const char *pp =
            strchr(inner, '%');
        if(pp != NULL)
        {
            char vn[CLI_VAR_NAMELEN];
            int nlen =
                (int)(pp - inner);
            if(nlen >= CLI_VAR_NAMELEN)
            {
                nlen =
                    CLI_VAR_NAMELEN - 1;
            }
            memcpy(vn, inner,
                   (size_t) nlen);
            vn[nlen] = '\0';
            const char *pat = pp + 1;
            const char *val =
                cli_var_lookup(vn);
            if(val != NULL)
            {
                int vl =
                    (int) strlen(val);
                int pl =
                    (int) strlen(pat);
                /* Find last occurrence */
                const char *last =
                    NULL;
                const char *s = val;
                while((s = strstr(
                           s, pat))
                      != NULL)
                {
                    last = s;
                    s++;
                }
                if(last != NULL
                   && (last + pl)
                   == (val + vl))
                {
                    int clen =
                        (int)(last
                              - val);
                    int avail =
                        maxlen - 1
                        - *opos;
                    if(clen > avail)
                    {
                        clen = avail;
                    }
                    memcpy(
                        out + *opos,
                        val,
                        (size_t) clen);
                    *opos += clen;
                }
                else
                {
                    emit_str(
                        out, opos,
                        maxlen, val);
                }
            }
            return;
        }
    }

    /* ${var^^} uppercase / ${var,,} lowercase
     * ${var^}  first char upper
     * ${var,}  first char lower */
    {
        /* Find ^ or , in inner */
        const char *cp =
            strchr(inner, '^');
        const char *cl =
            strchr(inner, ',');
        /* Pick the earlier one */
        const char *op = NULL;
        if(cp != NULL && cl != NULL)
        {
            op = (cp < cl) ? cp : cl;
        }
        else if(cp != NULL)
        {
            op = cp;
        }
        else if(cl != NULL)
        {
            op = cl;
        }
        if(op != NULL)
        {
            char vn[CLI_VAR_NAMELEN];
            int nlen =
                (int)(op - inner);
            if(nlen >= CLI_VAR_NAMELEN)
            {
                nlen =
                    CLI_VAR_NAMELEN - 1;
            }
            memcpy(vn, inner,
                   (size_t) nlen);
            vn[nlen] = '\0';
            const char *val =
                cli_var_lookup(vn);
            if(val != NULL)
            {
                char tmp[
                    CLI_VAR_VALLEN];
                strncpy(tmp, val,
                        CLI_VAR_VALLEN
                        - 1);
                tmp[CLI_VAR_VALLEN
                    - 1] = '\0';
                if(op[0] == '^'
                   && op[1] == '^')
                {
                    /* Uppercase all */
                    for(int k = 0;
                        tmp[k]
                        != '\0'; k++)
                    {
                        tmp[k] =
                            (char)
                            toupper(
                                (unsigned
                                 char)
                                tmp[k]);
                    }
                }
                else if(op[0] == '^')
                {
                    /* First char */
                    if(tmp[0] != '\0')
                    {
                        tmp[0] =
                            (char)
                            toupper(
                                (unsigned
                                 char)
                                tmp[0]);
                    }
                }
                else if(op[0] == ','
                        && op[1]
                        == ',')
                {
                    /* Lowercase all */
                    for(int k = 0;
                        tmp[k]
                        != '\0'; k++)
                    {
                        tmp[k] =
                            (char)
                            tolower(
                                (unsigned
                                 char)
                                tmp[k]);
                    }
                }
                else if(op[0] == ',')
                {
                    /* First char */
                    if(tmp[0] != '\0')
                    {
                        tmp[0] =
                            (char)
                            tolower(
                                (unsigned
                                 char)
                                tmp[0]);
                    }
                }
                emit_str(out, opos,
                         maxlen,
                         tmp);
            }
            return;
        }
    }

    /* ${var#pat} — strip shortest prefix */
    {
        const char *pp =
            strchr(inner, '#');
        if(pp != NULL)
        {
            char vn[CLI_VAR_NAMELEN];
            int nlen =
                (int)(pp - inner);
            if(nlen >= CLI_VAR_NAMELEN)
            {
                nlen =
                    CLI_VAR_NAMELEN - 1;
            }
            memcpy(vn, inner,
                   (size_t) nlen);
            vn[nlen] = '\0';
            const char *pat = pp + 1;
            const char *val =
                cli_var_lookup(vn);
            if(val != NULL)
            {
                int pl =
                    (int) strlen(pat);
                if(strncmp(val, pat,
                           (size_t) pl)
                   == 0)
                {
                    emit_str(
                        out, opos,
                        maxlen,
                        val + pl);
                }
                else
                {
                    emit_str(
                        out, opos,
                        maxlen, val);
                }
            }
            return;
        }
    }

    /* ${var:-default} ${var:=default}
     * ${var:+alt} ${var:?error}
     * ${var:offset:length} */
    {
        const char *col =
            strchr(inner, ':');
        if(col != NULL)
        {
            char vn[CLI_VAR_NAMELEN];
            int nlen =
                (int)(col - inner);
            if(nlen >= CLI_VAR_NAMELEN)
            {
                nlen =
                    CLI_VAR_NAMELEN - 1;
            }
            memcpy(vn, inner,
                   (size_t) nlen);
            vn[nlen] = '\0';
            char op = col[1];

            /* ${var:-default} */
            if(op == '-')
            {
                const char *val =
                    cli_var_lookup(vn);
                if(val != NULL
                   && val[0] != '\0')
                {
                    emit_str(
                        out, opos,
                        maxlen, val);
                }
                else
                {
                    emit_str(
                        out, opos,
                        maxlen,
                        col + 2);
                }
                return;
            }
            /* ${var:=default} */
            if(op == '=')
            {
                const char *val =
                    cli_var_lookup(vn);
                if(val != NULL
                   && val[0] != '\0')
                {
                    emit_str(
                        out, opos,
                        maxlen, val);
                }
                else
                {
                    cli_var_set(
                        vn, col + 2);
                    emit_str(
                        out, opos,
                        maxlen,
                        col + 2);
                }
                return;
            }
            /* ${var:+alt} */
            if(op == '+')
            {
                const char *val =
                    cli_var_lookup(vn);
                if(val != NULL
                   && val[0] != '\0')
                {
                    emit_str(
                        out, opos,
                        maxlen,
                        col + 2);
                }
                return;
            }
            /* ${var:?error} */
            if(op == '?')
            {
                const char *val =
                    cli_var_lookup(vn);
                if(val == NULL
                   || val[0] == '\0')
                {
                    fprintf(stderr,
                            "%s: %s\n",
                            vn,
                            col + 2);
                }
                else
                {
                    emit_str(
                        out, opos,
                        maxlen, val);
                }
                return;
            }

            /* ${var:offset:length} */
            int offset = (int) strtol(
                col + 1, NULL, 0);
            int slen = -1;
            const char *c2 =
                strchr(col + 1, ':');
            if(c2 != NULL)
            {
                slen = (int) strtol(
                    c2 + 1, NULL, 0);
            }
            const char *val =
                cli_var_lookup(vn);
            if(val != NULL)
            {
                int vl =
                    (int) strlen(val);
                if(offset < 0)
                {
                    offset =
                        vl + offset;
                }
                if(offset < 0)
                {
                    offset = 0;
                }
                if(offset >= vl)
                {
                    return;
                }
                int rem =
                    vl - offset;
                if(slen < 0
                   || slen > rem)
                {
                    slen = rem;
                }
                int avail =
                    maxlen - 1
                    - *opos;
                if(slen > avail)
                {
                    slen = avail;
                }
                memcpy(
                    out + *opos,
                    val + offset,
                    (size_t) slen);
                *opos += slen;
            }
            return;
        }
    }

    /* Plain ${var} */
    const char *val =
        cli_var_lookup(inner);
    if(val != NULL)
    {
        emit_str(out, opos, maxlen,
                 val);
    }
}

/**
 * @brief Expand $VAR and ${VAR} in place
 *
 * Handles string ops, arrays, and special
 * forms inside ${...}.
 */
void cli_expand_env(
    char *line,
    int   maxlen
)
{
    char out[STRINGMAXLEN_CLICMDLINE];
    int  opos = 0;
    int  i = 0;

    while(line[i] != '\0'
            && opos < maxlen - 1)
    {
        if(line[i] == '`')
        {
            i++; /* skip ` */
            char cmdsub[STRINGMAXLEN_CLICMDLINE];
            int  clen = 0;
            while(line[i] != '\0' && line[i] != '`' && clen < STRINGMAXLEN_CLICMDLINE - 1)
            {
                cmdsub[clen++] = line[i++];
            }
            if(line[i] == '`')
            {
                i++;
            }
            cmdsub[clen] = '\0';

            FILE *fp = popen(cmdsub, "r");
            if(fp)
            {
                char   resbuf[4096];
                size_t bytes_read = fread(resbuf, 1, sizeof(resbuf) - 1, fp);
                resbuf[bytes_read] = '\0';

                while(bytes_read > 0 &&
                      (resbuf[bytes_read - 1] == '\n' || resbuf[bytes_read - 1] == '\r'))
                {
                    resbuf[--bytes_read] = '\0';
                }

                for(size_t k = 0; k < bytes_read; k++)
                {
                    if(resbuf[k] == '\n' || resbuf[k] == '\r')
                    {
                        resbuf[k] = ' ';
                    }
                }

                for(size_t k = 0; k < bytes_read && opos < maxlen - 1; k++)
                {
                    out[opos++] = resbuf[k];
                }
                pclose(fp);
            }
            continue;
        }

        if(line[i] == '$')
        {
            /* Skip $(( — arithmetic */
            if(line[i + 1] == '('
               && line[i + 2] == '(')
            {
                out[opos++] = line[i++];
                continue;
            }
            /* Handle $( — command subst */
            if(line[i + 1] == '(')
            {
                i += 2; /* skip $( */
                char cmdsub[STRINGMAXLEN_CLICMDLINE];
                int  clen = 0;
                int  cdepth = 1;
                while(line[i] != '\0' && clen < STRINGMAXLEN_CLICMDLINE - 1)
                {
                    if(line[i] == '(')
                    {
                        cdepth++;
                    }
                    else if(line[i] == ')')
                    {
                        cdepth--;
                        if(cdepth == 0)
                        {
                            i++;
                            break;
                        }
                    }
                    cmdsub[clen++] = line[i++];
                }
                cmdsub[clen] = '\0';

                FILE *fp = popen(cmdsub, "r");
                if(fp)
                {
                    char   resbuf[4096];
                    size_t bytes_read = fread(resbuf, 1, sizeof(resbuf) - 1, fp);
                    resbuf[bytes_read] = '\0';

                    /* Trim trailing newlines */
                    while(bytes_read > 0 &&
                          (resbuf[bytes_read - 1] == '\n' || resbuf[bytes_read - 1] == '\r'))
                    {
                        resbuf[--bytes_read] = '\0';
                    }

                    /* Replace internal newlines with space */
                    for(size_t k = 0; k < bytes_read; k++)
                    {
                        if(resbuf[k] == '\n' || resbuf[k] == '\r')
                        {
                            resbuf[k] = ' ';
                        }
                    }

                    for(size_t k = 0; k < bytes_read && opos < maxlen - 1; k++)
                    {
                        out[opos++] = resbuf[k];
                    }
                    pclose(fp);
                }
                continue;
            }
            i++;
            if(line[i] == '{')
            {
                i++;
                char inner[512];
                int ilen = 0;
                int depth = 1;
                while(line[i] != '\0'
                      && ilen < 511)
                {
                    if(line[i] == '{')
                    {
                        depth++;
                    }
                    if(line[i] == '}')
                    {
                        depth--;
                        if(depth == 0)
                        {
                            i++;
                            break;
                        }
                    }
                    inner[ilen++] = line[i++];
                }
                inner[ilen] = '\0';
                expand_braced(out, &opos,
                              maxlen, inner);
            }
            else
            {
                /* $VAR — simple unbraced */
                char varname[256];
                int  vlen = 0;
                while(line[i] != '\0'
                      && vlen < 255)
                {
                    char c = line[i];
                    if(!((c >= 'A'
                          && c <= 'Z')
                         || (c >= 'a'
                             && c <= 'z')
                         || (c >= '0'
                             && c <= '9')
                         || c == '_'
                         || c == '?'))
                    {
                        break;
                    }
                    varname[vlen++] =
                        line[i++];
                    if(c == '?')
                    {
                        break;
                    }
                }
                varname[vlen] = '\0';
                const char *val =
                    cli_var_lookup(varname);
                if(val != NULL)
                {
                    emit_str(out, &opos,
                             maxlen, val);
                }
            }
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

/*
 * ============================================================
 *  Built-in cd / pwd
 * ============================================================
 */

errno_t cli_cd(void)
{
    const char *dir = getenv("HOME");
    if(data.cmdNBarg >= 2)
    {
        dir = data.cmdargtoken[1].val.string;
    }
    if(dir != NULL)
    {
        if(chdir(dir) != 0)
        {
            printf("cd: %s: %s\n", dir, strerror(errno));
            return RETURN_FAILURE;
        }
    }
    return RETURN_SUCCESS;
}

errno_t cli_pwd(void)
{
    char cwd[1024];
    if(getcwd(cwd, sizeof(cwd)) != NULL)
    {
        printf("%s\n", cwd);
        return RETURN_SUCCESS;
    }
    else
    {
        perror("pwd");
        return RETURN_FAILURE;
    }
}


/*
 * ============================================================
 *  History <N> Command
 * ============================================================
 */

errno_t cli_history_show(void)
{
#ifdef USE_READLINE
    int n = 20;  /* default */
    if(data.cmdNBarg >= 2)
    {
        n = atoi(
            data.cmdargtoken[1].val.string);
        if(n <= 0)
        {
            n = 20;
        }
    }

    HIST_ENTRY **hlist = history_list();
    if(hlist == NULL)
    {
        printf("No history\n");
        return RETURN_SUCCESS;
    }

    int total = history_length;
    int start = total - n;
    if(start < 0)
    {
        start = 0;
    }
    for(int i = start; i < total; i++)
    {
        printf(" %4d  %s\n",
               i + 1, hlist[i]->line);
    }
#else
    printf("Readline not available\n");
#endif
    return RETURN_SUCCESS;
}


/*
 * ============================================================
 *  Fuzzy History Search (searchhist)
 * ============================================================
 *
 * Search history for entries containing a
 * substring. Shows all matches with index.
 */

errno_t cli_searchhist(void)
{
#ifdef USE_READLINE
    if(data.cmdNBarg < 2)
    {
        printf("Usage: searchhist <pattern>\n");
        return RETURN_SUCCESS;
    }
    const char *pattern =
        data.cmdargtoken[1].val.string;

    HIST_ENTRY **hlist = history_list();
    if(hlist == NULL)
    {
        printf("No history\n");
        return RETURN_SUCCESS;
    }

    int total = history_length;
    int found = 0;
    for(int i = 0; i < total; i++)
    {
        if(strcasestr(hlist[i]->line,
                      pattern) != NULL)
        {
            /* Highlight matching substring */
            const char *pos =
                strcasestr(hlist[i]->line,
                           pattern);
            int pre = (int)(pos
                            - hlist[i]->line);
            int plen = (int) strlen(pattern);
            printf(" %4d  %.*s"
                   "\033[1;33m%.*s\033[0m"
                   "%s\n",
                   i + 1,
                   pre, hlist[i]->line,
                   plen, pos,
                   pos + plen);
            found++;
        }
    }
    if(found == 0)
    {
        printf("No history entries match"
               " '%s'\n", pattern);
    }
    else
    {
        printf("(%d match%s)\n",
               found,
               found == 1 ? "" : "es");
    }
#else
    printf("Readline not available\n");
#endif
    return RETURN_SUCCESS;
}


errno_t CLI_execute_line()
{
    DEBUG_TRACE_FSTART();

    char            *cmdargstring;
    int strmaxlen   = 200;
    char             str[strmaxlen];
    FILE            *fp;
    time_t           t;
    struct tm       *uttime;
    struct timespec *thetime =
        (struct timespec *) malloc(sizeof(struct timespec));
    char calctmpimname[STRINGMAXLEN_IMGNAME];

    /* Expand history (!! and !$) first */
    cli_history_expand();
    if(data.CLIcmdline[0] == '\0')
    {
        free(thetime);
        DEBUG_TRACE_FEXIT();
        return RETURN_SUCCESS;
    }

    /* Expand aliases before anything else */
    cli_alias_expand();

    /* Logical operators: && and ||
     * Split line at top-level && / || and
     * execute segments conditionally.
     * Skip && / || inside quotes or $(). */
    {
        const char *src = data.CLIcmdline;
        int depth = 0;   /* () nesting */
        int in_sq = 0;   /* single quote */
        int in_dq = 0;   /* double quote */
        int found = 0;
        int split_pos = -1;
        int op_len = 0;  /* 2 for && or || */
        int op_is_and = 0;

        for(int si = 0; src[si] != '\0'; si++)
        {
            char c = src[si];
            if(c == '\'' && !in_dq)
            {
                in_sq = !in_sq;
            }
            else if(c == '"' && !in_sq)
            {
                in_dq = !in_dq;
            }
            else if(!in_sq && !in_dq)
            {
                if(c == '(')
                {
                    depth++;
                }
                else if(c == ')' && depth > 0)
                {
                    depth--;
                }
                else if(depth == 0
                        && c == '&'
                        && src[si + 1] == '&')
                {
                    found = 1;
                    split_pos = si;
                    op_len = 2;
                    op_is_and = 1;
                    break;
                }
                else if(depth == 0
                        && c == '|'
                        && src[si + 1] == '|')
                {
                    found = 1;
                    split_pos = si;
                    op_len = 2;
                    op_is_and = 0;
                    break;
                }
            }
        }
        if(found && split_pos >= 0)
        {
            /* Execute left side */
            char left[STRINGMAXLEN_CLICMDLINE];
            strncpy(left, data.CLIcmdline,
                    (size_t) split_pos);
            left[split_pos] = '\0';
            strncpy(data.CLIcmdline, left,
                    STRINGMAXLEN_CLICMDLINE
                    - 1);
            data.CLIcmdline[
                STRINGMAXLEN_CLICMDLINE
                - 1] = '\0';
            errno_t lret = CLI_execute_line();
            int ok = (lret == RETURN_SUCCESS);
            /* Decide whether to run right */
            int run_right =
                (op_is_and && ok)
                || (!op_is_and && !ok);
            if(run_right)
            {
                const char *rp =
                    src + split_pos + op_len;
                while(*rp == ' '
                      || *rp == '\t')
                {
                    rp++;
                }
                strncpy(data.CLIcmdline, rp,
                        STRINGMAXLEN_CLICMDLINE
                        - 1);
                data.CLIcmdline[
                    STRINGMAXLEN_CLICMDLINE
                    - 1] = '\0';
                errno_t rret =
                    CLI_execute_line();
                free(thetime);
                DEBUG_TRACE_FEXIT();
                return rret;
            }
            free(thetime);
            DEBUG_TRACE_FEXIT();
            return lret;
        }
    }

    /* Pipe: cmd1 | cmd2
     * Capture stdout of left command into a
     * temp file, then feed it as stdin to
     * the right command. Only matches single
     * '|', not '||'. */
    {
        const char *src = data.CLIcmdline;
        int depth = 0;
        int in_sq = 0;
        int in_dq = 0;
        int pipe_pos = -1;

        for(int si = 0; src[si] != '\0'; si++)
        {
            char c = src[si];
            if(c == '\'' && !in_dq)
            {
                in_sq = !in_sq;
            }
            else if(c == '"' && !in_sq)
            {
                in_dq = !in_dq;
            }
            else if(!in_sq && !in_dq)
            {
                if(c == '(')
                {
                    depth++;
                }
                else if(c == ')'
                        && depth > 0)
                {
                    depth--;
                }
                else if(depth == 0
                        && c == '|'
                        && src[si + 1]
                        != '|')
                {
                    pipe_pos = si;
                    break;
                }
            }
        }
        if(pipe_pos >= 0)
        {
            /* Split at pipe */
            char left[STRINGMAXLEN_CLICMDLINE];
            strncpy(left, data.CLIcmdline,
                    (size_t) pipe_pos);
            left[pipe_pos] = '\0';
            const char *rp =
                data.CLIcmdline
                + pipe_pos + 1;
            while(*rp == ' ' || *rp == '\t')
            {
                rp++;
            }
            char right[STRINGMAXLEN_CLICMDLINE];
            strncpy(right, rp,
                    STRINGMAXLEN_CLICMDLINE
                    - 1);
            right[STRINGMAXLEN_CLICMDLINE
                  - 1] = '\0';

            /* Capture left stdout */
            FILE *tmpfp = tmpfile();
            if(tmpfp != NULL)
            {
                int saved_stdout =
                    dup(STDOUT_FILENO);
                dup2(fileno(tmpfp),
                     STDOUT_FILENO);

                strncpy(data.CLIcmdline,
                        left,
                        STRINGMAXLEN_CLICMDLINE
                        - 1);
                CLI_execute_line();

                fflush(stdout);
                dup2(saved_stdout,
                     STDOUT_FILENO);
                close(saved_stdout);

                /* Feed to right stdin */
                rewind(tmpfp);
                int saved_stdin =
                    dup(STDIN_FILENO);
                dup2(fileno(tmpfp),
                     STDIN_FILENO);

                strncpy(data.CLIcmdline,
                        right,
                        STRINGMAXLEN_CLICMDLINE
                        - 1);
                errno_t pret =
                    CLI_execute_line();

                dup2(saved_stdin,
                     STDIN_FILENO);
                close(saved_stdin);
                fclose(tmpfp);

                free(thetime);
                DEBUG_TRACE_FEXIT();
                return pret;
            }
        }
    }

    /* Dot-sourcing: ". file" → "source file"
     * Must check before script intercept */
    {
        const char *p = data.CLIcmdline;
        while(*p == ' ' || *p == '\t')
        {
            p++;
        }
        if(p[0] == '.' && p[1] == ' ')
        {
            char tmp[STRINGMAXLEN_CLICMDLINE];
            snprintf(tmp,
                     STRINGMAXLEN_CLICMDLINE,
                     "source %s", p + 2);
            strncpy(data.CLIcmdline, tmp,
                    STRINGMAXLEN_CLICMDLINE
                    - 1);
            data.CLIcmdline[
                STRINGMAXLEN_CLICMDLINE
                - 1] = '\0';
        }
    }

    /* Flow control: if/while/for/function
     * and user-defined function calls.
     * Must run BEFORE expansion so block
     * accumulator stores raw lines with
     * $VAR unexpanded. */
    if(cli_script_intercept(data.CLIcmdline))
    {
        data.CMDexecuted = 1;
        free(thetime);
        DEBUG_TRACE_FEXIT();
        return RETURN_SUCCESS;
    }

    /* Expand command substitution */
    cli_expand_cmdsub(data.CLIcmdline, STRINGMAXLEN_CLICMDLINE);

    /* Expand @fpsname.param tokens */
    cli_expand_fpsvar(data.CLIcmdline,
                      STRINGMAXLEN_CLICMDLINE);

    /* Expand tilde (~) to $HOME */
    cli_expand_tilde(data.CLIcmdline,
                     STRINGMAXLEN_CLICMDLINE);

    /* Expand environment variables ($VAR).
     * Runs before arith so $(( $n + 1 )) works.
     * cli_expand_env skips $(( tokens. */
    cli_expand_env(data.CLIcmdline,
                   STRINGMAXLEN_CLICMDLINE);

    /* Expand brace ranges {N..M} {N..M..S} */
    cli_expand_braces(data.CLIcmdline,
                      STRINGMAXLEN_CLICMDLINE);

    /* Expand arithmetic $(( expr )) */
    cli_expand_arith(data.CLIcmdline,
                     STRINGMAXLEN_CLICMDLINE);

    /* Expand filename globs (*.fits etc) */
    cli_expand_globs(data.CLIcmdline,
                     STRINGMAXLEN_CLICMDLINE);

    /* Log command to session log if active */
    cli_session_log_cmd(data.CLIcmdline);

    /* set -x: trace output */
    if(cli_flag_xtrace)
    {
        fprintf(stderr, "+ %s\n",
                data.CLIcmdline);
    }

    /* Output redirection: cmd > file,
     * cmd >> file. Scan from end for
     * unquoted > or >> and redirect. */
    {
        int redir_mode = 0; /* 1=trunc 2=app */
        int redir_pos = -1;
        int in_sq2 = 0;
        int in_dq2 = 0;
        int depth2 = 0;
        const char *cl2 = data.CLIcmdline;
        for(int ri = 0;
            cl2[ri] != '\0'; ri++)
        {
            if(cl2[ri] == '\''
               && !in_dq2)
            {
                in_sq2 = !in_sq2;
            }
            else if(cl2[ri] == '"'
                    && !in_sq2)
            {
                in_dq2 = !in_dq2;
            }
            else if(!in_sq2
                    && !in_dq2)
            {
                if(cl2[ri] == '(')
                {
                    depth2++;
                }
                else if(cl2[ri] == ')'
                        && depth2 > 0)
                {
                    depth2--;
                }
                else if(depth2 == 0
                        && cl2[ri] == '>')
                {
                    if(cl2[ri + 1] == '>')
                    {
                        redir_mode = 2;
                        redir_pos = ri;
                    }
                    else
                    {
                        redir_mode = 1;
                        redir_pos = ri;
                    }
                }
            }
        }
        if(redir_pos >= 0)
        {
            /* Extract filename */
            int fstart = redir_pos
                         + ((redir_mode
                             == 2) ? 2 : 1);
            while(data.CLIcmdline[fstart]
                  == ' '
                  || data.CLIcmdline[fstart]
                  == '\t')
            {
                fstart++;
            }
            char rfile[512];
            int fi = 0;
            while(data.CLIcmdline[fstart]
                  != '\0'
                  && data.CLIcmdline[fstart]
                  != ' '
                  && data.CLIcmdline[fstart]
                  != '\t'
                  && fi < 511)
            {
                rfile[fi++] =
                    data.CLIcmdline[
                        fstart++];
            }
            rfile[fi] = '\0';

            /* Truncate cmd at redir */
            data.CLIcmdline[
                redir_pos] = '\0';
            {
                int cl3 = redir_pos - 1;
                while(cl3 >= 0
                      && (data.CLIcmdline[
                              cl3] == ' '
                          || data.CLIcmdline[
                              cl3]
                          == '\t'))
                {
                    data.CLIcmdline[
                        cl3--] = '\0';
                }
            }

            /* Open file and redirect */
            FILE *rfp = fopen(
                rfile,
                (redir_mode == 2)
                ? "a" : "w");
            if(rfp != NULL)
            {
                int sv_out =
                    dup(STDOUT_FILENO);
                dup2(fileno(rfp),
                     STDOUT_FILENO);

                errno_t rret =
                    CLI_execute_line();

                fflush(stdout);
                dup2(sv_out,
                     STDOUT_FILENO);
                close(sv_out);
                fclose(rfp);

                free(thetime);
                DEBUG_TRACE_FEXIT();
                return rret;
            }
        }
    }

    /* Here-string: cmd <<< "string"
     * Provides string as stdin to cmd */
    {
        char *hs = strstr(
            data.CLIcmdline, "<<<");
        if(hs != NULL)
        {
            /* Split at <<< */
            *hs = '\0';
            const char *hsval =
                hs + 3;
            while(*hsval == ' '
                  || *hsval == '\t')
            {
                hsval++;
            }
            /* Strip quotes */
            int hvlen =
                (int) strlen(hsval);
            char hvbuf[STRINGMAXLEN_CLICMDLINE];
            if(hvlen >= 2
               && ((hsval[0] == '"'
                    && hsval[
                        hvlen - 1]
                    == '"')
                   || (hsval[0] == '\''
                       && hsval[
                           hvlen - 1]
                       == '\'')))
            {
                memcpy(hvbuf,
                       hsval + 1,
                       (size_t)
                       (hvlen - 2));
                hvbuf[hvlen - 2] = '\0';
            }
            else
            {
                strncpy(
                    hvbuf, hsval,
                    STRINGMAXLEN_CLICMDLINE
                    - 1);
                hvbuf[
                    STRINGMAXLEN_CLICMDLINE
                    - 1] = '\0';
            }

            /* Write to tmpfile, feed
             * as stdin */
            FILE *hsfp = tmpfile();
            if(hsfp != NULL)
            {
                fprintf(hsfp, "%s\n",
                        hvbuf);
                rewind(hsfp);
                int sv_in =
                    dup(STDIN_FILENO);
                dup2(fileno(hsfp),
                     STDIN_FILENO);

                errno_t hsret =
                    CLI_execute_line();

                dup2(sv_in,
                     STDIN_FILENO);
                close(sv_in);
                fclose(hsfp);

                free(thetime);
                DEBUG_TRACE_FEXIT();
                return hsret;
            }
        }
    }

    /* Background: cmd & */
    {
        int ll =
            (int) strlen(
                data.CLIcmdline);
        /* Scan backward for & */
        int bi = ll - 1;
        while(bi >= 0
              && (data.CLIcmdline[bi]
                  == ' '
                  || data.CLIcmdline[
                      bi]
                  == '\t'))
        {
            bi--;
        }
        if(bi >= 0
           && data.CLIcmdline[bi]
           == '&'
           && (bi == 0
               || data.CLIcmdline[
                      bi - 1]
               != '&'))
        {
            /* Strip trailing & */
            data.CLIcmdline[bi] =
                '\0';
            pid_t cpid = fork();
            if(cpid == 0)
            {
                /* Child */
                CLI_execute_line(
                    data.CLIcmdline);
                _exit(0);
            }
            else if(cpid > 0)
            {
                printf("[bg] %d\n",
                       (int) cpid);
                char pb[32];
                snprintf(
                    pb, sizeof(pb),
                    "%d",
                    (int) cpid);
                cli_var_set("!", pb);
            }
            free(thetime);
            DEBUG_TRACE_FEXIT();
            return 0;
        }
    }

    /* Subshell: (cmd1; cmd2) */
    {
        const char *sp =
            data.CLIcmdline;
        int spl =
            (int) strlen(sp);
        if(spl >= 3
           && sp[0] == '('
           && sp[spl - 1] == ')')
        {
            char sbuf[
                STRINGMAXLEN_CLICMDLINE
            ];
            memcpy(sbuf, sp + 1,
                   (size_t)(spl - 2));
            sbuf[spl - 2] = '\0';
            pid_t spid = fork();
            if(spid == 0)
            {
                /* Execute in child */
                /* Split on ; */
                char *tok =
                    strtok(sbuf, ";");
                while(tok != NULL)
                {
                    const char *st =
                        tok;
                    while(*st == ' '
                          || *st
                          == '\t')
                    {
                        st++;
                    }
                    if(*st != '\0')
                    {
                        CLI_execute_line(
                            (char *) st
                        );
                    }
                    tok =
                        strtok(
                            NULL,
                            ";");
                }
                _exit(0);
            }
            else if(spid > 0)
            {
                int wst;
                waitpid(spid,
                        &wst, 0);
                cli_last_retval =
                    WEXITSTATUS(wst);
            }
            free(thetime);
            DEBUG_TRACE_FEXIT();
            return 0;
        }
    }

    /* Here-string: cmd <<< "text" */
    {
        const char *hs =
            strstr(data.CLIcmdline,
                   "<<<");
        if(hs != NULL)
        {
            /* Extract command before
             * <<< */
            char hcmd[
                STRINGMAXLEN_CLICMDLINE
            ];
            int hcl =
                (int)(hs
                      - data.CLIcmdline);
            if(hcl
               >= STRINGMAXLEN_CLICMDLINE)
            {
                hcl =
                    STRINGMAXLEN_CLICMDLINE
                    - 1;
            }
            memcpy(hcmd,
                   data.CLIcmdline,
                   (size_t) hcl);
            hcmd[hcl] = '\0';
            /* Trim trailing ws */
            while(hcl > 0
                  && (hcmd[hcl - 1]
                      == ' '
                      || hcmd[hcl - 1]
                      == '\t'))
            {
                hcmd[--hcl] = '\0';
            }
            /* Get the text */
            const char *tp = hs + 3;
            while(*tp == ' '
                  || *tp == '\t')
            {
                tp++;
            }
            /* Strip quotes */
            char htxt[1024];
            strncpy(htxt, tp,
                    sizeof(htxt) - 1);
            htxt[sizeof(htxt) - 1] =
                '\0';
            int htl =
                (int) strlen(htxt);
            if(htl >= 2
               && ((htxt[0] == '"'
                    && htxt[htl - 1]
                    == '"')
                   || (htxt[0] == '\''
                       && htxt[
                           htl - 1]
                       == '\'')))
            {
                htxt[htl - 1] = '\0';
                memmove(htxt,
                        htxt + 1,
                        (size_t)
                        (htl - 1));
            }
            /* Create pipe */
            int pfd[2];
            if(pipe(pfd) == 0)
            {
                /* Write text to pipe */
                ssize_t wr_ignore;
                wr_ignore =
                write(pfd[1], htxt,
                      strlen(htxt));
                wr_ignore =
                write(pfd[1], "\n", 1);
                (void) wr_ignore;
                close(pfd[1]);
                /* Redirect stdin */
                int sv =
                    dup(STDIN_FILENO);
                dup2(pfd[0],
                     STDIN_FILENO);
                close(pfd[0]);
                CLI_execute_line(hcmd);
                dup2(sv,
                     STDIN_FILENO);
                close(sv);
            }
            free(thetime);
            DEBUG_TRACE_FEXIT();
            return 0;
        }
    }

    /* Stderr redirect: 2>&1, 2>/dev/null,
     * 2>file */
    {
        const char *se =
            strstr(data.CLIcmdline,
                   "2>");
        if(se != NULL)
        {
            /* Extract cmd before 2> */
            char scmd[
                STRINGMAXLEN_CLICMDLINE
            ];
            int scl =
                (int)(se
                      - data.CLIcmdline);
            if(scl
               >= STRINGMAXLEN_CLICMDLINE)
            {
                scl =
                    STRINGMAXLEN_CLICMDLINE
                    - 1;
            }
            memcpy(scmd,
                   data.CLIcmdline,
                   (size_t) scl);
            scmd[scl] = '\0';
            while(scl > 0
                  && (scmd[scl - 1]
                      == ' '
                      || scmd[scl - 1]
                      == '\t'))
            {
                scmd[--scl] = '\0';
            }
            const char *target =
                se + 2;
            while(*target == ' '
                  || *target == '\t')
            {
                target++;
            }
            int sv_err =
                dup(STDERR_FILENO);
            if(strncmp(target, "&1",
                       2) == 0)
            {
                dup2(STDOUT_FILENO,
                     STDERR_FILENO);
            }
            else
            {
                /* Strip trailing ws */
                char fname[256];
                int fi = 0;
                while(target[fi]
                      != '\0'
                      && target[fi]
                      != ' '
                      && target[fi]
                      != '\t'
                      && fi < 254)
                {
                    fname[fi] =
                        target[fi];
                    fi++;
                }
                fname[fi] = '\0';
                FILE *ef =
                    fopen(fname, "w");
                if(ef != NULL)
                {
                    dup2(fileno(ef),
                         STDERR_FILENO);
                    fclose(ef);
                }
            }
            CLI_execute_line(scmd);
            dup2(sv_err,
                 STDERR_FILENO);
            close(sv_err);
            free(thetime);
            DEBUG_TRACE_FEXIT();
            return 0;
        }
    }

    /* Input redirection: cmd < file
     * Scan for unquoted < that is not
     * << or <<<. */
    {
        const char *cl4 =
            data.CLIcmdline;
        int in_sq4 = 0, in_dq4 = 0;
        int depth4 = 0;
        int inr_pos = -1;
        for(int ri = 0;
            cl4[ri] != '\0'; ri++)
        {
            if(cl4[ri] == '\''
               && !in_dq4)
            {
                in_sq4 = !in_sq4;
            }
            else if(cl4[ri] == '"'
                    && !in_sq4)
            {
                in_dq4 = !in_dq4;
            }
            else if(!in_sq4
                    && !in_dq4)
            {
                if(cl4[ri] == '(')
                {
                    depth4++;
                }
                else if(cl4[ri] == ')'
                        && depth4 > 0)
                {
                    depth4--;
                }
                else if(depth4 == 0
                        && cl4[ri] == '<'
                        && cl4[ri + 1]
                        != '<')
                {
                    inr_pos = ri;
                    break;
                }
            }
        }
        if(inr_pos >= 0)
        {
            int fst = inr_pos + 1;
            while(data.CLIcmdline[fst]
                  == ' '
                  || data.CLIcmdline[fst]
                  == '\t')
            {
                fst++;
            }
            char infile[512];
            int ifi = 0;
            while(data.CLIcmdline[fst]
                  != '\0'
                  && data.CLIcmdline[fst]
                  != ' '
                  && data.CLIcmdline[fst]
                  != '\t'
                  && ifi < 511)
            {
                infile[ifi++] =
                    data.CLIcmdline[
                        fst++];
            }
            infile[ifi] = '\0';

            /* Truncate cmd at < */
            data.CLIcmdline[
                inr_pos] = '\0';
            {
                int cl5 = inr_pos - 1;
                while(cl5 >= 0
                      && (data.CLIcmdline[
                              cl5] == ' '
                          || data.CLIcmdline[
                              cl5]
                          == '\t'))
                {
                    data.CLIcmdline[
                        cl5--] = '\0';
                }
            }

            FILE *ifp =
                fopen(infile, "r");
            if(ifp != NULL)
            {
                int sv_in =
                    dup(STDIN_FILENO);
                dup2(fileno(ifp),
                     STDIN_FILENO);

                errno_t iret =
                    CLI_execute_line();

                dup2(sv_in,
                     STDIN_FILENO);
                close(sv_in);
                fclose(ifp);

                free(thetime);
                DEBUG_TRACE_FEXIT();
                return iret;
            }
        }
    }

    /* Check for array assignment: arr=(a b c) */
    if(cli_try_array_assign(data.CLIcmdline))
    {
        data.CMDexecuted = 1;
        free(thetime);
        DEBUG_TRACE_FEXIT();
        return RETURN_SUCCESS;
    }

    /* Check for variable assignment (VAR=val) */
    if(cli_try_var_assign(data.CLIcmdline))
    {
        data.CMDexecuted = 1;
        free(thetime);
        DEBUG_TRACE_FEXIT();
        return RETURN_SUCCESS;
    }

    /*
     * ---- Command chaining: ; && || ----
     *
     * Scan for the first unquoted chaining
     * operator and split there.
     */
    {
        char fullline[STRINGMAXLEN_CLICMDLINE];
        strncpy(fullline, data.CLIcmdline,
                STRINGMAXLEN_CLICMDLINE - 1);
        fullline[
            STRINGMAXLEN_CLICMDLINE - 1]
            = '\0';

        /* Find first chaining operator */
        int  chain_type = 0;
        /* 1=;  2=&&  3=||  */
        int  chain_off = -1;
        int  chain_len = 0;
        for(int ci = 0; fullline[ci] != '\0';
                ci++)
        {
            /* Skip quoted strings */
            if(fullline[ci] == '"')
            {
                ci++;
                while(fullline[ci] != '\0'
                        && fullline[ci] != '"')
                {
                    ci++;
                }
                if(fullline[ci] == '\0')
                {
                    break;
                }
                continue;
            }
            if(fullline[ci] == ';')
            {
                chain_type = 1;
                chain_off = ci;
                chain_len = 1;
                break;
            }
            if(fullline[ci] == '&'
                    && fullline[ci + 1] == '&')
            {
                chain_type = 2;
                chain_off = ci;
                chain_len = 2;
                break;
            }
            if(fullline[ci] == '|'
                    && fullline[ci + 1] == '|')
            {
                chain_type = 3;
                chain_off = ci;
                chain_len = 2;
                break;
            }
        }

        if(chain_off >= 0)
        {
            /* Extract first part */
            fullline[chain_off] = '\0';
            strncpy(data.CLIcmdline, fullline,
                    STRINGMAXLEN_CLICMDLINE - 1);
            data.CLIcmdline[
                STRINGMAXLEN_CLICMDLINE - 1]
                = '\0';

            /* Execute first part */
            errno_t ret1 = CLI_execute_line();

            /* Determine whether to run rest */
            int run_rest = 0;
            if(chain_type == 1)
            {
                run_rest = 1; /* ; always */
            }
            else if(chain_type == 2)
            {
                /* && only on success */
                run_rest =
                    (ret1 == RETURN_SUCCESS)
                    ? 1 : 0;
            }
            else if(chain_type == 3)
            {
                /* || only on failure */
                run_rest =
                    (ret1 != RETURN_SUCCESS)
                    ? 1 : 0;
            }

            if(run_rest)
            {
                const char *rest =
                    fullline + chain_off
                    + chain_len;
                while(*rest == ' '
                        || *rest == '\t')
                {
                    rest++;
                }
                if(*rest != '\0')
                {
                    strncpy(data.CLIcmdline,
                            rest,
                            STRINGMAXLEN_CLICMDLINE
                            - 1);
                    data.CLIcmdline[
                        STRINGMAXLEN_CLICMDLINE
                        - 1] = '\0';
                    CLI_execute_line();
                }
            }
            free(thetime);
            DEBUG_TRACE_FEXIT();
            return RETURN_SUCCESS;
        }
    }

    /* ---- Pipe to shell ---- */
    FILE *pipe_fp = NULL;
    int   saved_stdout_fd = -1;
    {
        char *pipe_pos =
            strchr(data.CLIcmdline, '|');
        if(pipe_pos != NULL)
        {
            *pipe_pos = '\0';
            const char *rhs = pipe_pos + 1;
            while(*rhs == ' ' || *rhs == '\t')
            {
                rhs++;
            }
            if(*rhs != '\0')
            {
                pipe_fp = popen(rhs, "w");
                if(pipe_fp != NULL)
                {
                    saved_stdout_fd =
                        dup(STDOUT_FILENO);
                    dup2(fileno(pipe_fp),
                         STDOUT_FILENO);
                }
            }
        }
    }

    /* ---- Output redirect to file ---- */
    FILE *redir_fp = NULL;
    int   saved_stdout_redir = -1;
    if(pipe_fp == NULL)
    {
        char *redir_pos =
            strchr(data.CLIcmdline, '>');
        if(redir_pos != NULL)
        {
            *redir_pos = '\0';
            const char *fname = redir_pos + 1;
            while(*fname == ' '
                    || *fname == '\t')
            {
                fname++;
            }
            if(*fname != '\0')
            {
                char fpath[500];
                strncpy(fpath, fname, 499);
                fpath[499] = '\0';
                {
                    size_t fl = strlen(fpath);
                    while(fl > 0
                            && (fpath[fl - 1]
                                == ' '
                                || fpath[fl - 1]
                                == '\t'
                                || fpath[fl - 1]
                                == '\n'))
                    {
                        fpath[--fl] = '\0';
                    }
                }
                redir_fp = fopen(fpath, "w");
                if(redir_fp != NULL)
                {
                    saved_stdout_redir =
                        dup(STDOUT_FILENO);
                    dup2(fileno(redir_fp),
                         STDOUT_FILENO);
                }
            }
        }
    }

#ifdef USE_READLINE
    add_history(data.CLIcmdline);
    if(data.autocomplete_history)
    {
        append_history(1, CLI_history_file());
        history_truncate_file(CLI_history_file(), 10000);
    }
#endif

    //
    // If line starts with !, use system()
    //
    if(data.CLIcmdline[0] == '!')
    {
        data.CLIcmdline[0] = ' ';
        if(system(data.CLIcmdline) != 0)
        {
            PRINT_ERROR("system call error");
            exit(4);
        }
        data.CMDexecuted = 1;
    }
    else if(data.CLIcmdline[0] == '#')
    {
        // do nothing... this is a comment
        data.CMDexecuted = 1;
    }
    else if(strncmp(data.CLIcmdline,
                    "echo ", 5) == 0
            || strcmp(data.CLIcmdline,
                      "echo") == 0)
    {
        /* Handle echo before tokenization
         * to avoid image name resolution */
        const char *args =
            data.CLIcmdline + 4;
        while(*args == ' ')
        {
            args++;
        }
        int nl = 1;
        if(strncmp(args, "-n ", 3) == 0)
        {
            nl = 0;
            args += 3;
            while(*args == ' ')
            {
                args++;
            }
        }
        printf("%s", args);
        if(nl)
        {
            printf("\n");
        }
        data.CMDexecuted = 1;
    }
    else if(strncmp(data.CLIcmdline,
                    "source ", 7) == 0)
    {
        /* Handle before tokenization so
         * file paths with dots are not
         * misinterpreted by the parser */
        const char *arg =
            data.CLIcmdline + 7;
        while(*arg == ' ' || *arg == '\t')
        {
            arg++;
        }
        if(*arg == '\0')
        {
            printf("Usage: source "
                   "<filename>\n");
        }
        else
        {
            data.cmdNBarg = 2;
            strncpy(
                data.cmdargtoken[1]
                .val.string,
                arg,
                sizeof(
                    data.cmdargtoken[1]
                    .val.string) - 1);
            cli_source();
        }
        data.CMDexecuted = 1;
    }
    else if(strncmp(data.CLIcmdline,
                    "include_once ", 13) == 0)
    {
        /* include_once <file> — source only
         * if not already sourced. Uses a
         * static table of resolved paths. */
        static char sourced[128][PATH_MAX];
        static int nsourced = 0;

        const char *arg =
            data.CLIcmdline + 13;
        while(*arg == ' ' || *arg == '\t')
        {
            arg++;
        }
        if(*arg == '\0')
        {
            printf("Usage: include_once "
                   "<filename>\n");
        }
        else
        {
            char rp[PATH_MAX];
            char *resolved =
                realpath(arg, rp);
            if(resolved == NULL)
            {
                printf("include_once: "
                       "%s: %s\n",
                       arg,
                       strerror(errno));
            }
            else
            {
                int found = 0;
                for(int k = 0;
                    k < nsourced; k++)
                {
                    if(strcmp(sourced[k],
                             rp) == 0)
                    {
                        found = 1;
                        break;
                    }
                }
                if(!found)
                {
                    if(nsourced < 128)
                    {
                        strncpy(
                            sourced[nsourced],
                            rp,
                            PATH_MAX - 1);
                        nsourced++;
                    }
                    data.cmdNBarg = 2;
                    strncpy(
                        data.cmdargtoken[1]
                        .val.string,
                        arg,
                        sizeof(
                            data.cmdargtoken[1]
                            .val.string) - 1);
                    cli_source();
                }
            }
        }
        data.CMDexecuted = 1;
    }
    else if(strncmp(data.CLIcmdline,
                    "savescript ", 11) == 0)
    {
        /* Handle before tokenization so
         * file paths with dots etc. are
         * not misinterpreted */
        const char *arg =
            data.CLIcmdline + 11;
        while(*arg == ' ' || *arg == '\t')
        {
            arg++;
        }
        if(*arg == '\0')
        {
            printf("Usage: savescript "
                   "<filename>\n");
        }
        else
        {
            /* Temporarily set cmdNBarg and
             * token for cli_savescript() */
            data.cmdNBarg = 2;
            strncpy(
                data.cmdargtoken[1]
                .val.string,
                arg,
                sizeof(
                    data.cmdargtoken[1]
                    .val.string) - 1);
            cli_savescript();
        }
        data.CMDexecuted = 1;
    }
    else if(strncmp(data.CLIcmdline,
                    "savehistory ", 12) == 0)
    {
        const char *arg =
            data.CLIcmdline + 12;
        while(*arg == ' ' || *arg == '\t')
        {
            arg++;
        }
        if(*arg == '\0')
        {
            printf("Usage: savehistory "
                   "<filename>\n");
        }
        else
        {
            data.cmdNBarg = 2;
            strncpy(
                data.cmdargtoken[1]
                .val.string,
                arg,
                sizeof(
                    data.cmdargtoken[1]
                    .val.string) - 1);
            cli_savehistory();
        }
        data.CMDexecuted = 1;
    }
    else if(strncmp(data.CLIcmdline,
                    "on_update ", 10) == 0)
    {
        /* on_update <stream> { cmd }
         * Wait for stream semaphore,
         * then execute cmd. */
        const char *arg =
            data.CLIcmdline + 10;
        while(*arg == ' ' || *arg == '\t')
        {
            arg++;
        }
        /* Parse stream name */
        char sname[200];
        {
            int si = 0;
            while(*arg != '\0'
                  && *arg != ' '
                  && *arg != '\t'
                  && si < 199)
            {
                sname[si++] = *arg++;
            }
            sname[si] = '\0';
        }
        while(*arg == ' ' || *arg == '\t')
        {
            arg++;
        }
        /* Skip optional { and } */
        if(*arg == '{')
        {
            arg++;
        }
        while(*arg == ' ' || *arg == '\t')
        {
            arg++;
        }
        /* Find end, strip } */
        char body[STRINGMAXLEN_CLICMDLINE];
        strncpy(body, arg,
                STRINGMAXLEN_CLICMDLINE - 1);
        body[
            STRINGMAXLEN_CLICMDLINE - 1]
            = '\0';
        {
            int blen = (int) strlen(body);
            while(blen > 0
                  && (body[blen - 1] == '}'
                      || body[blen - 1] == ' '
                      || body[blen - 1]
                      == '\t'))
            {
                blen--;
            }
            body[blen] = '\0';
        }
        if(sname[0] == '\0'
           || body[0] == '\0')
        {
            printf("Usage: on_update "
                   "<stream> "
                   "{ command }\n");
        }
        else
        {
            /* Connect to stream and
             * wait for semaphore */
            IMAGE img;
            if(ImageStreamIO_read_sharedmem_image_toIMAGE(
                   sname, &img)
               == IMAGESTREAMIO_SUCCESS)
            {
                int semidx =
                    ImageStreamIO_getsemwaitindex(
                        &img, 0);
                if(semidx >= 0)
                {
                    ImageStreamIO_semwait(
                        &img, semidx);
                    /* Execute body */
                    strncpy(
                        data.CLIcmdline,
                        body,
                        STRINGMAXLEN_CLICMDLINE
                        - 1);
                    data.CLIcmdline[
                        STRINGMAXLEN_CLICMDLINE
                        - 1] = '\0';
                    CLI_execute_line();
                }
            }
            else
            {
                printf("on_update: "
                       "stream %s not "
                       "found\n", sname);
            }
        }
        data.CMDexecuted = 1;
    }
    else if(strncmp(data.CLIcmdline,
                    "sleep ", 6) == 0
            || strcmp(data.CLIcmdline,
                     "sleep") == 0)
    {
        /* sleep <seconds> — float-capable
         * delay. Handle before tokenization
         * because the parser would try to
         * interpret decimals. */
        const char *arg =
            data.CLIcmdline + 5;
        while(*arg == ' ' || *arg == '\t')
        {
            arg++;
        }
        if(*arg == '\0')
        {
            printf("Usage: sleep "
                   "<seconds>\n");
        }
        else
        {
            double secs = strtod(arg, NULL);
            if(secs > 0.0)
            {
                usleep(
                    (useconds_t)
                    (secs * 1e6));
            }
        }
        data.CMDexecuted = 1;
    }
    else if(strncmp(data.CLIcmdline,
                    "printf ", 7) == 0)
    {
        /* printf "fmt" arg1 arg2 ...
         * Supports %d %f %s %% \n \t */
        const char *p =
            data.CLIcmdline + 7;
        while(*p == ' ' || *p == '\t')
        {
            p++;
        }
        /* Extract format string */
        char fmt[512];
        int fi = 0;
        char delim = '"';
        if(*p == '"' || *p == '\'')
        {
            delim = *p++;
            while(*p != '\0'
                  && *p != delim
                  && fi < 511)
            {
                fmt[fi++] = *p++;
            }
            if(*p == delim)
            {
                p++;
            }
        }
        else
        {
            while(*p != '\0'
                  && *p != ' '
                  && *p != '\t'
                  && fi < 511)
            {
                fmt[fi++] = *p++;
            }
        }
        fmt[fi] = '\0';
        /* Collect remaining args */
        char *args[16];
        int nargs = 0;
        while(*p != '\0' && nargs < 16)
        {
            while(*p == ' ' || *p == '\t')
            {
                p++;
            }
            if(*p == '\0')
            {
                break;
            }
            char abuf[256];
            int ai = 0;
            while(*p != '\0'
                  && *p != ' '
                  && *p != '\t'
                  && ai < 255)
            {
                abuf[ai++] = *p++;
            }
            abuf[ai] = '\0';
            args[nargs] =
                strdup(abuf);
            nargs++;
        }
        /* Print with format */
        {
            int ai = 0;
            for(int k = 0; fmt[k] != '\0';
                k++)
            {
                if(fmt[k] == '\\'
                   && fmt[k + 1] != '\0')
                {
                    k++;
                    if(fmt[k] == 'n')
                    {
                        putchar('\n');
                    }
                    else if(fmt[k] == 't')
                    {
                        putchar('\t');
                    }
                    else if(fmt[k] == '\\')
                    {
                        putchar('\\');
                    }
                    else
                    {
                        putchar('\\');
                        putchar(fmt[k]);
                    }
                }
                else if(fmt[k] == '%'
                        && fmt[k + 1]
                        != '\0')
                {
                    k++;
                    if(fmt[k] == '%')
                    {
                        putchar('%');
                    }
                    else if(fmt[k] == 'd'
                            && ai < nargs)
                    {
                        printf("%ld",
                               strtol(
                                   args[ai++],
                                   NULL, 0));
                    }
                    else if(fmt[k] == 'f'
                            && ai < nargs)
                    {
                        printf("%f",
                               strtod(
                                   args[ai++],
                                   NULL));
                    }
                    else if(fmt[k] == 's'
                            && ai < nargs)
                    {
                        printf("%s",
                               args[ai++]);
                    }
                    else if(fmt[k] == '.'
                            && ai < nargs)
                    {
                        /* Handle %.Nf */
                        char pfmt[16];
                        int pfi = 0;
                        pfmt[pfi++] = '%';
                        pfmt[pfi++] = '.';
                        k++;
                        while(fmt[k] >= '0'
                              && fmt[k] <= '9'
                              && pfi < 14)
                        {
                            pfmt[pfi++] =
                                fmt[k++];
                        }
                        pfmt[pfi++] =
                            fmt[k]; /* f */
                        pfmt[pfi] = '\0';
                        printf(pfmt,
                               strtod(
                                   args[ai++],
                                   NULL));
                    }
                    else
                    {
                        putchar('%');
                        putchar(fmt[k]);
                    }
                }
                else
                {
                    putchar(fmt[k]);
                }
            }
        }
        fflush(stdout);
        for(int k = 0; k < nargs; k++)
        {
            free(args[k]);
        }
        data.CMDexecuted = 1;
    }
    else if(strncmp(data.CLIcmdline,
                    "read ", 5) == 0
            || strcmp(data.CLIcmdline,
                     "read") == 0)
    {
        /* read [-p "prompt"] [-t N]
         * [-a arr] varname
         * Read line from stdin */
        const char *p =
            data.CLIcmdline + 4;
        while(*p == ' ' || *p == '\t')
        {
            p++;
        }
        /* Parse flags */
        int rd_timeout = -1;
        int rd_array = 0;
        char rd_prompt[256] = {'\0'};
        char rd_aname[CLI_VAR_NAMELEN]
            = {'\0'};
        while(p[0] == '-')
        {
            if(strncmp(p, "-p ", 3)
               == 0)
            {
                p += 3;
                while(*p == ' '
                      || *p == '\t')
                {
                    p++;
                }
                if(*p == '"'
                   || *p == '\'')
                {
                    char delim = *p++;
                    int pi = 0;
                    while(*p != '\0'
                          && *p
                          != delim
                          && pi < 254)
                    {
                        rd_prompt[pi++]
                            = *p++;
                    }
                    rd_prompt[pi] =
                        '\0';
                    if(*p == delim)
                    {
                        p++;
                    }
                }
                else
                {
                    int pi = 0;
                    while(*p != '\0'
                          && *p != ' '
                          && *p
                          != '\t'
                          && pi < 254)
                    {
                        rd_prompt[pi++]
                            = *p++;
                    }
                    rd_prompt[pi] =
                        '\0';
                }
            }
            else if(strncmp(
                        p, "-t ", 3)
                    == 0)
            {
                p += 3;
                while(*p == ' '
                      || *p == '\t')
                {
                    p++;
                }
                rd_timeout = (int)
                    strtol(p, NULL,
                           10);
                while(*p != '\0'
                      && *p != ' '
                      && *p != '\t')
                {
                    p++;
                }
            }
            else if(strncmp(
                        p, "-a ", 3)
                    == 0)
            {
                p += 3;
                while(*p == ' '
                      || *p == '\t')
                {
                    p++;
                }
                rd_array = 1;
                {
                    int ni = 0;
                    while(*p != '\0'
                          && *p != ' '
                          && *p
                          != '\t'
                          && ni
                          < CLI_VAR_NAMELEN
                          - 1)
                    {
                        rd_aname[ni++]
                            = *p++;
                    }
                    rd_aname[ni] =
                        '\0';
                }
            }
            else
            {
                /* Unknown flag */
                p++;
                while(*p != '\0'
                      && *p != ' '
                      && *p != '\t')
                {
                    p++;
                }
            }
            while(*p == ' '
                  || *p == '\t')
            {
                p++;
            }
        }
        /* Print prompt */
        if(rd_prompt[0] != '\0')
        {
            printf("%s", rd_prompt);
            fflush(stdout);
        }
        /* Timeout with select() */
        int rd_ok = 1;
        if(rd_timeout >= 0)
        {
            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(STDIN_FILENO,
                   &fds);
            struct timeval tv;
            tv.tv_sec = rd_timeout;
            tv.tv_usec = 0;
            int sr = select(
                STDIN_FILENO + 1,
                &fds, NULL, NULL,
                &tv);
            if(sr <= 0)
            {
                rd_ok = 0;
                cli_last_retval = 1;
            }
        }
        if(rd_ok)
        {
            char rbuf[1024];
            if(fgets(rbuf,
                     sizeof(rbuf),
                     stdin)
               != NULL)
            {
                /* Strip trailing
                 * newline */
                size_t rlen =
                    strlen(rbuf);
                while(rlen > 0
                      && (rbuf[
                              rlen - 1]
                          == '\n'
                          || rbuf[
                              rlen - 1]
                          == '\r'))
                {
                    rbuf[--rlen] =
                        '\0';
                }
                if(rd_array)
                {
                    /* Split into array
                     * elements */
                    for(int k = 0;
                        k
                        < CLI_MAX_ARRAYS;
                        k++)
                    {
                        if(!cli_arrays[
                            k].used)
                        {
                            cli_arrays[
                                k]
                                .used = 1;
                            strncpy(
                                cli_arrays[
                                    k]
                                .name,
                                rd_aname,
                                CLI_VAR_NAMELEN
                                - 1);
                            cli_arrays[
                                k]
                                .nelem
                                = 0;
                            char *tok
                                = strtok(
                                    rbuf,
                                    " \t");
                            while(tok
                                  != NULL
                                  && cli_arrays[
                                      k]
                                  .nelem
                                  < CLI_ARRAY_MAXELEM)
                            {
                                strncpy(
                                    cli_arrays[
                                        k]
                                    .elem[
                                        cli_arrays[
                                            k]
                                        .nelem],
                                    tok,
                                    CLI_VAR_VALLEN
                                    - 1);
                                cli_arrays[
                                    k]
                                    .nelem++;
                                tok
                                    = strtok(
                                        NULL,
                                        " \t");
                            }
                            break;
                        }
                    }
                }
                else if(*p != '\0')
                {
                    /* Scalar var */
                    char vname[
                        CLI_VAR_NAMELEN
                    ];
                    int vi = 0;
                    while(*p != '\0'
                          && *p != ' '
                          && *p
                          != '\t'
                          && vi
                          < CLI_VAR_NAMELEN
                          - 1)
                    {
                        vname[vi++] =
                            *p++;
                    }
                    vname[vi] = '\0';
                    cli_var_set(
                        vname, rbuf);
                }
                cli_last_retval = 0;
            }
            else
            {
                cli_last_retval = 1;
            }
        }
        data.CMDexecuted = 1;
    }
    else
    {
        // some initialization
        data.parseerror      = 0;
        data.calctmp_imindex = 0;
        for(int i = 0; i < NB_ARG_MAX; i++)
        {
            data.cmdargtoken[i].type          = CMDARGTOKEN_TYPE_UNSOLVED;
            data.cmdargtoken[i].val.string[0] = '\0';
        }

        // log command if CLIlogON active
        if(data.CLIlogON == 1)
        {
            t      = time(NULL);
            uttime = gmtime(&t);
            clock_gettime(CLOCK_MILK, thetime);

            snprintf(data.CLIlogname,
                     STRINGMAXLEN_FULLFILENAME,
                     "%s/logdir/%04d%02d%02d/%04d%02d%02d_CLI-%s.log",
                     getenv("HOME"),
                     1900 + uttime->tm_year,
                     1 + uttime->tm_mon,
                     uttime->tm_mday,
                     1900 + uttime->tm_year,
                     1 + uttime->tm_mon,
                     uttime->tm_mday,
                     data.processname);

            fp = fopen(data.CLIlogname, "a");
            if(fp == NULL)
            {
                printf("ERROR: cannot log into file %s\n", data.CLIlogname);
                EXECUTE_SYSTEM_COMMAND("mkdir -p %s/logdir/%04d%02d%02d\n",
                                       getenv("HOME"),
                                       1900 + uttime->tm_year,
                                       1 + uttime->tm_mon,
                                       uttime->tm_mday);
            }
            else
            {
                fprintf(fp,
                        "%04d/%02d/%02d %02d:%02d:%02d.%09ld %10s "
                        "%6ld %s\n",
                        1900 + uttime->tm_year,
                        1 + uttime->tm_mon,
                        uttime->tm_mday,
                        uttime->tm_hour,
                        uttime->tm_min,
                        uttime->tm_sec,
                        thetime->tv_nsec,
                        data.processname,
                        (long) getpid(),
                        data.CLIcmdline);
                fclose(fp);
            }
        }

        //
        data.cmdNBarg = 0;


        if(dcdebug > 0)
        {
        }

        // extract first word

        // First, split double-quote strings out
        // strings inside double quotes are not processed, and will be given type CMDARGTOKEN_TYPE_RAWSTRING
        int  rawstringmode = 0;
        char str1[500];
        strcpy(str1, data.CLIcmdline);

        char *tokengroup;
        char *rest = str1;
        if(str1[0] == '\"')
        {
            rawstringmode = 1;
        }

        while((tokengroup = strtok_r(rest, "\"", &rest)))
        {
            //printf(" TOKEN [%d]:  %s\n", rawstringmode, tokengroup);

            // always copy word in string, so that arg can be processed as string if needed
            //strcpy(data.cmdargtoken[data.cmdNBarg].val.string, cmdargstring);

            if(rawstringmode == 0)  // not in a raw string, process tokengroup
            {
                cmdargstring = strtok(tokengroup, " ");
                while(cmdargstring != NULL)  // iterate on words
                {
                    // printf("\t processing -- %s\n", cmdargstring);

                    snprintf(str, strmaxlen, "%s\n", cmdargstring);
                    cli_parse(str);

                    cmdargstring = strtok(NULL, " ");
                    data.cmdNBarg++;
                }
                rawstringmode = 1;
            }
            else
            {
                strcpy(data.cmdargtoken[data.cmdNBarg].val.string, tokengroup);
                data.cmdargtoken[data.cmdNBarg].type =
                    CMDARGTOKEN_TYPE_RAWSTRING;
                data.cmdNBarg++;
                rawstringmode = 0;
            }
        }
        data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_UNSOLVED;


        if(dcdebug > 0)
        {
            printf("DEBUG: %s %d: data.cmdNBarg = %ld\n", __func__, __LINE__,
                   data.cmdNBarg);
        }

        if(dcdebug > 1)
        {
            long i = 0;

            if(dcdebug > 0)
            {
                printf("DEBUG: %s %d: TOKEN %ld type : %d\n",
                       __func__, __LINE__,
                       i,
                       data.cmdargtoken[i].type);
            }

            while(data.cmdargtoken[i].type != 0)
            {

                printf("DEBUG: %s %d: TOKEN %ld/%ld   \"%s\"  type : %d\n",
                       __func__, __LINE__,
                       i,
                       data.cmdNBarg,
                       data.cmdargtoken[i].val.string,
                       data.cmdargtoken[i].type);
                if(data.cmdargtoken[i].type ==
                        CMDARGTOKEN_TYPE_FLOAT) // double
                {
                    printf(
                        "\t CMDARGTOKEN_TYPE_FLOAT           : "
                        "%g\n",
                        data.cmdargtoken[i].val.numf);
                }
                if(data.cmdargtoken[i].type == CMDARGTOKEN_TYPE_LONG)  // long
                {
                    printf(
                        "\t CMDARGTOKEN_TYPE_LONG           : "
                        "%ld\n",
                        data.cmdargtoken[i].val.numl);
                }
                if(data.cmdargtoken[i].type ==
                        CMDARGTOKEN_TYPE_STRING) // new variable/image
                {
                    printf(
                        "\t CMDARGTOKEN_TYPE_STRING        : "
                        "%s\n",
                        data.cmdargtoken[i].val.string);
                }
                if(data.cmdargtoken[i].type ==
                        CMDARGTOKEN_TYPE_EXISTINGIMAGE) // existing image
                {
                    printf(
                        "\t CMDARGTOKEN_TYPE_EXISTINGIMAGE : "
                        "%s\n",
                        data.cmdargtoken[i].val.string);
                }
                if(data.cmdargtoken[i].type ==
                        CMDARGTOKEN_TYPE_COMMAND) // command
                {
                    printf(
                        "\t CMDARGTOKEN_TYPE_COMMAND       : "
                        "%s\n",
                        data.cmdargtoken[i].val.string);
                }
                if(data.cmdargtoken[i].type ==
                        CMDARGTOKEN_TYPE_RAWSTRING) // unprocessed string
                {
                    printf(
                        "\t CMDARGTOKEN_TYPE_RAWSTRING    : "
                        "%s\n",
                        data.cmdargtoken[i].val.string);
                }

                i++;
            }
        }

        if(dcdebug > 0)
        {
            printf("DEBUG: %s %d: data.parseerror = %d\n",
                   __func__, __LINE__,
                   data.parseerror);
        }

        if(data.parseerror == 0)
        {
            if(data.cmdargtoken[0].type == CMDARGTOKEN_TYPE_COMMAND)
            {
                // Execute CLI command
                data.cmd[data.cmdindex]
                    .callcount++;

                struct timespec t0, t1;
                clock_gettime(CLOCK_MONOTONIC, &t0);

                data.CMDerrstatus =
                    data.cmd[data.cmdindex].fp();

                if(data.print_cmd_timing)
                {
                    clock_gettime(CLOCK_MONOTONIC, &t1);
                    double elapsed_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + 
                                        (t1.tv_nsec - t0.tv_nsec) / 1000000.0;
                    printf("Execution time: %.3f ms\n", elapsed_ms);
                }

                cli_save_last_argument();

                if(data.CMDerrstatus != RETURN_SUCCESS)
                {
                    // CLI function returns error
                    // print function key name and error code
                    printf(
                        "\n%c[%d;%dm ERROR %c[%d;m CLI "
                        "function %s returns %d\n",
                        (char) 27,
                        1,
                        31,
                        (char) 27,
                        0,
                        data.cmd[data.cmdindex].key,
                        data.CMDerrstatus);

                    if(dcerrorexit == 1)
                    {
                        printf(
                            "%c[%d;%dm -> EXIT CLI "
                            "%c[%d;m\n",
                            (char) 27,
                            1,
                            31,
                            (char) 27,
                            0);
                        dcexitcode = data.CMDerrstatus;

#ifndef NDEBUG
                        // output trace debug
                        write_tracedebugfile();
#endif
                    }
                }

                data.CMDexecuted = 1;
            }
        }
        else
        {
            if(dcerrorexit == 1)
            {
                dcexitcode = 1;
            }
        }

        for(int i = 0; i < data.calctmp_imindex; i++)
        {
            CREATE_IMAGENAME(calctmpimname, "_tmpcalc%d", i);
            if(image_ID(calctmpimname, dcimg, dcnimg) != -1)
            {
                if(dcdebug == 1)
                {
                    printf("Deleting %s\n", calctmpimname);
                }
                delete_image_ID(calctmpimname, DELETE_IMAGE_ERRMODE_WARNING);
            }
        }

        if(!((data.cmdargtoken[0].type == CMDARGTOKEN_TYPE_STRING) ||
                (data.cmdargtoken[0].type == CMDARGTOKEN_TYPE_RAWSTRING)))
        {
            data.CMDexecuted = 1;
        }
    }

    if((data.CMDexecuted == 0) && (data.CLIloopON == 1))
    {
        /* Attempt transparent OS shell fallback */
        int sys_ret = system(data.CLIcmdline);
        int os_not_found = 0;
        
        /* system() returns 127 << 8 if command not found by sh */
        if(sys_ret != -1 && ((sys_ret >> 8) & 0xff) == 127)
        {
            os_not_found = 1;
        }
        else
        {
            /* OS processed it (success, or other error), update ret val */
            if(sys_ret != -1)
            {
                cli_last_retval = (sys_ret >> 8) & 0xff;
            }
        }

        if(os_not_found)
        {
#ifdef USE_READLINE
            if(data.cmdNBarg > 0 && strlen(data.cmdargtoken[0].val.string) > 0)
            {
                const char *input_cmd = data.cmdargtoken[0].val.string;
                
                struct MatchNode {
                    int dist;
                    const char *cmd;
                } matches[3] = { {9999, NULL}, {9999, NULL}, {9999, NULL} };

                for(unsigned int i = 0; i < data.NBcmd; i++) {
                    int d = levenshtein_distance((const char*)input_cmd,
                        (const char*)data.cmd[i].key);
                    
                    if (d < matches[2].dist) {
                        matches[2].dist = d;
                        matches[2].cmd = data.cmd[i].key;
                        
                        if (matches[2].dist < matches[1].dist) {
                            struct MatchNode tmp = matches[1];
                            matches[1] = matches[2];
                            matches[2] = tmp;
                        }
                        if (matches[1].dist < matches[0].dist) {
                            struct MatchNode tmp = matches[0];
                            matches[0] = matches[1];
                            matches[1] = tmp;
                        }
                    }
                }

                if(matches[0].dist <= 4 && matches[0].cmd != NULL) {
                    printf(COLORRED "Command '%s' not found. " COLORRESET
                           "Did you mean:\n", input_cmd);
                    for (int m = 0; m < 3; m++) {
                        if (matches[m].cmd && matches[m].dist <= 4 && matches[m].dist < 9999) {
                            printf("  - " COLORHBOLDCYAN "%s" COLORRESET "\n", matches[m].cmd);
                        }
                    }
                } else {
                    printf(COLORRED "Command not found, or command with no effect\n" COLORRESET);
                }
            }
            else
#endif
            {
                printf(COLORRED
                       "Command not found, or command with no effect\n" COLORRESET);
            }
        }
    }

    /* Restore stdout if pipe was active */
    if(pipe_fp != NULL)
    {
        fflush(stdout);
        dup2(saved_stdout_fd, STDOUT_FILENO);
        close(saved_stdout_fd);
        pclose(pipe_fp);
    }
    /* Restore stdout if redirect was active */
    if(redir_fp != NULL)
    {
        fflush(stdout);
        dup2(saved_stdout_redir,
             STDOUT_FILENO);
        close(saved_stdout_redir);
        fclose(redir_fp);
    }

    free(thetime);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

#ifdef USE_READLINE

/** @brief Stores the current inline suggestion suffix
 *
 * Set by CLI_redisplay when a suggestion is shown.
 * Consumed by accept_suggestion when Right Arrow is pressed.
 */
static char *pending_suggestion = NULL;
static int   pending_replace_len = 0;

/**
 * @brief Accept the inline suggestion on Right Arrow
 *
 * If cursor is at end-of-line and a pending suggestion
 * exists, insert it. Otherwise, fall through to normal
 * cursor-right movement.
 */
static int accept_suggestion(
    int count,
    int key
)
{
    (void) count;
    (void) key;

    if(pending_suggestion && rl_point == rl_end)
    {
        if(pending_replace_len > 0
            && rl_end >= pending_replace_len)
        {
            int del_start =
                rl_end - pending_replace_len;
            rl_delete_text(del_start, rl_end);
            rl_point = del_start;
        }
        rl_insert_text(pending_suggestion);
        free(pending_suggestion);
        pending_suggestion = NULL;
        pending_replace_len = 0;
        rl_redisplay();
        return 0;
    }

    /* Not at EOL or no suggestion — normal right */
    return rl_forward_char(1, key);
}

/**
 * @brief Store the suggestion suffix for Right Arrow
 */
static void set_pending_suggestion(
    const char *text,
    int replace_len
)
{
    free(pending_suggestion);
    pending_suggestion = NULL;
    pending_replace_len = 0;
    if(text && strlen(text) > 0)
    {
        pending_suggestion =
            dupstr((char *) text);
        pending_replace_len = replace_len;
    }
}

/**
 * @brief Find the command index matching `firstword`
 *
 * Returns -1 if no match.
 */
static int find_command_match(const char *firstword)
{
    for(uint32_t cmdi = 0; cmdi < data.NBcmd; cmdi++)
    {
        if(strcmp(firstword, data.cmd[cmdi].key) == 0)
        {
            data.cmdindex = cmdi;
            return (int) cmdi;
        }
    }
    return -1;
}

/**
 * @brief Compute visible length of readline prompt
 *
 * Strips \001..\002 escape wrappers that readline
 * uses to mark non-printing characters.
 */
static int visible_prompt_len(void)
{
    const char *p = rl_display_prompt
                        ? rl_display_prompt
                        : "";
    int   len = 0;
    int   invisible = 0;

    for(; *p; p++)
    {
        if(*p == '\001')
        {
            invisible = 1;
        }
        else if(*p == '\002')
        {
            invisible = 0;
        }
        else if(!invisible)
        {
            len++;
        }
    }
    return len;
}

/**
 * @brief Get number of ghost chars we can print
 *
 * Returns max chars that can be printed after the
 * cursor without wrapping to the next terminal line.
 */
static int get_ghost_budget(void)
{
    struct winsize ws;
    int cols = 80; /* fallback */

    if(ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) >= 0
            && ws.ws_col > 0)
    {
        cols = ws.ws_col;
    }

    int cursor_col =
        (visible_prompt_len() + rl_point) % cols;

    int budget = cols - cursor_col - 1;
    if(budget < 0)
    {
        budget = 0;
    }
    return budget;
}

/**
 * @brief Print ghost text with truncation
 *
 * Prints up to budget visible chars from text
 * in the given ANSI style, then resets style.
 * Returns number of visible chars printed.
 */
static int print_ghost(
    const char *style,
    const char *text,
    int budget
)
{
    int tlen = (int) strlen(text);
    int plen = tlen < budget ? tlen : budget;

    if(plen <= 0)
    {
        return 0;
    }

    printf("%s%.*s\033[0m", style, plen, text);
    return plen;
}

/**
 * @brief State for the reserved hint area
 */
static int hint_area_active = 0;
static int cached_term_rows = 0;
static int cached_term_cols = 0;

/**
 * @brief Set up scroll region reserving bottom line
 *
 * Confines normal terminal output to lines 1..(rows-1)
 * so the bottom line stays fixed for hints.
 */
void CLI_setup_hint_area(void)
{
    struct winsize ws;
    if(ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) < 0
            || ws.ws_row <= 3)
    {
        hint_area_active = 0;
        return;
    }

    cached_term_rows = ws.ws_row;
    cached_term_cols = ws.ws_col;

    /* Ensure there's a free line below the cursor.
     * If the cursor is at the bottom of the screen, \n scrolls
     * the screen up by 1. \033[1A moves it back to its relative
     * position. This ensures the cursor is never at 'rows'
     * (the reserved hint line) before we save its position. */
    printf("\n\033[1A");

    /* Save the current cursor position using ANSI */
    printf("\033[s");

    /* Set scroll region to rows 1..(rows-1)
     * NOTE: DECSTBM moves cursor to home (1, 1) */
    printf("\033[1;%dr", cached_term_rows - 1);

    /* Clear the hint line (outside scroll region) */
    printf("\033[%d;1H\033[2K", cached_term_rows);

    /* Restore cursor to where it was using ANSI */
    printf("\033[u");
    fflush(stdout);

    hint_area_active = 1;
}

/**
 * @brief Reset scroll region to full terminal
 *
 * Call this before exiting readline mode or
 * when the CLI session ends.
 */
void CLI_cleanup_scroll_region(void)
{
    if(!hint_area_active)
    {
        return;
    }

    /* Save cursor position (ANSI — does NOT
     * touch scroll margins) */
    printf("\033[s");

    /* Move to hint line and erase it BEFORE
     * resetting scroll region, while the row
     * is still addressable at its cached pos */
    printf("\033[%d;1H\033[2K", cached_term_rows);

    /* Reset scroll region to full terminal.
     * This also moves cursor to (1,1). */
    printf("\033[r");

    /* Restore cursor to where it was before */
    printf("\033[u");

    fflush(stdout);

    hint_area_active = 0;
}

/**
 * @brief Update the hint area with function prototype
 *
 * Paints the reserved bottom line with the command
 * syntax when a known command is being typed.
 */
static void update_hint_area(void)
{
    if(!hint_area_active || !data.autocomplete_arghint)
    {
        return;
    }

    /* Single ANSI save cursor for the whole
     * operation (resize + hint painting) */
    printf("\033[s");

    /* Check for terminal resize */
    {
        struct winsize ws;
        if(ioctl(STDOUT_FILENO, TIOCGWINSZ,
                 &ws) >= 0 &&
                ws.ws_row > 3 &&
                (ws.ws_row != cached_term_rows ||
                 ws.ws_col != cached_term_cols))
        {
            cached_term_rows = ws.ws_row;
            cached_term_cols = ws.ws_col;
            /* Re-set scroll region (cursor jumps
             * to home, but we saved it above) */
            printf("\033[1;%dr",
                   cached_term_rows - 1);
            /* Clear new hint line */
            printf("\033[%d;1H\033[2K",
                   cached_term_rows);
        }
    }

    /* Move to hint line, clear it */
    printf("\033[%d;1H\033[2K",
           cached_term_rows);

    /* Check if first word is a known command */
    if(rl_line_buffer[0] != '\0')
    {
        char  buf[200];
        char *saveptr_hint = NULL;
        snprintf(buf, sizeof(buf), "%s",
                 rl_line_buffer);
        char *fw = strtok_r(
                       buf, " ", &saveptr_hint);

        if(fw != NULL)
        {
            int cmi = find_command_match(fw);
            if(cmi >= 0)
            {
                /* Count argument words after cmd
                 * to determine current arg */
                int argidx = 0;
                {
                    const char *p =
                        rl_line_buffer;
                    /* Skip command word */
                    while(*p && *p != ' ')
                    {
                        p++;
                    }
                    /* Count argument words */
                    int wcount = 0;
                    int in_word = 0;
                    while(*p)
                    {
                        if(*p != ' ')
                        {
                            if(!in_word)
                            {
                                wcount++;
                                in_word = 1;
                            }
                        }
                        else
                        {
                            in_word = 0;
                        }
                        p++;
                    }
                    if(rl_end > 0 &&
                            rl_line_buffer[
                                rl_end - 1]
                            == ' ')
                    {
                        /* Trailing space: about
                         * to type next arg */
                        argidx = wcount;
                    }
                    else
                    {
                        /* Mid-word: typing this
                         * arg (0-indexed) */
                        argidx = wcount > 0
                                 ? wcount - 1 : 0;
                    }
                }

                /* Print syntax with <> delimited
                 * tokens, highlighting active */
                const char *syn =
                    data.cmd[cmi].syntax;
                int col = 0;
                int tidx = 0;
                const char *p = syn;

                while(*p &&
                        col < cached_term_cols - 2)
                {
                    /* Skip whitespace between
                     * argument tokens */
                    if(*p == ' ')
                    {
                        printf(" ");
                        col++;
                        p++;
                        continue;
                    }

                    /* Find extent of this token */
                    const char *tstart = p;
                    if(*p == '<')
                    {
                        /* Scan to matching '>' */
                        while(*p && *p != '>')
                        {
                            p++;
                        }
                        if(*p == '>')
                        {
                            p++;
                        }
                    }
                    else
                    {
                        /* Non-<> word */
                        while(*p && *p != ' '
                                && *p != '<')
                        {
                            p++;
                        }
                    }
                    int tlen = (int)(p - tstart);
                    int avail =
                        cached_term_cols - 1 - col;
                    int plen = tlen < avail
                               ? tlen : avail;

                    if(*tstart == '<' &&
                            tidx == argidx)
                    {
                        printf("\033[1;97m"
                               "%.*s"
                               "\033[0m",
                               plen, tstart);
                    }
                    else
                    {
                        printf("\033[2m"
                               "%.*s"
                               "\033[0m",
                               plen, tstart);
                    }
                    col += plen;
                    if(*tstart == '<')
                    {
                        tidx++;
                    }
                }
            }
        }
    }

    /* ANSI restore cursor */
    printf("\033[u");
    fflush(stdout);
}

static void CLI_redisplay(void)
{
    /* Default or syntax-highlighted redisplay */
    rl_redisplay_function = NULL;
    if(data.syntax_highlight
            && rl_line_buffer[0] != '\0')
    {
        cli_highlight_redisplay();
    }
    else
    {
        rl_redisplay();
        fflush(stdout);
    }
    rl_redisplay_function = CLI_redisplay;

    /* Clear any stale suggestion */
    set_pending_suggestion(NULL, 0);

    if(data.autocomplete == 0)
    {
        return;
    }

    if(rl_line_buffer[0] == '\0')
    {
        update_hint_area();
        return;
    }


    if(rl_point != rl_end)
    {
        /* Update hint even when not at EOL */
        update_hint_area();
        return;
    }

    int budget = get_ghost_budget();
    if(budget <= 0)
    {
        update_hint_area();
        return;
    }

    int total_ghost = 0;

    /* ===== History-based suggestion ===== */
    if(data.autocomplete_history)
    {
        HIST_ENTRY **hist = history_list();
        if(hist)
        {
            int hlen = history_length;
            for(int i = hlen - 1; i >= 0; i--)
            {
                if(strncmp(hist[i]->line,
                           rl_line_buffer,
                           rl_end) == 0 &&
                        (int) strlen(
                            hist[i]->line) >
                        rl_end)
                {
                    const char *suffix =
                        hist[i]->line + rl_end;
                    int n = print_ghost(
                                "\033[38;5;245m",
                                suffix,
                                budget);
                    if(n > 0)
                    {
                        printf("\033[K");
                        printf("\033[%dD", n);
                        fflush(stdout);
                        set_pending_suggestion(
                            suffix, 0);
                    }
                    update_hint_area();
                    return;
                }
            }
        }
    }

    /* ===== Generator-based suggestion ===== */

    /* Find current word start */
    int start = 0;
    for(int i = rl_point - 1; i >= 0; i--)
    {
        if(rl_line_buffer[i] == ' ')
        {
            start = i + 1;
            break;
        }
    }

    char *text = rl_line_buffer + start;

    /* Determine matching mode */
    if((start == 0) ||
            (strncmp(rl_line_buffer, "cmd?",
                     strlen("cmd?")) == 0))
    {
        data.CLImatchMode =
            CLICOMPLETIONMODE_COMMANDS;
    }
    else
    {
        char  str[200];
        char *saveptr_comp = NULL;
        snprintf(str, 200, "%s", rl_line_buffer);
        char *firstword = strtok_r(
                str, " ", &saveptr_comp);

        int cmdimatch = -1;
        if(firstword != NULL)
        {
            cmdimatch =
                find_command_match(firstword);
        }

        /* If command has no <> argument tokens,
         * don't suggest arguments */
        if(cmdimatch >= 0)
        {
            const char *syn =
                data.cmd[cmdimatch].syntax;
            if(syn == NULL ||
                    strchr(syn, '<') == NULL)
            {
                update_hint_area();
                return;
            }
        }

        if((cmdimatch != -1) && (text[0] == '.'))
        {
            data.CLImatchMode =
                CLICOMPLETIONMODE_CMDARGS;
        }
        else
        {
            data.CLImatchMode =
                CLICOMPLETIONMODE_IMAGES;
        }
    }

    /* Get best match */
    char *match = CLI_generator(text, 0);

    if(match)
    {
        if(strncmp(match, text, strlen(text)) == 0)
        {
            char *suffix = match + strlen(text);
            int n = print_ghost(
                        "\033[38;5;245m",
                        suffix,
                        budget);
            if(n > 0)
            {
                total_ghost += n;
                set_pending_suggestion(
                    suffix, 0);
            }
        }
        else if(data.autocomplete_fuzzy)
        {
            char fzbuf[256];
            snprintf(fzbuf, sizeof(fzbuf),
                     " [%s]", match);
            int n = print_ghost(
                        "\033[38;5;245m",
                        fzbuf,
                        budget);
            total_ghost += n;
            set_pending_suggestion(
                match,
                (int) strlen(text));
        }
        free(match);
    }

    /* Erase rest of line + move cursor back */
    if(total_ghost > 0)
    {
        printf("\033[K");
        printf("\033[%dD", total_ghost);
        fflush(stdout);
    }

    update_hint_area();
}

void CLI_configure_readline()
{
    rl_redisplay_function = CLI_redisplay;

    if(data.autocomplete_history)
    {
        read_history(CLI_history_file());
    }

    /* Bind Right Arrow to accept suggestion
     * when at end-of-line */
    rl_bind_keyseq("\\e[C", accept_suggestion);
}
#else
void CLI_configure_readline() {}
void CLI_setup_hint_area(void) {}
void CLI_cleanup_scroll_region(void) {}
#endif

