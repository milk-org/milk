/**
 * @file CLIcore_UI.c
 *
 * @brief User input (UI) functions
 *
 */

#include <stdio.h>
#include <sys/ioctl.h>

#ifdef USE_READLINE
#include <readline/history.h>
#include <readline/readline.h>
#endif


#include "CLIcore.h"
#include "CommandLineInterface/calc.h"
#include "CommandLineInterface/calc_bison.h"

#include "COREMOD_memory/COREMOD_memory.h"
#include "timeutils.h"

#define CLICOMPLETIONMODE_COMMANDS 0
#define CLICOMPLETIONMODE_IMAGES   1
#define CLICOMPLETIONMODE_CMDARGS  2

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
        while(list_index1 < dcnimg)
        {
            int iok;
            iok = dcimg[list_index1].used;
            if(iok == 1)
            {
                name = dcimg[list_index1].name;
            }
            list_index1++;
            if(iok == 1)
            {
                if(generator_fuzzy_pass == 0)
                {
                    if(strncmp(name, text, len) == 0)
                    {
                        return (dupstr(name));
                    }
                }
                else
                {
                    if(strstr(name, text) != NULL)
                    {
                        return (dupstr(name));
                    }
                }
            }
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
        else
        {
            // match string with images
            data.CLImatchMode = CLICOMPLETIONMODE_IMAGES;
        }
    }

    matches = rl_completion_matches((char *) text, &CLI_generator);

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
                    if(dcdebug > 1)
                    {
                        printf("DEBUG: %s %d: calling yy_scan_string on \"%s\"\n", __func__, __LINE__,
                               str);
                    }
                    yy_scan_string(str);
                    data.calctmp_imindex = 0;
                    yyparse();
                    yylex_destroy();

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
                data.CMDerrstatus = data.cmd[data.cmdindex].fp();

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
#ifdef USE_READLINE
        if(data.cmdNBarg > 0 && strlen(data.cmdargtoken[0].val.string) > 0)
        {
            const char *input_cmd = data.cmdargtoken[0].val.string;
            int best_dist = 9999;
            const char *best_match = NULL;

            for(unsigned int i = 0; i < data.NBcmd; i++) {
                int d = levenshtein_distance((const char*)input_cmd, (const char*)data.cmd[i].key);
                if(d < best_dist) {
                    best_dist = d;
                    best_match = data.cmd[i].key;
                }
            }

            if(best_dist <= 3 && best_match != NULL) {
                printf(COLORRED "Command '%s' not found. " COLORRESET
                       "Did you mean " COLORHBOLDCYAN "'%s'" COLORRESET "?\n",
                       input_cmd, best_match);
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
        rl_insert_text(pending_suggestion);
        free(pending_suggestion);
        pending_suggestion = NULL;
        rl_redisplay();
        return 0;
    }

    /* Not at EOL or no suggestion — normal right */
    return rl_forward_char(1, key);
}

/**
 * @brief Store the suggestion suffix for Right Arrow
 */
static void set_pending_suggestion(const char *suffix)
{
    free(pending_suggestion);
    pending_suggestion = NULL;
    if(suffix && strlen(suffix) > 0)
    {
        pending_suggestion = dupstr((char *) suffix);
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

    /* Set scroll region to rows 1..(rows-1)
     * NOTE: DECSTBM moves cursor to home */
    printf("\033[1;%dr", cached_term_rows - 1);

    /* Clear the hint line (outside scroll region) */
    printf("\033[%d;1H\033[2K",
           cached_term_rows);

    /* Position cursor at last line of scroll
     * region. readline will print prompt here. */
    printf("\033[%d;1H",
           cached_term_rows - 1);
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

    /* Save cursor before scroll region reset */
    printf("\0337");

    /* Clear hint line */
    printf("\033[%d;1H\033[2K",
           cached_term_rows);

    /* Reset scroll region to full terminal
     * (also moves cursor to home) */
    printf("\033[r");

    /* Restore cursor */
    printf("\0338");
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

    /* Single DEC save cursor for the whole
     * operation (resize + hint painting) */
    printf("\0337");

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

    /* DEC restore cursor */
    printf("\0338");
    fflush(stdout);
}

static void CLI_redisplay(void)
{
    /* Default redisplay */
    rl_redisplay_function = NULL;
    rl_redisplay();
    fflush(stdout);
    rl_redisplay_function = CLI_redisplay;

    /* Clear any stale suggestion */
    set_pending_suggestion(NULL);

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
                            suffix);
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
                set_pending_suggestion(suffix);
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

