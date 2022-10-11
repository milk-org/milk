/**
 * @file CLIcore_UI.c
 *
 * @brief User input (UI) functions
 *
 */

#include <stdio.h>

#include <readline/history.h>
#include <readline/readline.h>


#include "CommandLineInterface/CLIcore.h"
#include "CommandLineInterface/calc.h"
#include "CommandLineInterface/calc_bison.h"

#include "COREMOD_memory/COREMOD_memory.h"
#include "timeutils.h"

#define CLICOMPLETIONMODE_COMMANDS 0
#define CLICOMPLETIONMODE_IMAGES   1
#define CLICOMPLETIONMODE_CMDARGS  2

#define COLORRESET     "\033[0m"
#define COLORRED       "\033[31m" /* Red */
#define COLORHBOLDCYAN "\e[0;96m" /* High Intensity Bold Cyan */

extern void yy_scan_string(const char *);
extern int  yylex_destroy(void);

/**
 * @brief Readline callback
 *
 **/
void rl_cb_linehandler(char *linein)
{
    if(NULL == linein)
    {
        return;
    }

    data.CLIexecuteCMDready = 1;

    // copy input into data.CLIcmdline
    strcpy(data.CLIcmdline, linein);
    CLI_execute_line();

    free(linein);
}

errno_t runCLI_prompt(char *promptstring, char *prompt)
{
    //int color_cyan = 36;

    if(strlen(promptstring) > 0)
    {
        if(data.processnameflag == 0)
        {
            sprintf(prompt, COLORHBOLDCYAN "%s > " COLORRESET, promptstring);
        }
        else
        {
            sprintf(prompt,
                    COLORHBOLDCYAN "%s-%s > " COLORRESET,
                    promptstring,
                    data.processname);
        }
    }
    else
    {
        sprintf(prompt, COLORHBOLDCYAN "%s > " COLORRESET, data.processname);
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

static char *CLI_generator(const char *text, int state)
{
    static unsigned int list_index;
    static unsigned int list_index1;
    static unsigned int len;
    char               *name;

    //printf("[generator %d %d %d]\n", state, data.CLImatchMode, list_index);

    if(!state)
    {
        list_index  = 0;
        list_index1 = 0;
        len         = strlen(text);
    }

    if(data.CLImatchMode == CLICOMPLETIONMODE_COMMANDS)
    {
        // search through list of commands
        while(list_index < data.NBcmd)
        {
            name = data.cmd[list_index].key;
            list_index++;
            if(strncmp(name, text, len) == 0)
            {
                return (dupstr(name));
            }
        }
    }

    if(data.CLImatchMode == CLICOMPLETIONMODE_IMAGES)
    {
        // search through list of images
        while(list_index1 < data.NB_MAX_IMAGE)
        {
            int iok;
            iok = data.image[list_index1].used;
            if(iok == 1)
            {
                name = data.image[list_index1].name;
                //	  printf("  name %d = %s %s\n", list_index1, data.image[list_index1].name, name);
            }
            list_index1++;
            if(iok == 1)
            {
                if(strncmp(name, text, len) == 0)
                {
                    return (dupstr(name));
                }
            }
        }
    }

    if(data.CLImatchMode == CLICOMPLETIONMODE_CMDARGS)
    {
        // search through command arguments and parameters
        while((int) list_index < data.cmd[data.cmdindex].nbarg)
        {
            name = data.cmd[data.cmdindex].argdata[list_index].fpstag;
            list_index++;
            if(strncmp(name, text, len) == 0)
            {
                return (dupstr(name));
            }
        }
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

errno_t write_tracedebugfile()
{
    pid_t thisPID = getpid();

    char fname[STRINGMAXLEN_FILENAME];
    WRITE_FILENAME(fname, "milk-codetracepoint.%05d.log", thisPID);

    printf("Writing output trace to file %s\n", fname);
    printf("data.testpointarrayinit = %d\n", data.testpointarrayinit);

    FILE *fp = fopen(fname, "w");
    if(fp != NULL)
    {
        for(uint64_t i = 0; i < CODETESTPOINTARRAY_NBCNT; i++)
        {
            long j = (i + data.testpointcnt) % CODETESTPOINTARRAY_NBCNT;

            uint64_t index =
                data.testpointarray[j].loopcnt * CODETESTPOINTARRAY_NBCNT + j;

            if(data.testpointarray[j].line != 0)
            {
                char timestring[20];
                mkUTtimestring_nanosec(timestring, data.testpointarray[j].time);

                // extract last word
                char str[STRINGMAXLEN_FULLFILENAME];
                strcpy(str, data.testpointarray[j].file);
                char *lastword = strrchr(str, '/') + 1;

                fprintf(fp,
                        "T %6ld %s %-20s %6d %-20s  %s\n",
                        index,
                        timestring,
                        lastword,
                        data.testpointarray[j].line,
                        data.testpointarray[j].func,
                        data.testpointarray[j].msg);
                fprintf(fp,
                        "       FTRACE %d ",
                        data.testpointarray[j].funclevel);
                for(int level = 0; level < data.testpointarray[j].funclevel;
                        level++)
                {
                    fprintf(fp,
                            " (%d) >> %ld:%s",
                            data.testpointarray[j].linestack[level],
                            data.testpointarray[j].fcntstack[level],
                            data.testpointarray[j].funcstack[level]);
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
    char             str[200];
    FILE            *fp;
    time_t           t;
    struct tm       *uttime;
    struct timespec *thetime =
        (struct timespec *) malloc(sizeof(struct timespec));

    add_history(data.CLIcmdline);

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
            clock_gettime(CLOCK_REALTIME, thetime);

            sprintf(data.CLIlogname,
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
                    sprintf(str, "%s\n", cmdargstring);
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

        if(data.Debug == 1)
        {
            long i = 0;
            while(data.cmdargtoken[i].type != 0)
            {

                printf("TOKEN %ld/%ld   \"%s\"  type : %d\n",
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

        if(data.parseerror == 0)
        {
            if(data.cmdargtoken[0].type == CMDARGTOKEN_TYPE_COMMAND)
            {
                if(data.Debug == 1)
                {
                    printf("EXECUTING COMMAND %ld (%s)\n",
                           data.cmdindex,
                           data.cmd[data.cmdindex].key);
                }
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

                    if(data.errorexit == 1)
                    {
                        printf(
                            "%c[%d;%dm -> EXIT CLI "
                            "%c[%d;m\n",
                            (char) 27,
                            1,
                            31,
                            (char) 27,
                            0);
                        data.exitcode = data.CMDerrstatus;

#ifndef NDEBUG
                        // output trace debug
                        write_tracedebugfile();
#endif
                    }
                }

                data.CMDexecuted = 1;
            }
        }

        for(int i = 0; i < data.calctmp_imindex; i++)
        {
            CREATE_IMAGENAME(calctmpimname, "_tmpcalc%d", i);
            //sprintf(calctmpimname, "_tmpcalc%ld", i);
            if(image_ID(calctmpimname) != -1)
            {
                if(data.Debug == 1)
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
        printf(COLORRED
               "Command not found, or command with no effect\n" COLORRESET);
    }

    free(thetime);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
