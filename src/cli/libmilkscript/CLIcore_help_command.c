/**
 * @file    CLIcore_help_command.c
 * @brief   Per-command help and search functions
 *
 * Contains help_command(), command_info_search(),
 * help(), and help_module().
 *
 * @see CLIcore_help.c for system info and listings.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <libgen.h>
#include <malloc.h>

#include <fitsio.h>
#include <regex.h>

#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <sys/ioctl.h>
#include "CLIcore.h"

#include "fps.h"
#include "fps_connect.h"

#define C_RST    "\033[0m"
#define C_TITLE  "\033[1;36m"
#define C_HDR    "\033[1;35m"
#define C_CMD    "\033[32m"
#define C_BOLD   "\033[1m"
#define C_NOTE   "\033[33m"
#define C_ERR    "\033[1;31m"

#ifndef COLORRESET
#define COLORRESET     C_RST
#endif
#ifndef COLORCMD
#define COLORCMD       C_CMD
#endif
#ifndef COLORINFO
#define COLORINFO      "\033[32m"
#endif
#ifndef COLORARGCLI
#define COLORARGCLI    "\033[36m"
#endif
#ifndef COLORARGnotCLI
#define COLORARGnotCLI "\033[35m"
#endif

extern int help_format_mode;

static int checkFlag64(
    uint64_t flags,
    uint64_t testflag,
    char    *flagdescription
)
{
    int rval = 0;

    // printf("--------- flags: %ld\n", flags);

    if(flags & testflag)
    {
        rval = 1;
        printf("    [%c[%d;%dm ON%c[%dm]  %s\n",
               (char) 27,
               1,
               32,
               (char) 27,
               0,
               flagdescription);
    }
    else
    {
        rval = 0;
        printf("    [%c[%d;%dmOFF%c[%dm]  %s\n",
               (char) 27,
               1,
               31,
               (char) 27,
               0,
               flagdescription);
    }
    return rval;
}

/**
 * @brief command help\n
 *
 * @param[in] cmdkey Commmand name
 */

errno_t help_command(
    const char *__restrict cmdkey
)
{
    int cOK = 0;

    for(unsigned int cmdi = 0; cmdi < data.NBcmd; cmdi++)
    {
        if(!strcmp(cmdkey, data.cmd[cmdi].key))
        {
            printf("\n");
            printf(COLORCMD "%s" COLORRESET
                   " in %s [%s]\n"
                   "\t" COLORINFO
                   "%s\n" COLORRESET,
                   data.cmd[cmdi].key,
                   data.cmd[cmdi].module,
                   data.module[
                       data.cmd[cmdi]
                       .moduleindex]
                   .shortname,
                   data.cmd[cmdi].info);

            printf("\t\033[33mexample>\033[0m"
                   " \033[1m%s\033[0m\n",
                   data.cmd[cmdi].example);
            printf("\t\033[2msrc: %s\033[0m"
                   "\n",
                   data.cmd[cmdi].srcfile);

            int FPSsupport = checkFlag64(data.cmd[cmdi].cmdsettings.flags,
                                         CLICMDFLAG_FPS,
                                         "FPS support");
            if(FPSsupport == 1)
            {
                if(checkFlag64(data.cmd[cmdi].cmdsettings.flags,
                               CLICMDFLAG_PROCINFO,
                               "processinfo support (..procinfo 0/1)") == 1)
                {
                    printf("        loopcntMax         : %ld\n",
                           data.cmd[cmdi].cmdsettings.procinfo_loopcntMax);
                    printf("      Triggering:\n");

                    printf("        triggermode        : %d ",
                           data.cmd[cmdi].cmdsettings.triggermode);
                    switch(data.cmd[cmdi].cmdsettings.triggermode)
                    {
                    case PROCESSINFO_TRIGGERMODE_IMMEDIATE:
                        printf("IMMEDIATE");
                        break;
                    case PROCESSINFO_TRIGGERMODE_CNT0:
                        printf("CNT0");
                        break;
                    case PROCESSINFO_TRIGGERMODE_CNT1:
                        printf("CNT1");
                        break;
                    case PROCESSINFO_TRIGGERMODE_SEMAPHORE:
                        printf("SEMAPHORE");
                        break;
                    case PROCESSINFO_TRIGGERMODE_DELAY:
                        printf("DELAY");
                        break;
                    case PROCESSINFO_TRIGGERMODE_CNT2:
                        printf("CNT2");
                        break;
                    default:
                        printf("unknown");
                        break;
                    }
                    printf("\n");

                    printf("        triggerstreamname  : %s\n",
                           data.cmd[cmdi].cmdsettings.triggerstreamname);

                    printf("        semindexrequested  : %d\n",
                           data.cmd[cmdi].cmdsettings.semindexrequested);

                    printf(
                        "        triggerdelay       : "
                        "%lld.%09ld\n",
                        (long long) data.cmd[cmdi]
                        .cmdsettings.triggerdelay.tv_sec,
                        data.cmd[cmdi].cmdsettings.triggerdelay.tv_nsec);

                    printf(
                        "        triggertimeout     : "
                        "%lld.%09ld\n",
                        (long long) data.cmd[cmdi]
                        .cmdsettings.triggertimeout.tv_sec,
                        data.cmd[cmdi].cmdsettings.triggertimeout.tv_nsec);

                    printf("      Resources:\n");
                    printf("        RT_priority        : %d\n",

                           data.cmd[cmdi].cmdsettings.RT_priority);

                    printf("        CPUmask            : ");

                    int nproc = sysconf(_SC_NPROCESSORS_ONLN);
                    for(int cpu = 0; cpu < nproc; cpu++)
                    {
                        printf(" %d",
                               CPU_ISSET(cpu,
                                         &data.cmd[cmdi].cmdsettings.CPUmask));
                    }
                    printf("\n");
                    printf("        MeasureTiming      : %d\n",
                           data.cmd[cmdi].cmdsettings.procinfo_MeasureTiming);
                }
            }

            printf("\n");
            printf("  CLI call arguments:\n");
            //printf("  CLI#       tagname             Value         description\n");

            int CLIargcnt = 0;
            for(int argi = 0; argi < data.cmd[cmdi].nbparam; argi++)
            {
                char valuestring[STRINGMAXLEN_CLICMDARG] = "???";

                switch(data.cmd[cmdi].argdata[argi].type)
                {
                case CLIARG_FLOAT32:
                    SNPRINTF_CHECK(valuestring,
                                   STRINGMAXLEN_CLICMDARG,
                                   "[float32]  %f",
                                   data.cmd[cmdi].argdata[argi].val.f32);
                    break;

                case CLIARG_FLOAT64:
                    SNPRINTF_CHECK(valuestring,
                                   STRINGMAXLEN_CLICMDARG,
                                   "[float64]  %lf",
                                   data.cmd[cmdi].argdata[argi].val.f64);
                    break;

                case CLIARG_ONOFF:
                    SNPRINTF_CHECK(valuestring,
                                   STRINGMAXLEN_CLICMDARG,
                                   "[ ONOFF ]  %ld",
                                   data.cmd[cmdi].argdata[argi].val.ui64);
                    break;

                case CLIARG_INT32:
                    SNPRINTF_CHECK(valuestring,
                                   STRINGMAXLEN_CLICMDARG,
                                   "[ int32 ]  %d",
                                   data.cmd[cmdi].argdata[argi].val.i32);
                    break;

                case CLIARG_UINT32:
                    SNPRINTF_CHECK(valuestring,
                                   STRINGMAXLEN_CLICMDARG,
                                   "[uint32 ]  %u",
                                   data.cmd[cmdi].argdata[argi].val.ui32);
                    break;

                case CLIARG_INT64:
                    SNPRINTF_CHECK(valuestring,
                                   STRINGMAXLEN_CLICMDARG,
                                   "[ int64 ]  %ld",
                                   data.cmd[cmdi].argdata[argi].val.i64);
                    break;

                case CLIARG_UINT64:
                    SNPRINTF_CHECK(valuestring,
                                   STRINGMAXLEN_CLICMDARG,
                                   "[uint64 ]  %lu",
                                   data.cmd[cmdi].argdata[argi].val.ui64);
                    break;

                case CLIARG_STR_NOT_IMG:
                    SNPRINTF_CHECK(valuestring,
                                   STRINGMAXLEN_CLICMDARG,
                                   "[ STRnI ]  %s",
                                   data.cmd[cmdi].argdata[argi].val.s);
                    break;

                case CLIARG_IMG:
                    SNPRINTF_CHECK(valuestring,
                                   STRINGMAXLEN_CLICMDARG,
                                   "[ STREAM]  %s",
                                   data.cmd[cmdi].argdata[argi].val.s);
                    break;

                case FPTYPE_FITSFILENAME:
                case FPTYPE_FILENAME:
                case FPTYPE_DIRNAME:
                case FPTYPE_EXECFILENAME:
                    SNPRINTF_CHECK(valuestring,
                                   STRINGMAXLEN_CLICMDARG,
                                   "[ FILE  ]  %s",
                                   data.cmd[cmdi].argdata[argi].val.s);
                    break;

                case FPTYPE_STRING:
                    SNPRINTF_CHECK(valuestring,
                                   STRINGMAXLEN_CLICMDARG,
                                   "[STRING ]  %s",
                                   data.cmd[cmdi].argdata[argi].val.s);
                    break;

                case FPTYPE_PID:
                    SNPRINTF_CHECK(valuestring,
                                   STRINGMAXLEN_CLICMDARG,
                                   "[  PID  ]  %ld",
                                   data.cmd[cmdi].argdata[argi].val.i64);
                    break;

                case FPTYPE_TIMESPEC:
                    SNPRINTF_CHECK(valuestring,
                                   STRINGMAXLEN_CLICMDARG,
                                   "[ TIME  ]  %s",
                                   data.cmd[cmdi].argdata[argi].val.s);
                    break;

                case FPTYPE_PROCESS:
                    SNPRINTF_CHECK(valuestring,
                                   STRINGMAXLEN_CLICMDARG,
                                   "[PROCESS]  %s",
                                   data.cmd[cmdi].argdata[argi].val.s);
                    break;

                case FPTYPE_FPSNAME:
                    SNPRINTF_CHECK(valuestring,
                                   STRINGMAXLEN_CLICMDARG,
                                   "[FPSNAME]  %s",
                                   data.cmd[cmdi].argdata[argi].val.s);
                    break;
                }

                if(data.cmd[cmdi].argdata[argi].fpflag & FPFLAG_PRIMARY_CLI_INPUT)
                {
                    printf("%6d   " COLORARGCLI " %-16s" COLORRESET " %-24s %s\n",
                           CLIargcnt,
                           data.cmd[cmdi].argdata[argi].fpstag,
                           valuestring,
                           data.cmd[cmdi].argdata[argi].descr);
                    CLIargcnt++;
                }
                else
                {
                    printf("[hidden] " COLORARGnotCLI " %-16s" COLORRESET " %-24s %s\n",
                           data.cmd[cmdi].argdata[argi].fpstag,
                           valuestring,
                           data.cmd[cmdi].argdata[argi].descr);
                }
            }

            printf("\n");

            cOK = 1;
        }
    }

    int foundsubstring  = 0;
    int foundregexmatch = 0;
    if(cOK == 0)
    {
        printf("Command \"%s\" does not exist. Partial matches:\n", cmdkey);

        regex_t regex;
        int     reti;
        /* Compile regular expression */
        reti = regcomp(&regex, cmdkey, REG_EXTENDED);
        if(reti)
        {
            fprintf(stderr, "Could not compile regex : \"%s\"\n", cmdkey);
            exit(1);
        }
        int        maxGroups = 8;
        regmatch_t groupArray[maxGroups];

        for(unsigned int cmdi = 0; cmdi < data.NBcmd; cmdi++)
        {

            int matchsubstring = 0;
            // look for substring match

            if(strstr(data.cmd[cmdi].key, cmdkey) != NULL)
            {
                foundsubstring = 1;
                matchsubstring = 1;
                printf(COLORCMD "%s" COLORRESET " in %s [%s]\n\t" COLORINFO
                       "%s\n" COLORRESET,
                       data.cmd[cmdi].key,
                       data.cmd[cmdi].module,
                       data.module[data.cmd[cmdi].moduleindex].shortname,
                       data.cmd[cmdi].info);
            }

            // Regular expression search
            if(matchsubstring == 0)
            {
                // Regular expression search
                reti = regexec(&regex,
                               data.cmd[cmdi].key,
                               maxGroups,
                               groupArray,
                               0);
                if(!reti)
                {
                    foundregexmatch = 1;
                    printf(COLORCMD "%s" COLORRESET " in %s [%s]\n\t" COLORINFO
                           "%s\n" COLORRESET,
                           data.cmd[cmdi].key,
                           data.cmd[cmdi].module,
                           data.module[data.cmd[cmdi].moduleindex].shortname,
                           data.cmd[cmdi].info);

                    char        *cursor = data.cmd[cmdi].key;
                    unsigned int offset = 0;
                    for(int g = 0; g < maxGroups; g++)
                    {
                        if(groupArray[g].rm_so == (regoff_t)((size_t) -1))
                        {
                            break; // No more groups
                        }

                        if(g == 0)
                        {
                            offset = groupArray[g].rm_eo;
                        }

                        char cursorCopy[strlen(cursor) + 1];
                        memcpy(cursorCopy, cursor,
                               strlen(cursor) + 1);
                        cursorCopy[groupArray[g].rm_eo] = 0;
                        /*printf("\t    Match Group %u: [%2u-%2u]: %s\n",
                               g, groupArray[g].rm_so, groupArray[g].rm_eo,
                               cursorCopy + groupArray[g].rm_so);*/
                    }
                    cursor += offset;
                }
                else if(reti == REG_NOMATCH)
                {
                    //puts("No match");
                }
                else
                {
                    char msgbuf[100];
                    regerror(reti, &regex, msgbuf, sizeof(msgbuf));
                    fprintf(stderr, "Regex match failed: %s\n", msgbuf);
                    exit(1);
                }
            }
        }

        regfree(&regex);

        if(foundsubstring == 0)
        {
            if(foundregexmatch == 0)
            {
                printf(
                    "\tNo substring or regex "
                    "match to \"%s\"\n",
                    cmdkey);
            }
        }
        return RETURN_FAILURE;
    }

    return RETURN_SUCCESS;
}


/**
 * @brief search for string in command info
 *
 */

errno_t command_info_search(const char *restrict searchstring)
{
    int foundsubstring  = 0;
    int foundregexmatch = 0;
    int colorcodecmd    = 31; // red
    int colorcodeinfo   = 32; // green

    regex_t regex;
    /* Compile regular expression */
    int reti = regcomp(&regex, searchstring, REG_EXTENDED);
    if(reti)
    {
        fprintf(stderr, "Could not compile regex : \"%s\"\n", searchstring);
        exit(1);
    }
    int        maxGroups = 8;
    regmatch_t groupArray[maxGroups];

    for(unsigned int cmdi = 0; cmdi < data.NBcmd; cmdi++)
    {

        int matchsubstring = 0;
        // look for substring match

        if(strstr(data.cmd[cmdi].info, searchstring) != NULL)
        {
            foundsubstring = 1;
            matchsubstring = 1;
            printf("%c[%d;%dm%s%c[%dm in %s [%s]\n\t%c[%d;%dm%s%c[%dm\n",
                   (char) 27,
                   1,
                   colorcodecmd,
                   data.cmd[cmdi].key,
                   (char) 27,
                   0,
                   data.cmd[cmdi].module,
                   data.module[data.cmd[cmdi].moduleindex].shortname,
                   (char) 27,
                   1,
                   colorcodeinfo,
                   data.cmd[cmdi].info,
                   (char) 27,
                   0);
        }

        // Regular expression search
        if(matchsubstring == 0)
        {
            // Regular expression search
            reti =
                regexec(&regex, data.cmd[cmdi].info, maxGroups, groupArray, 0);
            if(!reti)
            {
                foundregexmatch = 1;

                printf(
                    "%c[%d;%dm%s%c[%dm in %s "
                    "[%s]\n\t%c[%d;%dm%s%c[%dm\n",
                    (char) 27,
                    1,
                    colorcodecmd,
                    data.cmd[cmdi].key,
                    (char) 27,
                    0,
                    data.cmd[cmdi].module,
                    data.module[data.cmd[cmdi].moduleindex].shortname,
                    (char) 27,
                    1,
                    colorcodeinfo,
                    data.cmd[cmdi].info,
                    (char) 27,
                    0);

                char        *cursor = data.cmd[cmdi].info;
                unsigned int offset = 0;
                for(int g = 0; g < maxGroups; g++)
                {
                    if(groupArray[g].rm_so == (regoff_t)((size_t) -1))
                    {
                        break; // No more groups
                    }

                    if(g == 0)
                    {
                        offset = groupArray[g].rm_eo;
                    }

                    char cursorCopy[strlen(cursor) + 1];
                    memcpy(cursorCopy, cursor,
                           strlen(cursor) + 1);
                    cursorCopy[groupArray[g].rm_eo] = 0;
                    /*printf("\t    Match Group %u: [%2u-%2u]: %s\n",
                           g, groupArray[g].rm_so, groupArray[g].rm_eo,
                           cursorCopy + groupArray[g].rm_so);*/
                }
                cursor += offset;
            }
            else if(reti == REG_NOMATCH)
            {
                //puts("No match");
            }
            else
            {
                char msgbuf[100];
                regerror(reti, &regex, msgbuf, sizeof(msgbuf));
                fprintf(stderr, "Regex match failed: %s\n", msgbuf);
                exit(1);
            }
        }
    }

    regfree(&regex);

    if(foundsubstring == 0)
    {
        if(foundregexmatch == 0)
        {
            printf("\tNo substring or regex match to \"%s\"\n", searchstring);
        }
    }

    return RETURN_SUCCESS;
}




/**
 * @brief Top-level 'help' command handler.
 *
 * Dispatches to topic-specific help pages or the
 * general CLI help summary. Supports --json and
 * --porcelain output modes for machine-readable
 * output.
 */
errno_t help()
{
    int json_mode = 0;
    int porcelain_mode = 0;
    const char *topic = NULL;

    for (int arg = 1; arg < data.cmdNBarg; arg++)
    {
        if (data.cmdargtoken[arg].type == CMDARGTOKEN_TYPE_STRING || data.cmdargtoken[arg].type == CMDARGTOKEN_TYPE_RAWSTRING)
        {
            if (strcmp(data.cmdargtoken[arg].val.string, "--json") == 0)
            {
                json_mode = 1;
            }
            else if (strcmp(data.cmdargtoken[arg].val.string, "--porcelain") == 0)
            {
                porcelain_mode = 1;
            }
            else
            {
                topic = data.cmdargtoken[arg].val.string;
            }
        }
    }

    help_format_mode = json_mode ? 1 : (porcelain_mode ? 2 : 0);

    if (topic != NULL)
    {
        if (help_topic_dispatch(topic) != 0)
        {
            if (help_format_mode == 0) {
                printf(C_ERR "Unknown help topic: \"%s\"" C_RST "\n\n", topic);
                print_help_topic_list();
            }
        }
    }
    else
    {
        print_milk_cli_help();
    }

    return RETURN_SUCCESS;
}

/**
 * @brief Display readline keybinding help.
 *
 * Pages through the doc/helpreadline.md file
 * using the system pager.
 */
errno_t helpreadline()
{

    EXECUTE_SYSTEM_COMMAND_NOCHECK(
        "more %s/src/CommandLineInterface/doc/helpreadline.md",
        dcsourcedir);

    return RETURN_SUCCESS;
}

/**
 * @brief Show detailed help for a specific command.
 *
 * If a command name is provided as argument 1,
 * prints its full help (syntax, arguments, flags).
 * If no argument, falls back to list_commands().
 */
errno_t help_cmd()
{
    if((data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_STRING) ||
            (data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_EXISTINGIMAGE) ||
            (data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_COMMAND) ||
            (data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_RAWSTRING))
    {
        help_command(data.cmdargtoken[1].val.string);
    }
    else
    {
        list_commands();
    }

    return RETURN_SUCCESS;
}

/**
 * @brief Search command descriptions for a keyword.
 *
 * If a search string is provided as argument 1,
 * calls command_info_search() to find matching
 * commands. If no argument, lists all commands.
 */
errno_t cmdinfosearch()
{
    if((data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_STRING) ||
            (data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_EXISTINGIMAGE) ||
            (data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_COMMAND) ||
            (data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_RAWSTRING))
    {
        command_info_search(data.cmdargtoken[1].val.string);
    }
    else
    {
        list_commands();
    }

    return RETURN_SUCCESS;
}

/**
 * @brief Show module info or list all loaded modules.
 *
 * If a module name is provided as argument 1,
 * shows that module's commands. Otherwise lists
 * all loaded modules with version and description.
 */
errno_t help_module()
{

    if(data.cmdargtoken[1].type == 3)
    {
        list_commands_module(data.cmdargtoken[1].val.string);
    }
    else
    {
        long i;
        printf("\n");
        printf("%2s  %10s %32s %10s %7s    %20s %s\n",
               "#",
               "shortname",
               "Name",
               "Package",
               "Version",
               "last compiled",
               "description");
        printf(
            "--------------------------------------------------------------"
            "-----------------------------------------"
            "-------\n");
        for(i = 0; i < data.NBmodule; i++)
        {
            printf(
                "%2ld %10s \033[1m%32s\033[0m %10s %2d.%02d.%02d    "
                "%11s %8s  %s\n",
                i,
                data.module[i].shortname,
                data.module[i].name,
                data.module[i].package,
                data.module[i].versionmajor,
                data.module[i].versionminor,
                data.module[i].versionpatch,
                data.module[i].datestring,
                data.module[i].timestring,
                data.module[i].info);
        }
        printf(
            "--------------------------------------------------------------"
            "-----------------------------------------"
            "\n");
        printf("\n");
    }

    return RETURN_SUCCESS;
}


