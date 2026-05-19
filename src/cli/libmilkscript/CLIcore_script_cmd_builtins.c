/**
 * @file CLIcore_script_cmd_builtins.c
 * @brief Built-in shell commands handler
 */

#include <stdio.h>
#include <math.h>
#include <ctype.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <termios.h>
#include <readline/history.h>
#include <readline/readline.h>
#include "CLIcore.h"
#include "CLIcore/cli_calc_parser.h"
#include "CLIcore_script.h"
#include "CLIcore_UI_execute.h"
#include "CLIcore_UI_execute_internal.h"
#include <errno.h>
#include <fnmatch.h>
#include <glob.h>
#include <strings.h>
#include <spawn.h>
#include <sys/wait.h>
#include "COREMOD_memory/COREMOD_memory.h"
#include "timeutils.h"
#include "ImageStreamIO.h"
#include "fps_connect.h"
#include "fps_paramvalue.h"
#define CLICOMPLETIONMODE_COMMANDS 0
#define CLICOMPLETIONMODE_IMAGES   1
#define CLICOMPLETIONMODE_CMDARGS  2
#define CLICOMPLETIONMODE_FILES    3
#define CLICOMPLETIONMODE_FPSPARAMS 4
#define COLORRED       "\001\033[31m\002" /* Red */
#define COLORHBOLDCYAN "\001\e[0;96m\002" /* High Intensity Bold Cyan */
#define COLORDIMYELLOW "\033[2;33m" /* Dim Yellow (no RL wrap) */
#include <wordexp.h>
#define COLORRST       "\033[0m"    /* Reset (no RL wrap) */
#define RL_COLORRESET  "\001\033[0m\002"


/**
 * @brief Dispatch shell built-in commands.
 *
 * Handles cd, echo, printf, and other standard
 * shell builtins within the milk interpreter.
 */
int cli_handle_shell_builtins(void)
{
    if(data.CLIcmdline[0] == '!')
    {
        data.CLIcmdline[0] = ' ';
        printf(COLORDIMYELLOW
               "[shell] %s" COLORRST "\n",
               data.CLIcmdline);
        cli_export_vars_to_env();
        if(cli_run_external(
                    data.CLIcmdline) != 0)
        {
            PRINT_ERROR("shell command error");
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
                    "listim ", 7) == 0)
    {
        /* listim <pattern> — glob filter
         * Only intercept when pattern has
         * wildcard chars (* or ?).
         * Non-wildcard falls through to
         * normal registered command. */
        const char *pat =
            data.CLIcmdline + 7;
        while(*pat == ' ' || *pat == '\t')
        {
            pat++;
        }
        if(strchr(pat, '*') != NULL
                || strchr(pat, '?') != NULL)
        {
            /* Build glob pattern for
             * /dev/shm matching */
            char shmglob[512];
            snprintf(shmglob,
                     sizeof(shmglob),
                     "%s.im.shm", pat);
            DIR *dp = opendir("/dev/shm");
            if(dp != NULL)
            {
                struct dirent *de;
                int count = 0;
                while((de = readdir(dp))
                        != NULL)
                {
                    if(fnmatch(
                                shmglob,
                                de->d_name,
                                0) == 0)
                    {
                        /* Strip .im.shm
                         * suffix */
                        char nm[256];
                        strncpy(nm,
                                de->d_name,
                                sizeof(nm)
                                - 1);
                        nm[sizeof(nm) - 1]
                            = '\0';
                        char *sfx =
                            strstr(nm,
                                   ".im.shm");
                        if(sfx != NULL)
                        {
                            *sfx = '\0';
                        }
                        printf("  %s\n", nm);
                        count++;
                    }
                }
                closedir(dp);
                printf("%d stream(s) "
                       "matched\n",
                       count);
            }
            data.CMDexecuted = 1;
        }
        /* else: no wildcard, fall through
         * to normal listim command */
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
                    "printf ", 7) == 0)
    {
        /* Intercept printf before
         * tokenization so % and
         * backslash are preserved */
        const char *raw =
            data.CLIcmdline + 7;
        while(*raw == ' ')
        {
            raw++;
        }

        /* Tokenize manually: split on
         * spaces, respecting quotes */
        data.cmdNBarg = 1;
        strncpy(
            data.cmdargtoken[0].val.string,
            "printf",
            STRINGMAXLEN_CMDARGTOKEN_VAL - 1);

        const char *s = raw;
        while(*s != '\0'
                && data.cmdNBarg
                < NB_ARG_MAX)
        {
            while(*s == ' ')
            {
                s++;
            }
            if(*s == '\0')
            {
                break;
            }
            int ai = 0;
            if(*s == '"')
            {
                s++;
                while(*s != '\0'
                        && *s != '"'
                        && ai
                        < STRINGMAXLEN_CMDARGTOKEN_VAL
                        - 1)
                {
                    data.cmdargtoken[
                        data.cmdNBarg]
                    .val.string[ai++]
                        = *s++;
                }
                if(*s == '"')
                {
                    s++;
                }
            }
            else
            {
                while(*s != '\0'
                        && *s != ' '
                        && ai
                        < STRINGMAXLEN_CMDARGTOKEN_VAL
                        - 1)
                {
                    data.cmdargtoken[
                        data.cmdNBarg]
                    .val.string[ai++]
                        = *s++;
                }
            }
            data.cmdargtoken[
            data.cmdNBarg]
            .val.string[ai] = '\0';
            data.cmdNBarg++;
        }

        cli_cmd_printf();
        data.CMDexecuted = 1;
    }
    else if(strncmp(data.CLIcmdline,
                    "export ", 7) == 0
            || strcmp(data.CLIcmdline,
                      "export") == 0)
    {
        /* Intercept export before
         * tokenization so = in
         * VAR=value is preserved */
        const char *raw =
            data.CLIcmdline + 6;
        while(*raw == ' ')
        {
            raw++;
        }

        data.cmdNBarg = 1;
        strncpy(
            data.cmdargtoken[0].val.string,
            "export",
            STRINGMAXLEN_CMDARGTOKEN_VAL - 1);

        if(*raw != '\0')
        {
            int ai = 0;
            while(*raw != '\0'
                    && *raw != ' '
                    && ai
                    < STRINGMAXLEN_CMDARGTOKEN_VAL
                    - 1)
            {
                data.cmdargtoken[1]
                .val.string[ai++]
                    = *raw++;
            }
            data.cmdargtoken[1]
            .val.string[ai] = '\0';
            data.cmdNBarg = 2;
        }

        cli_cmd_export();
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
                    "on_update ", 10) == 0 ||
            strcmp(data.CLIcmdline,
                   "on_update") == 0)
    {
        /* on_update [-l] [-n N] <stream> { cmd }
         * Wait for stream semaphore,
         * then execute cmd.
         * -l: loop forever
         * -n N: loop N times */
        const char *arg = data.CLIcmdline;
        if(strncmp(data.CLIcmdline, "on_update ", 10) == 0)
        {
            arg += 10;
        }
        else
        {
            arg += 9;
        }
        while(*arg == ' ' || *arg == '\t')
        {
            arg++;
        }

        /* Parse flags */
        int loop_count = 1; /* default: once */
        while(*arg == '-')
        {
            if(strncmp(arg, "-l", 2) == 0
                    && (arg[2] == ' '
                        || arg[2] == '\t'
                        || arg[2] == '\0'))
            {
                loop_count = -1;
                arg += 2;
            }
            else if(strncmp(arg, "-n", 2)
                    == 0)
            {
                arg += 2;
                while(*arg == ' '
                        || *arg == '\t')
                {
                    arg++;
                }
                char *endptr = NULL;
                long nval = strtol(arg, &endptr, 10);
                if(endptr == arg || nval <= 0)
                {
                    fprintf(stderr,
                            "Invalid value for -n option: '%s' (expected positive integer)\n",
                            arg);
                    /* Treat invalid/zero as a no-op for loop_count */
                }
                else
                {
                    loop_count = (int) nval;
                    arg = endptr;
                }
            }
            else
            {
                /* Unknown flag — stop parsing */
                break;
            }
            while(*arg == ' '
                    || *arg == '\t')
            {
                arg++;
            }
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
                   "[-l] [-n N] "
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
                    /* Create processinfo for
                     * loop mode */
                    PROCESSINFO *procinfo =
                        NULL;
                    int is_loop =
                        (loop_count != 1);
                    if(is_loop)
                    {
                        char pname[64];
                        snprintf(pname,
                                 sizeof(pname),
                                 "on_update_%s",
                                 sname);
                        procinfo =
                            processinfo_shm_create(
                                pname,
                                PROCESSINFO_CTRLVAL_RUN);
                        if(procinfo == NULL)
                        {
                            PRINT_WARNING(
                                "processinfo_shm_create(%s) "
                                "failed; continuing without "
                                "process tracking",
                                pname);
                        }
                    }

                    int iter = 0;
                    int keep_going = 1;
                    while(keep_going
                            && !cli_break_flag)
                    {
                        /* Check procctl */
                        if(procinfo != NULL)
                        {
                            if(procinfo->CTRLval
                                    == PROCESSINFO_CTRLVAL_EXIT)
                            {
                                break;
                            }
                            while(procinfo
                                    ->CTRLval
                                    == PROCESSINFO_CTRLVAL_PAUSE)
                            {
                                usleep(10000);
                                if(cli_break_flag)
                                {
                                    break;
                                }
                            }
                        }

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

                        iter++;
                        if(procinfo != NULL)
                        {
                            procinfo->loopcnt
                                = iter;
                        }
                        if(loop_count > 0
                                && iter
                                >= loop_count)
                        {
                            keep_going = 0;
                        }
                    }
                    cli_break_flag = 0;

                    if(procinfo != NULL)
                    {
                        processinfo_cleanExit(
                            procinfo);
                    }
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
                    "on_fpschange ",
                    13) == 0)
    {
        /* on_fpschange [-l] [-n N]
         *   fpsname.param { cmd }
         * Poll FPS parameter, execute body
         * when it changes. */
        const char *arg =
            data.CLIcmdline + 13;
        while(*arg == ' '
                || *arg == '\t')
        {
            arg++;
        }

        /* Parse flags */
        int loop_count = 1;
        while(*arg == '-')
        {
            if(strncmp(arg, "-l", 2) == 0
                    && (arg[2] == ' '
                        || arg[2] == '\t'
                        || arg[2] == '\0'))
            {
                loop_count = -1;
                arg += 2;
            }
            else if(strncmp(arg, "-n", 2)
                    == 0)
            {
                arg += 2;
                while(*arg == ' '
                        || *arg == '\t')
                {
                    arg++;
                }
                char *endptr = NULL;
                long nval = strtol(
                                arg, &endptr, 10);
                if(endptr == arg
                        || nval <= 0)
                {
                    fprintf(stderr,
                            "Invalid value for"
                            " -n option: '%s'"
                            " (expected positive"
                            " integer)\n",
                            arg);
                }
                else
                {
                    loop_count = (int) nval;
                    arg = endptr;
                }
            }
            else
            {
                /* Unknown flag — stop parsing */
                break;
            }
            while(*arg == ' '
                    || *arg == '\t')
            {
                arg++;
            }
        }

        /* Extract fpsname.param */
        char fparg[256];
        {
            int ai = 0;
            while(*arg != '\0'
                    && *arg != ' '
                    && *arg != '\t'
                    && ai < 255)
            {
                fparg[ai++] = *arg++;
            }
            fparg[ai] = '\0';
        }
        /* Split at dot */
        char *dot = strchr(fparg, '.');
        if(dot == NULL)
        {
            printf(
                "on_fpschange: "
                "use fpsname.param\n");
            cli_last_retval = 1;
            data.CMDexecuted = 1;
        }
        else
        {
            *dot = '\0';
            const char *fpsn = fparg;
            const char *parn = dot + 1;
            /* Extract body between { } */
            while(*arg == ' '
                    || *arg == '\t')
            {
                arg++;
            }
            char body[
                STRINGMAXLEN_CLICMDLINE];
            body[0] = '\0';
            if(*arg == '{')
            {
                arg++;
                while(*arg == ' '
                        || *arg == '\t')
                {
                    arg++;
                }
                int bi = 0;
                while(*arg != '\0'
                        && *arg != '}'
                        && bi
                        < STRINGMAXLEN_CLICMDLINE
                        - 1)
                {
                    body[bi++] = *arg++;
                }
                body[bi] = '\0';
                /* trim trailing spaces */
                while(bi > 0
                        && (body[bi - 1]
                            == ' '
                            || body[bi - 1]
                            == '\t'))
                {
                    body[--bi] = '\0';
                }
            }
            /* Connect to FPS */
            FPS fps;
            if(
                fps_connect(
                    fpsn, &fps,
                    FPSCONNECT_SIMPLE)
                != EXIT_SUCCESS)
            {
                printf(
                    "on_fpschange: "
                    "cannot connect "
                    "to fps '%s'\n",
                    fpsn);
                cli_last_retval = 1;
            }
            else
            {
                long pidx =
                    functionparameter_GetParamIndex(
                        &fps, parn);
                if(pidx < 0)
                {
                    printf(
                        "on_fpschange: "
                        "param '%s' not "
                        "found\n", parn);
                    cli_last_retval = 1;
                }
                else
                {
                    /* Create processinfo
                     * for loop mode */
                    PROCESSINFO *procinfo =
                        NULL;
                    int is_loop =
                        (loop_count != 1);
                    if(is_loop)
                    {
                        char pname[64];
                        snprintf(pname,
                                 sizeof(pname),
                                 "on_fpschg_%s",
                                 fpsn);
                        procinfo =
                            processinfo_shm_create(
                                pname,
                                PROCESSINFO_CTRLVAL_RUN);
                        if(procinfo == NULL)
                        {
                            PRINT_WARNING(
                                "processinfo_shm_create(%s) "
                                "failed; continuing without "
                                "process tracking",
                                pname);
                        }
                    }

                    char prev[256];
                    functionparameter_GetParamValueString(
                        &fps.parray[pidx],
                        prev,
                        sizeof(prev));

                    int iter = 0;
                    int keep_going = 1;
                    while(keep_going
                            && !cli_break_flag)
                    {
                        /* Check procctl */
                        if(procinfo != NULL)
                        {
                            if(procinfo
                                    ->CTRLval
                                    == PROCESSINFO_CTRLVAL_EXIT)
                            {
                                break;
                            }
                            while(procinfo
                                    ->CTRLval
                                    == PROCESSINFO_CTRLVAL_PAUSE)
                            {
                                usleep(10000);
                                if(cli_break_flag)
                                {
                                    break;
                                }
                            }
                        }

                        /* Poll for change */
                        char cur[256];
                        for(;;)
                        {
                            usleep(100000);
                            if(cli_break_flag)
                            {
                                break;
                            }
                            if(procinfo
                                    != NULL
                                    && procinfo
                                    ->CTRLval
                                    != PROCESSINFO_CTRLVAL_RUN)
                            {
                                break;
                            }
                            functionparameter_GetParamValueString(
                                &fps
                                .parray[pidx],
                                cur,
                                sizeof(cur));
                            if(strcmp(cur,
                                      prev)
                                    != 0)
                            {
                                break;
                            }
                        }
                        if(cli_break_flag)
                        {
                            break;
                        }
                        if(procinfo != NULL
                                && procinfo
                                ->CTRLval
                                != PROCESSINFO_CTRLVAL_RUN)
                        {
                            break;
                        }

                        /* Execute body */
                        strncpy(prev, cur,
                                sizeof(prev)
                                - 1);
                        prev[sizeof(prev) - 1] = '\0';
                        strncpy(
                            data.CLIcmdline,
                            body,
                            STRINGMAXLEN_CLICMDLINE
                            - 1);
                        data.CLIcmdline[
                            STRINGMAXLEN_CLICMDLINE
                            - 1] = '\0';
                        CLI_execute_line();

                        iter++;
                        if(procinfo != NULL)
                        {
                            procinfo
                            ->loopcnt
                                = iter;
                        }
                        if(loop_count > 0
                                && iter
                                >= loop_count)
                        {
                            keep_going = 0;
                        }
                    }
                    cli_break_flag = 0;

                    if(procinfo != NULL)
                    {
                        processinfo_cleanExit(
                            procinfo);
                    }
                }
                fps_disconnect(
                    &fps);
            }
            data.CMDexecuted = 1;
        }
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
    else if(0) /* duplicate printf handler removed: printf is already handled earlier */
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
        return 0; // Not matched
    }
    return 1;
}
