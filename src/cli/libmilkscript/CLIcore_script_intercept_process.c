#include <stddef.h>
extern int cli_find_in_path(const char *cmd, char *outpath, size_t outsize);
extern int processinfo_procdirname(char *procdirname);
#include <stddef.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include "CLIcore.h"
#include "CLIcore_script.h"
#include "milkscript.h"
#include "CLIcore_utils.h"
#include "CLIcore_memory.h"
#include "CLIcore_modules.h"
#include "CLIcore_UI_execute.h"
#include "CLIcore_checkargs.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <poll.h>

extern int cli_block_level;
extern int cli_break_flag;
extern int CLI_trap_enable;
extern int cli_cmd_delay_us;


int cli_intercept_cmd_return(const char *p)
{
    if(strcmp(p, "return") == 0
       || starts_with(p, "return ")
       || starts_with(p, "return\t"))
    {
        const char *rv = p + 6;
        while(*rv == ' ' || *rv == '\t')
        {
            rv++;
        }
        if(*rv != '\0')
        {
            cli_last_retval =
                (int) strtol(rv, NULL, 0);
        }
        cli_return_flag = 1;
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_exit(const char *p)
{
    if(strcmp(p, "exit") == 0
       || starts_with(p, "exit ")
       || starts_with(p, "exit\t"))
    {
        int exitcode = 0;
        if(strlen(p) > 4)
        {
            const char *ev = p + 4;
            while(*ev == ' ' || *ev == '\t')
            {
                ev++;
            }
            if(*ev != '\0')
            {
                exitcode =
                    (int) strtol(ev,
                                 NULL, 0);
            }
        }
        
        cli_trap_run_exit();
        exit(exitcode);
    }
    return 0;
}

int cli_intercept_cmd_shift(const char *p)
{
    if(strcmp(p, "shift") == 0
       || starts_with(p, "shift ")
       || starts_with(p, "shift\t"))
    {
        int n = 1;
        if(strlen(p) > 5)
        {
            const char *sv = p + 5;
            while(*sv == ' ' || *sv == '\t')
            {
                sv++;
            }
            if(*sv != '\0')
            {
                n = (int) strtol(sv,
                                 NULL, 0);
            }
        }
        if(n < 1)
        {
            n = 1;
        }
        /* Shift $1..$9 by n positions */
        for(int i = 1;
            i < CLI_FUNC_MAXARGS; i++)
        {
            char dst[16], src[16];
            snprintf(dst, sizeof(dst),
                     "%d", i);
            snprintf(src, sizeof(src),
                     "%d", i + n);
            if(i + n < CLI_FUNC_MAXARGS)
            {
                const char *sv2 =
                    cli_var_get(src);
                if(sv2 != NULL)
                {
                    cli_var_set(dst, sv2);
                }
                else
                {
                    cli_var_unset(dst);
                }
            }
            else
            {
                cli_var_unset(dst);
            }
        }
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_procctl(const char *p)
{
    if(starts_with(p, "procctl ")
       || starts_with(p, "procctl\t"))
    {
        const char *ap = p + 7;
        while(*ap == ' ' || *ap == '\t')
        {
            ap++;
        }
        char pname[256];
        int nlen = 0;
        while(*ap && *ap != ' '
              && *ap != '\t'
              && nlen < 255)
        {
            pname[nlen++] = *ap++;
        }
        pname[nlen] = '\0';
        while(*ap == ' ' || *ap == '\t')
        {
            ap++;
        }
        int ctrlval = -1;
        if(strncmp(ap, "run", 3) == 0)
        {
            ctrlval = PROCESSINFO_CTRLVAL_RUN;
        }
        else if(strncmp(ap, "pause", 5) == 0)
        {
            ctrlval = PROCESSINFO_CTRLVAL_PAUSE;
        }
        else if(strncmp(ap, "step", 4) == 0)
        {
            ctrlval = PROCESSINFO_CTRLVAL_INCR;
        }
        else if(strncmp(ap, "stop", 4) == 0
                || strncmp(ap, "exit", 4)
                   == 0)
        {
            ctrlval = PROCESSINFO_CTRLVAL_EXIT;
        }
        if(ctrlval < 0)
        {
            printf(
                "procctl: unknown action "
                "'%s' (use run|pause|"
                "step|stop)\n", ap);
            return 1;
        }
        if(pinfolist != NULL)
        {
            pid_t fpid = 0;
            for(int pi = 0;
                pi < PROCESSINFOLISTSIZE;
                pi++)
            {
                if(pinfolist->active[pi]
                   && strcmp(
                       pinfolist
                           ->pnamearray[pi],
                       pname) == 0)
                {
                    fpid = pinfolist
                        ->PIDarray[pi];
                    break;
                }
            }
            if(fpid > 0)
            {
                char pfn[512];
                char pdname[256];
                processinfo_procdirname(
                    pdname);
                snprintf(pfn, sizeof(pfn),
                         "%s/proc.%d.shm",
                         pdname,
                         (int) fpid);
                int pfd = -1;
                PROCESSINFO *pi =
                    processinfo_shm_link(
                        pfn, &pfd);
                if(pi != MAP_FAILED
                   && pi != NULL)
                {
                    pi->CTRLval = ctrlval;
                    munmap(pi,
                        sizeof(PROCESSINFO));
                    close(pfd);
                }
                else if(pfd >= 0)
                {
                    close(pfd);
                }
            }
            else
            {
                printf(
                    "procctl: process "
                    "'%s' not found\n",
                    pname);
            }
        }
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_procwait(const char *p)
{
    if(starts_with(p, "procwait ")
       || starts_with(p, "procwait\t"))
    {
        const char *ap = p + 8;
        while(*ap == ' ' || *ap == '\t')
        {
            ap++;
        }
        char pname[256];
        int nlen = 0;
        while(*ap && *ap != ' '
              && *ap != '\t'
              && nlen < 255)
        {
            pname[nlen++] = *ap++;
        }
        pname[nlen] = '\0';
        while(*ap == ' ' || *ap == '\t')
        {
            ap++;
        }
        int tgt = -1;
        if(strncasecmp(ap, "INIT", 4) == 0)
        {
            tgt = PROCESSINFO_LOOPSTAT_INIT;
        }
        else if(strncasecmp(ap, "ACTIVE",
                6) == 0)
        {
            tgt = PROCESSINFO_LOOPSTAT_ACTIVE;
        }
        else if(strncasecmp(ap, "PAUSE",
                5) == 0)
        {
            tgt = PROCESSINFO_LOOPSTAT_PAUSE;
        }
        else if(strncasecmp(ap, "STOP",
                4) == 0)
        {
            tgt = PROCESSINFO_LOOPSTAT_STOP;
        }
        else if(strncasecmp(ap, "ERROR",
                5) == 0)
        {
            tgt = PROCESSINFO_LOOPSTAT_ERROR;
        }
        else
        {
            tgt = (int) strtol(ap, NULL, 0);
        }
        /* Skip state word */
        while(*ap && *ap != ' '
              && *ap != '\t')
        {
            ap++;
        }
        while(*ap == ' ' || *ap == '\t')
        {
            ap++;
        }
        double timeout = 30.0;
        if(*ap != '\0')
        {
            timeout = strtod(ap, NULL);
        }
        struct timespec slp;
        slp.tv_sec = 0;
        slp.tv_nsec = 100000000; /* 100ms */
        double elapsed = 0.0;
        cli_last_retval = 1;
        while(elapsed < timeout)
        {
            if(pinfolist != NULL)
            {
                for(int pi = 0;
                    pi < PROCESSINFOLISTSIZE;
                    pi++)
                {
                    if(pinfolist->active[pi]
                       && strcmp(
                           pinfolist
                               ->pnamearray[
                                   pi],
                           pname) == 0)
                    {
                        pid_t fpid =
                            pinfolist
                                ->PIDarray[
                                    pi];
                        char pfn[512];
                        char pdname[256];
                        processinfo_procdirname(
                            pdname);
                        snprintf(
                            pfn,
                            sizeof(pfn),
                            "%s/proc."
                            "%d.shm",
                            pdname,
                            (int) fpid);
                        int pfd = -1;
                        PROCESSINFO *pii =
                            processinfo_shm_link(
                                pfn, &pfd);
                        if(pii
                           != MAP_FAILED
                           && pii != NULL)
                        {
                            if(pii
                               ->loopstat
                               == tgt)
                            {
                                cli_last_retval
                                    = 0;
                            }
                            munmap(pii,
                                sizeof(
                                PROCESSINFO));
                            close(pfd);
                        }
                        else if(pfd >= 0)
                        {
                            close(pfd);
                        }
                        break;
                    }
                }
            }
            if(cli_last_retval == 0)
            {
                break;
            }
            nanosleep(&slp, NULL);
            elapsed += 0.1;
        }
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_procstat(const char *p)
{
    if(strcmp(p, "procstat") == 0
       || starts_with(p, "procstat ")
       || starts_with(p, "procstat\t"))
    {
        const char *ap = p + 8;
        while(*ap == ' ' || *ap == '\t')
        {
            ap++;
        }
        char filter[256];
        filter[0] = '\0';
        if(*ap != '\0')
        {
            strncpy(filter, ap,
                    sizeof(filter) - 1);
            filter[sizeof(filter) - 1]
                = '\0';
        }
        if(pinfolist != NULL)
        {
            char pdname[256];
            processinfo_procdirname(pdname);
            for(int pi = 0;
                pi < PROCESSINFOLISTSIZE;
                pi++)
            {
                if(!pinfolist->active[pi])
                {
                    continue;
                }
                if(filter[0] != '\0'
                   && strcmp(
                       pinfolist
                           ->pnamearray[pi],
                       filter) != 0)
                {
                    continue;
                }
                pid_t fpid =
                    pinfolist
                        ->PIDarray[pi];
                char pfn[512];
                snprintf(pfn,
                         sizeof(pfn),
                         "%s/proc.%d.shm",
                         pdname,
                         (int) fpid);
                int pfd = -1;
                PROCESSINFO *pii =
                    processinfo_shm_link(
                        pfn, &pfd);
                if(pii == MAP_FAILED
                   || pii == NULL)
                {
                    if(pfd >= 0)
                    {
                        close(pfd);
                    }
                    continue;
                }
                const char *stname =
                    "UNKNOWN";
                switch(pii->loopstat)
                {
                    case 0:
                        stname = "INIT";
                        break;
                    case 1:
                        stname = "ACTIVE";
                        break;
                    case 2:
                        stname = "PAUSED";
                        break;
                    case 3:
                        stname = "STOPPED";
                        break;
                    case 4:
                        stname = "ERROR";
                        break;
                    case 5:
                        stname = "SPINNING";
                        break;
                    case 6:
                        stname = "CRASHED";
                        break;
                }
                double hz = 0.0;
                if(pii->dtmedian_iter_ns
                   > 0)
                {
                    hz = 1.0e9
                        / (double)
                          pii
                          ->dtmedian_iter_ns;
                }
                double us =
                    (double)
                    pii->dtmedian_exec_ns
                    / 1000.0;
                printf(
                    "name=%s\n"
                    "pid=%d\n"
                    "loopstat=%s\n"
                    "loopcnt=%ld\n"
                    "loopfreq_hz=%.1f\n"
                    "exectime_us=%.1f\n"
                    "rtprio=%d\n"
                    "ctrlval=%d\n"
                    "missedframes=%lu\n"
                    "tmux=%s\n",
                    pii->name,
                    (int) pii->PID,
                    stname,
                    pii->loopcnt,
                    hz, us,
                    pii->RT_priority,
                    pii->CTRLval,
                    (unsigned long)
                    pii
                    ->triggermissedframe_cumul,
                    pii->tmuxname);
                munmap(pii,
                    sizeof(PROCESSINFO));
                close(pfd);
                if(filter[0] != '\0')
                {
                    break;
                }
                printf("---\n");
            }
        }
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_time(const char *p)
{
    if(starts_with(p, "time ")
       || starts_with(p, "time\t"))
    {
        const char *cmd = p + 4;
        while(*cmd == ' ' || *cmd == '\t')
        {
            cmd++;
        }
        struct timespec t0, t1;
        clock_gettime(
            CLOCK_MONOTONIC, &t0);
        CLI_execute_string(cmd);
        clock_gettime(
            CLOCK_MONOTONIC, &t1);
        double elapsed =
            (double)(t1.tv_sec - t0.tv_sec)
            + (double)(t1.tv_nsec
                       - t0.tv_nsec)
              / 1.0e9;
        printf(
            "\nreal\t%.3fs\n",
            elapsed);
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_assert(const char *p)
{
    if(starts_with(p, "assert ")
       || starts_with(p, "assert\t"))
    {
        const char *ap = p + 6;
        while(*ap == ' ' || *ap == '\t')
        {
            ap++;
        }
        if(*ap == '[')
        {
            ap++;
            const char *end =
                strrchr(ap, ']');
            if(end != NULL)
            {
                char cs[512];
                int clen =
                    (int)(end - ap);
                if(clen
                   >= (int) sizeof(cs))
                {
                    clen =
                        (int) sizeof(cs)
                        - 1;
                }
                memcpy(cs, ap,
                       (size_t) clen);
                cs[clen] = '\0';
                int result =
                    cli_eval_test(cs);
                if(!result)
                {
                    const char *msg =
                        end + 1;
                    while(*msg == ' '
                          || *msg == '\t')
                    {
                        msg++;
                    }
                    /* strip quotes */
                    if(*msg == '"'
                       || *msg == '\'')
                    {
                        msg++;
                    }
                    int mlen =
                        (int) strlen(msg);
                    if(mlen > 0
                       && (msg[mlen - 1]
                           == '"'
                           || msg[mlen - 1]
                              == '\''))
                    {
                        char mb[512];
                        strncpy(
                            mb, msg,
                            sizeof(mb) - 1);
                        mb[sizeof(mb) - 1]
                            = '\0';
                        if(mlen - 1
                           < (int)
                             sizeof(mb))
                        {
                            mb[mlen - 1]
                                = '\0';
                        }
                        printf(
                            "ASSERT "
                            "FAILED: "
                            "%s\n", mb);
                    }
                    else
                    {
                        printf(
                            "ASSERT "
                            "FAILED: "
                            "%s\n", msg);
                    }
                    cli_last_retval = 1;
                    if(cli_flag_errexit)
                    {
                        cli_trap_run(-1);
                    }
                }
            }
            else
            {
                printf(
                    "ERROR: malformed assert: "
                    "missing closing ']'\n");
                cli_last_retval = 1;
            }
        }
        else
        {
            printf(
                "ERROR: malformed assert: "
                "expected '[condition]' after "
                "'assert'\n");
            cli_last_retval = 1;
        }
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_watch(const char *p)
{
    if(starts_with(p, "watch ")
       || starts_with(p, "watch\t"))
    {
        const char *wp = p + 5;
        while(*wp == ' ' || *wp == '\t')
        {
            wp++;
        }
        if(*wp == '-' && *(wp + 1) == 'n')
        {
            double interval = 2.0;
            wp += 2;
            while(*wp == ' '
                  || *wp == '\t')
            {
                wp++;
            }
            interval = strtod(wp, NULL);
            while(*wp != ' '
                  && *wp != '\t'
                  && *wp != '\0')
            {
                wp++;
            }
            while(*wp == ' '
                  || *wp == '\t')
            {
                wp++;
            }
            struct timespec ts;
            ts.tv_sec =
                (time_t) interval;
            ts.tv_nsec =
                (long)((interval
                        - (double) ts.tv_sec)
                       * 1.0e9);
            while(!cli_break_flag)
            {
                printf(
                    "\033[2J\033[H"
                    "Every %.1fs: %s\n\n",
                    interval, wp);
                CLI_execute_string(wp);
                nanosleep(&ts, NULL);
            }
            cli_break_flag = 0;
            return 1;
        }
    }
    return 0;
}

