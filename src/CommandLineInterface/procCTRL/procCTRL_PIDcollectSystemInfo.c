#include <dirent.h>

#include "CLIcore.h"
#include <processtools.h>

#include "CommandLineInterface/timeutils.h"



// timing info collected to optimize this program
static struct timespec t1;
static struct timespec t2;
static struct timespec tdiff;


static double scantime_cpuset;
static double scantime_pstree;



// for Display Modes 2 and 3
//

int PIDcollectSystemInfo(PROCESSINFODISP *pinfodisp, int level)
{

    // COLLECT INFO FROM SYSTEM
    FILE *fp;
    char  fname[STRINGMAXLEN_FULLFILENAME];

    DEBUG_TRACEPOINT(" ");

    // cpuset

    int PID = pinfodisp->PID;

    DEBUG_TRACEPOINT(" ");

    clock_gettime(CLOCK_REALTIME, &t1);

    WRITE_FULLFILENAME(fname, "/proc/%d/task/%d/cpuset", PID, PID);

    fp = fopen(fname, "r");
    if(fp == NULL)
    {
        return -1;
    }
    if(fscanf(fp, "%s", pinfodisp->cpuset) != 1)
    {
        PRINT_ERROR("fscanf returns value != 1");
    }
    fclose(fp);
    clock_gettime(CLOCK_REALTIME, &t2);
    tdiff = timespec_diff(t1, t2);
    scantime_cpuset += 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

    //char   *line = NULL;
    //size_t  len  = 0;
    //ssize_t read;
    //char    string0[200];
    //char    string1[300];

    DEBUG_TRACEPOINT(" ");

    clock_gettime(CLOCK_REALTIME, &t1);
    if(level == 0)
    {
        //FILE *fpout;
        //char command[200];
        //char outstring[200];

        pinfodisp->subprocPIDarray[0] = PID;
        pinfodisp->NBsubprocesses     = 1;

        // if(pinfodisp->threads > 1) // look for children
        // {
        DIR           *dp;
        struct dirent *ep;
        char           dirname[STRINGMAXLEN_FULLFILENAME];

        // fprintf(stderr, "reading /proc/%d/task\n", PID);
        WRITE_FULLFILENAME(dirname, "/proc/%d/task/", PID);
        //sprintf(dirname, "/proc/%d/task/", PID);
        dp = opendir(dirname);

        if(dp != NULL)
        {
            while((ep = readdir(dp)))
            {
                if(ep->d_name[0] != '.')
                {
                    int subPID = atoi(ep->d_name);
                    if(subPID != PID)
                    {
                        pinfodisp->subprocPIDarray[pinfodisp->NBsubprocesses] =
                            atoi(ep->d_name);
                        pinfodisp->NBsubprocesses++;
                    }
                }
            }
            closedir(dp);
        }
        else
        {
            return -1;
        }
        // }
        // fprintf(stderr, "%d threads found\n", pinfodisp->NBsubprocesses);
        pinfodisp->threads = pinfodisp->NBsubprocesses;
    }
    clock_gettime(CLOCK_REALTIME, &t2);
    tdiff = timespec_diff(t1, t2);
    scantime_pstree += 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

    // read /proc/PID/status
#ifdef CMDPROC_PROCSTAT
    for(int spindex = 0; spindex < pinfodisp->NBsubprocesses; spindex++)
    {
        clock_gettime(CLOCK_REALTIME, &t1);
        PID = pinfodisp->subprocPIDarray[spindex];

        WRITE_FULLFILENAME(fname, "/proc/%d/status", PID);
        fp = fopen(fname, "r");
        if(fp == NULL)
        {
            return -1;
        }

        while((read = getline(&line, &len, fp)) != -1)
        {
            if(sscanf(line, "%31[^:]: %s", string0, string1) == 2)
            {
                if(spindex == 0)
                {
                    if(strcmp(string0, "Cpus_allowed_list") == 0)
                    {
                        strcpy(pinfodisp->cpusallowed, string1);
                    }

                    if(strcmp(string0, "Threads") == 0)
                    {
                        pinfodisp->threads = atoi(string1);
                    }
                }

                if(strcmp(string0, "VmRSS") == 0)
                {
                    pinfodisp->VmRSSarray[spindex] = atol(string1);
                }

                if(strcmp(string0, "nonvoluntary_ctxt_switches") == 0)
                {
                    pinfodisp->ctxtsw_nonvoluntary[spindex] = atoi(string1);
                }
                if(strcmp(string0, "voluntary_ctxt_switches") == 0)
                {
                    pinfodisp->ctxtsw_voluntary[spindex] = atoi(string1);
                }
            }
        }

        fclose(fp);
        if(line)
        {
            free(line);
        }
        line = NULL;
        len  = 0;

        clock_gettime(CLOCK_REALTIME, &t2);
        tdiff = timespec_diff(t1, t2);
        scantime_status += 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

        // read /proc/PID/stat
        clock_gettime(CLOCK_REALTIME, &t1);
        WRITE_FULLFILENAME(fname, "/proc/%d/stat", PID);

        int  stat_pid; // (1) The process ID.
        char stat_comm
        [20];        // (2) The filename of the executable, in parentheses.
        char stat_state; // (3)
        /* One of the following characters, indicating process state:
                            R  Running
                            S  Sleeping in an interruptible wait
                            D  Waiting in uninterruptible disk sleep
                            Z  Zombie
                            T  Stopped (on a signal) or (before Linux 2.6.33)
                            trace stopped
                            t  Tracing stop (Linux 2.6.33 onward)
                            W  Paging (only before Linux 2.6.0)
                            X  Dead (from Linux 2.6.0 onward)
                            x  Dead (Linux 2.6.33 to 3.13 only)
                            K  Wakekill (Linux 2.6.33 to 3.13 only)
                            W  Waking (Linux 2.6.33 to 3.13 only)
                            P  Parked (Linux 3.9 to 3.13 only)
                    */
        int          stat_ppid;    // (4) The PID of the parent of this process.
        int          stat_pgrp;    // (5) The process group ID of the process
        int          stat_session; // (6) The session ID of the process
        int          stat_tty_nr; // (7) The controlling terminal of the process
        int
        stat_tpgid; // (8) The ID of the foreground process group of the controlling terminal of the process
        unsigned int stat_flags; // (9) The kernel flags word of the process
        unsigned long
        stat_minflt; // (10) The number of minor faults the process has made which have not required loading a memory page from disk
        unsigned long
        stat_cminflt; // (11) The number of minor faults that the process's waited-for children have made
        unsigned long
        stat_majflt; // (12) The number of major faults the process has made which have required loading a memory page from disk
        unsigned long
        stat_cmajflt; // (13) The number of major faults that the process's waited-for children have made
        unsigned long
        stat_utime; // (14) Amount of time that this process has been scheduled in user mode, measured in clock ticks (divide by sysconf(_SC_CLK_TCK)).
        unsigned long
        stat_stime; // (15) Amount of time that this process has been scheduled in kernel mode, measured in clock ticks
        long
        stat_cutime; // (16) Amount of time that this process's waited-for children have been scheduled in user mode, measured in clock ticks
        long
        stat_cstime; // (17) Amount of time that this process's waited-for children have been scheduled in kernel mode, measured in clock ticks
        long
        stat_priority; // (18) (Explanation for Linux 2.6) For processes running a
        /*                  real-time scheduling policy (policy below; see
                            sched_setscheduler(2)), this is the negated schedul‐
                            ing priority, minus one; that is, a number in the
                            range -2 to -100, corresponding to real-time priori‐
                            ties 1 to 99.  For processes running under a non-
                            real-time scheduling policy, this is the raw nice
                            value (setpriority(2)) as represented in the kernel.
                            The kernel stores nice values as numbers in the
                            range 0 (high) to 39 (low), corresponding to the
                            user-visible nice range of -20 to 19.

                            Before Linux 2.6, this was a scaled value based on
                            the scheduler weighting given to this process.*/
        long
        stat_nice; // (19) The nice value (see setpriority(2)), a value in the range 19 (low priority) to -20 (high priority)
        long stat_num_threads; // (20) Number of threads in this process
        long stat_itrealvalue; // (21) hard coded as 0
        unsigned long long
        stat_starttime; // (22) The time the process started after system boot in clock ticks
        unsigned long stat_vsize; // (23)  Virtual memory size in bytes
        long
        stat_rss; // (24) Resident Set Size: number of pages the process has in real memory
        unsigned long
        stat_rsslim; // (25) Current soft limit in bytes on the rss of the process
        unsigned long
        stat_startcode; // (26) The address above which program text can run
        unsigned long
        stat_endcode; // (27) The address below which program text can run
        unsigned long
        stat_startstack; // (28) The address of the start (i.e., bottom) of the stack
        unsigned long
        stat_kstkesp; // (29) The current value of ESP (stack pointer), as found in the kernel stack page for the process
        unsigned long
        stat_kstkeip; // (30) The current EIP (instruction pointer)
        unsigned long
        stat_signal; // (31) The bitmap of pending signals, displayed as a decimal number.  Obsolete, because it does not provide information on real-time signals; use /proc/[pid]/status instead
        unsigned long
        stat_blocked; // (32) The bitmap of blocked signals, displayed as a decimal number.  Obsolete, because it does not provide information on real-time signals; use /proc/[pid]/status instead.
        unsigned long
        stat_sigignore; // (33) The bitmap of ignored signals, displayed as a decimal number.  Obsolete, because it does not provide information on real-time signals; use /proc/[pid]/status instead.
        unsigned long
        stat_sigcatch; // (34) The bitmap of ignored signals, displayed as a decimal number.  Obsolete, because it does not provide information on real-time signals; use /proc/[pid]/status instead.
        unsigned long
        stat_wchan; // (35) This is the "channel" in which the process is waiting.  It is the address of a location in the kernel where the process is sleeping.  The corresponding symbolic name can be found in /proc/[pid]/wchan.
        unsigned long
        stat_nswap; // (36) Number of pages swapped (not maintained)
        unsigned long
        stat_cnswap; // (37) Cumulative nswap for child processes (not maintained)
        int stat_exit_signal; // (38) Signal to be sent to parent when we die
        int stat_processor;   // (39) CPU number last executed on
        unsigned int
        stat_rt_priority; // (40) Real-time scheduling priority, a number in the range 1 to 99 for processes scheduled under a real-time policy, or 0, for non-real-time processes (see  sched_setscheduler(2)).
        unsigned int
        stat_policy; // (41) Scheduling policy (see sched_setscheduler(2))
        unsigned long long
        stat_delayacct_blkio_ticks; // (42) Aggregated block I/O delays, measured in clock ticks
        unsigned long
        stat_guest_time; // (43) Guest time of the process (time spent running a virtual CPU for a guest operating system), measured in clock ticks
        long
        stat_cguest_time; // (44) Guest time of the process's children, measured in clock ticks (divide by sysconf(_SC_CLK_TCK)).
        unsigned long
        stat_start_data; // (45) Address above which program initialized and uninitialized (BSS) data are placed
        unsigned long
        stat_end_data; // (46) ddress below which program initialized and uninitialized (BSS) data are placed
        unsigned long
        stat_start_brk; // (47) Address above which program heap can be expanded with brk(2)
        unsigned long
        stat_arg_start; // (48) Address above which program command-line arguments (argv) are placed
        unsigned long
        stat_arg_end; // (49) Address below program command-line arguments (argv) are placed
        unsigned long
        stat_env_start; // (50) Address above which program environment is placed
        unsigned long
        stat_env_end; // (51) Address below which program environment is placed
        long
        stat_exit_code; // (52) The thread's exit status in the form reported by waitpid(2)

        fp = fopen(fname, "r");
        int Nfields;
        if(fp == NULL)
        {
            return -1;
        }

        Nfields =
            fscanf(fp,
                   "%d %s %c %d %d %d %d %d %u %lu %lu %lu %lu %lu %lu %ld "
                   "%ld %ld %ld %ld %ld %llu %lu %ld %lu %lu %lu %lu "
                   "%lu %lu %lu %lu %lu %lu %lu %lu %lu %d %d %u %u %llu "
                   "%lu %ld %lu %lu %lu %lu %lu %lu %lu %ld\n",
                   &stat_pid, //  1
                   stat_comm,
                   &stat_state,
                   &stat_ppid,
                   &stat_pgrp,
                   &stat_session,
                   &stat_tty_nr,
                   &stat_tpgid,
                   &stat_flags,
                   &stat_minflt, //  10
                   &stat_cminflt,
                   &stat_majflt,
                   &stat_cmajflt,
                   &stat_utime,
                   &stat_stime,
                   &stat_cutime,
                   &stat_cstime,
                   &stat_priority,
                   &stat_nice,
                   &stat_num_threads, // 20
                   &stat_itrealvalue,
                   &stat_starttime,
                   &stat_vsize,
                   &stat_rss,
                   &stat_rsslim,
                   &stat_startcode,
                   &stat_endcode,
                   &stat_startstack,
                   &stat_kstkesp,
                   &stat_kstkeip, // 30
                   &stat_signal,
                   &stat_blocked,
                   &stat_sigignore,
                   &stat_sigcatch,
                   &stat_wchan,
                   &stat_nswap,
                   &stat_cnswap,
                   &stat_exit_signal,
                   &stat_processor,
                   &stat_rt_priority, // 40
                   &stat_policy,
                   &stat_delayacct_blkio_ticks,
                   &stat_guest_time,
                   &stat_cguest_time,
                   &stat_start_data,
                   &stat_end_data,
                   &stat_start_brk,
                   &stat_arg_start,
                   &stat_arg_end,
                   &stat_env_start, // 50
                   &stat_env_end,
                   &stat_exit_code);
        if(Nfields != 52)
        {
            PRINT_ERROR("fscanf returns value != 1");
            pinfodisp->processorarray[spindex] = stat_processor;
            pinfodisp->rt_priority             = stat_rt_priority;
        }
        else
        {
            pinfodisp->processorarray[spindex] = stat_processor;
            pinfodisp->rt_priority             = stat_rt_priority;
        }
        fclose(fp);

        pinfodisp->sampletimearray[spindex] =
            1.0 * t1.tv_sec + 1.0e-9 * t1.tv_nsec;

        pinfodisp->cpuloadcntarray[spindex] = (stat_utime + stat_stime);
        pinfodisp->memload                  = 0.0;

        clock_gettime(CLOCK_REALTIME, &t2);
        tdiff = timespec_diff(t1, t2);
        scantime_stat += 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
    }
#endif

    DEBUG_TRACEPOINT(" ");

    return 0;
}
