// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CLIcore_signals.c
 *
 * @brief signals and debugging
 *
 */

#include <stdarg.h>
#include <sys/resource.h> // getrlimit
#include <sys/stat.h>
#include <termios.h>

#include "CLIcore.h"

#include "CLIcore_UI_execute.h"

#include "timeutils.h"

/**
 * @brief Write entry into debug log
 *
 *
 */
errno_t write_process_log()
{
    static FILE *fplog;
    static long  logcnt = 0;
    char         fname[STRINGMAXLEN_FILENAME];
    pid_t        thisPID;

    thisPID = getpid();
    WRITE_FILENAME(fname, "logreport.%05d.log", thisPID);

    fplog = fopen(fname, "a");

    if (fplog != NULL)
    {
        char timestring[TIMESTRINGLEN];
        mkUTtimestring_nanosec(timestring, dctestpoint.time);

        fprintf(fplog, "%18ld  %s ", logcnt, timestring);

        {
            // extract last word
            char str[STRINGMAXLEN_FULLFILENAME];
            snprintf(str, sizeof(str), "%s", dctestpoint.file);
            char *lastword = strrchr(str, '/') + 1;
            fprintf(fplog, " %s", lastword);
        }

        fprintf(fplog, " %4d", dctestpoint.line);
        fprintf(fplog, " %s", dctestpoint.func);
        fprintf(fplog, "  %s\n", dctestpoint.msg);

        logcnt++;

        fclose(fplog);
    }

    return RETURN_SUCCESS;
}

/**
 * @brief Restore terminal echo mode.
 *
 * Re-enables ECHO on stdin. Called during signal
 * handling (crash/exit) to leave the terminal in
 * a usable state.
 */
static void set_terminal_echo_on()
{
    // Terminal settings
    struct termios termInfo;
    if (tcgetattr(0, &termInfo) == -1)
    {
        PRINT_ERROR("tcgetattr: %s", strerror(errno));
        exit(1);
    }
    termInfo.c_lflag |= ECHO; /* turn on ECHO */
    tcsetattr(0, TCSADRAIN, &termInfo);
}

/**
 * @brief Write formatted output to both stdout
 *        and a file stream.
 *
 * Used by write_process_exit_report() to produce
 * the crash report on disk while simultaneously
 * printing it to the terminal.
 */
static void fprintf_stdout(FILE *f, char const *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    va_start(ap, fmt);
    vfprintf(f, fmt, ap);
    va_end(ap);
}

/** @brief signal catching
 *
 */
errno_t set_signal_catch()
{
    // catch signals for clean exit
    if (sigaction(SIGTERM, &dcsigact, NULL) == -1)
    {
        printf("\ncan't catch SIGTERM\n");
    }

    if (sigaction(SIGINT, &dcsigact, NULL) == -1)
    {
        printf("\ncan't catch SIGINT\n");
    }

    if (sigaction(SIGABRT, &dcsigact, NULL) == -1)
    {
        printf("\ncan't catch SIGABRT\n");
    }

    if (sigaction(SIGBUS, &dcsigact, NULL) == -1)
    {
        printf("\ncan't catch SIGBUS\n");
    }

    if (sigaction(SIGSEGV, &dcsigact, NULL) == -1)
    {
        printf("\ncan't catch SIGSEGV\n");
    }

    if (sigaction(SIGHUP, &dcsigact, NULL) == -1)
    {
        printf("\ncan't catch SIGHUP\n");
    }

    if (sigaction(SIGPIPE, &dcsigact, NULL) == -1)
    {
        printf("\ncan't catch SIGPIPE\n");
    }

    return RETURN_SUCCESS;
}

/**
 * @brief Write to disk a process report
 *
 * This function is typically called upon crash to help debugging
 *
 * errortypestring describes the type of error or reason to issue report
 *
 */
/**
 * @brief Write a crash/exit report to disk.
 *
 * Dumps the last tracepoint, function call stack,
 * open file descriptors, and timing info into
 * exitreport-<reason>.<PID>.log.
 *
 * Typically called from the signal handler before
 * exit() so the developer has a post-mortem trail.
 *
 * @param errortypestring  Reason string (e.g.
 *        "SIGSEGV", "SIGBUS")
 */
errno_t write_process_exit_report(const char *__restrict errortypestring)
{
#ifndef NDEBUG
    FILE *fpexit;
    char  fname[STRINGMAXLEN_FILENAME];
    pid_t thisPID;
    long  fd_counter = 0;

    thisPID = getpid();

    WRITE_FILENAME(fname, "exitreport-%s.%05d.log", errortypestring, thisPID);

    printf("EXIT CONDITION < %s >: See report in file %s\n", errortypestring, fname);
    printf("    File    : %s\n", dctestpoint.file);
    printf("    Function: %s\n", dctestpoint.func);
    printf("    Line    : %d\n", dctestpoint.line);
    printf("    Message : %s\n", dctestpoint.msg);
    fflush(stdout);

    struct tm *uttime;
    time_t     tvsec0, tvsec1;

    fpexit = fopen(fname, "w");
    if (fpexit != NULL)
    {
        fprintf_stdout(fpexit, "PID : %d\n", thisPID);

        struct timespec tnow;
        //        time_t now;
        clock_gettime(CLOCK_MILK, &tnow);
        tvsec0 = tnow.tv_sec;
        uttime = gmtime(&tvsec0);
        fprintf_stdout(fpexit, "Time: %04d%02d%02dT%02d%02d%02d.%09ld\n\n", 1900 + uttime->tm_year,
                       1 + uttime->tm_mon, uttime->tm_mday, uttime->tm_hour, uttime->tm_min,
                       uttime->tm_sec, tnow.tv_nsec);

        fprintf_stdout(fpexit, "Last encountered test point\n");
        tvsec1 = dctestpoint.time.tv_sec;
        uttime = gmtime(&tvsec1);
        fprintf_stdout(fpexit, "    Time    : %04d%02d%02dT%02d%02d%02d.%09ld\n",
                       1900 + uttime->tm_year, 1 + uttime->tm_mon, uttime->tm_mday, uttime->tm_hour,
                       uttime->tm_min, uttime->tm_sec, dctestpoint.time.tv_nsec);

        double timediff =
            1.0 * (tvsec0 - tvsec1) + 1.0e-9 * (tnow.tv_nsec - dctestpoint.time.tv_nsec);
        fprintf_stdout(fpexit, "              %.9f sec ago\n", timediff);

        fprintf_stdout(fpexit, "    File    : %s\n", dctestpoint.file);
        fprintf_stdout(fpexit, "    Function: %s\n", dctestpoint.func);
        fprintf_stdout(fpexit, "    Line    : %d\n", dctestpoint.line);
        fprintf_stdout(fpexit, "    Message : %s\n", dctestpoint.msg);
        fprintf_stdout(fpexit, "\n");

        // write function trace
        write_tracedebugfile();

        // Check open file descriptors
        struct rlimit rlimits;
        int           max_fd_number;

        fprintf_stdout(fpexit, "File descriptors\n");
        getrlimit(RLIMIT_NOFILE, &rlimits);
        max_fd_number = getdtablesize();
        fprintf_stdout(fpexit, "    max_fd_number  : %d\n", max_fd_number);
        fprintf_stdout(fpexit, "    rlim_cur       : %lu\n", rlimits.rlim_cur);
        fprintf_stdout(fpexit, "    rlim_max       : %lu\n", rlimits.rlim_max);
        for (int i = 0; i <= max_fd_number; i++)
        {
            struct stat stats;

            fstat(i, &stats);
            if (errno != EBADF)
            {
                fd_counter++;
            }
        }
        fprintf_stdout(fpexit, "    Open files     : %ld\n", fd_counter);

        fclose(fpexit);
    }
#endif

    return RETURN_SUCCESS;
}

/**
 * @brief Signal handler
 *
 *
 */
void sig_handler(int signo)
{
    switch (signo)
    {
    case SIGINT:
        printf("PID %d sig_handler received SIGINT\n", CLIPID);
        dcsigINT = 1;
        set_terminal_echo_on();
        exit(EXIT_FAILURE);
        break;

    case SIGTERM:
        printf("PID %d sig_handler received SIGTERM\n", CLIPID);
        dcsigTERM = 1;
        set_terminal_echo_on();
        exit(EXIT_FAILURE);
        break;

    case SIGUSR1:
        printf("PID %d sig_handler received SIGUSR1\n", CLIPID);
        dcsigUSR1 = 1;
        break;

    case SIGUSR2:
        printf("PID %d sig_handler received SIGUSR2\n", CLIPID);
        dcsigUSR2 = 1;
        break;

    case SIGBUS: // exit program after SIGSEGV
        printf("PID %d sig_handler received SIGBUS \n", CLIPID);
        write_process_exit_report("SIGBUS");
        dcsigBUS = 1;
        set_terminal_echo_on();
        exit(EXIT_FAILURE);
        break;

    case SIGABRT:
        printf("PID %d sig_handler received SIGABRT\n", CLIPID);
        write_process_exit_report("SIGABRT");
        dcsigABRT = 1;
        set_terminal_echo_on();
        exit(EXIT_FAILURE);
        break;

    case SIGSEGV: // exit program after SIGSEGV
        printf("PID %d sig_handler received SIGSEGV\n", CLIPID);
        write_process_exit_report("SIGSEGV");
        dcsigSEGV = 1;
        set_terminal_echo_on();
        exit(EXIT_FAILURE);
        break;

    case SIGHUP:
        printf("PID %d sig_handler received SIGHUP\n", CLIPID);
        write_process_exit_report("SIGHUP");
        dcsigHUP = 1;
        set_terminal_echo_on();
        exit(EXIT_FAILURE);
        break;

    case SIGPIPE:
        printf("PID %d sig_handler received SIGPIPE\n", CLIPID);
        dcsigPIPE = 1;
        break;
    }
}
