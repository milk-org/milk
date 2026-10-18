// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_unit_script.c
 * @brief milk-fpsexec-milk-script (milk.script): a compute unit that runs a bash snippet on each frame.
 *
 * .cmd runs on each frame of .trigger, else on each tick of the delay loop, in one bash that loaded milk.bash
 * at start: a tick costs what its commands cost (a pipe round trip, ~0.1 ms, on top). A new .cmd applies from
 * the next tick. A tick that fails stops the loop, with its status in .error (`|| :` ignores a failure).
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "fps.h"
#include "milk_unit.h"
#include "processinfo_signals.h"

#define SCRIPT_NFIXED 3
#define SCRIPT_STATUS_FD 77 /* the shell writes a status line per tick there */

static FPS_APP_INFO app = {
    .fps_name    = "milkscript",
    .cmdkey      = "script",
    .description = "run a bash snippet on each frame",
    .description_long =
        "Runs the commands of .cmd on each frame of .trigger, in a bash that loaded milk.bash "
        "once. A tick that fails stops the loop; its status is kept in .error."
};

static char cmd[FUNCTION_PARAMETER_STRMAXLEN];
static char trigger[FUNCTION_PARAMETER_STRMAXLEN];
static char error[FUNCTION_PARAMETER_STRMAXLEN];

static FPS_CLI_BINDING bindings[SCRIPT_NFIXED] = {
    { ".cmd", cmd, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,
      "commands run on each frame", -2, -1 },
    { ".trigger", trigger, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT | FPFLAG_TRIGGER_STREAM,
      "a frame of this stream runs them", -2, -1 },
    { ".error", error, FPTYPE_STRING, 0, FPFLAG_DEFAULT_OUTPUT, "why the loop stopped", -2, -1 },
};

static CLICMDARGDEF farg[1];
CLICMDDATA          CLIcmddata = { "", "", CLICMD_FIELDS_DEFAULTS };
FPS_CMDSETTINGS_INIT(script, CLIcmddata, app)

static pid_t sh_pid = -1;
static int   sh_in  = -1; /* commands to the shell */
static int   sh_st  = -1; /* a status line per tick, from SCRIPT_STATUS_FD */

/** put_quoted() - write @s to the shell as a single-quoted word. */
static int put_quoted(const char *s)
{
    FILE *fp = fdopen(dup(sh_in), "w");

    if (fp == NULL)
    {
        return -1;
    }
    fputc('\'', fp);
    for (; *s != '\0'; s++)
    {
        (*s == '\'') ? fputs("'\\''", fp) : fputc(*s, fp);
    }
    fputc('\'', fp);
    return fclose(fp);
}

/** read_line() - the shell's next status line into @buf; 0, -1 if it exited, -2 if it was stopped. */
static int read_line(char *buf, size_t sz, double timeout, PROCESSINFO *pi)
{
    size_t          n    = 0;
    int             stop = 0;
    struct timespec t0;
    struct timespec t;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (;;)
    {
        struct pollfd p = { sh_st, POLLIN, 0 };
        int           r = poll(&p, 1, 100);
        clock_gettime(CLOCK_MONOTONIC, &t);
        double dt = (double) (t.tv_sec - t0.tv_sec) + 1e-9 * (double) (t.tv_nsec - t0.tv_nsec);
        if (!stop && (processinfo_signal_INT || processinfo_signal_TERM || processinfo_signal_HUP ||
                      (pi != NULL && pi->CTRLval == 3)))
        {
            stop = 1; /* interrupt the tick: the shell is in its own process group */
            kill(-sh_pid, SIGINT);
            clock_gettime(CLOCK_MONOTONIC, &t0);
            timeout = 2.0;
        }
        else if (dt > timeout)
        {
            kill(-sh_pid, SIGKILL);
            return stop ? -2 : -1;
        }
        if (r <= 0)
        {
            continue;
        }
        ssize_t k = read(sh_st, buf + n, 1);
        if (k <= 0)
        {
            return stop ? -2 : -1;
        }
        if (buf[n] == '\n')
        {
            buf[n] = '\0';
            return 0;
        }
        n = (n + 2 < sz) ? n + 1 : 0; /* not a status line: dropped */
    }
}

static void shell_stop(void)
{
    if (sh_pid <= 0)
    {
        return;
    }
    close(sh_in); /* end of input: bash exits */
    close(sh_st);
    for (int i = 0; i < 100 && waitpid(sh_pid, NULL, WNOHANG) == 0; i++)
    {
        usleep(10000);
    }
    kill(-sh_pid, SIGKILL);
    waitpid(sh_pid, NULL, WNOHANG);
    sh_pid = -1;
}

/** shell_start() - bash with milk.bash loaded, reading commands from a pipe; 0 or -1 (@err). */
static int shell_start(char *err, size_t errsz)
{
    char        dir[PATH_MAX];
    const char *inst = getenv("MILK_INSTALLDIR");
    int         in[2];
    int         st[2];
    char        line[64];

    if (inst != NULL)
    {
        snprintf(dir, sizeof(dir), "%s/bin", inst);
    }
    else
    {
        ssize_t n            = readlink("/proc/self/exe", dir, sizeof(dir) - 1);
        dir[(n > 0) ? n : 0] = '\0';
        *(strrchr(dir, '/') != NULL ? strrchr(dir, '/') : dir) = '\0';
    }
    if (pipe2(in, O_CLOEXEC) != 0 || pipe2(st, O_CLOEXEC) != 0)
    {
        snprintf(err, errsz, "pipe: %s", strerror(errno));
        return -1;
    }
    sh_pid = fork();
    if (sh_pid == 0)
    {
        setpgid(0, 0); /* Ctrl-C reaches the unit, which interrupts the tick itself */
        prctl(PR_SET_PDEATHSIG, SIGKILL);
        dup2(in[0], 0);
        dup2(st[1], SCRIPT_STATUS_FD);
        fcntl(0, F_SETFD, 0);
        fcntl(SCRIPT_STATUS_FD, F_SETFD, 0);
        execlp("bash", "bash", "--norc", "--noprofile", "-s", (char *) NULL);
        _exit(127);
    }
    close(in[0]);
    close(st[1]);
    sh_in = in[1];
    sh_st = st[0];
    if (sh_pid < 0)
    {
        snprintf(err, errsz, "fork: %s", strerror(errno));
        return -1;
    }
    signal(SIGPIPE, SIG_IGN);
    dprintf(sh_in, "source ");
    put_quoted(dir);
    dprintf(
        sh_in,
        "/milk.bash >/dev/null || exit 1\n"
        "__milk_unit_tick() { eval \"$__MILK_UNIT_CMD\" </dev/null; printf '%%d\\n' $? >&%d; }\n"
        "echo ready >&%d\n",
        SCRIPT_STATUS_FD, SCRIPT_STATUS_FD);
    if (read_line(line, sizeof(line), 60.0, NULL) != 0)
    {
        snprintf(err, errsz, "bash could not load %s/milk.bash", dir);
        return -1;
    }
    return 0;
}

static void report(PROCESSINFO *pi, const char *msg)
{
    fprintf(stderr, "milk.script %s: %s\n", dcfpsptr->md->name, msg);
    snprintf(error, sizeof(error), "%s", msg);
    functionparameter_SetParamValue_STRING(dcfpsptr, ".error", error);
    if (pi != NULL)
    {
        processinfo_WriteMessage(pi, msg);
    }
}

static errno_t compute(void)
{
    char        err[256] = "";
    long        pc       = functionparameter_GetParamIndex(dcfpsptr, ".cmd");
    uint64_t    built    = 0;
    const char *tname    = dcfpsptr->cmdset.triggerstreamname;

    error[0] = '\0';
    functionparameter_SetParamValue_STRING(dcfpsptr, ".error", "");
    if (tname[0] != '\0' && image_ID(tname, dcimg, dcnimg) == -1)
    {
        read_sharedmem_image(tname, dcimg, dcnimg); /* the processinfo trigger waits on it */
    }
    processinfo_CatchSignals(); /* Ctrl-C and runstop end the loop through its cleanup */

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT
    if (shell_start(err, sizeof(err)) != 0)
    {
        report(processinfo, err);
        processloopOK = 0;
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        char line[64];
        int  rc = 0;
        if (built != dcfpsptr->parray[pc].value_cnt || built == 0)
        {
            built = dcfpsptr->parray[pc].value_cnt;
            dprintf(sh_in, "__MILK_UNIT_CMD=");
            put_quoted(cmd);
            dprintf(sh_in, "\n");
        }
        dprintf(sh_in, "__milk_unit_tick\n");
        rc = read_line(line, sizeof(line), 1e9, processinfo);
        if (rc == 0 && atoi(line) != 0)
        {
            snprintf(err, sizeof(err), "status %d: %s", atoi(line), cmd);
            report(processinfo, err);
        }
        else if (rc == -1)
        {
            snprintf(err, sizeof(err), "the shell exited: %s", cmd);
            report(processinfo, err);
        }
        processloopOK = (rc == 0 && err[0] == '\0') ? processloopOK : 0;
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    shell_stop();
    return (err[0] != '\0') ? RETURN_FAILURE : RETURN_SUCCESS;
}

/** create() - (re)create the FPS from the command line: .cmd .trigger. */
static int create(const milk_unit_args *a)
{
    char *fixed[] = { cmd, trigger };

    for (int i = 0; i < 2; i++)
    {
        const char *v = milk_unit_pos(a, i);
        if (v != NULL && strlen(v) >= FUNCTION_PARAMETER_STRMAXLEN)
        {
            fprintf(stderr, "milk.script: %s is longer than %d characters%s\n",
                    bindings[i].fpskeyword, FUNCTION_PARAMETER_STRMAXLEN - 1,
                    (i == 0) ? ": source a file from it" : "");
            return 1;
        }
        snprintf(fixed[i], FUNCTION_PARAMETER_STRMAXLEN, "%s", (v != NULL) ? v : "");
    }
    return milk_unit_create(a, &app, bindings, SCRIPT_NFIXED);
}

int main(int argc, char **argv)
{
    milk_unit_args a;
    int            init;

    milk_unit_args_read(argc, argv, app.fps_name, &a);
    init = (strcmp(a.cmd, "fpsinit") == 0);
    if (!a.help && (init || (strcmp(a.cmd, "exec") == 0 && !milk_unit_fps_exists(a.name))))
    {
        int rc = create(&a);
        if (rc != 0 || init)
        {
            return rc; /* fpsinit is done here: the parameters come from this command line */
        }
    }
    return main_impl(argc, argv, &app, bindings, SCRIPT_NFIXED, farg, compute, NULL);
}
