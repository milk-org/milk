// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_unit_script.c
 * @brief milk-fpsexec-milk-script (milk.script): a compute unit that runs a bash snippet on each frame.
 *
 * .unit.cmd runs on each frame of .unit.trigger, else on each tick of the delay loop, in one bash
 * that loaded milk.bash at start: a tick costs what its commands cost (a pipe round trip, ~0.1 ms,
 * on top). A new .unit.cmd applies from the next tick. A tick that fails stops the loop, with its
 * status in .unit.error (`|| :` ignores a failure).
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
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

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "milk_argval.h"
#include "milk_unit.h"
#include "processinfo_signals.h"

#define SCRIPT_NFIXED 5
#define SCRIPT_MAXPAR 32
#define SCRIPT_MAXTEXT 65536
#define SCRIPT_STATUS_FD 77 /* the shell writes a status line per tick there */

static char cmd[FUNCTION_PARAMETER_STRMAXLEN];
static char trigger[FUNCTION_PARAMETER_STRMAXLEN];
static char params[FUNCTION_PARAMETER_STRMAXLEN];
static char file[FUNCTION_PARAMETER_STRMAXLEN];
static char error[FUNCTION_PARAMETER_STRMAXLEN];

/* Fixed unit-definition entries under .unit.*, set by unit options (not positional values);
 * user-declared parameters are appended at runtime */
#define SCRIPT_FIXED_PARAMS(X)                                                                    \
    X(".unit.cmd", cmd, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                 \
      "commands run on each frame (-c)")                                                          \
    X(".unit.trigger", trigger, FPTYPE_STREAMNAME, 0,                                             \
      FPFLAG_DEFAULT_INPUT | FPFLAG_TRIGGER_STREAM, "a frame of this stream runs them (-t)")      \
    X(".unit.params", params, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT, "declared parameters (-p)") \
    X(".unit.file", file, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,               \
      "commands from this file (-f)")                                                             \
    X(".unit.error", error, FPTYPE_STRING, 0, FPFLAG_DEFAULT_OUTPUT, "why the loop stopped")

static FPS_CLI_BINDING bindings[SCRIPT_NFIXED + SCRIPT_MAXPAR] = { SCRIPT_FIXED_PARAMS(
    FPS_X_BINDING) };
static CLICMDARGDEF    script_farg[] = { SCRIPT_FIXED_PARAMS(FPS_X_FARG) };
MILK_RUNNABLE_RT_DEFINE(script);
static int nbind = SCRIPT_NFIXED;

static union
{
    int64_t  i;
    uint64_t u;
    double   f;
    int32_t  b;
    char     s[FUNCTION_PARAMETER_STRMAXLEN];
} pval[SCRIPT_MAXPAR];
static char pkey[SCRIPT_MAXPAR][FUNCTION_PARAMETER_KEYWORD_STRMAXLEN + 1];
static char pdescr[SCRIPT_MAXPAR][FUNCTION_PARAMETER_DESCR_STRMAXLEN];

static errno_t compute(const MILK_RUNNABLE *self);

static MILK_RUNNABLE unit_script = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = {
        .fps_name    = "milkscript",
        .cmdkey      = "script",
        .description = "run a bash snippet on each frame",
        .description_long =
            "Runs bash commands (-c, or the file -f names) on each frame of the trigger stream "
            "(-t), or on each -loopd tick without one, in a bash that loaded milk.bash once. -p "
            "declares parameters exposed as shell variables: the values after the command word "
            "set them, in declaration order, and @set changes them while it runs. A tick that "
            "fails stops the loop; its status is kept in .unit.error.",
    },
    .srcfile     = __FILE__,
    .bindings    = bindings,
    .nb_bindings = SCRIPT_NFIXED,
    .farg        = script_farg,
    .nbarg       = (int) (sizeof(script_farg) / sizeof(script_farg[0])),
    .rt          = &script_rt,
    .compute     = compute,
};

static pid_t sh_pid = -1;
static int   sh_in  = -1; /* commands to the shell */
static int   sh_st  = -1; /* a status line per tick, from SCRIPT_STATUS_FD */

/**
 * is_fixed - Check whether a parameter name collides with reserved script unit names
 * @name: Candidate parameter name (without leading dot)
 *
 * Rejects the built-in frame index variable `_n` and `unit` (which holds the unit
 * definition entries `.unit.*`).
 *
 * Return: 1 if @name is reserved, 0 otherwise.
 */
static int is_fixed(const char *name)
{
    return strcmp(name, "_n") == 0 || strcmp(name, "unit") == 0;
}

/**
 * bind - Register a user-declared parameter in the script unit's FPS binding table
 * @name:   Parameter name (without leading dot)
 * @type:   FPS parameter type (`FPTYPE_*`)
 * @fpflag: FPS parameter flags (`FPFLAG_*`)
 * @descr:  Optional human-readable description string (may be NULL)
 *
 * Appends `.name` to `bindings[]` backed by `pval[]` after verifying that the
 * parameter limit (`SCRIPT_MAXPAR`) has not been reached and @name is not reserved.
 *
 * Return: 0 on success, -1 if @name is reserved or the parameter table is full.
 */
static int bind(const char *name, uint64_t type, uint64_t fpflag, const char *descr)
{
    int k = nbind - SCRIPT_NFIXED;

    if (k == SCRIPT_MAXPAR || is_fixed(name))
    {
        fprintf(stderr, "milk.script: parameter '%s': %s\n", name,
                is_fixed(name) ? "reserved" : "too many");
        return -1;
    }
    memset(&pval[k], 0, sizeof(pval[k]));
    snprintf(pkey[k], sizeof(pkey[k]), ".%s", name);
    snprintf(pdescr[k], sizeof(pdescr[k]), "%s", (descr != NULL) ? descr : "");
    bindings[nbind++] = (FPS_CLI_BINDING) { pkey[k], &pval[k], type, 1, fpflag, pdescr[k], -2, -1 };
    return 0;
}

/**
 * declare - Parse whitespace-separated parameter specifications and bind them
 * @text: Space-separated list of `"name:type[=default][:help]"` declarations
 *
 * Parses each parameter specification in @text, registers it via bind(), and
 * initializes its default value in `pval[]`.
 *
 * Return: 0 on success, -1 on syntax error or binding failure.
 */
static int declare(const char *text)
{
    char *buf = strdup(text);
    char *save;
    int   rc = (buf == NULL) ? -1 : 0;

    for (char *t = strtok_r(buf, " \t\n", &save); t != NULL && rc == 0;
         t       = strtok_r(NULL, " \t\n", &save))
    {
        milk_argspec_t d;
        CLICMDARGDATA  ad;
        char           spec[FUNCTION_PARAMETER_STRMAXLEN * 4];
        snprintf(spec, sizeof(spec), "%s", t);
        memset(&ad, 0, sizeof(ad));
        if (milk_argspec_parse(t, &d) != 0 || (d.def != NULL && d.type != FPTYPE_STRING &&
                                               milk_argval_parse(d.type, d.def, &ad) != 0))
        {
            fprintf(
                stderr,
                "milk.script: bad parameter '%s': name:type[=default][:help], type int uint float "
                "bool str stream fits\n",
                spec);
            rc = -1;
        }
        else if ((rc = bind(d.name, d.type, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, d.help)) == 0)
        {
            int k = nbind - SCRIPT_NFIXED - 1;
            switch (d.type)
            {
            case FPTYPE_INT64:
                pval[k].i = ad.val.i64;
                break;
            case FPTYPE_UINT64:
                pval[k].u = ad.val.ui64;
                break;
            case FPTYPE_FLOAT64:
                pval[k].f = ad.val.f64;
                break;
            case FPTYPE_ONOFF:
                pval[k].b = (ad.val.i64 != 0);
                break;
            default:
                snprintf(pval[k].s, sizeof(pval[k].s), "%s", (d.def != NULL) ? d.def : "");
            }
        }
    }
    free(buf);
    return rc;
}

/**
 * find_params - Discover and bind user-declared parameters from an existing FPS
 * @name: FPS instance name in shared memory
 *
 * Connects to @name and binds all active single-level non-fixed parameters, with their
 * current values, so a subsequent `run` or `set` invocation maps them into `bindings[]`.
 *
 * Return: 0 on success (or if @name is not yet created), -1 on binding failure.
 */
static int find_params(const char *name)
{
    FPS fps;
    int rc = 0;

    if (fps_connect(name, &fps, FPSCONNECT_SIMPLE) == -1)
    {
        return 0;
    }
    for (long i = 0; i < fps.md->NBparamMAX && rc == 0; i++)
    {
        const FPS_PARAM *p = &fps.parray[i];
        if ((p->fpflag & FPFLAG_ACTIVE) && p->keywordlevel == 2 && !is_fixed(p->keyword[1]) &&
            (p->type == FPTYPE_INT64 || p->type == FPTYPE_UINT64 || p->type == FPTYPE_FLOAT64 ||
             p->type == FPTYPE_ONOFF || p->type == FPTYPE_STRING || p->type == FPTYPE_STREAMNAME ||
             p->type == FPTYPE_FITSFILENAME))
        {
            rc = bind(p->keyword[1], p->type, p->fpflag, p->description);
            if (rc == 0)
            {
                sync_fps_to_local(&fps, i, &bindings[nbind - 1]);
            }
        }
    }
    fps_disconnect(&fps);
    return rc;
}

/**
 * source_text - Load script commands from @fpath if non-empty, else duplicate @inline_str
 * @inline_str: Inline command snippet from `.cmd`
 * @fpath:      Path to script file from `.file` (empty string if unused)
 * @err:        Output buffer receiving an error message on failure
 * @errsz:      Capacity of @err in bytes
 *
 * Reads up to `SCRIPT_MAXTEXT - 1` bytes from @fpath when specified, or duplicates
 * @inline_str when @fpath is empty.
 *
 * Return: Newly allocated command string (caller frees), or NULL on error.
 */
static char *source_text(const char *inline_str, const char *fpath, char *err, size_t errsz)
{
    if (fpath[0] == '\0')
    {
        return strdup(inline_str);
    }
    FILE *fp = fopen(fpath, "r");
    char *t  = (fp != NULL) ? calloc(1, SCRIPT_MAXTEXT) : NULL;
    if (t != NULL && fread(t, 1, SCRIPT_MAXTEXT - 1, fp) == SCRIPT_MAXTEXT - 1)
    {
        free(t);
        t = NULL;
        snprintf(err, errsz, "%s is longer than %d bytes", fpath, SCRIPT_MAXTEXT - 1);
    }
    else if (t == NULL)
    {
        snprintf(err, errsz, "cannot read %s", fpath);
    }
    if (fp != NULL)
    {
        fclose(fp);
    }
    return t;
}

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

/**
 * sync_shell_params - Push modified FPS user parameters into the persistent bash coprocess
 * @npar:  Number of user-declared parameters
 * @pcnt:  Per-parameter array tracking the last synced `value_cnt`
 * @force: Non-zero to emit all parameters regardless of `value_cnt` (used on frame 0)
 *
 * Writes shell variable assignments (`name=value`) to `sh_in` for any declared
 * parameter whose FPS `value_cnt` changed since the previous tick (or all parameters
 * when @force is non-zero).
 *
 * Return: void
 */
static void sync_shell_params(int npar, uint64_t *pcnt, int force)
{
    for (int k = 0; k < npar; k++)
    {
        const FPS_CLI_BINDING *b    = &bindings[SCRIPT_NFIXED + k];
        long                   pidx = b->_fps_pindex;
        if (!force && (pidx < 0 || dcfpsptr->parray[pidx].value_cnt == pcnt[k]))
        {
            continue;
        }
        if (pidx >= 0)
        {
            pcnt[k] = dcfpsptr->parray[pidx].value_cnt;
        }
        const char *name = b->fpskeyword + 1;
        switch (b->type)
        {
        case FPTYPE_INT64:
            dprintf(sh_in, "%s=%" PRId64 "\n", name, pval[k].i);
            break;
        case FPTYPE_UINT64:
            dprintf(sh_in, "%s=%" PRIu64 "\n", name, pval[k].u);
            break;
        case FPTYPE_FLOAT64:
        {
            CLICMDARGDATA ad = { .type = FPTYPE_FLOAT64, .val.f64 = pval[k].f };
            char          buf[64];
            milk_argval_format(&ad, buf, sizeof(buf));
            dprintf(sh_in, "%s=%s\n", name, buf);
            break;
        }
        case FPTYPE_ONOFF:
            dprintf(sh_in, "%s=%d\n", name, pval[k].b ? 1 : 0);
            break;
        default:
            dprintf(sh_in, "%s=", name);
            put_quoted(pval[k].s);
            dprintf(sh_in, "\n");
            break;
        }
    }
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

/**
 * shell_stop - Terminate and reap the persistent bash coprocess
 *
 * Closes the command and status pipes to signal EOF to the child bash process,
 * waits up to 1 second for graceful exit, and sends `SIGKILL` to its process group
 * if it has not exited.
 *
 * Return: void
 */
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
    dprintf(sh_in,
            "/milk.bash >/dev/null || exit 1\n"
            "__milk_unit_tick() { _n=$1; eval \"$__MILK_UNIT_CMD\" </dev/null; printf '%%d\\n' $? "
            ">&%d; }\n"
            "echo ready >&%d\n",
            SCRIPT_STATUS_FD, SCRIPT_STATUS_FD);
    if (read_line(line, sizeof(line), 60.0, NULL) != 0)
    {
        snprintf(err, errsz, "bash could not load %s/milk.bash", dir);
        return -1;
    }
    return 0;
}

/**
 * report - Record a runtime error message to stderr, the FPS `.unit.error` parameter, and processinfo
 * @pi:  Active `PROCESSINFO` handle (may be NULL)
 * @msg: Error message string
 *
 * Prints @msg to `stderr`, stores it in the `.unit.error` output parameter of the unit's
 * FPS, and writes it to the processinfo status message when @pi is active.
 *
 * Return: void
 */
static void report(PROCESSINFO *pi, const char *msg)
{
    fprintf(stderr, "milk.script %s: %s\n", dcfpsptr->md->name, msg);
    snprintf(error, sizeof(error), "%s", msg);
    functionparameter_SetParamValue_STRING(dcfpsptr, ".unit.error", error);
    if (pi != NULL)
    {
        processinfo_WriteMessage(pi, msg);
    }
}

/**
 * compute - Main processinfo frame loop driving the persistent bash coprocess
 * @self: Pointer to the `MILK_RUNNABLE` descriptor for this compute unit
 *
 * Launches a background bash shell preloaded with `milk.bash`, synchronizes modified
 * FPS parameters into shell variables before each tick, updates `__MILK_UNIT_CMD` when
 * `.unit.cmd` or `.unit.file` changes, and dispatches `__milk_unit_tick` on every frame or delay
 * tick until stopped or an unhandled non-zero status occurs.
 *
 * Return: `RETURN_SUCCESS` on clean loop exit, `RETURN_FAILURE` if a command tick failed.
 */
static errno_t compute(const MILK_RUNNABLE *self)
{
    char        err[512]            = "";
    long        pc                  = functionparameter_GetParamIndex(dcfpsptr, ".unit.cmd");
    long        pf                  = functionparameter_GetParamIndex(dcfpsptr, ".unit.file");
    uint64_t    built[2]            = { 0, 0 };
    uint64_t    pcnt[SCRIPT_MAXPAR] = { 0 };
    int         npar                = nbind - SCRIPT_NFIXED;
    int64_t     frame_n             = 0;
    const char *tname               = dcfpsptr->cmdset.triggerstreamname;

    error[0] = '\0';
    functionparameter_SetParamValue_STRING(dcfpsptr, ".unit.error", "");
    if (tname[0] != '\0' && image_ID(tname, dcimg, dcnimg) == -1)
    {
        read_sharedmem_image(tname, dcimg, dcnimg); /* the processinfo trigger waits on it */
    }
    processinfo_CatchSignals(); /* Ctrl-C and runstop end the loop through its cleanup */

    MILK_PROCINFO_INIT(self)
    if (shell_start(err, sizeof(err)) != 0)
    {
        report(processinfo, err);
        processloopOK = 0;
    }
    MILK_PROCINFO_LOOPSTART(self)
    {
        char line[64];
        int  rc = 0;
        sync_shell_params(npar, pcnt, frame_n == 0);
        if (frame_n == 0 || built[0] != dcfpsptr->parray[pc].value_cnt ||
            built[1] != dcfpsptr->parray[pf].value_cnt)
        {
            char *text = source_text(cmd, file, err, sizeof(err));
            built[0]   = dcfpsptr->parray[pc].value_cnt;
            built[1]   = dcfpsptr->parray[pf].value_cnt;
            if (text == NULL)
            {
                report(processinfo, err);
                processloopOK = 0;
            }
            else
            {
                dprintf(sh_in, "__MILK_UNIT_CMD=");
                put_quoted(text);
                dprintf(sh_in, "\n");
                free(text);
            }
        }
        if (err[0] == '\0')
        {
            dprintf(sh_in, "__milk_unit_tick %" PRId64 "\n", frame_n++);
            rc = read_line(line, sizeof(line), 1e9, processinfo);
            if (rc == 0 && atoi(line) != 0)
            {
                snprintf(err, sizeof(err), "status %d: %s", atoi(line),
                         (file[0] != '\0') ? file : cmd);
                report(processinfo, err);
            }
            else if (rc == -1)
            {
                snprintf(err, sizeof(err), "the shell exited: %s", (file[0] != '\0') ? file : cmd);
                report(processinfo, err);
            }
            processloopOK = (rc == 0 && err[0] == '\0') ? processloopOK : 0;
        }
    }
    MILK_PROCINFO_END(self)

    shell_stop();
    return (err[0] != '\0') ? RETURN_FAILURE : RETURN_SUCCESS;
}

/**
 * main - Entry point for the milk-fpsexec-milk-script standalone compute unit
 * @argc: Command-line argument count
 * @argv: Command-line argument vector
 *
 * The unit options -c -t -p -f set the unit definition (.unit.cmd .unit.trigger .unit.params
 * .unit.file); milk_unit_main() creates or updates the FPS and delegates to `milk_runnable_main()`.
 *
 * Return: 0 on success, 2 on a usage error, non-zero on other errors.
 */
int main(int argc, char **argv)
{
    static const milk_unit_opt opts[] = {
        { "-c", "--cmd", 0, 0, "write the commands in a file (-f)" },
        { "-t", "--trigger", 1, 0, NULL },
        { "-p", "--params", 2, 0, NULL },
        { "-f", "--file", 3, 1, NULL },
    };
    static const milk_unit_desc desc = {
        .tag         = "milk.script",
        .unit        = &unit_script,
        .bindings    = bindings,
        .nfixed      = SCRIPT_NFIXED,
        .nbind       = &nbind,
        .opts        = opts,
        .nopt        = (int) (sizeof(opts) / sizeof(opts[0])),
        .params_opt  = 2,
        .declare     = declare,
        .find_params = find_params,
        .help_options =
            "  -c, --cmd CMD         bash commands run on each tick (.unit.cmd;\n"
            "                        end with '|| :' to go on)\n"
            "  -t, --trigger STREAM  a frame of STREAM runs them (.unit.trigger)\n"
            "  -p, --params DECL     declare parameters (.unit.params):\n"
            "                        \"name:type[=default][:help] ...\",\n"
            "                        types int uint float bool str stream fits\n"
            "  -f, --file FILE       commands from FILE instead of -c (.unit.file)\n"
            "  fpsinit and exec create the FPS from them; on an existing FPS, -c -t -f update\n"
            "  .unit.* (applied from the next tick) and -p cannot change. Why the loop\n"
            "  stopped is recorded in .unit.error.\n"
            "  Example:\n"
            "    milk-fpsexec-milk-script -n mon -t cam -p \"log:str=cam.log\" \\\n"
            "        -c '@get s.cam.mean >> \"$log\"' exec\n",
    };

    return milk_unit_main(argc, argv, &desc);
}
