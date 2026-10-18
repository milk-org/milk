// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_bash_coproc.c
 * @brief Supervised bash co-process and in-process socket server lifecycle for milk-cli.
 *
 * Role & Goal:
 *   Manages the persistent bash co-process spawned by interactive `milk-cli`
 *   and `milk-cli <script.milk>`:
 *   - Creates an in-process UNIX domain listener socket (`@milk-cli-<uid>-<pid>`)
 *   - Forks a supervised `/bin/bash` co-process connected via fd 198 (commands)
 *     and fd 199 (NUL-terminated exit status + PWD synchronization)
 *   - Services incoming `milk-call` / loadable builtin socket requests in the
 *     parent `milk-cli` process while bash evaluates commands and control flow.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <errno.h>
#include <dirent.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <termios.h>
#include <unistd.h>

#include "CLIcore.h"
#include "CLIcore_script.h"
#include "CLIcore_signals.h"
#include "milk_bash_engine.h"
#include "milk_bash_engine_internal.h"
#include "milk_shell_lex.h"
#include "milk_server.h"
#include "milk_srvlog.h"

static int   g_listen_fd     = -1;
static int   g_exec_depth    = 0; // >0 while a command runs in bash (re-entrancy guard)
static int   g_cmd_pipe_w    = -1;
static int   g_status_pipe_r = -1;
static pid_t g_bash_pid      = -1;
static char  g_sockpath[128] = { '\0' };
static char  g_cmdfile[300]  = { '\0' }; /* translated command, `source`d by bash (keeps $LINENO) */
static char  g_label[64]     = "milk";   /* command-file name: bash error messages show it */
static int   g_bash_pidfd    = -1;       /* readable once bash exits (-1: poll waitpid instead) */
static pid_t g_engine_owner  = 0;        /* process that started bash (atexit guard) */
static int   g_inflight      = 0;        /* a command was sent and its status is not back yet */
static int   g_tty_handoff   = 0;        /* bash's process group owns the terminal */
static struct termios g_tty_modes;       /* terminal modes saved before the handoff */

/** tty_owner() - make @pgrp the terminal's foreground group (SIGTTOU blocked: we may be in the background). */
static void tty_owner(pid_t pgrp)
{
    sigset_t blk, old;
    sigemptyset(&blk);
    sigaddset(&blk, SIGTTOU);
    sigprocmask(SIG_BLOCK, &blk, &old);
    tcsetpgrp(STDIN_FILENO, pgrp);
    sigprocmask(SIG_SETMASK, &old, NULL);
}

/** tty_give() - hand the terminal to bash while a command runs, so Ctrl-C and `read` reach it. */
static void tty_give(void)
{
    if (g_tty_handoff || g_bash_pid <= 0 || !isatty(STDIN_FILENO) ||
        tcgetpgrp(STDIN_FILENO) != getpgrp())
    {
        return;
    }
    tcgetattr(STDIN_FILENO, &g_tty_modes);
    g_tty_handoff = 1;
    tty_owner(g_bash_pid);
}

/** tty_take() - take the terminal back from bash and restore its modes. */
static void tty_take(void)
{
    if (!g_tty_handoff)
    {
        return;
    }
    tty_owner(getpgrp());
    tcsetattr(STDIN_FILENO, TCSADRAIN, &g_tty_modes);
    g_tty_handoff = 0;
}

/** serve_one() - service one client request; milk-cli owns the terminal for its duration. */
static void serve_one(int *shutdown_flag)
{
    if (g_tty_handoff)
    {
        tty_owner(getpgrp());
    }
    milk_server_accept_and_handle(g_listen_fd, shutdown_flag);
    if (milk_server_take_call_interrupt() && g_bash_pid > 0)
    {
        kill(-g_bash_pid, SIGINT); /* Ctrl-C aborted the milk command: abort the command line too */
    }
    if (g_tty_handoff && g_bash_pid > 0)
    {
        tty_owner(g_bash_pid);
    }
}

/** close_engine_fds() - close the pipes, pidfd and listener of a bash that is gone. */
static void close_engine_fds(void)
{
    int *fds[] = { &g_cmd_pipe_w, &g_status_pipe_r, &g_listen_fd, &g_bash_pidfd };
    for (size_t i = 0; i < sizeof(fds) / sizeof(fds[0]); i++)
    {
        if (*fds[i] >= 0)
        {
            close(*fds[i]);
            *fds[i] = -1;
        }
    }
}

/** bash_exited() - 1 if bash has exited (without reaping it). */
static int bash_exited(void)
{
    if (g_bash_pid <= 0)
    {
        return 1;
    }
    if (g_bash_pidfd >= 0)
    {
        struct pollfd p = { g_bash_pidfd, POLLIN, 0 };
        return poll(&p, 1, 0) > 0;
    }
    siginfo_t si;
    si.si_pid = 0;
    return waitid(P_PID, (id_t) g_bash_pid, &si, WEXITED | WNOHANG | WNOWAIT) == 0 &&
           si.si_pid != 0;
}

/** bash_lost() - reap a bash that died during a command, report it, and return its status (128+sig if killed). */
static int bash_lost(void)
{
    int st = 0;
    waitpid(g_bash_pid, &st, 0);
    g_bash_pid = -1;
    g_inflight = 0;
    tty_take();
    close_engine_fds();
    int rc = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
    if (WIFSIGNALED(st))
    {
        fprintf(stderr, "milk: bash co-process killed by signal %d (%s); shell state is lost\n",
                WTERMSIG(st), strsignal(WTERMSIG(st)));
    }
    else
    {
        fprintf(stderr, "milk: bash co-process exited with status %d; shell state is lost\n", rc);
    }
    return (rc != 0) ? rc : 1;
}

/** engine_atexit() - do not leave bash behind if the process exits without milk_bash_engine_shutdown(). */
static void engine_atexit(void)
{
    if (getpid() != g_engine_owner || g_bash_pid <= 0)
    {
        return;
    }
    tty_take();
    kill(g_bash_pid, SIGHUP);
}

/**
 * @brief Check whether the bash execution engine is enabled for milk-cli.
 *
 * @return Always 1 (the bash engine is the sole execution engine)
 */
int milk_is_bash_engine_enabled(void)
{
    return 1;
}

static int engine_exec(const char *cmdline, int *out_exit_req);

/**
 * @brief Initialize the in-process milk socket server and supervised bash co-process.
 *
 * @return 0 on success, -1 on error
 */
int milk_bash_engine_init(void)
{
    if (g_listen_fd >= 0 && g_bash_pid > 0)
    {
        return 0;
    }

    milk_bash_funsub_disabled(); // probe bash once; result is exported to the coprocess env
    if (milk_bash_cmdfile_path(g_label, g_cmdfile, sizeof(g_cmdfile)) != 0)
    {
        return -1;
    }

    snprintf(g_sockpath, sizeof(g_sockpath), "@milk-cli-%d-%d", (int) getuid(), (int) getpid());
    g_listen_fd = milk_server_create_listener(g_sockpath);
    if (g_listen_fd < 0)
    {
        return -1;
    }
    milk_server_set_log_flags(MILK_SRVFLAG_INPROC);
    setenv("MILKCLI_SOCK", g_sockpath, 1);
    setenv("_MILK_ENGINE_SOCK", g_sockpath, 1); // `@server local` returns here

    int cmd_pipe[2]    = { -1, -1 };
    int status_pipe[2] = { -1, -1 };
    if (pipe2(cmd_pipe, O_CLOEXEC) != 0 || pipe2(status_pipe, O_CLOEXEC) != 0)
    {
        if (cmd_pipe[0] >= 0)
        {
            close(cmd_pipe[0]);
            close(cmd_pipe[1]);
        }
        close(g_listen_fd);
        g_listen_fd = -1;
        return -1;
    }

    cli_export_vars_to_env();
    fflush(stdout);
    fflush(stderr);

    pid_t pid = fork();
    if (pid < 0)
    {
        close(cmd_pipe[0]);
        close(cmd_pipe[1]);
        close(status_pipe[0]);
        close(status_pipe[1]);
        close(g_listen_fd);
        g_listen_fd = -1;
        return -1;
    }

    if (pid == 0)
    {
        /* Child: supervised bash co-process */
        setpgid(0, 0);
        close(g_listen_fd);
        close(cmd_pipe[1]);
        close(status_pipe[0]);

        dup2(cmd_pipe[0], 198);
        dup2(status_pipe[1], 199);
        if (cmd_pipe[0] != 198)
        {
            close(cmd_pipe[0]);
        }
        if (status_pipe[1] != 199)
        {
            close(status_pipe[1]);
        }
        fcntl(198, F_SETFD, 0);
        fcntl(199, F_SETFD, 0);

        const char *loop_script =
            "shopt -s expand_aliases\n"
            ": \"${FUNCNEST:=5000}\"\n" /* runaway recursion ends bash with a message, not a stack overflow */
            "if [[ -n \"${MILK_INSTALLDIR:-}\" && -f \"${MILK_INSTALLDIR}/bin/milk.bash\" ]]; "
            "then\n"
            "    builtin source \"${MILK_INSTALLDIR}/bin/milk.bash\"\n"
            "fi\n"
            "unalias() { builtin unalias \"$@\" 2>/dev/null || true; }\n"
            "__MILK_EXIT_REQ=0\n"
            "__MILK_EXIT_CODE=0\n"
            "__MILK_MAIN_PID=$BASHPID\n"
            "exit() {\n"
            "    local __rc=$?\n"
            "    if [[ $BASHPID != \"$__MILK_MAIN_PID\" ]]; then builtin exit \"${1:-$__rc}\"; fi\n"
            "    __MILK_EXIT_REQ=1\n"
            "    __MILK_EXIT_CODE=\"${1:-$__rc}\"\n"
            "    printf '%d\\t1\\t%s\\0' \"${__MILK_EXIT_CODE}\" \"${PWD}\" >&199 2>/dev/null || "
            "true\n"
            "    builtin exit \"${__MILK_EXIT_CODE}\"\n"
            "}\n"
            /* Ctrl-C: abort the running command (simulated returns out of every function and sourced
             * file, via an extdebug DEBUG trap) but keep the shell and its state */
            "__milk_abort_dbg() {\n"
            "    local __f\n"
            "    for __f in \"${FUNCNAME[@]:1}\"; do [[ \"${__f}\" == source ]] && return 2; done\n"
            "    return 0\n"
            "}\n"
            "__milk_on_int() {\n"
            "    (( __MILK_INTR )) && return\n"
            "    __MILK_INTR=1\n"
            "    __MILK_DBGTRAP=$(trap -p DEBUG)\n"
            "    if shopt -q extdebug; then __MILK_XDBG=1; else __MILK_XDBG=0; fi\n"
            "    shopt -s extdebug\n"
            /* a simulated return inside a RETURN trap (@defer's) would run that trap again, forever: the
             * entries it would run are left to _milk_defer_orphans */
            "    trap 'trap - RETURN; __milk_abort_dbg' DEBUG\n"
            "}\n"
            "__milk_intr_reset() {\n"
            "    trap - DEBUG\n"
            "    [[ -n \"${__MILK_DBGTRAP}\" ]] && eval \"${__MILK_DBGTRAP}\"\n"
            "    (( __MILK_XDBG )) || shopt -u extdebug\n"
            "    __MILK_INTR=0\n"
            "}\n"
            "__MILK_INTR=0\n"
            "trap '__milk_on_int' INT\n"
            "milkcloexec 198 199 2>/dev/null\n" /* children of bash do not inherit the engine pipes */
            "__milk_ret() { return \"$1\"; }\n"
            "__milk_rc=0\n"
            "while :; do\n"
            "    __milk_rd=0\n"
            "    IFS= read -r -d '' __milk_cmd <&198 || __milk_rd=$?\n" /* no errexit if the user set -e */
            "    (( __MILK_INTR )) && __milk_intr_reset\n" /* a late Ctrl-C must not hit the next command */
            "    (( __milk_rd > 128 )) && continue\n"
            "    (( __milk_rd == 0 )) && IFS= read -r -d '' __MILK_INTERACTIVE <&198 || break\n"
            "    if __milk_ret \"${__milk_rc}\"; then\n" /* each command starts with the previous $? */
            "        builtin source \"$__milk_cmd\" 198<&-\n"
            "    else\n"
            "        builtin source \"$__milk_cmd\" 198<&-\n"
            "    fi\n"
            "    __milk_rc=$?\n"
            "    if (( __MILK_INTR )); then __milk_intr_reset; __milk_rc=130; fi\n"
            "    _milk_defer_orphans\n" /* @defer entries that a Ctrl-C left behind */
            "    printf '%d\\t%d\\t%s\\0' \"${__milk_rc}\" \"${__MILK_EXIT_REQ}\" \"${PWD}\" "
            ">&199\n"
            "    if [[ ${__MILK_EXIT_REQ} -eq 1 ]]; then\n"
            "        builtin exit \"${__MILK_EXIT_CODE}\"\n"
            "    fi\n"
            "done\n";

        execl("/bin/bash", "bash", "--norc", "--noprofile", "-c", loop_script, (char *) NULL);
        _exit(127);
    }

    close(cmd_pipe[0]);
    close(status_pipe[1]);
    setpgid(pid, pid); /* also in the child: either may run first */
    g_cmd_pipe_w    = cmd_pipe[1];
    g_status_pipe_r = status_pipe[0];
    g_bash_pid      = pid;
#ifdef SYS_pidfd_open
    g_bash_pidfd = (int) syscall(SYS_pidfd_open, pid, 0);
#endif
    if (g_engine_owner != getpid())
    {
        g_engine_owner = getpid();
        atexit(engine_atexit);
    }

    int dummy_exit = 0;
    engine_exec(":", &dummy_exit); // internal: init may run inside an exec (lazy start / respawn)
    return 0;
}

/**
 * @brief Translate and execute a command or multi-line block in the bash co-process.
 *
 * @param cmdline      Command line or multi-line script text
 * @param out_exit_req Optional: set to 1 if script invoked `exit`/`quit`
 * @return Exit status of the bash command (0 = success)
 */
static int engine_exec(const char *cmdline, int *out_exit_req)
{
    if (out_exit_req != NULL)
    {
        *out_exit_req = 0;
    }
    if (cmdline == NULL)
    {
        return 0;
    }
    fflush(stdout);
    fflush(stderr);
    if (g_listen_fd < 0 || g_bash_pid <= 0)
    {
        if (milk_bash_engine_init() != 0)
        {
            return 1;
        }
    }

    if (bash_exited()) /* died between commands (e.g. killed from outside) */
    {
        waitpid(g_bash_pid, NULL, 0);
        g_bash_pid = -1;
        close_engine_fds();
        fprintf(
            stderr,
            "milk: bash co-process had exited; starting a new one (shell variables are lost)\n");
        if (milk_bash_engine_init() != 0)
        {
            return 1;
        }
    }

    msl_map_t chk;
    if (msl_scan(cmdline, strlen(cmdline), &chk) == 0)
    {
        int unterminated = chk.unterminated;
        msl_free(&chk);
        if (unterminated)
        {
            fprintf(stderr, "milk: syntax error: unterminated quote or substitution\n");
            return 2;
        }
    }

    char *translated = milk_translate_shorthand_to_bash(cmdline, 1);
    if (translated == NULL)
    {
        return 1;
    }

    if (dcerrorexit && strcmp(cmdline, ":") != 0)
    {
        size_t old_len = strlen(translated);
        char  *with_e  = (char *) malloc(old_len + 16);
        if (with_e != NULL)
        {
            snprintf(with_e, old_len + 16, "set -e; %s",
                     translated); // same line: $LINENO unchanged
            free(translated);
            translated = with_e;
        }
    }

    /* the text goes to a command file named after the script; the pipe carries its path */
    milk_bash_cmdfile_path(g_label, g_cmdfile, sizeof(g_cmdfile));
    int    cfd  = open(g_cmdfile, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    size_t tlen = strlen(translated);
    size_t sent = 0;
    while (cfd >= 0 && sent < tlen)
    {
        ssize_t n = write(cfd, translated + sent, tlen - sent);
        if (n < 0 && errno == EINTR)
        {
            continue;
        }
        if (n <= 0)
        {
            break;
        }
        sent += (size_t) n;
    }
    if (cfd >= 0)
    {
        close(cfd);
    }
    free(translated);
    if (cfd < 0 || sent < tlen)
    {
        fprintf(stderr, "milk: cannot write command file %s\n", g_cmdfile);
        return 1;
    }
    /* message: command-file path, then the interactive flag (bash reads it into __MILK_INTERACTIVE) */
    char   msg[sizeof(g_cmdfile) + 4];
    size_t plen = strlen(g_cmdfile) + 1;
    memcpy(msg, g_cmdfile, plen);
    msg[plen++]  = cli_is_interactive_mode() ? '1' : '0';
    msg[plen++]  = '\0';
    size_t psent = 0;
    while (psent < plen)
    {
        ssize_t n = write(g_cmd_pipe_w, msg + psent, plen - psent);
        if (n < 0 && errno == EINTR)
        {
            continue;
        }
        if (n <= 0)
        {
            unlink(g_cmdfile);
            return 1;
        }
        psent += (size_t) n;
    }

    char   sbuf[4096];
    size_t spos          = 0;
    int    shutdown_flag = 0;
    int    lost          = 0;
    dcsigINT             = 0;
    g_inflight           = 1;
    tty_give();
    cli_set_sigint_forward(1); /* SIGINT to milk-cli is passed on to bash, never a longjmp */

    while (spos < sizeof(sbuf) - 1)
    {
        if (dcsigINT)
        {
            dcsigINT = 0;
            kill(-g_bash_pid, SIGINT);
        }

        struct pollfd pfds[3] = {
            { g_listen_fd, POLLIN, 0 },
            { g_status_pipe_r, POLLIN | POLLHUP | POLLERR, 0 },
            { g_bash_pidfd, POLLIN, 0 },
        };
        int pr = poll(pfds, (g_bash_pidfd >= 0) ? 3 : 2, 50);
        if (pr < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            break;
        }

        if (pfds[0].revents & POLLIN)
        {
            serve_one(&shutdown_flag);
        }

        if (pfds[1].revents & (POLLIN | POLLHUP | POLLERR))
        {
            ssize_t r = read(g_status_pipe_r, sbuf + spos, sizeof(sbuf) - 1 - spos);
            if (r > 0)
            {
                spos += (size_t) r;
                if (memchr(sbuf + spos - (size_t) r, '\0', (size_t) r) != NULL)
                {
                    break;
                }
                continue;
            }
            if (r == 0 || (errno != EINTR && errno != EAGAIN))
            {
                lost = 1; /* EOF: bash is gone */
            }
        }

        if (lost || (pfds[2].revents & POLLIN) || (pr == 0 && g_bash_pidfd < 0 && bash_exited()))
        {
            cli_set_sigint_forward(0);
            int rc_lost = bash_lost();
            if (out_exit_req != NULL)
            {
                *out_exit_req =
                    !cli_is_interactive_mode(); /* a script stops; the prompt carries on */
            }
            cli_last_retval = rc_lost;
            return rc_lost;
        }
    }
    cli_set_sigint_forward(0);
    tty_take();
    g_inflight = 0;

    sbuf[spos]  = '\0';
    int   rc    = 0;
    int   exreq = 0;
    char *tab1  = strchr(sbuf, '\t');
    if (tab1 != NULL)
    {
        *tab1      = '\0';
        rc         = (int) strtol(sbuf, NULL, 10);
        char *tab2 = strchr(tab1 + 1, '\t');
        if (tab2 != NULL)
        {
            *tab2            = '\0';
            exreq            = (int) strtol(tab1 + 1, NULL, 10);
            const char *npwd = tab2 + 1;
            if (npwd[0] != '\0')
            {
                if (chdir(npwd) == 0)
                {
                    setenv("PWD", npwd, 1);
                }
            }
        }
    }

    if (out_exit_req != NULL)
    {
        *out_exit_req = exreq;
    }
    cli_last_retval = rc;
    unlink(g_cmdfile); /* bash has read it (source reads the whole file) */
    return rc;
}

int milk_bash_engine_busy(void)
{
    return g_exec_depth > 0;
}

int milk_bash_engine_listen_fd(void)
{
    return g_listen_fd;
}

void milk_bash_engine_serve(void)
{
    int shutdown_flag = 0;
    if (g_listen_fd >= 0)
    {
        serve_one(&shutdown_flag);
    }
}

int milk_bash_engine_exec(const char *cmdline, int *out_exit_req)
{
    if (g_exec_depth > 0)
    {
        // a milk command called from bash tried to run shell code: bash is blocked on our reply
        fprintf(stderr, "milk: cannot run a shell command from inside a milk command "
                        "(use bash constructs, e.g. 'time cmd' or a while loop)\n");
        if (out_exit_req != NULL)
        {
            *out_exit_req = 0;
        }
        return 1;
    }
    g_exec_depth++;
    int rc = engine_exec(cmdline, out_exit_req);
    g_exec_depth--;
    dcexitcode = rc; /* milk-cli exits with the status of the last command */
    return rc;
}

/**
 * @brief Read, translate, and execute a script file in the bash co-process.
 *
 * @param filename     Path to script file
 * @param out_exit_req Optional: set to 1 if script invoked `exit`/`quit`
 * @return Exit status of the script (0 = success)
 */
int milk_bash_engine_exec_file(const char *filename, int *out_exit_req)
{
    if (filename == NULL || filename[0] == '\0')
    {
        fprintf(stderr, "Usage: source <filename>\n");
        return 1;
    }
    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "source: %s: %s\n", filename, strerror(errno));
        return 1;
    }
    if (fseek(fp, 0, SEEK_END) != 0)
    {
        fclose(fp);
        return 1;
    }
    long fsz = ftell(fp);
    if (fsz < 0)
    {
        fclose(fp);
        return 1;
    }
    rewind(fp);

    char *buf = (char *) malloc((size_t) fsz + 1);
    if (buf == NULL)
    {
        fclose(fp);
        return 1;
    }
    size_t nread = fread(buf, 1, (size_t) fsz, fp);
    buf[nread]   = '\0';
    fclose(fp);

    const char *base = strrchr(filename, '/'); /* bash error messages name the script */
    snprintf(g_label, sizeof(g_label), "%s", (base != NULL) ? base + 1 : filename);
    int rc = milk_bash_engine_exec(buf, out_exit_req);
    snprintf(g_label, sizeof(g_label), "milk");
    free(buf);
    return rc;
}

/**
 * @brief Shut down the supervised bash co-process (running EXIT/defer traps)
 *        and close the in-process socket server.
 */
void milk_bash_engine_shutdown(void)
{
    if (g_cmd_pipe_w >= 0)
    {
        close(g_cmd_pipe_w); /* bash's read loop ends, then its EXIT trap runs */
        g_cmd_pipe_w = -1;
    }

    if (g_bash_pid > 0)
    {
        /* serve EXIT-trap milk calls for as long as bash runs; the trap may also read the terminal */
        int shutdown_flag = 0;
        tty_give();
        cli_set_sigint_forward(1);
        while (g_bash_pid > 0)
        {
            if (dcsigINT)
            {
                dcsigINT = 0;
                kill(-g_bash_pid, SIGINT);
            }
            struct pollfd pfds[3] = {
                { g_listen_fd, POLLIN, 0 },
                { g_status_pipe_r, POLLIN | POLLHUP | POLLERR, 0 },
                { g_bash_pidfd, POLLIN, 0 },
            };
            int pr = poll(pfds, 3, 50); /* negative fds are ignored */
            if (pr < 0 && errno != EINTR)
            {
                break;
            }
            if (pfds[0].revents & POLLIN)
            {
                serve_one(&shutdown_flag);
            }
            if (pfds[1].revents & (POLLIN | POLLHUP | POLLERR))
            {
                char drain[256];
                if (read(g_status_pipe_r, drain, sizeof(drain)) <= 0)
                {
                    close(g_status_pipe_r);
                    g_status_pipe_r = -1;
                }
            }
            if (waitpid(g_bash_pid, NULL, WNOHANG) == g_bash_pid)
            {
                g_bash_pid = -1;
            }
        }
        cli_set_sigint_forward(0);
        tty_take();
    }
    if (g_bash_pid > 0)
    {
        waitpid(g_bash_pid, NULL, 0);
        g_bash_pid = -1;
    }
    milk_bash_cmdfile_cleanup(); /* includes files left by aborted commands */
    g_cmdfile[0] = '\0';
    milk_server_close_log(1);
    close_engine_fds();
}

/**
 * @brief Abort an in-flight bash co-process command after a signal interrupt
 *        or fault-isolation `siglongjmp`.
 */
void milk_bash_engine_abort(void)
{
    g_exec_depth = 0; // the aborted exec was left via siglongjmp
    cli_set_sigint_forward(0);
    tty_take();
    if (g_bash_pid <= 0 || !g_inflight)
    {
        return;
    }

    /* interrupt the command and wait for bash to report it; only a bash that ignores that is killed */
    kill(-g_bash_pid, SIGINT);
    int synced        = 0;
    int shutdown_flag = 0;
    for (int iter = 0; iter < 200 && !synced; iter++)
    {
        struct pollfd pfds[2] = { { g_listen_fd, POLLIN, 0 },
                                  { g_status_pipe_r, POLLIN | POLLHUP | POLLERR, 0 } };
        if (poll(pfds, 2, 10) <= 0)
        {
            if (bash_exited())
            {
                bash_lost();
                return;
            }
            continue;
        }
        if (pfds[0].revents & POLLIN)
        {
            serve_one(&shutdown_flag);
        }
        if (pfds[1].revents & (POLLIN | POLLHUP | POLLERR))
        {
            char    drain[512];
            ssize_t r = read(g_status_pipe_r, drain, sizeof(drain));
            if (r <= 0)
            {
                bash_lost();
                return;
            }
            synced = (memchr(drain, '\0', (size_t) r) != NULL);
        }
    }
    g_inflight = 0;
    if (!synced)
    {
        fprintf(stderr, "milk: bash co-process does not respond to Ctrl-C; killing it\n");
        kill(g_bash_pid, SIGKILL);
        bash_lost();
    }
}
