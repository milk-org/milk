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
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "CLIcore.h"
#include "CLIcore_script.h"
#include "CLIcore_signals.h"
#include "milk_bash_engine.h"
#include "milk_bash_engine_internal.h"
#include "milk_server.h"
#include "milk_srvlog.h"

static int   g_listen_fd     = -1;
static int   g_cmd_pipe_w    = -1;
static int   g_status_pipe_r = -1;
static pid_t g_bash_pid      = -1;
static char  g_sockpath[128] = { '\0' };

/**
 * @brief Check whether the bash execution engine is enabled for milk-cli.
 *
 * @return Always 1 (the bash engine is the sole execution engine)
 */
int milk_is_bash_engine_enabled(void)
{
    return 1;
}

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

    snprintf(g_sockpath, sizeof(g_sockpath), "@milk-cli-%d-%d", (int) getuid(), (int) getpid());
    g_listen_fd = milk_server_create_listener(g_sockpath);
    if (g_listen_fd < 0)
    {
        return -1;
    }
    milk_server_set_log_flags(MILK_SRVFLAG_INPROC);
    setenv("MILKCLI_SOCK", g_sockpath, 1);

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
            "trap ':' INT\n"
            "if [[ -n \"${MILK_INSTALLDIR:-}\" && -f \"${MILK_INSTALLDIR}/bin/milk.bash\" ]]; "
            "then\n"
            "    builtin source \"${MILK_INSTALLDIR}/bin/milk.bash\"\n"
            "fi\n"
            "unalias() { builtin unalias \"$@\" 2>/dev/null || true; }\n"
            "__MILK_SOURCE_DEPTH=0\n"
            "source() {\n"
            "    if [[ $# -eq 0 ]]; then echo \"Usage: source <filename>\" >&2; return 1; fi\n"
            "    if [[ ! -f \"$1\" ]]; then echo \"Error: source: file not found: $1\" >&2; return "
            "1; fi\n"
            "    if (( __MILK_SOURCE_DEPTH >= 32 )); then\n"
            "        echo \"Error: maximum source recursion depth (32) exceeded\" >&2\n"
            "        return 1\n"
            "    fi\n"
            "    local __milk_f=\"$1\"; shift\n"
            "    (( __MILK_SOURCE_DEPTH++ ))\n"
            "    builtin source <(\"${MILK_INSTALLDIR}/bin/milk-call\" translate --legacy "
            "\"${__milk_f}\") \"$@\"\n"
            "    local __rc=$?\n"
            "    (( __MILK_SOURCE_DEPTH-- ))\n"
            "    return $__rc\n"
            "}\n"
            "declare -gA __MILK_INC_ONCE=()\n"
            "include_once() {\n"
            "    if [[ $# -eq 0 ]]; then echo \"Usage: include_once <filename>\" >&2; return 1; "
            "fi\n"
            "    if [[ ! -f \"$1\" ]]; then echo \"Error: include_once: file not found: $1\" >&2; "
            "return 1; fi\n"
            "    local __rp; __rp=\"$(realpath \"$1\" 2>/dev/null || echo \"$1\")\"\n"
            "    [[ -n \"${__MILK_INC_ONCE[$__rp]:-}\" ]] && return 0\n"
            "    __MILK_INC_ONCE[$__rp]=1\n"
            "    source \"$@\"\n"
            "}\n"
            "bookmark() {\n"
            "    if [[ \"${1:-}\" == \"run\" && $# -ge 2 ]]; then\n"
            "        local __bcmd\n"
            "        __bcmd=\"$(milkcall bookmark get \"$2\")\" || { echo \"${__bcmd}\"; return 1; "
            "}\n"
            "        eval \"${__bcmd}\"\n"
            "    else\n"
            "        milkcall bookmark \"$@\"\n"
            "    fi\n"
            "}\n"
            "__MILK_EXIT_REQ=0\n"
            "__MILK_EXIT_CODE=0\n"
            "exit() {\n"
            "    __MILK_EXIT_REQ=1\n"
            "    __MILK_EXIT_CODE=\"${1:-0}\"\n"
            "    printf '%d\\t1\\t%s\\0' \"${__MILK_EXIT_CODE}\" \"${PWD}\" >&199 2>/dev/null || "
            "true\n"
            "    builtin exit \"${__MILK_EXIT_CODE}\"\n"
            "}\n"
            "quit() { exit \"$@\"; }\n"
            "exitCLI() { exit \"$@\"; }\n"
            "while IFS= read -r -d '' __milk_cmd <&198; do\n"
            "    eval \"${__milk_cmd}\" 198<&-\n"
            "    __milk_rc=$?\n"
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
    g_cmd_pipe_w    = cmd_pipe[1];
    g_status_pipe_r = status_pipe[0];
    g_bash_pid      = pid;

    int dummy_exit = 0;
    milk_bash_engine_exec(":", &dummy_exit);
    return 0;
}

/**
 * @brief Translate and execute a command or multi-line block in the bash co-process.
 *
 * @param cmdline      Command line or multi-line script text
 * @param out_exit_req Optional: set to 1 if script invoked `exit`/`quit`
 * @return Exit status of the bash command (0 = success)
 */
int milk_bash_engine_exec(const char *cmdline, int *out_exit_req)
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

    int wstatus = 0;
    if (waitpid(g_bash_pid, &wstatus, WNOHANG) == g_bash_pid)
    {
        close(g_cmd_pipe_w);
        close(g_status_pipe_r);
        close(g_listen_fd);
        g_cmd_pipe_w    = -1;
        g_status_pipe_r = -1;
        g_listen_fd     = -1;
        g_bash_pid      = -1;
        if (milk_bash_engine_init() != 0)
        {
            return 1;
        }
    }

    if (milk_bash_has_unclosed_syntax(cmdline))
    {
        fprintf(stderr, "Syntax error: unexpected end of input\n");
        fflush(stderr);
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
            snprintf(with_e, old_len + 16, "set -e\n%s", translated);
            free(translated);
            translated = with_e;
        }
    }

    size_t tlen = strlen(translated) + 1;
    size_t sent = 0;
    while (sent < tlen)
    {
        ssize_t n = write(g_cmd_pipe_w, translated + sent, tlen - sent);
        if (n < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            free(translated);
            return 1;
        }
        sent += (size_t) n;
    }
    free(translated);

    char   sbuf[4096];
    size_t spos          = 0;
    int    shutdown_flag = 0;
    dcsigINT             = 0;

    while (spos < sizeof(sbuf) - 1)
    {
        if (dcsigINT)
        {
            dcsigINT = 0;
            if (g_bash_pid > 0)
            {
                kill(-g_bash_pid, SIGINT);
            }
        }

        struct pollfd pfds[2];
        pfds[0].fd      = g_listen_fd;
        pfds[0].events  = POLLIN;
        pfds[0].revents = 0;
        pfds[1].fd      = g_status_pipe_r;
        pfds[1].events  = POLLIN | POLLHUP | POLLERR;
        pfds[1].revents = 0;

        int pr = poll(pfds, 2, 50);
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
            milk_server_accept_and_handle(g_listen_fd, &shutdown_flag);
        }

        if (pfds[1].revents & (POLLIN | POLLHUP | POLLERR))
        {
            ssize_t r = read(g_status_pipe_r, sbuf + spos, sizeof(sbuf) - 1 - spos);
            if (r > 0)
            {
                int found_nul = 0;
                for (ssize_t k = 0; k < r; k++)
                {
                    if (sbuf[spos + (size_t) k] == '\0')
                    {
                        found_nul = 1;
                        break;
                    }
                }
                spos += (size_t) r;
                if (found_nul)
                {
                    break;
                }
            }
            else if (r == 0 || (r < 0 && errno != EINTR && errno != EAGAIN))
            {
                int st = 0;
                if (waitpid(g_bash_pid, &st, 0) == g_bash_pid)
                {
                    g_bash_pid = -1;
                }
                close(g_cmd_pipe_w);
                close(g_status_pipe_r);
                close(g_listen_fd);
                g_cmd_pipe_w    = -1;
                g_status_pipe_r = -1;
                g_listen_fd     = -1;
                int rc_err      = WIFEXITED(st) ? WEXITSTATUS(st) : 1;
                if (out_exit_req != NULL)
                {
                    *out_exit_req = 1;
                }
                cli_last_retval = (rc_err != 0) ? rc_err : 1;
                dcexitcode      = cli_last_retval;
                return cli_last_retval;
            }
        }
    }

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
    if (exreq != 0 || (dcerrorexit && rc != 0))
    {
        dcexitcode = rc;
    }
    cli_last_retval = rc;
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

    int rc = milk_bash_engine_exec(buf, out_exit_req);
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
        close(g_cmd_pipe_w);
        g_cmd_pipe_w = -1;
    }

    if (g_bash_pid > 0 && g_listen_fd >= 0 && g_status_pipe_r >= 0)
    {
        int shutdown_flag = 0;
        for (int iter = 0; iter < 200; iter++)
        {
            struct pollfd pfds[2];
            pfds[0].fd      = g_listen_fd;
            pfds[0].events  = POLLIN;
            pfds[0].revents = 0;
            pfds[1].fd      = g_status_pipe_r;
            pfds[1].events  = POLLIN | POLLHUP | POLLERR;
            pfds[1].revents = 0;

            int pr = poll(pfds, 2, 25);
            if (pr <= 0)
            {
                if (waitpid(g_bash_pid, NULL, WNOHANG) == g_bash_pid)
                {
                    g_bash_pid = -1;
                    break;
                }
                continue;
            }
            if (pfds[0].revents & POLLIN)
            {
                milk_server_accept_and_handle(g_listen_fd, &shutdown_flag);
            }
            if (pfds[1].revents & (POLLIN | POLLHUP | POLLERR))
            {
                char drain[256];
                if (read(g_status_pipe_r, drain, sizeof(drain)) <= 0)
                {
                    struct pollfd dpfd = { g_listen_fd, POLLIN, 0 };
                    while (poll(&dpfd, 1, 10) > 0 && (dpfd.revents & POLLIN))
                    {
                        milk_server_accept_and_handle(g_listen_fd, &shutdown_flag);
                    }
                    break;
                }
            }
        }
    }

    if (g_status_pipe_r >= 0)
    {
        close(g_status_pipe_r);
        g_status_pipe_r = -1;
    }
    if (g_bash_pid > 0)
    {
        waitpid(g_bash_pid, NULL, 0);
        g_bash_pid = -1;
    }
    milk_server_close_log(1);
    if (g_listen_fd >= 0)
    {
        close(g_listen_fd);
        g_listen_fd = -1;
    }
}

/**
 * @brief Abort an in-flight bash co-process command after a signal interrupt
 *        or fault-isolation `siglongjmp`.
 */
void milk_bash_engine_abort(void)
{
    if (g_bash_pid <= 0)
    {
        return;
    }

    kill(-g_bash_pid, SIGINT);

    int synced = 0;
    if (g_status_pipe_r >= 0)
    {
        int shutdown_flag = 0;
        for (int iter = 0; iter < 15; iter++)
        {
            struct pollfd pfds[2];
            pfds[0].fd      = g_listen_fd;
            pfds[0].events  = (g_listen_fd >= 0) ? POLLIN : 0;
            pfds[0].revents = 0;
            pfds[1].fd      = g_status_pipe_r;
            pfds[1].events  = POLLIN | POLLHUP | POLLERR;
            pfds[1].revents = 0;

            int pr = poll(pfds, 2, 10);
            if (pr <= 0)
            {
                continue;
            }
            if (pfds[0].revents & POLLIN)
            {
                milk_server_accept_and_handle(g_listen_fd, &shutdown_flag);
            }
            if (pfds[1].revents & (POLLIN | POLLHUP | POLLERR))
            {
                char    drain[512];
                ssize_t r = read(g_status_pipe_r, drain, sizeof(drain));
                if (r > 0)
                {
                    for (ssize_t k = 0; k < r; k++)
                    {
                        if (drain[k] == '\0')
                        {
                            synced = 1;
                            break;
                        }
                    }
                    if (synced)
                    {
                        break;
                    }
                }
                else
                {
                    break;
                }
            }
        }
    }

    if (!synced)
    {
        kill(-g_bash_pid, SIGKILL);
        waitpid(g_bash_pid, NULL, 0);
        g_bash_pid = -1;
        if (g_cmd_pipe_w >= 0)
        {
            close(g_cmd_pipe_w);
            g_cmd_pipe_w = -1;
        }
        if (g_status_pipe_r >= 0)
        {
            close(g_status_pipe_r);
            g_status_pipe_r = -1;
        }
        if (g_listen_fd >= 0)
        {
            close(g_listen_fd);
            g_listen_fd = -1;
        }
    }
}
