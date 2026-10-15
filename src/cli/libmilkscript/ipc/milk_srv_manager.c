// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_srv_manager.c
 * @brief Multi-server lifecycle manager (`@server status|list|start|use|local|stop|attach`).
 *
 * Role & Goal:
 *   Implements discovery, status inspection, named background daemon spawning
 *   (or optional `tmux` session hosting), active-server switching, and clean
 *   shutdown across all `milk-cli --server` instances owned by the user.
 *   Uses `milk_ipc_common.h` for socket handshakes, binary resolution, shell
 *   variable propagation, and double-fork daemon launching.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <ctype.h>
#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "milk_ipc_common.h"
#include "milk_srvlog.h"

extern void milk_srvlog_get_dir(char *out, size_t outsz);

#define C_RST "\033[0m"
#define C_TITLE "\033[1;36m"
#define C_HDR "\033[1;35m"
#define C_CMD "\033[32m"
#define C_DIM "\033[2m"

static int use_color_output(void)
{
    const char *nc = getenv("NO_COLOR");
    if (nc != NULL && nc[0] != '\0')
    {
        return 0;
    }
    return isatty(STDOUT_FILENO);
}

/**
 * @brief Start a persistent named `milk-cli --server` (daemon or inside `tmux`).
 *
 * @param srv_name Logical server name (`[A-Za-z0-9_-]+`)
 * @param use_tmux If non-zero, host inside `tmux` session `milk_srv_<srv_name>`
 * @param hooks    Optional shell callback hooks
 * @return 0 on success, 1 on error
 */
static int start_named_server(const char *srv_name, int use_tmux, const milk_client_hooks_t *hooks)
{
    if (srv_name == NULL || srv_name[0] == '\0')
    {
        fprintf(stderr, "Usage: @server start <name> [--tmux]\n");
        return 1;
    }
    for (size_t i = 0; srv_name[i] != '\0'; i++)
    {
        char c = srv_name[i];
        if (!isalnum((unsigned char) c) && c != '_' && c != '-')
        {
            fprintf(stderr, "ERROR: @server: invalid server name '%s' (use [A-Za-z0-9_-])\n",
                    srv_name);
            return 1;
        }
    }

    char sockpath[256];
    milk_srvlog_make_sock(srv_name, sockpath, sizeof(sockpath));

    if (milk_ipc_ping_socket(sockpath))
    {
        char lpath[512];
        if (milk_srvlog_shm_path(sockpath, lpath, sizeof(lpath)) == 0 && access(lpath, R_OK) == 0)
        {
            return 0;
        }
        milk_ipc_shutdown_socket(sockpath);
        for (int w = 0; w < 50 && milk_ipc_ping_socket(sockpath); w++)
        {
            struct timespec req = { 0, 5000000L };
            nanosleep(&req, NULL);
        }
    }

    char binpath[512];
    milk_ipc_resolve_cli_bin(hooks, binpath, sizeof(binpath));

    if (use_tmux)
    {
        char tmux_sess[64];
        snprintf(tmux_sess, sizeof(tmux_sess), "milk_srv_%s", srv_name);

        char shmdir[256];
        milk_srvlog_get_dir(shmdir, sizeof(shmdir));

        char cmd[2048];
        snprintf(cmd, sizeof(cmd),
                 "tmux kill-session -t %s 2>/dev/null || true; "
                 "tmux new-session -d -s %s \"MILK_SHM_DIR='%s' MILK_SRV_NAMED=1 "
                 "MILK_SRV_TMUX='%s' '%s' -q --server '%s'\"",
                 tmux_sess, tmux_sess, shmdir, tmux_sess, binpath, sockpath);
        if (system(cmd) != 0)
        {
            fprintf(stderr, "ERROR: @server: failed to launch tmux session '%s'\n", tmux_sess);
            return 1;
        }

        for (int iter = 0; iter < 500; iter++)
        {
            if (milk_ipc_ping_socket(sockpath))
            {
                return 0;
            }
            struct timespec req = { 0, 10000000L }; /* 10 ms */
            nanosleep(&req, NULL);
        }
        fprintf(stderr, "ERROR: @server: timed out waiting for tmux server '%s'\n", srv_name);
        return 1;
    }

    return (milk_ipc_spawn_daemon(binpath, sockpath, -1, -1, 1) == 0) ? 0 : 1;
}

static void format_uptime(time_t start_sec, char *out, size_t outsz)
{
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    long diff = (start_sec > 0 && now.tv_sec >= start_sec) ? (long) (now.tv_sec - start_sec) : 0;
    if (diff < 60)
    {
        snprintf(out, outsz, "%lds", diff);
    }
    else if (diff < 3600)
    {
        snprintf(out, outsz, "%ldm%02lds", diff / 60, diff % 60);
    }
    else
    {
        snprintf(out, outsz, "%ldh%02ldm", diff / 3600, (diff % 3600) / 60);
    }
}

/**
 * @brief Print detailed status for the active or specified server (`@server status`).
 */
static int cmd_server_status(const char *target_arg, const milk_client_hooks_t *hooks)
{
    char cur_sock[256];
    milk_ipc_resolve_sockpath(hooks, cur_sock, sizeof(cur_sock), 0);

    char sockpath[256];
    if (target_arg != NULL && target_arg[0] != '\0')
    {
        milk_srvlog_make_sock(target_arg, sockpath, sizeof(sockpath));
    }
    else
    {
        snprintf(sockpath, sizeof(sockpath), "%s", cur_sock);
    }

    unsigned int uid = 0;
    char         sname[64];
    milk_srvlog_parse_sock(sockpath, &uid, sname, sizeof(sname));

    int  alive = milk_ipc_ping_socket(sockpath);
    char path[512];
    milk_srvlog_shm_path(sockpath, path, sizeof(path));

    milk_srvlog_shm_t snap;
    int               has_shm = 0;
    int               fd      = open(path, O_RDONLY, 0600);
    if (fd >= 0)
    {
        milk_srvlog_shm_t *shm = (milk_srvlog_shm_t *) mmap(NULL, sizeof(milk_srvlog_shm_t),
                                                            PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (shm != MAP_FAILED)
        {
            if (shm->magic == MILK_SRVLOG_MAGIC)
            {
                snap    = *shm;
                has_shm = 1;
            }
            munmap(shm, sizeof(milk_srvlog_shm_t));
        }
    }

    int         color = use_color_output();
    const char *c_hdr = color ? C_TITLE : "";
    const char *c_ok  = color ? C_CMD : "";
    const char *c_dim = color ? C_DIM : "";
    const char *c_rst = color ? C_RST : "";

    const char *mode_str =
        milk_ipc_is_numeric_str(sname) ? "private (per-shell)" : "named (persistent)";
    if (has_shm)
    {
        if (snap.flags & MILK_SRVFLAG_INPROC)
        {
            mode_str = "in-process (milk-cli)";
        }
        else if (snap.flags & MILK_SRVFLAG_TMUX)
        {
            mode_str = "named (tmux)";
        }
        else if (snap.flags & MILK_SRVFLAG_NAMED)
        {
            mode_str = "named (persistent)";
        }
    }

    printf("%s[milk server status]%s\n", c_hdr, c_rst);
    printf("  Name      : %s%s\n", sname,
           (strcmp(sockpath, cur_sock) == 0) ? " (current session)" : "");
    printf("  Socket    : %s\n", sockpath);
    printf("  Mode      : %s\n", mode_str);

    if (alive)
    {
        char upbuf[32] = "-";
        if (has_shm)
        {
            format_uptime(snap.ts_created.tv_sec, upbuf, sizeof(upbuf));
        }
        printf("  State     : %sRUNNING%s (PID %d, uptime %s)\n", c_ok, c_rst,
               has_shm ? (int) snap.srv_pid : 0, upbuf);
    }
    else
    {
        printf("  State     : %sIDLE%s (not started yet — starts lazily on first command)\n", c_dim,
               c_rst);
    }

    if (has_shm && (snap.flags & MILK_SRVFLAG_TMUX) && snap.tmux_session[0] != '\0')
    {
        printf("  Tmux      : %s\n", snap.tmux_session);
    }
    if (has_shm && snap.cwd[0] != '\0')
    {
        printf("  Workdir   : %s\n", snap.cwd);
    }
    if (has_shm)
    {
        printf("  Resources : %u modules, %u local images, %u vars\n", snap.nb_modules,
               snap.nb_images, snap.nb_vars);
        printf("  Log       : %lu commands (%lu errors)\n", (unsigned long) snap.total_cmds,
               (unsigned long) snap.total_errors);
    }
    return 0;
}

/**
 * @brief List all active `milk-cli` servers for the current user (`@server list`).
 */
static int cmd_server_list(int json_mode, const milk_client_hooks_t *hooks)
{
    char cur_sock[256];
    milk_ipc_resolve_sockpath(hooks, cur_sock, sizeof(cur_sock), 0);

    char shmdir[256];
    milk_srvlog_get_dir(shmdir, sizeof(shmdir));

    char pfx[64];
    snprintf(pfx, sizeof(pfx), "milk-srv.%u.", (unsigned) getuid());
    size_t pfx_len = strlen(pfx);

    DIR *dp = opendir(shmdir);
    if (dp == NULL)
    {
        if (json_mode)
        {
            puts("[]");
        }
        return 0;
    }

    int color = use_color_output();
    if (!json_mode)
    {
        printf("%s%-3s %-14s %-9s %-8s %-10s %-7s %-6s %-6s %-6s %s%s\n", color ? C_HDR : "", "ACT",
               "NAME", "STATE", "PID", "MODE", "UPTIME", "IMAGES", "VARS", "CMDS", "SOCKET / TMUX",
               color ? C_RST : "");
    }
    else
    {
        putchar('[');
    }

    int            count     = 0;
    int            found_cur = 0;
    struct dirent *de;

    while ((de = readdir(dp)) != NULL)
    {
        size_t dlen = strlen(de->d_name);
        if (dlen <= pfx_len + 8 || strncmp(de->d_name, pfx, pfx_len) != 0 ||
            strcmp(de->d_name + dlen - 8, ".log.shm") != 0)
        {
            continue;
        }

        char fullpath[512];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", shmdir, de->d_name);
        int fd = open(fullpath, O_RDONLY, 0600);
        if (fd < 0)
        {
            continue;
        }
        milk_srvlog_shm_t *shm = (milk_srvlog_shm_t *) mmap(NULL, sizeof(milk_srvlog_shm_t),
                                                            PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (shm == MAP_FAILED)
        {
            continue;
        }

        if (shm->magic != MILK_SRVLOG_MAGIC)
        {
            munmap(shm, sizeof(milk_srvlog_shm_t));
            continue;
        }

        milk_srvlog_shm_t snap = *shm;
        munmap(shm, sizeof(milk_srvlog_shm_t));

        int pid_alive   = (snap.srv_pid > 0 && kill(snap.srv_pid, 0) == 0);
        int owner_alive = (snap.owner_pid > 0 && kill(snap.owner_pid, 0) == 0);
        int sock_alive  = pid_alive ? milk_ipc_ping_socket(snap.sockpath) : 0;

        if (!sock_alive && !owner_alive && strcmp(snap.sockpath, cur_sock) != 0)
        {
            unlink(fullpath);
            continue;
        }

        int is_cur = (strcmp(snap.sockpath, cur_sock) == 0);
        if (is_cur)
        {
            found_cur = 1;
        }

        const char *mode_s =
            (snap.flags & MILK_SRVFLAG_INPROC)
                ? "inproc"
                : ((snap.flags & MILK_SRVFLAG_TMUX)
                       ? "tmux"
                       : ((snap.flags & MILK_SRVFLAG_NAMED) ? "named" : "private"));
        char upbuf[32] = "-";
        if (sock_alive)
        {
            format_uptime(snap.ts_created.tv_sec, upbuf, sizeof(upbuf));
        }

        if (json_mode)
        {
            printf("%s{\"active\":%s,\"name\":\"%s\",\"state\":\"%s\",\"pid\":%d,"
                   "\"mode\":\"%s\",\"images\":%u,\"vars\":%u,\"cmds\":%lu,\"errors\":%lu,"
                   "\"socket\":\"%s\",\"tmux\":\"%s\"}",
                   (count > 0) ? "," : "", is_cur ? "true" : "false", snap.srv_name,
                   sock_alive ? "RUNNING" : "IDLE", (int) snap.srv_pid, mode_s, snap.nb_images,
                   snap.nb_vars, (unsigned long) snap.total_cmds, (unsigned long) snap.total_errors,
                   snap.sockpath, snap.tmux_session);
        }
        else
        {
            printf("%-3s %-14s %-9s %-8d %-10s %-7s %-6u %-6u %-6lu %s%s%s\n", is_cur ? "*" : "",
                   snap.srv_name, sock_alive ? "RUNNING" : "IDLE", (int) snap.srv_pid, mode_s,
                   upbuf, snap.nb_images, snap.nb_vars, (unsigned long) snap.total_cmds,
                   snap.sockpath, (snap.tmux_session[0] != '\0') ? "  tmux:" : "",
                   snap.tmux_session);
        }
        count++;
    }
    closedir(dp);

    if (!found_cur)
    {
        unsigned int uid = 0;
        char         sname[64];
        milk_srvlog_parse_sock(cur_sock, &uid, sname, sizeof(sname));
        int         alive  = milk_ipc_ping_socket(cur_sock);
        const char *mode_s = milk_ipc_is_numeric_str(sname) ? "private" : "named";
        if (json_mode)
        {
            printf("%s{\"active\":true,\"name\":\"%s\",\"state\":\"%s\",\"pid\":0,"
                   "\"mode\":\"%s\",\"images\":0,\"vars\":0,\"cmds\":0,\"errors\":0,"
                   "\"socket\":\"%s\",\"tmux\":\"\"}",
                   (count > 0) ? "," : "", sname, alive ? "RUNNING" : "IDLE", mode_s, cur_sock);
        }
        else
        {
            printf("%-3s %-14s %-9s %-8s %-10s %-7s %-6u %-6u %-6u %s\n", "*", sname,
                   alive ? "RUNNING" : "IDLE", "-", mode_s, "-", 0U, 0U, 0U, cur_sock);
        }
    }

    if (json_mode)
    {
        puts("]");
    }
    return 0;
}

/**
 * @brief Stop a single server by socket path, killing its tmux session if any.
 */
static int stop_one_server(const char *sockpath)
{
    char path[512];
    char tmux_sess[64] = "";
    if (milk_srvlog_shm_path(sockpath, path, sizeof(path)) == 0)
    {
        int fd = open(path, O_RDONLY, 0600);
        if (fd >= 0)
        {
            milk_srvlog_shm_t *shm = (milk_srvlog_shm_t *) mmap(NULL, sizeof(milk_srvlog_shm_t),
                                                                PROT_READ, MAP_SHARED, fd, 0);
            close(fd);
            if (shm != MAP_FAILED)
            {
                if (shm->magic == MILK_SRVLOG_MAGIC && (shm->flags & MILK_SRVFLAG_TMUX))
                {
                    snprintf(tmux_sess, sizeof(tmux_sess), "%s", shm->tmux_session);
                }
                munmap(shm, sizeof(milk_srvlog_shm_t));
            }
        }
    }

    if (milk_ipc_shutdown_socket(sockpath))
    {
        for (int w = 0; w < 40 && milk_ipc_ping_socket(sockpath); w++)
        {
            struct timespec req = { 0, 5000000L };
            nanosleep(&req, NULL);
        }
    }

    if (tmux_sess[0] != '\0')
    {
        char cmd[256];
        snprintf(cmd, sizeof(cmd), "tmux kill-session -t '%s' 2>/dev/null || true", tmux_sess);
        if (system(cmd) < 0)
        {
        }
    }
    if (path[0] != '\0')
    {
        unlink(path);
    }
    return 0;
}

static void print_server_help(void)
{
    printf("Usage: @server <subcommand> [args...]\n\n");
    printf("Manage, monitor, and switch milk-cli servers from bash.\n\n");
    printf("Subcommands:\n");
    printf("  status [name]           Show server status, resources, and log counters (default)\n");
    printf("  list, ls [--json]       List all active milk-cli servers for the current user\n");
    printf("  start <name> [--tmux]   Start a persistent named server (optionally in tmux)\n");
    printf("  use <name> [--tmux]     Switch current shell to named server <name>\n");
    printf("  local, detach           Switch current shell back to its private PID server\n");
    printf("  stop [name|--all]       Shut down a named server (or all servers with --all)\n");
    printf("  attach [name]           Attach to server's tmux session or live-tail its log\n");
    printf("  log [options] [pattern] Query, search, or live-tail the server command log (@log)\n");
}

static int handle_server_start_or_use(const char                *sub,
                                      int                        argc,
                                      char                     **argv,
                                      const milk_client_hooks_t *hooks)
{
    int         no_switch = 0;
    int         use_tmux  = 0;
    const char *sname     = NULL;
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--tmux") == 0 || strcmp(argv[i], "-tmux") == 0)
        {
            use_tmux = 1;
        }
        else if (strcmp(argv[i], "--no-switch") == 0 || strcmp(argv[i], "-d") == 0)
        {
            no_switch = 1;
        }
        else if (sname == NULL)
        {
            sname = argv[i];
        }
    }
    if (sname == NULL || sname[0] == '\0')
    {
        fprintf(stderr, "Usage: @server %s <name> [--tmux] [-d|--no-switch]\n", sub);
        return 1;
    }
    if (start_named_server(sname, use_tmux, hooks) != 0)
    {
        return 1;
    }
    char sockpath[256];
    milk_srvlog_make_sock(sname, sockpath, sizeof(sockpath));
    if (!no_switch)
    {
        milk_ipc_set_var(hooks, "MILKCLI_SOCK", sockpath, 1);
        milk_ipc_set_var(hooks, "_MILK_SERVER_NAME", sname, 1);
        printf("Switched to milk server '%s' (%s)%s.\n", sname, sockpath,
               use_tmux ? " [tmux]" : "");
    }
    else
    {
        printf("Started milk server '%s' (%s)%s.\n", sname, sockpath, use_tmux ? " [tmux]" : "");
    }
    return 0;
}

static int handle_server_stop(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    if (argc >= 2 && strcmp(argv[1], "--all") == 0)
    {
        char shmdir[256];
        milk_srvlog_get_dir(shmdir, sizeof(shmdir));
        char pfx[64];
        snprintf(pfx, sizeof(pfx), "milk-srv.%u.", (unsigned) getuid());
        size_t pfx_len = strlen(pfx);
        DIR   *dp      = opendir(shmdir);
        if (dp != NULL)
        {
            struct dirent *de;
            while ((de = readdir(dp)) != NULL)
            {
                size_t dlen = strlen(de->d_name);
                if (dlen > pfx_len + 8 && strncmp(de->d_name, pfx, pfx_len) == 0 &&
                    strcmp(de->d_name + dlen - 8, ".log.shm") == 0)
                {
                    char full[512];
                    snprintf(full, sizeof(full), "%s/%s", shmdir, de->d_name);
                    int fd = open(full, O_RDONLY, 0600);
                    if (fd >= 0)
                    {
                        milk_srvlog_shm_t *shm = (milk_srvlog_shm_t *) mmap(
                            NULL, sizeof(milk_srvlog_shm_t), PROT_READ, MAP_SHARED, fd, 0);
                        close(fd);
                        if (shm != MAP_FAILED)
                        {
                            char sp[128];
                            snprintf(sp, sizeof(sp), "%s", shm->sockpath);
                            munmap(shm, sizeof(milk_srvlog_shm_t));
                            if (sp[0] != '\0')
                            {
                                stop_one_server(sp);
                            }
                        }
                    }
                    unlink(full);
                }
            }
            closedir(dp);
        }
        return 0;
    }

    char cur_sock[256];
    milk_ipc_resolve_sockpath(hooks, cur_sock, sizeof(cur_sock), 0);
    char target_sock[256];
    if (argc >= 2 && argv[1][0] != '\0')
    {
        milk_srvlog_make_sock(argv[1], target_sock, sizeof(target_sock));
    }
    else
    {
        snprintf(target_sock, sizeof(target_sock), "%s", cur_sock);
    }
    stop_one_server(target_sock);

    if (strcmp(target_sock, cur_sock) == 0)
    {
        const char *orig_sock = milk_ipc_get_var(hooks, "_MILK_LOCAL_SOCK");
        if (orig_sock != NULL && orig_sock[0] != '\0')
        {
            milk_ipc_set_var(hooks, "MILKCLI_SOCK", orig_sock, 1);
        }
        milk_ipc_set_var(hooks, "_MILK_SERVER_NAME", "", 1);
    }
    return 0;
}

static int handle_server_attach(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    char cur_sock[256];
    milk_ipc_resolve_sockpath(hooks, cur_sock, sizeof(cur_sock), 0);
    char target_sock[256];
    if (argc >= 2 && argv[1][0] != '\0')
    {
        milk_srvlog_make_sock(argv[1], target_sock, sizeof(target_sock));
    }
    else
    {
        snprintf(target_sock, sizeof(target_sock), "%s", cur_sock);
    }

    char path[512];
    char tmux_sess[64] = "";
    if (milk_srvlog_shm_path(target_sock, path, sizeof(path)) == 0)
    {
        int fd = open(path, O_RDONLY, 0600);
        if (fd >= 0)
        {
            milk_srvlog_shm_t *shm = (milk_srvlog_shm_t *) mmap(NULL, sizeof(milk_srvlog_shm_t),
                                                                PROT_READ, MAP_SHARED, fd, 0);
            close(fd);
            if (shm != MAP_FAILED)
            {
                if (shm->magic == MILK_SRVLOG_MAGIC && (shm->flags & MILK_SRVFLAG_TMUX))
                {
                    snprintf(tmux_sess, sizeof(tmux_sess), "%s", shm->tmux_session);
                }
                munmap(shm, sizeof(milk_srvlog_shm_t));
            }
        }
    }

    if (tmux_sess[0] != '\0')
    {
        char cmd[256];
        snprintf(cmd, sizeof(cmd), "tmux attach -t '%s'", tmux_sess);
        return (system(cmd) == 0) ? 0 : 1;
    }
    char *f_argv[2] = { (char *) "-f", NULL };
    return milk_srvlog_cmd_log(target_sock, 1, f_argv, hooks);
}

/**
 * @brief Execute the `@server` management command (`status`, `list`, `start`, `use`, `local`, `stop`, `attach`).
 */
int milk_srvlog_cmd_server(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    const char *sub = (argc > 0 && argv[0] != NULL) ? argv[0] : "status";

    if (strcmp(sub, "-h") == 0 || strcmp(sub, "--help") == 0 || strcmp(sub, "help") == 0)
    {
        print_server_help();
        return 0;
    }
    if (strcmp(sub, "status") == 0)
    {
        return cmd_server_status((argc >= 2) ? argv[1] : NULL, hooks);
    }
    if (strcmp(sub, "list") == 0 || strcmp(sub, "ls") == 0)
    {
        return cmd_server_list(argc >= 2 && strcmp(argv[1], "--json") == 0, hooks);
    }
    if (strcmp(sub, "log") == 0)
    {
        char cur_sock[256];
        milk_ipc_resolve_sockpath(hooks, cur_sock, sizeof(cur_sock), 0);
        return milk_srvlog_cmd_log(cur_sock, argc - 1, argv + 1, hooks);
    }
    if (strcmp(sub, "start") == 0 || strcmp(sub, "use") == 0)
    {
        return handle_server_start_or_use(sub, argc, argv, hooks);
    }
    if (strcmp(sub, "local") == 0 || strcmp(sub, "detach") == 0)
    {
        const char *orig_sock = milk_ipc_get_var(hooks, "_MILK_LOCAL_SOCK");
        char        local_sock[256];
        if (orig_sock != NULL && orig_sock[0] != '\0')
        {
            snprintf(local_sock, sizeof(local_sock), "%s", orig_sock);
        }
        else
        {
            pid_t ref_pid = (hooks != NULL && hooks->get_var != NULL) ? getpid() : getppid();
            snprintf(local_sock, sizeof(local_sock), "@milk-cli-%u-%d", (unsigned) getuid(),
                     (int) ref_pid);
        }
        milk_ipc_set_var(hooks, "MILKCLI_SOCK", local_sock, 1);
        milk_ipc_set_var(hooks, "_MILK_SERVER_NAME", "", 1);
        printf("Switched to local shell server (%s).\n", local_sock);
        return 0;
    }
    if (strcmp(sub, "stop") == 0)
    {
        return handle_server_stop(argc, argv, hooks);
    }
    if (strcmp(sub, "attach") == 0)
    {
        return handle_server_attach(argc, argv, hooks);
    }

    fprintf(stderr, "ERROR: @server: unknown subcommand '%s'\n", sub);
    print_server_help();
    return 1;
}
