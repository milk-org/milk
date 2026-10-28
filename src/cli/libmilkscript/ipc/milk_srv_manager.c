// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_srv_manager.c
 * @brief Sessions: the `@session` keyword (status, ls, new, use, leave, who, end, prune, log).
 *
 * A session is a milk-cli server: a shell's own (named after the shell's PID, started on its first
 * command), milk-cli's or milk-script's (in its process), or a named one (`@session new NAME`), detached and shared by
 * the shells that use it. A shell records the session it uses in an override file
 * (milk_ipc_set_var()); units record the session that started them in their run.info. Servers start
 * and stop through milk_ipc_common.h and milk_detach.h.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "fps_unit.h"
#include "fps_unit_state.h"
#include "milk_detach.h"
#include "milk_ipc_common.h"
#include "milk_os.h"
#include "milk_srvlog.h"

extern void milk_srvlog_get_dir(char *out, size_t outsz);

#define C_RST "\033[0m"
#define C_TITLE "\033[1;36m"
#define C_HDR "\033[1;35m"
#define C_CMD "\033[32m"
#define C_DIM "\033[2m"

/* What @session reads of a server's log: its header, and when it last ran a command */
typedef struct
{
    pid_t           srv_pid;
    pid_t           owner_pid;
    uint32_t        flags;
    struct timespec ts_created;
    struct timespec ts_last; /* zero: no command yet */
    char            name[64];
    char            sockpath[128];
    char            owner_ident[64];
    char            cwd[256];
    uint32_t        idle_s;
    uint32_t        nb_modules;
    uint32_t        nb_images;
    uint32_t        nb_vars;
    uint64_t        total_cmds;
    uint64_t        total_errors;
} session_info_t;

/**
 * use_color_output - Check whether ANSI color escape sequences should be emitted
 *
 * Respects the `NO_COLOR` environment variable convention and checks whether
 * `STDOUT_FILENO` is attached to an interactive terminal.
 *
 * Return: 1 if color output should be used, 0 otherwise
 */
static int use_color_output(void)
{
    const char *nc = getenv("NO_COLOR");
    if (nc != NULL && nc[0] != '\0')
    {
        return 0;
    }
    return isatty(STDOUT_FILENO);
}

static int pid_alive(pid_t pid)
{
    return pid > 0 && (kill(pid, 0) == 0 || errno == EPERM);
}

/* Read the server log at @path. Return: 1, or 0 if there is none */
static int session_read_log(const char *path, session_info_t *si)
{
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
    {
        return 0;
    }
    milk_srvlog_shm_t *shm =
        (milk_srvlog_shm_t *) mmap(NULL, sizeof(milk_srvlog_shm_t), PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (shm == MAP_FAILED)
    {
        return 0;
    }
    int ok = (shm->magic == MILK_SRVLOG_MAGIC && shm->version == MILK_SRVLOG_VERSION);
    if (ok)
    {
        memset(si, 0, sizeof(*si));
        si->srv_pid      = shm->srv_pid;
        si->owner_pid    = shm->owner_pid;
        si->flags        = shm->flags;
        si->idle_s       = shm->idle_s;
        si->ts_created   = shm->ts_created;
        si->nb_modules   = shm->nb_modules;
        si->nb_images    = shm->nb_images;
        si->nb_vars      = shm->nb_vars;
        si->total_cmds   = shm->total_cmds;
        si->total_errors = shm->total_errors;
        snprintf(si->name, sizeof(si->name), "%s", shm->srv_name);
        snprintf(si->sockpath, sizeof(si->sockpath), "%s", shm->sockpath);
        snprintf(si->owner_ident, sizeof(si->owner_ident), "%s", shm->owner_ident);
        snprintf(si->cwd, sizeof(si->cwd), "%s", shm->cwd);
        uint64_t seq = shm->write_seq;
        if (seq > 0)
        {
            si->ts_last = shm->entries[(seq - 1) % MILK_SRVLOG_CAP].ts_start;
        }
    }
    munmap(shm, sizeof(milk_srvlog_shm_t));
    return ok;
}

/* The log of session @sockpath. Return: 1, or 0 if there is none */
static int session_read(const char *sockpath, session_info_t *si)
{
    char path[512];

    return milk_srvlog_shm_path(sockpath, path, sizeof(path)) == 0 && session_read_log(path, si);
}

/* @diff seconds, compact: 42s, 5m12s, 2h03m; format_uptime(): the time since @start_sec (CLOCK_REALTIME) */
static void format_secs(long diff, char *out, size_t outsz)
{
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

static void format_uptime(time_t start_sec, char *out, size_t outsz)
{
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    format_secs((start_sec > 0 && now.tv_sec >= start_sec) ? (long) (now.tv_sec - start_sec) : 0,
                out, outsz);
}

/* Its owner: 1 alive, 0 gone (the same pid now running another program counts as gone), -1 none */
static int owner_alive(const session_info_t *si)
{
    MILKDETACH_ID      id    = { 0 };
    unsigned long long ticks = 0;

    if (si->owner_pid <= 0)
    {
        return -1;
    }
    if (sscanf(si->owner_ident, "%llu %36s", &ticks, id.boot_id) != 2)
    {
        return pid_alive(si->owner_pid);
    }
    id.pid         = si->owner_pid;
    id.start_ticks = (uint64_t) ticks;
    return milkdetach_id_alive(&id);
}

/* A duration: a number of seconds, or with s, m, h or d. Return: 1, or 0 if @s is not one (> 0) */
static int parse_duration(const char *s, unsigned *out)
{
    char  *end = NULL;
    double v   = (s != NULL) ? strtod(s, &end) : 0.0;
    double mul = 1.0;

    if (s == NULL || end == s)
    {
        return 0;
    }
    if (*end != '\0')
    {
        const char         *units = "smhd";
        const char         *u     = strchr(units, *end);
        static const double f[]   = { 1.0, 60.0, 3600.0, 86400.0 };
        if (u == NULL || end[1] != '\0')
        {
            return 0;
        }
        mul = f[u - units];
    }
    v *= mul;
    if (!(v >= 1.0 && v <= 4294967295.0))
    {
        return 0;
    }
    *out = (unsigned) v;
    return 1;
}

/* A session name: [A-Za-z0-9_-]; a number names a shell's own session, accepted only with @numbers.
 * @who starts the messages ("@session new") */
static int session_name_ok(const char *name, const char *who, int numbers)
{
    if (name == NULL || name[0] == '\0')
    {
        fprintf(stderr, "Usage: %s <name>\n", who);
        return 0;
    }
    for (size_t i = 0; name[i] != '\0'; i++)
    {
        if (!isalnum((unsigned char) name[i]) && name[i] != '_' && name[i] != '-')
        {
            fprintf(stderr, "%s: invalid name '%s' (use [A-Za-z0-9_-])\n", who, name);
            return 0;
        }
    }
    if (!numbers && milk_ipc_is_numeric_str(name))
    {
        fprintf(stderr, "%s: '%s' is a number, the name of a shell's own session\n", who, name);
        return 0;
    }
    if (strlen(name) > 63) /* what the session's files, its status and milk-cli's prompt hold */
    {
        fprintf(stderr, "%s: the name '%.20s...' is longer than 63 characters\n", who, name);
        return 0;
    }
    return 1;
}

/* Where shells record the session they use (milk_ipc_set_var()) */
static const char *session_tmpdir(void)
{
    const char *t = getenv("TMPDIR");
    return (t != NULL && t[0] != '\0') ? t : "/tmp";
}

/* 1 if the shell that wrote override file @suffix (<pid>_<start ticks>) still runs: the same pid with
 * another start is a later process. Files from before the start time was recorded have the pid only. */
static int shell_alive(const char *suffix, pid_t *pid)
{
    char    *end   = NULL;
    uint64_t ticks = 0;

    *pid = (pid_t) strtol(suffix, &end, 10);
    if (*pid <= 0 || end == suffix)
    {
        return 0;
    }
    unsigned long long want = (*end == '_') ? strtoull(end + 1, NULL, 10) : 0;
    if (want == 0)
    {
        return pid_alive(*pid);
    }
    return milkos_proc_start_ticks(*pid, &ticks) == 0 && ticks == (uint64_t) want;
}

/*
 * The shells that use session @sockpath (NULL: none), from their override files, at most @max in
 * @pids; with @pruned, the files of exited shells are removed and counted there. Return: how many
 */
static int session_shells(const char *sockpath, pid_t *pids, int max, int *pruned)
{
    char   pfx[64];
    int    n   = 0;
    DIR   *dir = opendir(session_tmpdir());
    size_t plen;

    if (dir == NULL)
    {
        return 0;
    }
    plen = (size_t) snprintf(pfx, sizeof(pfx), ".milk_active_sock_%u_", (unsigned) getuid());
    for (struct dirent *de; (de = readdir(dir)) != NULL;)
    {
        char  path[PATH_MAX];
        char  sock[256] = "";
        pid_t pid       = 0;
        if (strncmp(de->d_name, pfx, plen) != 0)
        {
            continue;
        }
        snprintf(path, sizeof(path), "%s/%s", session_tmpdir(), de->d_name);
        if (!shell_alive(de->d_name + plen, &pid))
        {
            if (pruned != NULL && unlink(path) == 0)
            {
                (*pruned)++;
            }
            continue;
        }
        FILE *f = (sockpath != NULL) ? fopen(path, "r") : NULL;
        if (f != NULL)
        {
            if (fgets(sock, sizeof(sock), f) != NULL && strcmp(sock, sockpath) == 0)
            {
                if (n < max && pids != NULL)
                {
                    pids[n] = pid;
                }
                n++;
            }
            fclose(f);
        }
    }
    closedir(dir);
    return n;
}

int milk_srvlog_session_shells(const char *sockpath)
{
    return session_shells(sockpath, NULL, 0, NULL);
}

/* The live processes of units that session @name started (their run.info or conf.info names it);
 * with @stop, they are stopped. Return: how many */
static int session_units(const char *name, int stop)
{
    char base[PATH_MAX];
    int  n = 0;
    DIR *dir;

    if (name[0] == '\0' || fps_unit_base_dir(base, sizeof(base)) != 0 ||
        (dir = opendir(base)) == NULL)
    {
        return 0;
    }
    for (struct dirent *de; (de = readdir(dir)) != NULL;)
    {
        char   unit[STRINGMAXLEN_FPS_NAME];
        size_t len = strlen(de->d_name);
        if (len <= 6 || strcmp(de->d_name + len - 6, ".fps.d") != 0 || len - 6 >= sizeof(unit))
        {
            continue;
        }
        snprintf(unit, sizeof(unit), "%.*s", (int) (len - 6), de->d_name);
        for (int r = FPS_UNIT_RUN; r <= FPS_UNIT_CONF; r++)
        {
            FPS_UNIT_INFO info;
            if (fps_unit_info_read(unit, (FPS_UNIT_ROLE) r, &info) == 0 &&
                strcmp(info.session, name) == 0 && milkdetach_id_alive(&info.id))
            {
                n++;
                if (stop)
                {
                    fps_unit_stop(unit, (FPS_UNIT_ROLE) r, 0);
                }
            }
        }
    }
    closedir(dir);
    return n;
}

int milk_srvlog_session_list_units(const char *name, FILE *fp)
{
    char base[PATH_MAX];
    int  n = 0;
    DIR *dir;

    if (name[0] == '\0' || fps_unit_base_dir(base, sizeof(base)) != 0 ||
        (dir = opendir(base)) == NULL)
    {
        return 0;
    }
    for (struct dirent *de; (de = readdir(dir)) != NULL;)
    {
        char          unit[STRINGMAXLEN_FPS_NAME];
        FPS_UNIT_INFO run;
        FPS_UNIT_INFO conf;
        size_t        len = strlen(de->d_name);
        if (len <= 6 || strcmp(de->d_name + len - 6, ".fps.d") != 0 || len - 6 >= sizeof(unit))
        {
            continue;
        }
        snprintf(unit, sizeof(unit), "%.*s", (int) (len - 6), de->d_name);
        int by_run =
            fps_unit_info_read(unit, FPS_UNIT_RUN, &run) == 0 && strcmp(run.session, name) == 0;
        if (by_run || (fps_unit_info_read(unit, FPS_UNIT_CONF, &conf) == 0 &&
                       strcmp(conf.session, name) == 0))
        {
            fprintf(fp, "# unit %s: %s\n", unit,
                    (by_run && milkdetach_id_alive(&run.id)) ? "running" : "stopped");
            n++;
        }
    }
    closedir(dir);
    return n;
}

int milk_srvlog_session_join(const char *name, const char *bin, char *sock, size_t socksz)
{
    char              b[512];
    milk_spawn_opts_t opts = { .owner_rfd = -1, .creator_pid = getpid(), .is_named = 1 };

    if (!session_name_ok(name, "milk-cli --session", 0))
    {
        return 2;
    }
    milk_srvlog_make_sock(name, sock, socksz);
    if (milk_ipc_ping_socket(sock))
    {
        return 0;
    }
    if (bin == NULL)
    {
        milk_ipc_resolve_cli_bin(NULL, b, sizeof(b));
        bin = b;
    }
    return (milk_ipc_spawn_daemon(bin, sock, &opts) == 0) ? 0 : 1;
}

/* save FILE: the session's server writes FILE, a milk script, and FILE.d/ (milk_server_session_save()) */
static int cmd_save(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    char  path[PATH_MAX];
    char  cwd[PATH_MAX];
    char *av[3];

    if (argc != 2 || argv[1][0] == '\0')
    {
        fprintf(stderr, "Usage: @session save <file>\n");
        return 2;
    }
    if (argv[1][0] != '/' && getcwd(cwd, sizeof(cwd)) == NULL)
    {
        perror("@session save: getcwd");
        return 1;
    }
    snprintf(path, sizeof(path), "%s%s%s", (argv[1][0] == '/') ? "" : cwd,
             (argv[1][0] == '/') ? "" : "/", argv[1]);
    av[0] = (char *) "session";
    av[1] = (char *) "save";
    av[2] = path;
    return milk_client_call(3, av, 0, hooks);
}

/* Remove a server's output files (next to its log) */
static void unlink_output(const char *sockpath)
{
    char out[512];
    char old[520];

    if (milk_srvlog_out_path(sockpath, out, sizeof(out)) == 0)
    {
        snprintf(old, sizeof(old), "%s.1", out);
        unlink(out);
        unlink(old);
    }
}

/* 1 if a server listens on @sockpath, answering or not (a stopped one still accepts) */
static int session_listens(const char *sockpath)
{
    int fd = milk_ipc_connect_once(sockpath);
    if (fd >= 0)
    {
        close(fd);
    }
    return fd >= 0;
}

/**
 * @brief Stop a single server by socket path: its shutdown command, then signals if it stays (U18).
 * Return: 0 once it is gone, 1 if it is a busy milk-cli (it ends at its prompt), 2 if it still runs
 */
static int stop_one_server(const char *sockpath)
{
    char               path[512] = "";
    pid_t              srv_pid   = 0;
    uint32_t           flags     = 0;
    MILKDETACH_ID      id        = { 0 };
    int                have_id   = 0;
    int                has_log   = 0;
    unsigned long long ticks     = 0;

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
                if (shm->magic == MILK_SRVLOG_MAGIC && shm->version == MILK_SRVLOG_VERSION)
                {
                    has_log = 1;
                    srv_pid = shm->srv_pid;
                    flags   = shm->flags;
                    have_id = (sscanf(shm->ident, "%llu %36s", &ticks, id.boot_id) == 2);
                }
                munmap(shm, sizeof(milk_srvlog_shm_t));
            }
        }
    }
    // the identity it recorded: a pid alone may be reused once it is gone. A milk-cli session that
    // serves in its own process is only asked.
    id.pid         = srv_pid;
    id.start_ticks = (uint64_t) ticks;
    have_id = have_id && srv_pid > 0 && !(flags & MILK_SRVFLAG_INPROC) && milkdetach_id_alive(&id);

    if (milk_ipc_shutdown_socket(sockpath))
    {
        struct timespec t0;
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        do // 2 s for it to go: a ping may itself take that long
        {
            struct timespec req = { 0, 10000000L };
            nanosleep(&req, NULL);
            clock_gettime(CLOCK_MONOTONIC, &now);
        } while ((have_id ? milkdetach_id_alive(&id) : milk_ipc_ping_socket(sockpath)) &&
                 (now.tv_sec - t0.tv_sec) * 1000000000L + (now.tv_nsec - t0.tv_nsec) < 2000000000L);
    }
    if (have_id && milkdetach_id_alive(&id))
    {
        pid_t pgid = (getpgid(srv_pid) == srv_pid) ? srv_pid : 0;
        int   step = milkdetach_stop(&id, pgid, 0, 2000, 2000);
        fprintf(stderr, "milk: server %s ignored its shutdown command: %s%s\n", sockpath,
                (step > 0) ? "stopped by " : "still running after ",
                (step > 0) ? milkdetach_stop_step_name(step) : "SIGKILL");
        if (step <= 0)
        {
            return 2;
        }
    }
    else if (!have_id && session_listens(sockpath))
    {
        int inproc = (flags & MILK_SRVFLAG_INPROC) != 0;
        fprintf(stderr, "milk: %s still runs: %s\n", sockpath,
                inproc    ? "milk-cli's session ends when its command returns"
                : has_log ? "it ignored its shutdown command"
                          : "its log is in another MILK_SHM_DIR");
        return inproc ? 1 : 2; // its log stays: milk-cli removes its own as it exits
    }
    if (path[0] != '\0')
    {
        unlink(path);
    }
    unlink_output(sockpath);
    return 0;
}

/* engine_sock() - socket of milk-cli's session when running in its bash coprocess, else NULL */
static const char *engine_sock(const milk_client_hooks_t *hooks)
{
    const char *s = milk_ipc_get_var(hooks, "_MILK_ENGINE_SOCK");
    return (s != NULL && s[0] != '\0') ? s : NULL;
}

/* home_sock() - the session a shell returns to: milk-cli's own one, else the shell's own one */
static const char *home_sock(const milk_client_hooks_t *hooks)
{
    const char *s = engine_sock(hooks);
    return (s != NULL) ? s : milk_ipc_get_var(hooks, "_MILK_LOCAL_SOCK");
}

/* The shell now uses session @sockpath (@name; "" for its home) */
static void session_attach(const milk_client_hooks_t *hooks, const char *sockpath, const char *name)
{
    milk_ipc_set_var(hooks, "MILKCLI_SOCK", sockpath, 1);
    milk_ipc_set_var(hooks, "_MILK_SERVER_NAME", name, 1);
}

/* The session @arg names (else the shell's current one): @sockpath, @name */
static void session_target(const milk_client_hooks_t *hooks,
                           const char                *arg,
                           char                      *sockpath,
                           size_t                     socksz,
                           char                      *name,
                           size_t                     namesz)
{
    unsigned int uid = 0;

    if (arg != NULL && arg[0] != '\0')
    {
        milk_srvlog_make_sock(arg, sockpath, socksz);
    }
    else
    {
        milk_ipc_resolve_sockpath(hooks, sockpath, socksz, 0);
    }
    milk_srvlog_parse_sock(sockpath, &uid, name, namesz);
}

/* status [NAME] */
static int cmd_status(const char *arg, const milk_client_hooks_t *hooks)
{
    char           cur[256];
    char           sock[256];
    char           name[64];
    char           out[512];
    session_info_t si;

    milk_ipc_resolve_sockpath(hooks, cur, sizeof(cur), 0);
    session_target(hooks, arg, sock, sizeof(sock), name, sizeof(name));
    int has    = session_read(sock, &si);
    int alive  = milk_ipc_ping_socket(sock);
    int own    = milk_ipc_is_numeric_str(name);
    int inproc = has && (si.flags & MILK_SRVFLAG_INPROC);

    int         color = use_color_output();
    const char *c_hdr = color ? C_TITLE : "";
    const char *c_ok  = color ? C_CMD : "";
    const char *c_dim = color ? C_DIM : "";
    const char *c_rst = color ? C_RST : "";

    printf("%s[milk session]%s\n", c_hdr, c_rst);
    printf("  Name      : %s%s\n", name, (strcmp(sock, cur) == 0) ? " (this shell's)" : "");
    printf("  Socket    : %s\n", sock);
    printf("  Kind      : %s\n",
           !inproc ? (own ? "a shell's own (starts on its first command)" : "named")
           : (si.flags & MILK_SRVFLAG_SCRIPT) ? "milk-script's (in its process)"
                                              : "milk-cli's (in its process)");
    if (alive)
    {
        char up[32] = "-";
        if (has)
        {
            format_uptime(si.ts_created.tv_sec, up, sizeof(up));
        }
        printf("  State     : %s%s%s (PID %d, uptime %s)\n", c_ok,
               (has && owner_alive(&si) == 0) ? "ORPHANED" : "RUNNING", c_rst,
               has ? (int) si.srv_pid : 0, up);
    }
    else if (own)
    {
        printf("  State     : %sIDLE%s (starts on its first command)\n", c_dim, c_rst);
    }
    else
    {
        printf("  State     : %sENDED%s (@session new %s)\n", c_dim, c_rst, name);
    }
    if (has && si.owner_pid > 0)
    {
        printf("  Owner     : %d%s\n", (int) si.owner_pid,
               (owner_alive(&si) == 0) ? " (exited: the session is orphaned)" : "");
    }
    if (has && si.idle_s > 0)
    {
        char t[32];
        format_secs((long) si.idle_s, t, sizeof(t));
        printf("  Idle end  : after %s without a command\n", t);
    }
    if (has && !inproc && milk_srvlog_out_path(sock, out, sizeof(out)) == 0 &&
        access(out, F_OK) == 0)
    {
        printf("  Output    : %s\n", out);
    }
    if (has && si.cwd[0] != '\0')
    {
        printf("  Workdir   : %s\n", si.cwd);
    }
    if (has)
    {
        printf("  Resources : %u modules, %u local images, %u vars\n", si.nb_modules, si.nb_images,
               si.nb_vars);
        printf("  Log       : %lu commands (%lu errors)\n", (unsigned long) si.total_cmds,
               (unsigned long) si.total_errors);
    }
    int shells =
        session_shells(sock, NULL, 0, NULL) + (own && !inproc && pid_alive((pid_t) atol(name)));
    printf("  Shells    : %d\n", shells);
    if (!own)
    {
        printf("  Units     : %d\n", session_units(name, 0));
    }
    return 0;
}

/* ls [--json] */
static int cmd_ls(int json, const milk_client_hooks_t *hooks)
{
    char cur[256];
    char dirpath[256];
    char pfx[64];
    int  count = 0;
    int  found = 0;

    milk_ipc_resolve_sockpath(hooks, cur, sizeof(cur), 0);
    milk_srvlog_get_dir(dirpath, sizeof(dirpath));
    size_t plen  = (size_t) snprintf(pfx, sizeof(pfx), "milk-srv.%u.", (unsigned) getuid());
    DIR   *dir   = opendir(dirpath);
    int    color = use_color_output();

    if (json)
    {
        putchar('[');
    }
    else
    {
        printf("%s%-3s %-14s %-8s %-8s %-7s %-7s %-8s %-6s %-5s %s%s\n", color ? C_HDR : "", "ACT",
               "NAME", "STATE", "OWNER", "CLIENTS", "UPTIME", "LAST-USE", "IMAGES", "UNITS", "CWD",
               color ? C_RST : "");
    }
    for (struct dirent *de; dir != NULL && (de = readdir(dir)) != NULL;)
    {
        char           full[512];
        session_info_t si;
        size_t         len = strlen(de->d_name);
        if (len <= plen + 8 || strncmp(de->d_name, pfx, plen) != 0 ||
            strcmp(de->d_name + len - 8, ".log.shm") != 0)
        {
            continue;
        }
        snprintf(full, sizeof(full), "%s/%s", dirpath, de->d_name);
        if (!session_read_log(full, &si))
        {
            continue;
        }
        int alive = pid_alive(si.srv_pid) && milk_ipc_ping_socket(si.sockpath);
        int cur_s = (strcmp(si.sockpath, cur) == 0);
        if (!alive && !pid_alive(si.owner_pid) && !cur_s)
        {
            unlink(full); // a server gone with its owner: what @session prune also does
            unlink_output(si.sockpath);
            continue;
        }
        found |= cur_s;

        int  own        = milk_ipc_is_numeric_str(si.name);
        int  inproc     = (si.flags & MILK_SRVFLAG_INPROC) != 0;
        int  clients    = session_shells(si.sockpath, NULL, 0, NULL) +
                          (own && !inproc && pid_alive(si.owner_pid));
        int  units      = own ? 0 : session_units(si.name, 0);
        char up[32]     = "-";
        char last[32]   = "-";
        char owner[16]  = "-";
        char nunits[16] = "-";
        if (alive)
        {
            format_uptime(si.ts_created.tv_sec, up, sizeof(up));
        }
        if (si.ts_last.tv_sec > 0)
        {
            format_uptime(si.ts_last.tv_sec, last, sizeof(last));
        }
        if (si.owner_pid > 0)
        {
            snprintf(owner, sizeof(owner), "%d", (int) si.owner_pid);
        }
        if (!own)
        {
            snprintf(nunits, sizeof(nunits), "%d", units);
        }
        int         orphan = alive && owner_alive(&si) == 0;
        const char *state  = orphan ? "ORPHANED" : alive ? "RUNNING" : own ? "IDLE" : "ENDED";
        if (json)
        {
            printf("%s{\"active\":%s,\"name\":\"%s\",\"state\":\"%s\",\"pid\":%d,\"kind\":\"%s\","
                   "\"owner\":%d,\"clients\":%d,\"images\":%u,\"vars\":%u,\"cmds\":%lu,"
                   "\"errors\":%lu,\"units\":%d,\"idle\":%u,\"cwd\":\"%s\",\"socket\":\"%s\"}",
                   (count > 0) ? "," : "", cur_s ? "true" : "false", si.name, state,
                   (int) si.srv_pid,
                   !inproc                            ? (own ? "own" : "named")
                   : (si.flags & MILK_SRVFLAG_SCRIPT) ? "milk-script"
                                                      : "milk-cli",
                   (int) si.owner_pid, clients, si.nb_images, si.nb_vars,
                   (unsigned long) si.total_cmds, (unsigned long) si.total_errors, units, si.idle_s,
                   si.cwd, si.sockpath);
        }
        else
        {
            printf("%-3s %-14s %-8s %-8s %-7d %-7s %-8s %-6u %-5s %s\n", cur_s ? "*" : "", si.name,
                   state, owner, clients, up, last, si.nb_images, nunits, si.cwd);
        }
        count++;
    }
    if (dir != NULL)
    {
        closedir(dir);
    }
    if (!found)
    {
        unsigned int uid = 0;
        char         name[64];
        milk_srvlog_parse_sock(cur, &uid, name, sizeof(name));
        int         alive = milk_ipc_ping_socket(cur);
        int         own   = milk_ipc_is_numeric_str(name);
        const char *state = alive ? "RUNNING" : own ? "IDLE" : "ENDED";
        if (json)
        {
            printf("%s{\"active\":true,\"name\":\"%s\",\"state\":\"%s\",\"pid\":0,\"kind\":\"%s\","
                   "\"owner\":0,\"clients\":0,\"images\":0,\"vars\":0,\"cmds\":0,\"errors\":0,"
                   "\"units\":0,\"idle\":0,\"cwd\":\"\",\"socket\":\"%s\"}",
                   (count > 0) ? "," : "", name, state, own ? "own" : "named", cur);
        }
        else
        {
            printf("%-3s %-14s %-8s %-8s %-7d %-7s %-8s %-6u %-5s %s\n", "*", name, state, "-", 0,
                   "-", "-", 0U, own ? "-" : "0", "");
        }
    }
    if (json)
    {
        puts("]");
    }
    return 0;
}

/* The verbs @server had, for the hint (start, list, local, detach, stop, attach) */
static const char *session_old_verb(const char *verb)
{
    static const char *const map[][2] = { { "start", "new <name> (or use <name>)" },
                                          { "list", "ls" },
                                          { "local", "leave" },
                                          { "detach", "leave" },
                                          { "stop", "end" },
                                          { "attach", "log -f" } };

    for (size_t i = 0; i < sizeof(map) / sizeof(map[0]); i++)
    {
        if (strcmp(verb, map[i][0]) == 0)
        {
            return map[i][1];
        }
    }
    return NULL;
}

/* 1 if the NUL-separated arguments @args (@n bytes) hold `--server @sock` */
static int args_serve(const char *args, ssize_t n, const char *sock)
{
    for (const char *a = args; a < args + n; a += strlen(a) + 1)
    {
        const char *next = a + strlen(a) + 1;
        if (strcmp(a, "--server") == 0 && next < args + n && strcmp(next, sock) == 0)
        {
            return 1;
        }
    }
    return 0;
}

/*
 * session_other_dir() - The MILK_SHM_DIR of the server on @sock when its log is not in this shell's,
 * so `@session ls` here does not list it: the directory of the log file the server maps. Else "".
 */
static void session_other_dir(const char *sock, char *dir, size_t dirsz)
{
    char here[512];
    dir[0] = '\0';
    if (milk_srvlog_shm_path(sock, here, sizeof(here)) != 0 || access(here, F_OK) == 0)
    {
        return;
    }
    const char *file = strrchr(here, '/');
    DIR        *pd   = opendir("/proc");
    for (struct dirent *de; pd != NULL && dir[0] == '\0' && (de = readdir(pd)) != NULL;)
    {
        char path[64];
        char args[2048];
        if (!isdigit((unsigned char) de->d_name[0]))
        {
            continue;
        }
        snprintf(path, sizeof(path), "/proc/%.20s/cmdline", de->d_name); /* a PID */
        int     fd = open(path, O_RDONLY | O_CLOEXEC);
        ssize_t n  = (fd >= 0) ? read(fd, args, sizeof(args) - 1) : -1;
        if (fd >= 0)
        {
            close(fd);
        }
        if (n <= 0)
        {
            continue;
        }
        args[n] = '\0';
        if (!args_serve(args, n, sock))
        {
            continue;
        }
        snprintf(path, sizeof(path), "/proc/%.20s/maps", de->d_name);
        FILE *fp = fopen(path, "re");
        char  line[1024];
        while (fp != NULL && fgets(line, sizeof(line), fp) != NULL)
        {
            char *f = strstr(line, file);
            char *p = strchr(line, '/');
            if (f != NULL && p != NULL && p < f)
            {
                snprintf(dir, dirsz, "%.*s", (int) (f - p), p);
                break;
            }
        }
        if (fp != NULL)
        {
            fclose(fp);
        }
    }
    if (pd != NULL)
    {
        closedir(pd);
    }
}

/* new NAME [-d] [--idle T] */
static int cmd_new(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    const char       *name   = NULL;
    int               detach = 0;
    char              sock[256];
    char              bin[512];
    milk_spawn_opts_t opts = { .owner_rfd   = -1,
                               .creator_pid = milk_ipc_shell_pid(hooks),
                               .is_named    = 1 };

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-d") == 0)
        {
            detach = 1;
        }
        else if (strcmp(argv[i], "--idle") == 0)
        {
            if (i + 1 >= argc || !parse_duration(argv[++i], &opts.idle_s))
            {
                fprintf(stderr, "@session new: --idle takes a duration: 90, 30s, 10m, 2h or 1d\n");
                return 2;
            }
        }
        else if (strcmp(argv[i], "--tmux") == 0 || strcmp(argv[i], "-tmux") == 0)
        {
            fprintf(stderr,
                    "@session new: --tmux was removed: sessions start detached, their output "
                    "in the file @session status names\n");
            return 2;
        }
        else if (argv[i][0] == '-' || name != NULL)
        {
            fprintf(stderr, "Usage: @session new <name> [-d] [--idle <T>]\n");
            return 2;
        }
        else
        {
            name = argv[i];
        }
    }
    if (!session_name_ok(name, "@session new", 0))
    {
        return 2;
    }
    milk_srvlog_make_sock(name, sock, sizeof(sock));
    if (milk_ipc_ping_socket(sock))
    {
        char other[512];
        session_other_dir(sock, other, sizeof(other));
        fprintf(stderr, "@session new: session %s exists (@session use %s)%s%s%s\n", name, name,
                (other[0] != '\0') ? "; it runs with MILK_SHM_DIR=" : "", other,
                (other[0] != '\0') ? ", where @session ls lists it" : "");
        return 1;
    }
    milk_ipc_resolve_cli_bin(hooks, bin, sizeof(bin));
    if (milk_ipc_spawn_daemon(bin, sock, &opts) != 0)
    {
        return 1;
    }
    if (detach)
    {
        printf("Session %s started.\n", name);
        return 0;
    }
    session_attach(hooks, sock, name);
    printf("Session %s started; this shell uses it.\n", name);
    return 0;
}

/* use NAME */
static int cmd_use(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    char         sock[256];
    char         name[64];
    unsigned int uid = 0;

    if (argc != 2)
    {
        fprintf(stderr, "Usage: @session use <name>\n");
        return 2;
    }
    if (!session_name_ok(argv[1], "@session use", 1))
    {
        return 2;
    }
    milk_srvlog_make_sock(argv[1], sock, sizeof(sock));
    if (!milk_ipc_ping_socket(sock))
    {
        fprintf(stderr, "@session use: no session %s runs (@session new %s)\n", argv[1], argv[1]);
        return 1;
    }
    milk_srvlog_parse_sock(sock, &uid, name, sizeof(name));
    const char *home = home_sock(hooks);
    session_attach(hooks, sock, (home != NULL && strcmp(home, sock) == 0) ? "" : name);
    printf("This shell uses session %s.\n", name);
    return 0;
}

/* leave */
static int cmd_leave(const milk_client_hooks_t *hooks)
{
    const char *home = home_sock(hooks);
    char        sock[256];

    if (home != NULL && home[0] != '\0')
    {
        snprintf(sock, sizeof(sock), "%s", home);
    }
    else
    {
        pid_t ref = (hooks != NULL && hooks->get_var != NULL) ? getpid() : getppid();
        snprintf(sock, sizeof(sock), "@milk-cli-%u-%d", (unsigned) getuid(), (int) ref);
    }
    session_attach(hooks, sock, "");
    printf("This shell uses %s session again (%s).\n",
           (engine_sock(hooks) != NULL) ? "milk-cli's" : "its own", sock);
    return 0;
}

/* One line of @session who: the shell's PID, terminal and working directory */
static void print_shell(pid_t pid, const char *note)
{
    char    link[64];
    char    tty[256]      = "-";
    char    cwd[PATH_MAX] = "-";
    ssize_t n;

    snprintf(link, sizeof(link), "/proc/%d/fd/0", (int) pid);
    if ((n = readlink(link, tty, sizeof(tty) - 1)) > 0)
    {
        tty[n] = '\0';
    }
    snprintf(link, sizeof(link), "/proc/%d/cwd", (int) pid);
    if ((n = readlink(link, cwd, sizeof(cwd) - 1)) > 0)
    {
        cwd[n] = '\0';
    }
    printf("%-8d %-12s %s%s\n", (int) pid, (strncmp(tty, "/dev/", 5) == 0) ? tty + 5 : "-", cwd,
           note);
}

/* who [NAME] */
static int cmd_who(const char *arg, const milk_client_hooks_t *hooks)
{
    char           sock[256];
    char           name[64];
    pid_t          pids[256];
    session_info_t si;

    session_target(hooks, arg, sock, sizeof(sock), name, sizeof(name));
    int has    = session_read(sock, &si);
    int inproc = has && (si.flags & MILK_SRVFLAG_INPROC);
    int n      = session_shells(sock, pids, (int) (sizeof(pids) / sizeof(pids[0])), NULL);

    printf("%-8s %-12s %s\n", "PID", "TTY", "CWD");
    if (inproc && pid_alive(si.srv_pid))
    {
        print_shell(si.srv_pid,
                    (si.flags & MILK_SRVFLAG_SCRIPT) ? "  (milk-script)" : "  (milk-cli)");
    }
    else if (milk_ipc_is_numeric_str(name) && pid_alive((pid_t) atol(name)))
    {
        print_shell((pid_t) atol(name), "  (its own)");
    }
    for (int i = 0; i < n && i < (int) (sizeof(pids) / sizeof(pids[0])); i++)
    {
        print_shell(pids[i], "");
    }
    return 0;
}

/* end [NAME|--all] [--units] */
static int cmd_end(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    const char *arg   = NULL;
    const char *esock = engine_sock(hooks);
    int         all   = 0;
    int         units = 0;
    char        cur[256];
    char        sock[256];
    char        name[64];

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--all") == 0)
        {
            all = 1;
        }
        else if (strcmp(argv[i], "--units") == 0)
        {
            units = 1;
        }
        else if (argv[i][0] == '-' || arg != NULL)
        {
            fprintf(stderr, "Usage: @session end [<name>|--all] [--units]\n");
            return 2;
        }
        else
        {
            arg = argv[i];
        }
    }
    milk_ipc_resolve_sockpath(hooks, cur, sizeof(cur), 0);
    if (all)
    {
        char dirpath[256];
        char pfx[64];
        milk_srvlog_get_dir(dirpath, sizeof(dirpath));
        size_t plen = (size_t) snprintf(pfx, sizeof(pfx), "milk-srv.%u.", (unsigned) getuid());
        DIR   *dir  = opendir(dirpath);
        for (struct dirent *de; dir != NULL && (de = readdir(dir)) != NULL;)
        {
            char           full[512];
            session_info_t si;
            size_t         len = strlen(de->d_name);
            snprintf(full, sizeof(full), "%s/%s", dirpath, de->d_name);
            if (len <= plen + 8 || strncmp(de->d_name, pfx, plen) != 0 ||
                strcmp(de->d_name + len - 8, ".log.shm") != 0 || !session_read_log(full, &si) ||
                (esock != NULL && strcmp(si.sockpath, esock) == 0))
            {
                continue;
            }
            if (units && !milk_ipc_is_numeric_str(si.name))
            {
                session_units(si.name, 1);
            }
            stop_one_server(si.sockpath);
        }
        if (dir != NULL)
        {
            closedir(dir);
        }
        cmd_leave(hooks);
        return 0;
    }

    session_target(hooks, arg, sock, sizeof(sock), name, sizeof(name));
    if (esock != NULL && strcmp(sock, esock) == 0)
    {
        fprintf(stderr, "@session end: milk-cli's session ends with milk-cli (exit)\n");
        return 1;
    }
    int stopped = 0; // the units go first, also those of a session that has ended already
    if (units && milk_ipc_is_numeric_str(name))
    {
        fprintf(stderr, "@session end --units: units record named sessions only\n");
    }
    else if (units)
    {
        stopped = session_units(name, 1);
    }
    session_info_t si;
    if (!session_listens(sock) && !session_read(sock, &si))
    {
        if (stopped > 0)
        {
            printf("Session %s had ended; units stopped: %d.\n", name, stopped);
            return 0;
        }
        fprintf(stderr, "@session end: no session %s\n", name);
        return 1;
    }
    int left = stop_one_server(sock);
    if (strcmp(sock, cur) == 0 && home_sock(hooks) != NULL && strcmp(sock, home_sock(hooks)) != 0)
    {
        session_attach(hooks, home_sock(hooks), "");
    }
    if (left == 0)
    {
        printf("Session %s ended.\n", name);
    }
    return (left == 2) ? 1 : 0;
}

/* prune */
static int cmd_prune(const milk_client_hooks_t *hooks)
{
    char cur[256];
    char dirpath[256];
    char pfx[64];
    int  logs      = 0;
    int  servers   = 0;
    int  overrides = 0;

    milk_ipc_resolve_sockpath(hooks, cur, sizeof(cur), 0);
    milk_srvlog_get_dir(dirpath, sizeof(dirpath));
    size_t plen = (size_t) snprintf(pfx, sizeof(pfx), "milk-srv.%u.", (unsigned) getuid());
    DIR   *dir  = opendir(dirpath);
    for (struct dirent *de; dir != NULL && (de = readdir(dir)) != NULL;)
    {
        char           full[512];
        session_info_t si;
        size_t         len = strlen(de->d_name);
        snprintf(full, sizeof(full), "%s/%s", dirpath, de->d_name);
        if (len <= plen + 8 || strncmp(de->d_name, pfx, plen) != 0 ||
            strcmp(de->d_name + len - 8, ".log.shm") != 0 || !session_read_log(full, &si))
        {
            continue;
        }
        int alive = pid_alive(si.srv_pid) && milk_ipc_ping_socket(si.sockpath);
        int own   = milk_ipc_is_numeric_str(si.name) && !(si.flags & MILK_SRVFLAG_INPROC);
        if (alive && own && owner_alive(&si) == 0)
        {
            stop_one_server(si.sockpath); // a shell's own session that outlived its shell
            servers++;
        }
        else if (!alive && strcmp(si.sockpath, cur) != 0) // also while its owner runs, unlike ls
        {
            unlink(full);
            unlink_output(si.sockpath);
            logs++;
        }
    }
    if (dir != NULL)
    {
        closedir(dir);
    }
    session_shells(NULL, NULL, 0, &overrides);
    printf("Pruned ended-session logs: %d, orphaned servers: %d, exited-shell records: %d.\n", logs,
           servers, overrides);
    return 0;
}

/* print_session_help - usage of @session */
static void print_session_help(void)
{
    printf("Usage: @session [<verb> [args...]]\n\n");
    printf("A session is a milk-cli server: it holds images, calculator variables and settings.\n");
    printf(
        "Each shell has its own, started on its first command; a named session is shared by the\n");
    printf(
        "shells that use it, and lives until it is ended. Scripts started from a shell use its\n");
    printf("session. A session belongs to the MILK_SHM_DIR it started with.\n\n");
    printf("Verbs:\n");
    printf("  status [name]                This shell's session, or session <name> (default)\n");
    printf("  ls [--json]                  This user's sessions in this MILK_SHM_DIR\n");
    printf("  new <name> [-d] [--idle T]   Start session <name> and use it (-d: do not); fails if "
           "it runs.\n");
    printf("                               --idle: it ends after T (90, 30s, 10m, 2h, 1d) without "
           "a command\n");
    printf("                               while no shell uses it\n");
    printf("  use <name>                   Use session <name>; fails if it does not run\n");
    printf("  leave                        Use this shell's own session again\n");
    printf("  who [name]                   The shells that use a session\n");
    printf("  end [name|--all] [--units]   End a session, the one in use by default (--all: every "
           "one ls lists;\n");
    printf("                               --units: and the units it started)\n");
    printf("  prune                        Remove what ended sessions and exited shells left\n");
    printf("  save <file>                  Write the session's state as a milk script, images in "
           "<file>.d/\n");
    printf("  restore <file>               Run that script in this shell's session (units: listed, "
           "not started)\n");
    printf("  log [options] [pattern]      The session's command log (@log)\n");
}

/**
 * @brief Execute `@session` (status, ls, new, use, leave, who, end, prune, log).
 */
int milk_srvlog_cmd_session(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    const char *verb = (argc > 0 && argv[0] != NULL) ? argv[0] : "status";
    char        cur[256];

    if (strcmp(verb, "-h") == 0 || strcmp(verb, "--help") == 0 || strcmp(verb, "help") == 0)
    {
        print_session_help();
        return 0;
    }
    if (strcmp(verb, "status") == 0)
    {
        return cmd_status((argc >= 2) ? argv[1] : NULL, hooks);
    }
    if (strcmp(verb, "ls") == 0)
    {
        return cmd_ls(argc >= 2 && strcmp(argv[1], "--json") == 0, hooks);
    }
    if (strcmp(verb, "new") == 0)
    {
        return cmd_new(argc, argv, hooks);
    }
    if (strcmp(verb, "use") == 0)
    {
        return cmd_use(argc, argv, hooks);
    }
    if (strcmp(verb, "leave") == 0)
    {
        return cmd_leave(hooks);
    }
    if (strcmp(verb, "who") == 0)
    {
        return cmd_who((argc >= 2) ? argv[1] : NULL, hooks);
    }
    if (strcmp(verb, "end") == 0)
    {
        return cmd_end(argc, argv, hooks);
    }
    if (strcmp(verb, "prune") == 0)
    {
        return cmd_prune(hooks);
    }
    if (strcmp(verb, "save") == 0)
    {
        return cmd_save(argc, argv, hooks);
    }
    if (strcmp(verb, "restore") == 0)
    {
        fprintf(stderr,
                "@session restore: from a shell, which runs the file (source <file> does too)\n");
        return 2;
    }
    if (strcmp(verb, "log") == 0)
    {
        milk_ipc_resolve_sockpath(hooks, cur, sizeof(cur), 0);
        return milk_srvlog_cmd_log(cur, argc - 1, argv + 1, hooks);
    }
    const char *now = session_old_verb(verb);
    if (now != NULL)
    {
        fprintf(stderr, "@session: %s is @session %s\n", verb, now);
        return 2;
    }
    fprintf(stderr, "@session: unknown verb '%s'\n", verb);
    print_session_help();
    return 2;
}
