// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_srvlog_shm.c
 * @brief Shared-memory server ring-buffer log (`milk-srv.<uid>.<name>.log.shm`) and `@log` CLI.
 *
 * Role & Goal:
 *   Implements the lock-free `/dev/shm` circular log (`milk_srvlog_shm_t`,
 *   1024 entries) recording nanosecond timestamps, execution durations, exit
 *   codes, input commands, and captured stdout/stderr snippets for each
 *   `milk-cli` server. Also implements the `@log` / `@server log` search,
 *   regex filter, `--slow`, `--since`, `--errors`, `--json`, and `-f` live-tail
 *   commands.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <ctype.h>
#include <fcntl.h>
#include <regex.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "CLIcore.h"
#include "milk_ipc_common.h"
#include "milk_srvlog.h"

#define C_RST "\033[0m"
#define C_HDR "\033[1;35m"
#define C_CMD "\033[32m"
#define C_BOLD "\033[1m"
#define C_DIM "\033[2m"
#define C_ERR "\033[1;31m"

static volatile sig_atomic_t g_follow_sigint = 0;

static void follow_sigint_handler(int sig)
{
    (void) sig;
    g_follow_sigint = 1;
}

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
 * @brief Resolve the writable shared-memory directory for server `.log.shm` files.
 *
 * @param out   Output directory path buffer
 * @param outsz Size of @out in bytes
 */
void milk_srvlog_get_dir(char *out, size_t outsz)
{
    const char *env = getenv("MILK_SHM_DIR");
    if (env != NULL && env[0] != '\0' && access(env, W_OK) == 0)
    {
        snprintf(out, outsz, "%s", env);
        return;
    }
    if (dcshmdir[0] != '\0' && access(dcshmdir, W_OK) == 0)
    {
        snprintf(out, outsz, "%s", dcshmdir);
        return;
    }
    if (access("/dev/shm", W_OK) == 0)
    {
        snprintf(out, outsz, "/dev/shm");
        return;
    }
    const char *tmp = getenv("TMPDIR");
    snprintf(out, outsz, "%s", (tmp != NULL && tmp[0] != '\0') ? tmp : "/tmp");
}

/**
 * @brief Parse a `MILKCLI_SOCK` string into its numeric UID and logical server name.
 */
void milk_srvlog_parse_sock(const char   *sockpath,
                            unsigned int *out_uid,
                            char         *out_name,
                            size_t        namesz)
{
    unsigned int uid = (unsigned int) getuid();
    if (out_name != NULL && namesz > 0)
    {
        out_name[0] = '\0';
    }

    if (sockpath != NULL && strncmp(sockpath, "@milk-cli-", 10) == 0)
    {
        const char   *p    = sockpath + 10;
        char         *endp = NULL;
        unsigned long u    = strtoul(p, &endp, 10);
        if (endp != p && *endp == '-' && endp[1] != '\0')
        {
            uid = (unsigned int) u;
            if (out_name != NULL && namesz > 0)
            {
                snprintf(out_name, namesz, "%s", endp + 1);
            }
            if (out_uid != NULL)
            {
                *out_uid = uid;
            }
            return;
        }
    }

    if (out_uid != NULL)
    {
        *out_uid = uid;
    }
    if (out_name != NULL && namesz > 0)
    {
        const char *s = (sockpath != NULL && sockpath[0] != '\0') ? sockpath : "default";
        if (*s == '@' || *s == '/')
        {
            s++;
        }
        size_t j = 0;
        for (size_t i = 0; s[i] != '\0' && j + 1 < namesz; i++)
        {
            char c        = s[i];
            out_name[j++] = (isalnum((unsigned char) c) || c == '_' || c == '-') ? c : '_';
        }
        out_name[j] = '\0';
    }
}

/**
 * @brief Build a canonical `MILKCLI_SOCK` string from a logical server name or socket.
 */
void milk_srvlog_make_sock(const char *name_or_sock, char *out_sock, size_t socksz)
{
    if (name_or_sock == NULL || name_or_sock[0] == '\0')
    {
        snprintf(out_sock, socksz, "@milk-cli-%u-%d", (unsigned) getuid(), (int) getpid());
        return;
    }
    if (name_or_sock[0] == '@' || name_or_sock[0] == '/')
    {
        snprintf(out_sock, socksz, "%s", name_or_sock);
        return;
    }
    snprintf(out_sock, socksz, "@milk-cli-%u-%s", (unsigned) getuid(), name_or_sock);
}

/**
 * @brief Resolve the filesystem path to a server's `.log.shm` file.
 */
int milk_srvlog_shm_path(const char *sockpath, char *out_path, size_t pathsz)
{
    if (out_path == NULL || pathsz == 0)
    {
        return -1;
    }
    char         shmdir[256];
    unsigned int uid = 0;
    char         name[64];
    milk_srvlog_get_dir(shmdir, sizeof(shmdir));
    milk_srvlog_parse_sock(sockpath, &uid, name, sizeof(name));

    char   safe[64];
    size_t j = 0;
    for (size_t i = 0; name[i] != '\0' && j + 1 < sizeof(safe); i++)
    {
        char c    = name[i];
        safe[j++] = (isalnum((unsigned char) c) || c == '_' || c == '-') ? c : '_';
    }
    safe[j] = '\0';

    int n = snprintf(out_path, pathsz, "%s/milk-srv.%u.%s.log.shm", shmdir, uid, safe);
    return (n > 0 && (size_t) n < pathsz) ? 0 : -1;
}

/**
 * @brief Create or attach to a server's shared-memory log and initialize server metadata.
 */
milk_srvlog_shm_t *milk_srvlog_server_init(const char *sockpath, pid_t owner_pid, uint32_t flags)
{
    char path[512];
    if (milk_srvlog_shm_path(sockpath, path, sizeof(path)) != 0)
    {
        return NULL;
    }

    int fd = open(path, O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0)
    {
        return NULL;
    }
    if (ftruncate(fd, (off_t) sizeof(milk_srvlog_shm_t)) != 0)
    {
        close(fd);
        return NULL;
    }

    milk_srvlog_shm_t *shm = (milk_srvlog_shm_t *) mmap(NULL, sizeof(milk_srvlog_shm_t),
                                                        PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (shm == MAP_FAILED)
    {
        return NULL;
    }

    unsigned int uid = 0;
    char         sname[64];
    milk_srvlog_parse_sock(sockpath, &uid, sname, sizeof(sname));

    int preserve_entries = 0;
    if (shm->magic == MILK_SRVLOG_MAGIC && shm->version == MILK_SRVLOG_VERSION &&
        shm->srv_pid == 0 && strcmp(shm->sockpath, sockpath ? sockpath : "") == 0)
    {
        preserve_entries = 1;
    }

    if (!preserve_entries)
    {
        memset(shm, 0, sizeof(*shm));
        shm->magic   = MILK_SRVLOG_MAGIC;
        shm->version = MILK_SRVLOG_VERSION;
        clock_gettime(CLOCK_REALTIME, &shm->ts_created);
    }
    else if (shm->ts_created.tv_sec == 0)
    {
        clock_gettime(CLOCK_REALTIME, &shm->ts_created);
    }

    shm->srv_pid = getpid();
    shm->owner_pid =
        (owner_pid > 0) ? owner_pid : (milk_ipc_is_numeric_str(sname) ? (pid_t) atoi(sname) : 0);
    shm->flags |= flags;

    if (!milk_ipc_is_numeric_str(sname) || getenv("MILK_SRV_NAMED") != NULL)
    {
        shm->flags |= MILK_SRVFLAG_NAMED;
    }
    const char *tmux_env = getenv("MILK_SRV_TMUX");
    if (tmux_env != NULL && tmux_env[0] != '\0')
    {
        shm->flags |= MILK_SRVFLAG_TMUX;
        snprintf(shm->tmux_session, sizeof(shm->tmux_session), "%s", tmux_env);
    }

    snprintf(shm->srv_name, sizeof(shm->srv_name), "%s", sname);
    snprintf(shm->sockpath, sizeof(shm->sockpath), "%s", sockpath ? sockpath : "");
    if (getcwd(shm->cwd, sizeof(shm->cwd)) == NULL)
    {
        shm->cwd[0] = '\0';
    }
    return shm;
}

/**
 * @brief Update live server entity counts and working directory in shared memory.
 */
void milk_srvlog_server_update_stats(milk_srvlog_shm_t *shm,
                                     uint32_t           nb_modules,
                                     uint32_t           nb_images,
                                     uint32_t           nb_vars,
                                     const char        *cwd)
{
    if (shm == NULL || shm->magic != MILK_SRVLOG_MAGIC)
    {
        return;
    }
    shm->nb_modules = nb_modules;
    shm->nb_images  = nb_images;
    shm->nb_vars    = nb_vars;
    if (cwd != NULL && cwd[0] != '\0')
    {
        snprintf(shm->cwd, sizeof(shm->cwd), "%s", cwd);
    }
}

/**
 * @brief Append a completed command and its output snippet to the ring buffer.
 */
void milk_srvlog_record(milk_srvlog_shm_t     *shm,
                        const struct timespec *ts_start,
                        uint64_t               duration_ns,
                        pid_t                  client_pid,
                        uint8_t                msg_type,
                        int32_t                rc,
                        const char            *cmd,
                        const char            *output)
{
    if (shm == NULL || shm->magic != MILK_SRVLOG_MAGIC)
    {
        return;
    }

    uint64_t             seq = __atomic_add_fetch(&shm->write_seq, 1, __ATOMIC_SEQ_CST);
    milk_srvlog_entry_t *e   = &shm->entries[(seq - 1) % MILK_SRVLOG_CAP];

    __atomic_store_n(&e->seq, 0, __ATOMIC_RELAXED);
    if (ts_start != NULL)
    {
        e->ts_start = *ts_start;
    }
    else
    {
        clock_gettime(CLOCK_REALTIME, &e->ts_start);
    }
    e->duration_ns = duration_ns;
    e->client_pid  = client_pid;
    e->msg_type    = msg_type;
    e->rc          = rc;

    if (cmd != NULL)
    {
        snprintf(e->cmd, sizeof(e->cmd), "%s", cmd);
    }
    else
    {
        e->cmd[0] = '\0';
    }

    if (output != NULL && output[0] != '\0')
    {
        snprintf(e->output, sizeof(e->output), "%s", output);
        size_t olen = strlen(e->output);
        while (olen > 0 && (e->output[olen - 1] == '\n' || e->output[olen - 1] == '\r'))
        {
            e->output[--olen] = '\0';
        }
    }
    else
    {
        e->output[0] = '\0';
    }

    __atomic_store_n(&e->seq, seq, __ATOMIC_RELEASE);
    __atomic_add_fetch(&shm->total_cmds, 1, __ATOMIC_RELAXED);
    if (rc != 0)
    {
        __atomic_add_fetch(&shm->total_errors, 1, __ATOMIC_RELAXED);
    }
}

/**
 * @brief Append a client-side fast-path operation (`@get`/`@set`) to the session's SHM log.
 */
void milk_srvlog_client_record(const char            *sockpath,
                               const struct timespec *ts_start,
                               uint64_t               duration_ns,
                               uint8_t                msg_type,
                               int32_t                rc,
                               const char            *cmd,
                               const char            *output)
{
    if (sockpath == NULL || sockpath[0] == '\0')
    {
        return;
    }
    char path[512];
    if (milk_srvlog_shm_path(sockpath, path, sizeof(path)) != 0)
    {
        return;
    }

    int fd = open(path, O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0)
    {
        return;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || (size_t) st.st_size < sizeof(milk_srvlog_shm_t))
    {
        if (ftruncate(fd, (off_t) sizeof(milk_srvlog_shm_t)) != 0)
        {
            close(fd);
            return;
        }
    }

    milk_srvlog_shm_t *shm = (milk_srvlog_shm_t *) mmap(NULL, sizeof(milk_srvlog_shm_t),
                                                        PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (shm == MAP_FAILED)
    {
        return;
    }

    if (shm->magic != MILK_SRVLOG_MAGIC || shm->version != MILK_SRVLOG_VERSION)
    {
        unsigned int uid = 0;
        char         sname[64];
        milk_srvlog_parse_sock(sockpath, &uid, sname, sizeof(sname));
        memset(shm, 0, sizeof(*shm));
        shm->magic   = MILK_SRVLOG_MAGIC;
        shm->version = MILK_SRVLOG_VERSION;
        clock_gettime(CLOCK_REALTIME, &shm->ts_created);
        snprintf(shm->srv_name, sizeof(shm->srv_name), "%s", sname);
        snprintf(shm->sockpath, sizeof(shm->sockpath), "%s", sockpath);
        if (milk_ipc_is_numeric_str(sname))
        {
            shm->owner_pid = (pid_t) atoi(sname);
        }
        else
        {
            shm->flags |= MILK_SRVFLAG_NAMED;
        }
        if (getcwd(shm->cwd, sizeof(shm->cwd)) == NULL)
        {
            shm->cwd[0] = '\0';
        }
    }

    milk_srvlog_record(shm, ts_start, duration_ns, getpid(), msg_type, rc, cmd, output);
    munmap(shm, sizeof(milk_srvlog_shm_t));
}

/**
 * @brief Unmap the server's shared-memory log and optionally unlink the backing file.
 */
void milk_srvlog_server_close(milk_srvlog_shm_t *shm, int unlink_file)
{
    if (shm == NULL)
    {
        return;
    }
    char sock_copy[128];
    snprintf(sock_copy, sizeof(sock_copy), "%s", shm->sockpath);
    shm->srv_pid = 0;
    munmap(shm, sizeof(milk_srvlog_shm_t));

    if (unlink_file && sock_copy[0] != '\0')
    {
        char path[512];
        if (milk_srvlog_shm_path(sock_copy, path, sizeof(path)) == 0)
        {
            unlink(path);
        }
    }
}

static void print_json_escaped(const char *s)
{
    putchar('"');
    if (s != NULL)
    {
        for (const unsigned char *p = (const unsigned char *) s; *p != '\0'; p++)
        {
            if (*p == '"')
            {
                fputs("\\\"", stdout);
            }
            else if (*p == '\\')
            {
                fputs("\\\\", stdout);
            }
            else if (*p == '\n')
            {
                fputs("\\n", stdout);
            }
            else if (*p == '\r')
            {
                fputs("\\r", stdout);
            }
            else if (*p == '\t')
            {
                fputs("\\t", stdout);
            }
            else if (*p < 0x20)
            {
                printf("\\u%04x", (unsigned) *p);
            }
            else
            {
                putchar((int) *p);
            }
        }
    }
    putchar('"');
}

static time_t parse_since_spec(const char *spec)
{
    if (spec == NULL || spec[0] == '\0')
    {
        return 0;
    }
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);

    char  *endp = NULL;
    double val  = strtod(spec, &endp);
    if (endp != spec && endp != NULL && *endp != '\0' && endp[1] == '\0')
    {
        char u = (char) tolower((unsigned char) *endp);
        if (u == 's')
        {
            return now.tv_sec - (time_t) val;
        }
        if (u == 'm')
        {
            return now.tv_sec - (time_t) (val * 60.0);
        }
        if (u == 'h')
        {
            return now.tv_sec - (time_t) (val * 3600.0);
        }
        if (u == 'd')
        {
            return now.tv_sec - (time_t) (val * 86400.0);
        }
    }

    int hh = 0, mm = 0, ss = 0;
    if (sscanf(spec, "%d:%d:%d", &hh, &mm, &ss) >= 2 && strchr(spec, '-') == NULL)
    {
        struct tm tm_now;
        localtime_r(&now.tv_sec, &tm_now);
        tm_now.tm_hour = hh;
        tm_now.tm_min  = mm;
        tm_now.tm_sec  = ss;
        return mktime(&tm_now);
    }

    int yr = 0, mo = 0, dy = 0;
    if (sscanf(spec, "%d-%d-%d%*1[T ]%d:%d:%d", &yr, &mo, &dy, &hh, &mm, &ss) >= 5)
    {
        struct tm tmv;
        memset(&tmv, 0, sizeof(tmv));
        tmv.tm_year  = yr - 1900;
        tmv.tm_mon   = mo - 1;
        tmv.tm_mday  = dy;
        tmv.tm_hour  = hh;
        tmv.tm_min   = mm;
        tmv.tm_sec   = ss;
        tmv.tm_isdst = -1;
        return mktime(&tmv);
    }

    if (endp != spec && endp != NULL && *endp == '\0')
    {
        return (time_t) val;
    }
    return 0;
}

static int entry_matches_filters(const milk_srvlog_entry_t *e,
                                 int                        errors_only,
                                 uint64_t                   min_dur_ns,
                                 time_t                     since_sec,
                                 int                        has_regex,
                                 const regex_t             *re)
{
    if (e->seq == 0)
    {
        return 0;
    }
    if (errors_only && e->rc == 0)
    {
        return 0;
    }
    if (min_dur_ns > 0 && e->duration_ns < min_dur_ns)
    {
        return 0;
    }
    if (since_sec > 0 && e->ts_start.tv_sec < since_sec)
    {
        return 0;
    }
    if (has_regex && re != NULL)
    {
        if (regexec(re, e->cmd, 0, NULL, 0) != 0 && regexec(re, e->output, 0, NULL, 0) != 0)
        {
            return 0;
        }
    }
    return 1;
}

static void format_entry_line(const milk_srvlog_entry_t *e, int json_mode, int color)
{
    struct tm tmv;
    localtime_r(&e->ts_start.tv_sec, &tmv);
    long ms = e->ts_start.tv_nsec / 1000000L;

    if (json_mode)
    {
        char tbuf[64];
        snprintf(tbuf, sizeof(tbuf), "%04d-%02d-%02dT%02d:%02d:%02d.%03ld", tmv.tm_year + 1900,
                 tmv.tm_mon + 1, tmv.tm_mday, tmv.tm_hour, tmv.tm_min, tmv.tm_sec, ms);
        printf("{\"seq\":%lu,\"ts\":\"%s\",\"duration_ms\":%.3f,\"pid\":%d,\"rc\":%d,\"cmd\":",
               (unsigned long) e->seq, tbuf, (double) e->duration_ns / 1e6, (int) e->client_pid,
               (int) e->rc);
        print_json_escaped(e->cmd);
        fputs(",\"output\":", stdout);
        print_json_escaped(e->output);
        fputs("}\n", stdout);
        return;
    }

    char tbuf[32];
    snprintf(tbuf, sizeof(tbuf), "%02d:%02d:%02d.%03ld", tmv.tm_hour, tmv.tm_min, tmv.tm_sec, ms);

    double dur_ms = (double) e->duration_ns / 1e6;
    char   dbuf[32];
    if (dur_ms < 1000.0)
    {
        snprintf(dbuf, sizeof(dbuf), "%6.2f ms", dur_ms);
    }
    else
    {
        snprintf(dbuf, sizeof(dbuf), "%6.2f s ", dur_ms / 1000.0);
    }

    const char *rc_col = (e->rc == 0) ? (color ? C_CMD : "") : (color ? C_ERR : "");
    const char *rst    = color ? C_RST : "";
    const char *dim    = color ? C_DIM : "";
    const char *bold   = color ? C_BOLD : "";

    printf("%s%s%s  %s  %s%3d%s  %6d  %s%s%s\n", dim, tbuf, rst, dbuf, rc_col, (int) e->rc, rst,
           (int) e->client_pid, bold, e->cmd, rst);

    if (e->output[0] != '\0')
    {
        const char *p = e->output;
        while (*p != '\0')
        {
            const char *nl   = strchr(p, '\n');
            size_t      llen = (nl != NULL) ? (size_t) (nl - p) : strlen(p);
            printf("                                  %s-> %.*s%s\n",
                   (e->rc != 0 && color) ? C_ERR : dim, (int) llen, p, rst);
            if (nl == NULL)
            {
                break;
            }
            p = nl + 1;
        }
    }
}

static void print_log_help(void)
{
    printf("Usage: @log [options] [pattern]\n");
    printf("   or: @server log [options] [pattern]\n\n");
    printf("Inspect, search, or live-tail the server's shared-memory command & output log.\n\n");
    printf("Options:\n");
    printf("  -n, --last <N>        Show last N matching entries (default: 20)\n");
    printf("  -a, --all             Show all available entries in the ring buffer\n");
    printf("  -g, --grep <regex>    Case-insensitive POSIX extended regex on cmd & output\n");
    printf("  -e, --errors          Show only failed commands (rc != 0)\n");
    printf("      --slow <ms>       Show only commands slower than <ms> milliseconds\n");
    printf("      --since <spec>    Filter since relative time (30s, 5m, 1h) or HH:MM:SS\n");
    printf("  -f, --follow          Live-tail new commands and outputs in real time\n");
    printf("  -s, --server <name>   Query log of named server <name>\n");
    printf("      --json            Output matching entries as JSON Lines\n");
    printf("      --clear           Clear all entries in the server ring buffer\n");
    printf("  -h, --help            Show this help message\n");
}

/**
 * @brief Execute the `@log` / `@server log` query, search, or live-follow command.
 */
int milk_srvlog_cmd_log(const char                *default_sockpath,
                        int                        argc,
                        char                     **argv,
                        const milk_client_hooks_t *hooks)
{
    int         last_n      = 20;
    int         show_all    = 0;
    int         errors_only = 0;
    int         follow_mode = 0;
    int         json_mode   = 0;
    int         clear_mode  = 0;
    uint64_t    min_dur_ns  = 0;
    time_t      since_sec   = 0;
    const char *pattern     = NULL;
    const char *target_srv  = NULL;

    for (int i = 0; i < argc; i++)
    {
        const char *a = argv[i];
        if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0)
        {
            print_log_help();
            return 0;
        }
        else if ((strcmp(a, "-n") == 0 || strcmp(a, "--last") == 0) && i + 1 < argc)
        {
            last_n = atoi(argv[++i]);
            if (last_n <= 0)
            {
                last_n = 20;
            }
        }
        else if (strcmp(a, "-a") == 0 || strcmp(a, "--all") == 0)
        {
            show_all = 1;
        }
        else if ((strcmp(a, "-g") == 0 || strcmp(a, "--grep") == 0) && i + 1 < argc)
        {
            pattern = argv[++i];
        }
        else if (strcmp(a, "-e") == 0 || strcmp(a, "--errors") == 0)
        {
            errors_only = 1;
        }
        else if (strcmp(a, "--slow") == 0 && i + 1 < argc)
        {
            double ms  = strtod(argv[++i], NULL);
            min_dur_ns = (ms > 0.0) ? (uint64_t) (ms * 1e6) : 0;
        }
        else if (strcmp(a, "--since") == 0 && i + 1 < argc)
        {
            since_sec = parse_since_spec(argv[++i]);
        }
        else if (strcmp(a, "-f") == 0 || strcmp(a, "--follow") == 0)
        {
            follow_mode = 1;
        }
        else if ((strcmp(a, "-s") == 0 || strcmp(a, "--server") == 0) && i + 1 < argc)
        {
            target_srv = argv[++i];
        }
        else if (strcmp(a, "--json") == 0)
        {
            json_mode = 1;
        }
        else if (strcmp(a, "--clear") == 0)
        {
            clear_mode = 1;
        }
        else if (a[0] != '-' && pattern == NULL)
        {
            pattern = a;
        }
    }

    char sockpath[256];
    if (target_srv != NULL && target_srv[0] != '\0')
    {
        milk_srvlog_make_sock(target_srv, sockpath, sizeof(sockpath));
    }
    else if (default_sockpath != NULL && default_sockpath[0] != '\0')
    {
        snprintf(sockpath, sizeof(sockpath), "%s", default_sockpath);
    }
    else
    {
        milk_ipc_resolve_sockpath(hooks, sockpath, sizeof(sockpath), 0);
    }

    char path[512];
    if (milk_srvlog_shm_path(sockpath, path, sizeof(path)) != 0)
    {
        return 1;
    }

    int fd = open(path, clear_mode ? O_RDWR : O_RDONLY, 0600);
    if (fd < 0)
    {
        if (clear_mode)
        {
            return 0;
        }
        if (!json_mode)
        {
            printf("No command log entries yet for server '%s'.\n", sockpath);
        }
        return 0;
    }

    milk_srvlog_shm_t *shm = (milk_srvlog_shm_t *) mmap(
        NULL, sizeof(milk_srvlog_shm_t), clear_mode ? (PROT_READ | PROT_WRITE) : PROT_READ,
        MAP_SHARED, fd, 0);
    close(fd);
    if (shm == MAP_FAILED)
    {
        return 1;
    }

    if (shm->magic != MILK_SRVLOG_MAGIC)
    {
        munmap(shm, sizeof(milk_srvlog_shm_t));
        return 0;
    }

    if (clear_mode)
    {
        __atomic_store_n(&shm->write_seq, 0, __ATOMIC_SEQ_CST);
        __atomic_store_n(&shm->total_cmds, 0, __ATOMIC_RELAXED);
        __atomic_store_n(&shm->total_errors, 0, __ATOMIC_RELAXED);
        for (size_t i = 0; i < MILK_SRVLOG_CAP; i++)
        {
            shm->entries[i].seq = 0;
        }
        munmap(shm, sizeof(milk_srvlog_shm_t));
        return 0;
    }

    regex_t re;
    int     has_regex = 0;
    if (pattern != NULL && pattern[0] != '\0')
    {
        if (regcomp(&re, pattern, REG_EXTENDED | REG_ICASE | REG_NOSUB) != 0)
        {
            fprintf(stderr, "ERROR: @log: invalid regular expression '%s'\n", pattern);
            munmap(shm, sizeof(milk_srvlog_shm_t));
            return 1;
        }
        has_regex = 1;
    }

    int      color = use_color_output();
    uint64_t wseq  = __atomic_load_n(&shm->write_seq, __ATOMIC_ACQUIRE);
    uint64_t first = (wseq > MILK_SRVLOG_CAP) ? (wseq - MILK_SRVLOG_CAP + 1) : 1;

    milk_srvlog_entry_t *matched =
        (milk_srvlog_entry_t *) calloc(MILK_SRVLOG_CAP, sizeof(milk_srvlog_entry_t));
    if (matched == NULL)
    {
        if (has_regex)
        {
            regfree(&re);
        }
        munmap(shm, sizeof(milk_srvlog_shm_t));
        return 1;
    }

    size_t nmatch = 0;
    for (uint64_t s = first; s <= wseq; s++)
    {
        milk_srvlog_entry_t snap = shm->entries[(s - 1) % MILK_SRVLOG_CAP];
        if (snap.seq == s &&
            entry_matches_filters(&snap, errors_only, min_dur_ns, since_sec, has_regex, &re))
        {
            matched[nmatch++] = snap;
        }
    }

    size_t start_idx = 0;
    if (!show_all && (size_t) last_n < nmatch)
    {
        start_idx = nmatch - (size_t) last_n;
    }

    if (!json_mode && (nmatch > 0 || follow_mode))
    {
        printf("%s%-12s  %-9s  %3s  %6s  %s%s\n", color ? C_HDR : "", "TIME", "DURATION", "RC",
               "PID", "COMMAND & OUTPUT", color ? C_RST : "");
    }

    for (size_t i = start_idx; i < nmatch; i++)
    {
        format_entry_line(&matched[i], json_mode, color);
    }
    free(matched);
    fflush(stdout);

    if (follow_mode)
    {
        struct sigaction sa, old_sa;
        memset(&sa, 0, sizeof(sa));
        sa.sa_handler = follow_sigint_handler;
        sigemptyset(&sa.sa_mask);
        g_follow_sigint = 0;
        sigaction(SIGINT, &sa, &old_sa);

        uint64_t cur_seq = wseq;
        while (!g_follow_sigint)
        {
            uint64_t now_seq = __atomic_load_n(&shm->write_seq, __ATOMIC_ACQUIRE);
            if (now_seq > cur_seq)
            {
                if (now_seq - cur_seq > MILK_SRVLOG_CAP)
                {
                    cur_seq = now_seq - MILK_SRVLOG_CAP;
                }
                for (uint64_t s = cur_seq + 1; s <= now_seq; s++)
                {
                    milk_srvlog_entry_t snap = shm->entries[(s - 1) % MILK_SRVLOG_CAP];
                    if (snap.seq == s && entry_matches_filters(&snap, errors_only, min_dur_ns,
                                                               since_sec, has_regex, &re))
                    {
                        format_entry_line(&snap, json_mode, color);
                    }
                }
                fflush(stdout);
                cur_seq = now_seq;
            }
            else
            {
                struct timespec req = { 0, 15000000L }; /* 15 ms */
                nanosleep(&req, NULL);
            }
        }
        sigaction(SIGINT, &old_sa, NULL);
    }

    if (has_regex)
    {
        regfree(&re);
    }
    munmap(shm, sizeof(milk_srvlog_shm_t));
    return 0;
}
