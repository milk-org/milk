// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_srvlog.c
 * @brief Shared-memory server telemetry, multi-server management (@server),
 *        and high-performance searchable command/output ring-buffer log (@log).
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <regex.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "CLIcore.h"
#include "milk_proto.h"
#include "milk_srvlog.h"

#define C_RST "\033[0m"
#define C_TITLE "\033[1;36m"
#define C_HDR "\033[1;35m"
#define C_CMD "\033[32m"
#define C_BOLD "\033[1m"
#define C_DIM "\033[2m"
#define C_NOTE "\033[33m"
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

/* ============================================================
 *  Socket & Shared-Memory Path Resolution
 * ============================================================ */

static void get_srvlog_dir(char *out, size_t outsz)
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
 * @brief Parse a MILKCLI_SOCK string into its numeric UID and logical server name.
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
 * @brief Build a canonical MILKCLI_SOCK string from a logical server name or socket.
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
 * @brief Resolve the filesystem path to a server's .log.shm file.
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
    get_srvlog_dir(shmdir, sizeof(shmdir));
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

static int is_numeric_str(const char *s)
{
    if (s == NULL || s[0] == '\0')
    {
        return 0;
    }
    for (size_t i = 0; s[i] != '\0'; i++)
    {
        if (!isdigit((unsigned char) s[i]))
        {
            return 0;
        }
    }
    return 1;
}

/* ============================================================
 *  Server-Side SHM Lifecycle & Ring-Buffer Recording
 * ============================================================ */

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

    /* Preserve pre-existing entries if created by client fast-path in the same session */
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
        (owner_pid > 0) ? owner_pid : (is_numeric_str(sname) ? (pid_t) atoi(sname) : 0);
    shm->flags |= flags;

    if (!is_numeric_str(sname) || getenv("MILK_SRV_NAMED") != NULL)
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
        /* Trim trailing newlines for clean display */
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
 * @brief Append a client-side fast-path operation (@get/@set) to the session's SHM log.
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
        if (is_numeric_str(sname))
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

/* ============================================================
 *  Log Search, Filtering & Formatting (@log / @server log)
 * ============================================================ */

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

    /* Check relative duration: e.g. 30s, 5m, 2h, 1d */
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

    /* Check HH:MM:SS today */
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

    /* Check YYYY-MM-DD[T ]HH:MM:SS */
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
        const char *s = (hooks != NULL && hooks->get_var != NULL)
                            ? hooks->get_var("MILKCLI_SOCK", hooks->ctx)
                            : getenv("MILKCLI_SOCK");
        if (s != NULL && s[0] != '\0')
        {
            snprintf(sockpath, sizeof(sockpath), "%s", s);
        }
        else
        {
            snprintf(sockpath, sizeof(sockpath), "@milk-cli-%u-%d", (unsigned) getuid(),
                     (int) getppid());
        }
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

    /* Collect matching entries in chronological order */
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

/* ============================================================
 *  Multi-Server Management (@server status|list|start|use|local|stop|attach)
 * ============================================================ */

static int try_ping_socket(const char *sockpath)
{
    struct sockaddr_un addr;
    socklen_t          addrlen = 0;
    if (milk_sockaddr_init(sockpath, &addr, &addrlen) != 0)
    {
        return 0;
    }
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0)
    {
        return 0;
    }
    if (connect(fd, (struct sockaddr *) &addr, addrlen) != 0)
    {
        close(fd);
        return 0;
    }
    int ok = 0;
    if (milk_proto_send(fd, MILK_MSG_PING, 0, NULL, 0) == 0)
    {
        milk_frame_hdr_t hdr;
        uint8_t         *pl = NULL;
        if (milk_proto_recv(fd, &hdr, &pl, NULL) == 0 && hdr.type == MILK_MSG_DONE)
        {
            ok = 1;
        }
        free(pl);
    }
    close(fd);
    return ok;
}

static int send_shutdown_socket(const char *sockpath)
{
    struct sockaddr_un addr;
    socklen_t          addrlen = 0;
    if (milk_sockaddr_init(sockpath, &addr, &addrlen) != 0)
    {
        return 0;
    }
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0)
    {
        return 0;
    }
    if (connect(fd, (struct sockaddr *) &addr, addrlen) != 0)
    {
        close(fd);
        return 0;
    }
    milk_proto_send(fd, MILK_MSG_SHUTDOWN, 0, NULL, 0);
    milk_frame_hdr_t hdr;
    uint8_t         *pl = NULL;
    (void) milk_proto_recv(fd, &hdr, &pl, NULL);
    free(pl);
    close(fd);
    return 1;
}

static void resolve_current_sock(const milk_client_hooks_t *hooks, char *out, size_t outsz)
{
    const char *s = (hooks != NULL && hooks->get_var != NULL)
                        ? hooks->get_var("MILKCLI_SOCK", hooks->ctx)
                        : getenv("MILKCLI_SOCK");
    if (s != NULL && s[0] != '\0')
    {
        snprintf(out, outsz, "%s", s);
        return;
    }
    pid_t ref_pid = (hooks != NULL && hooks->get_var != NULL) ? getpid() : getppid();
    snprintf(out, outsz, "@milk-cli-%u-%d", (unsigned) getuid(), (int) ref_pid);
}

static void set_shell_var(const milk_client_hooks_t *hooks, const char *name, const char *val)
{
    if (hooks != NULL && hooks->set_var != NULL)
    {
        hooks->set_var(name, val, hooks->ctx);
    }
    setenv(name, val ? val : "", 1);

    const char *eval_fd_str = getenv("MILK_CALL_EVAL_FD");
    if (eval_fd_str != NULL && eval_fd_str[0] != '\0')
    {
        int efd = atoi(eval_fd_str);
        if (efd > 2)
        {
            char line[512];
            int  n = snprintf(line, sizeof(line), "export %s='%s'\n", name, val ? val : "");
            if (n > 0 && n < (int) sizeof(line))
            {
                if (write(efd, line, (size_t) n) < 0)
                {
                }
            }
        }
    }
}

static void resolve_milk_cli_binary(const milk_client_hooks_t *hooks, char *out, size_t outsz)
{
    const char *bin = (hooks != NULL && hooks->get_var != NULL)
                          ? hooks->get_var("MILKCLI_BIN", hooks->ctx)
                          : getenv("MILKCLI_BIN");
    if (bin != NULL && bin[0] != '\0' && access(bin, X_OK) == 0)
    {
        snprintf(out, outsz, "%s", bin);
        return;
    }
    const char *inst = (hooks != NULL && hooks->get_var != NULL)
                           ? hooks->get_var("MILK_INSTALLDIR", hooks->ctx)
                           : getenv("MILK_INSTALLDIR");
    if (inst != NULL && inst[0] != '\0')
    {
        char cand[512];
        snprintf(cand, sizeof(cand), "%s/bin/milk-cli", inst);
        if (access(cand, X_OK) == 0)
        {
            snprintf(out, outsz, "%s", cand);
            return;
        }
    }
    snprintf(out, outsz, "milk-cli");
}

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

    if (try_ping_socket(sockpath))
    {
        char lpath[512];
        if (milk_srvlog_shm_path(sockpath, lpath, sizeof(lpath)) == 0 && access(lpath, R_OK) == 0)
        {
            return 0;
        }
        send_shutdown_socket(sockpath);
        for (int w = 0; w < 50 && try_ping_socket(sockpath); w++)
        {
            struct timespec req = { 0, 5000000L };
            nanosleep(&req, NULL);
        }
    }

    char binpath[512];
    resolve_milk_cli_binary(hooks, binpath, sizeof(binpath));

    if (use_tmux)
    {
        char tmux_sess[64];
        snprintf(tmux_sess, sizeof(tmux_sess), "milk_srv_%s", srv_name);

        char shmdir[256];
        get_srvlog_dir(shmdir, sizeof(shmdir));

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

        /* Poll until socket is ready (up to 5 seconds) */
        for (int iter = 0; iter < 500; iter++)
        {
            if (try_ping_socket(sockpath))
            {
                return 0;
            }
            struct timespec req = { 0, 10000000L }; /* 10 ms */
            nanosleep(&req, NULL);
        }
        fprintf(stderr, "ERROR: @server: timed out waiting for tmux server '%s'\n", srv_name);
        return 1;
    }

    int ready_pipe[2] = { -1, -1 };
    if (pipe(ready_pipe) != 0)
    {
        return 1;
    }
    fcntl(ready_pipe[0], F_SETFD, FD_CLOEXEC);

    pid_t pid = fork();
    if (pid < 0)
    {
        close(ready_pipe[0]);
        close(ready_pipe[1]);
        return 1;
    }
    if (pid == 0)
    {
        close(ready_pipe[0]);
        setsid();
        pid_t pid2 = fork();
        if (pid2 < 0)
        {
            _exit(1);
        }
        if (pid2 > 0)
        {
            _exit(0);
        }

        int devnull = open("/dev/null", O_RDWR);
        if (devnull >= 0)
        {
            dup2(devnull, STDIN_FILENO);
            dup2(devnull, STDOUT_FILENO);
            dup2(devnull, STDERR_FILENO);
            if (devnull > 2)
            {
                close(devnull);
            }
        }
        setenv("MILK_SRV_NAMED", "1", 1);
        char rfd_str[32];
        snprintf(rfd_str, sizeof(rfd_str), "%d", ready_pipe[1]);
        execlp(binpath, binpath, "-q", "--server", sockpath, "--ready-fd", rfd_str, (char *) NULL);
        _exit(127);
    }

    close(ready_pipe[1]);
    while (waitpid(pid, NULL, 0) < 0 && errno == EINTR)
    {
    }

    struct pollfd pfd = { ready_pipe[0], POLLIN, 0 };
    int           pr  = poll(&pfd, 1, 5000);
    char          rbuf[16];
    ssize_t       rn = 0;
    if (pr > 0 && (pfd.revents & POLLIN))
    {
        rn = read(ready_pipe[0], rbuf, sizeof(rbuf));
    }
    close(ready_pipe[0]);
    return (rn > 0) ? 0 : 1;
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

static int cmd_server_status(const char *target_arg, const milk_client_hooks_t *hooks)
{
    char cur_sock[256];
    resolve_current_sock(hooks, cur_sock, sizeof(cur_sock));

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

    int  alive = try_ping_socket(sockpath);
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

    const char *mode_str = is_numeric_str(sname) ? "private (per-shell)" : "named (persistent)";
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

static int cmd_server_list(int json_mode, const milk_client_hooks_t *hooks)
{
    char cur_sock[256];
    resolve_current_sock(hooks, cur_sock, sizeof(cur_sock));

    char shmdir[256];
    get_srvlog_dir(shmdir, sizeof(shmdir));

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
        int sock_alive  = pid_alive ? try_ping_socket(snap.sockpath) : 0;

        if (!sock_alive && !owner_alive && strcmp(snap.sockpath, cur_sock) != 0)
        {
            /* Clean up stale SHM file from terminated process */
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
        int         alive  = try_ping_socket(cur_sock);
        const char *mode_s = is_numeric_str(sname) ? "private" : "named";
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

    if (send_shutdown_socket(sockpath))
    {
        for (int w = 0; w < 40 && try_ping_socket(sockpath); w++)
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

/**
 * @brief Execute the `@server` management command (status, list, start, use, local, stop, attach).
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
        const char *target = (argc >= 2) ? argv[1] : NULL;
        return cmd_server_status(target, hooks);
    }

    if (strcmp(sub, "list") == 0 || strcmp(sub, "ls") == 0)
    {
        int json_mode = (argc >= 2 && strcmp(argv[1], "--json") == 0);
        return cmd_server_list(json_mode, hooks);
    }

    if (strcmp(sub, "log") == 0)
    {
        char cur_sock[256];
        resolve_current_sock(hooks, cur_sock, sizeof(cur_sock));
        return milk_srvlog_cmd_log(cur_sock, argc - 1, argv + 1, hooks);
    }

    if (strcmp(sub, "start") == 0 || strcmp(sub, "use") == 0)
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
            set_shell_var(hooks, "MILKCLI_SOCK", sockpath);
            set_shell_var(hooks, "_MILK_SERVER_NAME", sname);
            printf("Switched to milk server '%s' (%s)%s.\n", sname, sockpath,
                   use_tmux ? " [tmux]" : "");
        }
        else
        {
            printf("Started milk server '%s' (%s)%s.\n", sname, sockpath,
                   use_tmux ? " [tmux]" : "");
        }
        return 0;
    }

    if (strcmp(sub, "local") == 0 || strcmp(sub, "detach") == 0)
    {
        const char *orig_sock = (hooks != NULL && hooks->get_var != NULL)
                                    ? hooks->get_var("_MILK_LOCAL_SOCK", hooks->ctx)
                                    : getenv("_MILK_LOCAL_SOCK");
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
        set_shell_var(hooks, "MILKCLI_SOCK", local_sock);
        set_shell_var(hooks, "_MILK_SERVER_NAME", "");
        printf("Switched to local shell server (%s).\n", local_sock);
        return 0;
    }

    if (strcmp(sub, "stop") == 0)
    {
        if (argc >= 2 && strcmp(argv[1], "--all") == 0)
        {
            char shmdir[256];
            get_srvlog_dir(shmdir, sizeof(shmdir));
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
        resolve_current_sock(hooks, cur_sock, sizeof(cur_sock));
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
            const char *orig_sock = (hooks != NULL && hooks->get_var != NULL)
                                        ? hooks->get_var("_MILK_LOCAL_SOCK", hooks->ctx)
                                        : getenv("_MILK_LOCAL_SOCK");
            if (orig_sock != NULL && orig_sock[0] != '\0')
            {
                set_shell_var(hooks, "MILKCLI_SOCK", orig_sock);
            }
            set_shell_var(hooks, "_MILK_SERVER_NAME", "");
        }
        return 0;
    }

    if (strcmp(sub, "attach") == 0)
    {
        char cur_sock[256];
        resolve_current_sock(hooks, cur_sock, sizeof(cur_sock));
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

    fprintf(stderr, "ERROR: @server: unknown subcommand '%s'\n", sub);
    print_server_help();
    return 1;
}
