// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_unit.c
 * @brief   Unit lifecycle (fps_unit.h): the starter side (start, stop, environment) and the worker side
 */

#include <ctype.h>
#include <dirent.h>
#include <fcntl.h>
#include <pwd.h>
#include <stdarg.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "computeunit.h"
#include "fps_unit.h"
#include "milk_detach.h"
#include "milk_notify.h"
#include "milk_os.h"
#include "milk_spawn.h"
#include "processinfo_list.h"
#include "processinfo_procdirname.h"

extern char **environ;

#define FPSUNIT_START_TIMEOUT_S 10.0
#define FPSUNIT_STOP_T2_MS 2000
#define FPSUNIT_STOP_T3_MS 2000
#define FPSUNIT_OUT_MAX (1024LL * 1024LL)
#define FPSUNIT_ENV_HEADER "# milk unit environment"

/* The process's own role, between fps_unit_role_begin() and fps_unit_role_end() */
static struct
{
    int           active;
    int           worker; /* started by fps_unit_start() */
    FPS_UNIT_ROLE role;
    int           notify_fd;
    int           failed; /* the starter already has the reason (ERRNO=) */
    int           info_fd;
    char          out[PATH_MAX];
    long long     out_max;
    int           marks;   /* stdout is the role's output file: marker lines go there */
    int           ready;   /* READY=1 sent */
    int           svc_fd;  /* the service manager's notify socket (NOTIFY_SOCKET), -1: none */
    double        wd_half; /* half of WATCHDOG_USEC in s, 0: no watchdog */
    double        wd_last;
    double        st_last; /* last STATUS= with the loop rate */
    long          st_loopcnt;
} fpsunit_self = { .notify_fd = -1, .info_fd = -1, .svc_fd = -1 };

time_t fps_unit_tick_deadline = 0;

static double fpsunit_now(void)
{
    struct timespec t;

    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + 1e-9 * (double) t.tv_nsec;
}

/* "=== <local time> <text>" to @fd */
static void fpsunit_vmark(int fd, const char *fmt, va_list ap)
{
    char      line[1024];
    time_t    t = time(NULL);
    struct tm tm;
    int       n;

    if (fd < 0)
    {
        return;
    }
    localtime_r(&t, &tm);
    n = (int) strftime(line, sizeof(line), "=== %Y-%m-%d %H:%M:%S ", &tm);
    n += vsnprintf(line + n, sizeof(line) - (size_t) n, fmt, ap);
    if (n > (int) sizeof(line) - 2)
    {
        n = (int) sizeof(line) - 2;
    }
    line[n++] = '\n';
    (void) !write(fd, line, (size_t) n);
}

static void fpsunit_mark(int fd, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void fpsunit_mark(int fd, const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    fpsunit_vmark(fd, fmt, ap);
    va_end(ap);
}

/* Marker line in the unit's <role>.out, if it exists */
static void fpsunit_mark_file(const char *name, FPS_UNIT_ROLE role, const char *fmt, ...)
    __attribute__((format(printf, 3, 4)));
static void fpsunit_mark_file(const char *name, FPS_UNIT_ROLE role, const char *fmt, ...)
{
    char    path[PATH_MAX];
    int     fd;
    va_list ap;

    if (fps_unit_role_path(name, role, "out", path, sizeof(path)) != 0 ||
        (fd = open(path, O_WRONLY | O_APPEND | O_NOFOLLOW | O_CLOEXEC)) < 0)
    {
        return;
    }
    va_start(ap, fmt);
    fpsunit_vmark(fd, fmt, ap);
    va_end(ap);
    close(fd);
}

/* A message for the user of a start or stop, unless @quiet */
static void fpsunit_say(int quiet, FILE *f, const char *fmt, ...)
    __attribute__((format(printf, 3, 4)));
static void fpsunit_say(int quiet, FILE *f, const char *fmt, ...)
{
    va_list ap;

    if (!quiet)
    {
        va_start(ap, fmt);
        vfprintf(f, fmt, ap);
        va_end(ap);
    }
}

/* "<program> pid N", as the starter or stopper of a unit */
static void fpsunit_me(char *buf, size_t size)
{
    char        exe[PATH_MAX] = "?";
    const char *base;

    milkos_self_exe(exe, sizeof(exe));
    base = strrchr(exe, '/');
    snprintf(buf, size, "%s pid %d%s", (base != NULL) ? base + 1 : exe, (int) getpid(),
             (geteuid() == 0) ? " (root)" : "");
}

/* What the worker wrote to @path after offset @from (at most the last 4 KiB), to stderr */
static void fpsunit_print_tail(const char *path, off_t from, int quiet)
{
    char buf[4096];
    int  fd = quiet ? -1 : open(path, O_RDONLY | O_CLOEXEC);

    if (fd < 0)
    {
        return;
    }
    off_t end = lseek(fd, 0, SEEK_END);
    if (end - from > (off_t) sizeof(buf) - 1)
    {
        from = end - (off_t) (sizeof(buf) - 1);
    }
    ssize_t n = (end > from) ? pread(fd, buf, (size_t) (end - from), from) : 0;
    close(fd);
    if (n <= 0)
    {
        return;
    }
    buf[n] = '\0';
    fprintf(stderr, "  --- %s ---\n", path);
    for (char *l = buf; *l != '\0';)
    {
        char *nl = strchr(l, '\n');
        if (nl != NULL)
        {
            *nl = '\0';
        }
        fprintf(stderr, "  %s\n", l);
        if (nl == NULL)
        {
            break;
        }
        l = nl + 1;
    }
}

static void fpsunit_owner_hint(const char *name, int quiet)
{
    char           shm[STRINGMAXLEN_SHMDIRNAME];
    char           path[PATH_MAX];
    struct stat    sb;
    struct passwd *pw;

    function_parameter_struct_shmdirname(shm);
    snprintf(path, sizeof(path), "%s/%s.fps.shm", shm, name);
    if (stat(path, &sb) != 0 || (pw = getpwuid(sb.st_uid)) == NULL)
    {
        fpsunit_say(quiet, stderr, "%s: permission denied\n", name);
        return;
    }
    fpsunit_say(quiet, stderr,
                "%s: the unit belongs to %s: run this as that user (sudo -u %s ...) or as root\n",
                name, pw->pw_name, pw->pw_name);
}

/* Take the unit lock, explaining a failure. Return: 0, or -1 */
static int fpsunit_lock(const char *name, int *fd, int quiet)
{
    if ((*fd = fps_unit_lock(name, 1)) >= 0)
    {
        return 0;
    }
    if (errno == ENOENT)
    {
        fpsunit_say(quiet, stderr, "%s: no such FPS\n", name);
    }
    else if (errno == EACCES || errno == EPERM)
    {
        fpsunit_owner_hint(name, quiet);
    }
    else
    {
        fpsunit_say(quiet, stderr, "%s: cannot lock the unit: %s\n", name, strerror(errno));
    }
    return -1;
}

/* ---- environment ---- */

typedef struct
{
    char **v; /* NULL-terminated, each entry malloc'ed */
    int    n;
    int    cap;
} FPSUNIT_ENV;

static int fpsunit_env_valid(const char *kv)
{
    size_t klen = strcspn(kv, "=");

    if (klen == 0 || kv[klen] != '=' || isdigit((unsigned char) kv[0]))
    {
        return 0;
    }
    for (size_t i = 0; i < klen; i++)
    {
        if (!isalnum((unsigned char) kv[i]) && kv[i] != '_')
        {
            return 0;
        }
    }
    return 1;
}

/* Variables a snapshot leaves out: the terminal's and session's, and the unit protocol's */
static int fpsunit_env_keep(const char *kv)
{
    static const char *const drop[] = { "TMUX",
                                        "STY=",
                                        "SSH_",
                                        "DISPLAY=",
                                        "TERM=",
                                        "PWD=",
                                        "OLDPWD=",
                                        "SHLVL=",
                                        "_=",
                                        "_MILK_SERVER_",
                                        "MILKCLI_SOCK=",
                                        "NOTIFY_SOCKET",
                                        "WATCHDOG_",
                                        "INVOCATION_ID=",
                                        "JOURNAL_STREAM=",
                                        "MANAGERPID=",
                                        "LISTEN_",
                                        "SYSTEMD_EXEC_PID=",
                                        NULL };

    if (!fpsunit_env_valid(kv) || strchr(kv, '\n') != NULL)
    {
        return 0;
    }
    for (int i = 0; drop[i] != NULL; i++)
    {
        if (strncmp(kv, drop[i], strlen(drop[i])) == 0)
        {
            return 0;
        }
    }
    return strncmp(kv, "MILK_UNIT_", 10) != 0 || strncmp(kv, "MILK_UNIT_OUT_MAX=", 18) == 0;
}

/* Add "KEY=VALUE" @kv (malloc'ed, owned from now on), replacing an entry with the same KEY */
static int fpsunit_env_put(FPSUNIT_ENV *e, char *kv)
{
    if (kv == NULL)
    {
        return -1;
    }
    size_t klen = strcspn(kv, "=") + 1;
    for (int i = 0; i < e->n; i++)
    {
        if (strncmp(e->v[i], kv, klen) == 0)
        {
            free(e->v[i]);
            e->v[i] = kv;
            return 0;
        }
    }
    if (e->n + 2 > e->cap)
    {
        int    cap = (e->cap > 0) ? 2 * e->cap : 256;
        char **v   = realloc(e->v, (size_t) cap * sizeof(char *));
        if (v == NULL)
        {
            free(kv);
            return -1;
        }
        e->v   = v;
        e->cap = cap;
    }
    e->v[e->n++] = kv;
    e->v[e->n]   = NULL;
    return 0;
}

static int fpsunit_env_set(FPSUNIT_ENV *e, const char *key, const char *val)
{
    size_t len = strlen(key) + strlen(val) + 2;
    char  *kv  = malloc(len);

    if (kv == NULL)
    {
        return -1;
    }
    snprintf(kv, len, "%s=%s", key, val);
    return fpsunit_env_put(e, kv);
}

static void fpsunit_env_free(FPSUNIT_ENV *e)
{
    for (int i = 0; i < e->n; i++)
    {
        free(e->v[i]);
    }
    free(e->v);
    memset(e, 0, sizeof(*e));
}

/* "KEY=VALUE" as KEY="VALUE", with \ " ` and $ escaped: systemd's EnvironmentFile= reads it back as is */
static void fpsunit_env_write(FILE *f, const char *kv)
{
    size_t klen = strcspn(kv, "=");

    fprintf(f, "%.*s=\"", (int) klen, kv);
    for (const char *c = kv + klen + 1; *c != '\0'; c++)
    {
        if (strchr("\\\"`$", *c) != NULL)
        {
            fputc('\\', f);
        }
        fputc(*c, f);
    }
    fputs("\"\n", f);
}

/* A line of the env file as "KEY=VALUE" (malloc'ed), NULL to skip it. @quoted: the file has the header,
 * so a value is read as EnvironmentFile= reads it; an older snapshot's are verbatim */
static char *fpsunit_env_parse(const char *line, int quoted)
{
    size_t      klen = strcspn(line, "=");
    const char *v    = line + klen + 1;
    char       *kv;
    char       *o;

    if (!fpsunit_env_valid(line))
    {
        return NULL; // also comments and empty lines
    }
    if (!quoted || (kv = malloc(strlen(line) + 1)) == NULL)
    {
        return quoted ? NULL : strdup(line);
    }
    memcpy(kv, line, klen + 1);
    o = kv + klen + 1;
    while (*v == ' ' || *v == '\t')
    {
        v++;
    }
    if (*v == '"')
    {
        for (v++; *v != '\0' && *v != '"'; v++)
        {
            v += (*v == '\\' && v[1] != '\0' && strchr("\\\"`$", v[1]) != NULL);
            *o++ = *v;
        }
    }
    else if (*v == '\'')
    {
        for (v++; *v != '\0' && *v != '\''; v++)
        {
            *o++ = *v;
        }
    }
    else
    {
        for (; *v != '\0'; v++)
        {
            v += (*v == '\\' && v[1] != '\0');
            *o++ = *v;
        }
        while (o > kv + klen + 1 && (o[-1] == ' ' || o[-1] == '\t' || o[-1] == '\r'))
        {
            o--;
        }
    }
    *o = '\0';
    return kv;
}

int fps_unit_env_capture(const char *name)
{
    char  path[PATH_MAX];
    char  tmp[PATH_MAX + 8];
    FILE *f;
    int   fd;

    if (fps_unit_dir_create(name) != 0 || fps_unit_path(name, "env", path, sizeof(path)) != 0)
    {
        return -1;
    }
    snprintf(tmp, sizeof(tmp), "%s.new", path);
    if ((fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW | O_CLOEXEC, 0600)) < 0)
    {
        return -1;
    }
    if ((f = fdopen(fd, "w")) == NULL)
    {
        close(fd);
        unlink(tmp);
        return -1;
    }
    fprintf(
        f,
        "%s, captured at fpsinit: KEY=\"value\" lines, as systemd's EnvironmentFile= reads them\n",
        FPSUNIT_ENV_HEADER);
    for (char **e = environ; *e != NULL; e++)
    {
        if (fpsunit_env_keep(*e))
        {
            fpsunit_env_write(f, *e);
        }
    }
    if (fclose(f) != 0 || rename(tmp, path) != 0) // a start never reads half a file
    {
        unlink(tmp);
        return -1;
    }
    return 0;
}

/* The overrides of a unit process's environment (U8): directories, threads, work directory; with
 * @role >= 0, the unit's name and role */
static int fpsunit_env_overrides(FPSUNIT_ENV *e,
                                 const char  *name,
                                 int          role,
                                 FPS         *fps,
                                 const char  *base)
{
    char shm[STRINGMAXLEN_SHMDIRNAME];
    char procdir[STRINGMAXLEN_FULLFILENAME];
    char num[32];
    int  rc = 0;

    function_parameter_struct_shmdirname(shm);
    processinfo_procdirname(procdir);
    rc |= fpsunit_env_set(e, "MILK_SHM_DIR", shm);
    rc |= fpsunit_env_set(e, "MILK_PROC_DIR", procdir);
    rc |= fpsunit_env_set(e, "MILK_UNIT_DIR", base);
    if (role >= 0 && functionparameter_GetParamIndex(fps, ".procinfo.NBthread") >= 0)
    {
        long nbthread = functionparameter_GetParamValue_INT64(fps, ".procinfo.NBthread");
        if (nbthread > 0)
        {
            snprintf(num, sizeof(num), "%ld", nbthread); // read by libgomp when it loads (U8)
            rc |= fpsunit_env_set(e, "OMP_NUM_THREADS", num);
        }
    }
    if (fps->md->workdir[0] != '\0')
    {
        rc |= fpsunit_env_set(e, "PWD", fps->md->workdir);
    }
    const char *session = getenv("_MILK_SERVER_NAME");
    if (session != NULL && session[0] != '\0')
    {
        rc |= fpsunit_env_set(e, "_MILK_SERVER_NAME", session);
    }
    if (role >= 0)
    {
        rc |= fpsunit_env_set(e, "MILK_UNIT_NAME", name);
        rc |= fpsunit_env_set(e, "MILK_UNIT_ROLE", fps_unit_role_name((FPS_UNIT_ROLE) role));
    }
    return rc;
}

/* The worker's envp: the snapshot (else the starter's environment), then the overrides (U8). A script
 * (@role < 0) gets the directories, not the worker protocol. */
static int fpsunit_env_build(FPSUNIT_ENV *e,
                             const char  *name,
                             int          role,
                             FPS         *fps,
                             const char  *base,
                             const char  *me,
                             int          ofd)
{
    char  path[PATH_MAX];
    char  num[32];
    FILE *f  = NULL;
    int   fd = -1;
    int   rc = 0;

    if (fps_unit_path(name, "env", path, sizeof(path)) == 0 &&
        (fd = open(path, O_RDONLY | O_CLOEXEC)) >= 0)
    {
        f = fdopen(fd, "r");
    }
    if (f != NULL)
    {
        char  *line   = NULL;
        size_t cap    = 0;
        int    quoted = -1;
        while (rc == 0 && getline(&line, &cap, f) > 0)
        {
            line[strcspn(line, "\n")] = '\0';
            if (quoted < 0)
            {
                quoted = (strncmp(line, FPSUNIT_ENV_HEADER, strlen(FPSUNIT_ENV_HEADER)) == 0);
            }
            char *kv = fpsunit_env_parse(line, quoted);
            rc       = (kv != NULL) ? fpsunit_env_put(e, kv) : 0;
        }
        free(line);
        fclose(f);
    }
    else
    {
        if (fd >= 0)
        {
            close(fd);
        }
        fpsunit_mark(
            ofd, "no environment snapshot (FPS made by an older version): the starter's is used");
        for (char **v = environ; rc == 0 && *v != NULL; v++)
        {
            rc = fpsunit_env_keep(*v) ? fpsunit_env_put(e, strdup(*v)) : 0;
        }
    }

    rc |= fpsunit_env_overrides(e, name, role, fps, base);
    if (role < 0)
    {
        return rc;
    }
    snprintf(num, sizeof(num), "%d", MILKDETACH_NOTIFY_FD);
    rc |= fpsunit_env_set(e, "MILK_UNIT_WORKER", "1");
    rc |= fpsunit_env_set(e, "MILK_UNIT_NOTIFY_FD", num);
    rc |= fpsunit_env_set(e, "MILK_UNIT_STARTER", me);
    return rc;
}

/* ---- roles that systemd runs (U17) ---- */

int fps_unit_systemd_service(char *unit, size_t size, int *user)
{
    const char *inv = getenv("INVOCATION_ID");

    return inv != NULL && inv[0] != '\0' && milkos_service_unit(unit, size, user) == 0;
}

/* The systemd service that runs @role of @name, unless a process milk started holds the role. Return: 1 or 0 */
static int fpsunit_managed(const char *name, FPS_UNIT_ROLE role, char *unit, size_t size, int *user)
{
    FPS_UNIT_INFO info;

    if (!fps_unit_systemd_read(name, role, unit, size, user))
    {
        return 0;
    }
    return !(fps_unit_info_read(name, role, &info) == 0 && milkdetach_id_alive(&info.id) &&
             strncmp(info.starter, "systemd:", 8) != 0);
}

/* `systemctl [--user] @op @unit`, its messages to ours unless @quiet; waits @timeout_s at most (0: until
 * it returns). Return: its exit status, or FPS_UNIT_START_TIMEOUT */
static int fpsunit_systemctl(const char *op,
                             const char *unit,
                             int         user,
                             double      timeout_s,
                             int         quiet)
{
    const char *argv[5] = { "systemctl", NULL, NULL, NULL, NULL };
    double      end     = fpsunit_now() + timeout_s;
    int         n       = 1;
    int         st      = 1;

    if (user)
    {
        argv[n++] = "--user";
    }
    argv[n++] = op;
    argv[n]   = unit;
    fflush(stdout);
    pid_t pid = milkspawn_start(argv, quiet ? -1 : STDOUT_FILENO, quiet ? -1 : STDERR_FILENO);
    if (pid < 0)
    {
        fpsunit_say(quiet, stderr, "cannot run systemctl: %s\n", strerror(errno));
        return 1;
    }
    while (milkspawn_poll(pid, &st) == 0)
    {
        struct timespec t = { 0, 20000000L };
        if (timeout_s > 0.0 && fpsunit_now() >= end)
        {
            kill(pid, SIGTERM); // the job goes on in the manager
            milkspawn_wait(pid);
            return FPS_UNIT_START_TIMEOUT;
        }
        nanosleep(&t, NULL);
    }
    return st;
}

/* Start (@start) or stop @role of @name through its systemd service @unit (a user manager's if @user) */
static int fpsunit_delegate(const char   *name,
                            FPS_UNIT_ROLE role,
                            int           start,
                            const char   *unit,
                            int           user,
                            double        timeout_s,
                            int           quiet)
{
    const char     *rn = fps_unit_role_name(role);
    const char     *u  = user ? " --user" : "";
    char            shm[STRINGMAXLEN_SHMDIRNAME];
    char            path[PATH_MAX];
    struct stat     sb;
    FPS_UNIT_INFO   info;
    FPS_UNIT_STATUS st;

    function_parameter_struct_shmdirname(shm);
    snprintf(path, sizeof(path), "%s/%s.fps.shm", shm, name);
    if (stat(path, &sb) != 0)
    {
        fpsunit_say(quiet, stderr, "%s: no such FPS\n", name);
        return 1;
    }
    if (user && geteuid() != sb.st_uid)
    {
        fpsunit_say(quiet, stderr,
                    "%s %s: run by %s in its owner's user manager: run this as that user\n", name,
                    rn, unit);
        return 1;
    }
    int alive = (fps_unit_info_read(name, role, &info) == 0 && milkdetach_id_alive(&info.id));
    if (!start && alive)
    {
        fps_unit_info_append(name, role, "stop_requested=1");
    }
    double t0 = fpsunit_now();
    int    rc =
        fpsunit_systemctl(start ? "start" : "stop", unit, user, start ? timeout_s : 0.0, quiet);
    double dt = fpsunit_now() - t0;
    if (rc == FPS_UNIT_START_TIMEOUT)
    {
        fpsunit_say(quiet, stderr,
                    "%s %s: not ready after %.1f s: %s is still starting (systemctl%s status %s)\n",
                    name, rn, dt, unit, u, unit);
        return rc;
    }
    if (rc != 0)
    {
        fpsunit_say(quiet, stderr,
                    "%s %s: systemctl%s %s %s failed (status %d); milk-fps-run --release %s gives "
                    "the unit back to milk\n",
                    name, rn, u, start ? "start" : "stop", unit, rc, name);
        return 1;
    }
    if (start)
    {
        fps_unit_status(name, role, &st);
        fpsunit_say(quiet, stdout, "%s %s: ready after %.2f s, pid %d (%s)\n", name, rn, dt,
                    (int) st.pid, unit);
    }
    else if (alive)
    {
        fpsunit_say(quiet, stdout, "%s %s: stopped by %s (%.2f s)\n", name, rn, unit, dt);
    }
    else
    {
        fpsunit_say(quiet, stdout, "%s %s: not running (%s)\n", name, rn, unit);
    }
    return 0;
}

/* ---- start ---- */

static double fpsunit_default_timeout(void)
{
    const char *v = getenv("MILK_UNIT_START_TIMEOUT");
    double      t = (v != NULL) ? atof(v) : 0.0;

    return (t > 0.0) ? t : FPSUNIT_START_TIMEOUT_S;
}

/* U2: only an fpsexec binary runs a unit */
static int fpsunit_is_fpsexec(const char *exe)
{
    const char *base = strrchr(exe, '/');

    return exe[0] == '/' && base != NULL && strstr(base + 1, "-fpsexec-") != NULL;
}

static void fpsunit_move(const char *src, const char *dst)
{
    char    buf[8192];
    ssize_t n;
    int     in;
    int     out;
    int     ok;

    if (rename(src, dst) == 0 || errno != EXDEV)
    {
        return;
    }
    // MILK_PROC_DIR on another filesystem: copy, then remove
    if ((in = open(src, O_RDONLY | O_CLOEXEC)) < 0)
    {
        return;
    }
    out = open(dst, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW | O_CLOEXEC, 0644);
    ok  = (out >= 0);
    while (ok && (n = read(in, buf, sizeof(buf))) > 0)
    {
        ok = (write(out, buf, (size_t) n) == n);
    }
    close(in);
    if (out >= 0)
    {
        close(out);
    }
    if (ok)
    {
        unlink(src);
    }
}

void fps_unit_procinfo_leftovers(const char *name, int keep)
{
    FPS_UNIT_INFO info;
    char          dst[PATH_MAX];
    char          src[PATH_MAX + 256];
    char          dir[STRINGMAXLEN_FULLFILENAME];
    char          prefix[STRINGMAXLEN_FPS_NAME + 8];
    int           moved = 0;
    DIR          *d;

    if (fps_unit_path(name, "run.procinfo.1", dst, sizeof(dst)) != 0)
    {
        return;
    }
    if (fps_unit_info_read(name, FPS_UNIT_RUN, &info) == 0 && !milkdetach_id_alive(&info.id) &&
        fps_unit_proc_path(name, &info, src, sizeof(src)) == 0 && access(src, F_OK) == 0)
    {
        keep ? fpsunit_move(src, dst) : (void) unlink(src);
        moved = 1;
    }
    processinfo_procdirname(dir);
    snprintf(prefix, sizeof(prefix), "proc.%s.", name);
    if ((d = opendir(dir)) != NULL)
    {
        size_t         plen = strlen(prefix);
        struct dirent *ent;
        while ((ent = readdir(d)) != NULL)
        {
            char *end = NULL;
            long  pid = 0;
            if (strncmp(ent->d_name, prefix, plen) == 0)
            {
                pid = strtol(ent->d_name + plen, &end, 10);
            }
            if (pid <= 0 || strcmp(end, ".shm") != 0 || kill((pid_t) pid, 0) == 0 || errno == EPERM)
            {
                continue;
            }
            snprintf(src, sizeof(src), "%s/%s", dir, ent->d_name);
            if (keep && !moved)
            {
                fpsunit_move(src, dst);
                moved = 1;
            }
            else
            {
                unlink(src);
            }
        }
        closedir(d);
    }
    processinfo_list_reap(0); // frees the registry slots whose file is gone
}

int fps_unit_start(const char *name, FPS_UNIT_ROLE role, double timeout_s, int flags)
{
    int              quiet = (flags & FPS_UNIT_QUIET) != 0;
    const char      *rn    = fps_unit_role_name(role);
    FPS              fps   = { 0 };
    FPS_UNIT_STATUS  st;
    FPSUNIT_ENV      env = { 0 };
    MILKDETACH_SPEC  spec;
    MILKDETACH_READY r;
    struct stat      sb;
    char             out[PATH_MAX];
    char             base[PATH_MAX];
    char             me[PATH_MAX + 32];
    char             cmd[STRINGMAXLEN_FPS_NAME + 16];
    gid_t            groups[256];
    int              lockfd;
    int              ofd = -1;
    int              rc  = 1;

    if (name[0] == '_')
    {
        fpsunit_say(quiet, stderr,
                    "%s: a local FPS lives in one process: it cannot be started as a unit\n", name);
        return 1;
    }
    if (timeout_s <= 0.0)
    {
        timeout_s = fpsunit_default_timeout();
    }
    fflush(stdout); // what the caller printed comes before the start's messages
    char unit[128];
    int  user;
    if (fpsunit_managed(name, role, unit, sizeof(unit), &user))
    {
        return fpsunit_delegate(name, role, 1, unit, user, timeout_s, quiet);
    }
    if (fpsunit_lock(name, &lockfd, quiet) != 0)
    {
        return 1;
    }
    fps.SMfd = -1;
    if (fstat(lockfd, &sb) != 0 || fps_connect(name, &fps, FPSCONNECT_SIMPLE) == -1)
    {
        fpsunit_say(quiet, stderr, "%s: no such FPS\n", name);
        goto out;
    }
    if (sb.st_mode & (S_IWGRP | S_IWOTH))
    {
        fpsunit_say(
            quiet, stderr,
            "%s: refused: the FPS file is writable by group or others, who could change what "
            "the unit runs (chmod 600 it)\n",
            name);
        goto out;
    }
    if (geteuid() != sb.st_uid && geteuid() != 0)
    {
        fpsunit_owner_hint(name, quiet);
        goto out;
    }
    if (!fpsunit_is_fpsexec(fps.md->execfullpath))
    {
        fpsunit_say(quiet, stderr,
                    "%s: not a unit: '%s' created the FPS, and only an fpsexec binary runs a unit "
                    "(<prefix>-fpsexec-<id> -n %s fpsinit)\n",
                    name, fps.md->execfullpath, name);
        goto out;
    }
    fps_unit_status_fps(&fps, role, &st, FPS_UNIT_LOCKED);
    if (st.state == FPS_UNIT_STARTING || st.state == FPS_UNIT_RUNNING)
    {
        fpsunit_say(quiet, stdout, "%s %s: already running, pid %d\n", name, rn, (int) st.pid);
        rc = 0;
        goto out;
    }
    if (st.state == FPS_UNIT_STOPPING)
    {
        fpsunit_say(quiet, stderr, "%s %s: pid %d is stopping\n", name, rn, (int) st.pid);
        goto out;
    }
    if (role == FPS_UNIT_RUN)
    {
        fps_unit_procinfo_leftovers(name, 1);
    }

    if (fps_unit_dir_create(name) != 0 || fps_unit_base_dir(base, sizeof(base)) != 0 ||
        fps_unit_role_path(name, role, "out", out, sizeof(out)) != 0 ||
        (ofd = milkdetach_open_output(out, 1)) < 0)
    {
        fpsunit_say(quiet, stderr, "%s %s: cannot create its output file in %s: %s\n", name, rn,
                    base, strerror(errno));
        goto out;
    }
    fpsunit_me(me, sizeof(me));
    fpsunit_mark(ofd, "%sstart by %s", rn, me);
    if (milkos_on_tmpfs(base) == 0)
    {
        fpsunit_mark(
            ofd, "warning: %s is not on tmpfs: writing here can stall the loop during writeback",
            base);
    }
    if (fpsunit_env_build(&env, name, (int) role, &fps, base, me, ofd) != 0)
    {
        fpsunit_say(quiet, stderr, "%s %s: out of memory\n", name, rn);
        goto out;
    }
    off_t mark = lseek(ofd, 0, SEEK_END);

    snprintf(cmd, sizeof(cmd), "%s:%sstart", name, rn);
    char *argv[] = { fps.md->execfullpath, cmd, (char *) "--fg", NULL };
    memset(&spec, 0, sizeof(spec));
    spec.path    = fps.md->execfullpath;
    spec.argv    = argv;
    spec.envp    = env.v;
    spec.workdir = (fps.md->workdir[0] != '\0') ? fps.md->workdir : NULL;
    spec.out_fd  = ofd;
    if (geteuid() == 0 && sb.st_uid != 0)
    {
        // the worker runs as the owner of the FPS, whose file names what it runs (U3)
        int ng = milkos_user_ids(sb.st_uid, &spec.gid, groups,
                                 (int) (sizeof(groups) / sizeof(groups[0])));
        if (ng < 0)
        {
            fpsunit_say(quiet, stderr, "%s %s: the FPS owner (uid %d) is not a known user\n", name,
                        rn, (int) sb.st_uid);
            goto out;
        }
        spec.drop    = 1;
        spec.uid     = sb.st_uid;
        spec.groups  = groups;
        spec.ngroups = ng;
    }

    // the milk-cli session of the starter, shown with the unit (D37); run.info records it too
    const char *session = getenv("_MILK_SERVER_NAME");
    snprintf(fps.md->session, sizeof(fps.md->session), "%s", (session != NULL) ? session : "");

    double t0  = fpsunit_now();
    int    rfd = milkdetach_start(&spec);
    if (rfd < 0)
    {
        fpsunit_say(quiet, stderr, "%s %s: cannot start a process: %s\n", name, rn,
                    strerror(errno));
        goto out;
    }
    int w = milkdetach_wait_ready(rfd, (int) (timeout_s * 1000.0), &r);
    close(rfd);
    double   dt     = fpsunit_now() - t0;
    uint32_t cmdbit = (role == FPS_UNIT_RUN) ? FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN
                                             : FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF;
    if (w == MILKDETACH_OK || w == MILKDETACH_TIMEOUT)
    {
        fps.md->status |= cmdbit;
        fps.md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;
    }
    if (w == MILKDETACH_OK)
    {
        fpsunit_mark(ofd, "ready after %.2f s, pid %d", dt, (int) r.pid);
        fpsunit_say(quiet, stdout, "%s %s: ready after %.2f s, pid %d\n", name, rn, dt,
                    (int) r.pid);
        rc = 0;
    }
    else if (w == MILKDETACH_TIMEOUT)
    {
        fpsunit_mark(ofd, "not ready after %.1f s: still starting", dt);
        fpsunit_say(quiet, stderr,
                    "%s %s: not ready after %.1f s: pid %d is still starting (milk-fps-log %s)\n",
                    name, rn, dt, (int) r.pid, name);
        rc = FPS_UNIT_START_TIMEOUT;
    }
    else
    {
        if (w == MILKDETACH_FAILED)
        {
            fpsunit_mark(ofd, "start failed: %s: %s", r.status, strerror(r.err));
            fpsunit_say(quiet, stderr, "%s %s: %s: %s\n", name, rn, r.status, strerror(r.err));
        }
        else
        {
            fpsunit_mark(ofd, "start failed: the worker exited before its loop started");
            fpsunit_say(quiet, stderr, "%s %s: the worker exited before its loop started%s%s\n",
                        name, rn, (r.status[0] != '\0') ? ": " : "", r.status);
        }
        fpsunit_print_tail(out, mark, quiet);
    }

out:
    if (ofd >= 0)
    {
        close(ofd);
    }
    fpsunit_env_free(&env);
    fps_disconnect(&fps);
    fps_unit_unlock(lockfd);
    return rc;
}

/* ---- stop ---- */

/* Wake a loop that waits on its trigger stream's semaphore. Return: 1 posted */
static int fpsunit_wake(const PROCESSINFO *pi)
{
    char  path[1024];
    IMAGE img;
    int   ok;

    if ((pi->triggermode != PROCESSINFO_TRIGGERMODE_SEMAPHORE &&
         pi->triggermode != PROCESSINFO_TRIGGERMODE_SEMAPHORE_PROP_TIMEOUTS) ||
        pi->triggerstreamname[0] == '\0' || pi->triggersem < 0 ||
        ImageStreamIO_filename(path, sizeof(path), pi->triggerstreamname) !=
            IMAGESTREAMIO_SUCCESS ||
        access(path, F_OK) != 0)
    {
        return 0;
    }
    memset(&img, 0, sizeof(img));
    if (ImageStreamIO_openIm(&img, pi->triggerstreamname) != IMAGESTREAMIO_SUCCESS)
    {
        return 0;
    }
    ok = (pi->triggersem < img.md->sem);
    if (ok)
    {
        ImageStreamIO_sempost(&img, pi->triggersem);
    }
    ImageStreamIO_closeIm(&img);
    return ok;
}

/* U6 request: CTRLval=3 in the run's own processinfo, then a wake. Return: ms to wait, 0 if no processinfo */
static int fpsunit_request_run(const char *name, const FPS_UNIT_INFO *info)
{
    char         path[PATH_MAX + 256];
    int          fd;
    PROCESSINFO *pi;
    long         wait_ms = 500;

    if (fps_unit_proc_path(name, info, path, sizeof(path)) != 0 ||
        (pi = processinfo_shm_link(path, &fd)) == MAP_FAILED)
    {
        return 0;
    }
    pi->CTRLval = PROCESSINFO_CTRLVAL_EXIT;
    if (!fpsunit_wake(pi))
    {
        // it sees the request when its wait times out (or its delay ends)
        long t = (long) pi->triggertimeout.tv_sec * 1000L + pi->triggertimeout.tv_nsec / 1000000L;
        if (t <= 0)
        {
            t = 2000;
        }
        if (pi->triggermode == PROCESSINFO_TRIGGERMODE_DELAY)
        {
            long d = (long) pi->triggerdelay.tv_sec * 1000L + pi->triggerdelay.tv_nsec / 1000000L;
            t      = (d > t) ? d : t;
        }
        wait_ms += t;
    }
    processinfo_shm_close(pi, fd);
    return (int) wait_ms;
}

/* MILK_UNIT_STOP_TIMEOUTS="T2,T3" in seconds */
static void fpsunit_stop_timeouts(int *t2, int *t3)
{
    const char *v = getenv("MILK_UNIT_STOP_TIMEOUTS");
    char       *end;

    *t2 = FPSUNIT_STOP_T2_MS;
    *t3 = FPSUNIT_STOP_T3_MS;
    if (v == NULL)
    {
        return;
    }
    double a = strtod(v, &end);
    if (end != v && a >= 0.0)
    {
        *t2 = (int) (a * 1000.0);
    }
    if (*end == ',')
    {
        const char *s = end + 1;
        double      b = strtod(s, &end);
        if (end != s && b >= 0.0)
        {
            *t3 = (int) (b * 1000.0);
        }
    }
}

static int fpsunit_wait_gone(const MILKDETACH_ID *id, int ms)
{
    double end = fpsunit_now() + ms / 1000.0;

    while (milkdetach_id_alive(id))
    {
        struct timespec t = { 0, 10000000L };
        if (fpsunit_now() >= end)
        {
            return 0;
        }
        nanosleep(&t, NULL);
    }
    return 1;
}

int fps_unit_stop(const char *name, FPS_UNIT_ROLE role, int flags)
{
    int             quiet = (flags & FPS_UNIT_QUIET) != 0;
    const char     *rn    = fps_unit_role_name(role);
    FPS             fps   = { 0 };
    FPS_UNIT_STATUS st;
    FPS_UNIT_INFO   info;
    int             lockfd;
    int             rc = 0;
    int             t2;
    int             t3;
    int             step;
    char            unit[128];
    int             user;

    if (fpsunit_managed(name, role, unit, sizeof(unit), &user))
    {
        return fpsunit_delegate(name, role, 0, unit, user, 0.0, quiet);
    }
    if (fpsunit_lock(name, &lockfd, quiet) != 0)
    {
        return 1;
    }
    fps.SMfd = -1;
    if (fps_connect(name, &fps, FPSCONNECT_SIMPLE) == -1)
    {
        fpsunit_say(quiet, stderr, "%s: no such FPS\n", name);
        fps_unit_unlock(lockfd);
        return 1;
    }
    fps_unit_status_fps(&fps, role, &st, FPS_UNIT_LOCKED);
    if (!fps_unit_state_alive(st.state))
    {
        fpsunit_say(quiet, stdout, "%s %s: not running\n", name, rn);
        goto out;
    }
    int own = (fps_unit_info_read(name, role, &info) == 0 && info.id.pid == st.pid &&
               milkdetach_id_alive(&info.id));
    if (!own)
    {
        // no record of its own (an older version, a run inside milk-cli): asked, never signalled
        memset(&info, 0, sizeof(info));
        if (milkdetach_id_of(st.pid, &info.id) != 0)
        {
            goto out;
        }
        info.inprocess = 1;
    }
    else
    {
        fps_unit_info_append(name, role, "stop_requested=1");
    }

    double t0 = fpsunit_now();
    int    first_ms;
    if (role == FPS_UNIT_RUN)
    {
        first_ms = fpsunit_request_run(name, &info);
    }
    else
    {
        fps.md->signal &= ~FUNCTION_PARAMETER_STRUCT_SIGNAL_CONFRUN;
        first_ms = 500 + (int) (fps.md->confwaitus / 1000);
    }
    fpsunit_stop_timeouts(&t2, &t3);
    if (info.inprocess)
    {
        step = fpsunit_wait_gone(&info.id, first_ms + t2) ? 0 : -1;
    }
    else
    {
        step = milkdetach_stop(&info.id, info.pgid, first_ms, t2, t3);
    }
    double dt = fpsunit_now() - t0;
    if (step < 0)
    {
        fpsunit_say(
            quiet, stderr, "%s %s: pid %d is still running%s\n", name, rn, (int) info.id.pid,
            info.inprocess ? ": it was not started as a unit, so it was asked, not signalled"
                           : " after SIGKILL");
        rc = 1;
        goto out;
    }
    if (own)
    {
        char line[64];
        snprintf(line, sizeof(line), "stop=%s", milkdetach_stop_step_name(step));
        fps_unit_info_append(name, role, line);
        fpsunit_mark_file(name, role, "stop: %s after %.2f s", milkdetach_stop_step_name(step), dt);
    }
    fps_unit_status_fps(&fps, role, &st, FPS_UNIT_LOCKED); // clears what a killed process left
    fps.md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;
    fpsunit_say(quiet, stdout, "%s %s: stopped (%s, %.2f s)\n", name, rn,
                milkdetach_stop_step_name(step), dt);

out:
    fps_disconnect(&fps);
    fps_unit_unlock(lockfd);
    return rc;
}

/* ---- a service manager's start (milk-fps-run) ---- */

int fps_unit_exec_service(const char *name, FPS_UNIT_ROLE role)
{
    const char     *rn  = fps_unit_role_name(role);
    FPS             fps = { 0 };
    FPS_UNIT_STATUS st;
    FPSUNIT_ENV     env = { 0 };
    struct stat     sb;
    char            base[PATH_MAX];
    char            exe[PATH_MAX];
    char            workdir[PATH_MAX] = "";
    char            cmd[STRINGMAXLEN_FPS_NAME + 16];
    int             lockfd;
    int             rc = 1;

    if (name[0] == '_')
    {
        fprintf(stderr, "%s: a local FPS lives in one process: it cannot run as a service\n", name);
        return 1;
    }
    if (fpsunit_lock(name, &lockfd, 0) != 0)
    {
        return 1;
    }
    fps.SMfd = -1;
    if (fstat(lockfd, &sb) != 0 || fps_connect(name, &fps, FPSCONNECT_SIMPLE) == -1)
    {
        fprintf(stderr, "%s: no such FPS\n", name);
        fps_unit_unlock(lockfd);
        return 1;
    }
    if (sb.st_mode & (S_IWGRP | S_IWOTH))
    {
        fprintf(stderr, "%s: refused: the FPS file is writable by group or others (chmod 600 it)\n",
                name);
        goto out;
    }
    if (geteuid() != sb.st_uid)
    {
        struct passwd *pw = getpwuid(sb.st_uid);
        fprintf(stderr, "%s: the unit belongs to %s: run the service as that user (User=%s)\n",
                name, (pw != NULL) ? pw->pw_name : "another user",
                (pw != NULL) ? pw->pw_name : "...");
        goto out;
    }
    if (!fpsunit_is_fpsexec(fps.md->execfullpath))
    {
        fprintf(stderr,
                "%s: not a unit: '%s' created the FPS, and only an fpsexec binary runs a unit "
                "(<prefix>-fpsexec-<id> -n %s fpsinit)\n",
                name, fps.md->execfullpath, name);
        goto out;
    }
    fps_unit_status_fps(&fps, role, &st, FPS_UNIT_LOCKED);
    if (fps_unit_state_alive(st.state))
    {
        fprintf(stderr, "%s %s: already running, pid %d: stop it first (milk-fps-%sstop %s)\n",
                name, rn, (int) st.pid, rn, name);
        goto out;
    }
    // the service's environment (its EnvironmentFile= is the snapshot), then the overrides (U8)
    for (char **v = environ; *v != NULL; v++)
    {
        if (fpsunit_env_valid(*v) && strncmp(*v, "MILK_UNIT_WORKER=", 17) != 0 &&
            strncmp(*v, "MILK_UNIT_NOTIFY_FD=", 20) != 0 &&
            strncmp(*v, "MILK_UNIT_STARTER=", 18) != 0 && fpsunit_env_put(&env, strdup(*v)) != 0)
        {
            goto out;
        }
    }
    if (fps_unit_base_dir(base, sizeof(base)) != 0 ||
        fpsunit_env_overrides(&env, name, (int) role, &fps, base) != 0)
    {
        goto out;
    }
    snprintf(exe, sizeof(exe), "%s", fps.md->execfullpath);
    snprintf(workdir, sizeof(workdir), "%s", fps.md->workdir);
    rc = 0;

out:
    fps_disconnect(&fps);
    fps_unit_unlock(lockfd);
    if (rc != 0)
    {
        fpsunit_env_free(&env);
        return 1;
    }
    if (workdir[0] != '\0' && chdir(workdir) != 0)
    {
        fprintf(stderr, "%s %s: cannot enter the work directory %s: %s\n", name, rn, workdir,
                strerror(errno));
        fpsunit_env_free(&env);
        return 1;
    }
    snprintf(cmd, sizeof(cmd), "%s:%sstart", name, rn);
    char *argv[] = { exe, cmd, (char *) "--fg", NULL };
    fflush(stdout);
    execve(exe, argv, env.v);
    fprintf(stderr, "%s %s: cannot run %s: %s\n", name, rn, exe, strerror(errno));
    fpsunit_env_free(&env);
    return 1;
}

int fps_unit_release(const char *name)
{
    char path[PATH_MAX];

    if (fps_unit_path(name, "systemd", path, sizeof(path)) != 0)
    {
        return 1;
    }
    if (unlink(path) != 0)
    {
        if (errno == ENOENT)
        {
            printf("%s: no systemd service runs it\n", name);
            return 0;
        }
        fprintf(stderr, "%s: cannot remove %s: %s\n", name, path, strerror(errno));
        return 1;
    }
    printf("%s: released: milk starts and stops it again\n", name);
    return 0;
}

/* ---- front-ends: tools that do not block them, scripts ---- */

/* bytes, with an optional K, M or G suffix */
static long long fpsunit_parse_size(const char *s, long long dflt)
{
    char     *end;
    long long v;

    if (s == NULL || s[0] == '\0')
    {
        return dflt;
    }
    v = strtoll(s, &end, 10);
    switch (toupper((unsigned char) *end))
    {
    case 'K':
        v <<= 10;
        break;
    case 'M':
        v <<= 20;
        break;
    case 'G':
        v <<= 30;
        break;
    }
    return (end != s && v > 0) ? v : dflt;
}


pid_t fps_unit_spawn(const char *name, const char *op, char *const envp[], int out_fd, int err_fd)
{
    static const char *const ops[] = { "runstart", "runstop",  "confstart",
                                       "confstop", "confstep", "rm" };
    char                     prog[32];
    char                     tool[PATH_MAX];
    size_t                   i = 0;

    while (i < sizeof(ops) / sizeof(ops[0]) && strcmp(op, ops[i]) != 0)
    {
        i++;
    }
    // a name starting with '-' would be read as an option
    if (i == sizeof(ops) / sizeof(ops[0]) || name == NULL || name[0] == '\0' || name[0] == '-')
    {
        errno = EINVAL;
        return -1;
    }
    snprintf(prog, sizeof(prog), "milk-fps-%s", op);
    milkspawn_tool_path(prog, tool, sizeof(tool));
    if (envp == NULL)
    {
        envp = environ;
    }
    if (strcmp(op, "rm") == 0)
    {
        const char *argv[] = { tool, "-f", "-x", name, NULL }; // stop it, then remove that FPS only
        return milkspawn_start_env(argv, envp, out_fd, err_fd);
    }
    const char *argv[] = { tool, name, NULL };
    return milkspawn_start_env(argv, envp, out_fd, err_fd);
}

int fps_unit_job_start(FPS_UNIT_JOB *job, const char *name, const char *op)
{
    int p[2];
    int err;

    memset(job, 0, sizeof(*job));
    job->fd = -1;
    if (milkos_pipe_cloexec(p) != 0)
    {
        return -1;
    }
    job->pid = fps_unit_spawn(name, op, NULL, p[1], p[1]);
    err      = errno;
    close(p[1]);
    if (job->pid < 0)
    {
        close(p[0]);
        job->pid = 0;
        errno    = err;
        return -1;
    }
    fcntl(p[0], F_SETFL, O_NONBLOCK);
    job->fd = p[0];
    job->t0 = fpsunit_now();
    snprintf(job->name, sizeof(job->name), "%s", name);
    snprintf(job->op, sizeof(job->op), "%s", op);
    return 0;
}

/* Keep the first non-empty line of what the tool prints, without terminal sequences */
static void fpsunit_job_take(FPS_UNIT_JOB *job, const char *buf, ssize_t n)
{
    for (ssize_t i = 0; i < n && job->state != 3; i++)
    {
        unsigned char c = (unsigned char) buf[i];
        if (job->state == 1) // after ESC
        {
            job->state = (c == '[') ? 2 : 0;
        }
        else if (job->state == 2) // CSI: parameters up to a final byte
        {
            job->state = (c >= 0x40 && c <= 0x7e) ? 0 : 2;
        }
        else if (c == 0x1b)
        {
            job->state = 1;
        }
        else if (c == '\n')
        {
            job->state = (job->len > 0) ? 3 : 0;
        }
        else if (c >= 0x20 && job->len < sizeof(job->msg) - 1)
        {
            job->msg[job->len++] = (char) c;
        }
    }
    job->msg[job->len] = '\0';
}

int fps_unit_job_poll(FPS_UNIT_JOB *job)
{
    char    buf[1024];
    ssize_t n = 1;
    int     status;

    while (job->fd >= 0 && (n = read(job->fd, buf, sizeof(buf))) != 0)
    {
        if (n < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            if (errno == EAGAIN)
            {
                break;
            }
            n = 0;
            break;
        }
        fpsunit_job_take(job, buf, n);
    }
    if (job->fd >= 0 && n == 0)
    {
        close(job->fd);
        job->fd = -1;
    }
    if (job->pid <= 0)
    {
        return -1;
    }
    int r = milkspawn_poll(job->pid, &status);
    if (r == 0)
    {
        return -1;
    }
    if (job->fd >= 0) // what it printed last
    {
        fcntl(job->fd, F_SETFL, 0);
        while ((n = read(job->fd, buf, sizeof(buf))) > 0 || (n < 0 && errno == EINTR))
        {
            fpsunit_job_take(job, buf, (n > 0) ? n : 0);
        }
        close(job->fd);
        job->fd = -1;
    }
    job->pid = 0;
    if (strncmp(job->msg, "ERROR [", 7) == 0 && strstr(job->msg, "]: ") != NULL)
    {
        // PRINT_ERROR's source location means nothing in a front-end
        char *rest = strstr(job->msg, "]: ") + 3;
        memmove(job->msg, rest, strlen(rest) + 1);
    }
    return (r < 0) ? 1 : status;
}

void fps_unit_job_cancel(FPS_UNIT_JOB *job)
{
    if (job->fd >= 0)
    {
        close(job->fd);
        job->fd = -1;
    }
    job->pid = 0; // the tool finishes on its own, reparented to init when the caller exits
}

int fps_unit_script(const char *name, const char *path, int flags)
{
    int              quiet = (flags & FPS_UNIT_QUIET) != 0;
    FPS              fps   = { 0 };
    FPSUNIT_ENV      env   = { 0 };
    MILKDETACH_SPEC  spec;
    MILKDETACH_READY r;
    struct stat      sb;
    char             out[PATH_MAX];
    char             base[PATH_MAX];
    char             me[PATH_MAX + 32];
    char             arg[STRINGMAXLEN_DIRNAME + STRINGMAXLEN_FPS_NAME + 8];
    gid_t            groups[256];
    int              ofd = -1;
    int              rc  = 1;

    fps.SMfd = -1;
    if (path == NULL || path[0] == '\0')
    {
        fpsunit_say(quiet, stderr, "%s: no script to run\n", name);
        return 1;
    }
    if (fps_connect(name, &fps, FPSCONNECT_SIMPLE) == -1 || fstat(fps.SMfd, &sb) != 0)
    {
        fpsunit_say(quiet, stderr, "%s: no such FPS\n", name);
        goto out;
    }
    if (geteuid() != sb.st_uid && geteuid() != 0)
    {
        fpsunit_owner_hint(name, quiet);
        goto out;
    }
    if (sb.st_mode & (S_IWGRP | S_IWOTH))
    {
        fpsunit_say(quiet, stderr,
                    "%s: refused: the FPS file is writable by group or others (chmod 600 it)\n",
                    name);
        goto out;
    }
    if (fps_unit_dir_create(name) != 0 || fps_unit_base_dir(base, sizeof(base)) != 0 ||
        fps_unit_path(name, "exec.out", out, sizeof(out)) != 0)
    {
        fpsunit_say(quiet, stderr, "%s: cannot create the unit directory: %s\n", name,
                    strerror(errno));
        goto out;
    }
    struct stat ob;
    long long   cap = fpsunit_parse_size(getenv("MILK_UNIT_OUT_MAX"), FPSUNIT_OUT_MAX);
    if ((ofd = milkdetach_open_output(out, stat(out, &ob) == 0 && ob.st_size > cap)) < 0)
    {
        fpsunit_say(quiet, stderr, "%s: cannot open %s: %s\n", name, out, strerror(errno));
        goto out;
    }
    fpsunit_me(me, sizeof(me));
    fpsunit_mark(ofd, "exec %s by %s", path, me);
    if (fpsunit_env_build(&env, name, -1, &fps, base, me, ofd) != 0)
    {
        fpsunit_say(quiet, stderr, "%s: out of memory\n", name);
        goto out;
    }
    snprintf(arg, sizeof(arg), "%s/%s.fps", fps.md->datadir, name);
    char *argv[] = { (char *) path, arg, NULL };
    memset(&spec, 0, sizeof(spec));
    spec.path      = path;
    spec.argv      = argv;
    spec.envp      = env.v;
    spec.workdir   = (fps.md->workdir[0] != '\0') ? fps.md->workdir : NULL;
    spec.out_fd    = ofd;
    spec.no_notify = 1;
    if (geteuid() == 0 && sb.st_uid != 0)
    {
        int ng = milkos_user_ids(sb.st_uid, &spec.gid, groups,
                                 (int) (sizeof(groups) / sizeof(groups[0])));
        if (ng < 0)
        {
            fpsunit_say(quiet, stderr, "%s: the FPS owner (uid %d) is not a known user\n", name,
                        (int) sb.st_uid);
            goto out;
        }
        spec.drop    = 1;
        spec.uid     = sb.st_uid;
        spec.groups  = groups;
        spec.ngroups = ng;
    }
    int rfd = milkdetach_start(&spec);
    if (rfd < 0)
    {
        fpsunit_say(quiet, stderr, "%s: cannot start a process: %s\n", name, strerror(errno));
        goto out;
    }
    int w = milkdetach_wait_ready(rfd, 5000, &r);
    close(rfd);
    if (w == MILKDETACH_FAILED)
    {
        fpsunit_mark(ofd, "exec failed: %s: %s", r.status, strerror(r.err));
        fpsunit_say(quiet, stderr, "%s: cannot run %s: %s: %s\n", name, path, r.status,
                    strerror(r.err));
        goto out;
    }
    fpsunit_say(quiet, stdout, "%s: %s runs, pid %d (output in %s)\n", name, path, (int) r.pid,
                out);
    rc = 0;

out:
    if (ofd >= 0)
    {
        close(ofd);
    }
    fpsunit_env_free(&env);
    fps_disconnect(&fps);
    return rc;
}

/* ---- worker side ---- */

/* A role that systemd runs: recorded, so that the milk tools go through systemctl; output rotated as a
 * worker's when it goes to the role's file (StandardOutput=append:) */
static void fpsunit_service_begin(const char *name, FPS_UNIT_ROLE role, const char *unit, int user)
{
    struct stat a;
    struct stat b;

    if (fps_unit_systemd_write(name, role, unit, user) != 0)
    {
        fprintf(stderr, "%s %s: cannot record that %s runs it: %s\n", name,
                fps_unit_role_name(role), unit, strerror(errno));
    }
    if (fps_unit_role_path(name, role, "out", fpsunit_self.out, sizeof(fpsunit_self.out)) != 0 ||
        fstat(STDOUT_FILENO, &a) != 0 || stat(fpsunit_self.out, &b) != 0 || a.st_dev != b.st_dev ||
        a.st_ino != b.st_ino)
    {
        fpsunit_self.out[0] = '\0';
        return;
    }
    fpsunit_self.marks   = 1;
    fpsunit_self.out_max = fpsunit_parse_size(getenv("MILK_UNIT_OUT_MAX"), FPSUNIT_OUT_MAX);
    setvbuf(stdout, NULL, _IOLBF, 0);
    milkdetach_output_rotate_if(fpsunit_self.out,
                                1); // the last run's output to .1, as at a milk start
    fpsunit_mark(STDOUT_FILENO, "%sstart by systemd:%s", fps_unit_role_name(role), unit);
}

int fps_unit_role_begin(const char *name, FPS_UNIT_ROLE role)
{
    const char *w            = getenv("MILK_UNIT_WORKER");
    char        starter[256] = "foreground";
    char        unit[128];
    int         user    = 0;
    int         service = 0;
    int         lockfd  = -1;

    fpsunit_self.worker = (w != NULL && strcmp(w, "1") == 0);
    fpsunit_self.role   = role;
    // a service manager that waits for readiness is told what a starter is told (U17)
    fpsunit_self.svc_fd  = milknotify_open();
    fpsunit_self.wd_half = (double) milknotify_watchdog_usec() / 2e6;
    if (fpsunit_self.svc_fd >= 0 && !fpsunit_self.worker &&
        fps_unit_systemd_service(unit, sizeof(unit), &user))
    {
        service = 1;
        snprintf(starter, sizeof(starter), "systemd:%s", unit);
    }
    if (fpsunit_self.worker)
    {
        const char *s = getenv("MILK_UNIT_STARTER");
        snprintf(starter, sizeof(starter), "%s", (s != NULL) ? s : "");
        fpsunit_self.notify_fd = milkdetach_notify_fd("MILK_UNIT_NOTIFY_FD");
        fpsunit_self.out_max   = fpsunit_parse_size(getenv("MILK_UNIT_OUT_MAX"), FPSUNIT_OUT_MAX);
        // for this process only: the programs it runs are not workers
        unsetenv("MILK_UNIT_WORKER");
        unsetenv("MILK_UNIT_STARTER");
        // a file is fully buffered: the last lines before a crash would be lost
        setvbuf(stdout, NULL, _IOLBF, 0);
        if (fps_unit_role_path(name, role, "out", fpsunit_self.out, sizeof(fpsunit_self.out)) == 0)
        {
            fpsunit_self.marks     = 1;
            fps_unit_tick_deadline = time(NULL) + 1;
        }
    }
    else if ((lockfd = fps_unit_lock(name, 1)) >= 0)
    {
        // a foreground run: one process per role at a time (U9); the starter holds the lock for workers
        FPS             fps = { 0 };
        FPS_UNIT_STATUS st;
        fps.SMfd = -1;
        if (fps_connect(name, &fps, FPSCONNECT_SIMPLE) != -1)
        {
            fps_unit_status_fps(&fps, role, &st, FPS_UNIT_LOCKED);
            fps_disconnect(&fps);
            if (fps_unit_state_alive(st.state))
            {
                fprintf(stderr, "%s %s: already running, pid %d: stop it first (%sstop)\n", name,
                        fps_unit_role_name(role), (int) st.pid, fps_unit_role_name(role));
                fps_unit_unlock(lockfd);
                return 1;
            }
        }
    }
    else if (errno == ENOENT)
    {
        return 0; // no FPS: nothing to record, the run fails to connect
    }
    fpsunit_self.info_fd = fps_unit_info_create(name, role, starter, 0);
    fps_unit_unlock(lockfd);
    fpsunit_self.active = 1;
    if (service)
    {
        fpsunit_service_begin(name, role, unit, user);
    }
    if (fpsunit_self.svc_fd >= 0 || fpsunit_self.out[0] != '\0')
    {
        fps_unit_tick_deadline = time(NULL) + 1;
    }
    if (role == FPS_UNIT_RUN)
    {
        processinfo_set_crash_record(fpsunit_self.info_fd, fpsunit_self.marks ? STDERR_FILENO : -1);
    }
    return 0;
}

void fps_unit_role_end(int code)
{
    char line[64];

    if (!fpsunit_self.active)
    {
        return;
    }
    fpsunit_self.active    = 0;
    fps_unit_tick_deadline = 0;
    if (fpsunit_self.role == FPS_UNIT_RUN)
    {
        processinfo_set_crash_record(-1, -1);
    }
    if (fpsunit_self.info_fd >= 0)
    {
        int n = snprintf(line, sizeof(line), "exit=%d\n", code);
        (void) !write(fpsunit_self.info_fd, line, (size_t) n);
        close(fpsunit_self.info_fd);
        fpsunit_self.info_fd = -1;
    }
    if (fpsunit_self.marks)
    {
        fflush(stdout);
        fpsunit_mark(STDOUT_FILENO, "exit %d", code);
    }
    if (fpsunit_self.notify_fd >= 0)
    {
        if (!fpsunit_self.failed)
        {
            snprintf(line, sizeof(line), "STATUS=exited with status %d\n", code);
            milkdetach_notify(fpsunit_self.notify_fd, line);
        }
        close(fpsunit_self.notify_fd);
        fpsunit_self.notify_fd = -1;
    }
    if (fpsunit_self.svc_fd >= 0)
    {
        snprintf(line, sizeof(line), "STOPPING=1\nSTATUS=exited with status %d\n", code);
        milknotify_send(fpsunit_self.svc_fd, line);
        close(fpsunit_self.svc_fd);
        fpsunit_self.svc_fd = -1;
    }
}

void fps_unit_notify_connected(void)
{
    char msg[64];

    if (fpsunit_self.notify_fd >= 0)
    {
        snprintf(msg, sizeof(msg), "MAINPID=%d\nSTATUS=connected\n", (int) getpid());
        milkdetach_notify(fpsunit_self.notify_fd, msg);
    }
    milknotify_send(fpsunit_self.svc_fd, "STATUS=connected\n");
}

void fps_unit_notify_failed(int err, const char *fmt, ...)
{
    char    status[256];
    char    msg[320];
    va_list ap;

    if (fpsunit_self.notify_fd < 0 && fpsunit_self.svc_fd < 0)
    {
        return;
    }
    va_start(ap, fmt);
    vsnprintf(status, sizeof(status), fmt, ap);
    va_end(ap);
    status[strcspn(status, "\n")] = '\0';
    snprintf(msg, sizeof(msg), "ERRNO=%d\nSTATUS=%s\n", err, status);
    if (fpsunit_self.notify_fd >= 0)
    {
        milkdetach_notify(fpsunit_self.notify_fd, msg);
        fpsunit_self.failed = 1;
    }
    milknotify_send(fpsunit_self.svc_fd, msg);
}

void fps_unit_notify_ready(void)
{
    if (fpsunit_self.notify_fd >= 0)
    {
        milkdetach_notify(fpsunit_self.notify_fd, "READY=1\n");
        close(fpsunit_self.notify_fd);
        fpsunit_self.notify_fd = -1;
    }
    if (fpsunit_self.svc_fd >= 0 && !fpsunit_self.ready)
    {
        fpsunit_self.ready   = 1;
        fpsunit_self.wd_last = fpsunit_now();
        milknotify_send(fpsunit_self.svc_fd, (fpsunit_self.wd_half > 0.0)
                                                 ? "READY=1\nSTATUS=running\nWATCHDOG=1\n"
                                                 : "READY=1\nSTATUS=running\n");
    }
}

/* A service manager's tick: WATCHDOG=1 every half watchdog interval, STATUS= with the loop rate */
static void fpsunit_service_tick(long loopcnt)
{
    char   msg[96];
    double now = fpsunit_now();

    if (fpsunit_self.wd_half > 0.0 && now - fpsunit_self.wd_last >= fpsunit_self.wd_half)
    {
        milknotify_send(fpsunit_self.svc_fd, "WATCHDOG=1\n");
        fpsunit_self.wd_last = now;
    }
    if (loopcnt < 0 || !fpsunit_self.ready)
    {
        return;
    }
    if (fpsunit_self.st_last > 0.0 && now > fpsunit_self.st_last)
    {
        snprintf(msg, sizeof(msg), "STATUS=running, %.1f Hz\n",
                 (double) (loopcnt - fpsunit_self.st_loopcnt) / (now - fpsunit_self.st_last));
        milknotify_send(fpsunit_self.svc_fd, msg);
    }
    fpsunit_self.st_last    = now;
    fpsunit_self.st_loopcnt = loopcnt;
}

void fps_unit_tick_run(long loopcnt)
{
    fps_unit_tick_deadline = time(NULL) + 1;
    if (fpsunit_self.out[0] != '\0' &&
        milkdetach_output_rotate_if(fpsunit_self.out, fpsunit_self.out_max) == 1)
    {
        fpsunit_mark(STDOUT_FILENO, "earlier output in %s.1", fpsunit_self.out);
    }
    if (fpsunit_self.svc_fd >= 0)
    {
        fpsunit_service_tick(loopcnt);
    }
}
