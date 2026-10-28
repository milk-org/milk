// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_unit_state.c
 * @brief   Unit directory, identity records, unit lock and fps_unit_status() (fps_unit_state.h)
 */

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>

#include "fps.h"
#include "fps_unit_state.h"
#include "milk_os.h"
#include "milk_path.h"
#include "processinfo_procdirname.h"

static const char *const unitst_roles[2]  = { "run", "conf" };
static const char *const unitst_states[6] = { "stopped",  "starting", "running",
                                              "stopping", "exited",   "failed" };

const char *fps_unit_role_name(FPS_UNIT_ROLE role)
{
    return unitst_roles[role == FPS_UNIT_CONF];
}

const char *fps_unit_state_name(FPS_UNIT_STATE state)
{
    return ((int) state >= 0 && state <= FPS_UNIT_FAILED) ? unitst_states[state] : "unknown";
}

int fps_unit_base_dir(char *buf, size_t size)
{
    const char *dir = getenv("MILK_UNIT_DIR");
    char        shm[STRINGMAXLEN_SHMDIRNAME];

    if (dir == NULL || dir[0] == '\0')
    {
        function_parameter_struct_shmdirname(shm);
        dir = shm;
    }
    int n = snprintf(buf, size, "%s", dir);
    return (n < 0 || (size_t) n >= size) ? -1 : 0;
}

int fps_unit_path(const char *name, const char *file, char *buf, size_t size)
{
    char base[PATH_MAX];
    int  n;

    if (fps_unit_base_dir(base, sizeof(base)) != 0)
    {
        return -1;
    }
    n = (file != NULL) ? snprintf(buf, size, "%s/%s.fps.d/%s", base, name, file)
                       : snprintf(buf, size, "%s/%s.fps.d", base, name);
    if (n < 0 || (size_t) n >= size)
    {
        errno = ENAMETOOLONG;
        return -1;
    }
    return 0;
}

int fps_unit_role_path(const char   *name,
                       FPS_UNIT_ROLE role,
                       const char   *ext,
                       char         *buf,
                       size_t        size)
{
    char file[32];

    snprintf(file, sizeof(file), "%s.%s", fps_unit_role_name(role), ext);
    return fps_unit_path(name, file, buf, size);
}

int fps_unit_dir_create(const char *name)
{
    char dir[PATH_MAX];

    if (fps_unit_path(name, NULL, dir, sizeof(dir)) != 0)
    {
        return -1;
    }
    return milkpath_mkdir_p(dir, 0755);
}

int fps_unit_dir_remove(const char *name)
{
    char           dir[PATH_MAX];
    DIR           *d;
    struct dirent *e;

    if (fps_unit_path(name, NULL, dir, sizeof(dir)) != 0 || (d = opendir(dir)) == NULL)
    {
        return -1;
    }
    while ((e = readdir(d)) != NULL)
    {
        if (strcmp(e->d_name, ".") != 0 && strcmp(e->d_name, "..") != 0)
        {
            unlinkat(dirfd(d), e->d_name, 0);
        }
    }
    closedir(d);
    return rmdir(dir);
}

int fps_unit_info_read(const char *name, FPS_UNIT_ROLE role, FPS_UNIT_INFO *info)
{
    char  path[PATH_MAX];
    char  line[1024];
    FILE *f  = NULL;
    int   fd = -1;

    memset(info, 0, sizeof(*info));
    if (fps_unit_role_path(name, role, "info", path, sizeof(path)) != 0 ||
        (fd = open(path, O_RDONLY | O_CLOEXEC)) < 0 || (f = fdopen(fd, "r")) == NULL)
    {
        if (fd >= 0)
        {
            close(fd);
        }
        return -1;
    }
    while (fgets(line, sizeof(line), f) != NULL)
    {
        char *eq = strchr(line, '=');
        if (eq == NULL)
        {
            continue;
        }
        *eq                     = '\0';
        const char *key         = line;
        char       *val         = eq + 1;
        val[strcspn(val, "\n")] = '\0';

        if (strcmp(key, "pid") == 0)
        {
            info->id.pid = (pid_t) atol(val);
        }
        else if (strcmp(key, "pgid") == 0)
        {
            info->pgid = (pid_t) atol(val);
        }
        else if (strcmp(key, "start_ticks") == 0)
        {
            info->id.start_ticks = strtoull(val, NULL, 10);
        }
        else if (strcmp(key, "boot_id") == 0)
        {
            snprintf(info->id.boot_id, sizeof(info->id.boot_id), "%s", val);
        }
        else if (strcmp(key, "started") == 0)
        {
            info->started = strtod(val, NULL);
        }
        else if (strcmp(key, "session") == 0)
        {
            snprintf(info->session, sizeof(info->session), "%s", val);
        }
        else if (strcmp(key, "starter") == 0)
        {
            snprintf(info->starter, sizeof(info->starter), "%s", val);
        }
        else if (strcmp(key, "procdir") == 0)
        {
            snprintf(info->procdir, sizeof(info->procdir), "%s", val);
        }
        else if (strcmp(key, "inprocess") == 0)
        {
            info->inprocess = atoi(val);
        }
        else if (strcmp(key, "exit") == 0 && !info->has_exit)
        {
            info->has_exit  = 1;
            info->exit_code = atoi(val);
        }
        else if (strcmp(key, "signal") == 0 && !info->has_exit)
        {
            info->has_exit = 1;
            snprintf(info->signame, sizeof(info->signame), "%s", val);
        }
        else if (strcmp(key, "stop_requested") == 0)
        {
            info->stop_requested = 1;
        }
        else if (strcmp(key, "stop") == 0)
        {
            snprintf(info->stop, sizeof(info->stop), "%s", val);
        }
    }
    fclose(f);
    return (info->id.pid > 0) ? 0 : -1;
}

int fps_unit_info_create(const char *name, FPS_UNIT_ROLE role, const char *starter, int inprocess)
{
    char            path[PATH_MAX];
    char            exe[PATH_MAX] = "";
    char            procdir[STRINGMAXLEN_FULLFILENAME];
    char            rec[4096];
    MILKDETACH_ID   id;
    struct timespec now;
    const char     *session = getenv("_MILK_SERVER_NAME");

    if (fps_unit_dir_create(name) != 0 ||
        fps_unit_role_path(name, role, "info", path, sizeof(path)) != 0 ||
        milkdetach_id_of(getpid(), &id) != 0)
    {
        return -1;
    }
    milkos_self_exe(exe, sizeof(exe));
    processinfo_procdirname(procdir);
    clock_gettime(CLOCK_REALTIME, &now);
    int n = snprintf(rec, sizeof(rec),
                     "pid=%d\npgid=%d\nstart_ticks=%llu\nboot_id=%s\nstarted=%.3f\nsession=%s\n"
                     "starter=%s\nexe=%s\nprocdir=%s\n%s",
                     (int) id.pid, (int) getpgrp(), (unsigned long long) id.start_ticks, id.boot_id,
                     (double) now.tv_sec + 1e-9 * (double) now.tv_nsec,
                     (session != NULL) ? session : "", (starter != NULL) ? starter : "", exe,
                     procdir, inprocess ? "inprocess=1\n" : "");
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_APPEND | O_NOFOLLOW | O_CLOEXEC, 0644);
    if (fd < 0)
    {
        return -1;
    }
    if (n < 0 || (size_t) n >= sizeof(rec) || write(fd, rec, (size_t) n) != n)
    {
        close(fd);
        return -1;
    }
    return fd;
}

int fps_unit_info_append(const char *name, FPS_UNIT_ROLE role, const char *line)
{
    char path[PATH_MAX];
    char buf[512];
    int  fd;
    int  n = snprintf(buf, sizeof(buf), "%s\n", line);

    if (n < 0 || (size_t) n >= sizeof(buf) ||
        fps_unit_role_path(name, role, "info", path, sizeof(path)) != 0 ||
        (fd = open(path, O_WRONLY | O_APPEND | O_NOFOLLOW | O_CLOEXEC)) < 0)
    {
        return -1;
    }
    int rc = (write(fd, buf, (size_t) n) == n) ? 0 : -1;
    close(fd);
    return rc;
}

int fps_unit_systemd_read(const char *name, FPS_UNIT_ROLE role, char *unit, size_t size, int *user)
{
    char  path[PATH_MAX];
    char  line[256];
    char  key[8];
    int   found = 0;
    FILE *f;

    if (fps_unit_path(name, "systemd", path, sizeof(path)) != 0 || (f = fopen(path, "r")) == NULL)
    {
        return 0;
    }
    int klen = snprintf(key, sizeof(key), "%s=", fps_unit_role_name(role));
    while (!found && fgets(line, sizeof(line), f) != NULL)
    {
        char *scope;
        line[strcspn(line, "\n")] = '\0';
        if (strncmp(line, key, (size_t) klen) == 0 && (scope = strchr(line + klen, ' ')) != NULL)
        {
            *scope = '\0';
            snprintf(unit, size, "%s", line + klen);
            *user = (strcmp(scope + 1, "user") == 0);
            found = (unit[0] != '\0');
        }
    }
    fclose(f);
    return found;
}

int fps_unit_systemd_write(const char *name, FPS_UNIT_ROLE role, const char *unit, int user)
{
    char  path[PATH_MAX];
    char  tmp[PATH_MAX + 8];
    char  other[256] = "";
    char  line[256];
    char  key[8];
    FILE *f;

    if (fps_unit_path(name, "systemd", path, sizeof(path)) != 0)
    {
        return -1;
    }
    // keep the other role's line
    int klen = snprintf(key, sizeof(key), "%s=", fps_unit_role_name(role));
    if ((f = fopen(path, "r")) != NULL)
    {
        while (fgets(line, sizeof(line), f) != NULL)
        {
            if (strncmp(line, key, (size_t) klen) != 0 && strchr(line, '=') != NULL)
            {
                snprintf(other, sizeof(other), "%s", line);
            }
        }
        fclose(f);
    }
    snprintf(tmp, sizeof(tmp), "%s.new", path);
    int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW | O_CLOEXEC, 0644);
    if (fd < 0 || (f = fdopen(fd, "w")) == NULL)
    {
        if (fd >= 0)
        {
            close(fd);
        }
        return -1;
    }
    fprintf(f, "%s%s%s %s\n", other, key, unit, user ? "user" : "system");
    if (fclose(f) != 0 || rename(tmp, path) != 0)
    {
        unlink(tmp);
        return -1;
    }
    return 0;
}

off_t fps_unit_tail_offset(int fd, off_t end, long nlines)
{
    char  buf[4096];
    off_t pos = end;
    long  nl  = 0;

    if (nlines <= 0)
    {
        return 0;
    }
    while (pos > 0)
    {
        size_t  len = (pos > (off_t) sizeof(buf)) ? sizeof(buf) : (size_t) pos;
        ssize_t n   = pread(fd, buf, len, pos - (off_t) len);
        if (n <= 0)
        {
            return 0;
        }
        for (ssize_t i = n - 1; i >= 0; i--)
        {
            // the newline that ends the file does not start a line
            if (buf[i] == '\n' && pos - (off_t) n + i != end - 1 && ++nl == nlines)
            {
                return pos - (off_t) n + i + 1;
            }
        }
        pos -= n;
    }
    return 0;
}

int fps_unit_out_tail(const char *name, FPS_UNIT_ROLE role, long nlines, char *buf, size_t size)
{
    char    path[PATH_MAX];
    int     fd;
    ssize_t n = 0;

    if (size == 0 || fps_unit_role_path(name, role, "out", path, sizeof(path)) != 0 ||
        (fd = open(path, O_RDONLY | O_CLOEXEC)) < 0)
    {
        return -1;
    }
    off_t end  = lseek(fd, 0, SEEK_END);
    off_t from = fps_unit_tail_offset(fd, end, nlines);
    int   cut  = (end - from > (off_t) size - 1);
    if (cut)
    {
        from = end - (off_t) (size - 1);
    }
    if (end > from)
    {
        n = pread(fd, buf, (size_t) (end - from), from);
    }
    close(fd);
    n        = (n > 0) ? n : 0;
    buf[n]   = '\0';
    char *nl = cut ? memchr(buf, '\n', (size_t) n) : NULL;
    if (nl != NULL && nl + 1 < buf + n) // no partial first line
    {
        n -= (nl + 1 - buf);
        memmove(buf, nl + 1, (size_t) n + 1);
    }
    return (int) n;
}

int fps_unit_proc_path(const char *name, const FPS_UNIT_INFO *info, char *buf, size_t size)
{
    char procdir[STRINGMAXLEN_FULLFILENAME];

    if (info->procdir[0] != '\0')
    {
        snprintf(procdir, sizeof(procdir), "%s", info->procdir);
    }
    else
    {
        processinfo_procdirname(procdir);
    }
    int n = snprintf(buf, size, "%s/proc.%s.%06d.shm", procdir, name, (int) info->id.pid);
    return (n < 0 || (size_t) n >= size) ? -1 : 0;
}

int fps_unit_lock(const char *name, int wait)
{
    char shm[STRINGMAXLEN_SHMDIRNAME];
    char path[PATH_MAX];
    int  fd;

    function_parameter_struct_shmdirname(shm);
    if (snprintf(path, sizeof(path), "%s/%s.fps.shm", shm, name) >= (int) sizeof(path))
    {
        errno = ENAMETOOLONG;
        return -1;
    }
    // a descriptor of our own: closing any other descriptor of the file keeps an OFD lock
    if ((fd = open(path, O_RDWR | O_CLOEXEC)) < 0)
    {
        return -1;
    }
    if (milkos_lock_file(fd, wait) != 0)
    {
        int err = errno;
        close(fd);
        errno = err;
        return -1;
    }
    return fd;
}

void fps_unit_unlock(int fd)
{
    if (fd >= 0)
    {
        milkos_unlock_file(fd);
        close(fd);
    }
}

static int unitst_pid_alive(pid_t pid)
{
    return pid > 0 && (kill(pid, 0) == 0 || errno == EPERM);
}

/* Clear the bits a dead process of @role left in the FPS */
static void unitst_repair(FPS *fps, FPS_UNIT_ROLE role, pid_t dead)
{
    if (role == FPS_UNIT_RUN)
    {
        fps->md->status &=
            ~(FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN | FUNCTION_PARAMETER_STRUCT_STATUS_RUN |
              FUNCTION_PARAMETER_STRUCT_STATUS_RUNLOOP);
        if (fps->md->runpid == dead || !unitst_pid_alive(fps->md->runpid))
        {
            fps->md->runpid = 0;
        }
    }
    else
    {
        fps->md->status &=
            ~(FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF | FUNCTION_PARAMETER_STRUCT_STATUS_CONF);
        fps->md->signal &= ~FUNCTION_PARAMETER_STRUCT_SIGNAL_CONFRUN;
        if (fps->md->confpid == dead || !unitst_pid_alive(fps->md->confpid))
        {
            fps->md->confpid = 0;
        }
    }
}

int fps_unit_status_fps(FPS *fps, FPS_UNIT_ROLE role, FPS_UNIT_STATUS *st, int flags)
{
    FPS_UNIT_INFO info;
    char          path[PATH_MAX];
    struct stat   sb;
    int           has   = (fps_unit_info_read(fps->md->name, role, &info) == 0);
    int           loop  = (role == FPS_UNIT_RUN)
                              ? (fps->md->status & FUNCTION_PARAMETER_STRUCT_STATUS_RUNLOOP) != 0
                              : (fps->md->signal & FUNCTION_PARAMETER_STRUCT_SIGNAL_CONFRUN) != 0;
    pid_t         fpid  = (role == FPS_UNIT_RUN) ? fps->md->runpid : fps->md->confpid;
    int           alive = has && milkdetach_id_alive(&info.id);

    memset(st, 0, sizeof(*st));
    st->out_size = -1;
    if (has)
    {
        st->pid = info.id.pid;
        snprintf(st->session, sizeof(st->session), "%s", info.session);
    }
    if (!alive && loop && fpid > 0 && (!has || fpid != info.id.pid) && unitst_pid_alive(fpid))
    {
        // a run with no record of its own (an older version, a run inside milk-cli): pid and loop bit
        st->state      = FPS_UNIT_RUNNING;
        st->pid        = fpid;
        st->session[0] = '\0';
    }
    else if (alive)
    {
        struct timespec now;
        clock_gettime(CLOCK_REALTIME, &now);
        st->state  = info.stop_requested ? FPS_UNIT_STOPPING
                     : loop              ? FPS_UNIT_RUNNING
                                         : FPS_UNIT_STARTING;
        st->uptime = (info.started > 0)
                         ? (double) now.tv_sec + 1e-9 * (double) now.tv_nsec - info.started
                         : 0.0;
    }
    else
    {
        if (!has)
        {
            st->state = FPS_UNIT_STOPPED;
        }
        else if (info.has_exit && info.signame[0] != '\0')
        {
            st->state = FPS_UNIT_FAILED;
            snprintf(st->last, sizeof(st->last), "killed by %s", info.signame);
        }
        else if (info.has_exit)
        {
            st->state = (info.exit_code == 0) ? FPS_UNIT_EXITED : FPS_UNIT_FAILED;
            snprintf(st->last, sizeof(st->last), "exit %d", info.exit_code);
        }
        else if (info.stop[0] != '\0')
        {
            st->state = FPS_UNIT_EXITED;
            snprintf(st->last, sizeof(st->last), "stopped by %s", info.stop);
        }
        else
        {
            st->state = FPS_UNIT_FAILED;
            snprintf(st->last, sizeof(st->last), "no exit record (killed)");
        }

        int lockfd = -1;
        if ((flags & FPS_UNIT_LOCKED) ||
            ((flags & FPS_UNIT_REPAIR) && (lockfd = fps_unit_lock(fps->md->name, 0)) >= 0))
        {
            unitst_repair(fps, role, has ? info.id.pid : 0);
        }
        fps_unit_unlock(lockfd);
    }
    if (fps_unit_role_path(fps->md->name, role, "out", path, sizeof(path)) == 0 &&
        stat(path, &sb) == 0)
    {
        st->out_size = (long long) sb.st_size;
    }
    char unit[128];
    int  user;
    if (fps_unit_systemd_read(fps->md->name, role, unit, sizeof(unit), &user))
    {
        snprintf(st->manager, sizeof(st->manager), "%s (%s)", unit, user ? "user" : "system");
    }
    return 0;
}

int fps_unit_status(const char *name, FPS_UNIT_ROLE role, FPS_UNIT_STATUS *st)
{
    FPS fps = { 0 };

    fps.SMfd = -1;
    if (fps_connect(name, &fps, FPSCONNECT_SIMPLE) == -1)
    {
        memset(st, 0, sizeof(*st));
        st->out_size = -1;
        return -1;
    }
    fps_unit_status_fps(&fps, role, st, FPS_UNIT_REPAIR);
    fps_disconnect(&fps);
    return 0;
}
