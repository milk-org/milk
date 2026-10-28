// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_ipc_common.c
 * @brief Shared socket, environment, binary lookup, and daemon-spawning helpers.
 *
 * Role & Goal:
 *   Implements the common IPC utilities declared in `milk_ipc_common.h` so
 *   that `milk_client_conn.c`, `milk_client_ops.c`, `milk_srvlog_shm.c`, and
 *   `milk_srv_manager.c` share a single implementation of socket addressing,
 *   shell variable propagation, server ping/shutdown, and double-fork daemon
 *   launching.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#include "milk_detach.h"
#include "milk_ipc_common.h"
#include "milk_os.h"
#include "milk_proto.h"
#include "milk_srvlog.h"

extern char **environ;

/**
 * @brief Read a variable from the calling shell (via hooks) or `getenv()`.
 *
 * @param hooks Optional shell callback hooks (may be NULL)
 * @param name  Variable name
 * @return Value string, or NULL if unset
 */
const char *milk_ipc_get_var(const milk_client_hooks_t *hooks, const char *name)
{
    if (hooks != NULL && hooks->get_var != NULL)
    {
        const char *v = hooks->get_var(name, hooks->ctx);
        if (v != NULL)
        {
            return v;
        }
    }
    return getenv(name);
}

/**
 * get_shell_override_file - Compute the per-shell active socket override file path
 * @hooks: Optional shell callback hooks for reading `_MILK_LOCAL_SOCK`
 * @out:   Output buffer receiving the file path
 * @outsz: Size of @out in bytes
 *
 * Determines the owning shell PID from `_MILK_LOCAL_SOCK` (or `getpid()`/`getppid()`) and its start
 * time, and formats a path in `$TMPDIR` used to persist `@session use` socket switches across
 * subshells and standalone `milk-call` invocations.
 *
 * Return: void
 */
pid_t milk_ipc_shell_pid(const milk_client_hooks_t *hooks)
{
    const char *local_sock = milk_ipc_get_var(hooks, "_MILK_LOCAL_SOCK");
    if (local_sock != NULL && local_sock[0] != '\0')
    {
        const char *last_dash = strrchr(local_sock, '-');
        if (last_dash != NULL && isdigit((unsigned char) last_dash[1]))
        {
            return (pid_t) atoi(last_dash + 1);
        }
    }
    return (hooks != NULL && hooks->get_var != NULL) ? getpid() : getppid();
}

static void get_shell_override_file(const milk_client_hooks_t *hooks, char *out, size_t outsz)
{
    static pid_t start_pid = 0; // the shell's start, read once per process: with its pid it names
    static uint64_t start_ticks = 0; // the file, so a later shell with the same pid starts afresh
    int             shell_pid   = (int) milk_ipc_shell_pid(hooks);
    if (shell_pid != start_pid)
    {
        start_ticks = 0;
        milkos_proc_start_ticks(shell_pid, &start_ticks);
        start_pid = shell_pid;
    }
    const char *tmpdir = getenv("TMPDIR");
    if (tmpdir == NULL || tmpdir[0] == '\0')
    {
        tmpdir = "/tmp";
    }
    snprintf(out, outsz, "%s/.milk_active_sock_%u_%d_%llu", tmpdir, (unsigned) getuid(), shell_pid,
             (unsigned long long) start_ticks);
}

/**
 * milk_ipc_is_identifier - Check whether a string is a valid POSIX shell identifier
 * @name: Candidate variable or symbol name
 *
 * Verifies that @name starts with an ASCII letter or underscore and contains only
 * ASCII alphanumeric characters or underscores, preventing shell injection when
 * constructing variable assignment statements.
 *
 * Return: 1 if @name is a valid identifier, 0 otherwise
 */
int milk_ipc_is_identifier(const char *name)
{
    if (name == NULL || !(isalpha((unsigned char) name[0]) || name[0] == '_'))
    {
        return 0;
    }
    for (const char *c = name; *c; c++)
    {
        if (!(isalnum((unsigned char) *c) || *c == '_'))
        {
            return 0;
        }
    }
    return 1;
}

/**
 * settable_var_name() - 1 if @name is a plain identifier that does not steer the shell itself.
 */
static int settable_var_name(const char *name)
{
    static const char *const deny[] = {
        "PATH", "PROMPT_COMMAND", "IFS",      "ENV", "PS0", "PS1", "PS2", "PS3",
        "PS4",  "SHELLOPTS",      "BASHOPTS", NULL
    };
    static const char *const deny_pfx[] = { "BASH_", "LD_", NULL };

    if (!milk_ipc_is_identifier(name))
    {
        return 0;
    }
    for (int i = 0; deny[i] != NULL; i++)
    {
        if (strcmp(name, deny[i]) == 0)
        {
            return 0;
        }
    }
    for (int i = 0; deny_pfx[i] != NULL; i++)
    {
        if (strncmp(name, deny_pfx[i], strlen(deny_pfx[i])) == 0)
        {
            return 0;
        }
    }
    return 1;
}

/**
 * write_eval_assignment() - write `[export ]name='val'` to @fd, single-quote escaping @val.
 */
static void write_eval_assignment(int fd, const char *name, const char *val, int do_export)
{
    size_t cap  = strlen(name) + 4 * strlen(val) + 16;
    char  *line = (char *) malloc(cap);
    if (line == NULL)
    {
        return;
    }
    size_t n = (size_t) snprintf(line, cap, "%s%s='", do_export ? "export " : "", name);
    for (const char *c = val; *c; c++)
    {
        if (*c == '\'')
        {
            memcpy(line + n, "'\\''", 4);
            n += 4;
        }
        else
        {
            line[n++] = *c;
        }
    }
    memcpy(line + n, "'\n", 2);
    n += 2;
    if (write(fd, line, n) < 0)
    {
    }
    free(line);
}

/**
 * @brief Set a variable in the calling shell, process environment, and eval FD.
 *
 * @param hooks             Optional shell callback hooks (may be NULL)
 * @param name              Variable name
 * @param val               Value string
 * @param export_in_eval_fd If non-zero, prefix `export ` on `MILK_CALL_EVAL_FD`
 */
void milk_ipc_set_var(const milk_client_hooks_t *hooks,
                      const char                *name,
                      const char                *val,
                      int                        export_in_eval_fd)
{
    const char *safe_val = (val != NULL) ? val : "";
    if (!settable_var_name(name))
    {
        fprintf(stderr, "milk: refusing to set shell variable %s\n",
                milk_ipc_is_identifier(name) ? name : "(invalid name)");
        return;
    }
    if (strcmp(name, "MILKCLI_SOCK") == 0)
    {
        char ovfile[512];
        char def_local[128];
        int  shell_pid =
            (hooks != NULL && hooks->get_var != NULL) ? (int) getpid() : (int) getppid();
        const char *local_sock  = milk_ipc_get_var(hooks, "_MILK_LOCAL_SOCK");
        const char *engine_sock = milk_ipc_get_var(hooks, "_MILK_ENGINE_SOCK");
        snprintf(def_local, sizeof(def_local), "@milk-cli-%u-%d", (unsigned) getuid(), shell_pid);
        get_shell_override_file(hooks, ovfile, sizeof(ovfile));
        if (safe_val[0] == '\0' || strcmp(safe_val, def_local) == 0 ||
            (local_sock != NULL && strcmp(safe_val, local_sock) == 0) ||
            (engine_sock != NULL && strcmp(safe_val, engine_sock) == 0))
        {
            unlink(ovfile);
        }
        else
        {
            int fd = open(ovfile, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC | O_NOFOLLOW, 0600);
            if (fd >= 0)
            {
                if (write(fd, safe_val, strlen(safe_val)) < 0)
                {
                }
                close(fd);
            }
        }
    }

    if (hooks != NULL && hooks->set_var != NULL)
    {
        hooks->set_var(name, safe_val, hooks->ctx);
        if (!export_in_eval_fd)
        {
            return;
        }
    }

    const char *eval_fd_str = getenv("MILK_CALL_EVAL_FD");
    if (eval_fd_str != NULL && eval_fd_str[0] != '\0')
    {
        int efd = atoi(eval_fd_str);
        if (efd > 2)
        {
            write_eval_assignment(efd, name, safe_val, export_in_eval_fd);
        }
    }
    setenv(name, safe_val, 1);
}

/** sq_append() - append @s to the @len/@cap buffer *@buf, single-quoted for the shell when @quote. */
static void sq_append(char **buf, size_t *len, size_t *cap, const char *s, int quote)
{
    size_t need = *len + 4 * strlen(s) + 3;
    if (*buf == NULL || need > *cap)
    {
        size_t ncap = (*cap < 1024) ? 1024 : *cap;
        while (ncap < need)
        {
            ncap *= 2;
        }
        char *nb = (char *) realloc(*buf, ncap);
        if (nb == NULL)
        {
            return;
        }
        *buf = nb;
        *cap = ncap;
    }
    char *o = *buf + *len;
    if (quote)
    {
        *o++ = '\'';
    }
    for (; *s != '\0'; s++)
    {
        if (quote && *s == '\'')
        {
            memcpy(o, "'\\''", 4);
            o += 4;
        }
        else
        {
            *o++ = *s;
        }
    }
    if (quote)
    {
        *o++ = '\'';
    }
    *o   = '\0';
    *len = (size_t) (o - *buf);
}

/**
 * milk_ipc_bind_script() - shell code binding @text to array @name: words (split on blanks) for an indexed
 * array, `key<TAB|=>value` lines for an associative one (@assoc). Caller frees; NULL if @name is refused.
 */
char *milk_ipc_bind_script(const char *name, int assoc, char *text)
{
    if (!settable_var_name(name))
    {
        fprintf(stderr, "milk: refusing to set shell variable %s\n",
                milk_ipc_is_identifier(name) ? name : "(invalid name)");
        return NULL;
    }
    char  *out = NULL;
    size_t len = 0;
    size_t cap = 0;
    char  *sp  = NULL;
    sq_append(&out, &len, &cap, assoc ? "declare -gA " : "", 0);
    sq_append(&out, &len, &cap, name, 0);
    sq_append(&out, &len, &cap, assoc ? "=()" : "=(", 0);
    for (char *tok = strtok_r(text, assoc ? "\r\n" : " \t\r\n", &sp); tok != NULL;
         tok       = strtok_r(NULL, assoc ? "\r\n" : " \t\r\n", &sp))
    {
        if (!assoc)
        {
            sq_append(&out, &len, &cap, tok, 1);
            sq_append(&out, &len, &cap, " ", 0);
            continue;
        }
        char *sep = strchr(tok, '\t');
        if (sep == NULL)
        {
            sep = strchr(tok, '=');
        }
        if (sep == NULL || sep == tok)
        {
            continue;
        }
        *sep = '\0';
        sq_append(&out, &len, &cap, "; ", 0);
        sq_append(&out, &len, &cap, name, 0);
        sq_append(&out, &len, &cap, "[", 0);
        sq_append(&out, &len, &cap, tok, 1);
        sq_append(&out, &len, &cap, "]=", 0);
        sq_append(&out, &len, &cap, sep + 1, 1);
    }
    sq_append(&out, &len, &cap, assoc ? "" : ")", 0);
    return out;
}

/** milk_ipc_eval() - run shell code @script in the calling shell: hooks->eval, else write it to MILK_CALL_EVAL_FD. */
void milk_ipc_eval(const milk_client_hooks_t *hooks, char *script)
{
    if (hooks != NULL && hooks->eval != NULL)
    {
        hooks->eval(script, hooks->ctx);
        return;
    }
    const char *e   = getenv("MILK_CALL_EVAL_FD");
    int         efd = (e != NULL && e[0] != '\0') ? atoi(e) : -1;
    if (efd > 2 && write(efd, script, strlen(script)) >= 0 && write(efd, "\n", 1) < 0)
    {
    }
}

/**
 * @brief Synchronize `MILK_SHM_DIR` from shell hooks into `setenv()`.
 *
 * @param hooks Optional shell callback hooks (may be NULL)
 */
void milk_ipc_sync_shm_env(const milk_client_hooks_t *hooks)
{
    if (hooks != NULL && hooks->get_var != NULL)
    {
        const char *sd = hooks->get_var("MILK_SHM_DIR", hooks->ctx);
        if (sd != NULL && sd[0] != '\0')
        {
            setenv("MILK_SHM_DIR", sd, 1);
        }
        const char *pd = hooks->get_var("MILK_PROC_DIR", hooks->ctx);
        if (pd != NULL && pd[0] != '\0')
        {
            setenv("MILK_PROC_DIR", pd, 1);
        }
    }
}

/**
 * @brief Resolve the active `MILKCLI_SOCK` address for the current shell.
 *
 * @param hooks            Optional shell callback hooks (may be NULL)
 * @param out              Output socket path buffer
 * @param outsz            Size of @out in bytes
 * @param persist_if_unset If non-zero and `MILKCLI_SOCK` was unset, store the
 *                         computed default back into the shell environment
 */
void milk_ipc_resolve_sockpath(const milk_client_hooks_t *hooks,
                               char                      *out,
                               size_t                     outsz,
                               int                        persist_if_unset)
{
    char ovfile[512];
    get_shell_override_file(hooks, ovfile, sizeof(ovfile));
    int         fd = open(ovfile, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    struct stat st;
    if (fd >= 0 && (fstat(fd, &st) != 0 || st.st_uid != getuid() || (st.st_mode & 022) != 0))
    {
        close(fd); // not ours, or writable by others: never trust it
        fd = -1;
    }
    if (fd >= 0)
    {
        char    buf[256];
        ssize_t n = read(fd, buf, sizeof(buf) - 1);
        close(fd);
        if (n > 0)
        {
            buf[n] = '\0';
            while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r'))
            {
                buf[--n] = '\0';
            }
            if (buf[0] != '\0')
            {
                snprintf(out, outsz, "%s", buf);
                if (hooks != NULL && hooks->set_var != NULL)
                {
                    hooks->set_var("MILKCLI_SOCK", out, hooks->ctx);
                }
                setenv("MILKCLI_SOCK", out, 1);
                return;
            }
        }
    }

    const char *s = milk_ipc_get_var(hooks, "MILKCLI_SOCK");
    if (s != NULL && s[0] != '\0')
    {
        snprintf(out, outsz, "%s", s);
        return;
    }
    pid_t ref_pid = (hooks != NULL && hooks->get_var != NULL) ? getpid() : getppid();
    snprintf(out, outsz, "@milk-cli-%u-%d", (unsigned) getuid(), (int) ref_pid);
    if (persist_if_unset)
    {
        if (hooks != NULL && hooks->set_var != NULL)
        {
            hooks->set_var("MILKCLI_SOCK", out, hooks->ctx);
        }
        setenv("MILKCLI_SOCK", out, 1);
    }
}

/**
 * @brief Locate the `milk-cli` executable path using `MILKCLI_BIN` or `MILK_INSTALLDIR`.
 *
 * @param hooks Optional shell callback hooks (may be NULL)
 * @param out   Output path buffer
 * @param outsz Size of @out in bytes
 */
void milk_ipc_resolve_cli_bin(const milk_client_hooks_t *hooks, char *out, size_t outsz)
{
    const char *bin = milk_ipc_get_var(hooks, "MILKCLI_BIN");
    if (bin != NULL && bin[0] != '\0' && access(bin, X_OK) == 0)
    {
        snprintf(out, outsz, "%s", bin);
        return;
    }
    const char *inst = milk_ipc_get_var(hooks, "MILK_INSTALLDIR");
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
#ifdef INSTALLDIR
    {
        char cand[512];
        snprintf(cand, sizeof(cand), "%s/bin/milk-cli", INSTALLDIR);
        if (access(cand, X_OK) == 0)
        {
            snprintf(out, outsz, "%s", cand);
            return;
        }
    }
#endif
    snprintf(out, outsz, "milk-cli");
}

/**
 * @brief Check whether a string consists solely of ASCII decimal digits.
 *
 * @param s Input string
 * @return 1 if non-empty and all digits, 0 otherwise
 */
int milk_ipc_is_numeric_str(const char *s)
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

/**
 * milk_ipc_peer_is_self - Verify that a UNIX domain socket peer belongs to the current UID
 * @fd: Connected UNIX domain socket file descriptor
 *
 * Queries `SO_PEERCRED` on @fd and compares the peer's UID against `getuid()` to
 * prevent connecting to or accepting commands from an abstract UNIX socket claimed
 * by another local user.
 *
 * Return: 1 if the peer UID matches `getuid()`, 0 otherwise
 */
int milk_ipc_peer_is_self(int fd)
{
    struct ucred cred;
    socklen_t    clen = sizeof(cred);
    return getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &cred, &clen) == 0 && cred.uid == getuid();
}

/**
 * @brief Attempt a single non-retrying UNIX domain socket `connect()`.
 *
 * @param sockpath Abstract (`@name`) or filesystem UNIX socket path
 * @return Connected socket fd (>= 0), or -1 on failure
 */
int milk_ipc_connect_once(const char *sockpath)
{
    struct sockaddr_un addr;
    socklen_t          addrlen = 0;
    if (milk_sockaddr_init(sockpath, &addr, &addrlen) != 0)
    {
        return -1;
    }
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0)
    {
        return -1;
    }
    if (connect(fd, (struct sockaddr *) &addr, addrlen) != 0)
    {
        close(fd);
        return -1;
    }
    if (!milk_ipc_peer_is_self(fd))
    {
        // abstract socket names are global: another user may have claimed ours
        fprintf(stderr, "milk: ignoring server at %s: it runs as another user\n", sockpath);
        close(fd);
        return -1;
    }
    return fd;
}

/**
 * @brief Send `MILK_MSG_PING` to @sockpath and verify `MILK_MSG_DONE` within 2 s.
 *
 * @param sockpath Target UNIX socket path
 * @return 1 if server responded to ping, 0 otherwise
 */
int milk_ipc_ping_socket(const char *sockpath)
{
    int fd = milk_ipc_connect_once(sockpath);
    if (fd < 0)
    {
        return 0;
    }
    struct timeval tv = { 2, 0 }; // a stopped server accepts the connection but never answers
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
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

/**
 * @brief Send `MILK_MSG_SHUTDOWN` to @sockpath and wait at most 2 s for acknowledgment.
 *
 * @param sockpath Target UNIX socket path
 * @return 1 if shutdown frame was delivered, 0 if server was unreachable
 */
int milk_ipc_shutdown_socket(const char *sockpath)
{
    int fd = milk_ipc_connect_once(sockpath);
    if (fd < 0)
    {
        return 0;
    }
    struct timeval tv = { 2, 0 }; // a stuck server does not hold its stopper: signals come next
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    milk_proto_send(fd, MILK_MSG_SHUTDOWN, 0, NULL, 0);
    milk_frame_hdr_t hdr;
    uint8_t         *pl = NULL;
    (void) milk_proto_recv(fd, &hdr, &pl, NULL);
    free(pl);
    close(fd);
    return 1;
}

/* @bin as an absolute path: resolved if it has a '/', else looked up on PATH. Return: 0 or -1 */
static int spawn_exe_path(const char *bin, char *out, size_t size)
{
    char        cand[PATH_MAX];
    char        real[PATH_MAX];
    const char *path = getenv("PATH");

    if (strchr(bin, '/') != NULL)
    {
        return (realpath(bin, real) != NULL && snprintf(out, size, "%s", real) < (int) size) ? 0
                                                                                             : -1;
    }
    for (const char *d = (path != NULL) ? path : ""; *d != '\0';)
    {
        size_t len = strcspn(d, ":");
        if (len > 0 &&
            snprintf(cand, sizeof(cand), "%.*s/%s", (int) len, d, bin) < (int) sizeof(cand) &&
            access(cand, X_OK) == 0 && realpath(cand, real) != NULL)
        {
            return (snprintf(out, size, "%s", real) < (int) size) ? 0 : -1;
        }
        d += len + (d[len] == ':');
    }
    return -1;
}

/* The server's environment: ours (entries, not copies) without a service manager's variables (only
 * a server it starts itself answers it), and MILK_SRV_NAMED=1 for a named server. free() it */
static char **spawn_env(int is_named)
{
    size_t n = 0;
    size_t k = 0;

    while (environ[n] != NULL)
    {
        n++;
    }
    char **e = calloc(n + 2, sizeof(char *));
    if (e == NULL)
    {
        return NULL;
    }
    for (size_t i = 0; i < n; i++)
    {
        if (strncmp(environ[i], "MILK_SRV_NAMED=", 15) != 0 &&
            strncmp(environ[i], "NOTIFY_SOCKET=", 14) != 0 &&
            strncmp(environ[i], "WATCHDOG_", 9) != 0)
        {
            e[k++] = environ[i];
        }
    }
    if (is_named)
    {
        e[k++] = (char *) "MILK_SRV_NAMED=1";
    }
    e[k] = NULL;
    return e;
}

/* The end of a server's output file @path (at most 2 KiB), to stderr */
static void spawn_print_tail(const char *path)
{
    char buf[2048];
    int  fd = open(path, O_RDONLY | O_CLOEXEC);

    if (fd < 0)
    {
        return;
    }
    off_t   end  = lseek(fd, 0, SEEK_END);
    off_t   from = (end > (off_t) sizeof(buf) - 1) ? end - (off_t) (sizeof(buf) - 1) : 0;
    ssize_t n    = (end > from) ? pread(fd, buf, (size_t) (end - from), from) : 0;
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

int milk_ipc_spawn_daemon(const char *binpath, const char *sockpath, const milk_spawn_opts_t *opts)
{
    char             exe[PATH_MAX];
    char             out[512];
    char             opid[32];
    char             cpid[32];
    char             idle[16];
    char             ready[8];
    char             owner[8];
    const char      *argv[14];
    MILKDETACH_SPEC  spec;
    MILKDETACH_READY r;
    int              n = 0;

    if (spawn_exe_path(binpath, exe, sizeof(exe)) != 0)
    {
        fprintf(stderr, "milk: cannot start a server: %s not found\n", binpath);
        return -1;
    }
    char **envp = spawn_env(opts->is_named);
    if (envp == NULL)
    {
        return -1;
    }
    // its own output, for its crash messages (rotated to .1 at each start, as a unit's)
    int ofd = (milk_srvlog_out_path(sockpath, out, sizeof(out)) == 0)
                  ? milkdetach_open_output(out, 1)
                  : -1;
    snprintf(ready, sizeof(ready), "%d", MILKDETACH_NOTIFY_FD);
    argv[n++] = exe;
    argv[n++] = "-q";
    argv[n++] = "--server";
    argv[n++] = sockpath;
    argv[n++] = "--ready-fd";
    argv[n++] = ready;
    if (opts->owner_rfd > 2)
    {
        snprintf(owner, sizeof(owner), "%d", MILKDETACH_PASS_FD);
        argv[n++] = "--owner-fd";
        argv[n++] = owner;
    }
    else if (opts->owner_pid > 0)
    {
        snprintf(opid, sizeof(opid), "%ld", (long) opts->owner_pid);
        argv[n++] = "--owner-pid";
        argv[n++] = opid;
    }
    if (opts->creator_pid > 0)
    {
        snprintf(cpid, sizeof(cpid), "%ld", (long) opts->creator_pid);
        argv[n++] = "--creator";
        argv[n++] = cpid;
    }
    if (opts->idle_s > 0)
    {
        snprintf(idle, sizeof(idle), "%u", opts->idle_s);
        argv[n++] = "--idle-timeout";
        argv[n++] = idle;
    }
    argv[n] = NULL;
    memset(&spec, 0, sizeof(spec));
    spec.path    = exe;
    spec.argv    = (char *const *) argv;
    spec.envp    = envp;
    spec.out_fd  = ofd;
    spec.pass_fd = opts->owner_rfd;

    int rfd = milkdetach_start(&spec);
    int err = errno;
    if (ofd >= 0)
    {
        close(ofd);
    }
    free(envp);
    if (rfd < 0)
    {
        fprintf(stderr, "milk: cannot start a server for %s: %s\n", sockpath, strerror(err));
        return -1;
    }
    int w = milkdetach_wait_ready(rfd, 5000, &r);
    close(rfd);
    if (w == MILKDETACH_OK)
    {
        return 0;
    }
    if (w == MILKDETACH_FAILED)
    {
        fprintf(stderr, "milk: the server for %s did not start: %s: %s\n", sockpath, r.status,
                strerror(r.err));
    }
    else if (w == MILKDETACH_TIMEOUT)
    {
        fprintf(stderr, "milk: the server for %s is not ready after 5 s (pid %d)\n", sockpath,
                (int) r.pid);
    }
    else
    {
        fprintf(stderr, "milk: the server for %s exited before it listened%s%s\n", sockpath,
                (r.status[0] != '\0') ? ": " : "", r.status);
    }
    if (ofd >= 0)
    {
        spawn_print_tail(out);
    }
    return -1;
}
