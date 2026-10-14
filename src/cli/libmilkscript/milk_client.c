// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_client.c
 * @brief Client implementation for communicating with milk-cli --server
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include "milk_client.h"
#include "milk_proto.h"
#include "milk_server.h"

/* ============================================================
 *  Synced variable name registry & signal state
 * ============================================================ */

#define MAX_SYNCED_VARS 256
static char g_synced_vars[MAX_SYNCED_VARS][64];
static int  g_nb_synced_vars = 0;

static pid_t g_owner_shell_pid = 0;
static int   g_owner_pipe_wfd  = -1;

static volatile sig_atomic_t g_client_sigint   = 0;
static volatile sig_atomic_t g_client_sigwinch = 0;

static void client_sigint_handler(int sig)
{
    (void) sig;
    g_client_sigint = 1;
}

static void client_sigwinch_handler(int sig)
{
    (void) sig;
    g_client_sigwinch = 1;
}

static void record_synced_var(const char *name)
{
    if (name == NULL || name[0] == '\0' || strlen(name) >= 64)
    {
        return;
    }
    for (int i = 0; i < g_nb_synced_vars; i++)
    {
        if (strcmp(g_synced_vars[i], name) == 0)
        {
            return;
        }
    }
    if (g_nb_synced_vars < MAX_SYNCED_VARS)
    {
        snprintf(g_synced_vars[g_nb_synced_vars], 64, "%s", name);
        g_nb_synced_vars++;
    }
}

static const char *client_get_var(const milk_client_hooks_t *hooks, const char *name)
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

static void client_set_var(const milk_client_hooks_t *hooks, const char *name, const char *val)
{
    if (hooks != NULL && hooks->set_var != NULL)
    {
        hooks->set_var(name, val, hooks->ctx);
        return;
    }
    /* If running in milk-call with MILK_CALL_EVAL_FD set, emit assignment */
    const char *eval_fd_str = getenv("MILK_CALL_EVAL_FD");
    if (eval_fd_str != NULL && eval_fd_str[0] != '\0')
    {
        int efd = atoi(eval_fd_str);
        if (efd > 2)
        {
            char line[512];
            int  n = snprintf(line, sizeof(line), "%s='%s'\n", name, val);
            if (n > 0 && n < (int) sizeof(line))
            {
                if (write(efd, line, (size_t) n) < 0)
                {
                }
            }
        }
    }
    setenv(name, val, 1);
}

/* ============================================================
 *  Server Discovery & Lazy Auto-Start
 * ============================================================ */

static void resolve_sockpath(const milk_client_hooks_t *hooks, char *out, size_t outsz)
{
    const char *s = client_get_var(hooks, "MILKCLI_SOCK");
    if (s != NULL && s[0] != '\0')
    {
        snprintf(out, outsz, "%s", s);
        return;
    }
    pid_t ref_pid = (hooks != NULL && hooks->get_var != NULL) ? getpid() : getppid();
    snprintf(out, outsz, "@milk-cli-%u-%d", (unsigned) getuid(), (int) ref_pid);
    if (hooks != NULL && hooks->set_var != NULL)
    {
        hooks->set_var("MILKCLI_SOCK", out, hooks->ctx);
    }
    setenv("MILKCLI_SOCK", out, 1);
}

static void resolve_milk_cli_bin(const milk_client_hooks_t *hooks, char *out, size_t outsz)
{
    const char *bin = client_get_var(hooks, "MILKCLI_BIN");
    if (bin != NULL && bin[0] != '\0' && access(bin, X_OK) == 0)
    {
        snprintf(out, outsz, "%s", bin);
        return;
    }
    const char *inst = client_get_var(hooks, "MILK_INSTALLDIR");
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

static int connect_once(const char *sockpath)
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
    return fd;
}

static int autostart_server(const milk_client_hooks_t *hooks, const char *sockpath)
{
    int ready_pipe[2] = { -1, -1 };
    if (pipe(ready_pipe) != 0)
    {
        return -1;
    }
    fcntl(ready_pipe[0], F_SETFD, FD_CLOEXEC);

    int owner_rfd = -1;
    /* If running inside the main bash process via libmilkbash.so, bind server lifetime to shell */
    if (hooks != NULL && hooks->get_var != NULL)
    {
        if (g_owner_shell_pid == 0)
        {
            g_owner_shell_pid = getpid();
        }
        if (getpid() == g_owner_shell_pid)
        {
            int owner_pipe[2] = { -1, -1 };
            if (pipe(owner_pipe) == 0)
            {
                if (g_owner_pipe_wfd >= 0)
                {
                    close(g_owner_pipe_wfd);
                }
                /* Move write end to a high fd with CLOEXEC so child commands don't inherit it */
                g_owner_pipe_wfd = fcntl(owner_pipe[1], F_DUPFD_CLOEXEC, 100);
                if (g_owner_pipe_wfd < 0)
                {
                    g_owner_pipe_wfd = owner_pipe[1];
                    fcntl(g_owner_pipe_wfd, F_SETFD, FD_CLOEXEC);
                }
                else
                {
                    close(owner_pipe[1]);
                }
                owner_rfd = owner_pipe[0];
            }
        }
    }

    char binpath[512];
    resolve_milk_cli_bin(hooks, binpath, sizeof(binpath));

    pid_t pid = fork();
    if (pid < 0)
    {
        close(ready_pipe[0]);
        close(ready_pipe[1]);
        if (owner_rfd >= 0)
        {
            close(owner_rfd);
        }
        return -1;
    }

    if (pid == 0)
    {
        /* Double-fork so the server daemon is reparented to init/subreaper and
         * never interferes with the calling shell's wait() / job table */
        close(ready_pipe[0]);
        if (g_owner_pipe_wfd >= 0)
        {
            close(g_owner_pipe_wfd);
        }
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
            if (devnull > 2)
            {
                close(devnull);
            }
        }

        char rfd_str[32];
        snprintf(rfd_str, sizeof(rfd_str), "%d", ready_pipe[1]);

        if (owner_rfd >= 0)
        {
            char ofd_str[32];
            snprintf(ofd_str, sizeof(ofd_str), "%d", owner_rfd);
            execlp(binpath, binpath, "-q", "--server", sockpath, "--ready-fd", rfd_str,
                   "--owner-fd", ofd_str, (char *) NULL);
        }
        else
        {
            execlp(binpath, binpath, "-q", "--server", sockpath, "--ready-fd", rfd_str,
                   (char *) NULL);
        }
        _exit(127);
    }

    close(ready_pipe[1]);
    if (owner_rfd >= 0)
    {
        close(owner_rfd);
    }

    /* Reap intermediate child immediately */
    while (waitpid(pid, NULL, 0) < 0 && errno == EINTR)
    {
    }

    /* Wait up to 5 seconds for READY notification */
    struct pollfd pfd = { ready_pipe[0], POLLIN, 0 };
    int           pr  = poll(&pfd, 1, 5000);
    char          rbuf[16];
    ssize_t       rn = 0;
    if (pr > 0 && (pfd.revents & POLLIN))
    {
        rn = read(ready_pipe[0], rbuf, sizeof(rbuf));
    }
    close(ready_pipe[0]);
    return (rn > 0) ? 0 : -1;
}

static int milk_client_connect(const milk_client_hooks_t *hooks,
                               char                      *sockpath_out,
                               size_t                     sockpath_sz)
{
    resolve_sockpath(hooks, sockpath_out, sockpath_sz);
    int fd = connect_once(sockpath_out);
    if (fd >= 0)
    {
        return fd;
    }

    if (autostart_server(hooks, sockpath_out) == 0)
    {
        fd = connect_once(sockpath_out);
        if (fd >= 0)
        {
            return fd;
        }
    }
    fprintf(stderr, "ERROR: milk: unable to connect to or start milk-cli --server on '%s'\n",
            sockpath_out);
    return -1;
}

/* ============================================================
 *  Variable extraction helper for CALL / CALC / TEST payloads
 * ============================================================ */

typedef struct
{
    char names[256][64];
    char vals[256][128];
    int  count;
} packed_vars_t;

static int is_numeric_literal(const char *s)
{
    if (s == NULL || s[0] == '\0')
    {
        return 0;
    }
    char *end = NULL;
    errno     = 0;
    (void) strtod(s, &end);
    if (errno == 0 && end != s && *end == '\0')
    {
        return 1;
    }
    return 0;
}

static void try_add_packed_var(packed_vars_t             *pv,
                               const milk_client_hooks_t *hooks,
                               const char                *ident,
                               int                        allow_any)
{
    if (pv->count >= 256 || ident == NULL || ident[0] == '\0' || strlen(ident) >= 64)
    {
        return;
    }
    for (int i = 0; i < pv->count; i++)
    {
        if (strcmp(pv->names[i], ident) == 0)
        {
            return;
        }
    }
    const char *val = client_get_var(hooks, ident);
    if (val != NULL && (allow_any || is_numeric_literal(val)) && strlen(val) < 128)
    {
        snprintf(pv->names[pv->count], 64, "%s", ident);
        snprintf(pv->vals[pv->count], 128, "%s", val);
        pv->count++;
    }
}

static void collect_identifiers_from_str(packed_vars_t             *pv,
                                         const milk_client_hooks_t *hooks,
                                         const char                *s)
{
    if (s == NULL)
    {
        return;
    }
    const char *p = s;
    while (*p != '\0')
    {
        if (isalpha((unsigned char) *p) || *p == '_')
        {
            const char *start = p;
            p++;
            while (isalnum((unsigned char) *p) || *p == '_')
            {
                p++;
            }
            size_t len = (size_t) (p - start);
            if (len > 0 && len < 64)
            {
                char ident[64];
                memcpy(ident, start, len);
                ident[len] = '\0';
                try_add_packed_var(pv, hooks, ident, 1);
            }
        }
        else
        {
            p++;
        }
    }
}

static void collect_PreviouslySyncedVars(packed_vars_t *pv, const milk_client_hooks_t *hooks)
{
    for (int i = 0; i < g_nb_synced_vars; i++)
    {
        try_add_packed_var(pv, hooks, g_synced_vars[i], 0);
    }
}

/* ============================================================
 *  Response loop with Ctrl-C & SIGWINCH forwarding
 * ============================================================ */

static void send_oob_signal(const char *sockpath, uint8_t msg_type)
{
    int cfd = connect_once(sockpath);
    if (cfd >= 0)
    {
        milk_proto_send(cfd, msg_type, 0, NULL, 0);
        close(cfd);
    }
}

static int await_server_response(int                        fd,
                                 const char                *sockpath,
                                 const milk_client_hooks_t *hooks,
                                 char                      *value_buf,
                                 size_t                     value_sz,
                                 int                       *got_value)
{
    if (got_value != NULL)
    {
        *got_value = 0;
    }
    if (value_buf != NULL && value_sz > 0)
    {
        value_buf[0] = '\0';
    }

    struct sigaction sa_int, old_int;
    struct sigaction sa_winch, old_winch;
    memset(&sa_int, 0, sizeof(sa_int));
    memset(&sa_winch, 0, sizeof(sa_winch));
    sa_int.sa_handler   = client_sigint_handler;
    sa_winch.sa_handler = client_sigwinch_handler;
    sigemptyset(&sa_int.sa_mask);
    sigemptyset(&sa_winch.sa_mask);
    sa_int.sa_flags   = 0;
    sa_winch.sa_flags = 0;

    g_client_sigint   = 0;
    g_client_sigwinch = 0;
    int had_sigint    = 0;

    sigaction(SIGINT, &sa_int, &old_int);
    sigaction(SIGWINCH, &sa_winch, &old_winch);

    int32_t final_rc = 1;

    for (;;)
    {
        if (g_client_sigint)
        {
            g_client_sigint = 0;
            had_sigint      = 1;
            /* Send CANCEL directly on the active connection so watcher sees it */
            milk_proto_send(fd, MILK_MSG_CANCEL, 0, NULL, 0);
        }
        if (g_client_sigwinch)
        {
            g_client_sigwinch = 0;
            milk_proto_send(fd, MILK_MSG_WINCH, 0, NULL, 0);
            (void) sockpath;
        }

        struct pollfd pfd = { fd, POLLIN | POLLHUP | POLLERR, 0 };
        int           pr  = poll(&pfd, 1, -1);
        if (pr < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            break;
        }

        milk_frame_hdr_t hdr;
        uint8_t         *pl = NULL;
        if (milk_proto_recv(fd, &hdr, &pl, NULL) != 0)
        {
            break;
        }

        if (hdr.type == MILK_MSG_VALUE)
        {
            if (pl != NULL && hdr.len > 0)
            {
                if (value_buf != NULL && value_sz > 0)
                {
                    snprintf(value_buf, value_sz, "%s", (const char *) pl);
                    if (got_value != NULL)
                    {
                        *got_value = 1;
                    }
                }
                else
                {
                    fputs((const char *) pl, stdout);
                    size_t sl = strlen((const char *) pl);
                    if (sl == 0 || ((const char *) pl)[sl - 1] != '\n')
                    {
                        fputc('\n', stdout);
                    }
                    fflush(stdout);
                }
            }
        }
        else if (hdr.type == MILK_MSG_SETVAR)
        {
            if (pl != NULL && hdr.len > 0)
            {
                const char *vname = (const char *) pl;
                size_t      o1    = strlen(vname) + 1;
                const char *vval  = (o1 < hdr.len) ? (const char *) (pl + o1) : "";
                client_set_var(hooks, vname, vval);
                record_synced_var(vname);
            }
        }
        else if (hdr.type == MILK_MSG_DEFCMD)
        {
            if (pl != NULL && hdr.len > 0 && hooks != NULL && hooks->def_cmd != NULL)
            {
                hooks->def_cmd((const char *) pl, hooks->ctx);
            }
        }
        else if (hdr.type == MILK_MSG_DONE)
        {
            if (pl != NULL && hdr.len >= sizeof(int32_t))
            {
                memcpy(&final_rc, pl, sizeof(int32_t));
            }
            else
            {
                final_rc = 0;
            }
            free(pl);
            break;
        }

        free(pl);
    }

    sigaction(SIGINT, &old_int, NULL);
    sigaction(SIGWINCH, &old_winch, NULL);

    if (had_sigint)
    {
        if (hooks != NULL && hooks->on_sigint != NULL)
        {
            hooks->on_sigint(hooks->ctx);
        }
        if (final_rc == 0)
        {
            final_rc = 130;
        }
    }

    (void) send_oob_signal;
    return (int) final_rc;
}

/* ============================================================
 *  Public Client Operations
 * ============================================================ */

int milk_client_call(int argc, char **argv, int async_flag, const milk_client_hooks_t *hooks)
{
    if (argc <= 0 || argv == NULL || argv[0] == NULL)
    {
        return 0;
    }

    char sockpath[256];
    int  fd = milk_client_connect(hooks, sockpath, sizeof(sockpath));
    if (fd < 0)
    {
        return 1;
    }

    /* Collect scalar variables referenced in argv[] and previously synced vars */
    packed_vars_t pv;
    pv.count = 0;
    for (int i = 1; i < argc; i++)
    {
        collect_identifiers_from_str(&pv, hooks, argv[i]);
    }
    collect_PreviouslySyncedVars(&pv, hooks);

    /* Collect environment variables to forward */
    static const char *const env_keys[] = { "MILK_SHM_DIR",      "MILK_QUIET",    "NO_COLOR",
                                            "MILK_FLOAT_DIGITS", "MILKCLI_SIGIL", NULL };
    char                     env_kv[8][256];
    uint32_t                 nenv = 0;
    for (int i = 0; env_keys[i] != NULL && nenv < 8; i++)
    {
        const char *ev = client_get_var(hooks, env_keys[i]);
        if (ev != NULL)
        {
            snprintf(env_kv[nenv], sizeof(env_kv[nenv]), "%s=%s", env_keys[i], ev);
            nenv++;
        }
    }

    /* Compute payload size */
    size_t cap = sizeof(uint32_t) * 3 + 256;
    for (int i = 0; i < argc; i++)
    {
        cap += strlen(argv[i]) + 1;
    }
    for (uint32_t i = 0; i < nenv; i++)
    {
        cap += strlen(env_kv[i]) + 1;
    }
    for (int i = 0; i < pv.count; i++)
    {
        cap += strlen(pv.names[i]) + 1 + strlen(pv.vals[i]) + 1;
    }

    uint8_t *buf = (uint8_t *) malloc(cap);
    if (buf == NULL)
    {
        close(fd);
        return 1;
    }

    uint32_t off   = 0;
    uint32_t uargc = (uint32_t) argc;
    memcpy(buf + off, &uargc, sizeof(uargc));
    off += (uint32_t) sizeof(uargc);
    for (int i = 0; i < argc; i++)
    {
        size_t l = strlen(argv[i]) + 1;
        memcpy(buf + off, argv[i], l);
        off += (uint32_t) l;
    }

    memcpy(buf + off, &nenv, sizeof(nenv));
    off += (uint32_t) sizeof(nenv);
    for (uint32_t i = 0; i < nenv; i++)
    {
        size_t l = strlen(env_kv[i]) + 1;
        memcpy(buf + off, env_kv[i], l);
        off += (uint32_t) l;
    }

    uint32_t nvars = (uint32_t) pv.count;
    memcpy(buf + off, &nvars, sizeof(nvars));
    off += (uint32_t) sizeof(nvars);
    for (int i = 0; i < pv.count; i++)
    {
        size_t nl = strlen(pv.names[i]) + 1;
        size_t vl = strlen(pv.vals[i]) + 1;
        memcpy(buf + off, pv.names[i], nl);
        off += (uint32_t) nl;
        memcpy(buf + off, pv.vals[i], vl);
        off += (uint32_t) vl;
    }

    int cwd_fd = open(".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    int fds[4] = { STDIN_FILENO, STDOUT_FILENO, STDERR_FILENO, cwd_fd };

    uint8_t flags = async_flag ? MILK_FLAG_ASYNC : 0;
    int     sret =
        milk_proto_send_with_fds(fd, MILK_MSG_CALL, flags, buf, off, fds, (cwd_fd >= 0) ? 4 : 3);
    if (cwd_fd >= 0)
    {
        close(cwd_fd);
    }
    free(buf);

    if (sret != 0)
    {
        close(fd);
        return 1;
    }

    int rc = await_server_response(fd, sockpath, hooks, NULL, 0, NULL);
    close(fd);
    return rc;
}

static void sync_shm_env(const milk_client_hooks_t *hooks)
{
    if (hooks != NULL && hooks->get_var != NULL)
    {
        const char *sd = hooks->get_var("MILK_SHM_DIR", hooks->ctx);
        if (sd != NULL && sd[0] != '\0')
        {
            setenv("MILK_SHM_DIR", sd, 1);
        }
    }
}

int milk_client_get(const char *spec, const char *varname, const milk_client_hooks_t *hooks)
{
    if (spec == NULL || spec[0] == '\0')
    {
        return 1;
    }

    sync_shm_env(hooks);

    char val[1024];
    val[0] = '\0';

    /* Fast path: direct /dev/shm read (~2 us, zero IPC) */
    if (milk_entity_get(spec, val, sizeof(val)) == 0)
    {
        if (varname != NULL && varname[0] != '\0')
        {
            client_set_var(hooks, varname, val);
        }
        else
        {
            printf("%s\n", val);
            fflush(stdout);
        }
        return 0;
    }

    /* Fallback: query server (handles server-local heap images) */
    char sockpath[256];
    int  fd = milk_client_connect(hooks, sockpath, sizeof(sockpath));
    if (fd < 0)
    {
        return 1;
    }

    uint32_t slen = (uint32_t) (strlen(spec) + 1);
    if (milk_proto_send(fd, MILK_MSG_GET, 0, spec, slen) != 0)
    {
        close(fd);
        return 1;
    }

    int got_val = 0;
    int rc      = await_server_response(fd, sockpath, hooks, val, sizeof(val), &got_val);
    close(fd);

    if (rc == 0 && got_val)
    {
        if (varname != NULL && varname[0] != '\0')
        {
            client_set_var(hooks, varname, val);
        }
        else
        {
            printf("%s\n", val);
            fflush(stdout);
        }
    }
    return rc;
}

int milk_client_set(const char                *spec,
                    const char                *val,
                    const char                *comment,
                    const milk_client_hooks_t *hooks)
{
    if (spec == NULL || spec[0] == '\0')
    {
        return 1;
    }

    sync_shm_env(hooks);

    /* Fast path: if setting FPS param or existing /dev/shm stream keyword */
    if (milk_entity_set(spec, val, comment) == 0)
    {
        return 0;
    }

    /* Fallback to server (for server-local heap image keywords) */
    char sockpath[256];
    int  fd = milk_client_connect(hooks, sockpath, sizeof(sockpath));
    if (fd < 0)
    {
        return 1;
    }

    size_t   l1  = strlen(spec) + 1;
    size_t   l2  = (val != NULL) ? (strlen(val) + 1) : 0;
    size_t   l3  = (comment != NULL) ? (strlen(comment) + 1) : 0;
    uint32_t tot = (uint32_t) (l1 + l2 + l3);

    uint8_t *buf = (uint8_t *) malloc(tot);
    if (buf == NULL)
    {
        close(fd);
        return 1;
    }
    memcpy(buf, spec, l1);
    if (l2 > 0)
    {
        memcpy(buf + l1, val, l2);
    }
    if (l3 > 0)
    {
        memcpy(buf + l1 + l2, comment, l3);
    }

    int fds[3] = { STDIN_FILENO, STDOUT_FILENO, STDERR_FILENO };
    int sret   = milk_proto_send_with_fds(fd, MILK_MSG_SET, 0, buf, tot, fds, 3);
    free(buf);
    if (sret != 0)
    {
        close(fd);
        return 1;
    }

    int rc = await_server_response(fd, sockpath, hooks, NULL, 0, NULL);
    close(fd);
    return rc;
}

static int client_calc_or_test(const char                *expr,
                               int                        is_test,
                               const char                *varname,
                               const milk_client_hooks_t *hooks)
{
    if (expr == NULL || expr[0] == '\0')
    {
        return is_test ? 1 : 0;
    }

    char sockpath[256];
    int  fd = milk_client_connect(hooks, sockpath, sizeof(sockpath));
    if (fd < 0)
    {
        return 2;
    }

    packed_vars_t pv;
    pv.count = 0;
    collect_identifiers_from_str(&pv, hooks, expr);
    collect_PreviouslySyncedVars(&pv, hooks);

    size_t elen = strlen(expr) + 1;
    size_t cap  = elen + sizeof(uint32_t);
    for (int i = 0; i < pv.count; i++)
    {
        cap += strlen(pv.names[i]) + 1 + strlen(pv.vals[i]) + 1;
    }

    uint8_t *buf = (uint8_t *) malloc(cap);
    if (buf == NULL)
    {
        close(fd);
        return 2;
    }

    uint32_t off = 0;
    memcpy(buf + off, expr, elen);
    off += (uint32_t) elen;

    uint32_t nvars = (uint32_t) pv.count;
    memcpy(buf + off, &nvars, sizeof(nvars));
    off += (uint32_t) sizeof(nvars);
    for (int i = 0; i < pv.count; i++)
    {
        size_t nl = strlen(pv.names[i]) + 1;
        size_t vl = strlen(pv.vals[i]) + 1;
        memcpy(buf + off, pv.names[i], nl);
        off += (uint32_t) nl;
        memcpy(buf + off, pv.vals[i], vl);
        off += (uint32_t) vl;
    }

    /* If varname is given (e.g. ${| @calc ...; }), capture stdout via a pipe */
    int cap_pipe[2] = { -1, -1 };
    int out_fd      = STDOUT_FILENO;
    if (!is_test && varname != NULL && varname[0] != '\0')
    {
        if (pipe(cap_pipe) == 0)
        {
            out_fd = cap_pipe[1];
        }
    }

    int cwd_fd = open(".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    int fds[4] = { STDIN_FILENO, out_fd, STDERR_FILENO, cwd_fd };

    uint8_t mtype = is_test ? MILK_MSG_TEST : MILK_MSG_CALC;
    int     sret  = milk_proto_send_with_fds(fd, mtype, 0, buf, off, fds, (cwd_fd >= 0) ? 4 : 3);
    if (cwd_fd >= 0)
    {
        close(cwd_fd);
    }
    if (cap_pipe[1] >= 0)
    {
        close(cap_pipe[1]);
    }
    free(buf);

    if (sret != 0)
    {
        if (cap_pipe[0] >= 0)
        {
            close(cap_pipe[0]);
        }
        close(fd);
        return 2;
    }

    int rc = await_server_response(fd, sockpath, hooks, NULL, 0, NULL);
    close(fd);

    if (cap_pipe[0] >= 0)
    {
        char    cbuf[512];
        ssize_t cn = read(cap_pipe[0], cbuf, sizeof(cbuf) - 1);
        close(cap_pipe[0]);
        if (cn > 0)
        {
            cbuf[cn] = '\0';
            while (cn > 0 && (cbuf[cn - 1] == '\n' || cbuf[cn - 1] == '\r'))
            {
                cbuf[--cn] = '\0';
            }
            client_set_var(hooks, varname, cbuf);
        }
    }

    return rc;
}

int milk_client_calc(const char *expr, const char *varname, const milk_client_hooks_t *hooks)
{
    return client_calc_or_test(expr, 0, varname, hooks);
}

int milk_client_test(const char *expr, const milk_client_hooks_t *hooks)
{
    return client_calc_or_test(expr, 1, NULL, hooks);
}

int milk_client_exists(const char *spec, int check_running, const milk_client_hooks_t *hooks)
{
    if (spec == NULL || spec[0] == '\0')
    {
        return 1;
    }

    sync_shm_env(hooks);

    /* Fast path: check /dev/shm directly */
    if (milk_entity_exists(spec, check_running))
    {
        return 0;
    }

    /* If checking a stream/image that might be server-local in milk-cli --server */
    const char *s = spec;
    while (*s == '@')
    {
        s++;
    }
    if (!check_running &&
        (strncmp(s, "s.", 2) == 0 || strncmp(s, "stream.", 7) == 0 || strchr(s, '.') == NULL))
    {
        char sockpath[256];
        resolve_sockpath(hooks, sockpath, sizeof(sockpath));
        int fd = connect_once(sockpath);
        if (fd >= 0)
        {
            size_t   slen = strlen(spec) + 1;
            uint8_t *pl   = (uint8_t *) malloc(slen + 1);
            if (pl != NULL)
            {
                pl[0] = check_running ? 'R' : 'E';
                memcpy(pl + 1, spec, slen);
                if (milk_proto_send(fd, MILK_MSG_EXISTS, 0, pl, (uint32_t) (slen + 1)) == 0)
                {
                    int rc = await_server_response(fd, sockpath, hooks, NULL, 0, NULL);
                    free(pl);
                    close(fd);
                    return rc;
                }
                free(pl);
            }
            close(fd);
        }
    }

    return 1;
}

int milk_client_waitfor(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    const char *target  = NULL;
    const char *arg2    = "1";
    const char *timeout = "0";

    for (int i = 0; i < argc; i++)
    {
        if ((strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--timeout") == 0) && i + 1 < argc)
        {
            timeout = argv[++i];
        }
        else if (strncmp(argv[i], "--timeout=", 10) == 0)
        {
            timeout = argv[i] + 10;
        }
        else if (target == NULL)
        {
            target = argv[i];
        }
        else
        {
            arg2 = argv[i];
        }
    }

    if (target == NULL || target[0] == '\0')
    {
        fprintf(stderr,
                "Usage: @waitfor <stream|condition> [nframes|poll_sec] [--timeout <sec>]\n");
        return 1;
    }

    char sockpath[256];
    int  fd = milk_client_connect(hooks, sockpath, sizeof(sockpath));
    if (fd < 0)
    {
        return 1;
    }

    size_t   l1  = strlen(target) + 1;
    size_t   l2  = strlen(arg2) + 1;
    size_t   l3  = strlen(timeout) + 1;
    uint32_t tot = (uint32_t) (l1 + l2 + l3);

    uint8_t *buf = (uint8_t *) malloc(tot);
    if (buf == NULL)
    {
        close(fd);
        return 1;
    }
    memcpy(buf, target, l1);
    memcpy(buf + l1, arg2, l2);
    memcpy(buf + l1 + l2, timeout, l3);

    int sret = milk_proto_send(fd, MILK_MSG_WAITFOR, 0, buf, tot);
    free(buf);
    if (sret != 0)
    {
        close(fd);
        return 1;
    }

    int rc = await_server_response(fd, sockpath, hooks, NULL, 0, NULL);
    close(fd);
    return rc;
}

int milk_client_grab(int argc, char **argv, const milk_client_hooks_t *hooks)
{
    const char *sname   = NULL;
    const char *nframes = NULL;
    const char *outname = NULL;
    const char *timeout = "0";

    for (int i = 0; i < argc; i++)
    {
        if ((strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--timeout") == 0) && i + 1 < argc)
        {
            timeout = argv[++i];
        }
        else if (strncmp(argv[i], "--timeout=", 10) == 0)
        {
            timeout = argv[i] + 10;
        }
        else if (sname == NULL)
        {
            sname = argv[i];
        }
        else if (nframes == NULL)
        {
            nframes = argv[i];
        }
        else if (outname == NULL)
        {
            outname = argv[i];
        }
    }

    if (sname == NULL || nframes == NULL || outname == NULL)
    {
        fprintf(stderr, "Usage: @grab <stream> <nframes> <out_cube> [--timeout <sec>]\n");
        return 1;
    }

    char sockpath[256];
    int  fd = milk_client_connect(hooks, sockpath, sizeof(sockpath));
    if (fd < 0)
    {
        return 1;
    }

    size_t   l1  = strlen(sname) + 1;
    size_t   l2  = strlen(nframes) + 1;
    size_t   l3  = strlen(outname) + 1;
    size_t   l4  = strlen(timeout) + 1;
    uint32_t tot = (uint32_t) (l1 + l2 + l3 + l4);

    uint8_t *buf = (uint8_t *) malloc(tot);
    if (buf == NULL)
    {
        close(fd);
        return 1;
    }
    memcpy(buf, sname, l1);
    memcpy(buf + l1, nframes, l2);
    memcpy(buf + l1 + l2, outname, l3);
    memcpy(buf + l1 + l2 + l3, timeout, l4);

    int sret = milk_proto_send(fd, MILK_MSG_GRAB, 0, buf, tot);
    free(buf);
    if (sret != 0)
    {
        close(fd);
        return 1;
    }

    int rc = await_server_response(fd, sockpath, hooks, NULL, 0, NULL);
    close(fd);
    return rc;
}

int milk_client_complete(const char                *prefix,
                         const char                *comp_type,
                         const milk_client_hooks_t *hooks)
{
    if (prefix == NULL)
    {
        prefix = "";
    }
    if (comp_type == NULL)
    {
        comp_type = "arg";
    }

    char sockpath[256];
    int  fd = milk_client_connect(hooks, sockpath, sizeof(sockpath));
    if (fd < 0)
    {
        return 1;
    }

    size_t   l1  = strlen(prefix) + 1;
    size_t   l2  = strlen(comp_type) + 1;
    uint32_t tot = (uint32_t) (l1 + l2);

    uint8_t *buf = (uint8_t *) malloc(tot);
    if (buf == NULL)
    {
        close(fd);
        return 1;
    }
    memcpy(buf, prefix, l1);
    memcpy(buf + l1, comp_type, l2);

    int sret = milk_proto_send(fd, MILK_MSG_COMPLETE, 0, buf, tot);
    free(buf);
    if (sret != 0)
    {
        close(fd);
        return 1;
    }

    int rc = await_server_response(fd, sockpath, hooks, NULL, 0, NULL);
    close(fd);
    return rc;
}

int milk_client_shutdown(const milk_client_hooks_t *hooks)
{
    char sockpath[256];
    resolve_sockpath(hooks, sockpath, sizeof(sockpath));
    int fd = connect_once(sockpath);
    if (fd < 0)
    {
        if (g_owner_pipe_wfd >= 0)
        {
            close(g_owner_pipe_wfd);
            g_owner_pipe_wfd = -1;
        }
        return 0;
    }

    milk_proto_send(fd, MILK_MSG_SHUTDOWN, 0, NULL, 0);
    int rc = await_server_response(fd, sockpath, hooks, NULL, 0, NULL);
    close(fd);

    if (g_owner_pipe_wfd >= 0)
    {
        close(g_owner_pipe_wfd);
        g_owner_pipe_wfd = -1;
    }
    return rc;
}
