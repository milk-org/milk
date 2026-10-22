// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_server_dispatch.c
 * @brief Command resolution, token population, and SET/CALC/TEST/CALL execution.
 *
 * Role & Goal:
 *   1. Resolves canonical (`ns.cmd`) and unambiguous short (`cmd`) command names
 *      in `data.cmd[]` and populates `data.cmdargtoken[]` directly from `argv[]`
 *      without shell re-splitting or `wordexp()`.
 *   2. Executes `MILK_MSG_SET` (`@set`) and `MILK_MSG_CALL` payloads, managing
 *      variable synchronization, fault-isolated command execution (`sigsetjmp`),
 *      dynamic module command notification (`MILK_MSG_DEFCMD`), and concurrent
 *      pipeline/Ctrl-C monitoring. Calculator (`@calc`/`@test`), `@assert`, and
 *      `@pipe` evaluation live in `milk_server_eval.c`.
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <ctype.h>
#include <poll.h>
#include <pthread.h>
#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <unistd.h>

#include "CLIcore.h"
#include "CLIcore_help.h"
#include "CLIcore_memory.h"
#include "CLIcore_script.h"
#include "CLIcore_setSHMdir.h"
#include "CLIcore_signals.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "cli_calc_parser.h"
#include "milk_shm_prop.h"
#include "milk_ipc_common.h"
#include "milk_entity_ns.h"
#include "milk_proto.h"
#include "milk_server_internal.h"
#include "milk_catalog.h"
#include "milk_cmd_args.h"
#include "milk_cmd_index.h"
#include "milk_srvlog.h"

static volatile sig_atomic_t g_call_interrupted = 0; /* a CALL was aborted by SIGINT */

/**
 * milk_server_take_call_interrupt - Consume and clear the sticky SIGINT-aborted CALL flag
 *
 * Allows the server event loop to detect whether the just-finished `MILK_MSG_CALL`
 * was aborted by `SIGINT` (`siglongjmp`) and report exit status 130 to the client.
 *
 * Return: 1 if the preceding CALL was interrupted by SIGINT, 0 otherwise.
 */
int milk_server_take_call_interrupt(void)
{
    int r              = g_call_interrupted;
    g_call_interrupted = 0;
    return r;
}

extern void handle_did_you_mean(const char *input_cmd);

typedef struct
{
    int          client_fd;
    int          listen_fd;
    int          wake_fd;
    pthread_t    main_tid;
    volatile int active;
} call_watcher_ctx_t;

/**
 * watcher_take_client() - serve a connection that arrives while a CALL runs; 1 if a thread took @cfd.
 */
static int watcher_take_client(const call_watcher_ctx_t *ctx, int cfd)
{
    struct timeval to = { 2, 0 };
    setsockopt(cfd, SOL_SOCKET, SO_RCVTIMEO, &to, sizeof(to));
    milk_frame_hdr_t ph;
    if (recv(cfd, &ph, sizeof(ph), MSG_PEEK | MSG_WAITALL) != (ssize_t) sizeof(ph))
    {
        return 0;
    }
    if (ph.type == MILK_MSG_SHUTDOWN)
    {
        milk_frame_hdr_t hdr;
        uint8_t         *pl = NULL;
        if (milk_proto_recv(cfd, &hdr, &pl, NULL) == 0)
        {
            int32_t rc         = 0;
            g_milk_server_stop = 1; /* the main loop stops once the CALL returns */
            milk_proto_send(cfd, MILK_MSG_DONE, 0, &rc, sizeof(rc));
        }
        free(pl);
        return 0;
    }
    if ((ph.type == MILK_MSG_WAITFOR || ph.type == MILK_MSG_GRAB) &&
        milk_server_spawn_stream_worker(cfd) == 0)
    {
        return 1;
    }
    pid_t pid = fork();
    if (pid > 0)
    {
        g_srv_children = 1;
    }
    if (pid == 0)
    {
        sigset_t none; /* this thread blocks all signals; the child must not */
        sigemptyset(&none);
        pthread_sigmask(SIG_SETMASK, &none, NULL);
        close(ctx->listen_fd);
        int dummy_shutdown = 0;
        milk_server_handle_client(cfd, -1, &dummy_shutdown);
        close(cfd);
        _exit(0);
    }
    return 0;
}

/**
 * @brief Background watcher thread active during `MILK_MSG_CALL` execution.
 *
 * Monitors @ctx->client_fd for `MILK_MSG_CANCEL` / `MILK_MSG_WINCH` / hangup
 * (forwarding `SIGINT` / `SIGWINCH` to the main server thread) and accepts
 * concurrent connections on @ctx->listen_fd: SHUTDOWN is recorded for the main loop,
 * WAITFOR/GRAB get a worker thread, anything else (e.g. a pipeline stage that needs
 * its own stdio) runs in a forked child.
 */
static void *call_watcher_thread(void *arg)
{
    call_watcher_ctx_t *ctx = (call_watcher_ctx_t *) arg;
    while (ctx->active)
    {
        struct pollfd pfds[3];
        pfds[0].fd      = ctx->client_fd;
        pfds[0].events  = POLLIN | POLLHUP | POLLERR;
        pfds[0].revents = 0;
        pfds[1].fd      = ctx->listen_fd;
        pfds[1].events  = POLLIN;
        pfds[1].revents = 0;
        pfds[2].fd      = ctx->wake_fd;
        pfds[2].events  = POLLIN;
        pfds[2].revents = 0;

        int nfds = (ctx->wake_fd >= 0) ? 3 : 2;
        int tms  = (ctx->wake_fd >= 0) ? 200 : 20;
        int pr   = poll(pfds, (nfds_t) nfds, tms);
        if (nfds == 3 && (pfds[2].revents & POLLIN))
        {
            uint64_t v;
            ssize_t rret = read(ctx->wake_fd, &v, sizeof(v)); /* the thread is reused: consume it */
            (void) rret;
            continue;
        }
        if (pr <= 0 || !ctx->active)
        {
            continue;
        }

        if (pfds[0].revents & (POLLIN | POLLHUP | POLLERR))
        {
            if (pfds[0].revents & POLLIN)
            {
                milk_frame_hdr_t hdr;
                uint8_t         *pl = NULL;
                if (milk_proto_recv(ctx->client_fd, &hdr, &pl, NULL) == 0)
                {
                    if (hdr.type == MILK_MSG_CANCEL)
                    {
                        dcsigINT = 1;
                        pthread_kill(ctx->main_tid, SIGINT);
                    }
                    else if (hdr.type == MILK_MSG_WINCH)
                    {
                        pthread_kill(ctx->main_tid, SIGWINCH);
                    }
                    free(pl);
                }
                else
                {
                    dcsigINT = 1;
                    pthread_kill(ctx->main_tid, SIGINT);
                    break;
                }
            }
            else
            {
                dcsigINT = 1;
                pthread_kill(ctx->main_tid, SIGINT);
                break;
            }
        }

        if ((pfds[1].revents & POLLIN) && ctx->active)
        {
            int cfd = accept4(ctx->listen_fd, NULL, NULL, SOCK_CLOEXEC);
            if (cfd >= 0)
            {
                if (milk_ipc_peer_is_self(cfd) && watcher_take_client(ctx, cfd))
                {
                    continue; /* a worker thread owns cfd now */
                }
                close(cfd);
            }
        }
    }
    return NULL;
}

/* One watcher thread serves every CALL in turn (no thread creation per CALL). */
static call_watcher_ctx_t g_wctx    = { -1, -1, -1, 0, 0 };
static pthread_mutex_t    g_wmu     = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t     g_wcv     = PTHREAD_COND_INITIALIZER;
static unsigned long      g_wgen    = 0; /* CALLs handed to the watcher */
static int                g_wbusy   = 0; /* the watcher is watching a CALL */
static int                g_wthread = 0; /* the watcher thread runs in this process */

/**
 * watcher_main - Persistent background thread parking between `MILK_MSG_CALL` executions
 * @arg: Unused thread argument
 *
 * Reuses a single detached thread across all CALL requests to avoid per-command
 * `pthread_create()` overhead.
 *
 * Return: Always NULL (never returns).
 */
static void *watcher_main(void *arg)
{
    (void) arg;
    unsigned long served = 0;
    pthread_mutex_lock(&g_wmu);
    for (;;)
    {
        while (!g_wctx.active || g_wgen == served)
        {
            pthread_cond_wait(&g_wcv, &g_wmu);
        }
        served  = g_wgen;
        g_wbusy = 1;
        pthread_mutex_unlock(&g_wmu);
        call_watcher_thread(&g_wctx); /* until the CALL ends or the client hangs up */
        pthread_mutex_lock(&g_wmu);
        g_wbusy = 0;
        pthread_cond_broadcast(&g_wcv);
    }
    return NULL;
}

/** watcher_begin() - watch @client_fd (and accept on @listen_fd) while this CALL runs; 0 if unavailable. */
static int watcher_begin(int client_fd, int listen_fd)
{
    pthread_mutex_lock(&g_wmu);
    if (!g_wthread)
    {
        g_wctx.wake_fd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
        sigset_t all, old; /* signals stay with the main thread (sig_handler may siglongjmp) */
        sigfillset(&all);
        pthread_sigmask(SIG_BLOCK, &all, &old);
        pthread_t tid;
        g_wthread = (g_wctx.wake_fd >= 0 && pthread_create(&tid, &attr, watcher_main, NULL) == 0);
        pthread_sigmask(SIG_SETMASK, &old, NULL);
        pthread_attr_destroy(&attr);
    }
    if (g_wthread)
    {
        g_wctx.client_fd = client_fd;
        g_wctx.listen_fd = listen_fd;
        g_wctx.main_tid  = pthread_self();
        g_wctx.active    = 1;
        g_wgen++;
        pthread_cond_signal(&g_wcv);
    }
    pthread_mutex_unlock(&g_wmu);
    return g_wthread;
}

/** watcher_end() - stop watching the CALL; returns once the watcher no longer uses its fds. */
static void watcher_end(void)
{
    pthread_mutex_lock(&g_wmu);
    g_wctx.active = 0;
    uint64_t one  = 1;
    ssize_t  wret = write(g_wctx.wake_fd, &one, sizeof(one));
    (void) wret;
    while (g_wbusy)
    {
        pthread_cond_wait(&g_wcv, &g_wmu);
    }
    pthread_mutex_unlock(&g_wmu);
}

/**
 * milk_server_watcher_atfork_child - Reset CALL watcher state in a forked child process
 *
 * Because `fork()` does not copy the background watcher thread, a forked pipeline
 * worker must reinitialize the watcher mutex/condvar and clear `g_wthread` so it
 * can spawn its own watcher if it executes a nested `MILK_MSG_CALL`.
 */
void milk_server_watcher_atfork_child(void)
{
    pthread_mutex_init(&g_wmu, NULL); /* the watcher thread is not copied into the child */
    pthread_cond_init(&g_wcv, NULL);
    if (g_wctx.wake_fd >= 0)
    {
        close(g_wctx.wake_fd);
    }
    g_wctx.wake_fd = -1;
    g_wctx.active  = 0;
    g_wbusy        = 0;
    g_wthread      = 0;
}

/**
 * @brief Resolve a command name: exact `ns.cmd`, or an unambiguous short `cmd` (bare @cmd shims).
 *
 * @param name      Command name without leading sigil
 * @param out_ambig Set to 1 if multiple modules define the same short name
 * @return Command index in `data.cmd[]`, or -1 if not found / ambiguous
 */
int milk_server_resolve_cmd_index(const char *name, int *out_ambig)
{
    *out_ambig = 0;
    if (name == NULL || name[0] == '\0')
    {
        return -1;
    }

    /* 1. Exact registered key (hash index) */
    int exact = milk_cmd_lookup(name);
    if (exact >= 0)
    {
        return exact;
    }

    /* 2. If no dot in name, search for unambiguous "<ns>.<name>" */
    if (strchr(name, '.') == NULL)
    {
        int match_idx = -1;
        int match_cnt = 0;
        for (unsigned int i = 0; i < data.NBcmd; i++)
        {
            const char *dot = strchr(data.cmd[i].key, '.');
            if (dot != NULL && strcmp(dot + 1, name) == 0 && (dot - data.cmd[i].key) > 0)
            {
                match_idx = (int) i;
                match_cnt++;
            }
            else if (data.cmd[i].moduleindex >= 0 &&
                     data.module[data.cmd[i].moduleindex].shortname[0] != '\0' &&
                     strcmp(data.cmd[i].key, name) == 0)
            {
                match_idx = (int) i;
                match_cnt++;
            }
        }
        if (match_cnt == 1)
        {
            return match_idx;
        }
        if (match_cnt > 1)
        {
            *out_ambig = 1;
            fprintf(stderr, "\033[1;31mERROR:\033[0m ambiguous short command '@%s'; matches:\n",
                    name);
            for (unsigned int i = 0; i < data.NBcmd; i++)
            {
                const char *dot = strchr(data.cmd[i].key, '.');
                if (dot != NULL && strcmp(dot + 1, name) == 0)
                {
                    fprintf(stderr, "  @%s\n", data.cmd[i].key);
                }
            }
            return -1;
        }
    }

    return -1;
}

/**
 * @brief Handle built-in script/shell interceptors before `data.cmd[]` lookup.
 *
 * @return 1 if handled (exit code written to `*out_rc`), 0 if not an interceptor
 */
static int try_dispatch_interceptor(const char *cmdname, int argc, char **argv, int *out_rc)
{
    if (strcmp(cmdname, "server") == 0)
    {
        *out_rc = milk_srvlog_cmd_server(argc - 1, argv + 1, NULL);
        return 1;
    }
    if (strcmp(cmdname, "log") == 0)
    {
        *out_rc = milk_srvlog_cmd_log(getenv("MILKCLI_SOCK"), argc - 1, argv + 1, NULL);
        return 1;
    }
    if ((strcmp(cmdname, "listim") == 0 || strcmp(cmdname, "mem.listim") == 0) && argc >= 2 &&
        strcmp(argv[1], "-h") != 0 && strcmp(argv[1], "--help") != 0 && strcmp(argv[1], "-?") != 0)
    {
        if (argc == 2 && argv[1][0] != '-' &&
            (strchr(argv[1], '*') != NULL || strchr(argv[1], '?') != NULL))
        {
            cli_listim_glob(argv[1]);
            *out_rc = 0;
            return 1;
        }
        int         json_mode      = 0;
        int         porcelain_mode = 0;
        int         quiet_mode     = 0;
        const char *pattern        = NULL;
        for (int i = 1; i < argc; i++)
        {
            const char *tok = argv[i];
            if (strcmp(tok, "--json") == 0)
            {
                json_mode = 1;
            }
            else if (strcmp(tok, "--porcelain") == 0)
            {
                porcelain_mode = 1;
            }
            else if (strcmp(tok, "-q") == 0 || strcmp(tok, "--quiet") == 0)
            {
                quiet_mode = 1;
            }
            else if (tok[0] != '-')
            {
                pattern = tok;
            }
        }
        if (json_mode || porcelain_mode || quiet_mode)
        {
            list_image_ID_with_opts(json_mode, porcelain_mode, quiet_mode, pattern);
            *out_rc = 0;
            return 1;
        }
    }
    return 0;
}

/**
 * @brief Populate `data.cmdargtoken[]` from `argv[]` for command @cmdidx.
 *
 * @return 0 on success, 1 on parse/modifier error
 */
static int populate_cmdargtokens(int cmdidx, int argc, char **argv)
{
    data.parseerror = 0;
    cli_calc_tmp_cleanup();
    for (int i = 0; i < NB_ARG_MAX; i++)
    {
        data.cmdargtoken[i].type          = CMDARGTOKEN_TYPE_UNSOLVED;
        data.cmdargtoken[i].val.string[0] = '\0';
    }

    data.cmdindex = cmdidx;
    snprintf(data.cmdargtoken[0].val.string, STRINGMAXLEN_CMDARGTOKEN_VAL, "%s",
             data.cmd[cmdidx].key);
    data.cmdargtoken[0].type = CMDARGTOKEN_TYPE_COMMAND;
    data.cmdNBarg            = 1;

    for (int i = 1; i < argc && data.cmdNBarg < NB_ARG_MAX - 1; i++)
    {
        const char *arg    = argv[i];
        size_t      arglen = strlen(arg);
        const char *brk    = strchr(arg, '[');
        int         is_slice =
            (arg[0] != '@' && brk != NULL && brk > arg && arglen > 0 && arg[arglen - 1] == ']');
        if (is_slice)
        {
            for (const char *p = arg; p < brk; p++)
            {
                if (!isalnum((unsigned char) *p) && *p != '_' && *p != '.' && *p != '-')
                {
                    is_slice = 0;
                    break;
                }
            }
        }
        snprintf(data.cmdargtoken[data.cmdNBarg].val.string, STRINGMAXLEN_CMDARGTOKEN_VAL, "%s",
                 arg);

        int            is_ns_stream_arg = 0;
        milk_ns_kind_t ak               = milk_entity_ns_kind(arg, NULL);
        if (!is_slice && (ak == MILK_NS_STREAM || ak == MILK_NS_LOCAL))
        {
            FPS_STREAMNAME_PARSED sp = fps_streamname_parse(arg);
            if (!sp.error && sp.loc != '\0')
            {
                is_ns_stream_arg = 1;
            }
        }

        /* `=` too: the expression parser would run `a=b` as an assignment */
        if (!is_slice && (arg[0] == '-' || arg[0] == '/' || strchr(arg, ' ') != NULL ||
                          strchr(arg, '\t') != NULL || strchr(arg, '=') != NULL || arg[0] == '@' ||
                          is_ns_stream_arg))
        {
            data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_RAWSTRING;
            int is_help_cmd                      = (strcmp(data.cmd[cmdidx].key, "help") == 0);
            if ((arg[0] == '@' || is_ns_stream_arg) && !is_help_cmd)
            {
                IMGID tmpid = imgid_make_from_name(arg);
                if (tmpid.stream_parse_error)
                {
                    imgid_free(&tmpid);
                    data.CMDerrstatus = RETURN_FAILURE;
                    return 1;
                }
                if (tmpid.name[0] != '\0')
                {
                    char shmpath[512];
                    snprintf(shmpath, sizeof(shmpath), "%s/%s.im.shm", dcshmdir, tmpid.name);
                    if (image_ID(tmpid.name, data.core.image, data.core.NB_MAX_IMAGE) != -1 ||
                        (tmpid.stream_loc != 'L' && access(shmpath, R_OK) == 0 &&
                         read_sharedmem_image(tmpid.name, data.core.image,
                                              data.core.NB_MAX_IMAGE) != -1))
                    {
                        data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_EXISTINGIMAGE;
                    }
                    else
                    {
                        data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_STRING;
                    }
                }
                imgid_free(&tmpid);
            }
        }
        else
        {
            char tokstr[512];
            snprintf(tokstr, sizeof(tokstr), "%s\n", arg);
            cli_parse(tokstr);
            if (!is_slice && data.cmdargtoken[data.cmdNBarg].type == CMDARGTOKEN_TYPE_STRING)
            {
                char shmpath[512];
                snprintf(shmpath, sizeof(shmpath), "%s/%s.im.shm", dcshmdir, arg);
                if (access(shmpath, R_OK) == 0 &&
                    read_sharedmem_image(arg, data.core.image, data.core.NB_MAX_IMAGE) != -1)
                {
                    data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_EXISTINGIMAGE;
                }
            }
            if (is_slice && data.cmdargtoken[data.cmdNBarg].type == CMDARGTOKEN_TYPE_RAWSTRING)
            {
                data.CMDerrstatus = RETURN_FAILURE;
                return 1;
            }
        }
        data.cmdNBarg++;
    }
    if ((strcmp(data.cmd[cmdidx].key, "rm") == 0 || strcmp(data.cmd[cmdidx].key, "mem.rm") == 0) &&
        data.cmdNBarg == 2)
    {
        snprintf(data.cmdargtoken[2].val.string, STRINGMAXLEN_CMDARGTOKEN_VAL, "0");
        data.cmdargtoken[2].val.numl = 0;
        data.cmdargtoken[2].type     = CMDARGTOKEN_TYPE_LONG;
        data.cmdNBarg                = 3;
    }
    data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_UNSOLVED;
    return 0;
}

/**
 * is_server_entity_spec_token - Check whether @s looks like an entity property path
 * @s: Argument token to test
 *
 * Used by `@set` argument parsing to distinguish an optional trailing comment
 * from the start of the next `<entity> [=] <value>` assignment in a multi-property
 * `@set` call.
 *
 * Return: 1 if @s begins with an entity namespace prefix and contains a property dot, 0 otherwise.
 */
static int is_server_entity_spec_token(const char *s)
{
    if (s == NULL)
    {
        return 0;
    }
    while (*s == '@')
    {
        s++;
    }
    if ((strncmp(s, "S:", 2) == 0 || strncmp(s, "L:", 2) == 0) && strchr(s + 2, '.') != NULL)
    {
        return 1;
    }
    const char *rest = s;
    return milk_entity_ns_kind(s, &rest) != MILK_NS_NONE && strchr(rest, '.') != NULL;
}

/**
 * milk_server_set_argv - Execute the `@set` built-in command across one or more entity properties
 * @argc: Argument count (including `set`)
 * @argv: Argument vector (`set <entity> [=] [value] [comment] ...`)
 *
 * Parses one or more entity property assignments (`@set f.fps.param = val [comment]`)
 * and applies each through `milk_shm_prop_set()`.
 *
 * Return: 0 if all properties were set, 1 if any property write failed or on usage error.
 */
int milk_server_set_argv(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: @set <entity> [=] [value] [comment] [...]\n");
        return 1;
    }
    int rc = 0;
    int i  = 1;
    while (i < argc)
    {
        const char *spec    = argv[i++];
        const char *val     = NULL;
        const char *comment = NULL;
        if (strchr(spec, '=') != NULL)
        {
            if (i < argc && !is_server_entity_spec_token(argv[i]))
            {
                comment = argv[i++];
            }
        }
        else
        {
            if (i < argc && strcmp(argv[i], "=") == 0)
            {
                i++;
                if (i < argc)
                {
                    val = argv[i++];
                }
            }
            else if (i < argc && !is_server_entity_spec_token(argv[i]))
            {
                val = argv[i++];
            }
            if (i < argc && !is_server_entity_spec_token(argv[i]))
            {
                comment = argv[i++];
            }
        }
        if (milk_shm_prop_set(spec, val, comment) != 0)
        {
            rc = 1;
        }
    }
    return rc;
}

/**
 * @brief Dispatch a parsed `argv[]` command vector to built-ins, entities, or `data.cmd[]`.
 *
 * @param argc Argument count
 * @param argv Argument vector
 * @return Exit status (0 on success, non-zero on error)
 */
int milk_server_dispatch_argv(int argc, char **argv)
{
    if (argc <= 0 || argv[0] == NULL)
    {
        return 0;
    }

    const char *cmdname = argv[0];
    if (cmdname[0] == '@')
    {
        cmdname++;
    }

    if (strcmp(cmdname, "ls") == 0)
    {
        return milk_ls_exec(argc - 1, argv + 1);
    }
    if (strcmp(cmdname, "module") == 0)
    {
        return milk_server_module(argc, argv);
    }
    if (strcmp(cmdname, "config") == 0)
    {
        return milk_server_config(argc, argv);
    }
    if (strcmp(cmdname, "history") == 0)
    {
        int handled = 0;
        int hrc     = milk_server_history(argc, argv, &handled); /* @history <sub>; else below */
        if (handled)
        {
            return hrc;
        }
    }
    if (strcmp(cmdname, "streamcat") == 0)
    {
        return milk_server_streamcat(argc, argv);
    }
    if (strcmp(cmdname, "streamwrite") == 0)
    {
        return milk_server_streamwrite(argc, argv);
    }
    if (strcmp(cmdname, "replay") == 0)
    {
        return milk_server_replay(argc, argv);
    }
    if (strcmp(cmdname, "snapshot") == 0)
    {
        return milk_shm_prop_snapshot(argc, argv);
    }
    if (strcmp(cmdname, "diff") == 0)
    {
        return milk_shm_prop_diff(argc, argv);
    }
    if (strcmp(cmdname, "restore") == 0)
    {
        return milk_shm_prop_restore(argc, argv);
    }
    if (strcmp(cmdname, "assert") == 0)
    {
        return milk_server_assert(argc, argv);
    }
    if (strcmp(cmdname, "set") == 0)
    {
        return milk_server_set_argv(argc, argv);
    }
    if (strcmp(cmdname, "pipe") == 0)
    {
        return milk_server_batch_or_pipe(argc, argv);
    }

    /* Direct entity path invocation (@s.im.xsize, @s.im.kw.KEY=val, @f.fps.param=val) */
    {
        const char *dot1  = strchr(cmdname, '.');
        size_t      nslen = (dot1 != NULL) ? (size_t) (dot1 - cmdname) : 0;
        const char *dot2  = (dot1 != NULL) ? strchr(dot1 + 1, '.') : NULL;
        if (dot2 != NULL && milk_is_entity_ns_prefix(cmdname, nslen))
        {
            const char *eq = strchr(cmdname, '=');
            if (eq != NULL)
            {
                return (milk_shm_prop_set(cmdname, NULL, (argc >= 2) ? argv[1] : NULL) == 0) ? 0
                                                                                             : 1;
            }
            if (argc >= 3 && strcmp(argv[1], "=") == 0)
            {
                return (milk_shm_prop_set(cmdname, argv[2], (argc >= 4) ? argv[3] : NULL) == 0) ? 0
                                                                                                : 1;
            }
            if (argc == 1)
            {
                const char *prop = dot2 + 1;
                if (((strcmp(prop, "confstart") == 0 || strcmp(prop, "confstop") == 0 ||
                      strcmp(prop, "confstep") == 0 || strcmp(prop, "runstart") == 0 ||
                      strcmp(prop, "runstop") == 0 || strcmp(prop, "reset") == 0) &&
                     milk_entity_ns_kind(cmdname, NULL) == MILK_NS_FPS) ||
                    ((strcmp(prop, "pause") == 0 || strcmp(prop, "resume") == 0 ||
                      strcmp(prop, "run") == 0 || strcmp(prop, "step") == 0 ||
                      strcmp(prop, "incr") == 0 || strcmp(prop, "stop") == 0 ||
                      strcmp(prop, "exit") == 0 || strcmp(prop, "kill") == 0) &&
                     milk_entity_ns_kind(cmdname, NULL) == MILK_NS_PROC))
                {
                    return (milk_shm_prop_set(cmdname, "1", NULL) == 0) ? 0 : 1;
                }
                char val[1024];
                if (milk_shm_prop_get(cmdname, val, sizeof(val)) == 0)
                {
                    printf("%s\n", val);
                    return 0;
                }
                return 1;
            }
        }
    }

    /* Reconstruct data.CLIcmdline for commands and interceptors that inspect it */
    {
        size_t      pos = 0;
        const char *k0  = (strcmp(cmdname, "mem.listim") == 0 && argc >= 2 &&
                           (strchr(argv[1], '*') != NULL || strchr(argv[1], '?') != NULL))
                              ? "listim"
                              : cmdname;
        pos += (size_t) snprintf(data.CLIcmdline + pos, sizeof(data.CLIcmdline) - pos, "%s", k0);
        for (int i = 1; i < argc && pos + 2 < sizeof(data.CLIcmdline); i++)
        {
            pos += (size_t) snprintf(data.CLIcmdline + pos, sizeof(data.CLIcmdline) - pos, " %s",
                                     argv[i]);
        }
    }

    int intercept_rc = 0;
    if (try_dispatch_interceptor(cmdname, argc, argv, &intercept_rc))
    {
        return intercept_rc;
    }

    int ambig  = 0;
    int cmdidx = milk_server_resolve_cmd_index(cmdname, &ambig);
    if (cmdidx < 0 && !ambig && milk_catalog_autoload(cmdname)) /* first use of its module */
    {
        cmdidx = milk_server_resolve_cmd_index(cmdname, &ambig);
    }
    if (cmdidx < 0)
    {
        if (ambig)
        {
            return 2;
        }
        fprintf(stderr, "\033[1;31mmilk:\033[0m unknown command '@%s'\n", cmdname);
        handle_did_you_mean(cmdname);
        return 127;
    }
    if (argc >= 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0))
    {
        return (help_command(data.cmd[cmdidx].key) == RETURN_SUCCESS) ? 0 : 1; /* nothing runs */
    }
    int argrc = milk_args_begin(cmdidx, &argc, argv); /* --save= --load= --default --no-run -- */
    if (argrc != 0)
    {
        return argrc;
    }

    static unsigned int ncall = 0;
    if ((ncall++ & 15U) ==
        0) /* it keeps 500 image / 100 variable slots spare: no need on every call */
    {
        memory_re_alloc();
    }

    {
        size_t pos = 0;
        pos += (size_t) snprintf(data.CLIcmdline + pos, sizeof(data.CLIcmdline) - pos, "%s",
                                 data.cmd[cmdidx].key);
        for (int i = 1; i < argc && pos + 2 < sizeof(data.CLIcmdline); i++)
        {
            pos += (size_t) snprintf(data.CLIcmdline + pos, sizeof(data.CLIcmdline) - pos, " %s",
                                     argv[i]);
        }
    }

    if (populate_cmdargtokens(cmdidx, argc, argv) != 0)
    {
        milk_args_end();
        return 1;
    }

    data.cmd[cmdidx].callcount++;
    data.CMDerrstatus = RETURN_FAILURE;

    const char *outvars = data.cmd[cmdidx].outvars; /* the variables it sets for its caller */
    uint64_t    outh[MILK_OUTVARS_MAX];
    int         nout = (outvars != NULL) ? milk_server_outvars_snap(outvars, outh) : 0;

    /* the command gets its own fault-isolation frame; the caller's (REPL) frame is restored after */
    sigjmp_buf outer_env;
    memcpy(outer_env, *cli_get_repl_env(), sizeof(sigjmp_buf));
    int outer_armed = cli_is_fault_isolation_armed();
    int outer_fwd   = cli_get_sigint_forward();
    int aborted     = 0;
    cli_set_sigint_forward(0);
    cli_fault_isolation_arm();
    if (sigsetjmp(*cli_get_repl_env(), 1) == 0)
    {
        data.CMDerrstatus = data.cmd[cmdidx].fp();
    }
    else
    {
        aborted            = dcsigINT ? 130 : 1;
        g_call_interrupted = g_call_interrupted || dcsigINT;
        dcsigINT           = 0;
        dcsigSEGV          = 0;
        dcsigBUS           = 0;
        dcsigABRT          = 0;
    }
    memcpy(*cli_get_repl_env(), outer_env, sizeof(sigjmp_buf));
    if (outer_armed)
    {
        cli_fault_isolation_arm();
    }
    else
    {
        cli_fault_isolation_disarm();
    }
    cli_set_sigint_forward(outer_fwd);
    milk_server_outvars_diff(outvars, outh, nout);
    int usage = milk_args_end();
    if (aborted)
    {
        return aborted;
    }
    if (usage)
    {
        cli_calc_tmp_cleanup();
        return 2;
    }

    if (data.CMDerrstatus != RETURN_SUCCESS)
    {
        printf("\n%c[%d;%dm ERROR %c[%d;m CLI function %s returns %d\n", (char) 27, 1, 31,
               (char) 27, 0, data.cmd[cmdidx].key, data.CMDerrstatus);
    }

    cli_calc_tmp_cleanup();

    return (data.CMDerrstatus == RETURN_SUCCESS) ? 0 : 1;
}

/**
 * @brief Execute a `MILK_MSG_SET` payload.
 */
int milk_server_exec_set(const uint8_t *payload, uint32_t len, char *log_cmd, size_t log_cmd_sz)
{
    if (payload == NULL || len == 0)
    {
        return 1;
    }
    const char *spec    = (const char *) payload;
    size_t      o1      = strlen(spec) + 1;
    const char *val     = (o1 < len) ? (const char *) (payload + o1) : NULL;
    size_t      o2      = (val != NULL) ? (o1 + strlen(val) + 1) : len;
    const char *comment = (o2 < len) ? (const char *) (payload + o2) : NULL;
    if (val != NULL && val[0] == '\0' && strchr(spec, '=') != NULL)
    {
        val = NULL; // empty value slot: the value is in "spec=value"
    }
    snprintf(log_cmd, log_cmd_sz, "@set %s = %s", spec, val ? val : "");
    return (milk_shm_prop_set(spec, val, comment) == 0) ? 0 : 1;
}

/* Environment overrides sent with a CALL apply to that request only (restored afterwards). */
#define CALL_ENV_MAX 8
typedef struct
{
    char  name[128];
    char *old; // NULL: was unset
} call_env_saved_t;
static call_env_saved_t g_call_env[CALL_ENV_MAX];
static int              g_call_env_n       = 0;
static int              g_call_shm_changed = 0;

/**
 * apply_shm_dir_env() - re-resolve the SHM dir and reopen the server log if MILK_SHM_DIR moved
 * (@force: re-resolve even when MILK_SHM_DIR is unset). Returns 1 if it re-resolved.
 */
static int apply_shm_dir_env(int force)
{
    const char *sd = getenv("MILK_SHM_DIR");
    if (!force && (sd == NULL || strcmp(dcshmdir, sd) == 0))
    {
        return 0;
    }
    int saved_q = dcquiet;
    dcquiet     = 1;
    setSHMdir();
    dcquiet = saved_q;
    if (g_srvlog_shm != NULL && g_srvlog_sockpath[0] != '\0')
    {
        uint32_t old_flags = g_srvlog_shm->flags;
        milk_srvlog_server_close(g_srvlog_shm, 0);
        g_srvlog_shm = milk_srvlog_server_init(g_srvlog_sockpath, 0, old_flags);
        milk_server_refresh_log_stats();
    }
    return 1;
}

/**
 * @brief Unpack environment updates from `MILK_MSG_CALL` payload and re-init SHM dir if changed.
 */
static void unpack_call_env(const uint8_t *payload, uint32_t len, uint32_t *offset_io)
{
    uint32_t off = *offset_io;
    g_call_env_n = 0;
    if (off + sizeof(uint32_t) > len)
    {
        return;
    }
    uint32_t nenv = 0;
    memcpy(&nenv, payload + off, sizeof(uint32_t));
    off += (uint32_t) sizeof(uint32_t);
    for (uint32_t i = 0; i < nenv && off < len; i++)
    {
        const char *kv = (const char *) (payload + off);
        size_t      kl = strnlen(kv, len - off);
        off += (uint32_t) (kl + 1);
        const char *eq = strchr(kv, '=');
        size_t      kn = (eq != NULL) ? (size_t) (eq - kv) : 0;
        if (eq == NULL || kn == 0 || kn >= sizeof(g_call_env[0].name) ||
            g_call_env_n >= CALL_ENV_MAX)
        {
            continue;
        }
        call_env_saved_t *e = &g_call_env[g_call_env_n++];
        memcpy(e->name, kv, kn);
        e->name[kn]     = '\0';
        const char *cur = getenv(e->name);
        e->old          = (cur != NULL) ? strdup(cur) : NULL;
        setenv(e->name, eq + 1, 1);
    }
    g_call_shm_changed = apply_shm_dir_env(0);
    *offset_io         = off;
}

/** restore_call_env() - undo unpack_call_env() so one client's environment never leaks to the next. */
static void restore_call_env(void)
{
    while (g_call_env_n > 0)
    {
        call_env_saved_t *e = &g_call_env[--g_call_env_n];
        if (e->old != NULL)
        {
            setenv(e->name, e->old, 1);
            free(e->old);
            e->old = NULL;
        }
        else
        {
            unsetenv(e->name);
        }
    }
    if (g_call_shm_changed)
    {
        apply_shm_dir_env(1);
        g_call_shm_changed = 0;
    }
}

/**
 * @brief Execute a `MILK_MSG_CALL` payload.
 */
int milk_server_exec_call(int            client_fd,
                          int            listen_fd,
                          const uint8_t *payload,
                          uint32_t       len,
                          char          *log_cmd,
                          size_t         log_cmd_sz)
{
    if (payload == NULL || len < sizeof(uint32_t))
    {
        return 1;
    }

    uint32_t off  = 0;
    uint32_t argc = 0;
    memcpy(&argc, payload + off, sizeof(uint32_t));
    off += (uint32_t) sizeof(uint32_t);

    if (argc == 0 || argc > len - off) // every argument needs at least its NUL byte
    {
        fprintf(stderr, "milk: malformed CALL request (argc %u)\n", argc);
        return 1;
    }
    char **argv = (char **) calloc((size_t) argc + 1, sizeof(char *));
    if (argv == NULL)
    {
        return 1;
    }
    size_t   lpos  = 0;
    uint32_t nargs = 0;
    for (uint32_t i = 0; i < argc && off < len; i++, nargs++)
    {
        argv[i] = (char *) (payload + off);
        off += (uint32_t) (strnlen(argv[i], len - off) + 1);
        if (lpos + 2 < log_cmd_sz)
        {
            if (i > 0)
            {
                log_cmd[lpos++] = ' ';
            }
            size_t alen = strlen(argv[i]);
            size_t rem  = log_cmd_sz - 1 - lpos;
            size_t cp   = (alen < rem) ? alen : rem;
            memcpy(log_cmd + lpos, argv[i], cp);
            lpos += cp;
            log_cmd[lpos] = '\0';
        }
    }
    if (nargs < argc)
    {
        fprintf(stderr, "milk: malformed CALL request (%u of %u arguments)\n", nargs, argc);
        free(argv);
        return 1;
    }

    unpack_call_env(payload, len, &off);

    if (off + sizeof(uint32_t) <= len)
    {
        uint32_t nvars = 0;
        memcpy(&nvars, payload + off, sizeof(uint32_t));
        off += (uint32_t) sizeof(uint32_t);
        milk_server_sync_bash_vars(payload, len, nvars, &off, 0);
    }

    milk_server_outvars_send(-1); /* nothing queued by an earlier request survives */
    unsigned int nbcmd_before  = data.NBcmd;
    int          digits_before = cli_float_digits;

    // also in forked workers (listen_fd < 0, ignored by poll): CANCEL/hang-up must stop them
    int has_watcher = watcher_begin(client_fd, listen_fd);

    int rc = milk_server_dispatch_argv((int) argc, argv);

    if (has_watcher)
    {
        watcher_end();
    }

    fflush(stdout);
    fflush(stderr);

    milk_server_outvars_send(client_fd);
    if (cli_float_digits != digits_before) /* the builtin formats scalar results itself */
    {
        char pkt[32];
        int  n = snprintf(pkt, sizeof(pkt), "MILK_DPDIGITS%c%d", '\0', cli_float_digits);
        milk_proto_send(client_fd, MILK_MSG_SETVAR, 0, pkt, (uint32_t) n + 1);
    }

    /* If a module was dynamically loaded (@module load), send DEFCMD */
    for (unsigned int ci = nbcmd_before; ci < data.NBcmd; ci++)
    {
        const char *ck = data.cmd[ci].key;
        if (ck[0] != '\0')
        {
            milk_proto_send(client_fd, MILK_MSG_DEFCMD, 0, ck, (uint32_t) (strlen(ck) + 1));
            int midx = data.cmd[ci].moduleindex;
            if (strchr(ck, '.') == NULL && midx >= 0 && data.module[midx].shortname[0] != '\0')
            {
                char qk[256];
                snprintf(qk, sizeof(qk), "%s.%s", data.module[midx].shortname, ck);
                milk_proto_send(client_fd, MILK_MSG_DEFCMD, 0, qk, (uint32_t) (strlen(qk) + 1));
            }
        }
    }

    free(argv);
    restore_call_env();
    return rc;
}
