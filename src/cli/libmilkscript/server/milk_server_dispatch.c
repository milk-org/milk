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
 *   2. Executes `MILK_MSG_SET`, `MILK_MSG_CALC` (`@=`), `MILK_MSG_TEST` (`@test`),
 *      and `MILK_MSG_CALL` payloads, managing variable synchronization,
 *      fault-isolated command execution (`sigsetjmp`), dynamic module command
 *      notification (`MILK_MSG_DEFCMD`), and concurrent pipeline/Ctrl-C monitoring.
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
#include "milk_proto.h"
#include "milk_server_internal.h"
#include "milk_srvlog.h"

extern void handle_did_you_mean(const char *input_cmd);
extern int  cli_intercept_cmd_procctl(const char *p);
extern int  cli_intercept_cmd_procwait(const char *p);
extern int  cli_intercept_cmd_procstat(const char *p);
extern int  cli_intercept_cmd_dpdigits(const char *p);

typedef struct
{
    int          client_fd;
    int          listen_fd;
    pthread_t    main_tid;
    volatile int active;
} call_watcher_ctx_t;

/**
 * @brief Background watcher thread active during `MILK_MSG_CALL` execution.
 *
 * Monitors @ctx->client_fd for `MILK_MSG_CANCEL` / `MILK_MSG_WINCH` / hangup
 * (forwarding `SIGINT` / `SIGWINCH` to the main server thread) and accepts
 * concurrent pipeline stage connections on @ctx->listen_fd in forked children.
 */
static void *call_watcher_thread(void *arg)
{
    call_watcher_ctx_t *ctx = (call_watcher_ctx_t *) arg;
    while (ctx->active)
    {
        struct pollfd pfds[2];
        pfds[0].fd      = ctx->client_fd;
        pfds[0].events  = POLLIN | POLLHUP | POLLERR;
        pfds[0].revents = 0;
        pfds[1].fd      = ctx->listen_fd;
        pfds[1].events  = POLLIN;
        pfds[1].revents = 0;

        int pr = poll(pfds, 2, 20);
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
                struct ucred cred;
                socklen_t    clen = sizeof(cred);
                if (getsockopt(cfd, SOL_SOCKET, SO_PEERCRED, &cred, &clen) == 0 &&
                    cred.uid == getuid())
                {
                    pid_t pid = fork();
                    if (pid == 0)
                    {
                        close(ctx->listen_fd);
                        int dummy_shutdown = 0;
                        milk_server_handle_client(cfd, -1, &dummy_shutdown);
                        close(cfd);
                        _exit(0);
                    }
                }
                close(cfd);
            }
        }
    }
    return NULL;
}

/**
 * @brief Resolve a command name (canonical `ns.cmd` or unambiguous short `cmd`).
 *
 * @param name      Command name without leading sigil
 * @param out_ambig Set to 1 if multiple modules define the same short name
 * @return Command index in `data.cmd[]`, or -1 if not found / ambiguous
 */
static int resolve_milk_cmd_index(const char *name, int *out_ambig)
{
    *out_ambig = 0;
    if (name == NULL || name[0] == '\0')
    {
        return -1;
    }

    if (strcmp(name, "imgstat") == 0 || strcmp(name, "info.imgstat") == 0)
    {
        name = "info.imstats";
    }
    else if (strcmp(name, "mk2Drnd") == 0 || strcmp(name, "imgen.mk2Drnd") == 0)
    {
        name = "imgen.mkrndim";
    }
    else if (strcmp(name, "savefits") == 0 || strcmp(name, "iofits.savefits") == 0)
    {
        name = "saveFITS";
    }
    else if (strcmp(name, "lm?") == 0)
    {
        name = "hm";
    }

    /* 1. Exact match on registered command key */
    for (unsigned int i = 0; i < data.NBcmd; i++)
    {
        if (strcmp(data.cmd[i].key, name) == 0)
        {
            return (int) i;
        }
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

    /* 3. If name is "<shortmod>.<cmd>", check module shortname + key */
    const char *dot = strchr(name, '.');
    if (dot != NULL)
    {
        size_t      mlen = (size_t) (dot - name);
        const char *sub  = dot + 1;
        for (unsigned int i = 0; i < data.NBcmd; i++)
        {
            int midx = data.cmd[i].moduleindex;
            if (midx >= 0)
            {
                const char *ms = data.module[midx].shortname;
                const char *mn = data.module[midx].name;
                if (((strlen(ms) == mlen && strncmp(ms, name, mlen) == 0) ||
                     (strlen(mn) == mlen && strncmp(mn, name, mlen) == 0)) &&
                    strcmp(data.cmd[i].key, sub) == 0)
                {
                    return (int) i;
                }
            }
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
    if (strcmp(cmdname, "procctl") == 0)
    {
        cli_last_retval = 0;
        cli_intercept_cmd_procctl(data.CLIcmdline);
        *out_rc = cli_last_retval;
        return 1;
    }
    if (strcmp(cmdname, "procwait") == 0)
    {
        cli_last_retval = 0;
        cli_intercept_cmd_procwait(data.CLIcmdline);
        *out_rc = cli_last_retval;
        return 1;
    }
    if (strcmp(cmdname, "procstat") == 0)
    {
        cli_last_retval = 0;
        cli_intercept_cmd_procstat(data.CLIcmdline);
        *out_rc = cli_last_retval;
        return 1;
    }
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
    if (strcmp(cmdname, "dpdigits") == 0)
    {
        cli_intercept_cmd_dpdigits(data.CLIcmdline);
        *out_rc = 0;
        return 1;
    }
    if ((strcmp(cmdname, "listim") == 0 || strcmp(cmdname, "mem.listim") == 0) && argc >= 2 &&
        argv[1][0] != '-' && (strchr(argv[1], '*') != NULL || strchr(argv[1], '?') != NULL))
    {
        cli_listim_glob(argv[1]);
        *out_rc = 0;
        return 1;
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
    data.parseerror      = 0;
    data.calctmp_imindex = 0;
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

        if (!is_slice && (arg[0] == '-' || arg[0] == '/' || strchr(arg, ' ') != NULL ||
                          strchr(arg, '\t') != NULL || arg[0] == '@'))
        {
            data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_RAWSTRING;
            int is_help_cmd                      = (strcmp(data.cmd[cmdidx].key, "h") == 0 ||
                                                    strcmp(data.cmd[cmdidx].key, "help") == 0 ||
                                                    strcmp(data.cmd[cmdidx].key, "hc") == 0 ||
                                                    strcmp(data.cmd[cmdidx].key, "cmd?") == 0 ||
                                                    strcmp(data.cmd[cmdidx].key, "hm") == 0 ||
                                                    strcmp(data.cmd[cmdidx].key, "m?") == 0);
            if (arg[0] == '@' && !is_help_cmd)
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
                    if (image_ID(tmpid.name, data.core.image, data.core.NB_MAX_IMAGE) != -1 ||
                        read_sharedmem_image(tmpid.name, data.core.image, data.core.NB_MAX_IMAGE) !=
                            -1)
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
    data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_UNSOLVED;
    return 0;
}

static int milk_server_batch_or_pipe(int argc, char **argv);

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
    static const char *const pfx[] = { "s.",   "f.",      "p.",    "q.", "fps.",
                                       "seq.", "stream.", "proc.", NULL };
    for (int k = 0; pfx[k] != NULL; k++)
    {
        size_t plen = strlen(pfx[k]);
        if (strncmp(s, pfx[k], plen) == 0 && strchr(s + plen, '.') != NULL)
        {
            return 1;
        }
    }
    return 0;
}

static int milk_server_set_argv(int argc, char **argv)
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
    if (cmdname[0] == '@' || cmdname[0] == ',')
    {
        cmdname++;
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
    if (strcmp(cmdname, "set") == 0)
    {
        return milk_server_set_argv(argc, argv);
    }
    if (strcmp(cmdname, "batch") == 0 || strcmp(cmdname, "pipe") == 0)
    {
        return milk_server_batch_or_pipe(argc, argv);
    }

    /* Direct entity path invocation (@s.im.xsize, @s.im.kw.KEY=val, @f.fps.param=val) */
    {
        const char *dot1  = strchr(cmdname, '.');
        size_t      nslen = (dot1 != NULL) ? (size_t) (dot1 - cmdname) : 0;
        const char *dot2  = (dot1 != NULL) ? strchr(dot1 + 1, '.') : NULL;
        if (dot2 != NULL && milk_server_is_entity_ns_prefix(cmdname, nslen))
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
                if ((strcmp(prop, "confstart") == 0 || strcmp(prop, "confstop") == 0 ||
                     strcmp(prop, "confstep") == 0 || strcmp(prop, "runstart") == 0 ||
                     strcmp(prop, "runstop") == 0 || strcmp(prop, "reset") == 0) &&
                    (strncmp(cmdname, "f.", 2) == 0 || strncmp(cmdname, "fps.", 4) == 0))
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
    int cmdidx = resolve_milk_cmd_index(cmdname, &ambig);
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

    memory_re_alloc();
    compute_image_memory();
    compute_nb_image();

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
        return 1;
    }

    data.cmd[cmdidx].callcount++;
    data.CMDerrstatus = RETURN_FAILURE;

    cli_fault_isolation_arm();
    if (sigsetjmp(*cli_get_repl_env(), 1) == 0)
    {
        data.CMDerrstatus = data.cmd[cmdidx].fp();
    }
    else
    {
        int was_sigint = dcsigINT;
        dcsigINT       = 0;
        dcsigSEGV      = 0;
        dcsigBUS       = 0;
        dcsigABRT      = 0;
        cli_fault_isolation_disarm();
        return was_sigint ? 130 : 1;
    }
    cli_fault_isolation_disarm();

    if (data.CMDerrstatus != RETURN_SUCCESS)
    {
        printf("\n%c[%d;%dm ERROR %c[%d;m CLI function %s returns %d\n", (char) 27, 1, 31,
               (char) 27, 0, data.cmd[cmdidx].key, data.CMDerrstatus);
    }

    for (int i = 0; i < data.calctmp_imindex; i++)
    {
        char calctmpimname[STRINGMAXLEN_IMGNAME];
        CREATE_IMAGENAME(calctmpimname, "_tmpcalc%d", i);
        if (image_ID(calctmpimname, dcimg, dcnimg) != -1)
        {
            delete_image_ID(calctmpimname, DELETE_IMAGE_ERRMODE_WARNING);
        }
    }
    data.calctmp_imindex = 0;

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
    snprintf(log_cmd, log_cmd_sz, "@set %s = %s", spec, val ? val : "");
    return (milk_shm_prop_set(spec, val, comment) == 0) ? 0 : 1;
}

/**
 * @brief Try handling an entity LHS assignment (`@f.fps.param = rhs_expr`) inside `@=`.
 *
 * @return 1 if @expr was an entity LHS assignment (status in `*out_rc`), 0 otherwise
 */
static int try_exec_entity_lhs_assignment(const char *expr, int *out_rc)
{
    const char *p = expr;
    while (*p == ' ' || *p == '\t')
    {
        p++;
    }
    const char *lstart = (*p == '@') ? (p + 1) : p;
    const char *le     = lstart;
    int         ndots  = 0;
    while (*le != '\0' && (isalnum((unsigned char) *le) || *le == '_' || *le == '.' || *le == '-'))
    {
        if (*le == '.')
        {
            ndots++;
        }
        le++;
    }
    const char *dot1  = strchr(lstart, '.');
    size_t      nslen = (dot1 != NULL && dot1 < le) ? (size_t) (dot1 - lstart) : 0;
    if (ndots < 2 || (size_t) (le - lstart) >= 256 ||
        !milk_server_is_entity_ns_prefix(lstart, nslen))
    {
        return 0;
    }

    const char *eq = le;
    while (*eq == ' ' || *eq == '\t')
    {
        eq++;
    }
    if (*eq != '=' || eq[1] == '=')
    {
        return 0;
    }

    char lhs[256];
    memcpy(lhs, lstart, (size_t) (le - lstart));
    lhs[le - lstart] = '\0';
    const char *rhs  = eq + 1;
    while (*rhs == ' ' || *rhs == '\t')
    {
        rhs++;
    }
    char exp_rhs[4096];
    milk_server_expand_entities_in_expr(rhs, exp_rhs, sizeof(exp_rhs));
    size_t rlen = strlen(exp_rhs);
    while (rlen > 0 && (exp_rhs[rlen - 1] == ' ' || exp_rhs[rlen - 1] == '\t'))
    {
        exp_rhs[--rlen] = '\0';
    }
    if (rlen >= 2 && ((exp_rhs[0] == '"' && exp_rhs[rlen - 1] == '"') ||
                      (exp_rhs[0] == '\'' && exp_rhs[rlen - 1] == '\'')))
    {
        exp_rhs[rlen - 1] = '\0';
        *out_rc           = (milk_shm_prop_set(lhs, exp_rhs + 1, NULL) == 0) ? 0 : 1;
        return 1;
    }

    char assign_expr[4200];
    snprintf(assign_expr, sizeof(assign_expr), "_milk_ent_rhs = %s", exp_rhs);
    int eval_rc = 2;
    int saved_q = dcquiet;
    dcquiet     = 1;
    cli_calc_eval_bash(assign_expr, 0, &eval_rc);
    dcquiet   = saved_q;
    long vidx = variable_ID("_milk_ent_rhs");
    if (eval_rc == 0 && vidx != -1)
    {
        char vbuf[128];
        if (data.core.variable[vidx].type == 1)
        {
            snprintf(vbuf, sizeof(vbuf), "%ld", data.core.variable[vidx].value.l);
        }
        else if (data.core.variable[vidx].type == 2)
        {
            snprintf(vbuf, sizeof(vbuf), "%s", data.core.variable[vidx].value.s);
        }
        else
        {
            snprintf(vbuf, sizeof(vbuf), "%.*g", cli_float_digits,
                     data.core.variable[vidx].value.f);
        }
        data.core.variable[vidx].used = 0;
        *out_rc                       = (milk_shm_prop_set(lhs, vbuf, NULL) == 0) ? 0 : 1;
    }
    else
    {
        *out_rc = (milk_shm_prop_set(lhs, exp_rhs, NULL) == 0) ? 0 : 1;
    }
    return 1;
}

static void substitute_underscore_token(const char *in,
                                        const char *last_img,
                                        char       *out,
                                        size_t      outsz)
{
    if (last_img == NULL || last_img[0] == '\0')
    {
        snprintf(out, outsz, "%s", in);
        return;
    }
    size_t pos   = 0;
    size_t ilen  = strlen(in);
    size_t llen  = strlen(last_img);
    int    in_sq = 0;

    for (size_t i = 0; i < ilen && pos + 1 < outsz;)
    {
        if (in[i] == '\'')
        {
            in_sq = !in_sq;
        }
        if (!in_sq && in[i] == '_')
        {
            int prev_ok = (i == 0) || (!isalnum((unsigned char) in[i - 1]) && in[i - 1] != '_' &&
                                       in[i - 1] != '.');
            int next_ok = (i + 1 >= ilen) || (!isalnum((unsigned char) in[i + 1]) &&
                                              in[i + 1] != '_' && in[i + 1] != '.');
            if (prev_ok && next_ok && pos + llen < outsz)
            {
                memcpy(out + pos, last_img, llen);
                pos += llen;
                i++;
                continue;
            }
        }
        out[pos++] = in[i++];
    }
    out[pos] = '\0';
}

static int exec_batch_calc_expr(const char *expr, int is_test, char *last_img, size_t last_img_sz)
{
    int rc = 0;
    if (is_test || !try_exec_entity_lhs_assignment(expr, &rc))
    {
        char expanded[8192];
        milk_server_expand_entities_in_expr(expr, expanded, sizeof(expanded));
        int eval_rc = 2;
        cli_calc_eval_bash(expanded, is_test, &eval_rc);
        rc = eval_rc;
        if (rc == 0 && !is_test)
        {
            const char *p = expanded;
            while (*p == ' ' || *p == '\t')
            {
                p++;
            }
            char   lhs[128];
            size_t l = 0;
            while (p[l] != '\0' && (isalnum((unsigned char) p[l]) || p[l] == '_') &&
                   l + 1 < sizeof(lhs))
            {
                lhs[l] = p[l];
                l++;
            }
            lhs[l] = '\0';
            if (l > 0 && image_ID(lhs, data.core.image, data.core.NB_MAX_IMAGE) != -1)
            {
                snprintf(last_img, last_img_sz, "%s", lhs);
            }
        }
    }
    return rc;
}

static int tokenize_step_line(const char *step,
                              char      **argv_out,
                              int         max_args,
                              char       *arena,
                              size_t      arena_sz)
{
    int         argc = 0;
    size_t      apos = 0;
    const char *p    = step;

    while (*p != '\0' && argc < max_args)
    {
        while (*p == ' ' || *p == '\t')
        {
            p++;
        }
        if (*p == '\0' || *p == '#')
        {
            break;
        }
        argv_out[argc++] = arena + apos;
        int in_sq        = 0;
        int in_dq        = 0;
        int br_depth     = 0;
        while (*p != '\0' && apos + 1 < arena_sz)
        {
            if (!in_sq && !in_dq && br_depth == 0 && (*p == ' ' || *p == '\t'))
            {
                break;
            }
            if (*p == '\\' && !in_sq && p[1] != '\0')
            {
                p++;
                arena[apos++] = *p++;
                continue;
            }
            if (*p == '\'' && !in_dq)
            {
                in_sq = !in_sq;
                p++;
                continue;
            }
            if (*p == '"' && !in_sq)
            {
                in_dq = !in_dq;
                p++;
                continue;
            }
            if (!in_sq && !in_dq)
            {
                if (*p == '[')
                {
                    br_depth++;
                }
                else if (*p == ']' && br_depth > 0)
                {
                    br_depth--;
                }
            }
            arena[apos++] = *p++;
        }
        arena[apos++] = '\0';
    }
    return argc;
}

static int exec_single_batch_step(const char *raw_step, char *last_img, size_t last_img_sz)
{
    while (*raw_step == ' ' || *raw_step == '\t' || *raw_step == '\r' || *raw_step == '\n')
    {
        raw_step++;
    }
    if (*raw_step == '\0' || *raw_step == '#')
    {
        return 0;
    }

    char step_buf[4096];
    substitute_underscore_token(raw_step, last_img, step_buf, sizeof(step_buf));
    size_t slen = strlen(step_buf);
    while (slen > 0 && (step_buf[slen - 1] == ' ' || step_buf[slen - 1] == '\t' ||
                        step_buf[slen - 1] == '\r' || step_buf[slen - 1] == '\n'))
    {
        step_buf[--slen] = '\0';
    }
    if (slen == 0)
    {
        return 0;
    }

    const char *s = step_buf;
    if (s[0] == '@' || s[0] == ',')
    {
        s++;
    }
    if (s[0] == ' ' || s[0] == '\t')
    {
        while (*s == ' ' || *s == '\t')
        {
            s++;
        }
        return exec_batch_calc_expr(s, 0, last_img, last_img_sz);
    }
    if (strncmp(s, "calc", 4) == 0 && (s[4] == ' ' || s[4] == '\t'))
    {
        s += 4;
        while (*s == ' ' || *s == '\t')
        {
            s++;
        }
        return exec_batch_calc_expr(s, 0, last_img, last_img_sz);
    }
    if (strncmp(s, "test", 4) == 0 && (s[4] == ' ' || s[4] == '\t'))
    {
        s += 4;
        while (*s == ' ' || *s == '\t')
        {
            s++;
        }
        return exec_batch_calc_expr(s, 1, last_img, last_img_sz);
    }

    char *s_argv[64];
    char  arena[4096];
    int   s_argc = tokenize_step_line(step_buf, s_argv, 64, arena, sizeof(arena));
    if (s_argc <= 0)
    {
        return 0;
    }

    const char *w0 = s_argv[0];
    if (w0[0] == '@' || w0[0] == ',')
    {
        w0++;
    }

    if (strcmp(w0, "set") == 0 && s_argc >= 2)
    {
        return milk_server_set_argv(s_argc, s_argv);
    }
    if (strcmp(w0, "get") == 0 && s_argc >= 2)
    {
        char val[16384];
        if (milk_shm_prop_get(s_argv[1], val, sizeof(val)) == 0)
        {
            printf("%s\n", val);
            return 0;
        }
        return 1;
    }

    int ambig  = 0;
    int cmdidx = resolve_milk_cmd_index(w0, &ambig);
    if (cmdidx < 0 && !ambig && strcmp(w0, "streamcat") != 0 && strcmp(w0, "streamwrite") != 0 &&
        (strchr(step_buf, '=') != NULL || strchr(step_buf, '(') != NULL ||
         strchr(step_buf, '+') != NULL || strchr(step_buf, '-') != NULL ||
         strchr(step_buf, '*') != NULL || strchr(step_buf, '/') != NULL))
    {
        return exec_batch_calc_expr(s, 0, last_img, last_img_sz);
    }

    int rc = milk_server_dispatch_argv(s_argc, s_argv);
    if (rc == 0)
    {
        for (int k = 1; k < s_argc; k++)
        {
            IMGID tmpid = imgid_make_from_name(s_argv[k]);
            if (!tmpid.stream_parse_error && tmpid.name[0] != '\0' &&
                image_ID(tmpid.name, data.core.image, data.core.NB_MAX_IMAGE) != -1)
            {
                snprintf(last_img, last_img_sz, "%s", tmpid.name);
            }
            imgid_free(&tmpid);
        }
    }
    return rc;
}

static int milk_server_batch_or_pipe(int argc, char **argv)
{
    char  *script = NULL;
    size_t slen   = 0;

    if (argc <= 1)
    {
        if (isatty(STDIN_FILENO))
        {
            return 0;
        }
        size_t cap = 4096;
        script     = (char *) malloc(cap);
        if (script == NULL)
        {
            return 1;
        }
        ssize_t nr;
        while ((nr = read(STDIN_FILENO, script + slen, cap - slen - 1)) > 0)
        {
            slen += (size_t) nr;
            if (slen + 1024 >= cap)
            {
                cap *= 2;
                char *ns = (char *) realloc(script, cap);
                if (ns == NULL)
                {
                    free(script);
                    return 1;
                }
                script = ns;
            }
        }
        script[slen] = '\0';
    }
    else if (argc == 2)
    {
        script = strdup(argv[1]);
        slen   = (script != NULL) ? strlen(script) : 0;
    }
    else
    {
        int    has_sep_tok = 0;
        size_t cap         = 64;
        for (int i = 1; i < argc; i++)
        {
            cap += strlen(argv[i]) + 4;
            if (strcmp(argv[i], "|>") == 0 || strcmp(argv[i], ";") == 0)
            {
                has_sep_tok = 1;
            }
        }
        script = (char *) malloc(cap);
        if (script == NULL)
        {
            return 1;
        }
        int multi_stage_args =
            (!has_sep_tok && (strchr(argv[1], ' ') != NULL || strchr(argv[1], '=') != NULL));
        for (int i = 1; i < argc; i++)
        {
            if (has_sep_tok && (strcmp(argv[i], "|>") == 0 || strcmp(argv[i], ";") == 0))
            {
                memcpy(script + slen, "; ", 2);
                slen += 2;
            }
            else
            {
                if (i > 1)
                {
                    if (multi_stage_args)
                    {
                        memcpy(script + slen, "; ", 2);
                        slen += 2;
                    }
                    else
                    {
                        script[slen++] = ' ';
                    }
                }
                size_t l = strlen(argv[i]);
                memcpy(script + slen, argv[i], l);
                slen += l;
            }
        }
        script[slen] = '\0';
    }

    if (script == NULL)
    {
        return 1;
    }

    char last_img[256] = "";
    long vidx_last     = variable_ID("_last_img");
    if (vidx_last != -1 && data.core.variable[vidx_last].type == 2)
    {
        snprintf(last_img, sizeof(last_img), "%s", data.core.variable[vidx_last].value.s);
    }

    int    rc       = 0;
    int    in_sq    = 0;
    int    in_dq    = 0;
    int    br_depth = 0;
    size_t seg_st   = 0;

    for (size_t i = 0; i <= slen; i++)
    {
        char c       = script[i];
        int  sep_len = 0;

        if (c == '\0')
        {
            sep_len = 1;
        }
        else if (c == '\\' && !in_sq && i + 1 < slen)
        {
            i++;
            continue;
        }
        else if (c == '\'' && !in_dq)
        {
            in_sq = !in_sq;
        }
        else if (c == '"' && !in_sq)
        {
            in_dq = !in_dq;
        }
        else if (!in_sq && !in_dq)
        {
            if (c == '[')
            {
                br_depth++;
            }
            else if (c == ']' && br_depth > 0)
            {
                br_depth--;
            }
            else if (br_depth == 0)
            {
                if (c == '\n' || c == ';')
                {
                    sep_len = 1;
                }
                else if (c == '|' && script[i + 1] == '>')
                {
                    sep_len = 2;
                }
            }
        }

        if (sep_len > 0)
        {
            size_t seglen = i - seg_st;
            char  *seg    = (char *) malloc(seglen + 1);
            if (seg != NULL)
            {
                memcpy(seg, script + seg_st, seglen);
                seg[seglen] = '\0';
                rc          = exec_single_batch_step(seg, last_img, sizeof(last_img));
                free(seg);
            }
            if (rc != 0 || c == '\0')
            {
                break;
            }
            i += (size_t) (sep_len - 1);
            seg_st = i + 1;
        }
    }

    if (last_img[0] != '\0')
    {
        create_variable_string_ID("_last_img", last_img);
    }

    free(script);
    return rc;
}

/**
 * @brief Execute a `MILK_MSG_CALC` (`@=`) or `MILK_MSG_TEST` (`@test`) payload.
 */
int milk_server_exec_calc_or_test(int            client_fd,
                                  int            is_test,
                                  const uint8_t *payload,
                                  uint32_t       len,
                                  char          *log_cmd,
                                  size_t         log_cmd_sz)
{
    if (payload == NULL || len == 0)
    {
        return is_test ? 1 : 0;
    }

    const char *expr = (const char *) payload;
    snprintf(log_cmd, log_cmd_sz, "%s %s", is_test ? "@test" : "@=", expr);
    uint32_t off = (uint32_t) (strlen(expr) + 1);
    if (off + sizeof(uint32_t) <= len)
    {
        uint32_t nvars = 0;
        memcpy(&nvars, payload + off, sizeof(uint32_t));
        off += (uint32_t) sizeof(uint32_t);
        milk_server_sync_bash_vars(payload, len, nvars, &off);
    }

    long          snap_nmax = 0;
    dcvar_snap_t *snap      = milk_server_snapshot_dcvar(&snap_nmax);
    int           rc        = 0;

    if (is_test || !try_exec_entity_lhs_assignment(expr, &rc))
    {
        char expanded[8192];
        milk_server_expand_entities_in_expr(expr, expanded, sizeof(expanded));
        int eval_rc = 2;
        cli_calc_eval_bash(expanded, is_test, &eval_rc);
        rc = eval_rc;
    }

    milk_server_send_modified_dcvars(client_fd, snap, snap_nmax);
    free(snap);
    return rc;
}

/**
 * @brief Unpack environment updates from `MILK_MSG_CALL` payload and re-init SHM dir if changed.
 */
static void unpack_call_env(const uint8_t *payload, uint32_t len, uint32_t *offset_io)
{
    uint32_t off = *offset_io;
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
        if (eq != NULL)
        {
            char   kname[128];
            size_t kn = (size_t) (eq - kv);
            if (kn < sizeof(kname))
            {
                memcpy(kname, kv, kn);
                kname[kn] = '\0';
                setenv(kname, eq + 1, 1);
                if (strcmp(kname, "MILK_SHM_DIR") == 0 && strcmp(dcshmdir, eq + 1) != 0)
                {
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
                }
            }
        }
    }
    *offset_io = off;
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

    char **argv = (char **) calloc(argc + 1, sizeof(char *));
    size_t lpos = 0;
    for (uint32_t i = 0; i < argc && off < len; i++)
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

    unpack_call_env(payload, len, &off);

    if (off + sizeof(uint32_t) <= len)
    {
        uint32_t nvars = 0;
        memcpy(&nvars, payload + off, sizeof(uint32_t));
        off += (uint32_t) sizeof(uint32_t);
        milk_server_sync_bash_vars(payload, len, nvars, &off);
    }

    long          snap_nmax    = 0;
    dcvar_snap_t *snap         = milk_server_snapshot_dcvar(&snap_nmax);
    unsigned int  nbcmd_before = data.NBcmd;

    call_watcher_ctx_t wctx;
    pthread_t          wtid;
    int                has_watcher = 0;
    if (listen_fd >= 0)
    {
        wctx.client_fd = client_fd;
        wctx.listen_fd = listen_fd;
        wctx.main_tid  = pthread_self();
        wctx.active    = 1;
        if (pthread_create(&wtid, NULL, call_watcher_thread, &wctx) == 0)
        {
            has_watcher = 1;
        }
    }

    int rc = milk_server_dispatch_argv((int) argc, argv);

    if (has_watcher)
    {
        wctx.active = 0;
        pthread_join(wtid, NULL);
    }

    fflush(stdout);
    fflush(stderr);

    milk_server_send_modified_dcvars(client_fd, snap, snap_nmax);
    free(snap);

    /* If a module was dynamically loaded (@mload / @soload), send DEFCMD */
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
    return rc;
}
