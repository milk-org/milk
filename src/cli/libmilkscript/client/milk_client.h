// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_client.h
 * @brief The client: one call per keyword or request that a shell sends to its session
 *        (milk-cli --server).
 *
 * Purpose:     the bash builtins and milk-call run module commands and keywords through one API,
 *              whether the work is done in the calling process or by the session.
 * Does:        declares milk_client_call() for module commands, one call per keyword (@get,
 *              @set, @calc, @test, @exists, @waitfor, @waitany, @sleep, @args, @grab, @session,
 *              @log, @ls), completion and shutdown; the hooks (milk_client_hooks_t) through
 *              which a call reads and sets the caller's variables, defines command functions
 *              and runs shell code; MILK_CLIENT_RC_TRANSPORT, the status (255) when the session
 *              cannot be reached.
 * Used by:     the builtins of libmilkbash.so (milk_bash_builtins.c, milk_bash_hooks.c), milk-call
 *              (milk_call_main.c); milk_ipc_common.h, milk_srvlog.h and milk_bash_defer.h (the
 *              hooks).
 * See also:    milk_client_ops.c, milk_client_stream.c, milk_client_args.c (the calls),
 *              milk_client_internal.h and milk_client_conn.c (the connection);
 *              docs/guide/sessions.md, "Using milk from Plain Bash".
 */

#ifndef MILK_CLIENT_H
#define MILK_CLIENT_H

#ifdef __cplusplus
extern "C"
{
#endif

/** Exit status of a client call that could not reach the server or lost it (distinct from "false"). */
#define MILK_CLIENT_RC_TRANSPORT 255

typedef struct
{
    const char *(*get_var)(const char *name, void *ctx);
    void (*set_var)(const char *name, const char *val, void *ctx);
    void (*def_cmd)(const char *cmdname, void *ctx);
    void (*on_sigint)(void *ctx);
    /* run shell code in the caller (NULL: written to MILK_CALL_EVAL_FD) */
    void (*eval)(char *script, void *ctx);
    void *ctx;
} milk_client_hooks_t;

/* milk_client_call() flags: ASYNC runs in a fork of the server, while it goes on; FRONTEND goes
 * to the session of the front end (milk-cli or milk-script) that runs this bash, named by
 * _MILK_FRONTEND_SOCK, whichever session the bash uses; NOWAIT fails at once while another
 * shell's command runs, rather than wait for it */
#define MILK_CALL_ASYNC 1
#define MILK_CALL_FRONTEND 2
#define MILK_CALL_NOWAIT 4

/**
     * @brief Invoke a registered milk command (`@<cmd> args...`) with MILK_CALL_* @flags.
     */
int milk_client_call(int argc, char **argv, int flags, const milk_client_hooks_t *hooks);

/**
     * @brief Query a milk entity (`s.<stream>.<field>`, `f.<fps>.<param>`, `p.<proc>.<field>`).
     *
     * If @p varname is non-NULL, binds the result directly to @p varname via
     * `hooks->set_var` instead of printing to stdout.
     */
int milk_client_get_buf(const char                *spec,
                        char                      *out,
                        size_t                     outsz,
                        const milk_client_hooks_t *hooks);

int milk_client_get(const char *spec, const char *varname, const milk_client_hooks_t *hooks);

/**
     * @brief `@get [--json|-j] [-v var | -a arr | -A assoc] <entity>`: print, or bind to a shell variable/array.
     */
int milk_client_get_argv(int argc, char **argv, const milk_client_hooks_t *hooks);

/**
     * @brief Set a milk FPS parameter or stream FITS keyword.
     */
int milk_client_set(const char                *spec,
                    const char                *val,
                    const char                *comment,
                    const milk_client_hooks_t *hooks);

/**
     * @brief Set one or more milk FPS parameters or stream FITS keywords (`@set`).
     */
int milk_client_set_argv(int argc, char **argv, const milk_client_hooks_t *hooks);

/** milk_client_calc_argv() - `@calc [-v a[,b...]] [-a arr] <expr...>`: -v binds the values of the last
 * statements (an assignment's value is the assigned value), -a the pixels of an image result. */
int milk_client_calc_argv(int argc, char **argv, const milk_client_hooks_t *hooks);

/**
     * @brief Evaluate a `@test` boolean condition (`0` = true, `1` = false, `2` = error).
     */
int milk_client_test(const char *expr, const milk_client_hooks_t *hooks);

/**
     * @brief Check whether a milk entity exists (`check_running=0`) or is actively running (`1`).
     */
int milk_client_exists(const char *spec, int check_running, const milk_client_hooks_t *hooks);

/**
     * @brief Wait for stream frames or an entity condition (`@waitfor`).
     */
int milk_client_waitfor(int argc, char **argv, const milk_client_hooks_t *hooks);

/**
     * @brief Wait for any of the given streams to advance its frame counter (`@waitany`).
     */
int milk_client_waitany(int argc, char **argv, const milk_client_hooks_t *hooks);

/**
 * milk_client_sleep_argv() - `@sleep SECONDS`, or `@sleep --every T|--hz F [--late L] VAR`: sleep to the
 * CLOCK_MONOTONIC deadline in VAR (ns; unset: one period from now), then advance VAR by a period,
 * skipping the ticks already missed (their count goes to L). 130 if Ctrl-C ended the sleep.
 */
int milk_client_sleep_argv(int argc, char **argv, const milk_client_hooks_t *hooks);

/**
 * milk_client_args() - `@args "name:type[=default][:help]"... -- VALUE...`: check the values as module
 * commands check theirs, then bind the names in the caller (`local` if @local). @fname names the caller
 * in messages. 0 bound, 1 after -h (help printed), 2 on a usage error (reported).
 */
int milk_client_args(int                        argc,
                     char                     **argv,
                     const char                *fname,
                     int                        local,
                     const milk_client_hooks_t *hooks);

/**
     * @brief Synchronously capture N consecutive stream frames into a 3D cube (`@grab`).
     */
int milk_client_grab(int argc, char **argv, const milk_client_hooks_t *hooks);

/**
     * @brief Query completion candidates from the server.
     */
int milk_client_complete(const char                *prefix,
                         const char                *comp_type,
                         const milk_client_hooks_t *hooks);

/**
     * @brief Shut down the session's `milk-cli --server`.
     */
int milk_client_shutdown(const milk_client_hooks_t *hooks);

/**
     * @brief Manage sessions (`@session status|ls|new|use|leave|who|end|prune|save`).
     */
int milk_client_session(int argc, char **argv, const milk_client_hooks_t *hooks);

/**
     * @brief Inspect or search the session's SHM command & output log (`@log`).
     */
int milk_client_log(int argc, char **argv, const milk_client_hooks_t *hooks);

/**
     * @brief Unified entity discovery (`@ls`).
     */
int milk_client_ls(int argc, char **argv, const milk_client_hooks_t *hooks);

#ifdef __cplusplus
}
#endif

/* A command of the client by name: the bash builtin milk<name> (and the keyword @<name>) and
 * `milk-call <name>` run the same function, on the same arguments */
typedef struct
{
    const char *name;
    int (*fn)(int argc, char **argv, const milk_client_hooks_t *hooks);
} milk_client_cmd_t;

/** milk_client_cmd() - the client command called @name (`get`, `calc`, `waitfor`, ...), or NULL. */
const milk_client_cmd_t *milk_client_cmd(const char *name);

#endif /* MILK_CLIENT_H */
