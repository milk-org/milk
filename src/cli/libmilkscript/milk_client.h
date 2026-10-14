// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_client.h
 * @brief Client routines for communicating with milk-cli --server
 *
 * Shared by both the loadable bash builtin module (`libmilkbash.so`)
 * and the standalone helper executable (`milk-call`).
 */

#ifndef MILK_CLIENT_H
#define MILK_CLIENT_H

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct
{
    const char *(*get_var)(const char *name, void *ctx);
    void (*set_var)(const char *name, const char *val, void *ctx);
    void (*def_cmd)(const char *cmdname, void *ctx);
    void (*on_sigint)(void *ctx);
    void *ctx;
} milk_client_hooks_t;

/**
     * @brief Invoke a registered milk command (`@<cmd> args...`).
     */
int milk_client_call(int argc, char **argv, int async_flag, const milk_client_hooks_t *hooks);

/**
     * @brief Query a milk entity (`s.<stream>.<field>`, `f.<fps>.<param>`, `p.<proc>.<field>`).
     *
     * If @p varname is non-NULL, binds the result directly to @p varname via
     * `hooks->set_var` instead of printing to stdout.
     */
int milk_client_get(const char *spec, const char *varname, const milk_client_hooks_t *hooks);

/**
     * @brief Set a milk FPS parameter or stream FITS keyword.
     */
int milk_client_set(const char                *spec,
                    const char                *val,
                    const char                *comment,
                    const milk_client_hooks_t *hooks);

/**
     * @brief Evaluate a `@calc` expression on the persistent server.
     *
     * If @p varname is non-NULL, captures any printed scalar result into
     * @p varname via `hooks->set_var`.
     */
int milk_client_calc(const char *expr, const char *varname, const milk_client_hooks_t *hooks);

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

#ifdef __cplusplus
}
#endif

#endif /* MILK_CLIENT_H */
