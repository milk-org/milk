/**
 * @file    fps_lifecycle.h
 * @brief   Generic FPS lifecycle functions
 *
 * Provides generic init/conf/run/stop functions that
 * module-specific code delegates to. Each function handles
 * both local (_prefix) and shared-memory FPS modes.
 */

#ifndef FPS_LIFECYCLE_H
#define FPS_LIFECYCLE_H

#include "fps.h"
#include "fps_cli_binding.h"

/** Compute function signature */
typedef errno_t (*fps_compute_fn)(void);


/**
 * @brief Initialize an FPS (standalone lifecycle).
 *
 * For local FPS: allocates in-process memory.
 * For shared FPS: uses FPS_INIT_STD_PREAMBLE.
 *
 * @param fps_name    FPS name
 * @param app_info    Application identity
 * @param bindings    Parameter bindings
 * @param nb_b        Number of bindings
 * @param procinfo    If nonzero, add processinfo
 *                    entries to FPS
 * @return            0 on success, -1 on failure
 */
int fps_generic_init(
    const char      *fps_name,
    FPS_APP_INFO    *app_info,
    FPS_CLI_BINDING *bindings,
    int              nb_b,
    int              procinfo
);


/**
 * @brief Run FPS conf loop with custom check.
 *
 * Like fps_generic_conf, but calls confcheck_fn
 * on every loop iteration (if non-NULL).
 *
 * @param fps_name      FPS name
 * @param loop          Loop flag (1 = continuous)
 * @param confcheck_fn  Called each iteration
 * @return              0 on success
 */
int fps_generic_conf_cb(
    const char *fps_name,
    int         loop,
    errno_t   (*confcheck_fn)(void)
);


/**
 * @brief Run FPS configuration loop (standalone).
 *
 * For local FPS: prints skip message.
 * For shared FPS: uses FPS_CONF_STD_BODY.
 *
 * @param fps_name  FPS name
 * @param loop      Loop flag (1 = continuous)
 * @return          0 on success
 */
int fps_generic_conf(
    const char *fps_name,
    int         loop
);


/**
 * @brief Run the FPS computation (standalone).
 *
 * Connects to FPS, syncs parameters, sets up
 * processinfo if enabled, and calls compute_fn.
 *
 * @param fps_name    FPS name
 * @param app_info    Application identity
 * @param farg        CLICMDARGDEF array
 * @param bindings    Parameter bindings
 * @param nb_b        Number of bindings
 * @param compute_fn  Module computation function
 * @return            0 on success
 */
int fps_generic_run(
    const char      *fps_name,
    FPS_APP_INFO    *app_info,
    CLICMDARGDEF    *farg,
    FPS_CLI_BINDING *bindings,
    int              nb_b,
    fps_compute_fn   compute_fn
);


/**
 * @brief Stop the run process.
 *
 * @param fps_name  FPS name
 * @return          0 on success
 */
int fps_generic_runstop(const char *fps_name);


/**
 * @brief Stop the configuration process.
 *
 * @param fps_name  FPS name
 * @return          0 on success
 */
int fps_generic_confstop(const char *fps_name);


#endif /* FPS_LIFECYCLE_H */
