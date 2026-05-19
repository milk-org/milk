/**
 * @file    fps_cli_init.h
 * @brief   Initialize FPS entries from a bindings array
 */

#ifndef FPS_CLI_INIT_H
#define FPS_CLI_INIT_H

#include "fps.h"
#include "fps_cli_binding.h"

/**
 * @brief Initialize FPS entries from bindings array.
 *
 * Iterates through the bindings and creates FPS entries
 * with the provided metadata and current local variable
 * values.
 *
 * @param fps         Target FPS structure
 * @param cmdkey      Command key string
 * @param description Human-readable description
 * @param bindings    Array of parameter bindings
 * @param nb_b        Number of bindings
 * @return            RETURN_SUCCESS on success
 */
errno_t fps_init_from_bindings(
    FPS             *fps,
    const char      *cmdkey,
    const char      *description,
    FPS_CLI_BINDING *bindings,
    int                        nb_b
);


#endif /* FPS_CLI_INIT_H */
