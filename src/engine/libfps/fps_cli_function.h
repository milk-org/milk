/**
 * @file    fps_cli_function.h
 * @brief   Generic CLIfunction and CLIADDCMD for FPS modules
 *
 * Modules include this header to get safe_fps_*
 * wrappers.  The underlying function pointers are
 * stored as void* in fps_cli_function_registry.h
 * (part of milkfps, no CLI dependency).  milkfpsCLI
 * registers its implementations at load time.
 *
 * This header casts void* to proper function types.
 */

#ifndef FPS_CLI_FUNCTION_H
#define FPS_CLI_FUNCTION_H

#include "fps.h"
#include "fps_cli_binding.h"
#include "fps_cli_function_registry.h"
#include "libmilkdata/milkdata_clicmd.h"

/** Compute function signature */
typedef errno_t (*fps_compute_fn)(void);

/**
 * @brief Typed function pointer for
 *        fps_generic_CLIfunction.
 */
typedef errno_t (*fps_generic_CLIfunction_fn)(
    FPS_APP_INFO    *app_info,
    CLICMDARGDEF    *farg,
    CLICMDDATA      *CLIcmddata,
    FPS_CLI_BINDING *bindings,
    int             nb_b,
    fps_compute_fn   compute_fn
);

/**
 * @brief Typed function pointer for
 *        fps_fill_farg_examples.
 */
typedef void (*fps_fill_farg_examples_fn)(
    CLICMDARGDEF    *farg,
    FPS_CLI_BINDING *bindings,
    int              nb_b
);


/* ---- NULL-safe wrappers ---- */

/**
 * @brief NULL-safe call to fps_generic_CLIfunction.
 */
static inline errno_t
safe_fps_generic_CLIfunction(
    FPS_APP_INFO    *app_info,
    CLICMDARGDEF    *farg,
    CLICMDDATA      *cd,
    FPS_CLI_BINDING *bindings,
    int             nb_b,
    fps_compute_fn   compute_fn
)
{
    if(fps_generic_CLIfunction_ptr)
    {
        fps_generic_CLIfunction_fn fn =
            (fps_generic_CLIfunction_fn)
            fps_generic_CLIfunction_ptr;
        return fn(app_info, farg, cd,
                  bindings, nb_b, compute_fn);
    }
    return 0;
}

/**
 * @brief NULL-safe call to fps_fill_farg_examples.
 */
static inline void
safe_fps_fill_farg_examples(
    CLICMDARGDEF    *farg,
    FPS_CLI_BINDING *bindings,
    int              nb_b
)
{
    if(fps_fill_farg_examples_ptr)
    {
        fps_fill_farg_examples_fn fn =
            (fps_fill_farg_examples_fn)
            fps_fill_farg_examples_ptr;
        fn(farg, bindings, nb_b);
    }
}


#endif /* FPS_CLI_FUNCTION_H */
