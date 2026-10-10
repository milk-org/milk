// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    milk_module_info.h
 * @brief   Module descriptor exported by every milk module (engine tier)
 *
 * Defines MILK_MODULE_INFO, the symbol __milk_module_info read by the CLI after dlopen(), and the
 * MODULE_DEPS macro. Free of any CLI dependency so that module sources need not include CLIcore.h.
 *
 * The legacy MILK_MODULE() macro stays in CLIcore.h; new-style modules use MILK_MODULE_RUNNABLES().
 */

#ifndef MILK_MODULE_INFO_H
#define MILK_MODULE_INFO_H

#include "milk_config.h"
#include "milkDebugTools.h"

struct MILK_RUNNABLE_;

/**
 * @brief Declare module dependencies.
 *
 * Place this macro before MILK_MODULE(...) / MILK_MODULE_RUNNABLES(...) in the module's main .c
 * file. Arguments are loadnames (matching mload convention), e.g.:
 *
 *     MODULE_DEPS("milkfft", "milkimage_gen")
 *
 * If a module has no deps, omit this macro entirely.
 */
#ifndef MODULE_DEPS
#    define MODULE_DEPS(...)                                                                        \
        static const char *_module_deps[] = { __VA_ARGS__ };                                        \
        static const int   _module_ndeps  = (int) (sizeof(_module_deps) / sizeof(_module_deps[0])); \
        static const int   _module_deps_defined = 1
#endif

/**
 * @brief Per-module registration descriptor.
 *
 * Export as the symbol __milk_module_info from the module's main .c file.
 * load_sharedobj() reads it after dlopen() and drives the full registration
 * sequence without ever needing __attribute__((constructor)).
 *
 * Use MILK_MODULE() (legacy) or MILK_MODULE_RUNNABLES() to define and export it.
 */
typedef struct
{
    const char *name;              /**< module name token */
    const char *shortname_default; /**< MODULE_SHORTNAME_DEFAULT */
    const char *description;       /**< MODULE_DESCRIPTION */
    const char *source_file;       /**< __FILE__ */
    const char *package;           /**< PROJECT_NAME */
    int         version_major;
    int         version_minor;
    int         version_patch;
    const char *date_string; /**< __DATE__ */
    const char *time_string; /**< __TIME__ */
    /** Module CLI initializer (init_module_CLI), or NULL */
    errno_t (*reg_call)(void);
    /** NULL-sentinel dep load names, or NULL */
    const char **deps;
    int          mod_registered;
    int          constructor_called;
    /** Runnables exposed to the CLI (registered by the CLI core), or NULL */
    const struct MILK_RUNNABLE_ *const *runnables;
    int                                 n_runnables;
} MILK_MODULE_INFO;

/** @brief Shared initializer for MILK_MODULE_INFO (internal). */
#define MILK_MODULE_INFO_INIT_(modname, cli_reg_call, _deps, _runnables, _n_runnables) \
    { .name               = #modname,                                                  \
      .shortname_default  = MODULE_SHORTNAME_DEFAULT,                                  \
      .description        = MODULE_DESCRIPTION,                                        \
      .source_file        = __FILE__,                                                  \
      .package            = MILK_PROJECT_NAME,                                         \
      .version_major      = MILK_VERSION_MAJOR,                                        \
      .version_minor      = MILK_VERSION_MINOR,                                        \
      .version_patch      = MILK_VERSION_PATCH,                                        \
      .date_string        = __DATE__,                                                  \
      .time_string        = __TIME__,                                                  \
      .reg_call           = (cli_reg_call),                                            \
      .deps               = (_deps),                                                   \
      .mod_registered     = 0,                                                         \
      .constructor_called = 0,                                                         \
      .runnables          = (_runnables),                                              \
      .n_runnables        = (_n_runnables) }

/**
 * @brief Define and export the descriptor of a module made of MILK_RUNNABLE units.
 *
 * The CLI core registers every runnable of @p runnables_array; no init function is needed.
 * Requires MODULE_SHORTNAME_DEFAULT and MODULE_DESCRIPTION to be defined before use.
 *
 * @param modname           Module name token (unquoted)
 * @param _deps             NULL-sentinel dep array (see MODULE_DEPS), or NULL
 * @param runnables_array   Array of `const MILK_RUNNABLE *` (a real array, not a pointer)
 */
#define MILK_MODULE_RUNNABLES(modname, _deps, runnables_array)        \
    MILK_MODULE_INFO __milk_module_info =                             \
        MILK_MODULE_INFO_INIT_(modname, NULL, _deps, runnables_array, \
                               (int) (sizeof(runnables_array) / sizeof((runnables_array)[0])))

#endif /* MILK_MODULE_INFO_H */
