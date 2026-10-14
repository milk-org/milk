// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    milk_runnable.h
 * @brief   MILK_RUNNABLE: one descriptor for a CLI command and/or an fpsexec (engine tier)
 *
 * A MILK_RUNNABLE gathers everything needed to invoke a unit of computation from
 *  - the milk CLI: the module lists it in the array given to MILK_MODULE_RUNNABLES(); the CLI core
 *    registers it after dlopen() (module code never calls RegisterCLIcmd);
 *  - a standalone executable: EXPORT_FPSEXEC(prefix, "id", unit) marks it; CMake generates a
 *    minimal main() that calls milk_runnable_main().
 *
 * Source files using it are compiled ONCE (with -DMILK_NO_CLI) and never include CLIcore.h directly.
 * Several runnables may live in one file: every runnable owns its own parameter table, statics and
 * compute function, and nothing is named implicitly.
 *
 * Declaration pattern (see plugins/test-src/runnabletest):
 * @code
 *   #define MYPARAMS(X) X(".gain", &gain, FPTYPE_FLOAT32, 1, FPFLAG_DEFAULT_INPUT, "gain")
 *   MILK_PARAMS_DEFINE(my, MYPARAMS);                     // my_bind[], my_farg[], my_rt
 *   static errno_t my_compute(const MILK_RUNNABLE *self); // uses MILK_PROCINFO_*(self)
 *   const MILK_RUNNABLE runnable_my = {
 *       .kind = MILK_RUNNABLE_FPS, .app = { .fps_name = "my", .cmdkey = "my", .description = "..." },
 *       .srcfile = __FILE__, MILK_PARAMS_REF(my), .compute = my_compute,
 *   };
 *   EXPORT_FPSEXEC(milk, "my", runnable_my);
 * @endcode
 *
 * WARNING / TODO (milk-perfbench): perfbench reads the MILK_BUILD tag with `strings <exe>`. With
 * compile-once modules the exe holds only the generated stub, so the tag (PGO/LTO/OPT flags) no
 * longer describes the compute code, which lives in the module library. Until perfbench is taught
 * to query the module's build flags, tags reported for new-style fpsexecs must not be trusted.
 */

#ifndef MILK_RUNNABLE_H
#define MILK_RUNNABLE_H

#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#include <sys/types.h>

#include "milk_config.h"

/* Same include order as CLIcore.h / CLIcore_standalone.h. Sources compiled with MILK_NO_CLI (the
 * single compile of a module) get the CLI-free stub header, as legacy standalone sources do. */
#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#endif
#include "libmilkdata/milkdata.h"

#include "ImageStreamIO/ImageStreamIO.h"
#include "ImageStreamIO/ImageStruct.h"

#include "fps.h"
#include "computeunit.h"
#include "processtools.h"
#include "timeutils.h"

#include "milkDebugTools.h"
#include "milk_types.h"

#include "fps_procinfo_macros.h"
#include "milk_module_info.h"


/** @brief What a MILK_RUNNABLE provides. */
typedef enum
{
    /** CLI-only command: .farg + .cli_func, arguments parsed by the CLI core */
    MILK_RUNNABLE_CLI = 1,
    /** FPS-based unit: .bindings + .farg + .compute; usable from CLI and as fpsexec */
    MILK_RUNNABLE_FPS = 2
} MILK_RUNNABLE_KIND;


/**
 * @brief Mutable per-runnable state (the MILK_RUNNABLE itself is const).
 *
 * Filled by milk_runnable_prepare(). cmddata.cmdsettings points to cmdsettings_default until the CLI
 * core re-points it to its own per-command settings.
 */
typedef struct MILK_RUNNABLE_RT_
{
    CLICMDDATA  cmddata;
    CMDSETTINGS cmdsettings_default;
} MILK_RUNNABLE_RT;


/** @brief Descriptor of a CLI command and/or fpsexec. Define as `const`, with external linkage. */
typedef struct MILK_RUNNABLE_
{
    MILK_RUNNABLE_KIND kind;

    /** fps_name (FPS shm name), cmdkey (CLI keyword), description (-h1 output) */
    FPS_APP_INFO app;

    /** __FILE__ of the definition */
    const char *srcfile;

    /** Extra CLICMDFLAG_* bits (e.g. CLICMDFLAG_PROCINFO) */
    uint64_t flags;

    /** CLI argument definitions */
    CLICMDARGDEF *farg;
    int           nbarg;

    /** FPS bindings (MILK_RUNNABLE_FPS) */
    FPS_CLI_BINDING *bindings;
    int              nb_bindings;

    /** Compute function (MILK_RUNNABLE_FPS): runs the processinfo loop */
    errno_t (*compute)(const struct MILK_RUNNABLE_ *self);

    /** Optional custom configuration check (MILK_RUNNABLE_FPS); reads dcfpsptr */
    errno_t (*confcheck)(void);

    /** CLI function (MILK_RUNNABLE_CLI), called once farg values are linked */
    errno_t (*cli_func)(void);

    /** Mutable state, see MILK_RUNNABLE_RT_DEFINE() */
    MILK_RUNNABLE_RT *rt;
} MILK_RUNNABLE;


/** @brief Define the mutable state block `name##_rt` of a runnable. */
#define MILK_RUNNABLE_RT_DEFINE(name) static MILK_RUNNABLE_RT name##_rt

/**
 * @brief Generate name##_bind[], name##_farg[] and name##_rt from an X-macro parameter table.
 *
 * @param name    Identifier prefix, unique in the file
 * @param PARAMS  X-macro: X(keyword, ptr, type, is_primary, fpflag, descr), see fps_cli_binding.h
 */
#define MILK_PARAMS_DEFINE(name, PARAMS)                              \
    static FPS_CLI_BINDING name##_bind[] = { PARAMS(FPS_X_BINDING) }; \
    static CLICMDARGDEF    name##_farg[] = { PARAMS(FPS_X_FARG) };    \
    MILK_RUNNABLE_RT_DEFINE(name)

/** @brief Designated initializers wiring MILK_PARAMS_DEFINE(name, ...) into a MILK_RUNNABLE. */
#define MILK_PARAMS_REF(name)                                                                     \
    .bindings = name##_bind, .nb_bindings = (int) (sizeof(name##_bind) / sizeof(name##_bind[0])), \
    .farg = name##_farg, .nbarg = (int) (sizeof(name##_farg) / sizeof(name##_farg[0])),           \
    .rt = &name##_rt


/**
 * @brief Mark a runnable as fpsexec; CMake generates `<prefix>-fpsexec-<id>`.
 *
 * Must stand alone on a line in a module source file, with a bare prefix token (milk or cacao), a
 * string literal id and the (non-static) runnable symbol:
 *
 *     EXPORT_FPSEXEC(milk, "myunit", runnable_myunit);
 *
 * CMake scans module sources for this line at configure time (cmake/milk_runnable.cmake).
 */
#define EXPORT_FPSEXEC(prefix, id, unit)                                             \
    _Static_assert(sizeof(id) > 1, "EXPORT_FPSEXEC: id must be a non-empty string"); \
    extern const MILK_RUNNABLE unit

/**
 * @brief Declare a runnable that may be missing from the module (its file dropped by an unmet
 *        MILK_CMAKE_MANDATE_*); its address is NULL then, and the CLI core skips it.
 *
 * See docs/developer/dependency_system.md.
 */
#define MILK_RUNNABLE_WEAK_DECL(unit) extern const MILK_RUNNABLE unit __attribute__((weak))


/** @brief Processinfo loop macros for a runnable's compute function, see INSERT_STD_PROCINFO_*. */
#define MILK_PROCINFO_INIT(self) _MILK_PROCINFO_INIT_IMPL((self)->rt->cmddata)
#define MILK_PROCINFO_LOOPSTART(self) _MILK_PROCINFO_LOOPSTART_IMPL((self)->rt->cmddata)
#define MILK_PROCINFO_END(self) _MILK_PROCINFO_END_IMPL((self)->rt->cmddata)


/**
 * @brief Fill r->rt->cmddata (key, description, args, flags, settings) from the descriptor.
 *
 * Idempotent; called by milk_runnable_main() and by the CLI core before registration.
 */
void milk_runnable_prepare(const MILK_RUNNABLE *r);

/**
 * @brief Make @p r the runnable served by milk_runnable_compute_thunk().
 *
 * @return Previously current runnable, to be restored by the caller.
 */
const MILK_RUNNABLE *milk_runnable_set_current(const MILK_RUNNABLE *r);

/** @brief fps_compute_fn calling compute(self) of the current runnable (internal glue). */
errno_t milk_runnable_compute_thunk(void);

/** @brief Run an FPS runnable through the generic CLI lifecycle (CLI core only). */
errno_t milk_runnable_cli_call(const MILK_RUNNABLE *r);

/**
 * @brief main() body of an fpsexec: parses the standalone command line and drives the FPS
 *        lifecycle (fpsinit, confstart, runstart...) of runnable @p r.
 */
int milk_runnable_main(int argc, char *argv[], const MILK_RUNNABLE *r);

/**
 * @brief Entry used by the generated main() stub.
 *
 * Single seam for future experiments (inlining milk_runnable_main, static dispatch...).
 */
#define MILK_FPSEXEC_MAIN(argc, argv, unit) milk_runnable_main((argc), (argv), (unit))

#endif /* MILK_RUNNABLE_H */
