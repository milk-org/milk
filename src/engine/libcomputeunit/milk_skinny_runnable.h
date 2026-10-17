// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef MILK_SKINNY_RUNNABLE_H
#define MILK_SKINNY_RUNNABLE_H

#include "fps_types.h"
#include "fps_cli_binding.h"

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

#endif /* MILK_SKINNY_RUNNABLE_H */
