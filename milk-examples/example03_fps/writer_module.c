// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file writer_module.c
 * @brief Milk CLI module integration for the pattern generator writer.
 */

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "writer.h"

/**
 * @brief CLI argument definition array.
 * Uses the WRITER_PARAMS X-Macro from writer.h.
 */
static CLICMDARGDEF farg[] = {
#define X_CLI_DEF(cli_type, fps_type, c_type, key, descr, def_str, def_val, ptr_addr, cli_flags) \
    { cli_type, key, descr, def_str, cli_flags, (void **) ptr_addr, NULL },
    WRITER_PARAMS(X_CLI_DEF)
#undef X_CLI_DEF
};

/** @brief Custom validation called during CLI configuration. */
static errno_t customCONFcheck(void)
{
    writer03_validate();
    return RETURN_SUCCESS;
}

MILK_RUNNABLE_RT_DEFINE(writer03);

extern const MILK_RUNNABLE runnable_example03_writer03;

/** @brief Called by the CLI core once the arguments are linked to the global parameter pointers. */
static errno_t writer03_func(void)
{
    const MILK_RUNNABLE *self = &runnable_example03_writer03;

    DEBUG_TRACE_FSTART();

    // Resolve/Create output stream
    IMGID outimg         = imgid_make_from_name(out_name_ptr);
    outimg.mdt->naxis    = 2;
    outimg.mdt->size[0]  = *width_ptr;
    outimg.mdt->size[1]  = *height_ptr;
    outimg.mdt->datatype = _DATATYPE_FLOAT;
    imcreateIMGID(&outimg);

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    // Core shared computation
    writer03_compute(dcfpsptr, processinfo, outimg.im);

    // Update shared memory
    processinfo_update_output_stream(processinfo, outimg.im, NULL);

    // Control loop frequency (100 Hz)
    usleep(10000);

    MILK_PROCINFO_END(self)

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_example03_writer03 = {
    .kind      = MILK_RUNNABLE_CLI,
    .app       = { .fps_name    = "writer03",
                   .cmdkey      = "writer03",
                   .description = "writer03 example with FPS" },
    .srcfile   = __FILE__,
    .farg      = farg,
    .nbarg     = (int) (sizeof(farg) / sizeof(farg[0])),
    .confcheck = customCONFcheck,
    .cli_func  = writer03_func,
    .rt        = &writer03_rt,
};
