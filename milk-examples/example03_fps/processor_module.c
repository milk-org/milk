// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file processor_module.c
 * @brief Milk CLI module integration for the ROI processor.
 *
 * Maps CLI arguments to the shared parameter pointers and provides the compute function called
 * by the CLI core.
 */

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "processor.h"

/**
 * @brief CLI argument definition array.
 *
 * Uses the PROCESSOR_PARAMS X-Macro from processor.h to generate the binding between CLI input
 * tokens and the global parameter pointers.
 */
static CLICMDARGDEF farg[] = {
#define X_CLI_DEF(cli_type, fps_type, c_type, key, descr, def_str, def_val, ptr_addr, cli_flags) \
    { cli_type, key, descr, def_str, cli_flags, (void **) ptr_addr, NULL },
    PROCESSOR_PARAMS(X_CLI_DEF)
#undef X_CLI_DEF
};

/** @brief Custom validation called during CLI configuration. */
static errno_t customCONFcheck(void)
{
    processor03_validate();
    return RETURN_SUCCESS;
}

MILK_RUNNABLE_RT_DEFINE(processor03);

extern const MILK_RUNNABLE runnable_example03_processor03;

/** @brief Called by the CLI core once the arguments are linked to the global parameter pointers. */
static errno_t processor03_func(void)
{
    const MILK_RUNNABLE *self = &runnable_example03_processor03;

    DEBUG_TRACE_FSTART();

    // Resolve input stream from name (pointer was set by CLI arg binding)
    IMGID inimg = imgid_make_from_name(in_name_ptr);
    resolveIMGID(&inimg, ERRMODE_WARN, dcimg, dcnimg);
    if (inimg.ID == -1)
    {
        return RETURN_FAILURE;
    }

    // Resolve/Create output stream
    IMGID outimg         = imgid_make_from_name(proc_out_name_ptr);
    outimg.mdt->naxis    = 2;
    outimg.mdt->size[0]  = *roi_size_ptr;
    outimg.mdt->size[1]  = *roi_size_ptr;
    outimg.mdt->datatype = _DATATYPE_FLOAT;
    imcreateIMGID(&outimg);

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    // Core computation logic shared with the standalone build
    processor03_compute(dcfpsptr, processinfo, inimg.im, outimg.im);

    // Standard output update: increment counters and post semaphores
    processinfo_update_output_stream(processinfo, outimg.im, inimg.im);

    MILK_PROCINFO_END(self)

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_example03_processor03 = {
    .kind      = MILK_RUNNABLE_CLI,
    .app       = { .fps_name    = "processor03",
                   .cmdkey      = "processor03",
                   .description = "processor03 example with FPS" },
    .srcfile   = __FILE__,
    .farg      = farg,
    .nbarg     = (int) (sizeof(farg) / sizeof(farg[0])),
    .confcheck = customCONFcheck,
    .cli_func  = processor03_func,
    .rt        = &processor03_rt,
};
