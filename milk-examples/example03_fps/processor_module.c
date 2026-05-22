/**
 * @file processor_module.c
 * @brief Milk CLI module integration for the ROI processor.
 *
 * Maps CLI arguments to the shared FPS parameters and provides the
 * compute wrapper required by the Milk framework.
 */

#include "CLIcore.h"
#include "processor.h"

/**
 * @brief CLI argument definition array.
 *
 * Uses the PROCESSOR_PARAMS X-Macro from processor.h to automatically generate
 * the binding between CLI input tokens and the global parameter pointers.
 */
static CLICMDARGDEF farg[] = {
#define X_CLI_DEF(cli_type, fps_type, c_type, key, descr, def_str, def_val, ptr_addr, cli_flags) \
    { cli_type, key, descr, def_str, cli_flags, (void **) ptr_addr, NULL },
    PROCESSOR_PARAMS(X_CLI_DEF)
#undef X_CLI_DEF
};

/**
 * @brief Custom configuration check for the CLI module.
 *
 * This is called by the framework's FPSCONF implementation.
 * In CLI mode, the global variables (in_name_ptr etc) are already linked
 * to the FPS entries via the STD_FARG_LINKfunction macro.
 */
static errno_t customCONFcheck()
{
    processor03_validate();
    return RETURN_SUCCESS;
}

/** @brief Command metadata definition. */
static CLICMDDATA CLIcmddata = { "processor03", "processor03 example with FPS",
                                 CLICMD_FIELDS_DEFAULTS };

/** @brief Displays detailed help when the user types 'processor03 -h' in the CLI. */
static errno_t help_function()
{
    if (data.fpsptr != NULL && data.fpsptr->md != NULL)
    {
        printf("%s\n", data.fpsptr->md->helptext);
    }
    else
    {
        printf("Processor03: ROI extraction example.\n");
    }
    return RETURN_SUCCESS;
}

/**
 * @brief Compute function wrapper for the Milk CLI.
 *
 * This function is called by the FPSRUNfunction macro. It handles
 * image resolution and creation before calling the shared compute logic.
 */
static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    // Resolve input stream from name (pointer was set by CLI arg binding)
    IMGID inimg = imgid_make_from_name(in_name_ptr);
    resolveIMGID(&inimg, ERRMODE_WARN, data.image, data.NB_MAX_IMAGE);

    // Resolve/Create output stream
    IMGID outimg = imgid_make_from_name(proc_out_name_ptr);
    if (inimg.ID == -1)
    {
        return RETURN_FAILURE;
    }
    outimg.naxis    = 2;
    outimg.size[0]  = *roi_size_ptr;
    outimg.size[1]  = *roi_size_ptr;
    outimg.datatype = _DATATYPE_FLOAT;
    imcreateIMGID(&outimg);

    // Start the standard ProcessInfo loop sequence
    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    // Call the core computation logic shared with the standalone build
    processor03_compute(data.fpsptr, processinfo, inimg.im, outimg.im);

    // Standard output update: increment counters and post semaphores
    processinfo_update_output_stream(processinfo, outimg.im, inimg.im);

    // End the loop sequence
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

/**
 * @brief Macro to generate standard FPS integration functions:
 * - FPSCONFfunction: Handles parameter configuration.
 * - FPSRUNfunction:  Handles the main compute loop.
 * - CLIfunction:     The entry point called by the milk shell.
 */
INSERT_STD_CLIfunction

    /**
 * @brief Registers the 'processor03' command with the Milk framework.
 * Called by the module initializer in example03fps_module.c.
 */
    errno_t
    CLIADDCMD_processor03()
{
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
