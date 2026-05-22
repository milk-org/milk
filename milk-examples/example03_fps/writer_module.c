/**
 * @file writer_module.c
 * @brief Milk CLI module integration for the pattern generator writer.
 */

#include "CLIcore.h"
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
static errno_t customCONFcheck()
{
    writer03_validate();
    return RETURN_SUCCESS;
}

/** @brief Command metadata definition. */
static CLICMDDATA CLIcmddata = { "writer03", "writer03 example with FPS", CLICMD_FIELDS_DEFAULTS };

/** @brief Displays detailed help when the user types 'writer03 -h' in the CLI. */
static errno_t help_function()
{
    if (data.fpsptr != NULL && data.fpsptr->md != NULL)
    {
        printf("%s\n", data.fpsptr->md->helptext);
    }
    else
    {
        printf("Writer03: Pattern generation example.\n");
    }
    return RETURN_SUCCESS;
}

/**
 * @brief Compute function wrapper for the Milk CLI.
 */
static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    // Resolve/Create output stream
    IMGID outimg    = imgid_make_from_name(out_name_ptr);
    outimg.naxis    = 2;
    outimg.size[0]  = *width_ptr;
    outimg.size[1]  = *height_ptr;
    outimg.datatype = _DATATYPE_FLOAT;
    imcreateIMGID(&outimg);

    // Standard ProcessInfo Loop Sequence
    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    // Core shared computation
    writer03_compute(data.fpsptr, processinfo, outimg.im);

    // Update shared memory
    processinfo_update_output_stream(processinfo, outimg.im, NULL);

    // Control loop frequency (100 Hz)
    usleep(10000);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

// Generate standard FPS integration functions
INSERT_STD_CLIfunction

    /**
 * @brief Registers the 'writer03' command with the Milk framework.
 * Called by the module initializer in example03fps_module.c.
 */
    errno_t
    CLIADDCMD_writer03()
{
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
