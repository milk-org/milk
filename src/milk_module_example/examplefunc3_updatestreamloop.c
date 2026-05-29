/**
 * @file    updatestreamloop.c
 * @brief   simple procinfo+fps example - brief, no comments, uses macros
 *
 * Example 3
 * Demonstates function that updates a stream
 */

#include "CLIcore.h"
#include "fps.h"

#include "COREMOD_memory/COREMOD_memory.h"

static FPS_APP_INFO FPS_app_info = { .fps_name    = "streamupdate",
                                     .cmdkey      = "streamupdate",
                                     .description = "update stream" };

// Variables local to this translation unit
static char inimname[FUNCTION_PARAMETER_STRMAXLEN] = "";

#define FPS_PARAMS(X) \
    X(".in_sname", inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input stream")


static FPS_CLI_BINDING my_bindings[] = { FPS_PARAMS(FPS_X_BINDING) };

static const int nb_bindings = sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = { FPS_PARAMS(FPS_X_FARG) };

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata = {
#else
static CLICMDDATA CLIcmddata = {
#endif
    "", "", CLICMD_FIELDS_DEFAULTS
};

FPS_CMDSETTINGS_INIT(dft, CLIcmddata, FPS_app_info)


// Wrapper function, used by all CLI calls
static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID img = imgid_make_from_name(inimname);
    resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    // Notify that the image is being changed.
    // This is required prior to modifying image content so that consumers can be informed.
    img.md->write = 1;
    if (img.ID == -1)
    {
        return RETURN_FAILURE;
    }

    // Insert code, or function(s) that perform operation(s) on image
    // If the code is very brief, it can be insterted right here, otherwise
    // it can be in a function, which may be made visible/accessible outside of this translation unit
    // if the function needs to be used outside of this call.

    // Call this to notify consumers that the image has been updated
    processinfo_update_output_stream(processinfo, img.im, NULL);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

// Register function in CLI
errno_t CLIADDCMD_milk_module_example__updatestreamloop()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);

    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
#endif

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)
#endif
