#include "CLIcore.h"
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"

// MILK_CMAKE_MANDATE_LAPACKE
#include "milk_blas_lapacke.h"

// 1.  FPS COMPONENT IDENTITY
static FPS_APP_INFO FPS_app_info = { .fps_name    = "mandlapacke",
                                     .cmdkey      = "mandlapacke",
                                     .description = "Dumb test for MILK_CMAKE_MANDATE_LAPACKE test",
                                     .description_long =
                                         "Dumb test for MILK_CMAKE_MANDATE_LAPACKE test" };
// 2.  LOCAL PARAMETER VARIABLES
static char some_string[FUNCTION_PARAMETER_STRMAXLEN] = "string";


// 3.  UNIFIED PARAMETER TABLE (X-Macro)
#define FPS_PARAMS(X)                                                                 \
    X(".some_string", some_string, FPTYPE_STRING_NOT_STREAM, 1, FPFLAG_DEFAULT_INPUT, \
      "Unused string")

// 4.  COMPUTATION LOGIC

// 5.  BINDINGS, FARG, AND CLI DATA
FPS_V2_SECTION5(FPS_PARAMS)

// 6.  COMPUTE WRAPPER
static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        printf("Lapacke function body...\n");
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

// 7.  MILK MODULE REGISTRATION
#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t COMPILERTEST_CLIADDCMD_MANDATE_LAPACKE()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC return RETURN_SUCCESS;
}
#endif

// 8.  STANDALONE ENTRY POINT
#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)
#endif
