/**
 * @file image_keyword_add.c
 * @brief Image keyword add module
 */

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include <stdlib.h>

/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */
static FPS_APP_INFO FPS_app_info = { .fps_name    = "imkwadd",
                                     .cmdkey      = "imkwadd",
                                     .description = "Add or update a keyword in an image" };

/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */
static char param_inimname[FUNCTION_PARAMETER_STRMAXLEN] = "im1";
static char param_kwname[FUNCTION_PARAMETER_STRMAXLEN]   = "KW1";
static char param_kwtype[FUNCTION_PARAMETER_STRMAXLEN]   = "D";
static char param_kwval[FUNCTION_PARAMETER_STRMAXLEN]    = "1.234";
static char param_comment[FUNCTION_PARAMETER_STRMAXLEN]  = "keyword comment";

/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */
#define FPS_PARAMS(X)                                                                            \
    X(".in_name", param_inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image")     \
    X(".kwname", param_kwname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "keyword name")           \
    X(".kwtype", param_kwtype, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "keyword type (L, D, S)") \
    X(".kwval", param_kwval, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "keyword value")            \
    X(".comment", param_comment, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "keyword comment")

/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */
static MILK_HOT errno_t fpsexec()
{
    char type = param_kwtype[0];
    if (type == 'L' || type == 'l')
    {
        long val = atol(param_kwval);
        image_write_keyword_L(param_inimname, param_kwname, val, param_comment);
    }
    else if (type == 'D' || type == 'd')
    {
        double val = atof(param_kwval);
        image_write_keyword_D(param_inimname, param_kwname, val, param_comment);
    }
    else if (type == 'S' || type == 's')
    {
        image_write_keyword_S(param_inimname, param_kwname, param_kwval, param_comment);
    }
    else
    {
        PRINT_ERROR("Invalid keyword type '%c'. Must be L, D, or S.", type);
        return RETURN_FAILURE;
    }
    return RETURN_SUCCESS;
}

/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */
FPS_V2_SECTION5(FPS_PARAMS)

/* ================================================================
 * 6.  COMPUTE WRAPPER (processinfo loop support)
 * ============================================================= */
static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    INSERT_STD_PROCINFO_COMPUTEFUNC_START fpsexec();

    INSERT_STD_PROCINFO_COMPUTEFUNC_END return RETURN_SUCCESS;
}

/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */
#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_COREMOD_memory__image_keyword_add()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC return RETURN_SUCCESS;
}
#endif

/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */
#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)
#endif
