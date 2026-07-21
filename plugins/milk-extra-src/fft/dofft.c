// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    dofft.c
 * @brief   FPS registration and standalone entry
 *          for FFT operations
 *
 * The actual FFT implementations live in:
 *   - dofft_1d.c  (1D complex + real FFTs)
 *   - dofft_2d.c  (2D complex + real FFTs)
 */

#include "dofft_internal.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "dofft",
    .cmdkey      = "dofft",
    .description = "perform 2D complex FFT",
    .description_long =
        "Compute the Fast Fourier Transform (FFT) of 1D or 2D data using FFTW. Supports "
        "real-to-complex and complex-to-complex transforms, forward and inverse directions."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char    dofft_inimname[FUNCTION_PARAMETER_STRMAXLEN];
static char    dofft_outimname[FUNCTION_PARAMETER_STRMAXLEN];
static int32_t dofft_dir = 0;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                           \
    X(".in_name", dofft_inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,   \
      "input complex image")                                                    \
    X(".out_name", dofft_outimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, \
      "output complex image")                                                   \
    X(".dir", &dofft_dir, FPTYPE_INT32, 1, FPFLAG_DEFAULT_INPUT, "FFT direction")

// Forward declarations
imageID do1dfft(const char *in_name, const char *out_name);

imageID do1drfft(const char *in_name, const char *out_name);

imageID do2dfft(const char *in_name, const char *out_name);


/* =========================================
 *  CMD 2: do1Dfft (2 args)
 * ======================================= */

static char p_1dfft_in[FUNCTION_PARAMETER_STRMAXLEN]  = "in";
static char p_1dfft_out[FUNCTION_PARAMETER_STRMAXLEN] = "out";

static FPS_APP_INFO FPS_app_info_1dfft = {
    .fps_name    = "do1Dfft",
    .cmdkey      = "do1Dfft",
    .description = "perform 1D complex->complex FFT",
    .description_long =
        "Compute the Fast Fourier Transform (FFT) of 1D or 2D data using FFTW. Supports "
        "real-to-complex and complex-to-complex transforms, forward and inverse directions."
};

#define FPS_PARAMS_1DFFT(X)                                                                      \
    X(".in_name", p_1dfft_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input complex image") \
    X(".out_name", p_1dfft_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output complex image")

static CLICMDDATA CLIcmddata_1dfft = { "", "", CLICMD_FIELDS_NOPARAM };

FPS_CMDSETTINGS_INIT(1dfft, CLIcmddata_1dfft, FPS_app_info_1dfft)


static errno_t __attribute__((unused)) compute_1dfft()
{
    do1dfft(p_1dfft_in, p_1dfft_out);
    return RETURN_SUCCESS;
}


/* =========================================
 *  CMD 3: do1Drfft (2 args)
 * ======================================= */

static char p_1drfft_in[FUNCTION_PARAMETER_STRMAXLEN]  = "in";
static char p_1drfft_out[FUNCTION_PARAMETER_STRMAXLEN] = "out";

static FPS_APP_INFO FPS_app_info_1drfft = {
    .fps_name    = "do1Drfft",
    .cmdkey      = "do1Drfft",
    .description = "perform 1D real->complex FFT",
    .description_long =
        "Compute the Fast Fourier Transform (FFT) of 1D or 2D data using FFTW. Supports "
        "real-to-complex and complex-to-complex transforms, forward and inverse directions."
};

#define FPS_PARAMS_1DRFFT(X)                                                                   \
    X(".in_name", p_1drfft_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input real image") \
    X(".out_name", p_1drfft_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output complex image")

static CLICMDDATA CLIcmddata_1drfft = { "", "", CLICMD_FIELDS_NOPARAM };

FPS_CMDSETTINGS_INIT(1drfft, CLIcmddata_1drfft, FPS_app_info_1drfft)


static errno_t __attribute__((unused)) compute_1drfft()
{
    do1drfft(p_1drfft_in, p_1drfft_out);
    return RETURN_SUCCESS;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t compute_function()
{
    do2dfft(dofft_inimname, dofft_outimname);
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#ifndef FPS_STANDALONE
static errno_t __attribute__((unused)) CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

static FPS_CLI_BINDING b_1dfft[]  = { FPS_PARAMS_1DFFT(FPS_X_BINDING) };
static CLICMDARGDEF    fa_1dfft[] = { FPS_PARAMS_1DFFT(FPS_X_FARG) };

static errno_t CLIfunction_1dfft(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info_1dfft, fa_1dfft, &CLIcmddata_1dfft, b_1dfft,
                                        sizeof(b_1dfft) / sizeof(FPS_CLI_BINDING), compute_1dfft);
}

static FPS_CLI_BINDING b_1drfft[]  = { FPS_PARAMS_1DRFFT(FPS_X_BINDING) };
static CLICMDARGDEF    fa_1drfft[] = { FPS_PARAMS_1DRFFT(FPS_X_FARG) };

static errno_t CLIfunction_1drfft(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info_1drfft, fa_1drfft, &CLIcmddata_1drfft,
                                        b_1drfft, sizeof(b_1drfft) / sizeof(FPS_CLI_BINDING),
                                        compute_1drfft);
}

errno_t CLIADDCMD_milkfft__dofft()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC

    safe_fps_fill_farg_examples(fa_1dfft, b_1dfft, sizeof(b_1dfft) / sizeof(FPS_CLI_BINDING));
    {
        int cmdi                     = RegisterCLIcmd(CLIcmddata_1dfft, CLIfunction_1dfft);
        CLIcmddata_1dfft.cmdsettings = &data.cmd[cmdi].cmdsettings;
    }

    safe_fps_fill_farg_examples(fa_1drfft, b_1drfft, sizeof(b_1drfft) / sizeof(FPS_CLI_BINDING));
    {
        int cmdi                      = RegisterCLIcmd(CLIcmddata_1drfft, CLIfunction_1drfft);
        CLIcmddata_1drfft.cmdsettings = &data.cmd[cmdi].cmdsettings;
    }

    return RETURN_SUCCESS;
}
#endif


/* ================================================================
 * 4b. Standalone-friendly FFT step
 * ============================================================= */

static void MILK_HOT __attribute__((unused)) fpsexec(IMAGE *imgin, IMAGE *imgout, int dir)
{
    int naxes[2] = { (int) imgin->md[0].size[1], (int) imgin->md[0].size[0] };
    if (imgin->md[0].datatype == _DATATYPE_COMPLEX_FLOAT)
    {
        fftwf_plan plan = fftwf_plan_dft_2d(naxes[0], naxes[1], (fftwf_complex *) imgin->array.CF,
                                            (fftwf_complex *) imgout->array.CF, dir, FFTW_ESTIMATE);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
    }
    else if (imgin->md[0].datatype == _DATATYPE_COMPLEX_DOUBLE)
    {
        fftw_plan plan = fftw_plan_dft_2d(naxes[0], naxes[1], (fftw_complex *) imgin->array.CD,
                                          (fftw_complex *) imgout->array.CD, dir, FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);
    }
}


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)
#endif
