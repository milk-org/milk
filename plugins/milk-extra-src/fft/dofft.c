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

#include "milk_runnable.h"
#include "dofft_internal.h"


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

#define FPS_PARAMS_1DFFT(X)                                                                      \
    X(".in_name", p_1dfft_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input complex image") \
    X(".out_name", p_1dfft_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output complex image")

static MILK_HOT errno_t dofft_1dfft_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    do1dfft(p_1dfft_in, p_1dfft_out);
    return RETURN_SUCCESS;
}


/* =========================================
 *  CMD 3: do1Drfft (2 args)
 * ======================================= */

static char p_1drfft_in[FUNCTION_PARAMETER_STRMAXLEN]  = "in";
static char p_1drfft_out[FUNCTION_PARAMETER_STRMAXLEN] = "out";

#define FPS_PARAMS_1DRFFT(X)                                                                   \
    X(".in_name", p_1drfft_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input real image") \
    X(".out_name", p_1drfft_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output complex image")

static MILK_HOT errno_t dofft_1drfft_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    do1drfft(p_1drfft_in, p_1drfft_out);
    return RETURN_SUCCESS;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(dofft, FPS_PARAMS);
MILK_PARAMS_DEFINE(dofft_1dfft, FPS_PARAMS_1DFFT);
MILK_PARAMS_DEFINE(dofft_1drfft, FPS_PARAMS_1DRFFT);


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t dofft_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    do2dfft(dofft_inimname, dofft_outimname);
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  RUNNABLE DESCRIPTORS
 * ============================================================= */

const MILK_RUNNABLE runnable_fft_dofft = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name    = "dofft",
              .cmdkey      = "dofft",
              .description = "perform 2D complex FFT",
              .description_long =
                  "Compute the Fast Fourier Transform (FFT) of 1D or 2D data using FFTW. Supports "
                  "real-to-complex and complex-to-complex transforms, forward and inverse "
                  "directions." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(dofft),
    .compute = dofft_compute,
};
EXPORT_FPSEXEC(milk, "fft-dofft", runnable_fft_dofft);

const MILK_RUNNABLE runnable_fft_dofft_1dfft = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name    = "do1Dfft",
              .cmdkey      = "do1Dfft",
              .description = "perform 1D complex->complex FFT",
              .description_long =
                  "Compute the Fast Fourier Transform (FFT) of 1D or 2D data using FFTW. Supports "
                  "real-to-complex and complex-to-complex transforms, forward and inverse "
                  "directions." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(dofft_1dfft),
    .compute = dofft_1dfft_compute,
};

const MILK_RUNNABLE runnable_fft_dofft_1drfft = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name    = "do1Drfft",
              .cmdkey      = "do1Drfft",
              .description = "perform 1D real->complex FFT",
              .description_long =
                  "Compute the Fast Fourier Transform (FFT) of 1D or 2D data using FFTW. Supports "
                  "real-to-complex and complex-to-complex transforms, forward and inverse "
                  "directions." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(dofft_1drfft),
    .compute = dofft_1drfft_compute,
};
