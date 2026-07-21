// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "psf_internal.h"

/* ================================================================== */
/* ================================================================== */
/*            MODULE INFO                                             */
/* ================================================================== */
/* ================================================================== */

// module default short name
// all CLI calls to this module functions will be <shortname>.<funcname>
// if set to "", then calls use <funcname>
#define MODULE_SHORTNAME_DEFAULT "psf"

// Module short description
#define MODULE_DESCRIPTION "Point Spread Function analysis"

//extern struct DATA data;

double FWHM_MEASURED;

/* ================================================================== */
/* ================================================================== */
/*            INITIALIZE LIBRARY                                      */
/* ================================================================== */
/* ================================================================== */

// Module initialization macro in CLIcore.h
// macro argument defines module name for bindings
//
INIT_MODULE_LIB(psf)

/* ================================================================== */
/* ================================================================== */
/*            COMMAND LINE INTERFACE (CLI) FUNCTIONS                  */
/* ================================================================== */
/* ================================================================== */

errno_t PSF_sequence_measure(const char *IDin_name, float PSFsizeEst, const char *outfname);

static char   psm_in[FUNCTION_PARAMETER_STRMAXLEN]  = "imc";
static double psm_size                              = 20.0;
static char   psm_out[FUNCTION_PARAMETER_STRMAXLEN] = "outimc.txt";

static FPS_APP_INFO FPS_app_info_psm = {
    .fps_name         = "psfseqmeas",
    .cmdkey           = "psfseqmeas",
    .description      = "measure PSF sequence",
    .description_long = "Measure PSF properties from a sequence of images. Computes centroid, "
                        "FWHM, Strehl ratio, and encircled energy."
};

#define FPS_PARAMS_PSM(X)                                                                   \
    X(".in_name", psm_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image cube")   \
    X(".psfsize", &psm_size, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "estimated PSF size") \
    X(".out_name", psm_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output file")

#include "fps.h"

static FPS_CLI_BINDING psm_bindings[]  = { FPS_PARAMS_PSM(FPS_X_BINDING) };
static const int       psm_nb_bindings = sizeof(psm_bindings) / sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF    farg[]          = { FPS_PARAMS_PSM(FPS_X_FARG) };
static CLICMDDATA      CLIcmddata      = { "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS     psm_cms         = { 0 };

static __attribute__((constructor)) void init_psm_cms(void)
{
    strncpy(CLIcmddata.key, FPS_app_info_psm.cmdkey, sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description, FPS_app_info_psm.description,
            sizeof(CLIcmddata.description) - 1);
    CLIcmddata.nbarg         = sizeof(farg) / sizeof(CLICMDARGDEF);
    CLIcmddata.funcfpscliarg = farg;
    CLIcmddata.flags         = CLICMDFLAG_FPS;
    if (CLIcmddata.cmdsettings == NULL)
    {
        CLIcmddata.cmdsettings = &psm_cms;
    }
}

static errno_t psm_compute(void)
{
    PSF_sequence_measure(psm_in, (float) psm_size, psm_out);
    return RETURN_SUCCESS;
}

static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info_psm, farg, &CLIcmddata, psm_bindings,
                                        psm_nb_bindings, psm_compute);
}

static errno_t init_module_CLI()
{
    safe_fps_fill_farg_examples(farg, psm_bindings, psm_nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
