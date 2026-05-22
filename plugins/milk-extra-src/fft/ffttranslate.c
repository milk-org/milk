/**
 * @file ffttranslate.c
 * @brief Translate image using FFT
 */

#include <math.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "fps.h"

#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "dofft.h"
#include "permut.h"

// Forward declaration
int fft_image_translate(const char *ID_name, const char *ID_out, double xtransl, double ytransl);

/* =========================================
 *  V2 PARAMS
 * ======================================= */

static char   p_in[FUNCTION_PARAMETER_STRMAXLEN]  = "im1";
static char   p_out[FUNCTION_PARAMETER_STRMAXLEN] = "im2";
static double p_xtransl                           = 2.3;
static double p_ytransl                           = -2.1;

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "transl",
    .cmdkey      = "transl",
    .description = "translate image via FFT",
    .description_long =
        "Translate (shift) an image by a sub-pixel offset using FFT phase multiplication. Applies "
        "a linear phase ramp in Fourier space for exact sub-pixel shifts."
};

#define FPS_PARAMS(X)                                                                   \
    X(".in_name", p_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image")      \
    X(".out_name", p_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")       \
    X(".xtransl", &p_xtransl, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "x translation") \
    X(".ytransl", &p_ytransl, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "y translation")

static FPS_CLI_BINDING my_bindings[] = { FPS_PARAMS(FPS_X_BINDING) };
static const int       nb_bindings   = sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = { FPS_PARAMS(FPS_X_FARG) };

static CLICMDDATA  CLIcmddata = { "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS cms        = { 0 };

static __attribute__((constructor)) void init_cms(void)
{
    strncpy(CLIcmddata.key, FPS_app_info.cmdkey, sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description, FPS_app_info.description, sizeof(CLIcmddata.description) - 1);
    if (CLIcmddata.cmdsettings == NULL)
    {
        CLIcmddata.cmdsettings = &cms;
    }
}

static MILK_HOT errno_t compute_function()
{
    fft_image_translate(p_in, p_out, p_xtransl, p_ytransl);
    return RETURN_SUCCESS;
}

#ifndef FPS_STANDALONE
static errno_t __attribute__((unused)) CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_milkfft__ffttranslate()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif


/*^-----------------------------------------------------------------------------
|
| COMMENT:  Inclusion of this routine requires inclusion of modules:
|           fft, gen_image
* DOES NOT WORK ON STREAM
+-----------------------------------------------------------------------------*/
int fft_image_translate(const char *ID_name, const char *ID_out, double xtransl, double ytransl)
{
    long ID;
    long naxes[2];
    //  int n0,n1;

    ID       = image_ID(ID_name, dcimg, dcnimg);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];

    //  fprintf( stdout, "[arith_image_translate %ld %ld %ld     %f %f]\n", ID, naxes[0], naxes[1], xtransl, ytransl);

    // n0 = (int) ((log10(naxes[0])/log10(2))+0.01);
    // n1 = (int) ((log10(naxes[0])/log10(2))+0.01);

    //  if ((n0==n1)&&(naxes[0]==(int) pow(2,n0))&&(naxes[1]==(int) pow(2,n1)))
    // {

    do2drfft(ID_name, "ffttmp1");
    mk_amph_from_complex("ffttmp1", "amptmp", "phatmp", 0);

    delete_image_ID("ffttmp1", DELETE_IMAGE_ERRMODE_WARNING);
    arith_make_slopexy("sltmp", naxes[0], naxes[1], xtransl * 2.0 * M_PI / naxes[0],
                       ytransl * 2.0 * M_PI / naxes[1]);
    permut("sltmp");

    arith_image_add("phatmp", "sltmp", "phatmp1");
    delete_image_ID("phatmp", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("sltmp", DELETE_IMAGE_ERRMODE_WARNING);

    mk_complex_from_amph("amptmp", "phatmp1", "ffttmp2", 0);
    delete_image_ID("amptmp", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("phatmp1", DELETE_IMAGE_ERRMODE_WARNING);
    do2dffti("ffttmp2", "ffttmp3");
    delete_image_ID("ffttmp2", DELETE_IMAGE_ERRMODE_WARNING);
    mk_reim_from_complex("ffttmp3", "retmp", "imtmp", 0);
    arith_image_cstmult("retmp", 1.0 / naxes[0] / naxes[1], ID_out);
    delete_image_ID("ffttmp3", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("retmp", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("imtmp", DELETE_IMAGE_ERRMODE_WARNING);
    // }
    // else
    //{
    // printf("Error: image size does not allow translation\n");
    //}

    return (0);
}
