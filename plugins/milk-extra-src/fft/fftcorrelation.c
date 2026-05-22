/**
 * @file fftcorrelation.c
 * @brief Correlate two images using FFT
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
imageID fft_correlation(const char *ID_name1, const char *ID_name2, const char *ID_nameout);

/* =========================================
 *  V2 PARAMS
 * ======================================= */

static char p_in1[FUNCTION_PARAMETER_STRMAXLEN] = "im1";
static char p_in2[FUNCTION_PARAMETER_STRMAXLEN] = "im2";
static char p_out[FUNCTION_PARAMETER_STRMAXLEN] = "outim";

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "fcorrel",
    .cmdkey      = "fcorrel",
    .description = "correlate two images",
    .description_long =
        "Compute the cross-correlation of two images via FFT. Multiplies the Fourier transforms "
        "and applies the inverse FFT to produce the correlation map."
};

#define FPS_PARAMS(X)                                                             \
    X(".in1", p_in1, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image 1") \
    X(".in2", p_in2, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image 2") \
    X(".out", p_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output correlation")

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
    fft_correlation(p_in1, p_in2, p_out);
    return RETURN_SUCCESS;
}

#ifndef FPS_STANDALONE
static errno_t __attribute__((unused)) CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_milkfft__fftcorrelation()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif


imageID fft_correlation(const char *ID_name1, const char *ID_name2, const char *ID_nameout)
{
    imageID ID1;
    imageID IDout;
    long    nelement;

    char ft1name[STRINGMAXLEN_IMGNAME];
    char ft2name[STRINGMAXLEN_IMGNAME];
    char fta1name[STRINGMAXLEN_IMGNAME];
    char fta2name[STRINGMAXLEN_IMGNAME];
    char ftp1name[STRINGMAXLEN_IMGNAME];
    char ftp2name[STRINGMAXLEN_IMGNAME];
    char fta12name[STRINGMAXLEN_IMGNAME];
    char ftp12name[STRINGMAXLEN_IMGNAME];
    char fftname[STRINGMAXLEN_IMGNAME];
    char fft1name[STRINGMAXLEN_IMGNAME];
    char fft1pname[STRINGMAXLEN_IMGNAME];

    ID1      = image_ID(ID_name1, dcimg, dcnimg);
    nelement = dcimg[ID1].md[0].nelement;

    WRITE_IMAGENAME(ft1name, "_ft1_%d", (int) getpid());
    do2drfft(ID_name1, ft1name);

    WRITE_IMAGENAME(ft2name, "_ft2_%d", (int) getpid());
    do2drfft(ID_name2, ft2name);

    WRITE_IMAGENAME(fta1name, "_%.60s_a_%d", ft1name, (int) getpid());
    WRITE_IMAGENAME(ftp1name, "_%.60s_p_%d", ft1name, (int) getpid());
    WRITE_IMAGENAME(fta2name, "_%.60s_a_%d", ft2name, (int) getpid());
    WRITE_IMAGENAME(ftp2name, "_%.60s_p_%d", ft2name, (int) getpid());
    WRITE_IMAGENAME(fta12name, "_%.60s_12a_%d", ft1name, (int) getpid());
    WRITE_IMAGENAME(ftp12name, "_%.60s_12p_%d", ft1name, (int) getpid());

    mk_amph_from_complex(ft1name, fta1name, ftp1name, 0);
    mk_amph_from_complex(ft2name, fta2name, ftp2name, 0);

    delete_image_ID(ft1name, DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID(ft2name, DELETE_IMAGE_ERRMODE_WARNING);

    arith_image_mult(fta1name, fta2name, fta12name);
    arith_image_sub(ftp1name, ftp2name, ftp12name);
    delete_image_ID(fta1name, DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID(fta2name, DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID(ftp1name, DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID(ftp2name, DELETE_IMAGE_ERRMODE_WARNING);

    arith_image_cstmult_inplace(fta12name, 1.0 / sqrt(nelement) / (1.0 * nelement));

    WRITE_IMAGENAME(fftname, "_fft_%d", (int) getpid());

    mk_complex_from_amph(fta12name, ftp12name, fftname, 0);
    delete_image_ID(fta12name, DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID(ftp12name, DELETE_IMAGE_ERRMODE_WARNING);

    WRITE_IMAGENAME(fft1name, "_fft1_%d", (int) getpid());

    do2dfft(fftname, fft1name);
    delete_image_ID(fftname, DELETE_IMAGE_ERRMODE_WARNING);

    WRITE_IMAGENAME(fft1pname, "_fft1p_%d", (int) getpid());

    mk_amph_from_complex(fft1name, ID_nameout, fft1pname, 0);
    permut(ID_nameout);

    delete_image_ID(fft1name, DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID(fft1pname, DELETE_IMAGE_ERRMODE_WARNING);

    IDout = image_ID(ID_nameout, dcimg, dcnimg);

    return IDout;
}
