/**
 * @file fconvolve.c
 * @brief Fourier-based convolution
 */

#include <math.h>

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"

#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "fft/fft.h"

// Forward declaration
imageID fconvolve(
    const char *__restrict name_in,
    const char *__restrict name_ke,
    const char *__restrict name_out);

static char fconv_in[FUNCTION_PARAMETER_STRMAXLEN]
    = "imin";
static char fconv_ke[FUNCTION_PARAMETER_STRMAXLEN]
    = "kernim";
static char fconv_out[FUNCTION_PARAMETER_STRMAXLEN]
    = "imout";

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "fconv",
    .cmdkey      = "fconv",
    .description =
        "Fourier-based convolution",
    .description_long =
        "Convolve two images using FFT-based multiplication in Fourier space. Efficient for large kernel sizes."
};

#define FPS_PARAMS(X) \
    X(".in_name", fconv_in, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input image") \
    X(".ke_name", fconv_ke, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "kernel image") \
    X(".out_name", fconv_out, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output image")

static FPS_CLI_BINDING FPS_bindings[] = {
    FPS_PARAMS(FPS_X_BINDING)
};
static const int nb_bindings =
    sizeof(FPS_bindings) /
    sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF farg[] = {
    FPS_PARAMS(FPS_X_FARG)
};
static CLICMDDATA CLIcmddata = {
    "", "", CLICMD_FIELDS_DEFAULTS
};
static CMDSETTINGS cms = {0};

static __attribute__((constructor))
void init_cms(void)
{
    strncpy(CLIcmddata.key,
            FPS_app_info.cmdkey,
            sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description,
            FPS_app_info.description,
            sizeof(CLIcmddata.description)
            - 1);
    CLIcmddata.nbarg =
        sizeof(farg) / sizeof(CLICMDARGDEF);
    CLIcmddata.funcfpscliarg = farg;
    CLIcmddata.flags = CLICMDFLAG_FPS;
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings = &cms;
    }
}

static MILK_HOT errno_t compute_function()
{
    fconvolve(fconv_in, fconv_ke, fconv_out);
    return RETURN_SUCCESS;
}

static errno_t __attribute__((unused)) CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg,
        &CLIcmddata,
        FPS_bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_image_filter__fconvolve()
{
    safe_fps_fill_farg_examples(
        farg, FPS_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

imageID fconvolve(const char *__restrict name_in,
                  const char *__restrict name_ke,
                  const char *__restrict name_out)
{
    imageID IDin;
    imageID IDke;
    long    naxes[2];
    imageID IDout;

    IDin     = image_ID(name_in, dcimg, dcnimg);
    naxes[0] = dcimg[IDin].md[0].size[0];
    naxes[1] = dcimg[IDin].md[0].size[1];
    IDke     = image_ID(name_ke, dcimg, dcnimg);
    if((naxes[0] != dcimg[IDke].md[0].size[0]) ||
            (naxes[1] != dcimg[IDke].md[0].size[1]))
    {
        fprintf(stderr,
                "ERROR in function fconvolve: image and kernel have different "
                "sizes\n");
        exit(0);
    }
    //  save_fl_fits(name_in,"test1.fits");
    // save_fl_fits(name_ke,"test2.fits");

    do2drfft(name_in, "infft");
    do2drfft(name_ke, "kefft");

    arith_image_Cmult("infft", "kefft", "outfft");
    delete_image_ID("infft", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("kefft", DELETE_IMAGE_ERRMODE_WARNING);
    do2dffti("outfft", "outfft1");
    delete_image_ID("outfft", DELETE_IMAGE_ERRMODE_WARNING);
    mk_reim_from_complex("outfft1", "tmpre", "tmpim", 0);

    //  save_fl_fits("tmpre","tmpre.fits");
    // save_fl_fits("tmpim","tmpim.fits");

    delete_image_ID("outfft1", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("tmpim", DELETE_IMAGE_ERRMODE_WARNING);
    arith_image_cstmult("tmpre", 1.0 / naxes[0] / naxes[1], name_out);
    delete_image_ID("tmpre", DELETE_IMAGE_ERRMODE_WARNING);
    permut(name_out);

    IDout = image_ID(name_out, dcimg, dcnimg);

    return IDout;
}

// to avoid edge effects
imageID fconvolve_padd(const char *__restrict name_in,
                       const char *__restrict name_ke,
                       long paddsize,
                       const char *__restrict name_out)
{
    imageID IDin;
    imageID IDke;
    imageID ID1;
    imageID ID2;
    imageID ID3;
    imageID IDout;
    long    naxes[2];
    long    naxespadd[2];
    long    ii, jj;

    IDin     = image_ID(name_in, dcimg, dcnimg);
    naxes[0] = dcimg[IDin].md[0].size[0];
    naxes[1] = dcimg[IDin].md[0].size[1];
    IDke     = image_ID(name_ke, dcimg, dcnimg);
    if((naxes[0] != dcimg[IDke].md[0].size[0]) ||
            (naxes[1] != dcimg[IDke].md[0].size[1]))
    {
        fprintf(stderr,
                "ERROR in function fconvolve: image and kernel have different "
                "sizes\n");
        exit(0);
    }

    naxespadd[0] = naxes[0] + 2 * paddsize;
    naxespadd[1] = naxes[1] + 2 * paddsize;

    // printf("new axes : %ld %ld\n",naxespadd[0],naxespadd[1]);

    create_2Dimage_ID("tmpimpadd", naxespadd[0], naxespadd[1], &ID1);
    create_2Dimage_ID("tmpkepadd", naxespadd[0], naxespadd[1], &ID2);
    create_2Dimage_ID("tmpim1padd", naxespadd[0], naxespadd[1], &ID3);

    for(ii = 0; ii < naxes[0]; ii++)
        for(jj = 0; jj < naxes[1]; jj++)
        {
            dcimg[ID1]
            .array.F[(jj + paddsize) * naxespadd[0] + (ii + paddsize)] =
                dcimg[IDin].array.F[jj * naxes[0] + ii];
            dcimg[ID2]
            .array.F[(jj + paddsize) * naxespadd[0] + (ii + paddsize)] =
                dcimg[IDke].array.F[jj * naxes[0] + ii];
            dcimg[ID3]
            .array.F[(jj + paddsize) * naxespadd[0] + (ii + paddsize)] =
                1.0;
        }

    //  list_image_ID();
    //  printf("Doing convolutions...");
    //  fflush(stdout);

    fconvolve("tmpimpadd", "tmpkepadd", "tmpconv1");
    fconvolve("tmpim1padd", "tmpkepadd", "tmpconv2");

    //  printf(" done\n");
    // fflush(stdout);

    delete_image_ID("tmpimpadd", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("tmpkepadd", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("tmpim1padd", DELETE_IMAGE_ERRMODE_WARNING);

    ID1 = image_ID("tmpconv1", dcimg, dcnimg);
    ID2 = image_ID("tmpconv2", dcimg, dcnimg);
    create_2Dimage_ID(name_out, naxes[0], naxes[1], &IDout);

    for(ii = 0; ii < naxes[0]; ii++)
        for(jj = 0; jj < naxes[1]; jj++)
        {
            dcimg[IDout].array.F[jj * naxes[0] + ii] =
                dcimg[ID1]
                .array.F[(jj + paddsize) * naxespadd[0] + (ii + paddsize)] /
                dcimg[ID2]
                .array.F[(jj + paddsize) * naxespadd[0] + (ii + paddsize)];
        }
    delete_image_ID("tmpconv1", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("tmpconv2", DELETE_IMAGE_ERRMODE_WARNING);

    return IDout;
}

imageID fconvolve_1(const char *__restrict name_in,
                    const char *__restrict kefft,
                    const char *__restrict name_out)
{
    /* FFT of kernel has already been done */
    imageID IDin;
    long    naxes[2];

    IDin     = image_ID(name_in, dcimg, dcnimg);
    naxes[0] = dcimg[IDin].md[0].size[0];
    naxes[1] = dcimg[IDin].md[0].size[1];

    do2drfft(name_in, "infft");

    arith_image_Cmult("infft", kefft, "outfft");
    delete_image_ID("infft", DELETE_IMAGE_ERRMODE_WARNING);
    do2dffti("outfft", "outfft1");
    delete_image_ID("outfft", DELETE_IMAGE_ERRMODE_WARNING);
    mk_reim_from_complex("outfft1", "tmpre", "tmpim", 0);
    delete_image_ID("outfft1", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("tmpim", DELETE_IMAGE_ERRMODE_WARNING);
    arith_image_cstmult("tmpre", 1.0 / naxes[0] / naxes[1], name_out);
    delete_image_ID("tmpre", DELETE_IMAGE_ERRMODE_WARNING);
    permut(name_out);
    imageID IDout = image_ID(name_out, dcimg, dcnimg);

    return IDout;
}

// if blocksize = 512, for images > 512x512, break image in 512x512 overlapping blocks
// kernel image must be blocksize
imageID fconvolveblock(const char *__restrict name_in,
                       const char *__restrict name_ke,
                       const char *__restrict name_out,
                       long blocksize)
{
    imageID IDin;
    imageID IDout;
    imageID IDtmp;
    imageID IDtmpout;
    imageID IDcnt;
    long    xsize, ysize;
    long    overlap;
    long    ii, jj, ii0, jj0;
    float   gain;
    float   alpha = 4.0;

    overlap = (long)(blocksize / 10);
    IDin    = image_ID(name_in, dcimg, dcnimg);
    xsize   = dcimg[IDin].md[0].size[0];
    ysize   = dcimg[IDin].md[0].size[1];

    create_2Dimage_ID(name_out, xsize, ysize, &IDout);

    create_2Dimage_ID("tmpblock", blocksize, blocksize, &IDtmp);

    create_2Dimage_ID("tmpcnt", xsize, ysize, &IDcnt);

    for(ii = 0; ii < xsize * ysize; ii++)
    {
        dcimg[IDcnt].array.F[ii] = 0.0f;
    }

    for(ii0 = 0; ii0 < xsize - overlap; ii0 += blocksize - overlap)
        for(jj0 = 0; jj0 < ysize - overlap; jj0 += blocksize - overlap)
        {
            for(ii = 0; ii < blocksize; ii++)
                for(jj = 0; jj < blocksize; jj++)
                {
                    if((ii0 + ii < xsize) && (jj0 + jj < ysize))
                    {
                        dcimg[IDtmp].array.F[jj * blocksize + ii] =
                            dcimg[IDin]
                            .array.F[(jj0 + jj) * xsize + (ii0 + ii)];
                    }
                    else
                    {
                        dcimg[IDtmp].array.F[jj * blocksize + ii] = 0.0f;
                    }
                }
            fconvolve("tmpblock", name_ke, "tmpblockc");
            IDtmpout = image_ID("tmpblockc", dcimg, dcnimg);
            for(ii = 0; ii < blocksize; ii++)
                for(jj = 0; jj < blocksize; jj++)
                {
                    if((ii0 + ii < xsize) && (jj0 + jj < ysize))
                    {
                        gain = 1.0;
                        if(ii < overlap)
                        {
                            gain *= pow(1.0 * (1.0 * ii / overlap), alpha);
                        }
                        if(jj < overlap)
                        {
                            gain *= pow(1.0 * (1.0 * jj / overlap), alpha);
                        }
                        if(ii > blocksize - overlap)
                        {
                            gain *=
                                pow(1.0 * (1.0 * (blocksize - ii) / overlap),
                                    alpha);
                        }
                        if(jj > blocksize - overlap)
                        {
                            gain *=
                                pow(1.0 * (1.0 * (blocksize - jj) / overlap),
                                    alpha);
                        }

                        dcimg[IDout]
                        .array.F[(jj0 + jj) * xsize + (ii0 + ii)] +=
                            gain *
                            dcimg[IDtmpout].array.F[jj * blocksize + ii];
                        dcimg[IDcnt]
                        .array.F[(jj0 + jj) * xsize + (ii0 + ii)] +=
                            gain * 1.0;
                    }
                }
        }
    //  save_fl_fits("tmpcnt","tmpcnt.fits");
    // exit(0);
    for(ii = 0; ii < xsize * ysize; ii++)
    {
        dcimg[IDout].array.F[ii] /= dcimg[IDcnt].array.F[ii] + 1.0e-8;
    }

    delete_image_ID("tmpcnt", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("tmpblock", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("tmpblockc", DELETE_IMAGE_ERRMODE_WARNING);

    return IDout;
}
