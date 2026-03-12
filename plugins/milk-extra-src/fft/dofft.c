/**
 * @file dofft.c
 * @brief Dofft module
 */

/** @file dofft.c
 */

#include <fftw3.h>

#include "dofft.h"

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif

#include "COREMOD_memory/COREMOD_memory.h"

#include "wisdom.h"
#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO/ImageStreamIO.h"

#define FFTWOPTMODE FFTW_ESTIMATE


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "dofft",
    .cmdkey      = "dofft",
    .description = "perform 2D complex FFT"
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char dofft_inimname[
    FUNCTION_PARAMETER_STRMAXLEN];
static char dofft_outimname[
    FUNCTION_PARAMETER_STRMAXLEN];
static int32_t dofft_dir = 0;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".in_name", dofft_inimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input complex image") \
    X(".out_name", dofft_outimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output complex image") \
    X(".dir", &dofft_dir, \
      FPTYPE_INT32, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "FFT direction")

// Forward declarations
imageID do1dfft(const char *in_name,
                const char *out_name);

imageID do1drfft(const char *in_name,
                 const char *out_name);

imageID do2dfft(const char *in_name,
                const char *out_name);


/* =========================================
 *  CMD 2: do1Dfft (2 args)
 * ======================================= */

static char p_1dfft_in[
    FUNCTION_PARAMETER_STRMAXLEN]
    = "in";
static char p_1dfft_out[
    FUNCTION_PARAMETER_STRMAXLEN]
    = "out";

static FPS_APP_INFO FPS_app_info_1dfft = {
    .fps_name    = "do1Dfft",
    .cmdkey      = "do1Dfft",
    .description =
        "perform 1D complex->complex FFT"
};

#define FPS_PARAMS_1DFFT(X) \
    X(".in_name", p_1dfft_in, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input complex image") \
    X(".out_name", p_1dfft_out, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output complex image")

static CLICMDDATA CLIcmddata_1dfft = {
    "", "", CLICMD_FIELDS_NOPARAM
};
static CMDSETTINGS cms_1dfft = {0};

static __attribute__((constructor))
void init_cms_1dfft(void)
{
    strncpy(CLIcmddata_1dfft.key,
            FPS_app_info_1dfft.cmdkey,
            sizeof(CLIcmddata_1dfft.key)
            - 1);
    strncpy(CLIcmddata_1dfft.description,
            FPS_app_info_1dfft.description,
            sizeof(
                CLIcmddata_1dfft
                .description) - 1);
    if (CLIcmddata_1dfft.cmdsettings
        == NULL) {
        CLIcmddata_1dfft.cmdsettings =
            &cms_1dfft;
    }
}

static errno_t compute_1dfft()
{
    do1dfft(p_1dfft_in, p_1dfft_out);
    return RETURN_SUCCESS;
}


/* =========================================
 *  CMD 3: do1Drfft (2 args)
 * ======================================= */

static char p_1drfft_in[
    FUNCTION_PARAMETER_STRMAXLEN]
    = "in";
static char p_1drfft_out[
    FUNCTION_PARAMETER_STRMAXLEN]
    = "out";

static FPS_APP_INFO FPS_app_info_1drfft = {
    .fps_name    = "do1Drfft",
    .cmdkey      = "do1Drfft",
    .description =
        "perform 1D real->complex FFT"
};

#define FPS_PARAMS_1DRFFT(X) \
    X(".in_name", p_1drfft_in, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input real image") \
    X(".out_name", p_1drfft_out, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output complex image")

static CLICMDDATA CLIcmddata_1drfft = {
    "", "", CLICMD_FIELDS_NOPARAM
};
static CMDSETTINGS cms_1drfft = {0};

static __attribute__((constructor))
void init_cms_1drfft(void)
{
    strncpy(CLIcmddata_1drfft.key,
            FPS_app_info_1drfft.cmdkey,
            sizeof(CLIcmddata_1drfft.key)
            - 1);
    strncpy(
        CLIcmddata_1drfft.description,
        FPS_app_info_1drfft.description,
        sizeof(
            CLIcmddata_1drfft
            .description) - 1);
    if (CLIcmddata_1drfft.cmdsettings
        == NULL) {
        CLIcmddata_1drfft.cmdsettings =
            &cms_1drfft;
    }
}

static errno_t compute_1drfft()
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
    do2dfft(dofft_inimname,
            dofft_outimname);
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#ifndef FPS_STANDALONE
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

static FPS_CLI_BINDING b_1dfft[] = {
    FPS_PARAMS_1DFFT(FPS_X_BINDING)
};
static CLICMDARGDEF fa_1dfft[] = {
    FPS_PARAMS_1DFFT(FPS_X_FARG)
};

static errno_t CLIfunction_1dfft(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_1dfft,
        fa_1dfft, &CLIcmddata_1dfft,
        b_1dfft,
        sizeof(b_1dfft) /
        sizeof(FPS_CLI_BINDING),
        compute_1dfft);
}

static FPS_CLI_BINDING b_1drfft[] = {
    FPS_PARAMS_1DRFFT(FPS_X_BINDING)
};
static CLICMDARGDEF fa_1drfft[] = {
    FPS_PARAMS_1DRFFT(FPS_X_FARG)
};

static errno_t CLIfunction_1drfft(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_1drfft,
        fa_1drfft, &CLIcmddata_1drfft,
        b_1drfft,
        sizeof(b_1drfft) /
        sizeof(FPS_CLI_BINDING),
        compute_1drfft);
}

errno_t
CLIADDCMD_milkfft__dofft()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC

    safe_fps_fill_farg_examples(
        fa_1dfft, b_1dfft,
        sizeof(b_1dfft) /
        sizeof(FPS_CLI_BINDING));
    {
        int cmdi = RegisterCLIcmd(
            CLIcmddata_1dfft,
            CLIfunction_1dfft);
        CLIcmddata_1dfft.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }

    safe_fps_fill_farg_examples(
        fa_1drfft, b_1drfft,
        sizeof(b_1drfft) /
        sizeof(FPS_CLI_BINDING));
    {
        int cmdi = RegisterCLIcmd(
            CLIcmddata_1drfft,
            CLIfunction_1drfft);
        CLIcmddata_1drfft.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }

    return RETURN_SUCCESS;
}
#endif

int array_index(long size)
{
    int i;

    switch(size)
    {
        case 1:
            i = 0;
            break;
        case 2:
            i = 1;
            break;
        case 4:
            i = 2;
            break;
        case 8:
            i = 3;
            break;
        case 16:
            i = 4;
            break;
        case 32:
            i = 5;
            break;
        case 64:
            i = 6;
            break;
        case 128:
            i = 7;
            break;
        case 256:
            i = 8;
            break;
        case 512:
            i = 9;
            break;
        case 1024:
            i = 10;
            break;
        case 2048:
            i = 11;
            break;
        case 4096:
            i = 12;
            break;
        case 8192:
            i = 13;
            break;
        case 16384:
            i = 14;
            break;
        default:
            i = 100;
    }

    return (i);
}

/* 1d complex -> complex fft */
// supports single and double precisions
//
imageID FFT_do1dfft(
    const char *__restrict in_name,
    const char *__restrict out_name,
    int dir)
{
    int            OK = 0;
    fftwf_plan     plan;
    fftw_plan      plan_double;
    fftwf_complex *inptr, *outptr;
    fftw_complex  *inptr_double;
    fftw_complex  *outptr_double;

    IMGID imgin =
        imgid_make_from_name(in_name);
    resolveIMGID(&imgin,
                 ERRMODE_ABORT,
                 dcimg, dcnimg);

    long naxis = imgin.md->naxis;
    uint8_t datatype =
        imgin.md->datatype;

    int *naxes = (int *) malloc(
        naxis * sizeof(int));
    if(naxes == NULL)
    {
        PRINT_ERROR(
            "malloc returns NULL pointer");
        abort();
    }

    IMGID imgout =
        imgid_make_from_name(out_name);
    imgout.mdt->naxis = naxis;
    for(long i = 0; i < naxis; i++)
    {
        imgout.mdt->size[i] =
            imgin.md->size[i];
        naxes[i] =
            (int) imgin.md->size[i];
    }
    imgout.mdt->datatype = datatype;
    imgout.mdt->shared = dcshareddft;
    imgout.mdt->NBkw = NB_KEYWNODE_MAX;
    imgout.im = (IMAGE *) calloc(
        1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    if(naxis == 1)
    {
        if(array_index(naxes[0]) != 100)
        {
            OK = 1;
            if(datatype
               == _DATATYPE_COMPLEX_FLOAT)
            {
                plan = fftwf_plan_dft_1d(
                    naxes[0],
                    (fftwf_complex *)
                        imgin.im->array.CF,
                    (fftwf_complex *)
                        imgout.im->array.CF,
                    dir, FFTWOPTMODE);
                fftwf_execute(plan);
                fftwf_destroy_plan(plan);
            }
            else
            {
                plan_double =
                    fftw_plan_dft_1d(
                        naxes[0],
                        (fftw_complex *)
                            imgin.im
                                ->array.CD,
                        (fftw_complex *)
                            imgout.im
                                ->array.CD,
                        dir, FFTWOPTMODE);
                fftw_execute(plan_double);
                fftw_destroy_plan(
                    plan_double);
            }
        }
        else
        {
            OK = 1;
            if(datatype
               == _DATATYPE_COMPLEX_FLOAT)
            {
                plan = fftwf_plan_dft_1d(
                    naxes[0],
                    (fftwf_complex *)
                        imgin.im->array.CF,
                    (fftwf_complex *)
                        imgout.im->array.CF,
                    dir, FFTWOPTMODE);
                fftwf_execute(plan);
                fftwf_destroy_plan(plan);
            }
            else
            {
                plan_double =
                    fftw_plan_dft_1d(
                        naxes[0],
                        (fftw_complex *)
                            imgin.im
                                ->array.CD,
                        (fftw_complex *)
                            imgout.im
                                ->array.CD,
                        dir, FFTWOPTMODE);
                fftw_execute(plan_double);
                fftw_destroy_plan(
                    plan_double);
            }
        }
    }

    if(naxis == 2)
    {
        if((naxes[1] == 1)
           && (array_index(naxes[0])
               != 100))
        {
            OK = 1;
            if(datatype
               == _DATATYPE_COMPLEX_FLOAT)
            {
                inptr =
                    (fftwf_complex *)
                        imgin.im->array.CF;
                outptr =
                    (fftwf_complex *)
                        imgout.im->array.CF;
                plan =
                    fftwf_plan_dft_1d(
                        naxes[0],
                        inptr, outptr,
                        dir, FFTWOPTMODE);
                fftwf_execute(plan);
                fftwf_destroy_plan(plan);
            }
            else
            {
                inptr_double =
                    (fftw_complex *)
                        imgin.im->array.CD;
                outptr_double =
                    (fftw_complex *)
                        imgout.im->array.CD;
                plan_double =
                    fftw_plan_dft_1d(
                        naxes[0],
                        inptr_double,
                        outptr_double,
                        dir, FFTWOPTMODE);
                fftw_execute(plan_double);
                fftw_destroy_plan(
                    plan_double);
            }
        }
        else
        {
            OK = 1;
            if(datatype
               == _DATATYPE_COMPLEX_FLOAT)
            {
                inptr =
                    (fftwf_complex *)
                    malloc(
                        sizeof(
                            fftwf_complex)
                        * naxes[0]);
                if(inptr == NULL)
                {
                    PRINT_ERROR(
                        "malloc returns"
                        " NULL pointer");
                    abort();
                }

                outptr =
                    (fftwf_complex *)
                    malloc(
                        sizeof(
                            fftwf_complex)
                        * naxes[0]);
                if(outptr == NULL)
                {
                    PRINT_ERROR(
                        "malloc returns"
                        " NULL pointer");
                    abort();
                }

                plan =
                    fftwf_plan_dft_1d(
                        naxes[0],
                        inptr, outptr,
                        dir, FFTWOPTMODE);

                for(long jj = 0;
                    jj < naxes[1]; jj++)
                {
                    memcpy(
                        (char *) inptr,
                        (char *)
                            imgin.im
                                ->array.CF
                        + sizeof(
                              fftwf_complex)
                          * jj * naxes[0],
                        sizeof(
                            fftwf_complex)
                        * naxes[0]);
                    fftwf_execute(plan);
                    memcpy(
                        (char *)
                            imgout.im
                                ->array.CF
                        + sizeof(
                              complex_float)
                          * jj * naxes[0],
                        outptr,
                        sizeof(
                            fftwf_complex)
                        * naxes[0]);
                }
                fftwf_destroy_plan(plan);
                free(inptr);
                free(outptr);
            }
            else
            {
                inptr_double =
                    (fftw_complex *)
                    malloc(
                        sizeof(fftw_complex)
                        * naxes[0]);
                if(inptr_double == NULL)
                {
                    PRINT_ERROR(
                        "malloc returns"
                        " NULL pointer");
                    abort();
                }

                outptr_double =
                    (fftw_complex *)
                    malloc(
                        sizeof(fftw_complex)
                        * naxes[0]);
                if(outptr_double == NULL)
                {
                    PRINT_ERROR(
                        "malloc returns"
                        " NULL pointer");
                    abort();
                }

                plan_double =
                    fftw_plan_dft_1d(
                        naxes[0],
                        inptr_double,
                        outptr_double,
                        dir, FFTWOPTMODE);

                for(long jj = 0;
                    jj < naxes[1]; jj++)
                {
                    memcpy(
                        (char *)
                            inptr_double,
                        (char *)
                            imgin.im
                                ->array.CD
                        + sizeof(
                              fftw_complex)
                          * jj * naxes[0],
                        sizeof(fftw_complex)
                        * naxes[0]);
                    fftw_execute(
                        plan_double);
                    memcpy(
                        (char *)
                            imgout.im
                                ->array.CD
                        + sizeof(
                              complex_double)
                          * jj * naxes[0],
                        outptr_double,
                        sizeof(fftw_complex)
                        * naxes[0]);
                }
                fftw_destroy_plan(
                    plan_double);
                free(inptr_double);
                free(outptr_double);
            }
        }
    }

    if(OK == 0)
    {
        printf(
            "Error : image dimension"
            " not appropriate"
            " for FFT\n");
    }
    free(naxes);

    return imgout.ID;
}

/* 1d real -> complex fft */
// supports single and double precision
imageID do1drfft(
    const char *__restrict in_name,
    const char *__restrict out_name)
{
    int            OK = 0;
    fftwf_plan     plan;
    fftw_plan      plan_double;
    fftwf_complex *outptr;
    fftw_complex  *outptr_double;
    float         *inptr;
    double        *inptr_double;

    IMGID imgin =
        imgid_make_from_name(in_name);
    resolveIMGID(&imgin,
                 ERRMODE_ABORT,
                 dcimg, dcnimg);

    long naxis = imgin.md->naxis;
    uint8_t datatype =
        imgin.md->datatype;

    int *naxes = (int *) malloc(
        naxis * sizeof(int));
    if(naxes == NULL)
    {
        PRINT_ERROR(
            "malloc returns NULL pointer");
        abort();
    }

    int fftaxis = 0;
    if(naxis == 3)
    {
        fftaxis = 2;
    }

    uint32_t naxesout[3];
    for(long i = 0; i < naxis; i++)
    {
        naxes[i] =
            (int) imgin.md->size[i];
        naxesout[i] =
            imgin.md->size[i];
        if(i == fftaxis)
        {
            naxesout[i] =
                imgin.md->size[i] / 2
                + 1;
        }
    }

    uint8_t outtype;
    if(datatype == _DATATYPE_DOUBLE)
    {
        outtype =
            _DATATYPE_COMPLEX_DOUBLE;
    }
    else
    {
        outtype =
            _DATATYPE_COMPLEX_FLOAT;
    }

    IMGID imgout =
        imgid_make_from_name(out_name);
    imgout.mdt->naxis = naxis;
    for(long i = 0; i < naxis; i++)
    {
        imgout.mdt->size[i] =
            naxesout[i];
    }
    imgout.mdt->datatype = outtype;
    imgout.mdt->shared = dcshareddft;
    imgout.mdt->NBkw = NB_KEYWNODE_MAX;
    imgout.im = (IMAGE *) calloc(
        1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    if(naxis == 2)
    {
        if((naxes[1] == 1)
           && (array_index(naxes[0])
               != 100))
        {
            OK = 1;
            if(datatype
               == _DATATYPE_FLOAT)
            {
                plan =
                    fftwf_plan_dft_r2c_1d(
                        naxes[0],
                        imgin.im->array.F,
                        (fftwf_complex *)
                            imgout.im
                                ->array.CF,
                        FFTWOPTMODE);
                fftwf_execute(plan);
                fftwf_destroy_plan(plan);
            }
            else
            {
                plan_double =
                    fftw_plan_dft_r2c_1d(
                        naxes[0],
                        imgin.im->array.D,
                        (fftw_complex *)
                            imgout.im
                                ->array.CD,
                        FFTWOPTMODE);
                fftw_execute(plan_double);
                fftw_destroy_plan(
                    plan_double);
            }
        }
        else
        {
            OK = 1;
            if(datatype
               == _DATATYPE_FLOAT)
            {
                inptr = (float *) malloc(
                    sizeof(float)
                    * naxes[0]);
                if(inptr == NULL)
                {
                    PRINT_ERROR(
                        "malloc returns"
                        " NULL pointer");
                    abort();
                }

                outptr =
                    (fftwf_complex *)
                    malloc(
                        sizeof(
                            fftwf_complex)
                        * naxes[0]);
                if(outptr == NULL)
                {
                    PRINT_ERROR(
                        "malloc returns"
                        " NULL pointer");
                    abort();
                }

                plan =
                    fftwf_plan_dft_r2c_1d(
                        naxes[0],
                        inptr, outptr,
                        FFTWOPTMODE);

                for(long jj = 0;
                    jj < naxes[1]; jj++)
                {
                    memcpy(
                        (char *) inptr,
                        (char *)
                            imgin.im
                                ->array.F
                        + sizeof(float)
                          * jj * naxes[0],
                        sizeof(float)
                        * naxes[0]);
                    fftwf_execute(plan);
                    memcpy(
                        (char *)
                            imgout.im
                                ->array.CF
                        + sizeof(
                              complex_float)
                          * jj
                          * naxesout[0],
                        outptr,
                        sizeof(
                            fftwf_complex)
                        * naxesout[0]);
                }
                fftwf_destroy_plan(plan);
                free(inptr);
                free(outptr);
            }
            else
            {
                inptr_double =
                    (double *) malloc(
                        sizeof(double)
                        * naxes[0]);
                if(inptr_double == NULL)
                {
                    PRINT_ERROR(
                        "malloc returns"
                        " NULL pointer");
                    abort();
                }

                outptr_double =
                    (fftw_complex *)
                    malloc(
                        sizeof(fftw_complex)
                        * naxes[0]);
                if(outptr_double == NULL)
                {
                    PRINT_ERROR(
                        "malloc returns"
                        " NULL pointer");
                    abort();
                }

                plan_double =
                    fftw_plan_dft_r2c_1d(
                        naxes[0],
                        inptr_double,
                        outptr_double,
                        FFTWOPTMODE);

                for(long jj = 0;
                    jj < naxes[1]; jj++)
                {
                    memcpy(
                        (char *)
                            inptr_double,
                        (char *)
                            imgin.im
                                ->array.D
                        + sizeof(double)
                          * jj * naxes[0],
                        sizeof(double)
                        * naxes[0]);
                    fftw_execute(
                        plan_double);
                    memcpy(
                        (char *)
                            imgout.im
                                ->array.CD
                        + sizeof(
                              complex_double)
                          * jj
                          * naxesout[0],
                        outptr_double,
                        sizeof(fftw_complex)
                        * naxesout[0]);
                }
                fftw_destroy_plan(
                    plan_double);
                free(inptr_double);
                free(outptr_double);
            }
        }
    }
    if(naxis == 3)
    {
        /* 1D FFT along last dimension */
        OK = 1;
        uint64_t xysize = naxes[0];
        xysize *= naxes[1];

        if(datatype == _DATATYPE_FLOAT)
        {
            inptr = (float *) malloc(
                sizeof(float) * naxes[2]);
            outptr =
                (fftwf_complex *)
                malloc(
                    sizeof(fftwf_complex)
                    * naxes[2]);

            plan =
                fftwf_plan_dft_r2c_1d(
                    naxes[2], inptr,
                    outptr, FFTWOPTMODE);
            for(uint32_t ii = 0;
                ii < xysize; ii++)
            {
                for(int i = 0;
                    i < naxes[2]; i++)
                {
                    inptr[i] =
                        imgin.im
                            ->array.F[
                                i * xysize
                                + ii];
                }
                fftwf_execute(plan);
                for(uint32_t i = 0;
                    i < naxesout[2]; i++)
                {
                    imgout.im
                        ->array.CF[
                            i * xysize
                            + ii].re =
                        outptr[i][0];
                    imgout.im
                        ->array.CF[
                            i * xysize
                            + ii].im =
                        outptr[i][1];
                }
            }
            free(inptr);
            free(outptr);
        }

        if(datatype == _DATATYPE_UINT16)
        {
            printf("UINT16 data type\n");
            inptr = (float *) malloc(
                sizeof(float) * naxes[2]);
            outptr =
                (fftwf_complex *)
                malloc(
                    sizeof(fftwf_complex)
                    * naxes[2]);

            plan =
                fftwf_plan_dft_r2c_1d(
                    naxes[2], inptr,
                    outptr, FFTWOPTMODE);
            for(uint32_t ii = 0;
                ii < xysize; ii++)
            {
                for(int i = 0;
                    i < naxes[2]; i++)
                {
                    inptr[i] =
                        1.0
                        * imgin.im
                              ->array
                              .UI16[
                                  i * xysize
                                  + ii];
                }
                fftwf_execute(plan);
                for(uint32_t i = 0;
                    i < naxesout[2]; i++)
                {
                    imgout.im
                        ->array.CF[
                            i * xysize
                            + ii].re =
                        outptr[i][0];
                    imgout.im
                        ->array.CF[
                            i * xysize
                            + ii].im =
                        outptr[i][1];
                }
            }
            free(inptr);
            free(outptr);
        }

        if(datatype == _DATATYPE_UINT32)
        {
            printf("UINT32 data type\n");
            inptr = (float *) malloc(
                sizeof(float) * naxes[2]);
            outptr =
                (fftwf_complex *)
                malloc(
                    sizeof(fftwf_complex)
                    * naxes[2]);

            plan =
                fftwf_plan_dft_r2c_1d(
                    naxes[2], inptr,
                    outptr, FFTWOPTMODE);
            for(uint32_t ii = 0;
                ii < xysize; ii++)
            {
                for(int i = 0;
                    i < naxes[2]; i++)
                {
                    inptr[i] =
                        1.0
                        * imgin.im
                              ->array
                              .UI32[
                                  i * xysize
                                  + ii];
                }
                fftwf_execute(plan);
                for(uint32_t i = 0;
                    i < naxesout[2]; i++)
                {
                    imgout.im
                        ->array.CF[
                            i * xysize
                            + ii].re =
                        outptr[i][0];
                    imgout.im
                        ->array.CF[
                            i * xysize
                            + ii].im =
                        outptr[i][1];
                }
            }
            free(inptr);
            free(outptr);
        }

        if(datatype == _DATATYPE_UINT64)
        {
            printf("UINT64 data type\n");
            inptr = (float *) malloc(
                sizeof(float) * naxes[2]);
            outptr =
                (fftwf_complex *)
                malloc(
                    sizeof(fftwf_complex)
                    * naxes[2]);

            plan =
                fftwf_plan_dft_r2c_1d(
                    naxes[2], inptr,
                    outptr, FFTWOPTMODE);
            for(uint32_t ii = 0;
                ii < xysize; ii++)
            {
                for(int i = 0;
                    i < naxes[2]; i++)
                {
                    inptr[i] =
                        1.0
                        * imgin.im
                              ->array
                              .UI64[
                                  i * xysize
                                  + ii];
                }
                fftwf_execute(plan);
                for(uint32_t i = 0;
                    i < naxesout[2]; i++)
                {
                    imgout.im
                        ->array.CF[
                            i * xysize
                            + ii].re =
                        outptr[i][0];
                    imgout.im
                        ->array.CF[
                            i * xysize
                            + ii].im =
                        outptr[i][1];
                }
            }
            free(inptr);
            free(outptr);
        }
    }

    if(OK == 0)
    {
        printf(
            "Error : image dimension"
            " not appropriate"
            " for FFT\n");
    }
    free(naxes);

    return imgout.ID;
}

imageID do1dfft(const char *__restrict in_name, const char *__restrict out_name)
{
    imageID IDout;

    IDout = FFT_do1dfft(in_name, out_name, -1);

    return (IDout);
}

imageID do1dffti(const char *__restrict in_name,
                 const char *__restrict out_name)
{
    imageID IDout;

    IDout = FFT_do1dfft(in_name, out_name, 1);

    return (IDout);
}

/* 2d complex fft */
// supports single and double precisions
imageID FFT_do2dfft(
    const char *in_name,
    const char *out_name,
    int dir)
{
    int        OK = 0;
    fftwf_plan plan;
    fftw_plan  plan_double;

    char ffttmpcpyname[
        STRINGMAXLEN_IMGNAME];

    IMGID imgin =
        imgid_make_from_name(in_name);
    resolveIMGID(&imgin,
                 ERRMODE_ABORT,
                 dcimg, dcnimg);

    long naxis = imgin.md->naxis;
    uint8_t datatype =
        imgin.md->datatype;

    int *naxes = (int *) malloc(
        naxis * sizeof(int));
    if(naxes == NULL)
    {
        PRINT_ERROR(
            "malloc returns NULL pointer");
        abort();
    }

    IMGID imgout =
        imgid_make_from_name(out_name);
    imgout.mdt->naxis = naxis;
    for(long i = 0; i < naxis; i++)
    {
        imgout.mdt->size[i] =
            imgin.md->size[i];
        naxes[i] =
            (int) imgin.md->size[i];
    }
    imgout.mdt->datatype = datatype;
    imgout.mdt->shared = dcshareddft;
    imgout.mdt->NBkw = NB_KEYWNODE_MAX;
    imgout.im = (IMAGE *) calloc(
        1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    /* swap first 2 axes for fftw */
    if(naxis > 1)
    {
        long tmp1 = naxes[0];
        naxes[0]  = naxes[1];
        naxes[1]  = tmp1;
    }

    if(naxis == 2)
    {
        OK = 1;

        if(datatype
           == _DATATYPE_COMPLEX_FLOAT)
        {
            plan = fftwf_plan_dft_2d(
                naxes[0], naxes[1],
                (fftwf_complex *)
                    imgin.im->array.CF,
                (fftwf_complex *)
                    imgout.im->array.CF,
                dir, FFTWOPTMODE);
            if(plan == NULL)
            {
                fprintf(
                    stdout,
                    "New FFT size"
                    " [do2dfft"
                    " %d x %d]:"
                    " optimizing ...",
                    naxes[1],
                    naxes[0]);
                fflush(stdout);

                WRITE_IMAGENAME(
                    ffttmpcpyname,
                    "_ffttmpcpyname_%d",
                    (int) getpid());
                copy_image_ID(
                    in_name,
                    ffttmpcpyname, 0);

                plan = fftwf_plan_dft_2d(
                    naxes[0], naxes[1],
                    (fftwf_complex *)
                        imgin.im
                            ->array.CF,
                    (fftwf_complex *)
                        imgout.im
                            ->array.CF,
                    dir, FFTWOPTMODE);
                copy_image_ID(
                    ffttmpcpyname,
                    in_name, 0);
                delete_image_ID(
                    ffttmpcpyname,
                    DELETE_IMAGE_ERRMODE_WARNING);
                export_wisdom();
                fprintf(stdout, "\n");
            }
            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
        }
        else
        {
            plan_double =
                fftw_plan_dft_2d(
                    naxes[0], naxes[1],
                    (fftw_complex *)
                        imgin.im
                            ->array.CD,
                    (fftw_complex *)
                        imgout.im
                            ->array.CD,
                    dir, FFTWOPTMODE);
            if(plan_double == NULL)
            {
                fprintf(
                    stdout,
                    "New FFT size"
                    " [do2dfft"
                    " %d x %d]:"
                    " optimizing ...",
                    naxes[1],
                    naxes[0]);
                fflush(stdout);

                WRITE_IMAGENAME(
                    ffttmpcpyname,
                    "_ffttmpcpyname_%d",
                    (int) getpid());
                copy_image_ID(
                    in_name,
                    ffttmpcpyname, 0);

                plan_double =
                    fftw_plan_dft_2d(
                        naxes[0],
                        naxes[1],
                        (fftw_complex *)
                            imgin.im
                                ->array.CD,
                        (fftw_complex *)
                            imgout.im
                                ->array.CD,
                        dir,
                        FFTWOPTMODE);
                copy_image_ID(
                    ffttmpcpyname,
                    in_name, 0);
                delete_image_ID(
                    ffttmpcpyname,
                    DELETE_IMAGE_ERRMODE_WARNING);
                export_wisdom();
                fprintf(stdout, "\n");
            }
            fftw_execute(plan_double);
            fftw_destroy_plan(
                plan_double);
        }
    }

    if(naxis == 3)
    {
        OK = 1;
        if(datatype
           == _DATATYPE_COMPLEX_FLOAT)
        {
            plan = fftwf_plan_many_dft(
                2, naxes, naxes[2],
                (fftwf_complex *)
                    imgin.im->array.CF,
                NULL, 1,
                naxes[0] * naxes[1],
                (fftwf_complex *)
                    imgout.im->array.CF,
                NULL, 1,
                naxes[0] * naxes[1],
                dir, FFTWOPTMODE);
            if(plan == NULL)
            {
                fprintf(
                    stdout,
                    "New FFT size"
                    " [do2dfft"
                    " %d x %d x %d]:"
                    " optimizing ...",
                    naxes[1],
                    naxes[0],
                    naxes[2]);
                fflush(stdout);

                WRITE_IMAGENAME(
                    ffttmpcpyname,
                    "_ffttmpcpyname_%d",
                    (int) getpid());
                copy_image_ID(
                    in_name,
                    ffttmpcpyname, 0);

                plan =
                    fftwf_plan_many_dft(
                        2, naxes,
                        naxes[2],
                        (fftwf_complex *)
                            imgin.im
                                ->array.CF,
                        NULL, 1,
                        naxes[0]
                        * naxes[1],
                        (fftwf_complex *)
                            imgout.im
                                ->array.CF,
                        NULL, 1,
                        naxes[0]
                        * naxes[1],
                        dir,
                        FFTWOPTMODE);
                copy_image_ID(
                    ffttmpcpyname,
                    in_name, 0);
                delete_image_ID(
                    ffttmpcpyname,
                    DELETE_IMAGE_ERRMODE_WARNING);
                export_wisdom();
                fprintf(stdout, "\n");
            }
            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
        }
        else
        {
            plan_double =
                fftw_plan_many_dft(
                    2, naxes, naxes[2],
                    (fftw_complex *)
                        imgin.im
                            ->array.CD,
                    NULL, 1,
                    naxes[0] * naxes[1],
                    (fftw_complex *)
                        imgout.im
                            ->array.CD,
                    NULL, 1,
                    naxes[0] * naxes[1],
                    dir, FFTWOPTMODE);
            if(plan_double == NULL)
            {
                fprintf(
                    stdout,
                    "New FFT size"
                    " [do2dfft"
                    " %d x %d x %d]:"
                    " optimizing ...",
                    naxes[1],
                    naxes[0],
                    naxes[2]);
                fflush(stdout);

                WRITE_IMAGENAME(
                    ffttmpcpyname,
                    "_ffttmpcpyname_%d",
                    (int) getpid());
                copy_image_ID(
                    in_name,
                    ffttmpcpyname, 0);

                plan_double =
                    fftw_plan_many_dft(
                        2, naxes,
                        naxes[2],
                        (fftw_complex *)
                            imgin.im
                                ->array.CD,
                        NULL, 1,
                        naxes[0]
                        * naxes[1],
                        (fftw_complex *)
                            imgout.im
                                ->array.CD,
                        NULL, 1,
                        naxes[0]
                        * naxes[1],
                        dir,
                        FFTWOPTMODE);
                copy_image_ID(
                    ffttmpcpyname,
                    in_name, 0);
                delete_image_ID(
                    ffttmpcpyname,
                    DELETE_IMAGE_ERRMODE_WARNING);
                export_wisdom();
                fprintf(stdout, "\n");
            }
            fftw_execute(plan_double);
            fftw_destroy_plan(
                plan_double);
        }
    }

    if(OK == 0)
    {
        printf(
            "Error : image dimension"
            " not appropriate"
            " for FFT\n");
    }

    free(naxes);

    return imgout.ID;
}

imageID do2dfft(const char *__restrict in_name, const char *__restrict out_name)
{
    imageID IDout;

    IDout = FFT_do2dfft(in_name, out_name, -1);

    return (IDout);
}

imageID do2dffti(const char *__restrict in_name,
                 const char *__restrict out_name)
{
    imageID IDout;

    IDout = FFT_do2dfft(in_name, out_name, 1);

    return (IDout);
}

/* real fft : real to complex */
// supports single and double precisions
imageID FFT_do2drfft(
    const char *__restrict in_name,
    const char *__restrict out_name,
    int dir)
{
    int *naxes; /* int format for fftw */

    int        OK = 0;
    fftwf_plan plan;
    fftw_plan  plan_double;
    long       tmp1;

    uint8_t datatypeout;

    IMGID imgin =
        imgid_make_from_name(in_name);
    resolveIMGID(&imgin,
                 ERRMODE_ABORT,
                 dcimg, dcnimg);

    uint8_t datatype =
        imgin.md->datatype;
    long naxis = imgin.md->naxis;

    naxes = (int *) malloc(
        naxis * sizeof(uint32_t));
    if(naxes == NULL)
    {
        PRINT_ERROR(
            "malloc returns NULL pointer");
        abort();
    }

    uint32_t naxestmp[3];
    for(int i = 0; i < naxis; i++)
    {
        naxes[i] =
            (int) imgin.md->size[i];
        naxestmp[i] =
            imgin.md->size[i];
        if(i == 0)
        {
            naxestmp[i] =
                imgin.md->size[i] / 2
                + 1;
        }
    }

    char ffttmpname[
        STRINGMAXLEN_IMGNAME];
    WRITE_IMAGENAME(ffttmpname,
                    "_ffttmp_%d",
                    (int) getpid());

    if(datatype == _DATATYPE_FLOAT)
    {
        datatypeout =
            _DATATYPE_COMPLEX_FLOAT;
    }
    else
    {
        datatypeout =
            _DATATYPE_COMPLEX_DOUBLE;
    }

    IMGID imgtmp =
        imgid_make_from_name(ffttmpname);
    imgtmp.mdt->naxis = naxis;
    for(int i = 0; i < naxis; i++)
    {
        imgtmp.mdt->size[i] =
            naxestmp[i];
    }
    imgtmp.mdt->datatype = datatypeout;
    imgtmp.mdt->shared = dcshareddft;
    imgtmp.mdt->NBkw = NB_KEYWNODE_MAX;
    imgtmp.im = (IMAGE *) calloc(
        1, sizeof(IMAGE));
    imgid_mkimage(&imgtmp);

    IMGID imgout =
        imgid_make_from_name(out_name);
    imgout.mdt->naxis = naxis;
    for(int i = 0; i < naxis; i++)
    {
        imgout.mdt->size[i] =
            imgin.md->size[i];
    }
    imgout.mdt->datatype = datatypeout;
    imgout.mdt->shared = dcshareddft;
    imgout.mdt->NBkw = NB_KEYWNODE_MAX;
    imgout.im = (IMAGE *) calloc(
        1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    if(naxis == 2)
    {
        OK = 1;

        if(datatype == _DATATYPE_FLOAT)
        {
            plan = fftwf_plan_dft_r2c_2d(
                       (int) naxes[1],
                       (int) naxes[0],
                       imgin.im->array.F,
                       (fftwf_complex *) imgtmp.im->array.CF,
                       FFTWOPTMODE);
            if(plan == NULL)
            {
                // if ( Debug > 2)
                fprintf(stdout,
                        "New FFT size [do2drfft %d x %d]: optimizing ...",
                        naxes[1],
                        naxes[0]);
                fflush(stdout);

                char ffttmpcpyname[STRINGMAXLEN_IMGNAME];
                WRITE_IMAGENAME(ffttmpcpyname, "_ffttmpcpy_%d", (int) getpid());

                copy_image_ID(in_name, ffttmpcpyname, 0);

                plan = fftwf_plan_dft_r2c_2d(
                           naxes[1],
                           naxes[0],
                           imgin.im->array.F,
                           (fftwf_complex *) imgtmp.im->array.CF,
                           FFTWOPTMODE);
                copy_image_ID(ffttmpcpyname, in_name, 0);
                delete_image_ID(ffttmpcpyname, DELETE_IMAGE_ERRMODE_WARNING);
                export_wisdom();
                fprintf(stdout, "\n");
            }
            fftwf_execute(plan);
            fftwf_destroy_plan(plan);

            if(dir == -1)
            {
                for(uint32_t ii = 0; ii < (uint32_t)(naxes[0] / 2 + 1); ii++)
                    for(uint32_t jj = 0; jj < (uint32_t) naxes[1]; jj++)
                    {
                        imgout.im->array.CF[jj * naxes[0] + ii] =
                            imgtmp.im->array.CF[jj * naxestmp[0] + ii];
                    }

                for(uint32_t ii = 1; ii < (uint32_t)(naxes[0] / 2 + 1); ii++)
                {
                    uint32_t jj = 0;
                    imgout.im->array.CF[jj * naxes[0] + (naxes[0] - ii)]
                    .re =
                        imgtmp.im->array.CF[jj * naxestmp[0] + ii].re;
                    imgout.im->array.CF[jj * naxes[0] + (naxes[0] - ii)]
                    .im =
                        -imgtmp.im->array.CF[jj * naxestmp[0] + ii].im;
                    for(uint32_t jj = 1; jj < (uint32_t) naxes[1]; jj++)
                    {
                        imgout.im->array.CF[jj * naxes[0] + (naxes[0] - ii)]
                        .re =
                            imgtmp.im->array.CF[(naxes[1] - jj) * naxestmp[0] + ii]
                            .re;
                        imgout.im->array.CF[jj * naxes[0] + (naxes[0] - ii)]
                        .im =
                            -imgtmp.im->array.CF[(naxes[1] - jj) * naxestmp[0] + ii]
                            .im;
                    }
                }
            }
        }
        else
        {
            plan_double = fftw_plan_dft_r2c_2d(
                              naxes[1],
                              naxes[0],
                              imgin.im->array.D,
                              (fftw_complex *) imgtmp.im->array.CD,
                              FFTWOPTMODE);
            if(plan_double == NULL)
            {
                // if ( Debug > 2)
                fprintf(stdout,
                        "New FFT size [do2drfft %d x %d]: optimizing ...",
                        naxes[1],
                        naxes[0]);
                fflush(stdout);

                char ffttmpcpyname[STRINGMAXLEN_IMGNAME];
                WRITE_IMAGENAME(ffttmpcpyname, "_ffttmpcpy_%d", (int) getpid());

                copy_image_ID(in_name, ffttmpcpyname, 0);

                plan_double = fftw_plan_dft_r2c_2d(
                                  naxes[1],
                                  naxes[0],
                                  imgin.im->array.D,
                                  (fftw_complex *) imgtmp.im->array.CD,
                                  FFTWOPTMODE);
                copy_image_ID(ffttmpcpyname, in_name, 0);
                delete_image_ID(ffttmpcpyname, DELETE_IMAGE_ERRMODE_WARNING);
                export_wisdom();
                fprintf(stdout, "\n");
            }
            fftw_execute(plan_double);
            fftw_destroy_plan(plan_double);

            if(dir == -1)
            {
                for(uint32_t ii = 0; ii < (uint32_t)(naxes[0] / 2 + 1); ii++)
                    for(uint32_t jj = 0; jj < (uint32_t) naxes[1]; jj++)
                    {
                        imgout.im->array.CD[jj * naxes[0] + ii] =
                            imgtmp.im->array.CD[jj * naxestmp[0] + ii];
                    }

                for(uint32_t ii = 1; ii < (uint32_t)(naxes[0] / 2 + 1); ii++)
                {
                    uint32_t jj = 0;
                    imgout.im->array.CD[jj * naxes[0] + (naxes[0] - ii)]
                    .re =
                        imgtmp.im->array.CD[jj * naxestmp[0] + ii].re;
                    imgout.im->array.CD[jj * naxes[0] + (naxes[0] - ii)]
                    .im =
                        -imgtmp.im->array.CD[jj * naxestmp[0] + ii].im;
                    for(uint32_t jj = 1; jj < (uint32_t) naxes[1]; jj++)
                    {
                        imgout.im->array.CD[jj * naxes[0] + (naxes[0] - ii)]
                        .re =
                            imgtmp.im->array.CD[(naxes[1] - jj) * naxestmp[0] + ii]
                            .re;
                        imgout.im->array.CD[jj * naxes[0] + (naxes[0] - ii)]
                        .im =
                            -imgtmp.im->array.CD[(naxes[1] - jj) * naxestmp[0] + ii]
                            .im;
                    }
                }
            }
        }
    }
    if(naxis == 3)
    {
        OK = 1;
        //idist = naxes[0]*naxes[1];

        // swapping first 2 axis
        tmp1     = naxes[0];
        naxes[0] = naxes[1];
        naxes[1] = tmp1;

        if(datatype == _DATATYPE_FLOAT)
        {
            plan = fftwf_plan_many_dft_r2c(
                       2,
                       naxes,
                       naxes[2],
                       imgin.im->array.F,
                       NULL,
                       1,
                       naxes[0] * naxes[1],
                       (fftwf_complex *) imgout.im->array.CF,
                       NULL,
                       1,
                       naxes[0] * naxes[1],
                       FFTWOPTMODE);
            if(plan == NULL)
            {
                //	  if ( Debug > 2) fprintf(stdout,"New FFT size [do2drfft %d x %d x %d]: optimizing ...",naxes[1],naxes[0],naxes[2]);
                fflush(stdout);

                char ffttmpcpyname[STRINGMAXLEN_IMGNAME];
                WRITE_IMAGENAME(ffttmpcpyname, "_ffttmpcpy_%d", (int) getpid());
                copy_image_ID(in_name, ffttmpcpyname, 0);

                plan = fftwf_plan_many_dft_r2c(
                           2,
                           naxes,
                           naxes[2],
                           imgin.im->array.F,
                           NULL,
                           1,
                           naxes[0] * naxes[1],
                           (fftwf_complex *) imgout.im->array.CF,
                           NULL,
                           1,
                           naxes[0] * naxes[1],
                           FFTWOPTMODE);

                copy_image_ID(ffttmpcpyname, in_name, 0);
                delete_image_ID(ffttmpcpyname, DELETE_IMAGE_ERRMODE_WARNING);
                export_wisdom();
                fprintf(stdout, "\n");
            }

            fftwf_execute(plan);
            fftwf_destroy_plan(plan);

            if(dir == -1)
            {
                // unswapping first 2 axis
                tmp1     = naxes[0];
                naxes[0] = naxes[1];
                naxes[1] = tmp1;

                for(uint32_t ii = 0; ii < (uint32_t)(naxes[0] / 2 + 1); ii++)
                    for(uint32_t jj = 0; jj < (uint32_t) naxes[1]; jj++)
                        for(uint32_t kk = 0; kk < (uint32_t) naxes[2]; kk++)
                        {
                            imgout.im->array.CF[naxes[0] * naxes[1] * kk +
                                               jj * naxes[0] + ii] =
                                          imgtmp.im->array.CF[naxestmp[0] * naxestmp[1] * kk +
                                                                jj * naxestmp[0] + ii];
                            if(ii != 0)
                            {
                                imgout.im->array.CF[naxes[0] * naxes[1] * kk +
                                                   jj * naxes[0] + (naxes[0] - ii)] =
                                              imgtmp.im->array
                                              .CF[naxestmp[0] * naxestmp[1] * kk +
                                                              jj * naxestmp[0] + ii];
                            }
                        }
            }
        }
        else
        {
            plan_double = fftw_plan_many_dft_r2c(
                              2,
                              naxes,
                              naxes[2],
                              imgin.im->array.D,
                              NULL,
                              1,
                              naxes[0] * naxes[1],
                              (fftw_complex *) imgout.im->array.CD,
                              NULL,
                              1,
                              naxes[0] * naxes[1],
                              FFTWOPTMODE);
            if(plan == NULL)
            {
                //	  if ( Debug > 2) fprintf(stdout,"New FFT size [do2drfft %d x %d x %d]: optimizing ...",naxes[1],naxes[0],naxes[2]);
                //				fflush(stdout);

                char ffttmpcpyname[STRINGMAXLEN_IMGNAME];
                WRITE_IMAGENAME(ffttmpcpyname, "_ffttmpcpy_%d", (int) getpid());

                copy_image_ID(in_name, ffttmpcpyname, 0);

                plan_double = fftw_plan_many_dft_r2c(
                                  2,
                                  naxes,
                                  naxes[2],
                                  imgin.im->array.D,
                                  NULL,
                                  1,
                                  naxes[0] * naxes[1],
                                  (fftw_complex *) imgout.im->array.CD,
                                  NULL,
                                  1,
                                  naxes[0] * naxes[1],
                                  FFTWOPTMODE);

                copy_image_ID(ffttmpcpyname, in_name, 0);
                delete_image_ID(ffttmpcpyname, DELETE_IMAGE_ERRMODE_WARNING);
                export_wisdom();
                fprintf(stdout, "\n");
            }

            fftwf_execute(plan);
            fftwf_destroy_plan(plan);

            if(dir == -1)
            {
                // unswapping first 2 axis
                tmp1     = naxes[0];
                naxes[0] = naxes[1];
                naxes[1] = tmp1;

                for(uint32_t ii = 0; ii < (uint32_t)(naxes[0] / 2 + 1); ii++)
                    for(uint32_t jj = 0; jj < (uint32_t) naxes[1]; jj++)
                        for(uint32_t kk = 0; kk < (uint32_t) naxes[2]; kk++)
                        {
                            imgout.im->array.CD[naxes[0] * naxes[1] * kk +
                                               jj * naxes[0] + ii] =
                                          imgtmp.im->array.CD[naxestmp[0] * naxestmp[1] * kk +
                                                                jj * naxestmp[0] + ii];
                            if(ii != 0)
                            {
                                imgout.im->array.CD[naxes[0] * naxes[1] * kk +
                                                   jj * naxes[0] + (naxes[0] - ii)] =
                                              imgtmp.im->array
                                              .CD[naxestmp[0] * naxestmp[1] * kk +
                                                              jj * naxestmp[0] + ii];
                            }
                        }
            }
        }
    }

    if(OK == 0)
    {
        printf("Error : image dimension not appropriate for FFT\n");
    }

    delete_image_ID(ffttmpname,
                    DELETE_IMAGE_ERRMODE_WARNING);

    free(naxes);

    return imgout.ID;
}

imageID do2drfft(const char *in_name, const char *out_name)
{
    imageID IDout;

    IDout = FFT_do2drfft(in_name, out_name, -1);

    return (IDout);
}

imageID do2drffti(const char *in_name, const char *out_name)
{
    imageID IDout;

    IDout = FFT_do2drfft(in_name, out_name, 1);

    return (IDout);
}


/* ================================================================
 * 4b. Standalone-friendly FFT step
 * ============================================================= */

static void MILK_HOT fpsexec(
    IMAGE *imgin,
    IMAGE *imgout,
    int dir)
{
    int naxes[2] = {
        (int) imgin->md[0].size[1],
        (int) imgin->md[0].size[0]
    };
    if (imgin->md[0].datatype
        == _DATATYPE_COMPLEX_FLOAT)
    {
        fftwf_plan plan =
            fftwf_plan_dft_2d(
                naxes[0], naxes[1],
                (fftwf_complex *)
                    imgin->array.CF,
                (fftwf_complex *)
                    imgout->array.CF,
                dir, FFTW_ESTIMATE);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
    }
    else if (imgin->md[0].datatype
             == _DATATYPE_COMPLEX_DOUBLE)
    {
        fftw_plan plan =
            fftw_plan_dft_2d(
                naxes[0], naxes[1],
                (fftw_complex *)
                    imgin->array.CD,
                (fftw_complex *)
                    imgout->array.CD,
                dir, FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);
    }
}


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    FPS_app_info,
    FPS_PARAMS,
    compute_function)
#endif