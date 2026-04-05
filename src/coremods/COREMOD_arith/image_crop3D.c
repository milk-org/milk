/**
 * @file    image_crop.c
 * @brief   crop functions
 *
 * Uses FPS V2 framework.
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"

#include "COREMOD_memory/COREMOD_memory.h"

/* forward decls */
imageID arith_image_crop(
    const char *ID_name,
    const char *ID_out,
    long *start, long *end,
    long cropdim);

imageID arith_image_extract2D(
    const char *in_name,
    const char *out_name,
    long size_x, long size_y,
    long xstart, long ystart);

imageID arith_image_extract3D(
    const char *in_name,
    const char *out_name,
    long size_x, long size_y,
    long size_z,
    long xstart, long ystart,
    long zstart);


/* ================================================================
 *  PARAMS
 * ============================================================= */

static char p_inname[
    FUNCTION_PARAMETER_STRMAXLEN]
    = "im";
static char p_outname[
    FUNCTION_PARAMETER_STRMAXLEN]
    = "ime";
static long long p_sizex = 256;
static long long p_sizey = 256;
static long long p_sizez = 5;
static long long p_xstart = 100;
static long long p_ystart = 100;
static long long p_zstart = 0;


/* ================================================================
 *  CMD 1: extractim (6 args)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_2d = {
    .fps_name    = "extractim",
    .cmdkey      = "extractim",
    .description = "crop 2D image"
};

#define FPS_PARAMS_2D(X) \
    X(".inname", p_inname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input image") \
    X(".outname", p_outname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output image") \
    X(".sizex", &p_sizex, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "size X") \
    X(".sizey", &p_sizey, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "size Y") \
    X(".xstart", &p_xstart, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "X start") \
    X(".ystart", &p_ystart, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "Y start")

static CLICMDDATA CLIcmddata_2d = {
    "", "", CLICMD_FIELDS_NOPARAM
};
FPS_CMDSETTINGS_INIT(2d, CLIcmddata_2d, FPS_app_info_2d)

static errno_t compute_2d()
{
    arith_image_extract2D(
        p_inname, p_outname,
        p_sizex, p_sizey,
        p_xstart, p_ystart);
    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 2: extract3Dim (8 args, primary)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "extract3Dim",
    .cmdkey      = "extract3Dim",
    .description = "crop 3D image"
};

#define FPS_PARAMS(X) \
    X(".inname", p_inname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input image") \
    X(".outname", p_outname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output image") \
    X(".sizex", &p_sizex, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "size X") \
    X(".sizey", &p_sizey, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "size Y") \
    X(".sizez", &p_sizez, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "size Z") \
    X(".xstart", &p_xstart, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "X start") \
    X(".ystart", &p_ystart, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "Y start") \
    X(".zstart", &p_zstart, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "Z start")

static FPS_CLI_BINDING my_bindings[] = {
    FPS_PARAMS(FPS_X_BINDING)
};

static const int nb_bindings =
    sizeof(my_bindings) /
    sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = {
    FPS_PARAMS(FPS_X_FARG)
};

static CLICMDDATA CLIcmddata = {
    "", "", CLICMD_FIELDS_DEFAULTS
};

FPS_CMDSETTINGS_INIT(main, CLIcmddata, FPS_app_info)

static MILK_HOT errno_t compute_function()
{
    arith_image_extract3D(
        p_inname, p_outname,
        p_sizex, p_sizey, p_sizez,
        p_xstart, p_ystart, p_zstart);
    return RETURN_SUCCESS;
}


/* ================================================================
 *  REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)

static FPS_CLI_BINDING bindings_2d[] = {
    FPS_PARAMS_2D(FPS_X_BINDING)
};
static const int nb_bindings_2d =
    sizeof(bindings_2d) /
    sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF farg_2d[] = {
    FPS_PARAMS_2D(FPS_X_FARG)
};

static errno_t CLIfunction_2d(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_2d,
        farg_2d, &CLIcmddata_2d,
        bindings_2d, nb_bindings_2d,
        compute_2d);
}

static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_COREMOD_arith__image_crop()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    safe_fps_fill_farg_examples(
        farg_2d, bindings_2d,
        nb_bindings_2d);

    {
        int cmdi = RegisterCLIcmd(
            CLIcmddata_2d,
            CLIfunction_2d);
        CLIcmddata_2d.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }
    {
        int cmdi = RegisterCLIcmd(
            CLIcmddata, CLIfunction);
        CLIcmddata.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }

    return RETURN_SUCCESS;
}
#endif

imageID arith_image_crop(const char *ID_name,
                         const char *ID_out,
                         long       *start,
                         long       *end,
                         long        cropdim)
{
    long      naxis;
    long      i;
    uint32_t *naxes    = NULL;
    uint32_t *naxesout = NULL;
    uint8_t   datatype;

    long start_c[3];
    long end_c[3];

    for(i = 0; i < 3; i++)
    {
        start_c[i] = 0;
        end_c[i]   = 0;
    }

    IMGID imgin =
        imgid_make_from_name(ID_name);
    resolveIMGID(&imgin,
                 ERRMODE_ABORT,
                 dcimg, dcnimg);

    naxis = imgin.md->naxis;
    if(naxis < 1)
    {
        PRINT_ERROR("naxis < 1");
        exit(0);
    }
    naxes = (uint32_t *) malloc(
        sizeof(uint32_t) * naxis);
    if(naxes == NULL)
    {
        PRINT_ERROR(
            "malloc() error,"
            " naxis = %ld", naxis);
        exit(0);
    }

    naxesout = (uint32_t *) malloc(
        sizeof(uint32_t) * naxis);
    if(naxesout == NULL)
    {
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    datatype = imgin.md->datatype;

    naxes[0]    = 0;
    naxesout[0] = 0;
    for(i = 0; i < naxis; i++)
    {
        naxes[i] =
            imgin.md->size[i];
        naxesout[i] =
            end[i] - start[i];
    }

    IMGID imgout =
        imgid_make_from_name(ID_out);
    imgout.mdt->naxis = naxis;
    for(i = 0; i < naxis; i++)
    {
        imgout.mdt->size[i] =
            naxesout[i];
    }
    imgout.mdt->datatype = datatype;
    imgout.mdt->shared = dcshareddft;
    imgout.mdt->NBkw = NB_KEYWNODE_MAX;
    imgout.im = (IMAGE *) calloc(
        1, sizeof(IMAGE));
    imgid_mkimage(&imgout);
    imageID IDout = imgout.ID;

    start_c[0] = start[0];
    if(start_c[0] < 0)
    {
        start_c[0] = 0;
    }
    end_c[0] = end[0];
    if(end_c[0] > naxes[0])
    {
        end_c[0] = naxes[0];
    }
    if(naxis > 1)
    {
        start_c[1] = start[1];
        if(start_c[1] < 0)
        {
            start_c[1] = 0;
        }
        end_c[1] = end[1];
        if(end_c[1] > naxes[1])
        {
            end_c[1] = naxes[1];
        }
    }
    if(naxis > 2)
    {
        start_c[2] = start[2];
        if(start_c[2] < 0)
        {
            start_c[2] = 0;
        }
        end_c[2] = end[2];
        if(end_c[2] > naxes[2])
        {
            end_c[2] = naxes[2];
        }
    }

    printf("CROP: \n");
    for(i = 0; i < 3; i++)
    {
        printf("axis %ld: %ld -> %ld\n", i, start_c[i], end_c[i]);
    }

    if(cropdim != naxis)
    {
        printf(
            "Error (arith_image_crop): cropdim [%ld] and naxis [%ld] are "
            "different\n",
            cropdim,
            naxis);
    }

    if(naxis == 1)
    {
        if(datatype == _DATATYPE_FLOAT)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
            {
                imgout.im->array.F[ii - start[0]] =
                    imgin.im->array.F[ii];
            }
        }
        else if(datatype == _DATATYPE_DOUBLE)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
            {
                imgout.im->array.D[ii - start[0]] =
                    imgin.im->array.D[ii];
            }
        }
        else if(datatype == _DATATYPE_COMPLEX_FLOAT)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
            {
                imgout.im->array.CF[ii - start[0]].re =
                    imgin.im->array.CF[ii].re;
                imgout.im->array.CF[ii - start[0]].im =
                    imgin.im->array.CF[ii].im;
            }
        }
        else if(datatype == _DATATYPE_COMPLEX_DOUBLE)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
            {
                imgout.im->array.CD[ii - start[0]].re =
                    imgin.im->array.CD[ii].re;
                imgout.im->array.CD[ii - start[0]].im =
                    imgin.im->array.CD[ii].im;
            }
        }
        else if(datatype == _DATATYPE_UINT8)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
            {
                imgout.im->array.UI8[ii - start[0]] =
                    imgin.im->array.UI8[ii];
            }
        }
        else if(datatype == _DATATYPE_UINT16)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
            {
                imgout.im->array.UI16[ii - start[0]] =
                    imgin.im->array.UI16[ii];
            }
        }
        else if(datatype == _DATATYPE_UINT32)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
            {
                imgout.im->array.UI32[ii - start[0]] =
                    imgin.im->array.UI32[ii];
            }
        }
        else if(datatype == _DATATYPE_UINT64)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
            {
                imgout.im->array.UI64[ii - start[0]] =
                    imgin.im->array.UI64[ii];
            }
        }
        else if(datatype == _DATATYPE_INT8)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
            {
                imgout.im->array.SI8[ii - start[0]] =
                    imgin.im->array.SI8[ii];
            }
        }
        else if(datatype == _DATATYPE_INT16)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
            {
                imgout.im->array.SI16[ii - start[0]] =
                    imgin.im->array.SI16[ii];
            }
        }
        else if(datatype == _DATATYPE_INT32)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
            {
                imgout.im->array.SI32[ii - start[0]] =
                    imgin.im->array.SI32[ii];
            }
        }
        else if(datatype == _DATATYPE_INT64)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
            {
                imgout.im->array.SI64[ii - start[0]] =
                    imgin.im->array.SI64[ii];
            }
        }
        else
        {
            PRINT_ERROR("invalid data type");
            exit(0);
        }
    }
    if(naxis == 2)
    {
        if(datatype == _DATATYPE_FLOAT)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    imgout.im->array.F[(jj - start[1]) * naxesout[0] +
                                              (ii - start[0])] =
                                                  imgin.im->array.F[jj * naxes[0] + ii];
        }
        else if(datatype == _DATATYPE_DOUBLE)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    imgout.im->array.D[(jj - start[1]) * naxesout[0] +
                                              (ii - start[0])] =
                                                  imgin.im->array.D[jj * naxes[0] + ii];
        }
        else if(datatype == _DATATYPE_COMPLEX_FLOAT)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                {
                    imgout.im->array
                    .CF[(jj - start[1]) * naxesout[0] + (ii - start[0])]
                    .re = imgin.im->array.CF[jj * naxes[0] + ii].re;
                    imgout.im->array
                    .CF[(jj - start[1]) * naxesout[0] + (ii - start[0])]
                    .im = imgin.im->array.CF[jj * naxes[0] + ii].im;
                }
        }
        else if(datatype == _DATATYPE_COMPLEX_DOUBLE)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                {
                    imgout.im->array
                    .CD[(jj - start[1]) * naxesout[0] + (ii - start[0])]
                    .re = imgin.im->array.CD[jj * naxes[0] + ii].re;
                    imgout.im->array
                    .CD[(jj - start[1]) * naxesout[0] + (ii - start[0])]
                    .im = imgin.im->array.CD[jj * naxes[0] + ii].im;
                }
        }
        else if(datatype == _DATATYPE_UINT8)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    imgout.im->array.UI8[(jj - start[1]) * naxesout[0] +
                                                (ii - start[0])] =
                                                    imgin.im->array.UI8[jj * naxes[0] + ii];
        }
        else if(datatype == _DATATYPE_UINT16)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    imgout.im->array.UI16[(jj - start[1]) * naxesout[0] +
                                                 (ii - start[0])] =
                                                     imgin.im->array.UI16[jj * naxes[0] + ii];
        }
        else if(datatype == _DATATYPE_UINT32)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    imgout.im->array.UI32[(jj - start[1]) * naxesout[0] +
                                                 (ii - start[0])] =
                                                     imgin.im->array.UI32[jj * naxes[0] + ii];
        }
        else if(datatype == _DATATYPE_UINT64)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    imgout.im->array.UI64[(jj - start[1]) * naxesout[0] +
                                                 (ii - start[0])] =
                                                     imgin.im->array.UI64[jj * naxes[0] + ii];
        }
        else if(datatype == _DATATYPE_INT8)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    imgout.im->array.SI8[(jj - start[1]) * naxesout[0] +
                                                (ii - start[0])] =
                                                    imgin.im->array.SI8[jj * naxes[0] + ii];
        }
        else if(datatype == _DATATYPE_INT16)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    imgout.im->array.SI16[(jj - start[1]) * naxesout[0] +
                                                 (ii - start[0])] =
                                                     imgin.im->array.SI16[jj * naxes[0] + ii];
        }
        else if(datatype == _DATATYPE_INT32)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    imgout.im->array.SI32[(jj - start[1]) * naxesout[0] +
                                                 (ii - start[0])] =
                                                     imgin.im->array.SI32[jj * naxes[0] + ii];
        }
        else if(datatype == _DATATYPE_INT64)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    imgout.im->array.SI64[(jj - start[1]) * naxesout[0] +
                                                 (ii - start[0])] =
                                                     imgin.im->array.SI64[jj * naxes[0] + ii];
        }
        else
        {
            PRINT_ERROR("invalid data type");
            exit(0);
        }
    }
    if(naxis == 3)
    {
        //	printf("naxis = 3\n");
        if(datatype == _DATATYPE_FLOAT)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    for(uint32_t kk = start_c[2]; kk < end_c[2]; kk++)
                    {
                        imgout.im->array.F
                        [(kk - start[2]) * naxesout[0] * naxesout[1] +
                                         (jj - start[1]) * naxesout[0] + (ii - start[0])] =
                             imgin.im->array.F[kk * naxes[0] * naxes[1] +
                                                      jj * naxes[0] + ii];
                    }
        }
        else if(datatype == _DATATYPE_DOUBLE)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    for(uint32_t kk = start_c[2]; kk < end_c[2]; kk++)
                    {
                        imgout.im->array.D
                        [(kk - start[2]) * naxesout[0] * naxesout[1] +
                                         (jj - start[1]) * naxesout[0] + (ii - start[0])] =
                             imgin.im->array.D[kk * naxes[0] * naxes[1] +
                                                      jj * naxes[0] + ii];
                    }
        }
        else if(datatype == _DATATYPE_COMPLEX_FLOAT)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    for(uint32_t kk = start_c[2]; kk < end_c[2]; kk++)
                    {
                        imgout.im->array
                        .CF[(kk - start[2]) * naxesout[0] * naxesout[1] +
                                            (jj - start[1]) * naxesout[0] + (ii - start[0])]
                        .re = imgin.im->array
                              .CF[kk * naxes[0] * naxes[1] +
                                     jj * naxes[0] + ii]
                              .re;
                        imgout.im->array
                        .CF[(kk - start[2]) * naxesout[0] * naxesout[1] +
                                            (jj - start[1]) * naxesout[0] + (ii - start[0])]
                        .im = imgin.im->array
                              .CF[kk * naxes[0] * naxes[1] +
                                     jj * naxes[0] + ii]
                              .im;
                    }
        }
        else if(datatype == _DATATYPE_COMPLEX_DOUBLE)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    for(uint32_t kk = start_c[2]; kk < end_c[2]; kk++)
                    {
                        imgout.im->array
                        .CD[(kk - start[2]) * naxesout[0] * naxesout[1] +
                                            (jj - start[1]) * naxesout[0] + (ii - start[0])]
                        .re = imgin.im->array
                              .CD[kk * naxes[0] * naxes[1] +
                                     jj * naxes[0] + ii]
                              .re;
                        imgout.im->array
                        .CD[(kk - start[2]) * naxesout[0] * naxesout[1] +
                                            (jj - start[1]) * naxesout[0] + (ii - start[0])]
                        .im = imgin.im->array
                              .CD[kk * naxes[0] * naxes[1] +
                                     jj * naxes[0] + ii]
                              .im;
                    }
        }
        else if(datatype == _DATATYPE_UINT8)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    for(uint32_t kk = start_c[2]; kk < end_c[2]; kk++)
                    {
                        imgout.im->array.UI8
                        [(kk - start[2]) * naxesout[0] * naxesout[1] +
                                         (jj - start[1]) * naxesout[0] + (ii - start[0])] =
                             imgin.im->array.UI8[kk * naxes[0] * naxes[1] +
                                           jj * naxes[0] + ii];
                    }
        }
        else if(datatype == _DATATYPE_UINT16)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    for(uint32_t kk = start_c[2]; kk < end_c[2]; kk++)
                    {
                        imgout.im->array.UI16
                        [(kk - start[2]) * naxesout[0] * naxesout[1] +
                                         (jj - start[1]) * naxesout[0] + (ii - start[0])] =
                             imgin.im->array.UI16[kk * naxes[0] * naxes[1] +
                                            jj * naxes[0] + ii];
                    }
        }
        else if(datatype == _DATATYPE_UINT32)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    for(uint32_t kk = start_c[2]; kk < end_c[2]; kk++)
                    {
                        imgout.im->array.UI32
                        [(kk - start[2]) * naxesout[0] * naxesout[1] +
                                         (jj - start[1]) * naxesout[0] + (ii - start[0])] =
                             imgin.im->array.UI32[kk * naxes[0] * naxes[1] +
                                            jj * naxes[0] + ii];
                    }
        }
        else if(datatype == _DATATYPE_UINT64)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    for(uint32_t kk = start_c[2]; kk < end_c[2]; kk++)
                    {
                        imgout.im->array.UI64
                        [(kk - start[2]) * naxesout[0] * naxesout[1] +
                                         (jj - start[1]) * naxesout[0] + (ii - start[0])] =
                             imgin.im->array.UI64[kk * naxes[0] * naxes[1] +
                                            jj * naxes[0] + ii];
                    }
        }
        else if(datatype == _DATATYPE_INT8)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    for(uint32_t kk = start_c[2]; kk < end_c[2]; kk++)
                    {
                        imgout.im->array.SI8
                        [(kk - start[2]) * naxesout[0] * naxesout[1] +
                                         (jj - start[1]) * naxesout[0] + (ii - start[0])] =
                             imgin.im->array.SI8[kk * naxes[0] * naxes[1] +
                                           jj * naxes[0] + ii];
                    }
        }
        else if(datatype == _DATATYPE_INT16)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    for(uint32_t kk = start_c[2]; kk < end_c[2]; kk++)
                    {
                        imgout.im->array.SI16
                        [(kk - start[2]) * naxesout[0] * naxesout[1] +
                                         (jj - start[1]) * naxesout[0] + (ii - start[0])] =
                             imgin.im->array.SI16[kk * naxes[0] * naxes[1] +
                                            jj * naxes[0] + ii];
                    }
        }
        else if(datatype == _DATATYPE_INT32)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    for(uint32_t kk = start_c[2]; kk < end_c[2]; kk++)
                    {
                        imgout.im->array.SI32
                        [(kk - start[2]) * naxesout[0] * naxesout[1] +
                                         (jj - start[1]) * naxesout[0] + (ii - start[0])] =
                             imgin.im->array.SI32[kk * naxes[0] * naxes[1] +
                                            jj * naxes[0] + ii];
                    }
        }
        else if(datatype == _DATATYPE_INT64)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    for(uint32_t kk = start_c[2]; kk < end_c[2]; kk++)
                    {
                        imgout.im->array.SI64
                        [(kk - start[2]) * naxesout[0] * naxesout[1] +
                                         (jj - start[1]) * naxesout[0] + (ii - start[0])] =
                             imgin.im->array.SI64[kk * naxes[0] * naxes[1] +
                                            jj * naxes[0] + ii];
                    }
        }
        else
        {
            PRINT_ERROR("invalid data type");
            exit(0);
        }
    }

    free(naxesout);
    free(naxes);

    return IDout;
}

imageID arith_image_extract2D(
    const char *in_name,
    const char *out_name,
    long        size_x,
    long        size_y,
    long        xstart,
    long        ystart)
{
    long        *start = NULL;
    long        *end   = NULL;
    imageID      IDout;
    uint_fast8_t k;

    IMGID img =
        imgid_make_from_name(in_name);
    resolveIMGID(&img,
                 ERRMODE_ABORT,
                 dcimg, dcnimg);
    int naxis = img.md->naxis;

    start = (long *) malloc(
        sizeof(long) * naxis);
    if(start == NULL)
    {
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    end = (long *) malloc(
        sizeof(long) * naxis);
    if(end == NULL)
    {
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    for(k = 0; k < naxis; k++)
    {
        start[k] = 0;
        end[k]   = img.md->size[k];
    }

    start[0] = xstart;
    start[1] = ystart;
    end[0]   = xstart + size_x;
    end[1]   = ystart + size_y;
    IDout = arith_image_crop(
        in_name, out_name,
        start, end, naxis);

    free(start);
    free(end);

    return IDout;
}

imageID arith_image_extract3D(const char *in_name,
                              const char *out_name,
                              long        size_x,
                              long        size_y,
                              long        size_z,
                              long        xstart,
                              long        ystart,
                              long        zstart)
{
    imageID IDout;
    long   *start = NULL;
    long   *end   = NULL;

    start = (long *) malloc(sizeof(long) * 3);
    if(start == NULL)
    {
        PRINT_ERROR("malloc() error");
        printf("params: %s %s %ld %ld %ld %ld %ld %ld \n",
               in_name,
               out_name,
               size_x,
               size_y,
               size_z,
               xstart,
               ystart,
               zstart);
        exit(0);
    }

    end = (long *) malloc(sizeof(long) * 3);
    if(end == NULL)
    {
        PRINT_ERROR("malloc() error");
        printf("params: %s %s %ld %ld %ld %ld %ld %ld \n",
               in_name,
               out_name,
               size_x,
               size_y,
               size_z,
               xstart,
               ystart,
               zstart);
        exit(0);
    }

    start[0] = xstart;
    start[1] = ystart;
    start[2] = zstart;
    end[0]   = xstart + size_x;
    end[1]   = ystart + size_y;
    end[2]   = zstart + size_z;
    IDout    = arith_image_crop(in_name, out_name, start, end, 3);

    free(start);
    free(end);

    return IDout;
}

