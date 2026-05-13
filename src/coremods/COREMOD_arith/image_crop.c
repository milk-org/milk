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
    .description = "crop 2D image",
    .description_long =
        "Extract a rectangular sub-region from a 2D or 3D image. Legacy implementation supporting both 2D crop (extractim) and 3D crop (extract3Dim) in a single file."
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

static errno_t __attribute__((unused)) compute_2d()
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
    .description = "crop 3D image",
    .description_long =
        "Extract a rectangular sub-region from a 2D or 3D image. Legacy implementation supporting both 2D crop (extractim) and 3D crop (extract3Dim) in a single file."
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

static const int __attribute__((unused)) nb_bindings =
    sizeof(my_bindings) /
    sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = {
    FPS_PARAMS(FPS_X_FARG)
};

static CLICMDDATA CLIcmddata = {
    "", "", CLICMD_FIELDS_DEFAULTS
};

FPS_CMDSETTINGS_INIT(main, CLIcmddata, FPS_app_info)

static MILK_HOT errno_t __attribute__((unused)) compute_function()
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
        return -1;
    }
    naxes = (uint32_t *) malloc(
        sizeof(uint32_t) * naxis);
    if(naxes == NULL)
    {
        PRINT_ERROR(
            "malloc() error,"
            " naxis = %ld", naxis);
        return -1;
    }

    naxesout = (uint32_t *) malloc(
        sizeof(uint32_t) * naxis);
    if(naxesout == NULL)
    {
        PRINT_ERROR("malloc() error");
        free(naxes);
        return -1;
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

    int typesize =
        ImageStreamIO_typesize(datatype);
    if(typesize <= 0)
    {
        PRINT_ERROR("invalid datatype %d",
                    (int) datatype);
        free(naxesout);
        free(naxes);
        return -1;
    }
    size_t elemsize = (size_t) typesize;

    if (naxis == 1)
    {
        long ncopy = end_c[0] - start_c[0];
        if (ncopy > 0)
        {
            __builtin_memcpy(
                (char *) imgout.im->array.raw
                    + (start_c[0] - start[0])
                    * elemsize,
                (char *) imgin.im->array.raw
                    + start_c[0] * elemsize,
                (size_t) ncopy * elemsize);
        }
    }
    else if (naxis == 2)
    {
        long row_elems =
            end_c[0] - start_c[0];
        if (row_elems > 0)
        {
            for (long jj = start_c[1];
                 jj < end_c[1]; jj++)
            {
                long dst_off =
                    ((jj - start[1])
                     * (long) naxesout[0]
                     + (start_c[0]
                        - start[0]))
                    * (long) elemsize;
                long src_off =
                    (jj * (long) naxes[0]
                     + start_c[0])
                    * (long) elemsize;
                __builtin_memcpy(
                    (char *) imgout.im->array.raw
                        + dst_off,
                    (char *) imgin.im->array.raw
                        + src_off,
                    (size_t) row_elems
                    * elemsize);
            }
        }
    }
    else if (naxis == 3)
    {
        long row_elems =
            end_c[0] - start_c[0];
        if (row_elems > 0)
        {
            long in_slice =
                (long) naxes[0]
                * (long) naxes[1];
            long out_slice =
                (long) naxesout[0]
                * (long) naxesout[1];

            for (long kk = start_c[2];
                 kk < end_c[2]; kk++)
            {
                for (long jj = start_c[1];
                     jj < end_c[1]; jj++)
                {
                    long dst_off =
                        ((kk - start[2])
                         * out_slice
                         + (jj - start[1])
                         * (long) naxesout[0]
                         + (start_c[0]
                            - start[0]))
                        * (long) elemsize;
                    long src_off =
                        (kk * in_slice
                         + jj
                         * (long) naxes[0]
                         + start_c[0])
                        * (long) elemsize;
                    __builtin_memcpy(
                        (char *) imgout.im->array.raw
                            + dst_off,
                        (char *) imgin.im->array.raw
                            + src_off,
                        (size_t) row_elems
                        * elemsize);
                }
            }
        }
    }
    else
    {
        PRINT_ERROR(
            "unsupported naxis = %ld",
            naxis);
        free(naxesout);
        free(naxes);
        return -1;
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
        return -1;
    }

    end = (long *) malloc(
        sizeof(long) * naxis);
    if(end == NULL)
    {
        PRINT_ERROR("malloc() error");
        free(start);
        return -1;
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
        PRINT_ERROR(
            "malloc() error, params: "
            "%s %s %ld %ld %ld %ld %ld %ld",
            in_name, out_name,
            size_x, size_y, size_z,
            xstart, ystart, zstart);
        return -1;
    }

    end = (long *) malloc(sizeof(long) * 3);
    if(end == NULL)
    {
        PRINT_ERROR(
            "malloc() error, params: "
            "%s %s %ld %ld %ld %ld %ld %ld",
            in_name, out_name,
            size_x, size_y, size_z,
            xstart, ystart, zstart);
        free(start);
        return -1;
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

