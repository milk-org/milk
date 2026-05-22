/**
 * @file imtoASCII.c
 * @brief Convert image file to ASCII
 */

#include "CLIcore.h"
#include "fps.h"

#include "COREMOD_memory/COREMOD_memory.h"

// Forward declaration
errno_t IMAGE_FORMAT_im_to_ASCII(const char *__restrict IDname, const char *__restrict foutname);

static char p_in[FUNCTION_PARAMETER_STRMAXLEN]  = "im";
static char p_out[FUNCTION_PARAMETER_STRMAXLEN] = "im.txt";

static FPS_APP_INFO FPS_app_info = { .fps_name    = "im2ascii",
                                     .cmdkey      = "im2ascii",
                                     .description = "convert image file to ASCII",
                                     .description_long =
                                         "Export image pixel values to a text (ASCII) file with "
                                         "configurable formatting and delimiters." };

#define FPS_PARAMS(X)                                                              \
    X(".in_name", p_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".out_name", p_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output ASCII file")

static FPS_CLI_BINDING my_bindings[] = { FPS_PARAMS(FPS_X_BINDING) };
static const int       nb_bindings   = sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF    farg[]        = { FPS_PARAMS(FPS_X_FARG) };
static CLICMDDATA      CLIcmddata    = { "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS     cms           = { 0 };

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
    IMAGE_FORMAT_im_to_ASCII(p_in, p_out);
    return RETURN_SUCCESS;
}

static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_image_format__imtoASCII()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

errno_t IMAGE_FORMAT_im_to_ASCII(const char *__restrict IDname, const char *__restrict foutname)
{
    long    ii;
    long    k;
    imageID ID;
    FILE   *fpout;
    long    naxis;
    long   *coord;
    long    npix;

    ID    = image_ID(IDname, dcimg, dcnimg);
    naxis = dcimg[ID].md[0].naxis;
    coord = (long *) malloc(sizeof(long) * naxis);
    if (coord == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    npix = 1;
    for (k = 0; k < naxis; k++)
    {
        npix *= dcimg[ID].md[0].size[k];
        coord[k] = 0;
    }

    printf("npix = %ld\n", npix);

    fpout = fopen(foutname, "w");

    for (ii = 0; ii < npix; ii++)
    {
        int kOK;

        for (k = 0; k < naxis; k++)
        {
            fprintf(fpout, "%4ld ", coord[k]);
        }
        switch (dcimg[ID].md[0].datatype)
        {
        case _DATATYPE_UINT8:
            fprintf(fpout, " %5u\n", dcimg[ID].array.UI8[ii]);
            break;
        case _DATATYPE_UINT16:
            fprintf(fpout, " %5u\n", dcimg[ID].array.UI16[ii]);
            break;
        case _DATATYPE_UINT32:
            fprintf(fpout, " %u\n", dcimg[ID].array.UI32[ii]);
            break;
        case _DATATYPE_UINT64:
            fprintf(fpout, " %lu\n", dcimg[ID].array.UI64[ii]);
            break;

        case _DATATYPE_INT8:
            fprintf(fpout, " %5d\n", dcimg[ID].array.SI8[ii]);
            break;
        case _DATATYPE_INT16:
            fprintf(fpout, " %5d\n", dcimg[ID].array.SI16[ii]);
            break;
        case _DATATYPE_INT32:
            fprintf(fpout, " %5d\n", dcimg[ID].array.SI32[ii]);
            break;
        case _DATATYPE_INT64:
            fprintf(fpout, " %5ld\n", dcimg[ID].array.SI64[ii]);
            break;

        case _DATATYPE_FLOAT:
            fprintf(fpout, " %f\n", dcimg[ID].array.F[ii]);
            break;
        case _DATATYPE_DOUBLE:
            fprintf(fpout, " %lf\n", dcimg[ID].array.D[ii]);
            break;
        }
        coord[0]++;

        k   = 0;
        kOK = 0;
        while ((kOK == 0) && (k < naxis))
        {
            if (coord[k] == dcimg[ID].md[0].size[k])
            {
                coord[k] = 0;
                coord[k + 1]++;
            }
            else
            {
                kOK = 1;
            }
            k++;
        }
    }
    fclose(fpout);

    free(coord);

    return RETURN_SUCCESS;
}
