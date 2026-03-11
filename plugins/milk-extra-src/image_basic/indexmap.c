/**
 * @file indexmap.c
 * @brief Map values using index map
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"

#include "COREMOD_memory/COREMOD_memory.h"

// Forward declaration
imageID image_basic_indexmap(
    const char *__restrict ID_index_name,
    const char *__restrict ID_values_name,
    const char *__restrict IDout_name);

static char p_idx[FUNCTION_PARAMETER_STRMAXLEN]
    = "imap";
static char p_val[FUNCTION_PARAMETER_STRMAXLEN]
    = "imval";
static char p_out[FUNCTION_PARAMETER_STRMAXLEN]
    = "outmap";

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "imindexmap",
    .cmdkey      = "imindexmap",
    .description =
        "map values using index map"
};

#define FPS_PARAMS(X) \
    X(".in_index", p_idx, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "index map image") \
    X(".in_values", p_val, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "values image") \
    X(".out_name", p_out, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output image")

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
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings = &cms;
    }
}

static errno_t compute_function()
{
    image_basic_indexmap(p_idx, p_val,
                         p_out);
    return RETURN_SUCCESS;
}

static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_image_basic__indexmap()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

imageID image_basic_indexmap(const char *__restrict ID_index_name,
                             const char *__restrict ID_values_name,
                             const char *__restrict IDout_name)
{
    imageID IDindex, IDvalues;
    imageID IDout;
    long    xsize, ysize, xysize;
    long    val_xsize, val_ysize, val_xysize;
    uint8_t datatype;
    uint8_t val_datatype;
    long    ii, i;

    IDindex  = image_ID(ID_index_name, dcimg, dcnimg);
    IDvalues = image_ID(ID_values_name, dcimg, dcnimg);

    xsize    = dcimg[IDindex].md[0].size[0];
    ysize    = dcimg[IDindex].md[0].size[1];
    xysize   = xsize * ysize;
    datatype = dcimg[IDindex].md[0].datatype;

    val_xsize    = dcimg[IDvalues].md[0].size[0];
    val_ysize    = dcimg[IDvalues].md[0].size[1];
    val_xysize   = val_xsize * val_ysize;
    val_datatype = dcimg[IDindex].md[0].datatype;

    create_2Dimage_ID(IDout_name, xsize, ysize, &IDout);

    if(val_datatype == _DATATYPE_FLOAT)
    {
        for(ii = 0; ii < xysize; ii++)
        {
            i = (long)(dcimg[IDindex].array.F[ii] + 0.1);
            if((i > -1) && (i < val_xysize))
            {
                dcimg[IDout].array.F[ii] = dcimg[IDvalues].array.F[i];
            }
        }
    }
    else
    {
        float *arrayf = (float *) malloc(sizeof(float) * val_xysize);
        if(arrayf == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        for(i = 0; i < val_xysize; i++)
        {
            arrayf[i] = (float) dcimg[IDvalues].array.D[i];
        }

        switch(datatype)
        {

            case _DATATYPE_DOUBLE:
                for(ii = 0; ii < xysize; ii++)
                {
                    i = (long)(dcimg[IDindex].array.D[ii] + 0.1);
                    if((i > -1) && (i < val_xysize))
                    {
                        dcimg[IDout].array.F[ii] = arrayf[i];
                    }
                }
                break;

            case _DATATYPE_UINT8:
                for(ii = 0; ii < xysize; ii++)
                {
                    i = (long) dcimg[IDindex].array.UI8[ii];
                    if((i > -1) && (i < val_xysize))
                    {
                        dcimg[IDout].array.F[ii] = arrayf[i];
                    }
                }
                break;

            case _DATATYPE_INT8:
                for(ii = 0; ii < xysize; ii++)
                {
                    i = (long) dcimg[IDindex].array.SI8[ii];
                    if((i > -1) && (i < val_xysize))
                    {
                        dcimg[IDout].array.F[ii] = arrayf[i];
                    }
                }
                break;

            case _DATATYPE_UINT16:
                for(ii = 0; ii < xysize; ii++)
                {
                    i = (long) dcimg[IDindex].array.UI16[ii];
                    if((i > -1) && (i < val_xysize))
                    {
                        dcimg[IDout].array.F[ii] = arrayf[i];
                    }
                }
                break;

            case _DATATYPE_INT16:
                for(ii = 0; ii < xysize; ii++)
                {
                    i = (long) dcimg[IDindex].array.SI16[ii];
                    if((i > -1) && (i < val_xysize))
                    {
                        dcimg[IDout].array.F[ii] = arrayf[i];
                    }
                }
                break;

            case _DATATYPE_UINT32:
                for(ii = 0; ii < xysize; ii++)
                {
                    i = (long) dcimg[IDindex].array.UI32[ii];
                    if((i > -1) && (i < val_xysize))
                    {
                        dcimg[IDout].array.F[ii] = arrayf[i];
                    }
                }
                break;

            case _DATATYPE_INT32:
                for(ii = 0; ii < xysize; ii++)
                {
                    i = (long) dcimg[IDindex].array.SI32[ii];
                    if((i > -1) && (i < val_xysize))
                    {
                        dcimg[IDout].array.F[ii] = arrayf[i];
                    }
                }
                break;

            case _DATATYPE_UINT64:
                for(ii = 0; ii < xysize; ii++)
                {
                    i = (long) dcimg[IDindex].array.UI64[ii];
                    if((i > -1) && (i < val_xysize))
                    {
                        dcimg[IDout].array.F[ii] = arrayf[i];
                    }
                }
                break;

            case _DATATYPE_INT64:
                for(ii = 0; ii < xysize; ii++)
                {
                    i = (long) dcimg[IDindex].array.SI64[ii];
                    if((i > -1) && (i < val_xysize))
                    {
                        dcimg[IDout].array.F[ii] = arrayf[i];
                    }
                }
                break;

            default:
                printf("ERROR: datatype not supported\n");
                free(arrayf);
                return EXIT_FAILURE;
                break;
        }
        free(arrayf);
    }

    return (IDout);
}
