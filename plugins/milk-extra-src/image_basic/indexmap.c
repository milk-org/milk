/**
 * @file indexmap.c
 * @brief Map values using index map
 */

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "fps.h"

#include "COREMOD_memory/COREMOD_memory.h"

// Forward declaration
imageID image_basic_indexmap(const char *__restrict ID_index_name,
                             const char *__restrict ID_values_name,
                             const char *__restrict IDout_name);

static char p_idx[FUNCTION_PARAMETER_STRMAXLEN] = "imap";
static char p_val[FUNCTION_PARAMETER_STRMAXLEN] = "imval";
static char p_out[FUNCTION_PARAMETER_STRMAXLEN] = "outmap";

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "imindexmap",
    .cmdkey           = "imindexmap",
    .description      = "map values using index map",
    .description_long = "Map pixel values through a lookup table defined by an index map. Each "
                        "output pixel takes the value at the index specified by the map."
};

#define FPS_PARAMS(X)                                                                    \
    X(".in_index", p_idx, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "index map image") \
    X(".in_values", p_val, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "values image")   \
    X(".out_name", p_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")

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
    image_basic_indexmap(p_idx, p_val, p_out);
    return RETURN_SUCCESS;
}

static errno_t __attribute__((unused)) CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_image_basic__indexmap()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

/**
 * Map values from one image to another
 * using an index map.
 */
imageID image_basic_indexmap(const char *__restrict ID_index_name,
                             const char *__restrict ID_values_name,
                             const char *__restrict IDout_name)
{
    IMGID imgidx = imgid_make_from_name(ID_index_name);
    resolveIMGID(&imgidx, ERRMODE_WARN, dcimg, dcnimg);
    if (imgidx.ID == -1)
    {
        return RETURN_FAILURE;
    }

    IMGID imgval = imgid_make_from_name(ID_values_name);
    resolveIMGID(&imgval, ERRMODE_WARN, dcimg, dcnimg);
    if (imgval.ID == -1)
    {
        return RETURN_FAILURE;
    }

    long    xsize    = imgidx.md->size[0];
    long    ysize    = imgidx.md->size[1];
    long    xysize   = xsize * ysize;
    uint8_t datatype = imgidx.md->datatype;

    long val_xsize  = imgval.md->size[0];
    long val_ysize  = imgval.md->size[1];
    long val_xysize = val_xsize * val_ysize;

    IMGID imgout       = imgid_make_from_name_2D(IDout_name, xsize, ysize);
    imgout.mdt->shared = 0;
    imgout.im          = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    if (datatype == _DATATYPE_FLOAT)
    {
        for (long ii = 0; ii < xysize; ii++)
        {
            long i = (long) (imgidx.im->array.F[ii] + 0.1);
            if ((i > -1) && (i < val_xysize))
            {
                imgout.im->array.F[ii] = imgval.im->array.F[i];
            }
        }
    }
    else
    {
        float *arrayf = (float *) malloc(sizeof(float) * val_xysize);
        if (arrayf == NULL)
        {
            PRINT_ERROR("malloc returns NULL "
                        "pointer");
            abort();
        }

        for (long i = 0; i < val_xysize; i++)
        {
            arrayf[i] = (float) imgval.im->array.D[i];
        }

        switch (datatype)
        {
        case _DATATYPE_DOUBLE:
            for (long ii = 0; ii < xysize; ii++)
            {
                long i = (long) (imgidx.im->array.D[ii] + 0.1);
                if ((i > -1) && (i < val_xysize))
                {
                    imgout.im->array.F[ii] = arrayf[i];
                }
            }
            break;

        case _DATATYPE_UINT8:
            for (long ii = 0; ii < xysize; ii++)
            {
                long i = (long) imgidx.im->array.UI8[ii];
                if ((i > -1) && (i < val_xysize))
                {
                    imgout.im->array.F[ii] = arrayf[i];
                }
            }
            break;

        case _DATATYPE_INT8:
            for (long ii = 0; ii < xysize; ii++)
            {
                long i = (long) imgidx.im->array.SI8[ii];
                if ((i > -1) && (i < val_xysize))
                {
                    imgout.im->array.F[ii] = arrayf[i];
                }
            }
            break;

        case _DATATYPE_UINT16:
            for (long ii = 0; ii < xysize; ii++)
            {
                long i = (long) imgidx.im->array.UI16[ii];
                if ((i > -1) && (i < val_xysize))
                {
                    imgout.im->array.F[ii] = arrayf[i];
                }
            }
            break;

        case _DATATYPE_INT16:
            for (long ii = 0; ii < xysize; ii++)
            {
                long i = (long) imgidx.im->array.SI16[ii];
                if ((i > -1) && (i < val_xysize))
                {
                    imgout.im->array.F[ii] = arrayf[i];
                }
            }
            break;

        case _DATATYPE_UINT32:
            for (long ii = 0; ii < xysize; ii++)
            {
                long i = (long) imgidx.im->array.UI32[ii];
                if ((i > -1) && (i < val_xysize))
                {
                    imgout.im->array.F[ii] = arrayf[i];
                }
            }
            break;

        case _DATATYPE_INT32:
            for (long ii = 0; ii < xysize; ii++)
            {
                long i = (long) imgidx.im->array.SI32[ii];
                if ((i > -1) && (i < val_xysize))
                {
                    imgout.im->array.F[ii] = arrayf[i];
                }
            }
            break;

        case _DATATYPE_UINT64:
            for (long ii = 0; ii < xysize; ii++)
            {
                long i = (long) imgidx.im->array.UI64[ii];
                if ((i > -1) && (i < val_xysize))
                {
                    imgout.im->array.F[ii] = arrayf[i];
                }
            }
            break;

        case _DATATYPE_INT64:
            for (long ii = 0; ii < xysize; ii++)
            {
                long i = (long) imgidx.im->array.SI64[ii];
                if ((i > -1) && (i < val_xysize))
                {
                    imgout.im->array.F[ii] = arrayf[i];
                }
            }
            break;

        default:
            printf("ERROR: datatype not "
                   "supported\n");
            free(arrayf);
            return EXIT_FAILURE;
            break;
        }
        free(arrayf);
    }

    return imgout.ID;
}
