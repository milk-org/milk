// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file indexmap.c
 * @brief Map values using index map
 */

#include "milk_runnable.h"
#include "computeunit.h"

#include "COREMOD_memory/COREMOD_memory.h"

// Forward declaration
imageID image_basic_indexmap(const char *__restrict ID_index_name,
                             const char *__restrict ID_values_name,
                             const char *__restrict IDout_name);

static char p_idx[FUNCTION_PARAMETER_STRMAXLEN] = "imap";
static char p_val[FUNCTION_PARAMETER_STRMAXLEN] = "imval";
static char p_out[FUNCTION_PARAMETER_STRMAXLEN] = "outmap";

#define FPS_PARAMS(X)                                                                    \
    X(".in_index", p_idx, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "index map image") \
    X(".in_values", p_val, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "values image")   \
    X(".out_name", p_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")

MILK_PARAMS_DEFINE(indexmap, FPS_PARAMS);

static MILK_HOT errno_t indexmap_compute(const MILK_RUNNABLE *self)
{
    (void) self;
    image_basic_indexmap(p_idx, p_val, p_out);
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

const MILK_RUNNABLE runnable_image_basic_indexmap = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name         = "imindexmap",
              .cmdkey           = "imindexmap",
              .description      = "map values using index map",
              .description_long = "Map pixel values through a lookup table defined by an index "
                                  "map. Each output pixel takes the value at the index specified "
                                  "by the map." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(indexmap),
    .compute = indexmap_compute,
};
