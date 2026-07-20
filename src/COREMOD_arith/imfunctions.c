// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    imfunctions.c
 * @brief   apply math functions to images
 *
 *
 */

#include <assert.h>
#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"
#include "mathfuncs.h"

#ifdef _OPENMP
#include <omp.h>
#define OMP_NELEMENT_LIMIT 100000
#endif

static double get_pixel_double(IMAGE *im, uint64_t index)
{
    switch(im->md[0].datatype)
    {
        case _DATATYPE_UINT8:   return (double)im->array.UI8[index];
        case _DATATYPE_INT8:    return (double)im->array.SI8[index];
        case _DATATYPE_UINT16:  return (double)im->array.UI16[index];
        case _DATATYPE_INT16:   return (double)im->array.SI16[index];
        case _DATATYPE_UINT32:  return (double)im->array.UI32[index];
        case _DATATYPE_INT32:   return (double)im->array.SI32[index];
        case _DATATYPE_UINT64:  return (double)im->array.UI64[index];
        case _DATATYPE_INT64:   return (double)im->array.SI64[index];
        case _DATATYPE_FLOAT:   return (double)im->array.F[index];
        case _DATATYPE_DOUBLE:  return (double)im->array.D[index];
        default: return 0.0;
    }
}

/* ------------------------------------------------------------------------- */
/* Functions for bison / flex                                                */
/* im : image
  d : double

  function_<inputformat>_<outputformat>__<math function input>_<math function output>

  examples:
  function_imim__dd_d  : input is (image, image), applies double,double -> double function

  ------------------------------------------------------------------------- */

errno_t arith_image_function_im_im__d_d_IMGID(
    IMGID *imgin,
    IMGID *imgout,
    double (*pt2function)(double))
{
    resolveIMGID(imgin, ERRMODE_ABORT);

    DEBUG_TRACEPOINT("arith_image_function_d_d  %s %s\n", imgin->name, imgout->name);

    imgout->naxis = imgin->md->naxis;
    for(uint8_t i = 0; i < imgin->md->naxis; i++)
    {
        imgout->size[i] = imgin->md->size[i];
    }

    imgout->datatype = _DATATYPE_FLOAT;
    if(imgin->md->datatype == _DATATYPE_DOUBLE)
    {
        imgout->datatype = _DATATYPE_DOUBLE;
    }
    imgout->shared = data.SHARED_DFT;
    imgout->NBkw   = NB_KEYWNODE_MAX;

    imcreateIMGID(imgout);

    uint_fast64_t nelement = imgin->md->nelement;
    uint8_t       datatype = imgin->md->datatype;

#ifdef _OPENMP
    #pragma omp parallel if (nelement > OMP_NELEMENT_LIMIT)
    {
#endif

    if(datatype == _DATATYPE_UINT8)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(uint_fast64_t ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] = (float) pt2function(
                                          (double)(imgin->im->array.UI8[ii]));
        }
    }
    if(datatype == _DATATYPE_UINT16)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(uint_fast64_t ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] = (float) pt2function(
                                          (double)(imgin->im->array.UI16[ii]));
        }
    }
    if(datatype == _DATATYPE_UINT32)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(uint_fast64_t ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] = (float) pt2function(
                                          (double)(imgin->im->array.UI32[ii]));
        }
    }
    if(datatype == _DATATYPE_UINT64)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(uint_fast64_t ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] = (float) pt2function(
                                          (double)(imgin->im->array.UI64[ii]));
        }
    }

    if(datatype == _DATATYPE_INT8)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(uint_fast64_t ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] = (float) pt2function(
                                          (double)(imgin->im->array.SI8[ii]));
        }
    }
    if(datatype == _DATATYPE_INT16)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(uint_fast64_t ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] = (float) pt2function(
                                          (double)(imgin->im->array.SI16[ii]));
        }
    }
    if(datatype == _DATATYPE_INT32)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(uint_fast64_t ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] = (float) pt2function(
                                          (double)(imgin->im->array.SI32[ii]));
        }
    }
    if(datatype == _DATATYPE_INT64)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(uint_fast64_t ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] = (float) pt2function(
                                          (double)(imgin->im->array.SI64[ii]));
        }
    }

    if(datatype == _DATATYPE_FLOAT)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(uint_fast64_t ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] =
                (float) pt2function((double)(imgin->im->array.F[ii]));
        }
    }
    if(datatype == _DATATYPE_DOUBLE)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(uint_fast64_t ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.D[ii] =
                pt2function(imgin->im->array.D[ii]);
        }
    }
#ifdef _OPENMP
    }
#endif

    DEBUG_TRACEPOINT("arith_image_function_d_d  DONE\n");

    return RETURN_SUCCESS;
}

errno_t arith_image_function_im_im__d_d(
    const char *__restrict ID_name,
    const char *__restrict ID_out,
    double (*pt2function)(double))
{
    IMGID imgin  = mkIMGID_from_name(ID_name);
    IMGID imgout = mkIMGID_from_name(ID_out);

    return arith_image_function_im_im__d_d_IMGID(&imgin, &imgout, pt2function);
}

errno_t arith_image_function_imd_im__dd_d_IMGID(
    IMGID *imgin,
    double v0,
    IMGID *imgout,
    double (*pt2function)(double, double))
{
    long ii;

    resolveIMGID(imgin, ERRMODE_ABORT);

    imgout->naxis = imgin->md->naxis;
    for(int i = 0; i < imgin->md->naxis; i++)
    {
        imgout->size[i] = imgin->md->size[i];
    }

    imgout->datatype = _DATATYPE_FLOAT;
    if(imgin->md->datatype == _DATATYPE_DOUBLE)
    {
        imgout->datatype = _DATATYPE_DOUBLE;
    }
    imgout->shared = data.SHARED_DFT;
    imgout->NBkw   = NB_KEYWNODE_MAX;

    imcreateIMGID(imgout);

    uint_fast64_t nelement = imgin->md->nelement;
    uint8_t       datatype = imgin->md->datatype;

#ifdef _OPENMP
    #pragma omp parallel if (nelement > OMP_NELEMENT_LIMIT)
    {
#endif

    if(datatype == _DATATYPE_UINT8)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] =
                (float) pt2function((double)(imgin->im->array.UI8[ii]),
                                    v0);
        }
    }
    if(datatype == _DATATYPE_UINT16)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] = (float) pt2function(
                                          (double)(imgin->im->array.UI16[ii]),
                                          v0);
        }
    }
    if(datatype == _DATATYPE_UINT32)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] = (float) pt2function(
                                          (double)(imgin->im->array.UI32[ii]),
                                          v0);
        }
    }
    if(datatype == _DATATYPE_UINT64)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] = (float) pt2function(
                                          (double)(imgin->im->array.UI64[ii]),
                                          v0);
        }
    }

    if(datatype == _DATATYPE_INT8)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] =
                (float) pt2function((double)(imgin->im->array.SI8[ii]),
                                    v0);
        }
    }
    if(datatype == _DATATYPE_INT16)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] = (float) pt2function(
                                          (double)(imgin->im->array.SI16[ii]),
                                          v0);
        }
    }
    if(datatype == _DATATYPE_INT32)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] = (float) pt2function(
                                          (double)(imgin->im->array.SI32[ii]),
                                          v0);
        }
    }
    if(datatype == _DATATYPE_INT64)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] = (float) pt2function(
                                          (double)(imgin->im->array.SI64[ii]),
                                          v0);
        }
    }

    if(datatype == _DATATYPE_FLOAT)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] =
                (float) pt2function((double)(imgin->im->array.F[ii]),
                                    v0);
        }
    }
    if(datatype == _DATATYPE_DOUBLE)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.D[ii] =
                pt2function(imgin->im->array.D[ii], v0);
        }
    }
#ifdef _OPENMP
    }
#endif

    DEBUG_TRACEPOINT("arith_image_function_d_d  DONE\n");

    return RETURN_SUCCESS;
}

errno_t arith_image_function_imd_im__dd_d(
    const char *__restrict ID_name,
    double      v0,
    const char *__restrict ID_out,
    double (*pt2function)(double, double))
{
    IMGID imgin  = mkIMGID_from_name(ID_name);
    IMGID imgout = mkIMGID_from_name(ID_out);

    return arith_image_function_imd_im__dd_d_IMGID(&imgin, v0, &imgout, pt2function);
}

errno_t arith_image_function_imdd_im__ddd_d_IMGID(IMGID *imgin,
        double      v0,
        double      v1,
        IMGID      *imgout,
        double (*pt2function)(double,
                              double,
                              double))
{
    long      ii;

    resolveIMGID(imgin, ERRMODE_ABORT);

    imgout->naxis = imgin->md->naxis;
    for(int i = 0; i < imgin->md->naxis; i++)
    {
        imgout->size[i] = imgin->md->size[i];
    }

    imgout->datatype = _DATATYPE_FLOAT;
    if(imgin->md->datatype == _DATATYPE_DOUBLE)
    {
        imgout->datatype = _DATATYPE_DOUBLE;
    }
    imgout->shared = data.SHARED_DFT;
    imgout->NBkw   = NB_KEYWNODE_MAX;

    imcreateIMGID(imgout);

    uint_fast64_t nelement = imgin->md->nelement;
    uint8_t       datatype = imgin->md->datatype;

#ifdef _OPENMP
    #pragma omp parallel if (nelement > OMP_NELEMENT_LIMIT)
    {
#endif

    if(datatype == _DATATYPE_UINT8)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] =
                (float) pt2function((double)(imgin->im->array.UI8[ii]),
                                    v0,
                                    v1);
        }
    }
    if(datatype == _DATATYPE_UINT16)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] = (float) pt2function(
                                          (double)(imgin->im->array.UI16[ii]),
                                          v0,
                                          v1);
        }
    }
    if(datatype == _DATATYPE_UINT32)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] = (float) pt2function(
                                          (double)(imgin->im->array.UI32[ii]),
                                          v0,
                                          v1);
        }
    }
    if(datatype == _DATATYPE_UINT64)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] = (float) pt2function(
                                          (double)(imgin->im->array.UI64[ii]),
                                          v0,
                                          v1);
        }
    }

    if(datatype == _DATATYPE_INT8)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] =
                (float) pt2function((double)(imgin->im->array.SI8[ii]),
                                    v0,
                                    v1);
        }
    }
    if(datatype == _DATATYPE_INT16)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] = (float) pt2function(
                                          (double)(imgin->im->array.SI16[ii]),
                                          v0,
                                          v1);
        }
    }
    if(datatype == _DATATYPE_INT32)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] = (float) pt2function(
                                          (double)(imgin->im->array.SI32[ii]),
                                          v0,
                                          v1);
        }
    }
    if(datatype == _DATATYPE_INT64)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] = (float) pt2function(
                                          (double)(imgin->im->array.SI64[ii]),
                                          v0,
                                          v1);
        }
    }

    if(datatype == _DATATYPE_FLOAT)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] =
                (float) pt2function((double)(imgin->im->array.F[ii]),
                                    v0,
                                    v1);
        }
    }
    if(datatype == _DATATYPE_DOUBLE)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.D[ii] =
                pt2function(imgin->im->array.D[ii], v0, v1);
        }
    }
#ifdef _OPENMP
    }
#endif

    DEBUG_TRACEPOINT("arith_image_function_d_d  DONE\n");

    return RETURN_SUCCESS;
}

errno_t arith_image_function_imdd_im__ddd_d(const char *ID_name,
        double      v0,
        double      v1,
        const char *ID_out,
        double (*pt2function)(double,
                              double,
                              double))
{
    IMGID imgin  = mkIMGID_from_name(ID_name);
    IMGID imgout = mkIMGID_from_name(ID_out);

    return arith_image_function_imdd_im__ddd_d_IMGID(&imgin, v0, v1, &imgout, pt2function);
}

/* ------------------------------------------------------------------------- */
/* image  -> image                                                           */
/* ------------------------------------------------------------------------- */

errno_t arith_image_function_1_1_byID(imageID ID,
                                      imageID IDout,
                                      double (*pt2function)(double))
{
    uint32_t *naxes = NULL;
    long      naxis;
    long      ii;
    long      nelement;
    uint8_t   datatype;
    //, datatypeout;
    long i;

    //  printf("arith_image_function_1_1\n");

    datatype = data.image[ID].md[0].datatype;
    naxis    = data.image[ID].md[0].naxis;
    naxes    = (uint32_t *) malloc(sizeof(uint32_t) * naxis);
    if(naxes == NULL)
    {
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    for(i = 0; i < naxis; i++)
    {
        naxes[i] = data.image[ID].md[0].size[i];
    }

    //    datatypeout = _DATATYPE_FLOAT;
    //    if(datatype == _DATATYPE_DOUBLE)
    //        datatypeout = _DATATYPE_DOUBLE;

    free(naxes);

    nelement = data.image[ID].md[0].nelement;

#ifdef _OPENMP
    #pragma omp parallel if (nelement > OMP_NELEMENT_LIMIT)
    {
#endif

        if(datatype == _DATATYPE_UINT8)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[IDout].array.F[ii] =
                    pt2function((double)(data.image[ID].array.UI8[ii]));
            }
        }

        if(datatype == _DATATYPE_UINT16)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[IDout].array.F[ii] =
                    pt2function((double)(data.image[ID].array.UI16[ii]));
            }
        }

        if(datatype == _DATATYPE_UINT32)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[IDout].array.F[ii] =
                    pt2function((double)(data.image[ID].array.UI32[ii]));
            }
        }

        if(datatype == _DATATYPE_UINT64)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[IDout].array.F[ii] =
                    pt2function((double)(data.image[ID].array.UI64[ii]));
            }
        }

        if(datatype == _DATATYPE_INT8)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[IDout].array.F[ii] =
                    pt2function((double)(data.image[ID].array.SI8[ii]));
            }
        }
        if(datatype == _DATATYPE_INT16)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[IDout].array.F[ii] =
                    pt2function((double)(data.image[ID].array.SI16[ii]));
            }
        }
        if(datatype == _DATATYPE_INT32)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[IDout].array.F[ii] =
                    pt2function((double)(data.image[ID].array.SI32[ii]));
            }
        }
        if(datatype == _DATATYPE_INT64)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[IDout].array.F[ii] =
                    pt2function((double)(data.image[ID].array.SI64[ii]));
            }
        }

        if(datatype == _DATATYPE_FLOAT)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[IDout].array.F[ii] =
                    pt2function((double)(data.image[ID].array.F[ii]));
            }
        }
        if(datatype == _DATATYPE_DOUBLE)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[IDout].array.D[ii] =
                    (double) pt2function((double)(data.image[ID].array.D[ii]));
            }
        }
#ifdef _OPENMP
    }
#endif

    return RETURN_SUCCESS;
}

errno_t arith_image_function_1_1_IMGID(IMGID *imgin,
                                       IMGID *imgout,
                                       double (*pt2function)(double))
{
    long ii;

    resolveIMGID(imgin, ERRMODE_ABORT);

    imgout->naxis = imgin->md->naxis;
    for(int i = 0; i < imgin->md->naxis; i++)
    {
        imgout->size[i] = imgin->md->size[i];
    }

    imgout->datatype = _DATATYPE_FLOAT;
    if(imgin->md->datatype == _DATATYPE_DOUBLE)
    {
        imgout->datatype = _DATATYPE_DOUBLE;
    }
    imgout->shared = data.SHARED_DFT;
    imgout->NBkw   = NB_KEYWNODE_MAX;

    imcreateIMGID(imgout);

    uint_fast64_t nelement = imgin->md->nelement;
    uint8_t       datatype = imgin->md->datatype;

#ifdef _OPENMP
    #pragma omp parallel if (nelement > OMP_NELEMENT_LIMIT)
    {
#endif

    if(datatype == _DATATYPE_UINT8)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] =
                pt2function((double)(imgin->im->array.UI8[ii]));
        }
    }
    if(datatype == _DATATYPE_UINT16)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] =
                pt2function((double)(imgin->im->array.UI16[ii]));
        }
    }
    if(datatype == _DATATYPE_UINT32)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] =
                pt2function((double)(imgin->im->array.UI32[ii]));
        }
    }
    if(datatype == _DATATYPE_UINT64)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] =
                pt2function((double)(imgin->im->array.UI64[ii]));
        }
    }

    if(datatype == _DATATYPE_INT8)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] =
                pt2function((double)(imgin->im->array.SI8[ii]));
        }
    }
    if(datatype == _DATATYPE_INT16)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] =
                pt2function((double)(imgin->im->array.SI16[ii]));
        }
    }
    if(datatype == _DATATYPE_INT32)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] =
                pt2function((double)(imgin->im->array.SI32[ii]));
        }
    }
    if(datatype == _DATATYPE_INT64)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] =
                pt2function((double)(imgin->im->array.SI64[ii]));
        }
    }

    if(datatype == _DATATYPE_FLOAT)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.F[ii] =
                pt2function((double)(imgin->im->array.F[ii]));
        }
    }

    if(datatype == _DATATYPE_DOUBLE)
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            imgout->im->array.D[ii] =
                (double) pt2function((double)(imgin->im->array.D[ii]));
        }
    }
#ifdef _OPENMP
    }
#endif

    return RETURN_SUCCESS;
}

errno_t arith_image_function_1_1(const char *ID_name,
                                 const char *ID_out,
                                 double (*pt2function)(double))
{
    IMGID imgin  = mkIMGID_from_name(ID_name);
    IMGID imgout = mkIMGID_from_name(ID_out);

    return arith_image_function_1_1_IMGID(&imgin, &imgout, pt2function);
}

// imagein -> imagein (in place)
errno_t arith_image_function_1_1_inplace_byID(imageID ID,
        double (*pt2function)(double))
{
    long    ii;
    long    nelement;
    uint8_t datatype;
    //, datatypeout;

    // printf("arith_image_function_1_1_inplace\n");

    datatype = data.image[ID].md[0].datatype;

    //datatypeout = _DATATYPE_FLOAT;
    //if(datatype == _DATATYPE_DOUBLE)
    //   datatypeout = _DATATYPE_DOUBLE;

    nelement = data.image[ID].md[0].nelement;

    data.image[ID].md[0].write = 0;
#ifdef _OPENMP
    #pragma omp parallel if (nelement > OMP_NELEMENT_LIMIT)
    {
#endif

        if(datatype == _DATATYPE_UINT8)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.F[ii] =
                    pt2function((double)(data.image[ID].array.UI8[ii]));
            }
        }
        if(datatype == _DATATYPE_UINT16)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.F[ii] =
                    pt2function((double)(data.image[ID].array.UI16[ii]));
            }
        }
        if(datatype == _DATATYPE_UINT32)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.F[ii] =
                    pt2function((double)(data.image[ID].array.UI32[ii]));
            }
        }
        if(datatype == _DATATYPE_UINT64)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.F[ii] =
                    pt2function((double)(data.image[ID].array.UI64[ii]));
            }
        }

        if(datatype == _DATATYPE_INT8)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.F[ii] =
                    pt2function((double)(data.image[ID].array.SI8[ii]));
            }
        }
        if(datatype == _DATATYPE_INT16)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.F[ii] =
                    pt2function((double)(data.image[ID].array.SI16[ii]));
            }
        }
        if(datatype == _DATATYPE_INT32)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.F[ii] =
                    pt2function((double)(data.image[ID].array.SI32[ii]));
            }
        }
        if(datatype == _DATATYPE_INT64)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.F[ii] =
                    pt2function((double)(data.image[ID].array.SI64[ii]));
            }
        }

        if(datatype == _DATATYPE_FLOAT)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.F[ii] =
                    pt2function((double)(data.image[ID].array.F[ii]));
            }
        }

        if(datatype == _DATATYPE_DOUBLE)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.D[ii] =
                    (double) pt2function((double)(data.image[ID].array.D[ii]));
            }
        }

#ifdef _OPENMP
    }
#endif

    data.image[ID].md[0].write = 0;
    data.image[ID].md[0].cnt0++;

    return RETURN_SUCCESS;
}

// imagein -> imagein (in place)
errno_t arith_image_function_1_1_inplace(const char *ID_name,
        double (*pt2function)(double))
{
    imageID ID;
    long    ii;
    long    nelement;
    uint8_t datatype;
    //, datatypeout;

    // printf("arith_image_function_1_1_inplace\n");

    ID       = image_ID(ID_name);
    datatype = data.image[ID].md[0].datatype;

    //    datatypeout = _DATATYPE_FLOAT;
    //    if(datatype == _DATATYPE_DOUBLE)
    //        datatypeout = _DATATYPE_DOUBLE;

    nelement = data.image[ID].md[0].nelement;

    data.image[ID].md[0].write = 0;
#ifdef _OPENMP
    #pragma omp parallel if (nelement > OMP_NELEMENT_LIMIT)
    {
#endif

        if(datatype == _DATATYPE_UINT8)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.F[ii] =
                    pt2function((double)(data.image[ID].array.UI8[ii]));
            }
        }
        if(datatype == _DATATYPE_UINT16)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.F[ii] =
                    pt2function((double)(data.image[ID].array.UI16[ii]));
            }
        }
        if(datatype == _DATATYPE_UINT32)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.F[ii] =
                    pt2function((double)(data.image[ID].array.UI32[ii]));
            }
        }
        if(datatype == _DATATYPE_UINT64)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.F[ii] =
                    pt2function((double)(data.image[ID].array.UI64[ii]));
            }
        }

        if(datatype == _DATATYPE_INT8)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.F[ii] =
                    pt2function((double)(data.image[ID].array.SI8[ii]));
            }
        }
        if(datatype == _DATATYPE_INT16)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.F[ii] =
                    pt2function((double)(data.image[ID].array.SI16[ii]));
            }
        }
        if(datatype == _DATATYPE_INT32)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.F[ii] =
                    pt2function((double)(data.image[ID].array.SI32[ii]));
            }
        }
        if(datatype == _DATATYPE_INT64)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.F[ii] =
                    pt2function((double)(data.image[ID].array.SI64[ii]));
            }
        }

        if(datatype == _DATATYPE_FLOAT)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.F[ii] =
                    pt2function((double)(data.image[ID].array.F[ii]));
            }
        }
        if(datatype == _DATATYPE_DOUBLE)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.D[ii] =
                    (double) pt2function((double)(data.image[ID].array.D[ii]));
            }
        }

#ifdef _OPENMP
    }
#endif

    data.image[ID].md[0].write = 0;
    data.image[ID].md[0].cnt0++;

    return RETURN_SUCCESS;
}

/* ------------------------------------------------------------------------- */
/* image, image  -> image                                                    */
/* ------------------------------------------------------------------------- */

errno_t arith_image_function_2_1_IMGID(
    IMGID *inimg1,
    IMGID *inimg2,
    IMGID *outimg,
    double (*pt2function)(double, double)
)
{
    DEBUG_TRACE_FSTART();

    resolveIMGID(inimg1, ERRMODE_ABORT);
    resolveIMGID(inimg2, ERRMODE_ABORT);

    resolveIMGID(outimg, ERRMODE_NULL);
    if( outimg->ID == -1)
    {
        copyIMGID(inimg1, outimg);
    }

    // output naxis is max of inputs
    outimg->naxis = inimg1->md->naxis;
    if ( inimg2->md->naxis > inimg1->md->naxis )
    {
        outimg->naxis = inimg2->md->naxis;
    }

    // axis expansion flags
    int in1expand[3];
    int in2expand[3];

    // check which coordinate needs to be expanded in computation
    //
    uint64_t nbpix = 1;
    uint64_t nbpix1 = 1;
    uint64_t nbpix2 = 1;
    for ( uint8_t axis = 0; axis < outimg->naxis; axis++)
    {
        in1expand[axis] = 1;
        in2expand[axis] = 1;

        uint32_t size1;
        if(axis < inimg1->md->naxis)
        {
            size1 = inimg1->md->size[axis];
        }
        else
        {
            size1 = 1;
        }
        nbpix1 *= size1;

        uint32_t size2;
        if(axis < inimg2->md->naxis)
        {
            size2 = inimg2->md->size[axis];
        }
        else
        {
            size2 = 1;
        }
        nbpix2 *= size2;

        if( size1 != size2 )
        {
            if( size1 == 1 )
            {
                in1expand[axis] = 0;
                outimg->size[axis] = size2;

            }
            else if ( size2 == 1)
            {
                in2expand[axis] = 0;
                outimg->size[axis] = size1;
            }
            else
            {
                PRINT_ERROR("axis %d size %u and %u incompatible", axis, size1, size2);
                abort();
            }
        }
        nbpix *= outimg->size[axis];
    }
    for ( uint8_t axis = outimg->naxis; axis<3; axis++)
    {
        outimg->size[axis] = 1;
        in1expand[axis] = 1;
        in2expand[axis] = 1;
    }

    outimg->datatype = _DATATYPE_FLOAT; // default
    if(inimg1->md->datatype == _DATATYPE_DOUBLE)
    {
        outimg->datatype = _DATATYPE_DOUBLE;
    }
    if(inimg2->md->datatype == _DATATYPE_DOUBLE)
    {
        outimg->datatype = _DATATYPE_DOUBLE;
    }

    imcreateIMGID(outimg);

    // Metadata caching
    uint64_t nelement = outimg->md->nelement;
    uint8_t datatype = outimg->datatype;
    uint8_t datatype1 = inimg1->md->datatype;
    uint8_t datatype2 = inimg2->md->datatype;

    // Fast path: images have same size and layout
    int fastpath = 1;
    if (inimg1->md->naxis != inimg2->md->naxis)
    {
        fastpath = 0;
    }
    else
    {
        for (uint8_t axis = 0; axis < outimg->naxis; axis++)
        {
            if (inimg1->md->size[axis] != inimg2->md->size[axis])
            {
                fastpath = 0;
                break;
            }
        }
    }

    if (fastpath)
    {
#ifdef _OPENMP
        #pragma omp parallel if (nelement > OMP_NELEMENT_LIMIT)
        {
#endif
            if (datatype == _DATATYPE_FLOAT && datatype1 == _DATATYPE_FLOAT && datatype2 == _DATATYPE_FLOAT)
            {
                float *outptr = outimg->im->array.F;
                float *in1ptr = inimg1->im->array.F;
                float *in2ptr = inimg2->im->array.F;
#ifdef _OPENMP
                #pragma omp for
#endif
                for (uint64_t ii = 0; ii < nelement; ii++)
                {
                    outptr[ii] = (float)pt2function((double)in1ptr[ii], (double)in2ptr[ii]);
                }
            }
            else if (datatype == _DATATYPE_DOUBLE && datatype1 == _DATATYPE_DOUBLE && datatype2 == _DATATYPE_DOUBLE)
            {
                double *outptr = outimg->im->array.D;
                double *in1ptr = inimg1->im->array.D;
                double *in2ptr = inimg2->im->array.D;
#ifdef _OPENMP
                #pragma omp for
#endif
                for (uint64_t ii = 0; ii < nelement; ii++)
                {
                    outptr[ii] = pt2function(in1ptr[ii], in2ptr[ii]);
                }
            }
            else if (datatype == _DATATYPE_FLOAT)
            {
                float *outptr = outimg->im->array.F;
#ifdef _OPENMP
                #pragma omp for
#endif
                for (uint64_t ii = 0; ii < nelement; ii++)
                {
                    double v1 = (datatype1 == _DATATYPE_DOUBLE) ? inimg1->im->array.D[ii] :
                                (datatype1 == _DATATYPE_FLOAT) ? inimg1->im->array.F[ii] :
                                get_pixel_double(inimg1->im, ii);
                    double v2 = (datatype2 == _DATATYPE_DOUBLE) ? inimg2->im->array.D[ii] :
                                (datatype2 == _DATATYPE_FLOAT) ? inimg2->im->array.F[ii] :
                                get_pixel_double(inimg2->im, ii);
                    outptr[ii] = (float)pt2function(v1, v2);
                }
            }
            else if (datatype == _DATATYPE_DOUBLE)
            {
                double *outptr = outimg->im->array.D;
#ifdef _OPENMP
                #pragma omp for
#endif
                for (uint64_t ii = 0; ii < nelement; ii++)
                {
                    double v1 = (datatype1 == _DATATYPE_DOUBLE) ? inimg1->im->array.D[ii] :
                                (datatype1 == _DATATYPE_FLOAT) ? inimg1->im->array.F[ii] :
                                get_pixel_double(inimg1->im, ii);
                    double v2 = (datatype2 == _DATATYPE_DOUBLE) ? inimg2->im->array.D[ii] :
                                (datatype2 == _DATATYPE_FLOAT) ? inimg2->im->array.F[ii] :
                                get_pixel_double(inimg2->im, ii);
                    outptr[ii] = pt2function(v1, v2);
                }
            }
#ifdef _OPENMP
        }
#endif
        DEBUG_TRACE_FEXIT();
        return RETURN_SUCCESS;
    }

    // slow path with broadcasting
    uint64_t * __restrict inpix1 = (uint64_t *) malloc(sizeof(uint64_t) * nbpix);
    uint64_t * __restrict inpix2 = (uint64_t *) malloc(sizeof(uint64_t) * nbpix);

    for ( uint32_t ii = 0; ii < outimg->size[0]; ii++ )
    {
        uint32_t ii1 = ii * in1expand[0];
        uint32_t ii2 = ii * in2expand[0];

        for ( uint32_t jj = 0; jj < outimg->size[1]; jj++ )
        {
            uint32_t jj1 = jj * in1expand[1];
            uint32_t jj2 = jj * in2expand[1];

            for ( uint32_t kk = 0; kk < outimg->size[2]; kk++ )
            {
                uint64_t outpixi = ii;
                outpixi +=  jj * outimg->size[0];
                outpixi +=  kk * outimg->size[1] * outimg->size[0];

                uint32_t kk1 = kk * in1expand[2];
                uint32_t kk2 = kk * in2expand[2];

                inpix1[outpixi] =  kk1 * inimg1->md->size[1] * inimg1->md->size[0] + jj1 * inimg1->md->size[0] + ii1;
                inpix2[outpixi] =  kk2 * inimg2->md->size[1] * inimg2->md->size[0] + jj2 * inimg2->md->size[0] + ii2;
            }
        }
    }

    double * ptr1array;
    int ptr1allocate = 0;
    if ( inimg1->md->datatype == _DATATYPE_DOUBLE ) { ptr1array = inimg1->im->array.D; }
    else {
        ptr1allocate = 1;
        ptr1array = (double *) malloc(sizeof(double) * nbpix1);
        for(uint64_t ii = 0; ii < nbpix1; ii++) ptr1array[ii] = get_pixel_double(inimg1->im, ii);
    }

    double * ptr2array;
    int ptr2allocate = 0;
    if ( inimg2->md->datatype == _DATATYPE_DOUBLE ) { ptr2array = inimg2->im->array.D; }
    else {
        ptr2allocate = 1;
        ptr2array = (double *) malloc(sizeof(double) * nbpix2);
        for(uint64_t ii = 0; ii < nbpix2; ii++) ptr2array[ii] = get_pixel_double(inimg2->im, ii);
    }

    if ( outimg->datatype == _DATATYPE_FLOAT )
    {
        for(uint64_t ii = 0; ii < nbpix; ii++ ) outimg->im->array.F[ii] = (float)pt2function(ptr1array[inpix1[ii]], ptr2array[inpix2[ii]]);
    }
    else if ( outimg->datatype == _DATATYPE_DOUBLE )
    {
        for(uint64_t ii = 0; ii < nbpix; ii++ ) outimg->im->array.D[ii] = pt2function(ptr1array[inpix1[ii]], ptr2array[inpix2[ii]]);
    }

    if(ptr1allocate == 1) free(ptr1array);
    if(ptr2allocate == 1) free(ptr2array);
    free(inpix1);
    free(inpix2);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

errno_t arith_img_function_2_1(
    IMGID inimg1,
    IMGID inimg2,
    IMGID *outimg,
    double (*pt2function)(double, double)
)
{
    return arith_image_function_2_1_IMGID(&inimg1, &inimg2, outimg, pt2function);
}

/* ------------------------------------------------------------------------- */
/* image, image  -> image                                                    */
/* ------------------------------------------------------------------------- */

errno_t arith_image_function_2_1(
    const char *ID_name1,
    const char *ID_name2,
    const char *ID_out,
    double (*pt2function)(double, double)
)
{
    IMGID inimg1 = mkIMGID_from_name(ID_name1);
    IMGID inimg2 = mkIMGID_from_name(ID_name2);
    IMGID outimg = mkIMGID_from_name(ID_out);
    return arith_image_function_2_1_IMGID(&inimg1, &inimg2, &outimg, pt2function);
}

errno_t arith_image_function_2_1_inplace_byID(
    imageID ID1,
    imageID ID2,
    double (*pt2function)(double, double)
)
{
    long    ii;
    long    nelement1, nelement2, nelement;
    uint8_t datatype1, datatype2;

    datatype1 = data.image[ID1].md[0].datatype;
    datatype2 = data.image[ID2].md[0].datatype;
    nelement1 = data.image[ID1].md[0].nelement;
    nelement2 = data.image[ID2].md[0].nelement;

    nelement = nelement1;
    if(nelement1 != nelement2)
    {
        PRINT_ERROR("images %ld and %ld have different number of elements\n",
                    ID1,
                    ID2);
        exit(0);
    }

    data.image[ID1].md[0].write = 1;

#ifdef _OPENMP
    #pragma omp parallel if (nelement > OMP_NELEMENT_LIMIT)
    {
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            double v1 = get_pixel_double(&data.image[ID1], ii);
            double v2 = get_pixel_double(&data.image[ID2], ii);
            if (datatype1 == _DATATYPE_FLOAT)
                data.image[ID1].array.F[ii] = (float)pt2function(v1, v2);
            else if (datatype1 == _DATATYPE_DOUBLE)
                data.image[ID1].array.D[ii] = pt2function(v1, v2);
        }
#ifdef _OPENMP
    }
#endif

    data.image[ID1].md[0].write = 0;
    data.image[ID1].md[0].cnt0++;

    return EXIT_SUCCESS;
}

errno_t arith_image_function_2_1_inplace(
    const char *ID_name1,
    const char *ID_name2,
    double (*pt2function)(double, double))
{
    imageID ID1;
    imageID ID2;

    ID1 = image_ID(ID_name1);
    ID2 = image_ID(ID_name2);

    arith_image_function_2_1_inplace_byID(ID1, ID2, pt2function);

    return EXIT_SUCCESS;
}

/* ------------------------------------------------------------------------- */
/* complex image, complex image  -> complex image                            */
/* ------------------------------------------------------------------------- */
// complex float (CF), complex float (CF) -> complex float (CF)
errno_t arith_image_function_CF_CF__CF(
    const char *ID_name1,
    const char *ID_name2,
    const char *ID_out,
    complex_float(*pt2function)(complex_float, complex_float))
{
    imageID   ID1;
    imageID   ID2;
    imageID   IDout;
    long      ii;
    uint32_t *naxes = NULL;
    long      nelement;
    long      naxis;
    uint8_t   datatype1; //, datatype2;
    long      i;

    ID1       = image_ID(ID_name1);
    ID2       = image_ID(ID_name2);
    datatype1 = data.image[ID1].md[0].datatype;
    //datatype2 = data.image[ID2].md[0].datatype;
    naxis = data.image[ID1].md[0].naxis;
    naxes = (uint32_t *) malloc(sizeof(uint32_t) * naxis);
    if(naxes == NULL) { PRINT_ERROR("malloc() error"); exit(0); }
    for(i = 0; i < naxis; i++) naxes[i] = data.image[ID1].md[0].size[i];

    create_image_ID(ID_out,
                    naxis,
                    naxes,
                    datatype1,
                    data.SHARED_DFT,
                    NB_KEYWNODE_MAX,
                    0,
                    &IDout);
    free(naxes);
    nelement = data.image[ID1].md[0].nelement;

#ifdef _OPENMP
    #pragma omp parallel if (nelement > OMP_NELEMENT_LIMIT)
    {
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            data.image[IDout].array.CF[ii] = pt2function(data.image[ID1].array.CF[ii], data.image[ID2].array.CF[ii]);
        }
#ifdef _OPENMP
    }
#endif
    return RETURN_SUCCESS;
}

// complex double (CD), complex double (CD) -> complex double (CD)
errno_t arith_image_function_CD_CD__CD(
    const char *ID_name1,
    const char *ID_name2,
    const char *ID_out,
    complex_double(*pt2function)(complex_double, complex_double))
{
    imageID   ID1;
    imageID   ID2;
    imageID   IDout;
    long      ii;
    uint32_t *naxes = NULL;
    long      nelement;
    long      naxis;
    uint8_t   datatype1; //, datatype2;
    long      i;

    ID1       = image_ID(ID_name1);
    ID2       = image_ID(ID_name2);
    datatype1 = data.image[ID1].md[0].datatype;
    //datatype2 = data.image[ID2].md[0].datatype;
    naxis = data.image[ID1].md[0].naxis;
    naxes = (uint32_t *) malloc(sizeof(uint32_t) * naxis);
    if(naxes == NULL) { PRINT_ERROR("malloc() error"); exit(0); }
    for(i = 0; i < naxis; i++) naxes[i] = data.image[ID1].md[0].size[i];

    create_image_ID(ID_out,
                    naxis,
                    naxes,
                    datatype1,
                    data.SHARED_DFT,
                    NB_KEYWNODE_MAX,
                    0,
                    &IDout);
    free(naxes);
    nelement = data.image[ID1].md[0].nelement;

#ifdef _OPENMP
    #pragma omp parallel if (nelement > OMP_NELEMENT_LIMIT)
    {
        #pragma omp for
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            data.image[IDout].array.CD[ii] = pt2function(data.image[ID1].array.CD[ii], data.image[ID2].array.CD[ii]);
        }
#ifdef _OPENMP
    }
#endif
    return RETURN_SUCCESS;
}

int arith_image_function_1f_1_IMGID(IMGID *imgin, double f1, IMGID *imgout, double (*pt2function)(double, double))
{
    long ii;
    resolveIMGID(imgin, ERRMODE_ABORT);
    imgout->naxis = imgin->md->naxis;
    for(int i = 0; i < imgin->md->naxis; i++) imgout->size[i] = imgin->md->size[i];
    imgout->datatype = (imgin->md->datatype == _DATATYPE_DOUBLE) ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT;
    imgout->shared = data.SHARED_DFT; imgout->NBkw = NB_KEYWNODE_MAX;
    imcreateIMGID(imgout);
    uint_fast64_t nelement = imgin->md->nelement;

    if (imgin->md->datatype == _DATATYPE_FLOAT && imgout->datatype == _DATATYPE_FLOAT)
    {
        float *ptr = imgin->im->array.F;
        float *out = imgout->im->array.F;
#ifdef _OPENMP
    #pragma omp parallel for if (nelement > OMP_NELEMENT_LIMIT)
#endif
        for(ii = 0; ii < nelement; ii++)
        {
             out[ii] = (float)pt2function((double)ptr[ii], f1);
        }
    }
    else if (imgin->md->datatype == _DATATYPE_DOUBLE && imgout->datatype == _DATATYPE_DOUBLE)
    {
        double *ptr = imgin->im->array.D;
        double *out = imgout->im->array.D;
#ifdef _OPENMP
    #pragma omp parallel for if (nelement > OMP_NELEMENT_LIMIT)
#endif
        for(ii = 0; ii < nelement; ii++)
        {
             out[ii] = pt2function(ptr[ii], f1);
        }
    }
    else
    {
#ifdef _OPENMP
    #pragma omp parallel for if (nelement > OMP_NELEMENT_LIMIT)
#endif
        for(ii = 0; ii < nelement; ii++)
        {
            double v = get_pixel_double(imgin->im, ii);
            if (imgout->datatype == _DATATYPE_FLOAT) imgout->im->array.F[ii] = (float)pt2function(v, f1);
            else imgout->im->array.D[ii] = pt2function(v, f1);
        }
    }
    return EXIT_SUCCESS;
}

int arith_image_function_1f_1(const char *ID_name, double f1, const char *ID_out, double (*pt2function)(double, double))
{
    IMGID imgin = mkIMGID_from_name(ID_name); IMGID imgout = mkIMGID_from_name(ID_out);
    return arith_image_function_1f_1_IMGID(&imgin, f1, &imgout, pt2function);
}

int arith_image_function_1f_1_inplace_byID(long ID, double f1, double (*pt2function)(double, double))
{
    long ii; uint_fast64_t nelement = data.image[ID].md[0].nelement;
#ifdef _OPENMP
    #pragma omp parallel for if (nelement > OMP_NELEMENT_LIMIT)
#endif
    for(ii = 0; ii < nelement; ii++)
    {
        double v = get_pixel_double(&data.image[ID], ii);
        if (data.image[ID].md[0].datatype == _DATATYPE_FLOAT) data.image[ID].array.F[ii] = (float)pt2function(v, f1);
        else if (data.image[ID].md[0].datatype == _DATATYPE_DOUBLE) data.image[ID].array.D[ii] = pt2function(v, f1);
    }
    return EXIT_SUCCESS;
}

int arith_image_function_1f_1_inplace(const char *ID_name, double f1, double (*pt2function)(double, double))
{
    return arith_image_function_1f_1_inplace_byID(image_ID(ID_name), f1, pt2function);
}

int arith_image_function_1ff_1_IMGID(IMGID *imgin, double f1, double f2, IMGID *imgout, double (*pt2function)(double, double, double))
{
    long ii; resolveIMGID(imgin, ERRMODE_ABORT);
    imgout->naxis = imgin->md->naxis;
    for(int i = 0; i < imgin->md->naxis; i++) imgout->size[i] = imgin->md->size[i];
    imgout->datatype = (imgin->md->datatype == _DATATYPE_DOUBLE) ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT;
    imgout->shared = data.SHARED_DFT; imgout->NBkw = NB_KEYWNODE_MAX;
    imcreateIMGID(imgout);
    uint_fast64_t nelement = imgin->md->nelement;
#ifdef _OPENMP
    #pragma omp parallel for if (nelement > OMP_NELEMENT_LIMIT)
#endif
    for(ii = 0; ii < nelement; ii++)
    {
        double v = get_pixel_double(imgin->im, ii);
        if (imgout->datatype == _DATATYPE_FLOAT) imgout->im->array.F[ii] = (float)pt2function(v, f1, f2);
        else imgout->im->array.D[ii] = pt2function(v, f1, f2);
    }
    return (0);
}

int arith_image_function_1ff_1(const char *ID_name, double f1, double f2, const char *ID_out, double (*pt2function)(double, double, double))
{
    IMGID imgin = mkIMGID_from_name(ID_name); IMGID imgout = mkIMGID_from_name(ID_out);
    return arith_image_function_1ff_1_IMGID(&imgin, f1, f2, &imgout, pt2function);
}

int arith_image_function_1ff_1_inplace_byID(long ID, double f1, double f2, double (*pt2function)(double, double, double))
{
    long ii; uint_fast64_t nelement = data.image[ID].md[0].nelement;
#ifdef _OPENMP
    #pragma omp parallel for if (nelement > OMP_NELEMENT_LIMIT)
#endif
    for(ii = 0; ii < nelement; ii++)
    {
        double v = get_pixel_double(&data.image[ID], ii);
        if (data.image[ID].md[0].datatype == _DATATYPE_FLOAT) data.image[ID].array.F[ii] = (float)pt2function(v, f1, f2);
        else if (data.image[ID].md[0].datatype == _DATATYPE_DOUBLE) data.image[ID].array.D[ii] = pt2function(v, f1, f2);
    }
    return (0);
}

int arith_image_function_1ff_1_inplace(const char *ID_name, double f1, double f2, double (*pt2function)(double, double, double))
{
    return arith_image_function_1ff_1_inplace_byID(image_ID(ID_name), f1, f2, pt2function);
}

/* Specialized optimized arithmetic functions */

#define ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(name, funcname) \
errno_t arith_image_##name##_optimized_IMGID(IMGID *imgin, IMGID *imgout) \
{ \
    DEBUG_TRACE_FSTART(); \
    resolveIMGID(imgin, ERRMODE_ABORT); \
    resolveIMGID(imgout, ERRMODE_NULL); \
    if(imgout->ID == -1) copyIMGID(imgin, imgout); \
    imcreateIMGID(imgout); \
    uint64_t nelement = imgout->md->nelement; \
    if(imgin->md->datatype == _DATATYPE_FLOAT && imgout->datatype == _DATATYPE_FLOAT) \
    { \
        float *p1 = imgin->im->array.F; \
        float *po = imgout->im->array.F; \
        _Pragma("omp parallel for if (nelement > OMP_NELEMENT_LIMIT)") \
        for(uint64_t i=0; i<nelement; i++) po[i] = (float)funcname((double)p1[i]); \
    } \
    else if(imgin->md->datatype == _DATATYPE_DOUBLE && imgout->datatype == _DATATYPE_DOUBLE) \
    { \
        double *p1 = imgin->im->array.D; \
        double *po = imgout->im->array.D; \
        _Pragma("omp parallel for if (nelement > OMP_NELEMENT_LIMIT)") \
        for(uint64_t i=0; i<nelement; i++) po[i] = funcname(p1[i]); \
    } \
    else \
    { \
        arith_image_function_1_1_IMGID(imgin, imgout, &P##name); \
    } \
    DEBUG_TRACE_FEXIT(); \
    return RETURN_SUCCESS; \
}

ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(acos, acos)
ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(asin, asin)
ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(atan, atan)
ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(ceil, ceil)
ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(cos, cos)
ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(cosh, cosh)
ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(exp, exp)
ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(fabs, fabs)
ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(floor, floor)
ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(ln, log)
ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(log, log10)
ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(sqrt, sqrt)
ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(sin, sin)
ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(sinh, sinh)
ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(tan, tan)
ARITH_UNARY_OPTIMIZED_FUNCTION_CALL(tanh, tanh)

#define ARITH_OPTIMIZED_FUNCTION(name, op) \
errno_t arith_image_##name##_optimized_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout) \
{ \
    DEBUG_TRACE_FSTART(); \
    resolveIMGID(imgin1, ERRMODE_ABORT); \
    resolveIMGID(imgin2, ERRMODE_ABORT); \
    resolveIMGID(imgout, ERRMODE_NULL); \
    if(imgout->ID == -1) copyIMGID(imgin1, imgout); \
    imcreateIMGID(imgout); \
    uint64_t nelement = imgout->md->nelement; \
    if(imgin1->md->datatype == _DATATYPE_FLOAT && imgin2->md->datatype == _DATATYPE_FLOAT && imgout->datatype == _DATATYPE_FLOAT) \
    { \
        float *p1 = imgin1->im->array.F; \
        float *p2 = imgin2->im->array.F; \
        float *po = imgout->im->array.F; \
        _Pragma("omp parallel for if (nelement > OMP_NELEMENT_LIMIT)") \
        for(uint64_t i=0; i<nelement; i++) po[i] = p1[i] op p2[i]; \
    } \
    else if(imgin1->md->datatype == _DATATYPE_DOUBLE && imgin2->md->datatype == _DATATYPE_DOUBLE && imgout->datatype == _DATATYPE_DOUBLE) \
    { \
        double *p1 = imgin1->im->array.D; \
        double *p2 = imgin2->im->array.D; \
        double *po = imgout->im->array.D; \
        _Pragma("omp parallel for if (nelement > OMP_NELEMENT_LIMIT)") \
        for(uint64_t i=0; i<nelement; i++) po[i] = p1[i] op p2[i]; \
    } \
    else \
    { \
        arith_image_function_2_1_IMGID(imgin1, imgin2, imgout, &P##name); \
    } \
    DEBUG_TRACE_FEXIT(); \
    return RETURN_SUCCESS; \
}

ARITH_OPTIMIZED_FUNCTION(add, +)
ARITH_OPTIMIZED_FUNCTION(sub, -)
ARITH_OPTIMIZED_FUNCTION(mult, *)
ARITH_OPTIMIZED_FUNCTION(div, /)

#define ARITH_CST_OPTIMIZED_FUNCTION(name, op) \
errno_t arith_image_cst##name##_optimized_IMGID(IMGID *imgin, double f1, IMGID *imgout) \
{ \
    DEBUG_TRACE_FSTART(); \
    resolveIMGID(imgin, ERRMODE_ABORT); \
    resolveIMGID(imgout, ERRMODE_NULL); \
    if(imgout->ID == -1) copyIMGID(imgin, imgout); \
    imcreateIMGID(imgout); \
    uint64_t nelement = imgout->md->nelement; \
    if(imgin->md->datatype == _DATATYPE_FLOAT && imgout->datatype == _DATATYPE_FLOAT) \
    { \
        float *p1 = imgin->im->array.F; \
        float *po = imgout->im->array.F; \
        float cf1 = (float)f1; \
        _Pragma("omp parallel for if (nelement > OMP_NELEMENT_LIMIT)") \
        for(uint64_t i=0; i<nelement; i++) po[i] = p1[i] op cf1; \
    } \
    else if(imgin->md->datatype == _DATATYPE_DOUBLE && imgout->datatype == _DATATYPE_DOUBLE) \
    { \
        double *p1 = imgin->im->array.D; \
        double *po = imgout->im->array.D; \
        _Pragma("omp parallel for if (nelement > OMP_NELEMENT_LIMIT)") \
        for(uint64_t i=0; i<nelement; i++) po[i] = p1[i] op f1; \
    } \
    else \
    { \
        arith_image_function_1f_1_IMGID(imgin, f1, imgout, &P##name); \
    } \
    DEBUG_TRACE_FEXIT(); \
    return RETURN_SUCCESS; \
}

ARITH_CST_OPTIMIZED_FUNCTION(add, +)
ARITH_CST_OPTIMIZED_FUNCTION(sub, -)
ARITH_CST_OPTIMIZED_FUNCTION(mult, *)
ARITH_CST_OPTIMIZED_FUNCTION(div, /)

#define ARITH_OPTIMIZED_FUNCTION_CALL(name, funcname) \
errno_t arith_image_##name##_optimized_IMGID(IMGID *imgin1, IMGID *imgin2, IMGID *imgout) \
{ \
    DEBUG_TRACE_FSTART(); \
    resolveIMGID(imgin1, ERRMODE_ABORT); \
    resolveIMGID(imgin2, ERRMODE_ABORT); \
    resolveIMGID(imgout, ERRMODE_NULL); \
    if(imgout->ID == -1) copyIMGID(imgin1, imgout); \
    imcreateIMGID(imgout); \
    uint64_t nelement = imgout->md->nelement; \
    if(imgin1->md->datatype == _DATATYPE_FLOAT && imgin2->md->datatype == _DATATYPE_FLOAT && imgout->datatype == _DATATYPE_FLOAT) \
    { \
        float *p1 = imgin1->im->array.F; \
        float *p2 = imgin2->im->array.F; \
        float *po = imgout->im->array.F; \
        _Pragma("omp parallel for if (nelement > OMP_NELEMENT_LIMIT)") \
        for(uint64_t i=0; i<nelement; i++) po[i] = (float)funcname((double)p1[i], (double)p2[i]); \
    } \
    else if(imgin1->md->datatype == _DATATYPE_DOUBLE && imgin2->md->datatype == _DATATYPE_DOUBLE && imgout->datatype == _DATATYPE_DOUBLE) \
    { \
        double *p1 = imgin1->im->array.D; \
        double *p2 = imgin2->im->array.D; \
        double *po = imgout->im->array.D; \
        _Pragma("omp parallel for if (nelement > OMP_NELEMENT_LIMIT)") \
        for(uint64_t i=0; i<nelement; i++) po[i] = funcname(p1[i], p2[i]); \
    } \
    else \
    { \
        arith_image_function_2_1_IMGID(imgin1, imgin2, imgout, &P##name); \
    } \
    DEBUG_TRACE_FEXIT(); \
    return RETURN_SUCCESS; \
}

ARITH_OPTIMIZED_FUNCTION_CALL(pow, pow)
ARITH_OPTIMIZED_FUNCTION_CALL(fmod, fmod)
ARITH_OPTIMIZED_FUNCTION_CALL(minv, fmin)
ARITH_OPTIMIZED_FUNCTION_CALL(maxv, fmax)