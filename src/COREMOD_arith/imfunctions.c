/**
 * @file    imfunctions.c
 * @brief   apply math functions to images
 *
 *
 */

#include <assert.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

#ifdef _OPENMP
#include <omp.h>
#define OMP_NELEMENT_LIMIT 1000000
#endif

/* ------------------------------------------------------------------------- */
/* Functions for bison / flex                                                */
/* im : image
  d : double

  function_<inputformat>_<outputformat>__<math function input>_<math function output>

  examples:
  function_imim__dd_d  : input is (image, image), applies double,double -> double function

  ------------------------------------------------------------------------- */

errno_t arith_image_function_im_im__d_d(
    const char * __restrict ID_name,
    const char * __restrict ID_out,
    double (*pt2function)(double))
{
    imageID   ID;
    imageID   IDout;
    long      naxis;

    uint8_t   datatype, datatypeout;


    DEBUG_TRACEPOINT("arith_image_function_d_d  %s %s\n", ID_name, ID_out);

    ID       = image_ID(ID_name);
    datatype = data.image[ID].md[0].datatype;
    naxis    = data.image[ID].md[0].naxis;
    uint32_t naxes[3];


    for(uint8_t i = 0; i < naxis; i++)
    {
        naxes[i] = data.image[ID].md[0].size[i];
    }

    datatypeout = _DATATYPE_FLOAT;
    if(datatype == _DATATYPE_DOUBLE)
    {
        datatypeout = _DATATYPE_DOUBLE;
    }

    create_image_ID(ID_out,
                    naxis,
                    naxes,
                    datatypeout,
                    data.SHARED_DFT,
                    data.NBKEYWORD_DFT,
                    0,
                    &IDout);


    uint_fast64_t nelement = data.image[ID].md[0].nelement;

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
                data.image[IDout].array.F[ii] = (float) pt2function(
                                                    (double)(data.image[ID].array.UI8[ii]));
            }
        }
        if(datatype == _DATATYPE_UINT16)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(uint_fast64_t ii = 0; ii < nelement; ii++)
            {
                data.image[IDout].array.F[ii] = (float) pt2function(
                                                    (double)(data.image[ID].array.UI16[ii]));
            }
        }
        if(datatype == _DATATYPE_UINT32)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(uint_fast64_t ii = 0; ii < nelement; ii++)
            {
                data.image[IDout].array.F[ii] = (float) pt2function(
                                                    (double)(data.image[ID].array.UI32[ii]));
            }
        }
        if(datatype == _DATATYPE_UINT64)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(uint_fast64_t ii = 0; ii < nelement; ii++)
            {
                data.image[IDout].array.F[ii] = (float) pt2function(
                                                    (double)(data.image[ID].array.UI64[ii]));
            }
        }

        if(datatype == _DATATYPE_INT8)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(uint_fast64_t ii = 0; ii < nelement; ii++)
            {
                data.image[IDout].array.F[ii] = (float) pt2function(
                                                    (double)(data.image[ID].array.SI8[ii]));
            }
        }
        if(datatype == _DATATYPE_INT16)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(uint_fast64_t ii = 0; ii < nelement; ii++)
            {
                data.image[IDout].array.F[ii] = (float) pt2function(
                                                    (double)(data.image[ID].array.SI16[ii]));
            }
        }
        if(datatype == _DATATYPE_INT32)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(uint_fast64_t ii = 0; ii < nelement; ii++)
            {
                data.image[IDout].array.F[ii] = (float) pt2function(
                                                    (double)(data.image[ID].array.SI32[ii]));
            }
        }
        if(datatype == _DATATYPE_INT64)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(uint_fast64_t ii = 0; ii < nelement; ii++)
            {
                data.image[IDout].array.F[ii] = (float) pt2function(
                                                    (double)(data.image[ID].array.SI64[ii]));
            }
        }

        if(datatype == _DATATYPE_FLOAT)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(uint_fast64_t ii = 0; ii < nelement; ii++)
            {
                data.image[IDout].array.F[ii] =
                    (float) pt2function((double)(data.image[ID].array.F[ii]));
            }
        }
        if(datatype == _DATATYPE_DOUBLE)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(uint_fast64_t ii = 0; ii < nelement; ii++)
            {
                data.image[IDout].array.D[ii] =
                    pt2function(data.image[ID].array.D[ii]);
            }
        }
#ifdef _OPENMP
    }
#endif

    DEBUG_TRACEPOINT("arith_image_function_d_d  DONE\n");

    return RETURN_SUCCESS;
}





errno_t arith_image_function_imd_im__dd_d(
    const char * __restrict ID_name,
    double      v0,
    const char * __restrict ID_out,
    double (*pt2function)(double, double))
{
    imageID   ID;
    imageID   IDout;
    uint32_t *naxes = NULL;
    long      naxis;
    long      ii;
    long      nelement;
    uint8_t   datatype, datatypeout;
    long      i;

    ID       = image_ID(ID_name);
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

    datatypeout = _DATATYPE_FLOAT;
    if(datatype == _DATATYPE_DOUBLE)
    {
        datatypeout = _DATATYPE_DOUBLE;
    }

    create_image_ID(ID_out,
                    naxis,
                    naxes,
                    datatypeout,
                    data.SHARED_DFT,
                    data.NBKEYWORD_DFT,
                    0,
                    &IDout);
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
                    (float) pt2function((double)(data.image[ID].array.UI8[ii]),
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
                data.image[IDout].array.F[ii] = (float) pt2function(
                                                    (double)(data.image[ID].array.UI16[ii]),
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
                data.image[IDout].array.F[ii] = (float) pt2function(
                                                    (double)(data.image[ID].array.UI32[ii]),
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
                data.image[IDout].array.F[ii] = (float) pt2function(
                                                    (double)(data.image[ID].array.UI64[ii]),
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
                data.image[IDout].array.F[ii] =
                    (float) pt2function((double)(data.image[ID].array.SI8[ii]),
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
                data.image[IDout].array.F[ii] = (float) pt2function(
                                                    (double)(data.image[ID].array.SI16[ii]),
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
                data.image[IDout].array.F[ii] = (float) pt2function(
                                                    (double)(data.image[ID].array.SI32[ii]),
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
                data.image[IDout].array.F[ii] = (float) pt2function(
                                                    (double)(data.image[ID].array.SI64[ii]),
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
                data.image[IDout].array.F[ii] =
                    (float) pt2function((double)(data.image[ID].array.F[ii]),
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
                data.image[IDout].array.D[ii] =
                    pt2function(data.image[ID].array.D[ii], v0);
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
    imageID   ID;
    imageID   IDout;
    uint32_t *naxes = NULL;
    long      naxis;
    long      ii;
    long      nelement;
    uint8_t   datatype, datatypeout;
    long      i;

    ID       = image_ID(ID_name);
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

    datatypeout = _DATATYPE_FLOAT;
    if(datatype == _DATATYPE_DOUBLE)
    {
        datatypeout = _DATATYPE_DOUBLE;
    }

    create_image_ID(ID_out,
                    naxis,
                    naxes,
                    datatypeout,
                    data.SHARED_DFT,
                    data.NBKEYWORD_DFT,
                    0,
                    &IDout);
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
                    (float) pt2function((double)(data.image[ID].array.UI8[ii]),
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
                data.image[IDout].array.F[ii] = (float) pt2function(
                                                    (double)(data.image[ID].array.UI16[ii]),
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
                data.image[IDout].array.F[ii] = (float) pt2function(
                                                    (double)(data.image[ID].array.UI32[ii]),
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
                data.image[IDout].array.F[ii] = (float) pt2function(
                                                    (double)(data.image[ID].array.UI64[ii]),
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
                data.image[IDout].array.F[ii] =
                    (float) pt2function((double)(data.image[ID].array.SI8[ii]),
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
                data.image[IDout].array.F[ii] = (float) pt2function(
                                                    (double)(data.image[ID].array.SI16[ii]),
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
                data.image[IDout].array.F[ii] = (float) pt2function(
                                                    (double)(data.image[ID].array.SI32[ii]),
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
                data.image[IDout].array.F[ii] = (float) pt2function(
                                                    (double)(data.image[ID].array.SI64[ii]),
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
                data.image[IDout].array.F[ii] =
                    (float) pt2function((double)(data.image[ID].array.F[ii]),
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
                data.image[IDout].array.D[ii] =
                    pt2function(data.image[ID].array.D[ii], v0, v1);
            }
        }
#ifdef _OPENMP
    }
#endif

    DEBUG_TRACEPOINT("arith_image_function_d_d  DONE\n");

    return RETURN_SUCCESS;
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

errno_t arith_image_function_1_1(const char *ID_name,
                                 const char *ID_out,
                                 double (*pt2function)(double))
{
    imageID   ID;
    imageID   IDout;
    uint32_t *naxes = NULL;
    long      naxis;
    long      ii;
    long      nelement;
    uint8_t   datatype, datatypeout;
    long      i;

    //  printf("arith_image_function_1_1\n");

    ID       = image_ID(ID_name);
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

    datatypeout = _DATATYPE_FLOAT;
    if(datatype == _DATATYPE_DOUBLE)
    {
        datatypeout = _DATATYPE_DOUBLE;
    }

    create_image_ID(ID_out,
                    naxis,
                    naxes,
                    datatypeout,
                    data.SHARED_DFT,
                    data.NBKEYWORD_DFT,
                    0,
                    &IDout);
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

errno_t arith_image_function_2_1(const char *ID_name1,
                                 const char *ID_name2,
                                 const char *ID_out,
                                 double (*pt2function)(double, double))
{
    imageID   ID1;
    imageID   ID2;
    imageID   IDout;
    long      ii, kk;
    uint32_t *naxes  = NULL; // input, output
    uint32_t *naxes2 = NULL;
    long      nelement1, nelement2, nelement;
    uint8_t   naxis, naxis2;
    uint8_t   datatype1, datatype2, datatypeout;

    int  op3D2Dto3D = 0; // 3D image, 2D image -> 3D image
    long xysize;

    ID1 = image_ID(ID_name1);
    ID2 = image_ID(ID_name2);

    //list_image_ID(); //TEST
    DEBUG_TRACEPOINT("%s  IDs : %ld %ld\n", __FUNCTION__, ID1, ID2);

    if(ID1 == -1)
    {
        PRINT_WARNING("Image %s does not exist: cannot proceed\n", ID_name1);
        return 1;
    }

    if(ID2 == -1)
    {
        PRINT_WARNING("Image %s does not exist: cannot proceed\n", ID_name2);
        return 1;
    }

    datatype1 = data.image[ID1].md[0].datatype;
    datatype2 = data.image[ID2].md[0].datatype;
    naxis     = data.image[ID1].md[0].naxis;
    naxis2    = data.image[ID2].md[0].naxis;

    naxes = (uint32_t *) malloc(sizeof(uint32_t) * naxis);
    if(naxes == NULL)
    {
        PRINT_ERROR("malloc() error");
        abort();
    }

    naxes2 = (uint32_t *) malloc(sizeof(uint32_t) * naxis);
    if(naxes2 == NULL)
    {
        PRINT_ERROR("malloc() error");
        abort();
    }

    for(uint8_t i = 0; i < naxis; i++)
    {
        naxes[i] = data.image[ID1].md[0].size[i];
    }
    for(uint8_t i = 0; i < naxis2; i++)
    {
        naxes2[i] = data.image[ID2].md[0].size[i];
    }

    datatypeout = _DATATYPE_FLOAT; // default

    // other cases

    // DOUBLE * -> DOUBLE
    if(datatype1 == _DATATYPE_DOUBLE)
    {
        datatypeout = _DATATYPE_DOUBLE;
    }

    // * DOUBLE -> DOUBLE
    if(datatype2 == _DATATYPE_DOUBLE)
    {
        datatypeout = _DATATYPE_DOUBLE;
    }

    create_image_ID(ID_out,
                    naxis,
                    naxes,
                    datatypeout,
                    data.SHARED_DFT,
                    data.NBKEYWORD_DFT,
                    0,
                    &IDout);

    nelement1 = data.image[ID1].md[0].nelement;
    nelement2 = data.image[ID2].md[0].nelement;

    // test if 3D 2D -> 3D operation

    op3D2Dto3D = 0;
    xysize     = 0;
    if((naxis == 3) && (naxis2 == 2))
    {
        DEBUG_TRACEPOINT("naxes:  %ld %ld     %ld %ld\n",
               (long) naxes[0],
               (long) naxes2[0],
               (long) naxes[1],
               (long) naxes2[1]);
        if((naxes[0] == naxes2[0]) && (naxes[1] == naxes2[1]))
        {
            op3D2Dto3D = 1;
            xysize     = naxes[0] * naxes[1];
            DEBUG_TRACEPOINT("input : 3D im, 2D im -> output : 3D im\n");
            //list_image_ID();
        }
    }

    nelement = nelement1;
    if(op3D2Dto3D == 0)
        if(nelement1 != nelement2)
        {
            PRINT_ERROR(
                "images %s and %s have different number of elements ( %ld "
                "%ld )\n",
                ID_name1,
                ID_name2,
                nelement1,
                nelement2);
            exit(0);
        }

    //# ifdef _OPENMP
    //    #pragma omp parallel if (nelement>OMP_NELEMENT_LIMIT)
    //    {
    //# endif

    // ID1 datatype  UINT8
    if(datatype1 == _DATATYPE_UINT8)
    {
        if(datatype2 == _DATATYPE_UINT8)  // UINT8 UINT8 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI8[ii]),
                                    (double)(data.image[ID2].array.UI8[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI8[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI8[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_UINT16)  // UINT8 UINT16 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI8[ii]),
                                    (double)(data.image[ID2].array.UI16[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI8[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI16[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_UINT32)  // UINT8 UINT32 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI8[ii]),
                                    (double)(data.image[ID2].array.UI32[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI8[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI32[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_UINT64)  // UINT8 UINT64 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI8[ii]),
                                    (double)(data.image[ID2].array.UI64[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI8[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI64[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT8)  // UINT8 INT8 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI8[ii]),
                                    (double)(data.image[ID2].array.SI8[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI8[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI8[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT16)  // UINT8 INT16 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI8[ii]),
                                    (double)(data.image[ID2].array.SI16[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI8[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI16[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT32)  // UINT8 INT32 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI8[ii]),
                                    (double)(data.image[ID2].array.SI32[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI8[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI32[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT64)  // UINT8 INT64 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI8[ii]),
                                    (double)(data.image[ID2].array.SI64[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI8[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI64[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_FLOAT)  // UINT8 FLOAT -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI8[ii]),
                                    (double)(data.image[ID2].array.F[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI8[kk * xysize + ii]),
                                (double)(data.image[ID2].array.F[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_DOUBLE)  // UINT8 DOUBLE -> DOUBLE
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.D[ii] =
                        pt2function((double)(data.image[ID1].array.UI8[ii]),
                                    data.image[ID2].array.D[ii]);
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.D[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI8[kk * xysize + ii]),
                                data.image[ID2].array.D[ii]);
                    }
        }
    }

    // ID1 datatype  UINT16

    if(datatype1 == _DATATYPE_UINT16)
    {
        if(datatype2 == _DATATYPE_UINT8)  // UINT16 UINT8 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI16[ii]),
                                    (double)(data.image[ID2].array.UI8[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI16[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI8[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_UINT16)  // UINT16 UINT16 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI16[ii]),
                                    (double)(data.image[ID2].array.UI16[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI16[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI16[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_UINT32)  // UINT16 UINT32 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI16[ii]),
                                    (double)(data.image[ID2].array.UI32[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI16[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI32[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_UINT64)  // UINT16 UINT64 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI16[ii]),
                                    (double)(data.image[ID2].array.UI64[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI16[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI64[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT8)  // UINT16 INT8 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI16[ii]),
                                    (double)(data.image[ID2].array.SI8[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI16[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI8[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT16)  // UINT16 INT16 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI16[ii]),
                                    (double)(data.image[ID2].array.SI16[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI16[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI16[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT32)  // UINT16 INT32 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI16[ii]),
                                    (double)(data.image[ID2].array.SI32[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI16[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI32[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT64)  // UINT16 INT64 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI16[ii]),
                                    (double)(data.image[ID2].array.SI64[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI16[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI64[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_FLOAT)  // UINT16 FLOAT -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI16[ii]),
                                    (double)(data.image[ID2].array.F[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI16[kk * xysize + ii]),
                                (double)(data.image[ID2].array.F[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_DOUBLE)  // UINT16 DOUBLE -> DOUBLE
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.D[ii] =
                        pt2function((double)(data.image[ID1].array.UI16[ii]),
                                    data.image[ID2].array.D[ii]);
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.D[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI16[kk * xysize + ii]),
                                data.image[ID2].array.D[ii]);
                    }
        }
    }

    // ID1 datatype  UINT32
    if(datatype1 == _DATATYPE_UINT32)
    {

        if(datatype2 == _DATATYPE_UINT8)  // UINT32 UINT8 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI32[ii]),
                                    (double)(data.image[ID2].array.UI8[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI32[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI8[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_UINT16)  // UINT32 UINT16 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI32[ii]),
                                    (double)(data.image[ID2].array.UI16[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI32[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI16[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_UINT32)  // UINT32 UINT32 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI32[ii]),
                                    (double)(data.image[ID2].array.UI32[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI32[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI32[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_UINT64)  // UINT32 UINT64 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI32[ii]),
                                    (double)(data.image[ID2].array.UI64[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI32[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI64[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT8)  // UINT32 INT8 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI32[ii]),
                                    (double)(data.image[ID2].array.SI8[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI32[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI8[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT16)  // UINT32 INT16 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI32[ii]),
                                    (double)(data.image[ID2].array.SI16[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI32[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI16[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT32)  // UINT32 INT32 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI32[ii]),
                                    (double)(data.image[ID2].array.SI32[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI32[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI32[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT64)  // UINT32 INT64 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI32[ii]),
                                    (double)(data.image[ID2].array.SI64[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI32[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI64[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_FLOAT)  // UINT32 FLOAT -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI32[ii]),
                                    (double)(data.image[ID2].array.F[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI32[kk * xysize + ii]),
                                (double)(data.image[ID2].array.F[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_DOUBLE)  // UINT32 DOUBLE -> DOUBLE
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.D[ii] =
                        pt2function((double)(data.image[ID1].array.UI32[ii]),
                                    data.image[ID2].array.D[ii]);
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.D[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI32[kk * xysize + ii]),
                                data.image[ID2].array.D[ii]);
                    }
        }
    }

    // ID1 datatype  UINT64
    if(datatype1 == _DATATYPE_UINT64)
    {
        if(datatype2 == _DATATYPE_UINT8)  // UINT64 UINT8 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI64[ii]),
                                    (double)(data.image[ID2].array.UI8[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI64[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI8[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_UINT16)  // UINT64 UINT16 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI64[ii]),
                                    (double)(data.image[ID2].array.UI16[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI64[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI16[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_UINT32)  // UINT64 UINT32 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI64[ii]),
                                    (double)(data.image[ID2].array.UI32[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI64[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI32[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_UINT64)  // UINT64 UINT64 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI64[ii]),
                                    (double)(data.image[ID2].array.UI64[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI64[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI64[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT8)  // UINT64 INT8 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI64[ii]),
                                    (double)(data.image[ID2].array.SI8[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI64[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI8[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT16)  // UINT64 INT16 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI64[ii]),
                                    (double)(data.image[ID2].array.SI16[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI64[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI16[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT32)  // UINT64 INT32 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI64[ii]),
                                    (double)(data.image[ID2].array.SI32[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI64[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI32[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT64)  // UINT64 INT64 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI64[ii]),
                                    (double)(data.image[ID2].array.SI64[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI64[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI64[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_FLOAT)  // UINT64 FLOAT -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.UI64[ii]),
                                    (double)(data.image[ID2].array.F[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI64[kk * xysize + ii]),
                                (double)(data.image[ID2].array.F[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_DOUBLE)  // UINT64 DOUBLE -> DOUBLE
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.D[ii] =
                        pt2function((double)(data.image[ID1].array.UI64[ii]),
                                    data.image[ID2].array.D[ii]);
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.D[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.UI64[kk * xysize + ii]),
                                data.image[ID2].array.D[ii]);
                    }
        }
    }

    // ID1 datatype  INT8

    if(datatype1 == _DATATYPE_INT8)
    {
        if(datatype2 == _DATATYPE_UINT8)  // INT8 UINT8 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI8[ii]),
                                    (double)(data.image[ID2].array.UI8[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI8[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI8[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_UINT16)  // INT8 UINT16 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI8[ii]),
                                    (double)(data.image[ID2].array.UI16[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI8[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI16[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_UINT32)  // INT8 UINT32 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI8[ii]),
                                    (double)(data.image[ID2].array.UI32[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI8[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI32[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_UINT64)  // INT8 UINT64 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI8[ii]),
                                    (double)(data.image[ID2].array.UI64[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI8[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI64[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT8)  // INT8 INT8 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI8[ii]),
                                    (double)(data.image[ID2].array.SI8[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI8[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI8[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT16)  // INT8 INT16 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI8[ii]),
                                    (double)(data.image[ID2].array.SI16[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI8[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI16[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT32)  // INT8 INT32 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI8[ii]),
                                    (double)(data.image[ID2].array.SI32[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI8[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI32[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT64)  // INT8 INT64 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI8[ii]),
                                    (double)(data.image[ID2].array.SI64[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI8[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI64[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_FLOAT)  // INT8 FLOAT -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI8[ii]),
                                    (double)(data.image[ID2].array.F[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI8[kk * xysize + ii]),
                                (double)(data.image[ID2].array.F[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_DOUBLE)  // INT8 DOUBLE -> DOUBLE
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.D[ii] =
                        pt2function((double)(data.image[ID1].array.SI8[ii]),
                                    data.image[ID2].array.D[ii]);
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.D[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI8[kk * xysize + ii]),
                                data.image[ID2].array.D[ii]);
                    }
        }
    }

    // ID1 datatype  INT16

    if(datatype1 == _DATATYPE_INT16)
    {
        if(datatype2 == _DATATYPE_UINT8)  // INT16 UINT8 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI16[ii]),
                                    (double)(data.image[ID2].array.UI8[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI16[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI8[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_UINT16)  // INT16 UINT16 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI16[ii]),
                                    (double)(data.image[ID2].array.UI16[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI16[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI16[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_UINT32)  // INT16 UINT32 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI16[ii]),
                                    (double)(data.image[ID2].array.UI32[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI16[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI32[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_UINT64)  // INT16 UINT64 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI16[ii]),
                                    (double)(data.image[ID2].array.UI64[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI16[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI64[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT8)  // INT16 INT8 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI16[ii]),
                                    (double)(data.image[ID2].array.SI8[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI16[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI8[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT16)  // INT16 INT16 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI16[ii]),
                                    (double)(data.image[ID2].array.SI16[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI16[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI16[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT32)  // INT16 INT32 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI16[ii]),
                                    (double)(data.image[ID2].array.SI32[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI16[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI32[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT64)  // INT16 INT64 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI16[ii]),
                                    (double)(data.image[ID2].array.SI64[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI16[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI64[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_FLOAT)  // INT16 FLOAT -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI16[ii]),
                                    (double)(data.image[ID2].array.F[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI16[kk * xysize + ii]),
                                (double)(data.image[ID2].array.F[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_DOUBLE)  // INT16 DOUBLE -> DOUBLE
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.D[ii] =
                        pt2function((double)(data.image[ID1].array.SI16[ii]),
                                    data.image[ID2].array.D[ii]);
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.D[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI16[kk * xysize + ii]),
                                data.image[ID2].array.D[ii]);
                    }
        }
    }

    // ID1 datatype  INT32

    if(datatype1 == _DATATYPE_INT32)
    {
        if(datatype2 == _DATATYPE_UINT8)  // INT32 UINT8 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI32[ii]),
                                    (double)(data.image[ID2].array.UI8[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI32[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI8[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_UINT16)  // INT32 UINT16 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI32[ii]),
                                    (double)(data.image[ID2].array.UI16[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI32[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI16[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_UINT32)  // INT32 UINT32 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI32[ii]),
                                    (double)(data.image[ID2].array.UI32[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI32[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI32[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_UINT64)  // INT32 UINT64 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI32[ii]),
                                    (double)(data.image[ID2].array.UI64[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI32[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI64[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT8)  // INT32 INT8 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI32[ii]),
                                    (double)(data.image[ID2].array.SI8[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI32[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI8[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT16)  // INT32 INT16 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI32[ii]),
                                    (double)(data.image[ID2].array.SI16[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI32[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI16[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT32)  // INT32 INT32 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI32[ii]),
                                    (double)(data.image[ID2].array.SI32[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI32[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI32[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT64)  // INT32 INT64 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI32[ii]),
                                    (double)(data.image[ID2].array.SI64[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI32[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI64[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_FLOAT)  // INT32 FLOAT -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI32[ii]),
                                    (double)(data.image[ID2].array.F[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI32[kk * xysize + ii]),
                                (double)(data.image[ID2].array.F[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_DOUBLE)  // INT32 DOUBLE -> DOUBLE
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.D[ii] =
                        pt2function((double)(data.image[ID1].array.SI32[ii]),
                                    data.image[ID2].array.D[ii]);
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.D[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI32[kk * xysize + ii]),
                                data.image[ID2].array.D[ii]);
                    }
        }
    }

    // ID1 datatype  INT64
    if(datatype1 == _DATATYPE_INT64)
    {
        if(datatype2 == _DATATYPE_UINT8)  // INT64 UINT8 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI64[ii]),
                                    (double)(data.image[ID2].array.UI8[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI64[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI8[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_UINT16)  // INT64 UINT16 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI64[ii]),
                                    (double)(data.image[ID2].array.UI16[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI64[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI16[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_UINT32)  // INT64 UINT32 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI64[ii]),
                                    (double)(data.image[ID2].array.UI32[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI64[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI32[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_UINT64)  // INT64 UINT64 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI64[ii]),
                                    (double)(data.image[ID2].array.UI64[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI64[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI64[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT8)  // INT64 INT8 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI64[ii]),
                                    (double)(data.image[ID2].array.SI8[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI64[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI8[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT16)  // INT64 INT16 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI64[ii]),
                                    (double)(data.image[ID2].array.SI16[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI64[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI16[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT32)  // INT64 INT32 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI64[ii]),
                                    (double)(data.image[ID2].array.SI32[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI64[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI32[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT64)  // INT64 INT64 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI64[ii]),
                                    (double)(data.image[ID2].array.SI64[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI64[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI64[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_FLOAT)  // INT64 FLOAT -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.SI64[ii]),
                                    (double)(data.image[ID2].array.F[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI64[kk * xysize + ii]),
                                (double)(data.image[ID2].array.F[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_DOUBLE)  // INT64 DOUBLE -> DOUBLE
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.D[ii] =
                        pt2function((double)(data.image[ID1].array.SI64[ii]),
                                    data.image[ID2].array.D[ii]);
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.D[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.SI64[kk * xysize + ii]),
                                data.image[ID2].array.D[ii]);
                    }
        }
    }

    // ID1 datatype  FLOAT
    if(datatype1 == _DATATYPE_FLOAT)
    {
        if(datatype2 == _DATATYPE_UINT8)  // FLOAT UINT8 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.F[ii]),
                                    (double)(data.image[ID2].array.UI8[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.F[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI8[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_UINT16)  // FLOAT UINT16 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.F[ii]),
                                    (double)(data.image[ID2].array.UI16[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.F[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI16[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_UINT32)  // FLOAT UINT32 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.F[ii]),
                                    (double)(data.image[ID2].array.UI32[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.F[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI32[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_UINT64)  // FLOAT UINT64 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.F[ii]),
                                    (double)(data.image[ID2].array.UI64[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.F[kk * xysize + ii]),
                                (double)(data.image[ID2].array.UI64[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT8)  // FLOAT INT8 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.F[ii]),
                                    (double)(data.image[ID2].array.SI8[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.F[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI8[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT16)  // FLOAT INT16 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.F[ii]),
                                    (double)(data.image[ID2].array.SI16[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.F[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI16[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT32)  // FLOAT INT32 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.F[ii]),
                                    (double)(data.image[ID2].array.SI32[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.F[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI32[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT64)  // FLOAT INT64 -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.F[ii]),
                                    (double)(data.image[ID2].array.SI64[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.F[kk * xysize + ii]),
                                (double)(data.image[ID2].array.SI64[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_FLOAT)  // FLOAT FLOAT -> FLOAT
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.F[ii]),
                                    (double)(data.image[ID2].array.F[ii]));
                }

            if(op3D2Dto3D == 1)
            {
                //# ifdef _OPENMP
                //                #pragma omp for
                //# endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.F[kk * xysize + ii]),
                                (double)(data.image[ID2].array.F[ii]));
                    }
            }
        }

        if(datatype2 == _DATATYPE_DOUBLE)  // FLOAT DOUBLE -> DOUBLE
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.D[ii] =
                        pt2function((double)(data.image[ID1].array.F[ii]),
                                    data.image[ID2].array.D[ii]);
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.D[kk * xysize + ii] =
                            pt2function(
                                (double)(data.image[ID1]
                                         .array.F[kk * xysize + ii]),
                                data.image[ID2].array.D[ii]);
                    }
        }
    }

    // ID1 datatype  DOUBLE

    if(datatype1 == _DATATYPE_DOUBLE)
    {
        if(datatype2 == _DATATYPE_UINT8)  // DOUBLE UINT8 -> DOUBLE
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function(data.image[ID1].array.D[ii],
                                    (double)(data.image[ID2].array.UI8[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                data.image[ID1].array.D[kk * xysize + ii],
                                (double)(data.image[ID2].array.UI8[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_UINT16)  // DOUBLE UINT16 -> DOUBLE
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function(data.image[ID1].array.D[ii],
                                    (double)(data.image[ID2].array.UI16[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                data.image[ID1].array.D[kk * xysize + ii],
                                (double)(data.image[ID2].array.UI16[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_UINT32)  // DOUBLE UINT32 -> DOUBLE
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function(data.image[ID1].array.D[ii],
                                    (double)(data.image[ID2].array.UI32[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                data.image[ID1].array.D[kk * xysize + ii],
                                (double)(data.image[ID2].array.UI32[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_UINT64)  // DOUBLE UINT64 -> DOUBLE
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function(data.image[ID1].array.D[ii],
                                    (double)(data.image[ID2].array.UI64[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                data.image[ID1].array.D[kk * xysize + ii],
                                (double)(data.image[ID2].array.UI64[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT8)  // DOUBLE INT8 -> DOUBLE
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function(data.image[ID1].array.D[ii],
                                    (double)(data.image[ID2].array.SI8[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                data.image[ID1].array.D[kk * xysize + ii],
                                (double)(data.image[ID2].array.SI8[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT16)  // DOUBLE INT16 -> DOUBLE
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function(data.image[ID1].array.D[ii],
                                    (double)(data.image[ID2].array.SI16[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                data.image[ID1].array.D[kk * xysize + ii],
                                (double)(data.image[ID2].array.SI16[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT32)  // DOUBLE INT32 -> DOUBLE
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function(data.image[ID1].array.D[ii],
                                    (double)(data.image[ID2].array.SI32[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                data.image[ID1].array.D[kk * xysize + ii],
                                (double)(data.image[ID2].array.SI32[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_INT64)  // DOUBLE INT64 -> DOUBLE
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function(data.image[ID1].array.D[ii],
                                    (double)(data.image[ID2].array.SI64[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                data.image[ID1].array.D[kk * xysize + ii],
                                (double)(data.image[ID2].array.SI64[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_FLOAT)  // DOUBLE FLOAT -> DOUBLE
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.F[ii] =
                        pt2function(data.image[ID1].array.D[ii],
                                    (double)(data.image[ID2].array.F[ii]));
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[kk * xysize + ii] =
                            pt2function(
                                data.image[ID1].array.D[kk * xysize + ii],
                                (double)(data.image[ID2].array.F[ii]));
                    }
        }

        if(datatype2 == _DATATYPE_DOUBLE)  // DOUBLE DOUBLE -> DOUBLE
        {
            if(op3D2Dto3D == 0)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[IDout].array.D[ii] =
                        pt2function(data.image[ID1].array.D[ii],
                                    data.image[ID2].array.D[ii]);
                }

            if(op3D2Dto3D == 1)
#ifdef _OPENMP
                #pragma omp for
#endif
                for(kk = 0; kk < naxes[2]; kk++)
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.D[kk * xysize + ii] =
                            pt2function(
                                data.image[ID1].array.D[kk * xysize + ii],
                                data.image[ID2].array.D[ii]);
                    }
        }
    }

    //# ifdef _OPENMP
    //    }
    //# endif

    free(naxes);
    free(naxes2);

    return RETURN_SUCCESS;
}

errno_t arith_image_function_2_1_inplace_byID(
    imageID ID1, imageID ID2, double (*pt2function)(double, double))
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
#endif

        // FLOAT
        if(datatype1 == _DATATYPE_FLOAT)
        {
            if(datatype2 == _DATATYPE_UINT8)  // FLOAT <- UINT8
            {
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[ID1].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.F[ii]),
                                    (double)(data.image[ID2].array.UI8[ii]));
                }
            }

            if(datatype2 == _DATATYPE_UINT16)  // FLOAT <- UINT16
            {
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[ID1].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.F[ii]),
                                    (double)(data.image[ID2].array.UI16[ii]));
                }
            }

            if(datatype2 == _DATATYPE_UINT32)  // FLOAT <- UINT32
            {
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[ID1].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.F[ii]),
                                    (double)(data.image[ID2].array.UI32[ii]));
                }
            }

            if(datatype2 == _DATATYPE_UINT64)  // FLOAT <- UINT64
            {
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[ID1].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.F[ii]),
                                    (double)(data.image[ID2].array.UI64[ii]));
                }
            }

            if(datatype2 == _DATATYPE_INT8)  // FLOAT <- INT8
            {
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[ID1].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.F[ii]),
                                    (double)(data.image[ID2].array.SI8[ii]));
                }
            }

            if(datatype2 == _DATATYPE_INT16)  // FLOAT <- INT16
            {
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[ID1].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.F[ii]),
                                    (double)(data.image[ID2].array.SI16[ii]));
                }
            }

            if(datatype2 == _DATATYPE_INT32)  // FLOAT <- INT32
            {
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[ID1].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.F[ii]),
                                    (double)(data.image[ID2].array.SI32[ii]));
                }
            }

            if(datatype2 == _DATATYPE_INT64)  // FLOAT <- INT64
            {
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[ID1].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.F[ii]),
                                    (double)(data.image[ID2].array.SI64[ii]));
                }
            }

            if(datatype2 == _DATATYPE_FLOAT)  // FLOAT <- FLOAT
            {
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[ID1].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.F[ii]),
                                    (double)(data.image[ID2].array.F[ii]));
                }
            }

            if(datatype2 == _DATATYPE_DOUBLE)  // FLOAT <- DOUBLE
            {
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[ID1].array.F[ii] =
                        pt2function((double)(data.image[ID1].array.F[ii]),
                                    data.image[ID2].array.D[ii]);
                }
            }
        }

        // DOUBLE
        if(datatype1 == _DATATYPE_DOUBLE)
        {
            if(datatype2 == _DATATYPE_UINT8)  // DOUBLE <- UINT8
            {
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[ID1].array.D[ii] =
                        pt2function(data.image[ID1].array.D[ii],
                                    (double)(data.image[ID2].array.UI8[ii]));
                }
            }

            if(datatype2 == _DATATYPE_UINT16)  // DOUBLE <- UINT16
            {
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[ID1].array.D[ii] =
                        pt2function(data.image[ID1].array.D[ii],
                                    (double)(data.image[ID2].array.UI16[ii]));
                }
            }

            if(datatype2 == _DATATYPE_UINT32)  // DOUBLE <- UINT32
            {
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[ID1].array.D[ii] =
                        pt2function(data.image[ID1].array.D[ii],
                                    (double)(data.image[ID2].array.UI32[ii]));
                }
            }

            if(datatype2 == _DATATYPE_UINT64)  // DOUBLE <- UINT64
            {
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[ID1].array.D[ii] =
                        pt2function(data.image[ID1].array.D[ii],
                                    (double)(data.image[ID2].array.UI64[ii]));
                }
            }

            if(datatype2 == _DATATYPE_INT8)  // DOUBLE <- INT8
            {
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[ID1].array.D[ii] =
                        pt2function(data.image[ID1].array.D[ii],
                                    (double)(data.image[ID2].array.SI8[ii]));
                }
            }

            if(datatype2 == _DATATYPE_INT16)  // DOUBLE <- INT16
            {
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[ID1].array.D[ii] =
                        pt2function(data.image[ID1].array.D[ii],
                                    (double)(data.image[ID2].array.SI16[ii]));
                }
            }

            if(datatype2 == _DATATYPE_INT32)  // DOUBLE <- INT32
            {
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[ID1].array.D[ii] =
                        pt2function(data.image[ID1].array.D[ii],
                                    (double)(data.image[ID2].array.SI32[ii]));
                }
            }

            if(datatype2 == _DATATYPE_INT64)  // DOUBLE <- INT64
            {
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[ID1].array.D[ii] =
                        pt2function(data.image[ID1].array.D[ii],
                                    (double)(data.image[ID2].array.SI64[ii]));
                }
            }

            if(datatype2 == _DATATYPE_FLOAT)  // DOUBLE <- FLOAT
            {
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[ID1].array.D[ii] =
                        pt2function(data.image[ID1].array.D[ii],
                                    (double)(data.image[ID2].array.F[ii]));
                }
            }

            if(datatype2 == _DATATYPE_DOUBLE)  // DOUBLE <- DOUBLE
            {
#ifdef _OPENMP
                #pragma omp for
#endif
                for(ii = 0; ii < nelement; ii++)
                {
                    data.image[ID1].array.D[ii] =
                        pt2function(data.image[ID1].array.D[ii],
                                    data.image[ID2].array.D[ii]);
                }
            }
        }

        if((datatype1 == _DATATYPE_COMPLEX_FLOAT) &&
                (datatype2 == _DATATYPE_COMPLEX_FLOAT))
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID1].array.CF[ii].re =
                    pt2function((double)(data.image[ID1].array.CF[ii].re),
                                (double)(data.image[ID2].array.CF[ii].re));
                data.image[ID1].array.CF[ii].im =
                    pt2function((double)(data.image[ID1].array.CF[ii].im),
                                (double)(data.image[ID2].array.CF[ii].im));
            }
        }

        if((datatype1 == _DATATYPE_COMPLEX_DOUBLE) &&
                (datatype2 == _DATATYPE_COMPLEX_DOUBLE))
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID1].array.CD[ii].re =
                    pt2function((double)(data.image[ID1].array.CD[ii].re),
                                (double)(data.image[ID2].array.CD[ii].re));
                data.image[ID1].array.CD[ii].im =
                    pt2function((double)(data.image[ID1].array.CD[ii].im),
                                (double)(data.image[ID2].array.CD[ii].im));
            }
        }

#ifdef _OPENMP
    }
#endif

    data.image[ID1].md[0].write = 0;
    data.image[ID1].md[0].cnt0++;

    return EXIT_SUCCESS;
}

errno_t arith_image_function_2_1_inplace(const char *ID_name1,
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
    if(naxes == NULL)
    {
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    for(i = 0; i < naxis; i++)
    {
        naxes[i] = data.image[ID1].md[0].size[i];
    }

    create_image_ID(ID_out,
                    naxis,
                    naxes,
                    datatype1,
                    data.SHARED_DFT,
                    data.NBKEYWORD_DFT,
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
            data.image[IDout].array.CF[ii] =
                pt2function(data.image[ID1].array.CF[ii],
                            data.image[ID2].array.CF[ii]);
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
    if(naxes == NULL)
    {
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    for(i = 0; i < naxis; i++)
    {
        naxes[i] = data.image[ID1].md[0].size[i];
    }

    create_image_ID(ID_out,
                    naxis,
                    naxes,
                    datatype1,
                    data.SHARED_DFT,
                    data.NBKEYWORD_DFT,
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
            data.image[IDout].array.CD[ii] =
                pt2function(data.image[ID1].array.CD[ii],
                            data.image[ID2].array.CD[ii]);
        }
#ifdef _OPENMP
    }
#endif

    return RETURN_SUCCESS;
}

/* ------------------------------------------------------------------------- */
/* image, double  -> image                                                */
/* ------------------------------------------------------------------------- */

int arith_image_function_1f_1(const char *ID_name,
                              double      f1,
                              const char *ID_out,
                              double (*pt2function)(double, double))
{
    long      ID;
    long      IDout;
    long      ii;
    uint32_t *naxes = NULL;
    long      nelement;
    long      naxis;
    uint8_t   datatype, datatypeout;
    long      i;

    ID       = image_ID(ID_name);
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

    datatypeout = _DATATYPE_FLOAT;
    if(datatype == _DATATYPE_DOUBLE)
    {
        datatypeout = _DATATYPE_DOUBLE;
    }

    create_image_ID(ID_out,
                    naxis,
                    naxes,
                    datatypeout,
                    data.SHARED_DFT,
                    data.NBKEYWORD_DFT,
                    0,
                    &IDout);

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
                    pt2function((double)(data.image[ID].array.UI8[ii]), f1);
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
                    pt2function((double)(data.image[ID].array.UI16[ii]), f1);
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
                    pt2function((double)(data.image[ID].array.UI32[ii]), f1);
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
                    pt2function((double)(data.image[ID].array.UI64[ii]), f1);
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
                    pt2function((double)(data.image[ID].array.SI8[ii]), f1);
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
                    pt2function((double)(data.image[ID].array.SI16[ii]), f1);
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
                    pt2function((double)(data.image[ID].array.SI32[ii]), f1);
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
                    pt2function((double)(data.image[ID].array.SI64[ii]), f1);
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
                    pt2function((double)(data.image[ID].array.F[ii]), f1);
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
                    (double) pt2function((double)(data.image[ID].array.D[ii]),
                                         f1);
            }
        }
#ifdef _OPENMP
    }
#endif

    return EXIT_SUCCESS;
}

int arith_image_function_1f_1_inplace_byID(
    long ID, double f1, double (*pt2function)(double, double))
{
    long    ii;
    long    nelement;
    uint8_t datatype;

    datatype = data.image[ID].md[0].datatype;
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
                data.image[ID].array.F[ii] =
                    pt2function((double)(data.image[ID].array.UI8[ii]), f1);
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
                    pt2function((double)(data.image[ID].array.UI16[ii]), f1);
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
                    pt2function((double)(data.image[ID].array.UI32[ii]), f1);
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
                    pt2function((double)(data.image[ID].array.UI64[ii]), f1);
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
                    pt2function((double)(data.image[ID].array.SI8[ii]), f1);
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
                    pt2function((double)(data.image[ID].array.SI16[ii]), f1);
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
                    pt2function((double)(data.image[ID].array.SI32[ii]), f1);
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
                    pt2function((double)(data.image[ID].array.SI64[ii]), f1);
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
                    pt2function((double)(data.image[ID].array.F[ii]), f1);
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
                    (double) pt2function((double)(data.image[ID].array.D[ii]),
                                         f1);
            }
        }
#ifdef _OPENMP
    }
#endif

    return EXIT_SUCCESS;
}

int arith_image_function_1f_1_inplace(const char *ID_name,
                                      double      f1,
                                      double (*pt2function)(double, double))
{
    long ID;
    ID = image_ID(ID_name);

    return (arith_image_function_1f_1_inplace_byID(ID, f1, pt2function));
}

/* ------------------------------------------------------------------------- */
/* image, double, double -> image                                      */
/* ------------------------------------------------------------------------- */

int arith_image_function_1ff_1(const char *ID_name,
                               double      f1,
                               double      f2,
                               const char *ID_out,
                               double (*pt2function)(double, double, double))
{
    long      ID;
    long      IDout;
    long      ii;
    uint32_t *naxes = NULL;
    long      nelement;
    long      naxis;
    uint8_t   datatype;
    uint8_t   datatypeout;
    long      i;

    ID       = image_ID(ID_name);
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
    datatypeout = _DATATYPE_FLOAT;
    if(datatype == _DATATYPE_DOUBLE)
    {
        datatypeout = _DATATYPE_DOUBLE;
    }

    create_image_ID(ID_out,
                    naxis,
                    naxes,
                    datatypeout,
                    data.SHARED_DFT,
                    data.NBKEYWORD_DFT,
                    0,
                    &IDout);
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
                    pt2function((double)(data.image[ID].array.UI8[ii]),
                                f1,
                                f2);
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
                    pt2function((double)(data.image[ID].array.UI16[ii]),
                                f1,
                                f2);
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
                    pt2function((double)(data.image[ID].array.UI32[ii]),
                                f1,
                                f2);
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
                    pt2function((double)(data.image[ID].array.UI64[ii]),
                                f1,
                                f2);
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
                    pt2function((double)(data.image[ID].array.SI8[ii]),
                                f1,
                                f2);
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
                    pt2function((double)(data.image[ID].array.SI16[ii]),
                                f1,
                                f2);
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
                    pt2function((double)(data.image[ID].array.SI32[ii]),
                                f1,
                                f2);
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
                    pt2function((double)(data.image[ID].array.SI64[ii]),
                                f1,
                                f2);
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
                    pt2function((double)(data.image[ID].array.F[ii]), f1, f2);
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
                    pt2function((double)(data.image[ID].array.D[ii]), f1, f2);
            }
        }
#ifdef _OPENMP
    }
#endif

    return (0);
}

int arith_image_function_1ff_1_inplace(const char *ID_name,
                                       double      f1,
                                       double      f2,
                                       double (*pt2function)(double,
                                               double,
                                               double))
{
    long    ID;
    long    ii;
    long    nelement;
    uint8_t datatype;

    ID       = image_ID(ID_name);
    datatype = data.image[ID].md[0].datatype;
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
                data.image[ID].array.UI8[ii] = (uint8_t) pt2function(
                                                   (double)(data.image[ID].array.UI8[ii]),
                                                   f1,
                                                   f2);
            }
        }
        if(datatype == _DATATYPE_UINT16)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.UI16[ii] = (uint16_t) pt2function(
                                                    (double)(data.image[ID].array.UI16[ii]),
                                                    f1,
                                                    f2);
            }
        }
        if(datatype == _DATATYPE_UINT32)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.UI32[ii] = (uint32_t) pt2function(
                                                    (double)(data.image[ID].array.UI32[ii]),
                                                    f1,
                                                    f2);
            }
        }
        if(datatype == _DATATYPE_UINT64)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.UI64[ii] = (uint64_t) pt2function(
                                                    (double)(data.image[ID].array.UI64[ii]),
                                                    f1,
                                                    f2);
            }
        }

        if(datatype == _DATATYPE_INT8)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.SI8[ii] = (int8_t) pt2function(
                                                   (double)(data.image[ID].array.SI8[ii]),
                                                   f1,
                                                   f2);
            }
        }
        if(datatype == _DATATYPE_INT16)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.SI16[ii] = (int16_t) pt2function(
                                                    (double)(data.image[ID].array.SI16[ii]),
                                                    f1,
                                                    f2);
            }
        }
        if(datatype == _DATATYPE_INT32)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.SI32[ii] = (int32_t) pt2function(
                                                    (double)(data.image[ID].array.SI32[ii]),
                                                    f1,
                                                    f2);
            }
        }
        if(datatype == _DATATYPE_INT64)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.SI64[ii] = (int64_t) pt2function(
                                                    (double)(data.image[ID].array.SI64[ii]),
                                                    f1,
                                                    f2);
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
                    pt2function((double)(data.image[ID].array.F[ii]), f1, f2);
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
                    pt2function((double)(data.image[ID].array.D[ii]), f1, f2);
            }
        }
#ifdef _OPENMP
    }
#endif

    return (0);
}

int arith_image_function_1ff_1_inplace_byID(long   ID,
        double f1,
        double f2,
        double (*pt2function)(double,
                              double,
                              double))
{
    long    ii;
    long    nelement;
    uint8_t datatype;

    datatype = data.image[ID].md[0].datatype;
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
                data.image[ID].array.UI8[ii] = (uint8_t) pt2function(
                                                   (double)(data.image[ID].array.UI8[ii]),
                                                   f1,
                                                   f2);
            }
        }
        if(datatype == _DATATYPE_UINT16)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.UI16[ii] = (uint16_t) pt2function(
                                                    (double)(data.image[ID].array.UI16[ii]),
                                                    f1,
                                                    f2);
            }
        }
        if(datatype == _DATATYPE_UINT32)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.UI32[ii] = (uint32_t) pt2function(
                                                    (double)(data.image[ID].array.UI32[ii]),
                                                    f1,
                                                    f2);
            }
        }
        if(datatype == _DATATYPE_UINT64)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.UI64[ii] = (uint64_t) pt2function(
                                                    (double)(data.image[ID].array.UI64[ii]),
                                                    f1,
                                                    f2);
            }
        }

        if(datatype == _DATATYPE_INT32)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.SI8[ii] = (int8_t) pt2function(
                                                   (double)(data.image[ID].array.SI8[ii]),
                                                   f1,
                                                   f2);
            }
        }
        if(datatype == _DATATYPE_INT32)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.SI16[ii] = (int16_t) pt2function(
                                                    (double)(data.image[ID].array.SI16[ii]),
                                                    f1,
                                                    f2);
            }
        }
        if(datatype == _DATATYPE_INT32)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.SI32[ii] = (int32_t) pt2function(
                                                    (double)(data.image[ID].array.SI32[ii]),
                                                    f1,
                                                    f2);
            }
        }
        if(datatype == _DATATYPE_INT32)
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for(ii = 0; ii < nelement; ii++)
            {
                data.image[ID].array.SI64[ii] = (int64_t) pt2function(
                                                    (double)(data.image[ID].array.SI64[ii]),
                                                    f1,
                                                    f2);
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
                    pt2function((double)(data.image[ID].array.F[ii]), f1, f2);
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
                    pt2function((double)(data.image[ID].array.D[ii]), f1, f2);
            }
        }
#ifdef _OPENMP
    }
#endif

    return (0);
}
