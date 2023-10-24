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
                    NB_KEYWNODE_MAX,
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
                    NB_KEYWNODE_MAX,
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
                    NB_KEYWNODE_MAX,
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
                    NB_KEYWNODE_MAX,
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

errno_t arith_img_function_2_1(
    IMGID inimg1,
    IMGID inimg2,
    IMGID *outimg,
    double (*pt2function)(double, double)
)
{
    DEBUG_TRACE_FSTART();

    resolveIMGID(&inimg1, ERRMODE_ABORT);
    resolveIMGID(&inimg2, ERRMODE_ABORT);


    resolveIMGID(outimg, ERRMODE_NULL);
    if( outimg->ID == -1)
    {
        copyIMGID(&inimg1, outimg);
    }

    // toggles to 1 if image sizes are incompatible
    int axiserror = 0;

    // output naxis is max of inputs
    outimg->naxis = inimg1.md->naxis;
    if ( inimg2.md->naxis > inimg1.md->naxis )
    {
        outimg->naxis = inimg2.md->naxis;
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
        printf("Checking axis %u\n", axis);

        in1expand[axis] = 1;
        in2expand[axis] = 1;

        // convention: size=1 if > naxis

        uint32_t size1;
        if(axis < inimg1.md->naxis)
        {
            size1 = inimg1.md->size[axis];
        }
        else
        {
            size1 = 1;
        }
        nbpix1 *= size1;

        uint32_t size2;
        if(axis < inimg2.md->naxis)
        {
            size2 = inimg2.md->size[axis];
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
                printf("Expanding im1 axis %d to %u\n", axis, size2);
                outimg->size[axis] = size2;

            }
            else if ( size2 == 1)
            {
                in2expand[axis] = 0;
                printf("Expanding im2 axis %d to %u\n", axis, size1);
                outimg->size[axis] = size1;
            }
            else
            {
                axiserror = 1;
                PRINT_ERROR("axis %d size %u and %u incompatible", axis, size1, size2);
                abort();
            }
        }

        printf("    size   %5u  %5u   %5u\n", size1, size2, outimg->size[axis]);

        nbpix *= outimg->size[axis];
    }
    for ( uint8_t axis = outimg->naxis; axis<3; axis++)
    {
        // convention
        outimg->size[axis] = 1;
        in1expand[axis] = 1;
        in2expand[axis] = 1;
    }


    outimg->datatype = _DATATYPE_FLOAT; // default
    // other cases

    // DOUBLE * -> DOUBLE
    if(inimg1.md->datatype == _DATATYPE_DOUBLE)
    {
        outimg->datatype = _DATATYPE_DOUBLE;
    }

    // * DOUBLE -> DOUBLE
    if(inimg2.md->datatype == _DATATYPE_DOUBLE)
    {
        outimg->datatype = _DATATYPE_DOUBLE;
    }

    createimagefromIMGID(outimg);


    // build mapping between output and input pixel indices

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

                inpix1[outpixi] =  kk1 * inimg1.md->size[1] * inimg1.md->size[0];
                inpix1[outpixi] += jj1 * inimg1.md->size[0];
                inpix1[outpixi] += ii1;

                inpix2[outpixi] =  kk2 * inimg2.md->size[1] * inimg2.md->size[0];
                inpix2[outpixi] += jj2 * inimg2.md->size[0];
                inpix2[outpixi] += ii2;
            }
        }
    }



    // TYPE CONVERSION TO DOUBLES

    double * ptr1array;
    int ptr1allocate = 0;
    if ( inimg1.md->datatype == _DATATYPE_DOUBLE )
    {
        ptr1array = inimg1.im->array.D;
    }
    else
    {
        ptr1allocate = 1;
        ptr1array = (double *) malloc(sizeof(double) * nbpix1);

        if(inimg1.md->datatype == _DATATYPE_UINT8) {
            for(uint64_t ii = 0; ii < nbpix1; ii++)
                ptr1array[ii] = (double) (inimg1.im->array.UI8[inpix1[ii]]);
        }
        if(inimg1.md->datatype == _DATATYPE_INT8) {
            for(uint64_t ii = 0; ii < nbpix1; ii++)
                ptr1array[ii] = (double) (inimg1.im->array.SI8[inpix1[ii]]);
        }

        if(inimg1.md->datatype == _DATATYPE_UINT16) {
            for(uint64_t ii = 0; ii < nbpix1; ii++)
                ptr1array[ii] = (double) (inimg1.im->array.UI16[inpix1[ii]]);
        }
        if(inimg1.md->datatype == _DATATYPE_INT16) {
            for(uint64_t ii = 0; ii < nbpix1; ii++)
                ptr1array[ii] = (double) (inimg1.im->array.SI16[inpix1[ii]]);
        }

        if(inimg1.md->datatype == _DATATYPE_UINT32) {
            for(uint64_t ii = 0; ii < nbpix1; ii++)
                ptr1array[ii] = (double) (inimg1.im->array.UI32[inpix1[ii]]);
        }
        if(inimg1.md->datatype == _DATATYPE_INT32) {
            for(uint64_t ii = 0; ii < nbpix1; ii++)
                ptr1array[ii] = (double) (inimg1.im->array.SI32[inpix1[ii]]);
        }

        if(inimg1.md->datatype == _DATATYPE_UINT64) {
            for(uint64_t ii = 0; ii < nbpix1; ii++)
                ptr1array[ii] = (double) (inimg1.im->array.UI64[inpix1[ii]]);
        }
        if(inimg1.md->datatype == _DATATYPE_INT64) {
            for(uint64_t ii = 0; ii < nbpix1; ii++)
                ptr1array[ii] = (double) (inimg1.im->array.SI64[inpix1[ii]]);
        }

        if(inimg1.md->datatype == _DATATYPE_FLOAT) {
            for(uint64_t ii = 0; ii < nbpix1; ii++)
                ptr1array[ii] = (double) (inimg1.im->array.F[inpix1[ii]]);
        }
    }






    double * ptr2array;
    int ptr2allocate = 0;
    if ( inimg2.md->datatype == _DATATYPE_DOUBLE )
    {
        ptr2array = inimg2.im->array.D;
    }
    else
    {
        ptr2allocate = 1;
        ptr2array = (double *) malloc(sizeof(double) * nbpix2);

        if(inimg2.md->datatype == _DATATYPE_UINT8) {
            for(uint64_t ii = 0; ii < nbpix2; ii++)
                ptr2array[ii] = (double) (inimg2.im->array.UI8[inpix1[ii]]);
        }
        if(inimg2.md->datatype == _DATATYPE_INT8) {
            for(uint64_t ii = 0; ii < nbpix2; ii++)
                ptr2array[ii] = (double) (inimg2.im->array.SI8[inpix1[ii]]);
        }

        if(inimg2.md->datatype == _DATATYPE_UINT16) {
            for(uint64_t ii = 0; ii < nbpix2; ii++)
                ptr2array[ii] = (double) (inimg2.im->array.UI16[inpix1[ii]]);
        }
        if(inimg2.md->datatype == _DATATYPE_INT16) {
            for(uint64_t ii = 0; ii < nbpix2; ii++)
                ptr2array[ii] = (double) (inimg2.im->array.SI16[inpix1[ii]]);
        }

        if(inimg2.md->datatype == _DATATYPE_UINT32) {
            for(uint64_t ii = 0; ii < nbpix2; ii++)
                ptr2array[ii] = (double) (inimg2.im->array.UI32[inpix1[ii]]);
        }
        if(inimg2.md->datatype == _DATATYPE_INT32) {
            for(uint64_t ii = 0; ii < nbpix2; ii++)
                ptr2array[ii] = (double) (inimg2.im->array.SI32[inpix1[ii]]);
        }

        if(inimg2.md->datatype == _DATATYPE_UINT64) {
            for(uint64_t ii = 0; ii < nbpix2; ii++)
                ptr2array[ii] = (double) (inimg2.im->array.UI64[inpix1[ii]]);
        }
        if(inimg2.md->datatype == _DATATYPE_INT64) {
            for(uint64_t ii = 0; ii < nbpix2; ii++)
                ptr2array[ii] = (double) (inimg2.im->array.SI64[inpix1[ii]]);
        }

        if(inimg2.md->datatype == _DATATYPE_FLOAT) {
            for(uint64_t ii = 0; ii < nbpix2; ii++)
                ptr2array[ii] = (double) (inimg2.im->array.F[inpix1[ii]]);
        }
    }



    if ( outimg->datatype == _DATATYPE_FLOAT )
    {
        for(uint64_t ii = 0; ii < nbpix; ii++ )
        {
            outimg->im->array.F[ii] =
                pt2function( ptr1array[inpix1[ii]],
                             ptr2array[inpix2[ii]] );
        }
    }


    if ( outimg->datatype == _DATATYPE_DOUBLE )
    {
        for(uint64_t ii = 0; ii < nbpix; ii++ )
        {
            outimg->im->array.D[ii] =
                pt2function( ptr1array[inpix1[ii]],
                             ptr2array[inpix2[ii]] );
        }
    }


    if(ptr1allocate == 1)
    {
        free(ptr1array);
    }

    if(ptr2allocate == 1)
    {
        free(ptr2array);
    }


    free(inpix1);
    free(inpix2);


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
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
    printf("%s %d\n", __FILE__, __LINE__);
    fflush(stdout);


    IMGID inimg1 = mkIMGID_from_name(ID_name1);
    resolveIMGID(&inimg1, ERRMODE_ABORT);

    IMGID inimg2 = mkIMGID_from_name(ID_name2);
    resolveIMGID(&inimg2, ERRMODE_ABORT);

    IMGID outimg = mkIMGID_from_name(ID_out);

    printf("%s %d\n", __FILE__, __LINE__);
    fflush(stdout);

    arith_img_function_2_1(
        inimg1,
        inimg2,
        &outimg,
        pt2function
    );

    return RETURN_SUCCESS;
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
                    NB_KEYWNODE_MAX,
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
                    NB_KEYWNODE_MAX,
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
