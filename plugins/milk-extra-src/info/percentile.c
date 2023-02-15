/** @file percentile.c
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"


float img_percentile_float(const char *ID_name, float p)
{
    imageID  ID;
    uint32_t naxes[2];
    float    value = 0;
    float   *array;
    uint64_t nelements;
    uint64_t n;

    ID        = image_ID(ID_name);
    naxes[0]  = data.image[ID].md[0].size[0];
    naxes[1]  = data.image[ID].md[0].size[1];
    nelements = naxes[0] * naxes[1];

    array = (float *) malloc(nelements * sizeof(float));
    if(array == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    for(unsigned long ii = 0; ii < nelements; ii++)
    {
        array[ii] = data.image[ID].array.F[ii];
    }

    quick_sort_float(array, nelements);

    n = (uint64_t)(p * naxes[1] * naxes[0]);
    if(n > 0)
    {
        if(n > (nelements - 1))
        {
            n = (nelements - 1);
        }
    }
    value = array[n];
    free(array);

    printf("percentile %f = %f (%ld)\n", p, value, n);

    return (value);
}

double img_percentile_double(const char *ID_name, double p)
{
    imageID  ID;
    uint32_t naxes[2];
    double   value = 0;
    double  *array;
    uint64_t nelements;
    uint64_t n;

    ID        = image_ID(ID_name);
    naxes[0]  = data.image[ID].md[0].size[0];
    naxes[1]  = data.image[ID].md[0].size[1];
    nelements = naxes[0] * naxes[1];

    array = (double *) malloc(nelements * sizeof(double));
    if(array == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    for(unsigned long ii = 0; ii < nelements; ii++)
    {
        array[ii] = data.image[ID].array.F[ii];
    }

    quick_sort_double(array, nelements);

    n = (uint64_t)(p * naxes[1] * naxes[0]);
    if(n > 0)
    {
        if(n > (nelements - 1))
        {
            n = (nelements - 1);
        }
    }
    value = array[n];
    free(array);

    return (value);
}

double img_percentile(const char *ID_name, double p)
{
    imageID ID;
    uint8_t datatype;
    double  value = 0.0;

    ID       = image_ID(ID_name);
    datatype = data.image[ID].md[0].datatype;

    if(datatype == _DATATYPE_FLOAT)
    {
        value = (double) img_percentile_float(ID_name, (float) p);
    }
    if(datatype == _DATATYPE_DOUBLE)
    {
        value = img_percentile_double(ID_name, p);
    }

    return value;
}
