/**
 * @file    info.c
 * @brief   Information about images
 *
 * Computes information about images
 *
 *
 *
 */

#define MODULE_SHORTNAME_DEFAULT "info"
#define MODULE_DESCRIPTION       "Image information and statistics"

#include "CommandLineInterface/CLIcore.h"
#include "cubeMatchMatrix.h"
#include "cubestats.h"
#include "image_stats.h"
#include "imagemon.h"
#include "improfile.h"

int infoscreen_wcol;
int infoscreen_wrow; // window size

INIT_MODULE_LIB(info)

static errno_t init_module_CLI()
{
    cubeMatchMatrix_addCLIcmd();
    cubestats_addCLIcmd();

    CLIADDCMD_info__imagemon();

    image_stats_addCLIcmd();
    improfile_addCLIcmd();

    return RETURN_SUCCESS;
}

/* number of pixels brighter than value */
/*long brighter(
    const char *ID_name,
    double      value
)
{
    imageID   ID;
    uint32_t  naxes[2];
    long      brighter, fainter;

    ID = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    brighter = 0;
    fainter = 0;
    for(unsigned long jj = 0; jj < naxes[1]; jj++)
        for(unsigned long ii = 0; ii < naxes[0]; ii++)
        {
            if(data.image[ID].array.F[jj * naxes[0] + ii] > value)
            {
                brighter++;
            }
            else
            {
                fainter++;
            }
        }
    printf("brighter %ld   fainter %ld\n", brighter, fainter);

    return(brighter);
}
*/

/*
errno_t img_nbpix_flux(
    const char *ID_name
)
{
    imageID   ID;
    uint32_t  naxes[2];
    double    value = 0;
    double   *array;
    uint64_t  nelements, i;

    ID = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];
    nelements = naxes[0] * naxes[1];

    array = (double *) malloc(naxes[1] * naxes[0] * sizeof(double));
    for(unsigned long jj = 0; jj < naxes[1]; jj++)
        for(unsigned long ii = 0; ii < naxes[0]; ii++)
        {
            array[jj * naxes[0] + ii] = data.image[ID].array.F[jj * naxes[0] +
ii];
        }

    quick_sort_double(array, nelements);

    for(i = 0; i < nelements; i++)
    {
        value += array[i];
        printf("%ld  %20.18e\n", i, value);
    }

    free(array);

    return RETURN_SUCCESS;
}
*/

/*
errno_t img_histoc_float(
    const char *ID_name,
    const char *fname
)
{
    FILE       *fp;
    imageID     ID;
    uint32_t    naxes[2];
    float       value = 0;
    float      *array;
    uint64_t    nelements;

    ID = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];
    nelements = naxes[0] * naxes[1];

    array = (float *) malloc(naxes[1] * naxes[0] * sizeof(float));
    for(unsigned long jj = 0; jj < naxes[1]; jj++)
        for(unsigned long ii = 0; ii < naxes[0]; ii++)
        {
            array[jj * naxes[0] + ii] = data.image[ID].array.F[jj * naxes[0] +
ii];
        }

    quick_sort_float(array, nelements);

    if((fp = fopen(fname, "w")) == NULL)
    {
        printf("ERROR: cannot open file \"%s\"\n", fname);
        exit(0);
    }
    value = 0.0;
    for(unsigned long ii = 0; ii < nelements; ii++)
    {
        value += array[ii];
        if(ii > 0.99 * nelements)
        {
            fprintf(fp, "%ld %g %g\n", nelements - ii, value, array[ii]);
        }
    }

    fclose(fp);
    free(array);

    return RETURN_SUCCESS;
}




errno_t img_histoc_double(
    const char *ID_name,
    const char *fname
)
{
    FILE       *fp;
    imageID     ID;
    uint32_t    naxes[2];
    double      value = 0;
    double     *array;
    uint64_t    nelements;

    ID = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];
    nelements = naxes[0] * naxes[1];

    array = (double *) malloc(naxes[1] * naxes[0] * sizeof(double));
    for(unsigned long jj = 0; jj < naxes[1]; jj++)
        for(unsigned long ii = 0; ii < naxes[0]; ii++)
        {
            array[jj * naxes[0] + ii] = data.image[ID].array.F[jj * naxes[0] +
ii];
        }

    quick_sort_double(array, nelements);

    if((fp = fopen(fname, "w")) == NULL)
    {
        printf("ERROR: cannot open file \"%s\"\n", fname);
        exit(0);
    }
    value = 0.0;
    for(unsigned long ii = 0; ii < nelements; ii++)
    {
        value += array[ii];
        if(ii > 0.99 * nelements)
        {
            fprintf(fp, "%ld %g %g\n", nelements - ii, value, array[ii]);
        }
    }

    fclose(fp);
    free(array);

    return RETURN_SUCCESS;
}
*/

/*
errno_t make_histogram(
    const char *ID_name,
    const char *ID_out_name,
    double      min,
    double      max,
    long        nbsteps
)
{
    imageID ID, ID_out;
    uint32_t naxes[2];
    long n;

    ID = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    create_2Dimage_ID(ID_out_name, nbsteps, 1);
    ID_out = image_ID(ID_out_name);
    for(unsigned long jj = 0; jj < naxes[1]; jj++)
        for(unsigned long ii = 0; ii < naxes[0]; ii++)
        {
            n = (long)((data.image[ID].array.F[jj * naxes[0] + ii] - min) /
                       (max - min) * nbsteps);
            if((n > 0) && (n < nbsteps))
            {
                data.image[ID_out].array.F[n] += 1;
            }
        }
    return RETURN_SUCCESS;
}
*/

/*
double ssquare(const char *ID_name)
{
    int ID;
    uint32_t naxes[2];
    double ssquare;

    ID = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    ssquare = 0;
    for(unsigned long jj = 0; jj < naxes[1]; jj++)
        for(unsigned long ii = 0; ii < naxes[0]; ii++)
        {
            ssquare = ssquare + data.image[ID].array.F[jj * naxes[0] + ii] *
                      data.image[ID].array.F[jj * naxes[0] + ii];
        }
    return(ssquare);
}




double rms_dev(const char *ID_name)
{
    int ID;
    uint32_t naxes[2];
    double ssquare, rms;
    double constant;

    ID = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    ssquare = 0;
    constant = arith_image_total(ID_name) / naxes[0] / naxes[1];
    for(unsigned long jj = 0; jj < naxes[1]; jj++)
        for(unsigned long ii = 0; ii < naxes[0]; ii++)
        {
            ssquare = ssquare + (data.image[ID].array.F[jj * naxes[0] + ii] -
constant) * (data.image[ID].array.F[jj * naxes[0] + ii] - constant);
        }
    rms = sqrt(ssquare / naxes[1] / naxes[0]);
    return(rms);
}
*/

/*
double img_min(const char *ID_name)
{
    int ID;
    double min;

    ID = image_ID(ID_name);

    min = data.image[ID].array.F[0];
    for(unsigned long ii = 0; ii < data.image[ID].md[0].nelement; ii++)
        if(min > data.image[ID].array.F[ii])
        {
            min = data.image[ID].array.F[ii];
        }

    return(min);
}



double img_max(const char *ID_name)
{
    int ID;
    double max;

    ID = image_ID(ID_name);

    max = data.image[ID].array.F[0];
    for(unsigned long ii = 0; ii < data.image[ID].md[0].nelement; ii++)
        if(max < data.image[ID].array.F[ii])
        {
            max = data.image[ID].array.F[ii];
        }

    return(max);
}
*/

/*

errno_t printpix(
    const char *ID_name,
    const char *filename
)
{
    imageID       ID;
//    uint64_t      nelements;
    long          nbaxis;
    uint32_t      naxes[3];
    FILE         *fp;

    long iistep = 1;
    long jjstep = 1;

    ID = variable_ID("_iistep");
    if(ID != -1)
    {
        iistep = (long)(0.1 + data.variable[ID].value.f);
        printf("iistep = %ld\n", iistep);
    }
    ID = variable_ID("_jjstep");
    if(ID != -1)
    {
        jjstep = (long)(0.1 + data.variable[ID].value.f);
        printf("jjstep = %ld\n", jjstep);
    }

    if((fp = fopen(filename, "w")) == NULL)
    {
        printf("ERROR: cannot open file \"%s\"\n", filename);
        exit(0);
    }

    ID = image_ID(ID_name);
    nbaxis = data.image[ID].md[0].naxis;
    if(nbaxis == 2)
    {
        naxes[0] = data.image[ID].md[0].size[0];
        naxes[1] = data.image[ID].md[0].size[1];
        //nelements = naxes[0] * naxes[1];
        for(unsigned long ii = 0; ii < naxes[0]; ii += iistep)
        {
            for(unsigned long jj = 0; jj < naxes[1]; jj += jjstep)
            {
                //  fprintf(fp,"%f ",data.image[ID].array.F[jj*naxes[0]+ii]);
                fprintf(fp, "%ld %ld %g\n", ii, jj, data.image[ID].array.F[jj *
naxes[0] + ii]);
            }
            fprintf(fp, "\n");
        }
    }
    if(nbaxis == 3)
    {
        naxes[0] = data.image[ID].md[0].size[0];
        naxes[1] = data.image[ID].md[0].size[1];
        naxes[2] = data.image[ID].md[0].size[2];
        //nelements = naxes[0] * naxes[1];
        for(unsigned long ii = 0; ii < naxes[0]; ii += iistep)
            for(unsigned long jj = 0; jj < naxes[1]; jj += jjstep)
                for(unsigned long kk = 0; kk < naxes[2]; kk++)
                {
                    fprintf(fp, "%ld %ld %ld %f\n", ii, jj, kk,
                            data.image[ID].array.F[kk * naxes[1]*naxes[0] + jj
* naxes[0] + ii]);
                }

    }
    fclose(fp);

    return RETURN_SUCCESS;
}
*/

/* uses the repartition function F of the normal distribution law */
/* F(0) = 0.5 */
/* F(-0.1 * sig) = 0.460172162723 */
/* F(-0.2 * sig) = 0.420740290562 */
/* F(-0.3 * sig) = 0.382088577811 */
/* F(-0.4 * sig) = 0.34457825839 */
/* F(-0.5 * sig) = 0.308537538726 */
/* F(-0.6 * sig) = 0.27425311775 */
/* F(-0.7 * sig) = 0.241963652223 */
/* F(-0.8 * sig) = 0.211855398584 */
/* F(-0.9 * sig) = 0.184060125347 */
/* F(-1.0 * sig) = 0.158655253931 */
/* F(-1.1 * sig) = 0.135666060946 */
/* F(-1.2 * sig) = 0.115069670222 */
/* F(-1.3 * sig) = 0.0968004845855 */
/*
double background_photon_noise(
    const char *ID_name
)
{
    imageID        ID;
    uint32_t       naxes[2];
    double         value1, value2, value3, value;
    double        *array;
    uint64_t       nelements;

    ID = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];
    nelements = naxes[0] * naxes[1];

    array = (double *) malloc(naxes[1] * naxes[0] * sizeof(double));
    for(unsigned long jj = 0; jj < naxes[1]; jj++)
        for(unsigned long ii = 0; ii < naxes[0]; ii++)
        {
            array[jj * naxes[0] + ii] = data.image[ID].array.F[jj * naxes[0] +
ii];
        }

    quick_sort_double(array, nelements);


    // calculation using F(-0.9*sig) and F(-1.3*sig)
    value1 = array[(long)(0.184060125347 * naxes[1] * naxes[0])] -
array[(long)( 0.0968004845855 * naxes[1] * naxes[0])]; value1 /= (1.3 - 0.9);
    printf("(-1.3 -0.9) %f\n", value1);

    // calculation using F(-0.6*sig) and F(-1.3*sig)
    value2 = array[(long)(0.27425311775 * naxes[1] * naxes[0])] - array[(long)(
                 0.0968004845855 * naxes[1] * naxes[0])];
    value2 /= (1.3 - 0.6);
    printf("(-1.3 -0.6) %f\n", value2);

    // calculation using F(-0.3*sig) and F(-1.3*sig)
    value3 = array[(long)(0.382088577811 * naxes[1] * naxes[0])] -
array[(long)( 0.0968004845855 * naxes[1] * naxes[0])]; value3 /= (1.3 - 0.3);
    printf("(-1.3 -0.3) %f\n", value3);

    value = value3;

    free(array);

    return(value);
}
*/

/*
errno_t test_structure_function(
    const char *ID_name,
    long        NBpoints,
    const char *ID_out
)
{
    imageID        ID, ID1, ID2;
    long           ii1, ii2, jj1, jj2, i, ii, jj;
    uint32_t       naxes[2];
    //uint64_t       nelements;
    double         v1, v2;

    ID = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    //nelements = naxes[0]*naxes[1];

    ID1 = create_2Dimage_ID("tmp1", naxes[0], naxes[1]);
    ID2 = create_2Dimage_ID("tmp2", naxes[0], naxes[1]);

    for(i = 0; i < NBpoints; i++)
    {
        ii1 = (long)(data.INVRANDMAX * rand() * naxes[0]);
        jj1 = (long)(data.INVRANDMAX * rand() * naxes[1]);
        ii2 = (long)(data.INVRANDMAX * rand() * naxes[0]);
        jj2 = (long)(data.INVRANDMAX * rand() * naxes[1]);
        v1 = data.image[ID].array.F[jj1 * naxes[0] + ii1];
        v2 = data.image[ID].array.F[jj2 * naxes[0] + ii2];
        ii = (ii1 - ii2);
        if(ii < 0)
        {
            ii = -ii;
        }
        jj = (jj1 - jj2);
        if(jj < 0)
        {
            jj = -jj;
        }
        data.image[ID1].array.F[jj * naxes[0] + ii] += (v1 - v2) * (v1 - v2);
        data.image[ID2].array.F[jj * naxes[0] + ii] += 1.0;
    }
    arith_image_div("tmp1", "tmp2", ID_out);


    return RETURN_SUCCESS;
}



imageID full_structure_function(
    const char *ID_name,
    long        NBpoints,
    const char *ID_out
)
{
    imageID   ID, ID1, ID2;
    long      ii1, ii2, jj1, jj2;
    uint32_t  naxes[2];
    double    v1, v2;
    long      i = 0;
    long      STEP1 = 2;
    long      STEP2 = 3;

    ID = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    ID1 = create_2Dimage_ID("tmp1", naxes[0], naxes[1]);
    ID2 = create_2Dimage_ID("tmp2", naxes[0], naxes[1]);


    for(ii1 = 0; ii1 < naxes[0]; ii1 += STEP1)
    {
        printf(".");
        for(jj1 = 0; jj1 < naxes[1]; jj1 += STEP1)
        {
            if(i < NBpoints)
            {
                i++;
                fflush(stdout);
                for(ii2 = 0; ii2 < naxes[0]; ii2 += STEP2)
                    for(jj2 = 0; jj2 < naxes[1]; jj2 += STEP2)
                        if((ii2 > ii1) && (jj2 > jj1))
                        {
                            v1 = data.image[ID].array.F[jj1 * naxes[0] + ii1];
                            v2 = data.image[ID].array.F[jj2 * naxes[0] + ii2];
                            data.image[ID1].array.F[(jj2 - jj1)*naxes[0] + ii2
- ii1] += (v1 - v2) * (v1 - v2); data.image[ID2].array.F[(jj2 - jj1)*naxes[0] +
ii2 - ii1] += 1.0;
                        }
            }
        }
    }
    printf("\n");

    ID = arith_image_div("tmp1", "tmp2", ID_out);

    return ID;
}

*/
