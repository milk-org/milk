/** @file fit2Dcossin.c
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"

// fits a 2D image as a sum of cosines and sines
int filter_fit2Dcossin(const char *__restrict IDname, float radius)
{
    imageID ID;
    imageID IDres;
    imageID IDfit;
    long    size;
    long    NBfrequ1D = 25;
    long    NBfrequ;
    float  *coscoeff;
    float  *sincoeff;
    float   frequStep = 1.0;
    float   x, y;
    long    i, j, ii, jj, i1, j1;
    float   tmp, tmpc, tmpc1, tmpc2, tmps, tmps1, tmps2;
    long    iter;
    long    NBiter = 5000;
    float   error;
    long    errorcnt;
    float   gain = 0.0;
    float   Gain = 0.2;
    float   gtmpc, gtmps;
    float   rlim = 0.98;
    FILE   *fp;
    long    tmpl;
    long    step = 2;
    float   coeffc, coeffs;
    long    ii1;
    float  *xarray;
    float  *yarray;
    float  *rarray;

    NBfrequ = NBfrequ1D * (2 * NBfrequ1D - 1);

    coscoeff = (float *) malloc(sizeof(float) * NBfrequ);
    if(coscoeff == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    sincoeff = (float *) malloc(sizeof(float) * NBfrequ);
    if(sincoeff == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    for(i = 0; i < NBfrequ1D; i++)
        for(j = 0; j < 2 * NBfrequ1D - 1; j++)
        {
            coscoeff[j * NBfrequ1D + i] = 0.0;
            sincoeff[j * NBfrequ1D + i] = 0.0;
            //	printf("%ld %ld -> %g %g\n",i,(j-NBfrequ1D+1),coscoeff[j*NBfrequ1D+i],sincoeff[j*NBfrequ1D+i]);
        }

    if(1 == 0)
    {
        fp = fopen("fitcoeff.dat", "r");
        for(i = 0; i < NBfrequ1D; i++)
            for(j = 0; j < 2 * NBfrequ1D - 1; j++)
            {
                if(fscanf(fp,
                          "%ld %ld %ld %g %g\n",
                          &i,
                          &j,
                          &tmpl,
                          &coscoeff[j * NBfrequ1D + i],
                          &sincoeff[j * NBfrequ1D + i]) != 5)
                {
                    printf("ERROR: fscanf, %s line %d\n", __FILE__, __LINE__);
                    exit(0);
                }
            }
        fclose(fp);
        /*
        fp = fopen("fitcoeff1.dat","w");
        for(i=0;i<NBfrequ1D;i++)
        for(j=0;j<2*NBfrequ1D-1;j++)
          fprintf(fp,"%ld %ld %ld %.20g %.20g\n",i,j,j-NBfrequ1D+1,coscoeff[j*NBfrequ1D+i],sincoeff[j*NBfrequ1D+i]);
          fclose(fp);*/
    }
    //  exit(0);

    ID   = image_ID(IDname);
    size = data.image[ID].md[0].size[0];
    printf("SIZE = %ld\n", size);
    create_2Dimage_ID("residual", size, size, &IDres);
    create_2Dimage_ID("fitim", size, size, &IDfit);

    xarray = (float *) malloc(sizeof(float) * size * size);
    if(xarray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    yarray = (float *) malloc(sizeof(float) * size * size);
    if(yarray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    rarray = (float *) malloc(sizeof(float) * size * size);
    if(rarray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    for(ii = 0; ii < size; ii += step)
        for(jj = 0; jj < size; jj += step)
        {
            ii1         = jj * size + ii;
            x           = 1.0 * (ii - size / 2) / radius;
            y           = 1.0 * (jj - size / 2) / radius;
            xarray[ii1] = x;
            yarray[ii1] = y;
            rarray[ii1] = sqrt(x * x + y * y);
        }

    for(ii = 0; ii < size * size; ii++)
    {
        data.image[IDres].array.F[ii] = data.image[ID].array.F[ii];
        data.image[IDfit].array.F[ii] = 0.0;
    }
    for(iter = 0; iter < NBiter; iter++)
    {
        if((iter == 0) || (iter == NBiter - 1))
        {
            gain = 0.0;
        }
        else
        {
            gain = Gain;
        }
        // initialize IDfit
        for(ii = 0; ii < size; ii += step)
            for(jj = 0; jj < size; jj += step)
            {
                data.image[IDfit].array.F[jj * size + ii] = 0.0;
            }
        for(i1 = 0; i1 < NBfrequ1D; i1++)
            for(j1 = 0; j1 < 2 * NBfrequ1D - 1; j1++)
            {
                coeffc = coscoeff[j1 * NBfrequ1D + i1];
                coeffs = sincoeff[j1 * NBfrequ1D + i1];
                for(ii = 0; ii < size; ii += step)
                    for(jj = 0; jj < size; jj += step)
                    {
                        ii1 = jj * size + ii;
                        if(rarray[ii1] < rlim)
                        {
                            tmp = frequStep *
                                  (xarray[ii1] * i1 +
                                   yarray[ii1] * (j1 - NBfrequ1D + 1));
                            tmpc = cos(tmp);
                            tmps = sin(tmp);
                            data.image[IDfit].array.F[ii1] += coeffc * tmpc;
                            data.image[IDfit].array.F[ii1] += coeffs * tmps;
                        }
                    }
            }

        for(i = 0; i < NBfrequ1D; i++)
            for(j = 0; j < 2 * NBfrequ1D - 1; j++)
            {
                tmpc1 = 0.0;
                tmpc2 = 0.0;
                tmps1 = 0.0;
                tmps2 = 0.0;
                for(ii = 0; ii < size; ii += step)
                    for(jj = 0; jj < size; jj += step)
                    {
                        ii1 = jj * size + ii;
                        if(rarray[ii1] < rlim)
                        {
                            tmp =
                                frequStep * (xarray[ii1] * i +
                                             yarray[ii1] * (j - NBfrequ1D + 1));
                            tmpc = cos(tmp);
                            tmps = sin(tmp);

                            tmpc1 += tmpc * tmpc;
                            tmpc2 += data.image[IDres].array.F[ii1] * tmpc;

                            tmps1 += tmps * tmps;
                            tmps2 += data.image[IDres].array.F[ii1] * tmps;
                        }
                    }
                if(tmpc1 > 1e-8)
                {
                    tmpc = tmpc2 / tmpc1;
                }
                else
                {
                    tmpc = 0.0;
                }

                if(tmps1 > 1e-8)
                {
                    tmps = tmps2 / tmps1;
                }
                else
                {
                    tmps = 0.0;
                }

                //  printf("%ld (%ld,%ld) : %g %g\n",iter,i,(j-NBfrequ1D+1),tmpc,tmps);
                coscoeff[j * NBfrequ1D + i] += gain * tmpc;
                sincoeff[j * NBfrequ1D + i] += gain * tmps;
                gtmpc = gain * tmpc;
                gtmps = gain * tmps;

                for(ii = 0; ii < size; ii += step)
                    for(jj = 0; jj < size; jj += step)
                    {
                        ii1 = jj * size + ii;
                        if(rarray[ii1] < 1.0)
                        {
                            tmp =
                                frequStep * (xarray[ii1] * i +
                                             yarray[ii1] * (j - NBfrequ1D + 1));
                            tmpc = cos(tmp);
                            tmps = sin(tmp);
                            data.image[IDfit].array.F[ii1] += gtmpc * tmpc;
                            data.image[IDfit].array.F[ii1] += gtmps * tmps;
                        }
                    }

                error    = 0.0;
                errorcnt = 0;
                for(ii = 0; ii < size; ii += step)
                    for(jj = 0; jj < size; jj += step)
                    {
                        ii1 = jj * size + ii;
                        if(rarray[ii1] < 1.0)
                        {
                            data.image[IDres].array.F[ii1] =
                                data.image[ID].array.F[ii1] -
                                data.image[IDfit].array.F[ii1];
                            if(rarray[ii1] < rlim)
                            {
                                error += data.image[IDres].array.F[ii1] *
                                         data.image[IDres].array.F[ii1];
                                errorcnt++;
                            }
                        }
                    }
            }
        printf("iter %ld / %ld   error = %g\n",
               iter,
               NBiter,
               sqrt(error / errorcnt));
        save_fl_fits("fitim", "fitim");
        save_fl_fits("residual", "residual");

        fp = fopen("fitcoeff.dat", "w");
        for(i = 0; i < NBfrequ1D; i++)
            for(j = 0; j < 2 * NBfrequ1D - 1; j++)
            {
                fprintf(fp,
                        "%ld %ld %ld %.20g %.20g\n",
                        i,
                        j,
                        j - NBfrequ1D + 1,
                        coscoeff[j * NBfrequ1D + i],
                        sincoeff[j * NBfrequ1D + i]);
            }
        fclose(fp);
    }
    free(coscoeff);
    free(sincoeff);

    free(xarray);
    free(yarray);
    free(rarray);

    return (0);
}
