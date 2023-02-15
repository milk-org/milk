/** @file fit1D.c
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"
#include "statistic/statistic.h"

int filter_fit1D(const char *__restrict fname, long NBpts)
{
    FILE  *fp;
    float *xarray;
    float *yarray;
    long   i;
    long   iter;
    long   NBiter = 10000000;
    float *CX;  //,CX,CX2,CX3,CX4,CX5;
    float *CXb; //,CXb,CX2b,CX3b,CX4b,CX5b;
    long   PolyOrder = 10;
    long   k;
    float  amp;
    float  x, value, bvalue, tmp;
    float  cnt, coeff;

    xarray = (float *) malloc(sizeof(float) * NBpts);
    if(xarray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    yarray = (float *) malloc(sizeof(float) * NBpts);
    if(yarray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    CX = (float *) malloc(sizeof(float) * PolyOrder);
    if(CX == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    CXb = (float *) malloc(sizeof(float) * PolyOrder);
    if(CXb == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    fp = fopen(fname, "r");
    for(i = 0; i < NBpts; i++)
    {
        if(fscanf(fp, "%f %f\n", &xarray[i], &yarray[i]) != 2)
        {
            printf("ERROR: fscanf, %s line %d\n", __FILE__, __LINE__);
            exit(0);
        }
        printf("%ld %f %.10f\n", i, xarray[i], yarray[i]);
    }
    fclose(fp);

    for(k = 0; k < PolyOrder; k++)
    {
        CX[k] = 0.0;
    }

    if(1 == 0)
    {
        // + side
    }
    else
    {
        // - side  RMS = 68 nm
        CX[0] = -1.39745e-07;
        CX[1] = 3.21897e-05;
        CX[2] = -0.00011522;
        CX[3] = 0.000149478;
        CX[4] = -6.17691e-05;
        CX[5] = -2.94572e-06;
    }
    for(k = 0; k < PolyOrder; k++)
    {
        CXb[k] = CX[k];
    }

    bvalue = 1000000.0;
    for(iter = 0; iter < NBiter; iter++)
    {
        amp = 1.0e-7;
        if(iter > 0)
            for(k = 0; k < PolyOrder; k++)
            {
                CX[k] = CXb[k] + amp * 2.0 * (ran1() - 0.5);
            }

        value = 0.0;
        cnt   = 0.0;
        for(i = 0; i < NBpts; i++)
        {
            x   = xarray[i];
            tmp = 0.0;
            for(k = 0; k < PolyOrder; k++)
            {
                tmp += CX[k] * pow(x, k);
            }
            coeff = pow(1.0 + 5.0 * exp(-10.0 * x * x), 2.0);
            value += coeff * (tmp - yarray[i]) * (tmp - yarray[i]);
            cnt += coeff;
        }
        value = sqrt(value / cnt);
        //      printf("value = %g\n");
        if(iter == 0)
        {
            bvalue = value;
        }
        else
        {
            if(value < bvalue)
            {
                for(k = 0; k < PolyOrder; k++)
                {
                    CXb[k] = CX[k];
                }
                bvalue = value;
                printf("BEST VALUE = %g\n", value);
                printf("f(r) = ");
                printf(" %g", CX[0]);
                for(k = 1; k < PolyOrder; k++)
                {
                    printf(" + r**%ld*%g", k, CX[k]);
                }
                printf("\n");

                for(k = 0; k < PolyOrder; k++)
                {
                    printf("CX[%ld] = %g\n", k, CX[k]);
                }
            }
        }
    }

    free(xarray);
    free(yarray);
    free(CX);
    free(CXb);

    return (0);
}
