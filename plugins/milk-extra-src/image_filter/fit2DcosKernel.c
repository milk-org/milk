/** @file fit2DcosKernel.c
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "fit2Dcossin.h"

errno_t filter_fit2DcosKernel(const char *__restrict IDname, float radius)
{
    DEBUG_TRACE_FSTART();

    imageID ID, ID1, ID2, ID3;
    long    size;
    long    NBgridpts1D = 20;
    long    NBgridpts;
    long    ii, jj, i, j;
    float  *x0array;
    float  *y0array;
    float  *Varraytmp;
    float  *Varray;
    float  *Varraycnt;
    float   x, y, r, x1, y1;
    float   xstep, ystep;
    float   value;
    long    cnt;
    float   error;
    long    NBiter = 10;
    long    iter;
    float   cosa, tmp, tmp1, tmp2;

    float CX1_1 = -3.52106e-05;
    float CX2_1 = 0.000104827;
    float CX3_1 = -0.000156806;
    float CX4_1 = 0.000106682;
    float CX5_1 = -2.61437e-05;

    float CX1_2 = 3.16388e-05;
    float CX2_2 = -0.000125175;
    float CX3_2 = 0.000203591;
    float CX4_2 = -0.000146805;
    float CX5_2 = 3.87538e-05;

    // filter_fit1D("prof1m.dat",199);
    // exit(0);

    filter_fit2Dcossin(IDname, radius);

    NBgridpts = NBgridpts1D * NBgridpts1D;

    x0array   = (float *) malloc(sizeof(float) * NBgridpts);
    y0array   = (float *) malloc(sizeof(float) * NBgridpts);
    Varray    = (float *) malloc(sizeof(float) * NBgridpts);
    Varraytmp = (float *) malloc(sizeof(float) * NBgridpts);
    Varraycnt = (float *) malloc(sizeof(float) * NBgridpts);

    for(i = 0; i < NBgridpts1D; i++)
        for(j = 0; j < NBgridpts1D; j++)
        {
            x0array[j * NBgridpts1D + i]   = -1.0 + 2.0 * i / (NBgridpts1D - 1);
            y0array[j * NBgridpts1D + i]   = -1.0 + 2.0 * j / (NBgridpts1D - 1);
            Varray[j * NBgridpts1D + i]    = 0.0;
            Varraycnt[j * NBgridpts1D + i] = 0.0;
            //	printf("%ld %ld  %f %f\n",i,j,x0array[j*NBgridpts1D+i],y0array[j*NBgridpts1D+i]);
        }
    xstep = x0array[0 * NBgridpts1D + 1] - x0array[0 * NBgridpts1D + 0];
    ystep = y0array[1 * NBgridpts1D + 0] - y0array[0 * NBgridpts1D + 0];

    ID   = image_ID(IDname);
    size = data.image[ID].md[0].size[0];

    FUNC_CHECK_RETURN(create_2Dimage_ID("testim", size, size, &ID1));
    FUNC_CHECK_RETURN(create_2Dimage_ID("fitim", size, size, &ID2));
    FUNC_CHECK_RETURN(create_2Dimage_ID("residual", size, size, &ID3));

    for(ii = 0; ii < size; ii++)
        for(jj = 0; jj < size; jj++)
        {
            x    = (1.0 * ii - size / 2) / radius;
            y    = (1.0 * jj - size / 2) / radius;
            r    = sqrt(x * x + y * y);
            cosa = x / (r + 0.000001);
            //Cs = 3.34e-05;
            tmp1 = r * CX1_1 + r * r * CX2_1 + r * r * r * CX3_1 +
                   r * r * r * r * CX4_1 + r * r * r * r * r * CX5_1;
            tmp2 = r * CX1_2 + r * r * CX2_2 + r * r * r * CX3_2 +
                   r * r * r * r * CX4_2 + r * r * r * r * r * CX5_2;

            //	Cs = 3.3e-05;
            //	tmp1 = -r*Cs + r*r*0.000104827 - r*r*r*0.000156806 + r*r*r*r*0.000106682 - r*r*r*r*r*2.61437e-05;
            //	tmp2 = r*Cs - r*r*0.000125175 + r*r*r*0.000203591 - r*r*r*r*0.000146805 + r*r*r*r*r*3.87538e-05;

            //	tmp1 = -2.70e-5*exp(-7.0*pow((r-0.013),2.0))*r-1.05e-7;
            //	tmp2 = -2.70e-5*exp(-7.0*pow(((-r)-0.013),2.0))*(-r)-1.05e-7;

            if(r < 1.0)
            {
                tmp = tmp1 * (1.0 + cosa) / 2.0 + tmp2 * (1.0 - cosa) / 2.0;
            }
            else
            {
                tmp = 0.0;
            }
            tmp = 0.0;
            // for 15
            //tmp += 4.8e-6*exp(-280.0*(x-0.007)*(x-0.007))*(x-0.007)*exp(-200.0*y*y);
            tmp += -1.2e-7 * exp(-80.0 * r * r) + 1.4e-7 * exp(-40.0 * r * r);

            data.image[ID1].array.F[jj * size + ii] = tmp;
            data.image[ID2].array.F[jj * size + ii] =
                data.image[ID1].array.F[jj * size + ii];
            data.image[ID3].array.F[jj * size + ii] =
                data.image[ID].array.F[jj * size + ii] -
                data.image[ID2].array.F[jj * size + ii];
        }

    save_fl_fits("fitim", "fitim");
    save_fl_fits("residual", "residual0");
    //   exit(0);

    for(iter = 0; iter < NBiter; iter++)
    {
        for(i = 0; i < NBgridpts1D; i++)
            for(j = 0; j < NBgridpts1D; j++)
            {
                Varraytmp[j * NBgridpts1D + i] = 0.0;
                Varraycnt[j * NBgridpts1D + i] = 0.0;
            }

        for(ii = 0; ii < size; ii++)
            for(jj = 0; jj < size; jj++)
            {
                x = (1.0 * ii - size / 2) / radius;
                y = (1.0 * jj - size / 2) / radius;
                r = sqrt(x * x + y * y);
                if(r < 1.0)
                    for(i = 0; i < NBgridpts1D; i++)
                        for(j = 0; j < NBgridpts1D; j++)
                        {
                            x1 = (x - x0array[j * NBgridpts1D + i]) / xstep;
                            y1 = (y - y0array[j * NBgridpts1D + i]) / ystep;
                            if((fabs(x1) < 1.0) && (fabs(y1) < 1.0))
                            {
                                //value = (fabs(x1)-1.0)*(fabs(y1)-1.0); //0.25*(cos(x1*PI)+1.0)*(cos(y1*PI)+1.0);
                                value = 0.25 * (cos(x1 * PI) + 1.0) *
                                        (cos(y1 * PI) + 1.0);
                                Varraytmp[j * NBgridpts1D + i] +=
                                    value *
                                    data.image[ID3].array.F[jj * size + ii];
                                Varraycnt[j * NBgridpts1D + i] += value;
                            }
                        }
            }
        for(i = 0; i < NBgridpts1D; i++)
            for(j = 0; j < NBgridpts1D; j++)
            {
                if(Varraycnt[j * NBgridpts1D + i] > 1.0)
                {
                    Varraytmp[j * NBgridpts1D + i] /=
                        Varraycnt[j * NBgridpts1D + i];
                }
                else
                {
                    Varraytmp[j * NBgridpts1D + i] = 0.0;
                }
            }

        for(i = 0; i < NBgridpts1D; i++)
            for(j = 0; j < NBgridpts1D; j++)
            {
                Varray[j * NBgridpts1D + i] += Varraytmp[j * NBgridpts1D + i];
            }

        for(ii = 0; ii < size; ii++)
            for(jj = 0; jj < size; jj++)
            {
                data.image[ID2].array.F[jj * size + ii] =
                    data.image[ID1].array.F[jj * size + ii];
            }

        for(ii = 0; ii < size; ii++)
            for(jj = 0; jj < size; jj++)
            {
                x = (1.0 * ii - size / 2) / radius;
                y = (1.0 * jj - size / 2) / radius;
                r = sqrt(x * x + y * y);
                if(r < 1.0)
                    for(i = 0; i < NBgridpts1D; i++)
                        for(j = 0; j < NBgridpts1D; j++)
                        {
                            x1 = (x - x0array[j * NBgridpts1D + i]) / xstep;
                            y1 = (y - y0array[j * NBgridpts1D + i]) / ystep;
                            if((fabs(x1) < 1.0) && (fabs(y1) < 1.0))
                            {
                                //value = (fabs(x1)-1.0)*(fabs(y1)-1.0);
                                value = 0.25 * (cos(x1 * PI) + 1.0) *
                                        (cos(y1 * PI) + 1.0);
                                data.image[ID2].array.F[jj * size + ii] +=
                                    value * Varray[j * NBgridpts1D + i];
                            }
                        }
            }
        cnt   = 0;
        error = 0.0;
        for(ii = 0; ii < size; ii++)
            for(jj = 0; jj < size; jj++)
            {
                x = (1.0 * ii - size / 2) / radius;
                y = (1.0 * jj - size / 2) / radius;
                r = sqrt(x * x + y * y);
                if(r < 1.0)
                {
                    data.image[ID3].array.F[jj * size + ii] =
                        data.image[ID].array.F[jj * size + ii] -
                        data.image[ID2].array.F[jj * size + ii];
                    error += data.image[ID3].array.F[jj * size + ii] *
                             data.image[ID3].array.F[jj * size + ii];
                    cnt++;
                }
            }
        printf("Iteration %ld / %ld    error = %g RMS\n",
               iter,
               NBiter,
               sqrt(error / cnt));

        save_fl_fits("residual", "residual");
        save_fl_fits("fitim", "fitim");
    }

    free(x0array);
    free(y0array);
    free(Varray);
    free(Varraycnt);

    DEBUG_TRACE_FEXIT();

    return (0);
}
