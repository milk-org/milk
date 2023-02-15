/** @file percentile_interpolation.c
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"

#include "image_gen/image_gen.h"

#include "fconvolve.h"

imageID FILTER_percentile_interpol_fast(const char *ID_name,
                                        const char *IDout_name,
                                        double      perc,
                                        long        boxrad)
{
    imageID ID, ID1, IDout;
    long    step;
    long    ii, jj, ii1, jj1, ii2, jj2;
    long    iis, iie, jjs, jje;
    long    xsize, ysize, xsize1, ysize1;
    double *array;
    double  v00, v01, v10, v11;
    double  u, t, ii1f, jj1f, x, y;
    long    cnt;
    long    pixstep = 5;
    long    IDpercmask; // optional mask file

    step = (long)(0.7 * boxrad);
    if(step < 1)
    {
        step = 1;
    }

    ID    = image_ID(ID_name);
    xsize = data.image[ID].md[0].size[0];
    ysize = data.image[ID].md[0].size[1];

    xsize1 = (long)(xsize / step);
    ysize1 = (long)(ysize / step);

    create_2Dimage_ID("_tmppercintf", xsize1, ysize1, &ID1);

    // identify mask if it exists
    IDpercmask = image_ID("_percmask");

    array = (double *) malloc(sizeof(double) * boxrad * boxrad * 4);
    if(array == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    for(ii1 = 0; ii1 < xsize1; ii1++)
        for(jj1 = 0; jj1 < ysize1; jj1++)
        {
            x = 1.0 * (ii1 + 0.5) / xsize1 * xsize;
            y = 1.0 * (jj1 + 0.5) / ysize1 * ysize;

            iis = (long)(x - boxrad);
            if(iis < 0)
            {
                iis = 0;
            }

            iie = (long)(x + boxrad);
            if(iie > xsize)
            {
                iie = xsize;
            }

            jjs = (long)(y - boxrad);
            if(jjs < 0)
            {
                jjs = 0;
            }

            jje = (long)(y + boxrad);
            if(jje > ysize)
            {
                jje = ysize;
            }

            cnt = 0;
            if(IDpercmask == -1)
            {
                for(ii = iis; ii < iie; ii += pixstep)
                    for(jj = jjs; jj < jje; jj += pixstep)
                    {
                        array[cnt] = data.image[ID].array.F[jj * xsize + ii];
                        cnt++;
                    }
            }
            else
            {
                for(ii = iis; ii < iie; ii += pixstep)
                    for(jj = jjs; jj < jje; jj += pixstep)
                    {
                        if(data.image[IDpercmask].array.F[jj * xsize + ii] >
                                0.5)
                        {
                            array[cnt] =
                                data.image[ID].array.F[jj * xsize + ii];
                            cnt++;
                        }
                    }
            }
            quick_sort_double(array, cnt);

            data.image[ID1].array.F[jj1 * xsize1 + ii1] =
                array[(long)(perc * cnt)];
            //	data.image[IDx].array.F[jj1*xsize1+ii1] = 0.5*(iis+iie);
            //data.image[IDy].array.F[jj1*xsize1+ii1] = 0.5*(jjs+jje);
        }
    free(array);

    create_2Dimage_ID(IDout_name, xsize, ysize, &IDout);

    for(ii = 0; ii < xsize; ii++)
        for(jj = 0; jj < ysize; jj++)
        {
            ii1f = 1.0 * ii / xsize * xsize1;
            jj1f = 1.0 * jj / ysize * ysize1;
            ii1  = (long)(ii1f);
            jj1  = (long)(jj1f);

            ii2 = ii1 + 1;
            jj2 = jj1 + 1;

            while(ii2 > xsize1 - 1)
            {
                ii1--;
                ii2--;
            }

            while(jj2 > ysize1 - 1)
            {
                jj1--;
                jj2--;
            }

            u = ii1f - ii1;
            t = jj1f - jj1;

            v00 = data.image[ID1].array.F[jj1 * xsize1 + ii1];
            v10 = data.image[ID1].array.F[jj1 * xsize1 + ii2];
            v01 = data.image[ID1].array.F[jj2 * xsize1 + ii1];
            v11 = data.image[ID1].array.F[jj2 * xsize1 + ii2];

            data.image[IDout].array.F[jj * xsize + ii] =
                (1.0 - u) * (1.0 - t) * v00 + (1.0 - u) * t * v01 +
                u * (1.0 - t) * v10 + u * t * v11;
        }

    delete_image_ID("_tmppercintf", DELETE_IMAGE_ERRMODE_WARNING);

    return IDout;
}

//
// improvement of the spatial median filter
// percentile can be selected different than 50% percentile (parameter perc)
// spatial smoothing parameter (sigma)
//
// this algorithm tests values and build the final map from these tests
// works well for smooth images, with perc between 0.1 and 0.9
//
imageID FILTER_percentile_interpol(const char *__restrict ID_name,
                                   const char *__restrict IDout_name,
                                   double perc,
                                   double sigma)
{
    imageID     ID, IDout, IDtmp, ID2;
    long        NBstep = 10;
    double      Imin, Imax;
    long        xsize, ysize;
    double     *array;
    long        IDc;
    long        k;
    double     *varray;
    long        ii;
    double      value;
    double      range;
    long        IDkern;
    long double tot;
    long        k1, k2;
    double      x, v1, v2;
    double      pstart, pend;

    ID = image_ID(ID_name);

    xsize = data.image[ID].md[0].size[0];
    ysize = data.image[ID].md[0].size[1];

    array = (double *) malloc(sizeof(double) * xsize * ysize);
    if(array == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    varray = (double *) malloc(sizeof(double) * NBstep);
    if(varray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    for(ii = 0; ii < xsize * ysize; ii++)
    {
        array[ii] = data.image[ID].array.F[ii];
    }
    quick_sort_double(array, xsize * ysize);

    pstart = 0.8 * perc - 0.05;
    pend   = 1.2 * perc + 0.05;
    if(pstart < 0.01)
    {
        pstart = 0.01;
    }
    if(pend > 0.99)
    {
        pend = 0.99;
    }

    Imin = array[(long)(pstart * xsize * ysize)];
    Imax = array[(long)(pend * xsize * ysize)];

    range = Imax - Imin;
    Imin -= 0.1 * range;
    Imax += 0.1 * range;

    for(k = 0; k < NBstep; k++)
    {
        varray[k] = Imin + 1.0 * k / (NBstep - 1) * (Imax - Imin);
    }

    free(array);

    printf("Testing %ld values in range %g -> %g\n", NBstep, Imin, Imax);
    fflush(stdout);

    create_3Dimage_ID("_testpercim", xsize, ysize, NBstep, &IDc);

    IDkern = make_gauss("_kern", xsize, ysize, sigma, 1.0);
    tot    = 0.0;
    for(ii = 0; ii < xsize * ysize; ii++)
    {
        tot += data.image[IDkern].array.F[ii];
    }
    for(ii = 0; ii < xsize * ysize; ii++)
    {
        data.image[IDkern].array.F[ii] /= tot;
    }

    create_2Dimage_ID("_testpercim1", xsize, ysize, &IDtmp);
    for(k = 0; k < NBstep; k++)
    {
        printf("   %ld/%ld threshold = %f\n", k, NBstep, varray[k]);
        for(ii = 0; ii < xsize * ysize; ii++)
        {
            value = data.image[ID].array.F[ii];
            if(value < varray[k])
            {
                data.image[IDtmp].array.F[ii] = 1.0;
            }
            else
            {
                data.image[IDtmp].array.F[ii] = 0.0;
            }
        }

        fconvolve_padd("_testpercim1",
                       "_kern",
                       (long)(3.0 * sigma),
                       "_testpercim2");

        ID2 = image_ID("_testpercim2");
        for(ii = 0; ii < xsize * ysize; ii++)
        {
            data.image[IDc].array.F[k * xsize * ysize + ii] =
                data.image[ID2].array.F[ii];
        }
        delete_image_ID("_testpercim2", DELETE_IMAGE_ERRMODE_WARNING);
    }

    create_2Dimage_ID(IDout_name, xsize, ysize, &IDout);
    for(ii = 0; ii < xsize * ysize; ii++)
    {
        k  = 0;
        k1 = 0;
        k2 = 0;
        v1 = 0.0;
        v2 = 0.0;
        while((v2 < perc) && (k < NBstep - 1))
        {
            k++;
            v1 = v2;
            k1 = k2;
            v2 = data.image[IDc].array.F[k * xsize * ysize + ii];
            k2 = k;
        }
        // ideally, v1<perc<v2
        if((v1 < perc) && (perc < v2))
        {
            x = (perc - v1) / (v2 - v1);
            data.image[IDout].array.F[ii] =
                (1.0 - x) * varray[k1] + x * varray[k2];
        }
        else
        {
            if(v1 > perc)
            {
                data.image[IDout].array.F[ii] = varray[0];
            }
            else
            {
                data.image[IDout].array.F[ii] = varray[NBstep - 1];
            }
        }
    }

    //  save_fl_fits("_testpercim","_testpercim.fits");
    delete_image_ID("_kern", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("_testpercim", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("_testpercim1", DELETE_IMAGE_ERRMODE_WARNING);
    free(varray);

    return IDout;
}
