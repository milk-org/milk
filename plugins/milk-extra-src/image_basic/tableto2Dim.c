/** @file tableto2Dim.c
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"

#include "image_filter/image_filter.h"
#include "image_gen/image_gen.h"

#include "kdtree/kdtree.h"

/* ----------------------------------------------------------------------
 *
 * turns a list of 2D points into an image by interpolation
 *
 *
 * ---------------------------------------------------------------------- */

imageID basic_tableto2Dim(const char *__restrict fname,
                          float xmin,
                          float xmax,
                          float ymin,
                          float ymax,
                          long  xsize,
                          long  ysize,
                          const char *__restrict ID_name,
                          float convsize)
{
    FILE   *fp;
    imageID ID;
    float   x, y;
    long    ii, jj;
    double  tot, cnt, cntx, cnty, totx, toty, slx, sly, xave, yave, vave, totsx,
            totsy;
    long          i;
    long          NBpts;
    double       *xarray = NULL;
    double       *yarray = NULL;
    double       *varray = NULL;
    void         *ptree  = NULL;
    double        buf[2];
    double        pt[2];
    double        radius, radius0;
    double       *pv       = NULL;
    struct kdres *presults = NULL;
    double        dist;
    double       *pos;
    //  double tmp1;
    int ok;

    long long cnttotal    = 0;
    long long cntrejected = 0;
    double    valm, val0, val1;

    // nearest points
    long    NBnpt, NBnptmax;
    double *pt_x   = NULL;
    double *pt_y   = NULL;
    double *pt_val = NULL;
    float  *pt_val_cp;
    double *pt_coeff  = NULL;
    double *pt_coeff1 = NULL;

    long IDslx, IDsly, IDxerr, IDyerr;

    double radiusmax = 50.0;

    printf("table : %s\n", fname);
    printf("range : %f -> %f    %f -> %f\n", xmin, xmax, ymin, ymax);
    printf("output: %s (%ld x %ld)\n", ID_name, xsize, ysize);
    printf("kernel size = %f\n", convsize);
    printf("radiusmax = %f\n", radiusmax);

    pos = (double *) malloc(sizeof(double) * 2);
    if(pos == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }
    pos[0] = 0.0;
    pos[1] = 0.0;

    // load table into array
    NBpts  = file_number_lines(fname);
    xarray = (double *) malloc(sizeof(double) * NBpts);
    if(xarray == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    yarray = (double *) malloc(sizeof(double) * NBpts);
    if(yarray == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    varray = (double *) malloc(sizeof(double) * NBpts);
    if(varray == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    if((fp = fopen(fname, "r")) == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("fopen() error");
        exit(0);
    }

    for(i = 0; i < NBpts; i++)
    {
        if(fscanf(fp, "%lf %lf %lf\n", &xarray[i], &yarray[i], &varray[i]) !=
                3)
        {
            fprintf(stderr,
                    "%c[%d;%dm ERROR: fscanf [ %s  %s  %d ] %c[%d;m\n",
                    (char) 27,
                    1,
                    31,
                    __FILE__,
                    __func__,
                    __LINE__,
                    (char) 27,
                    0);
            exit(0);
        }
    }
    fclose(fp);

    printf("%ld points read\n", NBpts);
    fflush(stdout);

    /* create a k-d tree for 2-dimensional points */
    ptree = kd_create(2);

    /* add nodes to the tree */
    for(i = 0; i < NBpts; i++)
    {
        buf[0] = xarray[i];
        buf[1] = yarray[i];
        kd_insert(ptree, buf, &varray[i]);
    }

    create_2Dimage_ID(ID_name, xsize, ysize, &ID);

    create_2Dimage_ID("tmp2dinterpslx", xsize, ysize, &IDslx);
    create_2Dimage_ID("tmp2dinterpsly", xsize, ysize, &IDsly);
    create_2Dimage_ID("tmp2dinterpxerr", xsize, ysize, &IDxerr);
    create_2Dimage_ID("tmp2dinterpyerr", xsize, ysize, &IDyerr);

    // automatically set radius0 such that if points are randomly distributed, a circle of radius radius0 includes sqrt(NBpts) points
    //  tmp1 = sqrt(NBpts);
    //  if(tmp1<100.0)
    //   tmp1 = 100.0;
    // if(tmp1>1000.0)
    //  tmp1 = 1000.0;
    // radius0 = sqrt(tmp1/PI)*sqrt((xmax-xmin)*(ymax-ymin))/sqrt(NBpts);
    radius0 = 5.0 * convsize / sqrt(1.0 * xsize * ysize);
    radius0 *= sqrt((xmax - xmin) * (ymax - ymin));

    NBnpt    = 15000;
    NBnptmax = NBnpt;
    pt_x     = (double *) malloc(sizeof(double) * NBnpt);
    if(pt_x == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    pt_y = (double *) malloc(sizeof(double) * NBnpt);
    if(pt_y == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    pt_val = (double *) malloc(sizeof(double) * NBnpt);
    if(pt_val == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    pt_coeff = (double *) malloc(sizeof(double) * NBnpt);
    if(pt_coeff == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    pt_coeff1 = (double *) malloc(sizeof(double) * NBnpt);
    if(pt_coeff1 == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    pt_val_cp = (float *) malloc(sizeof(float) * NBnpt);
    if(pt_val_cp == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    pt_x[0]      = 0.0;
    pt_y[0]      = 0.0;
    pt_val[0]    = 0.0;
    pt_coeff[0]  = 0.0;
    pt_coeff1[0] = 0.0;
    pt_val_cp[0] = 0.0;

    printf("radius = %g\n", radius0);
    fflush(stdout);

    printf("\n");
    for(ii = 0; ii < xsize; ii++)
    {
        printf("\r[%ld/%ld]   ", ii, xsize);
        fflush(stdout);
        for(jj = 0; jj < ysize; jj++)
        {
            //	printf("[%ld %ld]\n",ii,jj);
            //	fflush(stdout);

            x = (float)(1.0 * xmin + 1.0 * (xmax - xmin) * ii / xsize);
            y = (float)(1.0 * ymin + 1.0 * (ymax - ymin) * jj / ysize);

            /* find points closest to the origin and within distance radius */
            pt[0] = x;
            pt[1] = y;

            radius = radius0;
            ok     = 0;
            while(ok == 0)
            {
                presults = kd_nearest_range(ptree, pt, radius);

                //presults = kd_nearest( ptree, pt );

                /* print out all the points found in results */
                //  printf( "[%g %g] found %d results (radius = %f):\n", x,y, kd_res_size(presults), radius );
                if((kd_res_size(presults) < 30) && (radius < radiusmax))
                {
                    radius *= 1.5;
                    //	  printf("        radius -> %f\n",radius);
                    //  fflush(stdout);
                }
                else
                {
                    ok = 1;
                }
            }

            if(radius < 0.99 * radiusmax)
            {
                NBnpt = kd_res_size(presults);
                //printf("NBnpt = %ld\n",NBnpt);
                //fflush(stdout);
                if(NBnpt > NBnptmax)
                {
                    pt_x      = realloc(pt_x, sizeof(double) * NBnpt);
                    pt_y      = realloc(pt_y, sizeof(double) * NBnpt);
                    pt_val    = realloc(pt_val, sizeof(double) * NBnpt);
                    pt_coeff  = realloc(pt_coeff, sizeof(double) * NBnpt);
                    pt_coeff1 = realloc(pt_coeff1, sizeof(double) * NBnpt);
                    NBnptmax  = NBnpt;
                    //  printf("Reallocation to %ld points\n",NBnpt);
                    // fflush(stdout);
                }

                i = 0;
                while(kd_res_end(presults) == 0)
                {
                    /* get the data and position of the current result item */
                    pv = (double *) kd_res_item(presults, pos);

                    /* compute the distance of the current result from the pt */
                    dist = sqrt((pos[0] - pt[0]) * (pos[0] - pt[0]) +
                                (pos[1] - pt[1]) * (pos[1] - pt[1]));

                    pt_x[i]      = pos[0];
                    pt_y[i]      = pos[1];
                    pt_val[i]    = *pv;
                    pt_val_cp[i] = (float) pt_val[i];
                    pt_coeff[i] =
                        pow((1.0 + cos(M_PI * dist / radius0)) / 2.0, 2.0);
                    pt_coeff1[i] = pow(dist / radius0, 2.0) *
                                   (1.0 + cos(M_PI * dist / radius0)) / 2.0;
                    /* go to the next entry */
                    free(pv);
                    kd_res_next(presults);

                    i++;
                }

                // reject outliers
                // sort values
                quick_sort_float(pt_val_cp, NBnpt);
                valm = pt_val_cp[(long)(NBnpt / 2)];  // median
                val0 = pt_val_cp[(long)(0.3 * NBnpt)];
                val1 = pt_val_cp[(long)(0.7 * NBnpt)];
                for(i = 0; i < NBnpt; i++)
                {
                    cnttotal++;
                    if(fabs(pt_val[i] - valm) > 3.0 * (val1 - val0))
                    {
                        pt_coeff[i]  = 0.0;
                        pt_coeff1[i] = 0.0;
                        cntrejected++;
                    }
                }

                tot  = 0.0;
                totx = 0.0;
                toty = 0.0;
                cnt  = 0.0;
                for(i = 0; i < NBnpt; i++)
                {
                    tot += pt_val[i] * pt_coeff[i];
                    totx += pt_x[i] * pt_coeff[i];
                    toty += pt_y[i] * pt_coeff[i];
                    cnt += pt_coeff[i];
                }

                xave = totx / cnt;
                yave = toty / cnt;
                vave = tot / cnt;

                totsx = 0.0;
                totsy = 0.0;
                cntx  = 0.0;
                cnty  = 0.0;

                for(i = 0; i < NBnpt; i++)
                {
                    if(fabs(pt_x[i] - xave) > 0.01 * radius0)
                    {
                        cntx += pt_coeff1[i];
                        totsx += (pt_val[i] - vave) / (pt_x[i] - xave) *
                                 pt_coeff1[i];
                    }
                    if(fabs(pt_y[i] - yave) > 0.01 * radius0)
                    {
                        cnty += pt_coeff1[i];
                        totsy += (pt_val[i] - vave) / (pt_y[i] - yave) *
                                 pt_coeff1[i];
                    }
                }
                if(cntx < 0.0001)
                {
                    cntx = 0.0001;
                }
                if(cnty < 0.0001)
                {
                    cnty = 0.0001;
                }
                slx = totsx / cntx;
                sly = totsy / cnty;

                data.image[ID].array.F[jj * xsize + ii] =
                    (float) vave; //vave + (x-xave)*slx + (y-yave)*sly;

                data.image[IDxerr].array.F[jj * xsize + ii] =
                    (float)(x - xave);
                data.image[IDyerr].array.F[jj * xsize + ii] =
                    (float)(y - yave);
                data.image[IDslx].array.F[jj * xsize + ii] = (float)(slx);
                data.image[IDsly].array.F[jj * xsize + ii] = (float)(sly);
            }
        }
    }

    free(pos);
    printf("\n");

    printf("fraction of points rejected = %g\n",
           (double)(1.0 * cntrejected / cnttotal));

    free(pt_x);
    free(pt_y);
    free(pt_val);
    free(pt_val_cp);
    free(pt_coeff);
    free(pt_coeff1);

    free(xarray);
    free(yarray);
    free(varray);
    if(presults != NULL)
    {
        kd_res_free(presults);
    }
    if(presults != NULL)
    {
        PRINT_ERROR(
            "presults was not freed by kd_res_free. Attempting free(presults) "
            "and continuing.");
        free(presults);
    }
    kd_free(ptree);
    if(ptree != NULL)
    {
        PRINT_ERROR(
            "ptree was not freed by kd_free. Attempting free(ptree) and "
            "continuing.");
        free(ptree);
    }
    save_fl_fits(ID_name, "tmp2dinterp.fits");

    make_gauss("kerg",
               xsize,
               ysize,
               convsize,
               (float) 1.0); //(long) (10.0*convsize+2.0));

    fconvolve_padd("tmp2dinterpxerr",
                   "kerg",
                   (long)(10.0 * convsize + 2.0),
                   "tmp2dinterpxerrg");
    fconvolve_padd("tmp2dinterpyerr",
                   "kerg",
                   (long)(10.0 * convsize + 2.0),
                   "tmp2dinterpyerrg");
    fconvolve_padd("tmp2dinterpslx",
                   "kerg",
                   (long)(10.0 * convsize + 2.0),
                   "tmp2dinterpslxg");
    fconvolve_padd("tmp2dinterpsly",
                   "kerg",
                   (long)(10.0 * convsize + 2.0),
                   "tmp2dinterpslyg");

    delete_image_ID("tmp2dinterpxerr", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("tmp2dinterpyerr", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("tmp2dinterpslx", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("tmp2dinterpsly", DELETE_IMAGE_ERRMODE_WARNING);

    IDxerr = image_ID("tmp2dinterpxerrg");
    IDyerr = image_ID("tmp2dinterpyerrg");
    IDslx  = image_ID("tmp2dinterpslxg");
    IDsly  = image_ID("tmp2dinterpslyg");
    ID     = image_ID(ID_name);

    for(ii = 0; ii < xsize; ii++)
        for(jj = 0; jj < ysize; jj++)
        {
            //xerr = data.image[IDxerr].array.F[jj*xsize+ii];
            //yerr = data.image[IDyerr].array.F[jj*xsize+ii];
            slx = data.image[IDslx].array.F[jj * xsize + ii];
            sly = data.image[IDsly].array.F[jj * xsize + ii];
            //	data.image[ID].array.F[jj*xsize+ii] += xerr*slx+yerr*sly;
        }

    delete_image_ID("tmp2dinterpxerrg", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("tmp2dinterpyerrg", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("tmp2dinterpslxg", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("tmp2dinterpslyg", DELETE_IMAGE_ERRMODE_WARNING);

    ID = image_ID(ID_name);
    delete_image_ID("kerg", DELETE_IMAGE_ERRMODE_WARNING);

    return (ID);
}
