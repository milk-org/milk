/**
 * @file    linopt_imtools.c
 * @brief   linear optimization tools
 *
 * CPU-based lineal algebra tools: decomposition, SVD etc...
 *
 *
 */

/* ================================================================== */
/* ================================================================== */
/*            MODULE INFO                                             */
/* ================================================================== */
/* ================================================================== */

// module default short name
// all CLI calls to this module functions will be <shortname>.<funcname>
// if set to "", then calls use <funcname>
#define MODULE_SHORTNAME_DEFAULT "lintools"

// Module short description
#define MODULE_DESCRIPTION "Image linear decomposition and optimization tools"

#include <ctype.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_multimin.h>
#include <malloc.h>
#include <math.h>
#include <sched.h>
#include <semaphore.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#include <time.h>

#include <fitsio.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"

#include "linalgebra/linalgebra.h"
#include "info/info.h"
#include "linopt_imtools/linopt_imtools.h"
#include "statistic/statistic.h"

#include "CommandLineInterface/timeutils.h"

#include "compute_SVDdecomp.h"
#include "compute_SVDpseudoInverse.h"
#include "image_construct.h"
#include "image_fitModes.h"
#include "image_to_vec.h"
#include "imcube_crossproduct.h"
#include "lin1Dfit.h"
#include "makeCPAmodes.h"
#include "makeCosRadModes.h"
#include "mask_to_pixtable.h"

/*
static long NBPARAM;
static long double C0;
// polynomial coeff (degree = 1)
static long double *polycoeff1 = NULL;
// polynomial coeff (degree = 2)
static long double *polycoeff2 = NULL;
static long dfcnt = 0;
*/

/* ================================================================== */
/* ================================================================== */
/*            INITIALIZE LIBRARY                                      */
/* ================================================================== */
/* ================================================================== */

// Module initialization macro in CLIcore.h
// macro argument defines module name for bindings
//
INIT_MODULE_LIB(linopt_imtools)

/*

errno_t linopt_imtools_image_construct_stream_cli()
{
    if(
        CLI_checkarg(1, 4) +
        CLI_checkarg(2, 4) +
        CLI_checkarg(3, 4)
        == 0)
    {
        linopt_imtools_image_construct_stream(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.string,
            data.cmdargtoken[3].val.string
        );

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}
*/

static errno_t init_module_CLI()
{

    // CONVERSION

    CLIADDCMD_linopt_imtools__mask_to_pixtable();

    CLIADDCMD_linopt_imtools__image_to_vec();

    CLIADDCMD_linopt_imtools__vec_to_2DImage();

    // CREATE MODES

    CLIADDCMD_linopt_imtools__makeCosRadModes();

    CLIADDCMD_linopt_imtools__makeCPAmodes();

    // LINEAR DECOMPOSITION

    CLIADDCMD_linopt_imtools__imcube_crossproduct();

    CLIADDCMD_linopt_imtools__image_fitModes();

    CLIADDCMD_linopt_imtools__image_construct();

    /*   RegisterCLIcommand(
           "imlinconstructs",
           __FILE__,
           linopt_imtools_image_construct_stream_cli,
           "construct image as linear sum of modes (stream mode)",
           "<modes> <coeffs> <outim>", "imlinconstructs modes coeffs outim",
           "long linopt_imtools_image_construct_stream(const char *IDmodes_name, const char *IDcoeff_name, const char *IDout_name)");
    */

    CLIADDCMD_linopt_imtools__compute_SVDdecomp();

    CLIADDCMD_linopt_imtools__compute_SVDpseudoinverse();

    CLIADDCMD_linopt_imtools__lin1Dfits();

    // OPTIMIZATION

    CLIADDCMD_linopt_imtools__linRM_from_inout();

    return RETURN_SUCCESS;
}

/* =============================================================================================== */
/* =============================================================================================== */
/*                                                                                                 */
/* 3. CREATE MODES                                                                                 */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */

// r0pix is r=1 in pixel unit

imageID linopt_imtools_make1Dpolynomials(const char *IDout_name,
        long        NBpts,
        long        MaxOrder,
        float       r0pix)
{
    DEBUG_TRACE_FSTART();

    imageID IDout;
    long    xsize, ysize, zsize;
    long    ii, kk;

    xsize = NBpts;
    ysize = 1;
    zsize = MaxOrder;

    FUNC_CHECK_RETURN(
        create_3Dimage_ID(IDout_name, xsize, ysize, zsize, &IDout));

    for(kk = 0; kk < zsize; kk++)
    {
        for(ii = 0; ii < xsize; ii++)
        {
            float r                                    = 1.0 * ii / r0pix;
            data.image[IDout].array.F[kk * xsize + ii] = pow(r, 1.0 * kk);
        }
    }

    DEBUG_TRACE_FEXIT();
    return IDout;
}

/* --------------------------------------------------------------- */
/*                                                                 */
/*           Functions for optimization                            */
/*                                                                 */
/* --------------------------------------------------------------- */

/*
double linopt_imtools_opt_f(
    const gsl_vector *v,
    __attribute__((unused)) void *params
)
{
    double value;
    long k, l, n;


    n = NBPARAM;
    value = C0;
    for(k = 0; k < n; k++)
    {
        value += polycoeff1[k] * gsl_vector_get(v, k);
    }
    for(k = 0; k < n; k++)
        for(l = 0; l < n; l++)
        {
            value += polycoeff2[l * n + k] * gsl_vector_get(v, k) * gsl_vector_get(v, l);
        }

    return(value);
}
*/

/*
void linopt_imtools_opt_df(
    const gsl_vector *v,
    void             *params,
    gsl_vector       *df
)
{
    double epsilon = 1.0e-8;
    long i, j;
    double v1, v2;
    gsl_vector *vcp;

    vcp = gsl_vector_alloc(NBPARAM);
    //v0 = linopt_imtools_opt_f (v, params);

    for(i = 0; i < NBPARAM; i++)
    {
        for(j = 0; j < NBPARAM; j++)
        {
            gsl_vector_set(vcp, j, gsl_vector_get(v, j));
        }
        gsl_vector_set(vcp, i, gsl_vector_get(v, i) + epsilon);
        v1 = linopt_imtools_opt_f(vcp, params);
        gsl_vector_set(vcp, i, gsl_vector_get(v, i) - epsilon);
        v2 = linopt_imtools_opt_f(vcp, params);
        gsl_vector_set(df, i, (double)((v1 - v2) / (2.0 * epsilon)));
    }

    if(0)
    {
        printf("%ld df = (", dfcnt);
        for(i = 0; i < NBPARAM; i++)
        {
            printf(" %g", gsl_vector_get(df, i));
        }
        printf(" )\n");
    }
    dfcnt ++;

    if(dfcnt > 50)
    {
        exit(0);
    }

    gsl_vector_free(vcp);
}
*/

/*

void linopt_imtools_opt_fdf(
    const gsl_vector *x,
    void             *params,
    double           *f,
    gsl_vector       *df
)
{
    *f = linopt_imtools_opt_f(x, params);
    linopt_imtools_opt_df(x, params, df);
}
*/

/*
// FLOAT only
imageID linopt_imtools_image_construct_stream(
    const char *IDmodes_name,
    const char *IDcoeff_name,
    const char *IDout_name
)
{
    imageID IDout;
    imageID IDmodes;
    imageID IDcoeff;
    long ii, kk;
    long xsize, ysize, zsize;
    long sizexy;
    int semval;
    uint64_t cnt = 0;
    int RT_priority = 80; //any number from 0-99
    struct sched_param schedpar;
    int NOSEM = 1; // ignore input semaphore, use counter


    schedpar.sched_priority = RT_priority;
    sched_setscheduler(0, SCHED_FIFO,
                       &schedpar); //other option is SCHED_RR, might be faster



    IDmodes = image_ID(IDmodes_name);
    //datatype = data.image[IDmodes].md[0].datatype;

    xsize = data.image[IDmodes].md[0].size[0];
    ysize = data.image[IDmodes].md[0].size[1];
    zsize = data.image[IDmodes].md[0].size[2];

    sizexy = xsize * ysize;

    if(variable_ID("NOSEM") != -1)
    {
        NOSEM = 1;
    }
    else
    {
        NOSEM = 0;
    }

    IDout = image_ID(IDout_name);
    IDcoeff = image_ID(IDcoeff_name);

    while(1 == 1)
    {
        if((data.image[IDcoeff].md[0].sem == 0) || (NOSEM == 1))
        {
            while(cnt == data.image[IDcoeff].md[0].cnt0) // test if new frame exists
            {
                usleep(5);
            }
            cnt = data.image[IDcoeff].md[0].cnt0;
        }
        else
        {
            sem_wait(data.image[IDcoeff].semptr[0]);
        }

        for(ii = 0; ii < sizexy; ii++)
        {
            data.image[IDout].array.F[ii] = 0.0;
        }

        data.image[IDout].md[0].write = 1;
        for(kk = 0; kk < zsize; kk++)
            for(ii = 0; ii < sizexy; ii++)
            {
                data.image[IDout].array.F[ii] += data.image[IDcoeff].array.F[kk] *
                                                 data.image[IDmodes].array.F[kk * sizexy + ii];
            }
        sem_getvalue(data.image[IDout].semptr[0], &semval);
        if(semval < SEMAPHORE_MAXVAL)
        {
            sem_post(data.image[IDout].semptr[0]);
        }

        data.image[IDout].md[0].cnt0++;
        data.image[IDout].md[0].write = 0;
    }

    return IDout;
}

*/

/*
//
// match a single image (ID_name) to a linear sum of images within IDref_name
// result is a 1D array of coefficients in IDsol_name
//
double linopt_imtools_match_slow(
    const char *ID_name,
    const char *IDref_name,
    const char *IDmask_name,
    const char *IDsol_name,
    const char *IDout_name
)
{
    long ID, IDref, IDmask, IDsol, IDout;
    long naxes[2];
    long n; // number of reference frames
    long ii, k, l;

    long double val;
    long double valbest;

    // initial random search
    long riter;
    long riterMax = 1000000;

    long double v0;
    long double *tarray = NULL; // temporary array to store values for fixed pixel


    // ref image coefficients (solutions)
    long double *alpha = NULL;
    long double *alphabest = NULL;
    long double ampl;

    //
    //  the optimization problem is first rewritten as a 2nd degree polynomial of alpha values
    //  val = V0 + SUM_{k=0...n-1}{polycoeff1[k]*alpha[k] + SUM_{k=0...n-1}{l=0...k}{polycoeff2[k,l]*alpha[k]*alpha[l]}
    //

    long iter = 0;
    double *params;
    const gsl_multimin_fdfminimizer_type *T;
    gsl_multimin_fdfminimizer *sminimizer;
    long i;
    gsl_vector *x;
    gsl_multimin_function_fdf opt_func;
    int status;

    //  printf("Input params : %s %s %s\n",ID_name,IDref_name,IDsol_name);

    params = (double *) malloc(sizeof(double) * 1);
    if(params == NULL) {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    params[0] = 0.0;


    ID = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    IDmask = image_ID(IDmask_name);
    IDref = image_ID(IDref_name);
    n = data.image[IDref].md[0].size[2];

    printf("Number of points = %ld x %ld\n", naxes[0]*naxes[1], n);


    alpha = (long double *) malloc(sizeof(long double) * n);
    if(alpha == NULL)
    {
        PRINT_ERROR("Cannot allocate memory");
        exit(0);
    }
    alphabest = (long double *) malloc(sizeof(long double) * n);
    if(alphabest == NULL)
    {
        PRINT_ERROR("Cannot allocate memory");
        exit(0);
    }



    polycoeff1 = (long double *) malloc(sizeof(long double) * n);
    if(polycoeff1 == NULL)
    {
        PRINT_ERROR("Cannot allocate memory");
        exit(0);
    }
    polycoeff2 = (long double *) malloc(sizeof(long double) * n * n);
    if(polycoeff2 == NULL)
    {
        PRINT_ERROR("Cannot allocate memory");
        exit(0);
    }

    tarray = (long double *) malloc(sizeof(long double) * n);
    if(tarray == NULL)
    {
        PRINT_ERROR("Cannot allocate memory");
        exit(0);
    }



    // initialize all coeffs to zero
    C0 = 0.0;
    for(k = 0; k < n; k++)
    {
        alpha[k] = 1.0 / n;
        polycoeff1[k] = 0.0;
        for(l = 0; l < n; l++)
        {
            polycoeff2[l * n + k] = 0.0;
        }
    }

    // compute polynomial coefficients
    for(ii = 0; ii < naxes[0]*naxes[1]; ii++)
    {
        v0 = (long double)(data.image[ID].array.F[ii] * data.image[IDmask].array.F[ii]);
        for(k = 0; k < n; k++)
        {
            tarray[k] = (long double)(data.image[IDref].array.F[naxes[0] * naxes[1] * k +
                                      ii] * data.image[IDmask].array.F[ii]);
        }
        C0 += v0 * v0;
        for(k = 0; k < n; k++)
        {
            polycoeff1[k] += -2.0 * v0 * tarray[k];
        }
        for(k = 0; k < n; k++)
            for(l = 0; l < n; l++)
            {
                polycoeff2[l * n + k] += tarray[k] * tarray[l];
            }
    }

    // find solution
    //   val = C0 + SUM_{k=0...n-1}{polycoeff1[k]*alpha[k] + SUM_{k=0...n-1}{l=0...k}{polycoeff2[k,l]*alpha[k]*alpha[l]}
    //
    val = C0;
    for(k = 0; k < n; k++)
    {
        val += polycoeff1[k] * alpha[k];
    }
    for(k = 0; k < n; k++)
        for(l = 0; l < n; l++)
        {
            val += polycoeff2[l * n + k] * alpha[k] * alpha[l];
        }


    for(k = 0; k < n; k++)
    {
        printf("%g ", (double) alpha[k]);
    }
    printf("-> %g\n", (double) val);
    for(k = 0; k < n; k++)
    {
        alphabest[k] = alpha[k];
    }
    valbest = val;





    for(riter = 0; riter < riterMax; riter++)
    {
        ampl = pow(ran1(), 4.0);
        for(k = 0; k < n; k++)
        {
            alpha[k] = alphabest[k] + ampl * (1.0 - 2.0 * ran1()) / n;
        }

        val = C0;
        for(k = 0; k < n; k++)
        {
            val += polycoeff1[k] * alpha[k];
        }
        for(k = 0; k < n; k++)
            for(l = 0; l < n; l++)
            {
                val += polycoeff2[l * n + k] * alpha[k] * alpha[l];
            }
        if(val < valbest)
        {
            //printf("[%ld/%ld] ",riter,riterMax);
            //for(k=0;k<n;k++)
            //  printf(" %g ", (double) alpha[k]);
            //printf("-> %g\n", (double) val);
            for(k = 0; k < n; k++)
            {
                alphabest[k] = alpha[k];
            }
            valbest = val;
        }
    }

    NBPARAM = n;

    x = gsl_vector_alloc(n);

    for(i = 0; i < n; i++)
    {
        gsl_vector_set(x, i, alphabest[i]);
    }
    printf("Value = %g\n", linopt_imtools_opt_f(x, params));



    opt_func.n = n;
    opt_func.f = &linopt_imtools_opt_f;
    opt_func.df = &linopt_imtools_opt_df;
    opt_func.fdf = &linopt_imtools_opt_fdf;
    opt_func.params = &params;

    x = gsl_vector_alloc(n);

    for(i = 0; i < n; i++)
    {
        gsl_vector_set(x, i, alphabest[i]);
    }

    T = gsl_multimin_fdfminimizer_vector_bfgs2;
    sminimizer = gsl_multimin_fdfminimizer_alloc(T, n);

    gsl_multimin_fdfminimizer_set(sminimizer, &opt_func, x, 1.0e-5, 0.1);

    do
    {
        iter++;
        dfcnt = 0;
        status = gsl_multimin_fdfminimizer_iterate(sminimizer);
        if(status)
        {
            break;
        }
        status = gsl_multimin_test_gradient(sminimizer->gradient, 1e-5);
        if(status == GSL_SUCCESS)
        {
            printf("Minimum found at:\n");
            printf("%5ld : ", iter);
            //for(i=0;i<n;i++)
            // printf("%.8f ",gsl_vector_get(sminimizer->x, i));
            printf("    %10.8f\n", sminimizer->f);
        }
    }
    while(status == GSL_CONTINUE && iter < 1000);

    for(i = 0; i < n; i++)
    {
        alphabest[i] = gsl_vector_get(sminimizer->x, i);
    }

    for(i = 0; i < n; i++)
    {
        gsl_vector_set(x, i, alphabest[i]);
    }
    printf("Value after minimization = %g\n", linopt_imtools_opt_f(x, params));

    gsl_multimin_fdfminimizer_free(sminimizer);
    gsl_vector_free(x);


    create_2Dimage_ID(IDsol_name, n, 1, &IDsol);
    for(i = 0; i < n; i++)
    {
        data.image[IDsol].array.F[i] = alphabest[i];
    }



    // compute residual

    create_2Dimage_ID(IDout_name, naxes[0], naxes[1], &IDout);

    for(ii = 0; ii < naxes[0]*naxes[1]; ii++)
    {
        data.image[IDout].array.F[ii] = 0.0;
    }
    for(k = 0; k < n; k++)
        for(ii = 0; ii < naxes[0]*naxes[1]; ii++)
        {
            data.image[IDout].array.F[ii] += alphabest[k] *
                                             data.image[IDref].array.F[naxes[0] * naxes[1] * k + ii];
        }


    free(alpha);
    alpha = NULL;
    free(alphabest);
    alphabest = NULL;
    free(polycoeff1);
    polycoeff1 = NULL;
    free(polycoeff2);
    polycoeff2 = NULL;
    free(tarray);
    tarray = NULL;

    free(params);

    return((double) val);
}

*/

// match a single image (ID_name) to a linear sum of images within IDref_name
// result is a 1D array of coefficients in IDsol_name
//
// n = number of observations
// p = number of variables
//
// ID_name is input, size (n,1)
// IDsol_name must contain initial solution
//
/*
double linopt_imtools_match(
    const char *ID_name,
    const char *IDref_name,
    const char *IDmask_name,
    const char *IDsol_name,
    const char *IDout_name
)
{
    gsl_multifit_linear_workspace *work;
    uint32_t n, p;
    imageID ID, IDref, IDmask, IDsol, IDout;
    long naxes[3];
    long i, j, k, ii;
    gsl_matrix *X;
    gsl_vector *y; // measurements
    gsl_vector *c;
    gsl_vector *w;
    gsl_matrix *cov;
    double chisq;

    ID = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];
    n = naxes[0] * naxes[1];
    IDmask = image_ID(IDmask_name);
    IDref = image_ID(IDref_name);
    p = data.image[IDref].md[0].size[2];

    // some verification
    if(IDref == -1)
    {
        PRINT_ERROR("input ref missing\n");
    }
    if(IDmask == -1)
    {
        PRINT_ERROR("input mask missing\n");
    }
    if(data.image[IDmask].md[0].size[0] != data.image[ID].md[0].size[0])
    {
        PRINT_ERROR("mask size[0] is wrong\n");
    }
    if(data.image[IDmask].md[0].size[1] != data.image[ID].md[0].size[1])
    {
        PRINT_ERROR("mask size[1] is wrong\n");
    }


    printf("n,p = %ld %ld\n", (long) n, (long) p);
    fflush(stdout);

    y = gsl_vector_alloc(n);  // measurements
    for(i = 0; i < n; i++)
    {
        gsl_vector_set(y, i, data.image[ID].array.F[i]);
    }

    w = gsl_vector_alloc(n);
    for(i = 0; i < n; i++)
    {
        gsl_vector_set(w, i, data.image[IDmask].array.F[i]);
    }

    X = gsl_matrix_alloc(n, p);
    for(i = 0; i < n; i++)
        for(j = 0; j < p; j++)
        {
            gsl_matrix_set(X, i, j, data.image[IDref].array.F[j * n + i]);
        }
    c = gsl_vector_alloc(p);  // solution (coefficients)
    cov = gsl_matrix_alloc(p, p);

    work = gsl_multifit_linear_alloc(n, p);

    printf("-");
    fflush(stdout);
    gsl_multifit_wlinear(X, w, y, c, cov, &chisq, work);
    printf("-");
    fflush(stdout);

    create_2Dimage_ID(IDsol_name, p, 1, &IDsol);
    for(i = 0; i < p; i++)
    {
        data.image[IDsol].array.F[i] = gsl_vector_get(c, i);
    }

    gsl_multifit_linear_free(work);
    gsl_vector_free(y);
    gsl_vector_free(w);
    gsl_matrix_free(X);
    gsl_vector_free(c);
    gsl_matrix_free(cov);

    printf(" . ");
    fflush(stdout);

    // compute residual
    create_2Dimage_ID(IDout_name, naxes[0], naxes[1], &IDout);
    for(ii = 0; ii < naxes[0]*naxes[1]; ii++)
    {
        data.image[IDout].array.F[ii] = 0.0;
    }
    for(k = 0; k < p; k++)
        for(ii = 0; ii < naxes[0]*naxes[1]; ii++)
        {
            data.image[IDout].array.F[ii] += data.image[IDsol].array.F[k] *
                                             data.image[IDref].array.F[naxes[0] * naxes[1] * k + ii];
        }

    return(chisq);
}

*/
