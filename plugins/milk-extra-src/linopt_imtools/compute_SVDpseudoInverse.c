#include <math.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_matrix.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_iofits/savefits.h"
#include "CommandLineInterface/timeutils.h"
#include "linalgebra/linalgebra.h"

// Local variables pointers
static char   *inimname;
static char   *outimname;
static double *SVD_epsilon;
static long   *max_NBmodes;
static char   *outimVTmatname;
static long   *useGPU;

static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".inim",
        "input image",
        "im",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimname,
        NULL
    },
    {
        CLIARG_STR,
        ".outim",
        "output image",
        "outim",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimname,
        NULL
    },
    {
        CLIARG_FLOAT64,
        ".svdeps",
        "SVD cutoff",
        "0.001",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &SVD_epsilon,
        NULL
    },
    {
        CLIARG_INT64,
        ".maxNBmode",
        "Maximum NB modes",
        "10000",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &max_NBmodes,
        NULL
    },
    {
        CLIARG_STR,
        ".outimVT",
        "output VT matrix",
        "outVTmat",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimVTmatname,
        NULL
    },
    {
        CLIARG_INT64,
        ".GPU",
        "use GPU",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &useGPU,
        NULL
    }
};

static CLICMDDATA CLIcmddata =
{
    "impsinvsvd", "compute pseudoinverse", CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}

//
// Computes control matrix
// Conventions:
//   m: number of actuators (= NB_MODES)
//   n: number of sensors  (= # of pixels)
//
// This implementation computes the eigenvalue decomposition of transpose(M) x M, so it is efficient if n>>m, as transpose(M) x M is size m x m
//
errno_t
linopt_compute_SVDpseudoInverse(
    const char *ID_Rmatrix_name,
    const char *ID_Cmatrix_name,
    double      SVDeps,
    long        MaxNBmodes,
    const char *ID_VTmatrix_name,
    imageID    *outID
) /* works for m != n */
{
    DEBUG_TRACE_FSTART();

    FILE       *fp;
    char        fname[200];
    gsl_matrix *matrix_D;  /* this is the input response matrix */
    gsl_matrix *matrix_Ds; /* this is the output pseudo inverse of D */
    gsl_matrix *matrix_Dtra;
    gsl_matrix *matrix_DtraD;
    gsl_matrix *matrix_DtraDinv;
    gsl_matrix *matrix_DtraD_evec;
    gsl_matrix *matrix1;
    gsl_matrix *matrix2;
    gsl_vector *matrix_DtraD_eval;
    gsl_eigen_symmv_workspace *w;

    gsl_matrix *matrix_save;

    long      m;
    long      n;
    imageID   ID_Rmatrix, ID_Cmatrix, ID_VTmatrix;
    uint32_t *arraysizetmp;
    double    egvlim;
    long      nbmodesremoved;

    uint8_t datatype;

    long MaxNBmodes1, mode;

    // Timing
    int             timing = 1;
    struct timespec t0, t1, t2, t3, t4, t5, t6, t7;
    double          t01d, t12d, t23d, t34d, t45d, t56d, t67d;
    struct timespec tdiff;

    int     testmode = 0;
    imageID ID_AtA;
    imageID ID;

    printf("[CPU (gsl) SVD start]");
    fflush(stdout);

    if(timing == 1)
    {
        clock_gettime(CLOCK_MILK, &t0);
    }

    arraysizetmp = (uint32_t *) malloc(sizeof(uint32_t) * 3);
    if(arraysizetmp == NULL)
    {
        FUNC_RETURN_FAILURE("malloc returns NULL pointer");
    }

    ID_Rmatrix = image_ID(ID_Rmatrix_name);
    if(ID_Rmatrix == -1)
    {
        printf("ERROR: matrix %s not found in memory\n", ID_Rmatrix_name);
        exit(0);
    }
    datatype = data.image[ID_Rmatrix].md[0].datatype;
    if(data.image[ID_Rmatrix].md[0].naxis == 3)
    {
        n = data.image[ID_Rmatrix].md[0].size[0] *
            data.image[ID_Rmatrix].md[0].size[1];
        m = data.image[ID_Rmatrix].md[0].size[2];
        printf("3D image -> %ld %ld\n", n, m);
        fflush(stdout);
    }
    else
    {
        n = data.image[ID_Rmatrix].md[0].size[0];
        m = data.image[ID_Rmatrix].md[0].size[1];
        printf("2D image -> %ld %ld\n", n, m);
        fflush(stdout);
    }

    /* in this procedure, m=number of actuators/modes, n=number of WFS elements */
    //  long m = smao[0].NBmode;
    // long n = smao[0].NBwfselem;

    printf("m = %ld , n = %ld \n", m, n);
    fflush(stdout);

    matrix_DtraD_eval = gsl_vector_alloc(m);
    matrix_D          = gsl_matrix_alloc(n, m);
    matrix_Ds         = gsl_matrix_alloc(m, n);
    matrix_Dtra       = gsl_matrix_alloc(m, n);
    matrix_DtraD      = gsl_matrix_alloc(m, m);
    matrix_DtraDinv   = gsl_matrix_alloc(m, m);
    matrix_DtraD_evec = gsl_matrix_alloc(m, m);

    /* write matrix_D */
    if(datatype == _DATATYPE_FLOAT)
    {
        for(int k = 0; k < m; k++)
            for(int ii = 0; ii < n; ii++)
            {
                gsl_matrix_set(matrix_D,
                               ii,
                               k,
                               data.image[ID_Rmatrix].array.F[k * n + ii]);
            }
    }
    else
    {
        for(int k = 0; k < m; k++)
            for(int ii = 0; ii < n; ii++)
            {
                gsl_matrix_set(matrix_D,
                               ii,
                               k,
                               data.image[ID_Rmatrix].array.D[k * n + ii]);
            }
    }

    if(timing == 1)
    {
        clock_gettime(CLOCK_MILK, &t1);
    }

    /* compute DtraD */
    gsl_blas_dgemm(CblasTrans,
                   CblasNoTrans,
                   1.0,
                   matrix_D,
                   matrix_D,
                   0.0,
                   matrix_DtraD);

    if(testmode == 1)
    {
        // TEST
        FUNC_CHECK_RETURN(create_2Dimage_ID("AtA", m, m, &ID_AtA));

        for(int ii = 0; ii < m; ii++)
            for(int jj = 0; jj < m; jj++)
            {
                data.image[ID_AtA].array.F[jj * m + ii] =
                    (float) gsl_matrix_get(matrix_DtraD, ii, jj);
            }
        save_fits("AtA", "test_AtA.fits");
    }

    if(timing == 1)
    {
        clock_gettime(CLOCK_MILK, &t2);
    }

    /* compute the inverse of DtraD */

    /* first, compute the eigenvalues and eigenvectors */
    w           = gsl_eigen_symmv_alloc(m);
    matrix_save = gsl_matrix_alloc(m, m);
    gsl_matrix_memcpy(matrix_save, matrix_DtraD);
    gsl_eigen_symmv(matrix_save, matrix_DtraD_eval, matrix_DtraD_evec, w);
    gsl_matrix_free(matrix_save);
    gsl_eigen_symmv_free(w);

    if(timing == 1)
    {
        clock_gettime(CLOCK_MILK, &t3);
    }

    gsl_eigen_symmv_sort(matrix_DtraD_eval,
                         matrix_DtraD_evec,
                         GSL_EIGEN_SORT_ABS_DESC);

    if(timing == 1)
    {
        clock_gettime(CLOCK_MILK, &t4);
    }

    //  printf("Eigenvalues\n");
    //  fflush(stdout);

    // Write eigenvalues
    sprintf(fname, "eigenv.dat");
    if((fp = fopen(fname, "w")) == NULL)
    {
        printf("ERROR: cannot create file \"%s\"\n", fname);
        exit(0);
    }
    for(int k = 0; k < m; k++)
    {
        fprintf(fp,
                "%d %g %g\n",
                k,
                sqrt(gsl_vector_get(matrix_DtraD_eval, k)),
                gsl_vector_get(matrix_DtraD_eval, k));
    }
    fclose(fp);

    //  for(k=0; k<m; k++)
    //    printf("Mode %ld eigenvalue = %g\n", k, gsl_vector_get(matrix_DtraD_eval,k));
    egvlim      = SVDeps * SVDeps * gsl_vector_get(matrix_DtraD_eval, 0);
    MaxNBmodes1 = MaxNBmodes;
    if(MaxNBmodes1 > m)
    {
        MaxNBmodes1 = m;
    }
    if(MaxNBmodes1 > n)
    {
        MaxNBmodes1 = n;
    }
    mode = 0;
    while((mode < MaxNBmodes1) &&
            (gsl_vector_get(matrix_DtraD_eval, mode) > egvlim))
    {
        mode++;
    }
    printf("Keeping %ld modes  (SVDeps = %g-> %g, MaxNBmodes = %ld -> %ld)\n",
           mode,
           SVDeps,
           egvlim,
           MaxNBmodes,
           MaxNBmodes1);
    MaxNBmodes1 = mode;

    // Write rotation matrix
    arraysizetmp[0] = m;
    arraysizetmp[1] = m;

    FUNC_CHECK_RETURN(create_image_ID(ID_VTmatrix_name,
                                      2,
                                      arraysizetmp,
                                      datatype,
                                      0,
                                      0,
                                      0,
                                      &ID_VTmatrix));

    if(datatype == _DATATYPE_FLOAT)
    {
        for(int ii = 0; ii < m; ii++)   // modes
            for(int k = 0; k < m; k++)  // modes
            {
                data.image[ID_VTmatrix].array.F[k * m + ii] =
                    (float) gsl_matrix_get(matrix_DtraD_evec, k, ii);
            }
    }
    else
    {
        for(int ii = 0; ii < m; ii++)   // modes
            for(int k = 0; k < m; k++)  // modes
            {
                data.image[ID_VTmatrix].array.D[k * m + ii] =
                    gsl_matrix_get(matrix_DtraD_evec, k, ii);
            }
    }

    if(testmode == 1)
    {
        save_fits(ID_VTmatrix_name, "test_VT.fits");
    }

    /* second, build the "inverse" of the diagonal matrix of eigenvalues (matrix1) */
    nbmodesremoved = 0;
    matrix1        = gsl_matrix_alloc(m, m);
    for(int ii1 = 0; ii1 < m; ii1++)  // mode
        for(int jj1 = 0; jj1 < m; jj1++)
        {
            if(ii1 == jj1)
            {
                if(ii1 > MaxNBmodes1 - 1)
                {
                    gsl_matrix_set(matrix1, ii1, jj1, 0.0);
                    nbmodesremoved++;
                }
                else
                {
                    gsl_matrix_set(matrix1,
                                   ii1,
                                   jj1,
                                   1.0 /
                                   gsl_vector_get(matrix_DtraD_eval, ii1));
                }
            }
            else
            {
                gsl_matrix_set(matrix1, ii1, jj1, 0.0);
            }
        }
    // printf("%ld modes removed\n", nbmodesremoved);
    // printf("Compute inverse\n");
    // fflush(stdout);


    if(timing == 1)
    {
        clock_gettime(CLOCK_MILK, &t5);
    }

    /* third, compute the "inverse" of DtraD */
    matrix2 = gsl_matrix_alloc(m, m);
    gsl_blas_dgemm(CblasNoTrans,
                   CblasNoTrans,
                   1.0,
                   matrix_DtraD_evec,
                   matrix1,
                   0.0,
                   matrix2);
    gsl_blas_dgemm(CblasNoTrans,
                   CblasTrans,
                   1.0,
                   matrix2,
                   matrix_DtraD_evec,
                   0.0,
                   matrix_DtraDinv);
    gsl_matrix_free(matrix1);
    gsl_matrix_free(matrix2);

    if(testmode == 1)
    {
        FUNC_CHECK_RETURN(create_2Dimage_ID("M2", m, m, &ID));

        for(int ii = 0; ii < m; ii++)
            for(int jj = 0; jj < m; jj++)
            {
                data.image[ID].array.F[jj * m + ii] =
                    gsl_matrix_get(matrix_DtraDinv, ii, jj);
            }
        save_fits("M2", "test_M2.fits");
    }

    gsl_blas_dgemm(CblasNoTrans,
                   CblasTrans,
                   1.0,
                   matrix_DtraDinv,
                   matrix_D,
                   0.0,
                   matrix_Ds);

    if(data.image[ID_Rmatrix].md[0].naxis == 3)
    {
        arraysizetmp[0] = data.image[ID_Rmatrix].md[0].size[0];
        arraysizetmp[1] = data.image[ID_Rmatrix].md[0].size[1];
        arraysizetmp[2] = m;
    }
    else
    {
        arraysizetmp[0] = n;
        arraysizetmp[1] = m;
    }

    FUNC_CHECK_RETURN(create_image_ID(ID_Cmatrix_name,
                                      data.image[ID_Rmatrix].md[0].naxis,
                                      arraysizetmp,
                                      datatype,
                                      0,
                                      0,
                                      0,
                                      &ID_Cmatrix));

    if(timing == 1)
    {
        clock_gettime(CLOCK_MILK, &t6);
    }

    /* write result */
    if(datatype == _DATATYPE_FLOAT)
    {
        for(int ii = 0; ii < n; ii++)   // sensors
            for(int k = 0; k < m; k++)  // actuator modes
            {
                data.image[ID_Cmatrix].array.F[k * n + ii] =
                    (float) gsl_matrix_get(matrix_Ds, k, ii);
            }
    }
    else
    {
        for(int ii = 0; ii < n; ii++)   // sensors
            for(int k = 0; k < m; k++)  // actuator modes
            {
                data.image[ID_Cmatrix].array.D[k * n + ii] =
                    gsl_matrix_get(matrix_Ds, k, ii);
            }
    }

    if(testmode == 1)
    {
        save_fits(ID_Cmatrix_name, "test_Ainv.fits");
    }

    if(timing == 1)
    {
        clock_gettime(CLOCK_MILK, &t7);
    }

    gsl_vector_free(matrix_DtraD_eval);
    gsl_matrix_free(matrix_D);
    gsl_matrix_free(matrix_Ds);
    gsl_matrix_free(matrix_Dtra);
    gsl_matrix_free(matrix_DtraD);
    gsl_matrix_free(matrix_DtraDinv);
    gsl_matrix_free(matrix_DtraD_evec);

    free(arraysizetmp);

    printf("[CPU pseudo-inverse done]\n");
    fflush(stdout);

    if(timing == 1)
    {
        tdiff = timespec_diff(t0, t1);
        t01d  = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

        tdiff = timespec_diff(t1, t2);
        t12d  = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

        tdiff = timespec_diff(t2, t3);
        t23d  = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

        tdiff = timespec_diff(t3, t4);
        t34d  = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

        tdiff = timespec_diff(t4, t5);
        t45d  = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

        tdiff = timespec_diff(t5, t6);
        t56d  = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

        tdiff = timespec_diff(t6, t7);
        t67d  = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

        printf("Timing info: \n");
        printf("  0-1	%12.3f ms\n", t01d * 1000.0);
        printf("  1-2	%12.3f ms\n", t12d * 1000.0);
        printf("  2-3	%12.3f ms\n", t23d * 1000.0);
        printf("  3-4	%12.3f ms\n", t34d * 1000.0);
        printf("  4-5	%12.3f ms\n", t45d * 1000.0);
        printf("  5-6	%12.3f ms\n", t56d * 1000.0);
        printf("  6-7	%12.3f ms\n", t67d * 1000.0);
    }

    if(outID != NULL)
    {
        *outID = ID_Cmatrix;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    if(*useGPU == 0)
    {
        printf("==== CPU =====\n");
        linopt_compute_SVDpseudoInverse(inimname,
                                        outimname,
                                        *SVD_epsilon,
                                        *max_NBmodes,
                                        outimVTmatname,
                                        NULL);
    }
    else
    {
        printf("==== GPU =====\n");
#ifdef HAVE_MAGMA
        LINALGEBRA_magma_compute_SVDpseudoInverse(inimname,
                                                outimname,
                                                *SVD_epsilon,
                                                *max_NBmodes,
                                                outimVTmatname,
                                                0,
                                                1,
                                                64,
                                                0, // GPU device
                                                NULL);
#endif
    }

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t
CLIADDCMD_linopt_imtools__compute_SVDpseudoinverse()
{
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
