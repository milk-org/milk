#include "CommandLineInterface/CLIcore.h"

#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_matrix.h>

// Local variables pointers
static char *imcinname;
static char *outimname;
static char *outcoeffname;

static CLICMDARGDEF farg[] = {{
        CLIARG_IMG,
        ".inc",
        "input 3D cube",
        "imc",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &imcinname,
        NULL
    },
    {
        CLIARG_STR,
        ".outm",
        "output modes",
        "outm",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimname,
        NULL
    },
    {
        CLIARG_STR,
        ".outcoeff",
        "output coeffs",
        "outcoeff",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outcoeffname,
        NULL
    }
};

static CLICMDDATA CLIcmddata =
{
    "imsvd", "Singular values decomposition", CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}




// rotation matrix written as SVD_VTm

errno_t linopt_compute_SVDdecomp(const char *IDin_name,
                                 const char *IDout_name,
                                 const char *IDcoeff_name,
                                 imageID    *outID)
{
    DEBUG_TRACE_FSTART();

    imageID                    IDin;
    imageID                    IDout;
    imageID                    IDcoeff;
    gsl_matrix                *matrix_D; /* input */
    gsl_matrix                *matrix_Dtra;
    gsl_matrix                *matrix_DtraD;
    gsl_matrix                *matrix_DtraD_evec;
    gsl_vector                *matrix_DtraD_eval;
    gsl_eigen_symmv_workspace *w;
    gsl_matrix                *matrix_save;

    long      m;
    long      n;
    uint32_t *arraysizetmp;

    imageID ID_VTmatrix;

    arraysizetmp = (uint32_t *) malloc(sizeof(uint32_t) * 3);
    if(arraysizetmp == NULL)
    {
        FUNC_RETURN_FAILURE("malloc returns NULL pointer");
    }

    printf("[SVD start]");
    fflush(stdout);

    IDin = image_ID(IDin_name);

    n = data.image[IDin].md[0].size[0] * data.image[IDin].md[0].size[1];
    m = data.image[IDin].md[0].size[2];

    matrix_DtraD_eval = gsl_vector_alloc(m);
    matrix_D          = gsl_matrix_alloc(n, m);
    matrix_Dtra       = gsl_matrix_alloc(m, n);
    matrix_DtraD      = gsl_matrix_alloc(m, m);
    matrix_DtraD_evec = gsl_matrix_alloc(m, m);

    /* write matrix_D */
    for(long k = 0; k < m; k++)
    {
        for(long ii = 0; ii < n; ii++)
        {
            gsl_matrix_set(matrix_D,
                           ii,
                           k,
                           data.image[IDin].array.F[k * n + ii]);
        }
    }
    /* compute DtraD */
    gsl_blas_dgemm(CblasTrans,
                   CblasNoTrans,
                   1.0,
                   matrix_D,
                   matrix_D,
                   0.0,
                   matrix_DtraD);

    /* compute the inverse of DtraD */

    /* first, compute the eigenvalues and eigenvectors */
    w           = gsl_eigen_symmv_alloc(m);
    matrix_save = gsl_matrix_alloc(m, m);
    gsl_matrix_memcpy(matrix_save, matrix_DtraD);
    gsl_eigen_symmv(matrix_save, matrix_DtraD_eval, matrix_DtraD_evec, w);

    gsl_matrix_free(matrix_save);
    gsl_eigen_symmv_free(w);
    gsl_eigen_symmv_sort(matrix_DtraD_eval,
                         matrix_DtraD_evec,
                         GSL_EIGEN_SORT_ABS_DESC);

    create_2Dimage_ID(IDcoeff_name, m, 1, &IDcoeff);

    for(long k = 0; k < m; k++)
    {
        data.image[IDcoeff].array.F[k] = gsl_vector_get(matrix_DtraD_eval, k);
    }

    /** Write rotation matrix to go from DM modes to eigenmodes */
    arraysizetmp[0] = m;
    arraysizetmp[1] = m;
    ID_VTmatrix     = image_ID("SVD_VTm");
    if(ID_VTmatrix != -1)
    {
        delete_image_ID("SVD_VTm", DELETE_IMAGE_ERRMODE_WARNING);
    }
    create_image_ID("SVD_VTm",
                    2,
                    arraysizetmp,
                    _DATATYPE_FLOAT,
                    0,
                    0,
                    0,
                    &ID_VTmatrix);
    for(long ii = 0; ii < m; ii++)   // modes
        for(long k = 0; k < m; k++)  // modes
        {
            data.image[ID_VTmatrix].array.F[k * m + ii] =
                (float) gsl_matrix_get(matrix_DtraD_evec, k, ii);
        }

    /// Compute SVD decomp

    FUNC_CHECK_RETURN(create_3Dimage_ID(IDout_name,
                                        data.image[IDin].md[0].size[0],
                                        data.image[IDin].md[0].size[1],
                                        data.image[IDin].md[0].size[2],
                                        &IDout));

    for(long kk = 0; kk < m; kk++)  /// eigen mode index
    {
        //        printf("eigenmode %4ld / %4ld  %g\n", kk, m, data.image[IDcoeff].array.F[kk]);
        //       fflush(stdout);
        for(long kk1 = 0; kk1 < m; kk1++)
        {
            for(long ii = 0; ii < n; ii++)
            {
                data.image[IDout].array.F[kk * n + ii] +=
                    data.image[ID_VTmatrix].array.F[kk1 * m + kk] *
                    data.image[IDin].array.F[kk1 * n + ii];
            }
        }
    }

    //   delete_image_ID("SVD_VTm");

    free(arraysizetmp);

    gsl_matrix_free(matrix_D);
    gsl_matrix_free(matrix_Dtra);
    gsl_matrix_free(matrix_DtraD);
    gsl_matrix_free(matrix_DtraD_evec);
    gsl_vector_free(matrix_DtraD_eval);

    printf("[SVD done]\n");
    fflush(stdout);

    if(outID != NULL)
    {
        *outID = IDout;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    linopt_compute_SVDdecomp(imcinname, outimname, outcoeffname, NULL);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions

errno_t
CLIADDCMD_linopt_imtools__compute_SVDdecomp()
{
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
