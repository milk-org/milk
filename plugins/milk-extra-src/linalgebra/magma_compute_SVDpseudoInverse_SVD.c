/**
 * @file magma_compute_SVDpseudoInverse_SVD.c
 * @brief Magma compute svdpseudoinverse svd module
 */

/** @file magma_compute_SVDpseudoInverse_SVD.c
 */

// MILK_CMAKE_MANDATE_CUDA
// MILK_CMAKE_REQUEST_MAGMA

#ifdef HAVE_MAGMA
#    include "magma_lapack.h"
#    include "magma_v2.h"

#    include <stdio.h>
#    include <stdlib.h>
#    include <string.h>
#    include <math.h>
#    include <stdint.h>
#    include <stdbool.h>
#    include <unistd.h>

#    ifdef MILK_NO_CLI
#        include "CLIcore_standalone.h"
#    else
#        include "libmilkdata/milkdata.h"
#        include "milkDebugTools.h"
#        include "fps.h"
#        include "ImageStreamIO/ImageStreamIO.h"
#    endif
#    include "COREMOD_memory/COREMOD_memory.h"

#    ifndef max
#        define max(a, b) ((a) > (b) ? (a) : (b))
#    endif
#    ifndef min
#        define min(a, b) ((a) < (b) ? (a) : (b))
#    endif

// ==========================================
// Forward declaration(s)
// ==========================================

int LINALGEBRA_magma_compute_SVDpseudoInverse_SVD(const char *ID_Rmatrix_name,
                                                  const char *ID_Cmatrix_name,
                                                  double      SVDeps,
                                                  long        MaxNBmodes,
                                                  const char *ID_VTmatrix_name);

// ==========================================
// Gen 4 V2 CLI command: linalgebrapsinvSVD
// ==========================================

static char         ps_r[FUNCTION_PARAMETER_STRMAXLEN]  = "matA";
static char         ps_c[FUNCTION_PARAMETER_STRMAXLEN]  = "matAinv";
static double       ps_eps                              = 0.01;
static int64_t      ps_nm                               = 100;
static char         ps_vt[FUNCTION_PARAMETER_STRMAXLEN] = "VTmat";
static FPS_APP_INFO FPS_app_info_ps = { .fps_name    = "linalgebrapsinvSVD",
                                        .cmdkey      = "linalgebrapsinvSVD",
                                        .description = "pseudo inverse via direct SVD",
                                        .description_long =
                                            "Compute SVD for pseudo-inverse using MAGMA. Low-level "
                                            "SVD computation step with GPU acceleration." };
#    define FPS_PARAMS_PS(X)                                                       \
        X(".inmat", ps_r, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input mat") \
        X(".outmat", ps_c, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output")       \
        X(".svdeps", &ps_eps, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "SVD eps")  \
        X(".nbmodes", &ps_nm, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "max modes")  \
        X(".vtmat", ps_vt, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "VT matrix")
#    include "fps.h"
static FPS_CLI_BINDING                   ps_b[]     = { FPS_PARAMS_PS(FPS_X_BINDING) };
static const int                         ps_nb      = sizeof(ps_b) / sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF                      farg[]     = { FPS_PARAMS_PS(FPS_X_FARG) };
static CLICMDDATA                        CLIcmddata = { "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS                       ps_cms     = { 0 };
static __attribute__((constructor)) void init_ps(void)
{
    strncpy(CLIcmddata.key, FPS_app_info_ps.cmdkey, sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description, FPS_app_info_ps.description,
            sizeof(CLIcmddata.description) - 1);
    CLIcmddata.nbarg         = sizeof(farg) / sizeof(CLICMDARGDEF);
    CLIcmddata.funcfpscliarg = farg;
    CLIcmddata.flags         = CLICMDFLAG_FPS;
    if (!CLIcmddata.cmdsettings)
    {
        CLIcmddata.cmdsettings = &ps_cms;
    }
}
static errno_t ps_compute(void)
{
    LINALGEBRA_magma_compute_SVDpseudoInverse_SVD(ps_r, ps_c, ps_eps, (long) ps_nm, ps_vt);
    return RETURN_SUCCESS;
}
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info_ps, farg, &CLIcmddata, ps_b, ps_nb,
                                        ps_compute);
}

errno_t magma_compute_SVDpseudoInverse_SVD_addCLIcmd()
{
    safe_fps_fill_farg_examples(farg, ps_b, ps_nb);
    INSERT_STD_CLIREGISTERFUNC;

    return RETURN_SUCCESS;
}

//
// Computes control matrix
// Conventions:
//   m: number of actuators
//   n: number of sensors
int LINALGEBRA_magma_compute_SVDpseudoInverse_SVD(const char *ID_Rmatrix_name,
                                                  const char *ID_Cmatrix_name,
                                                  double      SVDeps,
                                                  long        MaxNBmodes,
                                                  const char *ID_VTmatrix_name)
{
    uint32_t   *arraysizetmp;
    magma_int_t M, N, min_mn;
    long        m, n, ii, jj, k;
    long        ID_Rmatrix;
    long        ID_Cmatrix;
    uint8_t     datatype;

    magma_int_t lda, ldu, ldv;
    //float dummy[1];
    float      *a, *h_R; // a, h_R - mxn  matrices
    float      *U, *VT;  // u - mxm matrix , vt - nxn  matrix  on the  host
    float      *S1;      //  vectors  of  singular  values
    magma_int_t info;
    //float  work[1];				// used in  difference  computations
    float        *h_work; //  h_work  - workspace
    magma_int_t   lwork;  //  workspace  size
    real_Double_t gpu_time;
    //real_Double_t cpu_time;

    FILE *fp;
    char  fname[200];
    long  ID_VTmatrix;
    float egvlim;
    long  MaxNBmodes1, mode;

    arraysizetmp = (uint32_t *) malloc(sizeof(uint32_t) * 3);

    ID_Rmatrix = image_ID(ID_Rmatrix_name, dcimg, dcnimg);
    datatype   = dcimg[ID_Rmatrix].md[0].datatype;

    if (dcimg[ID_Rmatrix].md[0].naxis == 3)
    {
        n = dcimg[ID_Rmatrix].md[0].size[0] * dcimg[ID_Rmatrix].md[0].size[1];
        m = dcimg[ID_Rmatrix].md[0].size[2];
        printf("3D image -> %ld %ld\n", n, m);
        fflush(stdout);
    }
    else
    {
        n = dcimg[ID_Rmatrix].md[0].size[0];
        m = dcimg[ID_Rmatrix].md[0].size[1];
        printf("2D image -> %ld %ld\n", n, m);
        fflush(stdout);
    }

    M = n;
    N = m;

    lda = M;
    ldu = M;
    ldv = N;

    min_mn = min(M, N);

    //printf("INITIALIZE MAGMA\n");
    //fflush(stdout);

    /* in this procedure, m=number of actuators/modes, n=number of WFS elements */
    //   printf("magma :    M = %ld , N = %ld\n", (long) M, (long) N);
    //fflush(stdout);

    magma_init(); // initialize Magma
    //  Allocate  host  memory
    magma_smalloc_cpu(&a, lda * N);             // host  memory  for a
    magma_smalloc_cpu(&VT, ldv * N);            // host  memory  for vt
    magma_smalloc_cpu(&U, M * M);               // host  memory  for u
    magma_smalloc_cpu(&S1, min_mn);             // host  memory  for s1
    magma_smalloc_pinned(&h_R, lda * N);        // host  memory  for r
    magma_int_t nb = magma_get_sgesvd_nb(M, N); // opt. block  size
    lwork          = (M + N) * nb + 3 * min_mn;
    magma_smalloc_pinned(&h_work, lwork); // host  mem. for  h_work

    // write input h_R matrix
    if (datatype == _DATATYPE_FLOAT)
    {
        for (k = 0; k < m; k++)
        {
            for (ii = 0; ii < n; ii++)
            {
                h_R[k * n + ii] = dcimg[ID_Rmatrix].array.F[k * n + ii];
            }
        }
    }
    else
    {
        for (k = 0; k < m; k++)
        {
            for (ii = 0; ii < n; ii++)
            {
                h_R[k * n + ii] = dcimg[ID_Rmatrix].array.D[k * n + ii];
            }
        }
    }

    //printf("M = %ld   N = %ld\n", (long) M, (long) N);
    //printf("=============== lwork = %ld\n", (long) lwork);
    gpu_time = magma_wtime();
    magma_sgesvd(MagmaSomeVec, MagmaAllVec, M, N, h_R, lda, S1, U, ldu, VT, ldv, h_work, lwork,
                 &info);
    gpu_time = magma_wtime() - gpu_time;
    if (info != 0)
    {
        printf("magma_sgesvd returned error %d: %s.\n", (int) info, magma_strerror(info));
    }

    //printf("sgesvd gpu time: %7.5f\n", gpu_time );

    // Write eigenvalues
    snprintf(fname, sizeof(fname),\n "eigenv.dat.magma");
    if ((fp = fopen(fname, "w")) == NULL)
    {
        printf("ERROR: cannot create file \"%s\"\n", fname);
        exit(0);
    }
    for (k = 0; k < min_mn; k++)
    {
        fprintf(fp, "%5ld %20g %20g\n", k, S1[k], S1[k] / S1[0]);
    }
    fclose(fp);

    egvlim = SVDeps * S1[0];

    MaxNBmodes1 = MaxNBmodes;
    if (MaxNBmodes1 > M)
    {
        MaxNBmodes1 = M;
    }
    if (MaxNBmodes1 > N)
    {
        MaxNBmodes1 = N;
    }
    mode = 0;
    while ((mode < MaxNBmodes1) && (S1[mode] > egvlim))
    {
        mode++;
    }
    MaxNBmodes1 = mode;

    //printf("Keeping %ld modes  (SVDeps = %g)\n", MaxNBmodes1, SVDeps);
    // Write rotation matrix
    arraysizetmp[0] = m;
    arraysizetmp[1] = m;

    {
        IMGID imgvt         = imgid_make_from_name(ID_VTmatrix_name);
        imgvt.mdt->naxis    = 2;
        imgvt.mdt->size[0]  = arraysizetmp[0];
        imgvt.mdt->size[1]  = arraysizetmp[1];
        imgvt.mdt->datatype = _DATATYPE_FLOAT;
        imgvt.im            = (IMAGE *) calloc(1, sizeof(IMAGE));
        imgid_mkimage(&imgvt);
        ID_VTmatrix = imgvt.ID;
    }

    if (datatype == _DATATYPE_FLOAT)
    {
        for (ii = 0; ii < m; ii++) // modes
        {
            for (k = 0; k < m; k++) // modes
            {
                dcimg[ID_VTmatrix].array.F[k * m + ii] = (float) VT[k * m + ii];
            }
        }
    }
    else
    {
        for (ii = 0; ii < m; ii++) // modes
        {
            for (k = 0; k < m; k++) // modes
            {
                dcimg[ID_VTmatrix].array.D[k * m + ii] = (double) VT[k * m + ii];
            }
        }
    }

    if (dcimg[ID_Rmatrix].md[0].naxis == 3)
    {
        arraysizetmp[0] = dcimg[ID_Rmatrix].md[0].size[0];
        arraysizetmp[1] = dcimg[ID_Rmatrix].md[0].size[1];
        arraysizetmp[2] = m;
    }
    else
    {
        arraysizetmp[0] = n;
        arraysizetmp[1] = m;
    }

    {
        IMGID imgcm      = imgid_make_from_name(ID_Cmatrix_name);
        imgcm.mdt->naxis = dcimg[ID_Rmatrix].md[0].naxis;
        for (int a = 0; a < imgcm.mdt->naxis; a++)
        {
            imgcm.mdt->size[a] = arraysizetmp[a];
        }
        imgcm.mdt->datatype = datatype;
        imgcm.im            = (IMAGE *) calloc(1, sizeof(IMAGE));
        imgid_mkimage(&imgcm);
        ID_Cmatrix = imgcm.ID;
    }

    // compute pseudo-inverse
    // M+ = V Sig^-1 UT
    for (ii = 0; ii < M; ii++)
    {
        for (jj = 0; jj < N; jj++)
        {
            for (mode = 0; mode < MaxNBmodes1 - 1; mode++)
            {
                dcimg[ID_Cmatrix].array.F[jj * M + ii] +=
                    VT[jj * N + mode] * U[mode * M + ii] / S1[mode];
            }
        }
    }

    magma_free_cpu(a);         // free  host  memory
    magma_free_cpu(VT);        // free  host  memory
    magma_free_cpu(S1);        // free  host  memory
    magma_free_cpu(U);         // free  host  memory
    magma_free_pinned(h_work); // free  host  memory
    magma_free_pinned(h_R);    // free  host  memory

    magma_finalize(); //  finalize  Magma

    free(arraysizetmp);

    //    printf("[CM magma SVD done]\n");
    //   fflush(stdout);

    return (ID_Cmatrix);
}

#endif
