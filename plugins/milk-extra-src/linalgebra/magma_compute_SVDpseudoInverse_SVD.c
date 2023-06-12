/** @file magma_compute_SVDpseudoInverse_SVD.c
 */

#ifdef HAVE_CUDA

#ifdef HAVE_MAGMA
#include "magma_lapack.h"
#include "magma_v2.h"

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

// ==========================================
// Forward declaration(s)
// ==========================================

int LINALGEBRA_magma_compute_SVDpseudoInverse_SVD(const char *ID_Rmatrix_name,
        const char *ID_Cmatrix_name,
        double      SVDeps,
        long        MaxNBmodes,
        const char *ID_VTmatrix_name);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

errno_t LINALGEBRA_magma_compute_SVDpseudoInverse_SVD_cli()
{
    if(CLI_checkarg(1, 4) + CLI_checkarg(2, 3) + CLI_checkarg(3, 1) +
            CLI_checkarg(4, 2) + CLI_checkarg(5, 3) ==
            0)
    {
        LINALGEBRA_magma_compute_SVDpseudoInverse_SVD(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.string,
            data.cmdargtoken[3].val.numf,
            data.cmdargtoken[4].val.numl,
            data.cmdargtoken[5].val.string);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

// ==========================================
// Register CLI command(s)
// ==========================================

errno_t magma_compute_SVDpseudoInverse_SVD_addCLIcmd()
{

    RegisterCLIcommand(
        "linalgebrapsinvSVD",
        __FILE__,
        LINALGEBRA_magma_compute_SVDpseudoInverse_SVD_cli,
        "compute pseudo inverse with direct SVD",
        "<input matrix [string]> <output pseudoinv [string]> <eps [float]> "
        "<NBmodes [long]> <VTmat [string]>",
        "linalgebrapsinvSVD matA matAinv 0.01 100 VTmat",
        "int LINALGEBRA_magma_compute_SVDpseudoInverse_SVD(const char "
        "*ID_Rmatrix_name, const char *ID_Cmatrix_name, "
        "double SVDeps, long MaxNBmodes, const char *ID_VTmatrix_name);");

    return RETURN_SUCCESS;
}

//
// Computes control matrix
// Conventions:
//   m: number of actuators
//   n: number of sensors
int LINALGEBRA_magma_compute_SVDpseudoInverse_SVD(
    const char *ID_Rmatrix_name,
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

    ID_Rmatrix = image_ID(ID_Rmatrix_name);
    datatype   = data.image[ID_Rmatrix].md[0].datatype;

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
    if(datatype == _DATATYPE_FLOAT)
    {
        for(k = 0; k < m; k++)
            for(ii = 0; ii < n; ii++)
            {
                h_R[k * n + ii] = data.image[ID_Rmatrix].array.F[k * n + ii];
            }
    }
    else
    {
        for(k = 0; k < m; k++)
            for(ii = 0; ii < n; ii++)
            {
                h_R[k * n + ii] = data.image[ID_Rmatrix].array.D[k * n + ii];
            }
    }

    //printf("M = %ld   N = %ld\n", (long) M, (long) N);
    //printf("=============== lwork = %ld\n", (long) lwork);
    gpu_time = magma_wtime();
    magma_sgesvd(MagmaSomeVec,
                 MagmaAllVec,
                 M,
                 N,
                 h_R,
                 lda,
                 S1,
                 U,
                 ldu,
                 VT,
                 ldv,
                 h_work,
                 lwork,
                 &info);
    gpu_time = magma_wtime() - gpu_time;
    if(info != 0)
    {
        printf("magma_sgesvd returned error %d: %s.\n",
               (int) info,
               magma_strerror(info));
    }

    //printf("sgesvd gpu time: %7.5f\n", gpu_time );

    // Write eigenvalues
    sprintf(fname, "eigenv.dat.magma");
    if((fp = fopen(fname, "w")) == NULL)
    {
        printf("ERROR: cannot create file \"%s\"\n", fname);
        exit(0);
    }
    for(k = 0; k < min_mn; k++)
    {
        fprintf(fp, "%5ld %20g %20g\n", k, S1[k], S1[k] / S1[0]);
    }
    fclose(fp);

    egvlim = SVDeps * S1[0];

    MaxNBmodes1 = MaxNBmodes;
    if(MaxNBmodes1 > M)
    {
        MaxNBmodes1 = M;
    }
    if(MaxNBmodes1 > N)
    {
        MaxNBmodes1 = N;
    }
    mode = 0;
    while((mode < MaxNBmodes1) && (S1[mode] > egvlim))
    {
        mode++;
    }
    MaxNBmodes1 = mode;

    //printf("Keeping %ld modes  (SVDeps = %g)\n", MaxNBmodes1, SVDeps);
    // Write rotation matrix
    arraysizetmp[0] = m;
    arraysizetmp[1] = m;

    create_image_ID(ID_VTmatrix_name,
                    2,
                    arraysizetmp,
                    _DATATYPE_FLOAT,
                    0,
                    0,
                    0,
                    &ID_VTmatrix);

    if(datatype == _DATATYPE_FLOAT)
    {
        for(ii = 0; ii < m; ii++)   // modes
            for(k = 0; k < m; k++)  // modes
            {
                data.image[ID_VTmatrix].array.F[k * m + ii] =
                    (float) VT[k * m + ii];
            }
    }
    else
    {
        for(ii = 0; ii < m; ii++)   // modes
            for(k = 0; k < m; k++)  // modes
            {
                data.image[ID_VTmatrix].array.D[k * m + ii] =
                    (double) VT[k * m + ii];
            }
    }

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

    create_image_ID(ID_Cmatrix_name,
                    data.image[ID_Rmatrix].md[0].naxis,
                    arraysizetmp,
                    datatype,
                    0,
                    0,
                    0,
                    &ID_Cmatrix);

    // compute pseudo-inverse
    // M+ = V Sig^-1 UT
    for(ii = 0; ii < M; ii++)
        for(jj = 0; jj < N; jj++)
            for(mode = 0; mode < MaxNBmodes1 - 1; mode++)
            {
                data.image[ID_Cmatrix].array.F[jj * M + ii] +=
                    VT[jj * N + mode] * U[mode * M + ii] / S1[mode];
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

#endif
