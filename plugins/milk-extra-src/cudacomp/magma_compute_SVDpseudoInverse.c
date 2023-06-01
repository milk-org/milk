/** @file magma_compute_SVDpseudoInverse.c
 */

#ifdef HAVE_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <device_types.h>
#include <pthread.h>

#ifdef HAVE_MAGMA

#include "magma_lapack.h"
#include "magma_v2.h"

#include "CommandLineInterface/CLIcore.h"
#include "CommandLineInterface/timeutils.h"

#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"

#include "cudacomp_types.h"

extern int INIT_MAGMA;

// queue for default magma device
extern magma_queue_t magmaqueue;

static long MAGMAloop_iter = 0;

/*
static double *magma_h_A;
static double *magma_d_A;
static double *magma_d_AtA;
static double *magma_h_AtA;
static double *magma_w1; // eigenvalues
static double *magma_h_R;
static double *magma_h_work;
static double *magma_d_VT1;
static double *magma_h_VT1;
static double *magma_d_M2;
static double *magma_d_Ainv;
static double *magma_h_Ainv;
static double *magma_h_M2;
//static double *magma_h_S; //singular values
//static double *magma_d_U; //left singular vectors
//static double *magma_d_VT; //right singular vectors
//static double *magma_d_B;


static float *magmaf_h_A;
static float *magmaf_d_A;
static float *magmaf_d_AtA;
static float *magmaf_h_AtA;
static float *magmaf_w1; // eigenvalues
static float *magmaf_h_R;
static float *magmaf_h_work;
static float *magmaf_d_VT1;
static float *magmaf_h_VT1;
static float *magmaf_d_M2;
static float *magmaf_d_Ainv;
static float *magmaf_h_Ainv;
static float *magmaf_h_M2;
//static float *magmaf_h_S; //singular values
//static float *magmaf_d_U; //left singular vectors
//static float *magmaf_d_VT; //right singular vectors
//static float *magmaf_d_B;
*/

static magma_int_t  magma_aux_iwork[1];
static magma_int_t  magma_lwork, magma_liwork;
static magma_int_t *magma_iwork;

// ==========================================
// Forward declaration(s)
// ==========================================

errno_t CUDACOMP_magma_compute_SVDpseudoInverse(const char *ID_Rmatrix_name,
        const char *ID_Cmatrix_name,
        double      SVDeps,
        long        MaxNBmodes,
        const char *ID_VTmatrix_name,
        int         LOOPmode,
        int         testmode,
        int         precision,
        int         GPUdevice,
        imageID    *outID);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t CUDACOMP_magma_compute_SVDpseudoInverse_cli()
{
    if(CLI_checkarg(1, 4) + CLI_checkarg(2, 3) + CLI_checkarg(3, 1) +
            CLI_checkarg(4, 2) + CLI_checkarg(5, 3) + CLI_checkarg(6, 2) +
            CLI_checkarg(7, 1) + CLI_checkarg(8, 1) ==
            0)
    {
        CUDACOMP_magma_compute_SVDpseudoInverse(data.cmdargtoken[1].val.string,
                                                data.cmdargtoken[2].val.string,
                                                data.cmdargtoken[3].val.numf,
                                                data.cmdargtoken[4].val.numl,
                                                data.cmdargtoken[5].val.string,
                                                0,
                                                0,
                                                64,
                                                0,
                                                NULL);

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

errno_t magma_compute_SVDpseudoInverse_addCLIcmd()
{

    RegisterCLIcommand("cudacomppsinv",
                       __FILE__,
                       CUDACOMP_magma_compute_SVDpseudoInverse_cli,
                       "compute pseudo inverse",
                       "<input matrix [string]> <output pseudoinv [string]> "
                       "<eps [float]> <NBmodes [long]> <VTmat [string]>",
                       "cudacomppsinv matA matAinv 0.01 100 VTmat 0 1e-4 1e-7",
                       "int CUDACOMP_magma_compute_SVDpseudoInverse(const char "
                       "*ID_Rmatrix_name, const char *ID_Cmatrix_name, double "
                       "SVDeps, long MaxNBmodes, const char *ID_VTmatrix_name, "
                       "int LOOPmode, int PSINV_MODE, double qdwh_s, float "
                       "qdwh_tol)");

    return RETURN_SUCCESS;
}

/**
 *  @brief Computes matrix pseudo-inverse (AT A)^-1 AT, using eigenvector/eigenvalue decomposition of AT A
 *
 *
 * Computes pseuso inverse of a matrix.\n
 * Column-major representation used to match magma and lapack.\n
 * When viewed as an image, matrix leading dimension is size[0] = horizontal axis. When viewed in an image viewer, the first column is on the bottom side with the first element in bottom left corner, so the matrix appears rotated counter-clockwise by 90deg from its conventional representation where first column is on the left side.\n
 * Returns transpose of pseudoinverse.\n
 *
 *
 *
 * ## Matrix representation details
 *
 * Using column-major indexing\n
 * When viewed as a FITS file, the first matrix column (= vector) appears as the bottom line of the FITS image.\n
 * First matrix element is bottom left corner, second element is immediately to the right of it.
 *
 * Noting elements as a[row,column] = a[i,j], elements are accessed as in memory as:
 * 		a[ j * M + i ]
 *
 * FITS file representation (ds9 view) starts from bottom left corner.
 *
 * 		a[000,N-1] -> a[001,N-1] -> ... -> a[M-1,N-1]
 * 		............................................. ^
 * 		a[000,001] -> a[001,001] -> ... -> a[M-1,001] ^
 * 		a[000,000] -> a[001,000] -> ... -> a[M-1,000] ^     : this is the first matrix row
 *
 * Note that a tall input matrix (M>N) will appear short if viewed as an image.
 * To view the FITS file in the conventional matrix view, rotate by 90 deg clockwise.
 *
 *
 *
 * ## Application Notes
 *
 *  Use LOOPmode = 1 for computing the same size SVD, same input and output location
 *
 * ### Use case: Response matrix to compute control matrix
 *
 * When using function to invert AO response matrix with AOloopControl module, input is 2D or 3D image:
 * 		M: number of sensors    (AO control) =  size[0] (2D) = size[0]*size[1] (3D)
 * 		N: number of actuators  (AO control) =  size[1] (2D) =         size[2] (3D)
 *
 * 	We assume M>N
 *
 *
 * ### Use case: Predictive control
 *
 * When using function to compute pseudo-inverse of data matrix (predictive control), input matrix is a 2D image which is the Transpose of the data matrix.
 *		M: number of measurements samples  = size[0] (2D)
 *		N: dimension of each measurement   = size[1] (2D)
 *
 * We assume M>N
 *
 *
 *
 *
 * ## Algorithm details and main computation steps
 *
 * Notations:
 * 	AT is transpose of A
 * 	A+ is pseudo inverse of A
 *
 *  Computes pseudo-inverse : A+ = (AT A)^-1 AT
 *  Inverse of AT A is computed by SVD
 *
 * SVD:   A = U S V^T
 *   U are eigenvectors of A A^T
 *   V are eigenvectors of A^T A, computed at step 4 below
 *
 * Linear algebra reminder: equivalence between (AT A)^-1 AT and V S^-1 UT
 *
 * Definition of pseudoinverse:
 * A+ = (AT A)^-1 AT
 * singular value decomposition of A = U S VT
 * A+ = ( V S UT U S VT )^-1 V S UT
 * Since U is unitary, UT U = Id ->
 * A+ = ( V S^2 VT )^-1 V S UT
 * A+ = VT^-1 S^-2 V^-1 V S UT
 * A+ = V S^-1 UT
 *
 *  Main steps (non-QDWH):
 *
 *  STEP 1 :   Fill input data into magmaf_h_A on host
 *
 *  STEP 2 :   Copy input data to GPU                                 -> magmaf_d_A        (MxN matrix on device)
 *
 *  STEP 3 :   Compute  trans(A) x A   : magmaf_d_A x magmaf_d_A      -> magmaf_d_AtA      (NxN matrix on device)
 *
 *  STEP 4 :   Compute eigenvalues and eigenvectors of A^T A          -> magmaf_d_AtA      (NxN matrix on device)
 *     Calls magma_ssyevd_gpu :
 *     Compute the eigenvalues and optionally eigenvectors of a symmetric real matrix in single precision, GPU interface, big matrix.
 *     This function computes in single precision all eigenvalues and, optionally, eigenvectors of a real symmetric matrix A defined on the device.
 *     The  first parameter can take the values MagmaVec,'V' or MagmaNoVec,'N' and answers the question whether the eigenvectors are desired.
 *     If the eigenvectors are desired, it uses a divide and conquer algorithm.  The symmetric matrix A can be stored in lower (MagmaLower,'L')
 *     or upper  (MagmaUpper,'U') mode. If the eigenvectors are desired, then on exit A contains orthonormal eigenvectors.
 *     The eigenvalues are stored in an array w
 *
 *  STEP 5 :   Set eigenvalue limit
 *
 *  STEP 6 :   Write eigenvectors to V^T matrix
 *
 *  STEP 7 :   Write eigenvectors/eigenvalue to magma_h_VT1 if eigenvalue > limit
 *           Copy to magma_d_VT1
 *
 *  STEP 8 :   Compute M2 = VT1 VT. M2 is (AT A)^-1
 *
 *  STEP 9 :   Compute Ainv = M2 A. This is the pseudo inverse
 *
 * @note SVDeps^2 is applied as a limit to the eigenvectors of AT A, which are equal to the squares of the singular values of A, so this is equivalent to applying SVDeps as a limit on the singular values of A
 * @note When used to compute AO control matrix, N=number of actuators/modes, M=number of WFS elements
 * @note EIGENVALUES are good to about 1e-6 of peak eigenvalue in single precision, much better with double precision
 * @note 2.5x faster in single precision
 *
 * @note If provided with an additional data matrix named "", an additional Matrix Matrix product between Ainv and the provided matrix will be performed. This feature is used for predictive control and sensor fusion to create a control matrix.
 *
 * TEST MODE OUTPOUT
 *
 * non-QDWH mode:
 *
 * test_mA.fits               content of magmaf_h_A
 * test_mAtA.fits             content of transpose(A) x A = magmaf_d_AtA (output of STEP 3)
 * test_eigenv.dat            list of eigenvalues
 * test_SVDmodes.log          number of singular values kept
 * test_mM2.fits              matrix M2 (output of STEP 8)
 * test_mVT.fits              matrix transpose(V) = eigenvectors (output of step 6)
 * test_mAinv.fits            transpose of pseudoinverse
 * test_AinvA.fits            product of Ainv with A, should be close to identity matrix size NxN
 *
 *
 * QDWH mode:
 *
 * test_mA.QDWH.fits          content of magmaf_h_A
 * test_Aorig.QDWH.txt        content of magmaf_h_A prior to calling psinv function
 * test_sv.QDWH.dat           singular values after call to psinv function
 * test_SVDmodes.QDWH.log     number of singular values kept (note : independent form pseudo-inverse computation)
 * test_mAinv.QDWH.fits       transpose of pseudoinverse
 * test_AinvA.QDWH.fits       product of Ainv with A, should be close to identity matrix size NxN
 */

errno_t CUDACOMP_magma_compute_SVDpseudoInverse(const char *ID_Rmatrix_name,
        const char *ID_Cmatrix_name,
        double      SVDeps,
        long        MaxNBmodes,
        const char *ID_VTmatrix_name,
        int         LOOPmode,
        int         testmode,
        int         precision,
        int         GPUdevice,
        imageID    *outID)
{
    DEBUG_TRACE_FSTART();

    // input identifier
    imageID ID_Rmatrix;
    uint8_t datatype;

    // output result
    imageID ID_Cmatrix;

    // matrix size
    magma_int_t N, M;
    magma_int_t info;

    imageID ID_PFfmdat = -1; // optional final M-M product

    /// Timing tests
    // int timing = 1;                                                        /**< 1 if timing test ON, 0 otherwise */

    // timers
    struct timespec t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13;

    // times in sec
    double t01d, t12d, t23d, t34d, t45d, t56d, t67d, t78d, t89d, t910d, t1011d,
           t1112d, t1213d, t013d;

    long MaxNBmodes1, mode;

    // TESTING FLAGS
    int VERBOSE_CUDACOMP_magma_compute_SVDpseudoInverse = 1;

    // 1 if single precision, 0 if double precision
    int MAGMAfloat = 0;
    if(precision == 32)
    {
        MAGMAfloat = 1;
    }

    int magmaXmode = 0; // expert mode, uses magma_ssyevdx_gpu
    // this does not speed up computation
    magma_int_t mout;

    int dAinvMODE = 0;

    //  if(timing==1)
    clock_gettime(CLOCK_MILK, &t0);

    /**
     *
     *
     * MATRIX REPRESENTATION CONVENTION
     *

     *
     */

    ///
    /// MAGMA uses column-major matrices. For matrix A with dimension (M,N), element A(i,j) is A[ j*M + i]
    /// i = 0 ... M : row index, coefficient of a vector
    /// j = 0 ... N : column index, vector index
    /// M is the matrix leading dimension = lda
    /// M = number of rows
    /// N = number of columns
    /// (assuming here that vector = column of the matrix)
    ///

    ID_Rmatrix = image_ID(ID_Rmatrix_name);
    datatype   = data.image[ID_Rmatrix].md[0].datatype;

    if(data.image[ID_Rmatrix].md[0].naxis == 3)
    {
        /// each column (N=cst) of A is a z=cst slice of image Rmatrix
        M = data.image[ID_Rmatrix].md[0].size[0] *
            data.image[ID_Rmatrix].md[0].size[1];

        N = data.image[ID_Rmatrix].md[0].size[2];

        if(VERBOSE_CUDACOMP_magma_compute_SVDpseudoInverse == 1)
        {
            printf("3D image -> %ld %ld\n", (long) M, (long) N);
            fflush(stdout);
        }
    }
    else
    {
        /// each column (N=cst) of A is a line (y=cst) of Rmatrix (90 deg rotation)
        M = data.image[ID_Rmatrix].md[0].size[0];

        N = data.image[ID_Rmatrix].md[0].size[1];

        if(VERBOSE_CUDACOMP_magma_compute_SVDpseudoInverse == 1)
        {
            printf("2D image -> %ld %ld\n", (long) M, (long) N);
            fflush(stdout);
        }
    }

    //TEST
    //for(ii=0;ii<N;ii++)
    //data.image[ID_Rmatrix].array.F[ii*M+ii] += 1.0;

    if(VERBOSE_CUDACOMP_magma_compute_SVDpseudoInverse == 1)
    {
        printf("magma :    M = %ld , N = %ld\n", (long) M, (long) N);
        fflush(stdout);
    }

    /// Initialize MAGMA if needed
    if(INIT_MAGMA == 0)
    {
        if(VERBOSE_CUDACOMP_magma_compute_SVDpseudoInverse == 1)
        {
            printf("INITIALIZE MAGMA\n");
            fflush(stdout);
        }
        magma_init();
        magma_print_environment();

        INIT_MAGMA = 1;
    }

    printf("Selecting device\n");
    fflush(stdout);
    magma_int_t     num_dev;
    magma_device_t *devicearray =
        (magma_device_t *) malloc(sizeof(magma_device_t) * 10);
    magma_getdevices(devicearray, 10, &num_dev);
    printf("%d devices detected\n", num_dev);

    printf("Selecting device %d\n", GPUdevice);
    magma_setdevice(devicearray[GPUdevice]);

    fflush(stdout);

    double *magma_h_A;
    double *magma_d_A;
    double *magma_d_AtA;
    double *magma_h_AtA;
    double *magma_w1; // eigenvalues
    double *magma_h_R;
    double *magma_h_work;
    double *magma_d_VT1;
    double *magma_h_VT1;
    double *magma_d_M2;
    double *magma_d_Ainv;
    double *magma_h_Ainv;
    double *magma_h_M2;

    float *magmaf_h_A;
    float *magmaf_d_A;
    float *magmaf_d_AtA;
    float *magmaf_h_AtA;
    float *magmaf_w1; // eigenvalues
    float *magmaf_h_R;
    float *magmaf_h_work;
    float *magmaf_d_VT1;
    float *magmaf_h_VT1;
    float *magmaf_d_M2;
    float *magmaf_d_Ainv;
    float *magmaf_h_Ainv;
    float *magmaf_h_M2;


    // =================================================================
    //             MEMORY ALLOCATION
    //
    // (for single precision, magma_ -> magmaf_)
    //
    // ----- QSWHpartial --------
    // magma_h_A
    // magma_d_A
    // magma_h_S
    // magma_d_U
    // magma_d_VT
    // magma_d_B
    //
    // ----- std magma ----------
    // magma_h_A
    // magma_d_A
    // magma_h_AtA
    // magma_d_AtA
    // magma_h_VT1
    // magma_d_VT1
    // magma_d_M2
    //
    // =================================================================

    if(MAGMAloop_iter == 0)  /// memory is only allocated on first pass
    {
        if(MAGMAfloat == 0)  // double
        {
            printf("MAGMA allocating double %d x %d = %ld byte\n",
                   (int) M,
                   (int) N,
                   sizeof(double) * M * N);

            //TESTING_MALLOC_DEV(magma_d_A, M * N);
            if(MAGMA_SUCCESS != magma_dmalloc(&magma_d_A, M * N))
            {
                fprintf(stderr, "!!!! magma_malloc failed\n");
                magma_finalize();
                exit(-1);
            }
            TESTING_DMALLOC_CPU(magma_h_A, M * N);

            TESTING_DMALLOC_CPU(magma_h_AtA, N * N);
            TESTING_DMALLOC_DEV(magma_d_AtA, N * N);

            TESTING_DMALLOC_CPU(magma_h_VT1, N * N);
            TESTING_DMALLOC_DEV(magma_d_VT1, N * N);
            TESTING_DMALLOC_DEV(magma_d_M2, N * N);
        }
        else
        {
            TESTING_SMALLOC_CPU(magmaf_h_A, M * N);
            printf("Allocating magmaf_d_A on device ...\n");
            fflush(stdout);
            TESTING_SMALLOC_DEV(magmaf_d_A, M * N);
            printf(" ... done\n");
            fflush(stdout);

            TESTING_SMALLOC_CPU(magmaf_h_AtA, N * N);
            TESTING_SMALLOC_DEV(magmaf_d_AtA, N * N);

            TESTING_SMALLOC_CPU(magmaf_h_VT1, N * N);
            TESTING_SMALLOC_DEV(magmaf_d_VT1, N * N);
            TESTING_SMALLOC_DEV(magmaf_d_M2, N * N);
        }
    }

    if(MAGMAloop_iter == 0)
    {
        magma_queue_create(devicearray[GPUdevice], &magmaqueue);
    }

    // if(timing==1)
    magma_queue_sync(magmaqueue);
    clock_gettime(CLOCK_MILK, &t1);


    // ****************************************************
    // STEP 1 :   Fill input data into magmaf_h_A on host
    // ****************************************************
    // magma array is column-major.
    //

    if(datatype == _DATATYPE_FLOAT)
    {
        if(MAGMAfloat == 1)
        {
            if((testmode == 1))
            {
                // need magmaf_h_A, otherwise, straight to magmaf_d_A

                memcpy(magmaf_h_A,
                       data.image[ID_Rmatrix].array.F,
                       sizeof(float) * M * N);
                // copy from host to device
                magma_ssetmatrix(M,
                                 N,
                                 magmaf_h_A,
                                 M,
                                 magmaf_d_A,
                                 M,
                                 magmaqueue);
            }
            else
            {
                magma_ssetmatrix(M,
                                 N,
                                 data.image[ID_Rmatrix].array.F,
                                 M,
                                 magmaf_d_A,
                                 M,
                                 magmaqueue);
            }
        }
        else
        {
            for(long ii = 0; ii < M * N; ii++)
            {
                magma_h_A[ii] = data.image[ID_Rmatrix].array.F[ii];
            }

            // copy from host to device
            magma_dsetmatrix(M, N, magma_h_A, M, magma_d_A, M, magmaqueue);
        }
    }
    else
    {
        if(MAGMAfloat == 1)
        {
            for(long ii = 0; ii < M * N; ii++)
            {
                magmaf_h_A[ii] = data.image[ID_Rmatrix].array.D[ii];
            }

            // copy from host to device
            magma_ssetmatrix(M, N, magmaf_h_A, M, magmaf_d_A, M, magmaqueue);
        }
        else
        {
            if(testmode == 1)  // need magma_h_A for testing
            {
                //for(ii=0; ii<M*N; ii++)
                //    magma_h_A[ii] = data.image[ID_Rmatrix].array.D[ii];
                memcpy(magma_h_A,
                       data.image[ID_Rmatrix].array.D,
                       sizeof(double) * M * N);
                // copy from host to device
                magma_dsetmatrix(M, N, magma_h_A, M, magma_d_A, M, magmaqueue);
            }
            else
            {
                magma_dsetmatrix(M,
                                 N,
                                 data.image[ID_Rmatrix].array.D,
                                 M,
                                 magma_d_A,
                                 M,
                                 magmaqueue);
            }
        }
    }


    if(testmode == 1)
    {
        imageID ID_A;

        FUNC_CHECK_RETURN(create_2Dimage_ID("mA", M, N, &ID_A));

        if(MAGMAfloat == 1)
        {
            for(long ii = 0; ii < M * N; ii++)
            {
                data.image[ID_A].array.F[ii] = magmaf_h_A[ii];
            }
        }
        else
        {
            for(long ii = 0; ii < M * N; ii++)
            {
                data.image[ID_A].array.F[ii] = magma_h_A[ii];
            }
        }

        FUNC_CHECK_RETURN(save_fits("mA", "test_mA.QDWH.fits"));

        FUNC_CHECK_RETURN(delete_image_ID("mA", DELETE_IMAGE_ERRMODE_WARNING));
    }

    // ****************************************************
    // STEP 2 :   Copy input data from CPU to GPU
    // ****************************************************


    // copy from host to device
    //

    if(MAGMAloop_iter == 0)
    {
        if(MAGMAfloat == 1)
        {
            TESTING_SMALLOC_CPU(magmaf_h_Ainv, N * M);
        }
        else
        {
            TESTING_DMALLOC_CPU(magma_h_Ainv, N * M);
        }
    }

    {
        // START STD MAGMA ===============================================

        magma_queue_sync(magmaqueue);
        clock_gettime(CLOCK_MILK, &t2);

        // ****************************************************
        // STEP 3 :   Compute trans(A) x A    : magmaf_d_A x magmaf_d_A      -> magmaf_d_AtA      (NxN matrix on device)
        // ****************************************************

        if(MAGMAfloat == 1)
        {
            magma_ssyrk(MagmaLower,
                        MagmaTrans,
                        N,
                        M,
                        1.0,
                        magmaf_d_A,
                        M,
                        0.0,
                        magmaf_d_AtA,
                        N,
                        magmaqueue);
            magmablas_ssymmetrize(MagmaLower, N, magmaf_d_AtA, N, magmaqueue);

            // Slower alternative
            //magma_sgemm(  MagmaTrans, MagmaNoTrans, N, N, M, 1.0, magmaf_d_A, M, magmaf_d_A, M, 0.0,  magmaf_d_AtA, N, magmaqueue);
        }
        else
        {
            magma_dgemm(MagmaTrans,
                        MagmaNoTrans,
                        N,
                        N,
                        M,
                        1.0,
                        magma_d_A,
                        M,
                        magma_d_A,
                        M,
                        0.0,
                        magma_d_AtA,
                        N,
                        magmaqueue);
        }

        if(testmode == 1)
        {
            // copy from GPU to CPU
            if(MAGMAfloat == 1)
            {
                magma_sgetmatrix(N,
                                 N,
                                 magmaf_d_AtA,
                                 N,
                                 magmaf_h_AtA,
                                 N,
                                 magmaqueue);
            }
            else
            {
                magma_dgetmatrix(N,
                                 N,
                                 magma_d_AtA,
                                 N,
                                 magma_h_AtA,
                                 N,
                                 magmaqueue);
            }

            imageID ID_AtA;
            FUNC_CHECK_RETURN(create_2Dimage_ID("mAtA", N, N, &ID_AtA));
            if(MAGMAfloat == 1)
            {
                for(long ii = 0; ii < N * N; ii++)
                {
                    data.image[ID_AtA].array.F[ii] = magmaf_h_AtA[ii];
                }
            }
            else
            {
                for(long ii = 0; ii < N * N; ii++)
                {
                    data.image[ID_AtA].array.F[ii] = magma_h_AtA[ii];
                }
            }
            FUNC_CHECK_RETURN(save_fits("mAtA", "test_mAtA.fits"));
            FUNC_CHECK_RETURN(
                delete_image_ID("mAtA", DELETE_IMAGE_ERRMODE_IGNORE));
        }

        //if(timing==1)
        magma_queue_sync(magmaqueue);
        clock_gettime(CLOCK_MILK, &t3);

        // ****************************************************
        // STEP 4 :   Compute eigenvalues and eigenvectors of AT A   -> magmaf_d_AtA      (NxN matrix on device)
        //
        // SVD of AT A = V S^2 VT
        // calls function magma_ssyevd_gpu
        //
        //
        // ****************************************************


        if(MAGMAloop_iter == 0)
        {
            // get workspace size
            if(MAGMAfloat == 1)
            {
                float auxf_work[1];

                if(magmaXmode == 1)
                {
                    magma_ssyevdx_gpu(MagmaVec,
                                      MagmaRangeI,
                                      MagmaLower,
                                      N,
                                      NULL,
                                      N,
                                      0.0,
                                      1.0,
                                      N - MaxNBmodes,
                                      N,
                                      NULL,
                                      NULL,
                                      NULL,
                                      N,
                                      auxf_work,
                                      -1,
                                      magma_aux_iwork,
                                      -1,
                                      &info);
                }
                else
                {
                    magma_ssyevd_gpu(MagmaVec,
                                     MagmaLower,
                                     N,
                                     NULL,
                                     N,
                                     NULL,
                                     NULL,
                                     N,
                                     auxf_work,
                                     -1,
                                     magma_aux_iwork,
                                     -1,
                                     &info);
                }
                // -> change to 2-stage magma SVD
                // evd -> evr
                // PALSMA

                // alt -> LQ reduction -> SVD magma_dgsvd (more stable numerically)

                magma_lwork = (magma_int_t) MAGMA_S_REAL(auxf_work[0]);
            }
            else
            {
                double aux_work[1];

                magma_dsyevd_gpu(MagmaVec,
                                 MagmaLower,
                                 N,
                                 NULL,
                                 N,
                                 NULL,
                                 NULL,
                                 N,
                                 aux_work,
                                 -1,
                                 magma_aux_iwork,
                                 -1,
                                 &info);
                magma_lwork = (magma_int_t) MAGMA_S_REAL(aux_work[0]);
            }

            magma_liwork = magma_aux_iwork[0];
        }


        if(MAGMAloop_iter == 0)
        {
            if(MAGMAfloat == 1)
            {
                TESTING_MALLOC_CPU(magma_iwork, magma_int_t, magma_liwork);
                TESTING_MALLOC_PIN(magmaf_h_work, float, magma_lwork);
                TESTING_MALLOC_CPU(magmaf_w1, float, N);
                TESTING_MALLOC_PIN(magmaf_h_R, float, N * N);
            }
            else
            {
                TESTING_MALLOC_CPU(magma_iwork, magma_int_t, magma_liwork);
                TESTING_MALLOC_PIN(magma_h_work, double, magma_lwork);
                TESTING_MALLOC_CPU(magma_w1, double, N);
                TESTING_MALLOC_PIN(magma_h_R, double, N * N);
            }
        }

        //if(timing==1)
        magma_queue_sync(magmaqueue);
        clock_gettime(CLOCK_MILK, &t4);

        if(MAGMAfloat == 1)
        {
            // SSYEVD computes all eigenvalues and, optionally, eigenvectors of a real symmetric matrix A
            if(magmaXmode == 1)
            {
                magma_ssyevdx_gpu(MagmaVec,
                                  MagmaRangeI,
                                  MagmaLower,
                                  N,
                                  magmaf_d_AtA,
                                  N,
                                  0.0,
                                  1.0,
                                  N - MaxNBmodes,
                                  N,
                                  &mout,
                                  magmaf_w1,
                                  magmaf_h_R,
                                  N,
                                  magmaf_h_work,
                                  magma_lwork,
                                  magma_iwork,
                                  magma_liwork,
                                  &info);
            }
            else
            {
                magma_ssyevd_gpu(MagmaVec,
                                 MagmaLower,
                                 N,
                                 magmaf_d_AtA,
                                 N,
                                 magmaf_w1,
                                 magmaf_h_R,
                                 N,
                                 magmaf_h_work,
                                 magma_lwork,
                                 magma_iwork,
                                 magma_liwork,
                                 &info);
            }
        }
        else
        {
            // CODE CAN HANG HERE - THIS HAPPENS ONCE OUT OF multiple 1000s EXECUTIONS WHEN RUNNING IN A LOOP.. SEEMS TO BE A MAGMA ISSUE

            // SSYEVD computes all eigenvalues and, optionally, eigenvectors of a real symmetric matrix A
            magma_dsyevd_gpu(MagmaVec,
                             MagmaLower,
                             N,
                             magma_d_AtA,
                             N,
                             magma_w1,
                             magma_h_R,
                             N,
                             magma_h_work,
                             magma_lwork,
                             magma_iwork,
                             magma_liwork,
                             &info);

            if(VERBOSE_CUDACOMP_magma_compute_SVDpseudoInverse == 1)
            {
                printf(" DONE\n");
                fflush(stdout);
            }
        }

        if(LOOPmode == 0)
        {
            TESTING_FREE_CPU(magma_iwork);

            if(MAGMAfloat == 1)
            {
                TESTING_FREE_PIN(magmaf_h_R);
            }
            else
            {
                TESTING_FREE_PIN(magma_h_R);
            }

            if(MAGMAfloat == 1)
            {
                TESTING_FREE_PIN(magmaf_h_work);
            }
            else
            {
                TESTING_FREE_PIN(magma_h_work);
            }
        }

        //if(timing==1)
        magma_queue_sync(magmaqueue);
        clock_gettime(CLOCK_MILK, &t5);


        if(testmode == 1)
        {
            char fname[STRINGMAXLEN_FILENAME];
            WRITE_FILENAME(fname, "eigenv.dat");
            FILE *fp;

            if((fp = fopen(fname, "w")) == NULL)
            {
                printf("ERROR: cannot create file \"%s\"\n", fname);
                abort();
            }
            if(MAGMAfloat == 1)
            {
                for(long k = 0; k < N; k++)
                {
                    fprintf(fp,
                            "%5ld %20.8g  %20.8f  %g\n",
                            k,
                            magmaf_w1[N - k - 1],
                            magmaf_w1[N - k - 1] / magmaf_w1[N - 1],
                            SVDeps * SVDeps);
                }
            }
            else
            {
                for(long k = 0; k < N; k++)
                {
                    fprintf(fp,
                            "%5ld %20.8g  %20.8f  %g\n",
                            k,
                            magma_w1[N - k - 1],
                            magma_w1[N - k - 1] / magma_w1[N - 1],
                            SVDeps * SVDeps);
                }
            }
            fclose(fp);
        }

        /// w1 values are the EIGENVALUES of AT A
        /// Note: w1 values are the SQUARE of the singular values of A

        // ****************************************************
        // STEP 5 :   Set eigenvalue limit
        // ****************************************************
        DEBUG_TRACEPOINT("Set eigenvalue limit");
        double egvlim;
        if(MAGMAfloat == 1)
        {
            egvlim = SVDeps * SVDeps * magmaf_w1[N - 1];
        }
        else
        {
            egvlim = SVDeps * SVDeps * magma_w1[N - 1];
        }

        MaxNBmodes1 = MaxNBmodes;
        if(MaxNBmodes1 > N)
        {
            MaxNBmodes1 = N;
        }
        if(MaxNBmodes1 > M)
        {
            MaxNBmodes1 = M;
        }
        mode = 0;

        if(MAGMAfloat == 1)
        {
            while((mode < MaxNBmodes1) && (magmaf_w1[N - mode - 1] > egvlim))
            {
                mode++;
            }
        }
        else
        {
            while((mode < MaxNBmodes1) && (magma_w1[N - mode - 1] > egvlim))
            {
                mode++;
            }
        }

        if(VERBOSE_CUDACOMP_magma_compute_SVDpseudoInverse == 1)
        {
            printf(
                "Keeping %ld modes  (SVDeps = %g -> %g, MaxNBmodes = %ld -> "
                "%ld)\n",
                mode,
                SVDeps,
                egvlim,
                MaxNBmodes,
                MaxNBmodes1);
            fflush(stdout);
        }

        if(testmode == 1)
        {
            FILE *fp = fopen("test_SVDmodes.log", "w");
            fprintf(fp, "%6ld %6ld\n", mode, MaxNBmodes1);
            fclose(fp);
        }
        MaxNBmodes1 = mode;
        printf("Keeping %ld modes  (SVDeps = %g)\n", MaxNBmodes1, SVDeps);

        // ****************************************************
        // STEP 6 :   Write eigenvectors to VT matrix
        // ****************************************************
        DEBUG_TRACEPOINT("Write eigenvectors");
        // eigenvectors are in magma_d_AtA (device), copy them to magma_h_AtA (host)

        if(MAGMAfloat == 1)
        {
            magma_sgetmatrix(N,
                             N,
                             magmaf_d_AtA,
                             N,
                             magmaf_h_AtA,
                             N,
                             magmaqueue);
        }
        else
        {
            magma_dgetmatrix(N, N, magma_d_AtA, N, magma_h_AtA, N, magmaqueue);
        }

        // copy eigenvectors from magma_h_AtA to VT
        {
            imageID ID_VT;
            FUNC_CHECK_RETURN(
                create_2Dimage_ID(ID_VTmatrix_name, N, N, &ID_VT));

            if(MAGMAfloat == 1)
            {
                for(long ii = 0; ii < N; ii++)
                    for(long jj = 0; jj < N; jj++)
                    {
                        data.image[ID_VT].array.F[jj * N + ii] =
                            magmaf_h_AtA[(N - ii - 1) * N + jj];
                    }
            }
            else
            {
                for(long ii = 0; ii < N; ii++)
                    for(long jj = 0; jj < N; jj++)
                    {
                        data.image[ID_VT].array.F[jj * N + ii] =
                            magma_h_AtA[(N - ii - 1) * N + jj];
                    }
            }
        }

        if(testmode == 1)
        {
            FUNC_CHECK_RETURN(save_fits(ID_VTmatrix_name, "test_mVT.fits"));
        }

        // ****************************************************
        // STEP 7 :   Write eigenvectors/eigenvalue to magma_h_VT1 if eigenvalue > limit
        //          Copy to magma_d_VT1
        // ****************************************************
        DEBUG_TRACEPOINT(
            "Write eigenvectors/eigenvalue to magma_h_VT1 if eigenvalue > "
            "limit");

        if(MAGMAfloat == 1)
        {
            for(long ii = 0; ii < N; ii++)
                for(long jj = 0; jj < N; jj++)
                {
                    if(N - jj - 1 < MaxNBmodes1)
                    {
                        magmaf_h_VT1[ii * N + jj] =
                            magmaf_h_AtA[jj * N + ii] / magmaf_w1[jj];
                    }
                    else
                    {
                        magmaf_h_VT1[ii * N + jj] = 0.0;
                    }
                }
            magma_ssetmatrix(N,
                             N,
                             magmaf_h_VT1,
                             N,
                             magmaf_d_VT1,
                             N,
                             magmaqueue);
        }
        else
        {
            for(long ii = 0; ii < N; ii++)
                for(long jj = 0; jj < N; jj++)
                {
                    if(N - jj - 1 < MaxNBmodes1)
                    {
                        magma_h_VT1[ii * N + jj] =
                            magma_h_AtA[jj * N + ii] / magma_w1[jj];
                    }
                    else
                    {
                        magma_h_VT1[ii * N + jj] = 0.0;
                    }
                }
            magma_dsetmatrix(N, N, magma_h_VT1, N, magma_d_VT1, N, magmaqueue);
        }

        if(LOOPmode == 0)
        {
            if(MAGMAfloat == 1)
            {
                TESTING_FREE_CPU(magmaf_h_VT1);
                TESTING_FREE_CPU(magmaf_w1);
            }
            else
            {
                TESTING_FREE_CPU(magma_h_VT1);
                TESTING_FREE_CPU(magma_w1);
            }
        }

        //if(timing==1)
        magma_queue_sync(magmaqueue);
        clock_gettime(CLOCK_MILK, &t6);

        // ****************************************************
        // STEP 8 :   Compute M2 = VT1 VT = (AT A)^-1
        // ****************************************************
        DEBUG_TRACEPOINT("Compute M2 = VT1 VT = (AT A)^-1");

        if(MAGMAfloat == 1)
        {
            magma_sgemm(MagmaTrans,
                        MagmaTrans,
                        N,
                        N,
                        N,
                        1.0,
                        magmaf_d_VT1,
                        N,
                        magmaf_d_AtA,
                        N,
                        0.0,
                        magmaf_d_M2,
                        N,
                        magmaqueue);
        }
        else
        {
            magma_dgemm(MagmaTrans,
                        MagmaTrans,
                        N,
                        N,
                        N,
                        1.0,
                        magma_d_VT1,
                        N,
                        magma_d_AtA,
                        N,
                        0.0,
                        magma_d_M2,
                        N,
                        magmaqueue);

            if(VERBOSE_CUDACOMP_magma_compute_SVDpseudoInverse == 1)
            {
                printf("-> DONE\n");
                fflush(stdout);
            }
        }

        if(testmode == 1)
        {
            imageID ID_M2;

            FUNC_CHECK_RETURN(create_2Dimage_ID("mM2", N, N, &ID_M2));

            DEBUG_TRACEPOINT("Computing mM2");

            if(MAGMAfloat == 1)
            {
                TESTING_SMALLOC_CPU(magmaf_h_M2, N * N);
                magma_sgetmatrix(N,
                                 N,
                                 magmaf_d_M2,
                                 N,
                                 magmaf_h_M2,
                                 N,
                                 magmaqueue);

                for(long ii = 0; ii < N; ii++)
                    for(long jj = 0; jj < N; jj++)
                    {
                        data.image[ID_M2].array.F[jj * N + ii] =
                            magmaf_h_M2[jj * N + ii];
                    }
            }
            else
            {
                TESTING_DMALLOC_CPU(magma_h_M2, N * N);
                magma_dgetmatrix(N,
                                 N,
                                 magma_d_M2,
                                 N,
                                 magma_h_M2,
                                 N,
                                 magmaqueue);

                for(long ii = 0; ii < N; ii++)
                    for(long jj = 0; jj < N; jj++)
                    {
                        data.image[ID_M2].array.F[jj * N + ii] =
                            magma_h_M2[jj * N + ii];
                    }
            }
            DEBUG_TRACEPOINT("Saving mM2");
            FUNC_CHECK_RETURN(save_fits("mM2", "test_mM2.fits"));
            FUNC_CHECK_RETURN(
                delete_image_ID("mM2", DELETE_IMAGE_ERRMODE_WARNING));

            //	magma_dsetmatrix( N, N, h_M2, N, d_M2, N, magmaqueue);
            if(MAGMAfloat == 1)
            {
                TESTING_FREE_CPU(magmaf_h_M2);
            }
            else
            {
                TESTING_FREE_CPU(magma_h_M2);
            }
        }

        if(LOOPmode == 0)
        {
            if(MAGMAfloat == 1)
            {
                TESTING_FREE_DEV(magmaf_d_VT1);
            }
            else
            {
                TESTING_FREE_DEV(magma_d_VT1);
            }
        }

        //if(timing==1)
        magma_queue_sync(magmaqueue);
        clock_gettime(CLOCK_MILK, &t7);

        // ****************************************************
        // STEP 9 :   Compute Ainv = M2 A = (AT A)^-1 A
        // ****************************************************
        DEBUG_TRACEPOINT("Compute Ainv = M2 A = (AT A)^-1 A");

        // compute Ainv = M2 A
        if(MAGMAloop_iter == 0)
        {
            dAinvMODE = 1;
            if(MAGMAfloat == 1)
            {
                TESTING_SMALLOC_DEV(magmaf_d_Ainv, N * M);
            }
            else
            {
                TESTING_DMALLOC_DEV(magma_d_Ainv, N * M);
            }
        }

        if(MAGMAfloat == 1)
        {
            magma_sgemm(MagmaNoTrans,
                        MagmaNoTrans,
                        M,
                        N,
                        N,
                        1.0,
                        magmaf_d_A,
                        M,
                        magmaf_d_M2,
                        N,
                        0.0,
                        magmaf_d_Ainv,
                        M,
                        magmaqueue);
        }
        else
        {
            DEBUG_TRACEPOINT("double precision running magma_dgemm");
            magma_dgemm(MagmaNoTrans,
                        MagmaNoTrans,
                        M,
                        N,
                        N,
                        1.0,
                        magma_d_A,
                        M,
                        magma_d_M2,
                        N,
                        0.0,
                        magma_d_Ainv,
                        M,
                        magmaqueue);
            DEBUG_TRACEPOINT("double precision magma_dgemm done");
        }

        DEBUG_TRACEPOINT("free");
        if(LOOPmode == 0)
        {
            if(MAGMAfloat == 1)
            {
                TESTING_FREE_DEV(magmaf_d_M2);
            }
            else
            {
                TESTING_FREE_DEV(magma_d_M2);
            }
        }

        //if(timing==1)
        magma_queue_sync(magmaqueue);
        clock_gettime(CLOCK_MILK, &t8);

        DEBUG_TRACEPOINT("set result");
        if(MAGMAfloat == 1)
        {
            magma_sgetmatrix(M,
                             N,
                             magmaf_d_Ainv,
                             M,
                             magmaf_h_Ainv,
                             M,
                             magmaqueue);
        }
        else
        {
            magma_dgetmatrix(M,
                             N,
                             magma_d_Ainv,
                             M,
                             magma_h_Ainv,
                             M,
                             magmaqueue);
        }

        DEBUG_TRACEPOINT("end of magma computation");

    } // END STD MAGMA =================================================
    // End of QDWHPartial / MAGMA conditional

    //
    // At this point, pseudo-inverse is in magma_h_Ainv or magmaf_h_Ainv
    //


    if(testmode == 1)
    {
        imageID ID_Ainv;
        FUNC_CHECK_RETURN(create_2Dimage_ID("mAinv", M, N, &ID_Ainv));
        if(MAGMAfloat == 1)
        {

            {
                for(long ii = 0; ii < M; ii++)
                    for(long jj = 0; jj < N; jj++)
                    {
                        data.image[ID_Ainv].array.F[jj * M + ii] =
                            magmaf_h_Ainv[jj * M + ii];
                    }
            }
        }
        else
        {
            for(long ii = 0; ii < M; ii++)
                for(long jj = 0; jj < N; jj++)
                {
                    data.image[ID_Ainv].array.F[jj * M + ii] =
                        magma_h_Ainv[jj * M + ii];
                }
        }

        FUNC_CHECK_RETURN(save_fits("mAinv", "test_mAinv.fits"));
        FUNC_CHECK_RETURN(
            delete_image_ID("mAinv", DELETE_IMAGE_ERRMODE_IGNORE));
    }

    //if(timing==1)
    magma_queue_sync(magmaqueue);
    clock_gettime(CLOCK_MILK, &t9);

    if(MAGMAloop_iter == 0)
    {
        uint32_t *arraysizetmp;
        arraysizetmp = (uint32_t *) malloc(sizeof(uint32_t) *
                                           data.image[ID_Rmatrix].md[0].naxis);

        if(data.image[ID_Rmatrix].md[0].naxis == 3)
        {
            arraysizetmp[0] = data.image[ID_Rmatrix].md[0].size[0];
            arraysizetmp[1] = data.image[ID_Rmatrix].md[0].size[1];
            arraysizetmp[2] = N;
        }
        else
        {
            arraysizetmp[0] = M;
            arraysizetmp[1] = N;
        }

        FUNC_CHECK_RETURN(create_image_ID(ID_Cmatrix_name,
                                          data.image[ID_Rmatrix].md[0].naxis,
                                          arraysizetmp,
                                          datatype,
                                          0,
                                          0,
                                          0,
                                          &ID_Cmatrix));

        free(arraysizetmp);
    }
    else
    {
        ID_Cmatrix = image_ID(ID_Cmatrix_name);
    }

    magma_queue_sync(magmaqueue);
    clock_gettime(CLOCK_MILK, &t10);

    if(VERBOSE_CUDACOMP_magma_compute_SVDpseudoInverse == 1)
    {
        printf("write result\n");
        fflush(stdout);
    }

    if(datatype == _DATATYPE_FLOAT)
    {
        if(MAGMAfloat == 1)
        {

            memcpy(data.image[ID_Cmatrix].array.F,
                   magmaf_h_Ainv,
                   sizeof(float) * M * N);
        }
        else
        {
            for(long ii = 0; ii < M * N; ii++)
            {
                data.image[ID_Cmatrix].array.F[ii] = (float) magma_h_Ainv[ii];
            }
        }
    }
    else
    {
        // sensors : M
        // actuator modes: N
        if(MAGMAfloat == 1)
        {
            for(long ii = 0; ii < M * N; ii++)
            {
                data.image[ID_Cmatrix].array.D[ii] = magmaf_h_Ainv[ii];
            }
        }
        else
        {
            memcpy(data.image[ID_Cmatrix].array.D,
                   magma_h_Ainv,
                   sizeof(double) * M * N);
        }
    }

    //if(timing==1)
    magma_queue_sync(magmaqueue);
    clock_gettime(CLOCK_MILK, &t11);

    if(testmode == 1)  // compute product of Ainv with A
    {
        if(MAGMAfloat == 1)
        {
            magma_sgemm(MagmaTrans,
                        MagmaNoTrans,
                        N,
                        N,
                        M,
                        1.0,
                        magmaf_d_A,
                        M,
                        magmaf_d_Ainv,
                        M,
                        0.0,
                        magmaf_d_AtA,
                        N,
                        magmaqueue);
        }
        else
        {
            magma_dgemm(MagmaTrans,
                        MagmaNoTrans,
                        N,
                        N,
                        M,
                        1.0,
                        magma_d_A,
                        M,
                        magma_d_Ainv,
                        M,
                        0.0,
                        magma_d_AtA,
                        N,
                        magmaqueue);
        }

        imageID ID_AinvA;

        FUNC_CHECK_RETURN(create_2Dimage_ID("AinvA", N, N, &ID_AinvA));

        // copy from GPU to CPU
        if(MAGMAfloat == 1)
        {
            magma_sgetmatrix(N,
                             N,
                             magmaf_d_AtA,
                             N,
                             magmaf_h_AtA,
                             N,
                             magmaqueue);
        }
        else
        {
            magma_dgetmatrix(N, N, magma_d_AtA, N, magma_h_AtA, N, magmaqueue);
        }

        if(datatype == _DATATYPE_FLOAT)
        {
            if(MAGMAfloat == 1)
            {
                memcpy(data.image[ID_AinvA].array.F,
                       magmaf_h_AtA,
                       sizeof(float) * N * N);
            }
            else
            {
                for(long ii = 0; ii < N * N; ii++)
                {
                    data.image[ID_AinvA].array.F[ii] = magma_h_AtA[ii];
                }
            }
        }
        else
        {
            if(MAGMAfloat == 1)
            {
                for(long ii = 0; ii < N * N; ii++)
                {
                    data.image[ID_AinvA].array.D[ii] = magmaf_h_AtA[ii];
                }
            }
            else
            {
                memcpy(data.image[ID_AinvA].array.D,
                       magma_h_AtA,
                       sizeof(double) * M * N);
            }
        }

        FUNC_CHECK_RETURN(save_fits("AinvA", "test_AinvA.fits"));
        FUNC_CHECK_RETURN(
            delete_image_ID("AinvA", DELETE_IMAGE_ERRMODE_IGNORE));
    }

    magma_queue_sync(magmaqueue);
    clock_gettime(CLOCK_MILK, &t12);

    ID_PFfmdat = image_ID("PFfmdat");
    if(ID_PFfmdat != -1)
    {
        printf("Transp(Ainv)     N x M   = %d x %d\n", N, M);
        printf("PFfmdat  M x K           = %d x %d\n",
               data.image[ID_PFfmdat].md[0].size[0],
               data.image[ID_PFfmdat].md[0].size[1]);
        long K = data.image[ID_PFfmdat].md[0].size[1];
        printf("K = %ld\n", K);

        float *magmaf_d_PFfmdat;
        float *magmaf_d_PF;
        float *magmaf_h_PF;

        TESTING_SMALLOC_DEV(magmaf_d_PFfmdat, M * K);
        TESTING_SMALLOC_DEV(magmaf_d_PF, N * K);
        TESTING_SMALLOC_CPU(magmaf_h_PF, N * K);

        magma_sgetmatrix(N, K, magmaf_d_PF, N, magmaf_h_PF, N, magmaqueue);

        magma_ssetmatrix(M,
                         K,
                         data.image[ID_PFfmdat].array.F,
                         M,
                         magmaf_d_PFfmdat,
                         M,
                         magmaqueue);



        magma_sgetmatrix(N, K, magmaf_d_PF, N, magmaf_h_PF, N, magmaqueue);

        magma_sgemm(MagmaTrans,
                    MagmaNoTrans,
                    N,
                    K,
                    M,
                    1.0,
                    magmaf_d_Ainv,
                    M,
                    magmaf_d_PFfmdat,
                    M,
                    0.0,
                    magmaf_d_PF,
                    N,
                    magmaqueue);


        magma_sgetmatrix(N, K, magmaf_d_PF, N, magmaf_h_PF, N, magmaqueue);

        imageID ID_PF;
        FUNC_CHECK_RETURN(create_2Dimage_ID("psinvPFmat", N, K, &ID_PF));

        memcpy(data.image[ID_PF].array.F, magmaf_h_PF, sizeof(float) * N * K);
        FUNC_CHECK_RETURN(save_fits("psinvPFmat", "psinvPFmat.fits"));

        TESTING_FREE_DEV(magmaf_d_PFfmdat);
        TESTING_FREE_DEV(magmaf_d_PF);
        TESTING_FREE_CPU(magmaf_h_PF);
    }

    magma_queue_sync(magmaqueue);
    clock_gettime(CLOCK_MILK, &t13);

    if(LOOPmode ==
            0) /// if pseudo-inverse is only computed once, these arrays can be freed
    {
        if(MAGMAfloat == 1)
        {
            TESTING_FREE_CPU(magmaf_h_A);
        }
        else
        {
            TESTING_FREE_CPU(magma_h_A);
        }
    }

    if(LOOPmode == 0)
    {
        if(MAGMAfloat == 1)
        {
            TESTING_FREE_DEV(magmaf_d_A);

            if(dAinvMODE == 1)
            {
                TESTING_FREE_DEV(magmaf_d_Ainv);
            }

            TESTING_FREE_CPU(magmaf_h_Ainv);
            TESTING_FREE_DEV(magmaf_d_AtA);
            TESTING_FREE_CPU(magmaf_h_AtA);
        }
        else
        {
            TESTING_FREE_DEV(magma_d_A);

            if(dAinvMODE == 1)
            {
                TESTING_FREE_DEV(magma_d_Ainv);
            }

            TESTING_FREE_CPU(magma_h_Ainv);
            TESTING_FREE_DEV(magma_d_AtA);
            TESTING_FREE_CPU(magma_h_AtA);
        }
    }

    if(LOOPmode == 0)
    {
        free(devicearray);
        magma_queue_destroy(magmaqueue);
        magma_finalize(); //  finalize  Magma
    }

    //if(timing==1)
    //{
    t01d = timespec_diff_double(t0, t1);

    t12d   = timespec_diff_double(t1, t2);
    t23d   = timespec_diff_double(t2, t3);
    t34d   = timespec_diff_double(t3, t4);
    t45d   = timespec_diff_double(t4, t5);
    t56d   = timespec_diff_double(t5, t6);
    t67d   = timespec_diff_double(t6, t7);
    t78d   = timespec_diff_double(t7, t8);
    t89d   = timespec_diff_double(t8, t9);
    t910d  = timespec_diff_double(t9, t10);
    t1011d = timespec_diff_double(t10, t11);
    t1112d = timespec_diff_double(t11, t12);
    t1213d = timespec_diff_double(t12, t13);
    t013d  = timespec_diff_double(t0, t13);

    if(VERBOSE_CUDACOMP_magma_compute_SVDpseudoInverse == 1)
    {
        printf("%6ld  Timing info: \n", MAGMAloop_iter);
        printf("  0-1	[setup]                           %12.3f ms\n",
               t01d * 1000.0);
        printf("  1-2	[copy input to GPU]               %12.3f ms\n",
               t12d * 1000.0);

        printf("  2-3	[compute trans(A) x A]            %12.3f ms\n",
               t23d * 1000.0);
        printf("  3-4	[setup]                           %12.3f ms\n",
               t34d * 1000.0);
        printf("  4-5	[Compute eigenvalues]             %12.3f ms\n",
               t45d * 1000.0);
        printf("  5-6	[Select eigenvalues]              %12.3f ms\n",
               t56d * 1000.0);
        printf("  6-7	[Compute M2]                      %12.3f ms\n",
               t67d * 1000.0);
        printf("  7-8	[Compute Ainv]                    %12.3f ms\n",
               t78d * 1000.0);
        printf("  8-9	[Get Ainv from GPU]               %12.3f ms\n",
               t89d * 1000.0);

        printf("  9-10	[output setup]                    %12.3f ms\n",
               t910d * 1000.0);
        printf("  10-11	[Write output array]              %12.3f ms\n",
               t1011d * 1000.0);
        printf("  11-12	[Test output]                     %12.3f ms\n",
               t1112d * 1000.0);
        printf("  12-13	[Optional gemm]                   %12.3f ms\n",
               t1213d * 1000.0);
        printf("\n");
        printf(" TOTAL 0-13     %12.3f ms\n", t013d * 1000.0);
        fflush(stdout);
    }
    //}

    if(VERBOSE_CUDACOMP_magma_compute_SVDpseudoInverse == 1)
    {
        printf("\n");
        fflush(stdout);
    }

    if(LOOPmode == 1)
    {
        MAGMAloop_iter++;
    }

    list_image_ID();

    if(outID != NULL)
    {
        *outID = ID_Cmatrix;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

#endif

#endif
