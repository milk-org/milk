/**
 * @file test_MVMextract.c
 * @brief Round-trip correctness test for MVM mode extraction
 *
 * Tests both CPU (BLAS) and GPU (cuBLAS) MVM paths by
 * creating orthogonal modes, synthesizing an input image
 * from known coefficients, running the MVM, and verifying
 * that the extracted coefficients match.
 *
 * Exit code:
 *   0 = all tests passed
 *   1 = test failure
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef HAVE_CUDA
#    include <cublas_v2.h>
#    include <cuda_runtime.h>
#endif

#include "ImageStreamIO/ImageStreamIO.h"
#include "ImageStreamIO/ImageStruct.h"
#include "MVM_CPU.h"
#include "libfps/milk_help.h"


/* --------------------------------------------------------
 * Test configuration
 * ------------------------------------------------------ */

#define NPIX_X 16              /* image width */
#define NPIX_Y 16              /* image height */
#define NPIX (NPIX_X * NPIX_Y) /* = 256 */
#define NMODES 5               /* number of modes */
#define TOL 1.0e-4f            /* relative tolerance */

/* Stream names for SHM cleanup */
#define SNAME_MODES "test_mvm_modes"
#define SNAME_INPUT "test_mvm_input"


/* --------------------------------------------------------
 * Helper: build orthogonal modes matrix
 *
 * Each mode k is a vector of length npix with a
 * distinctive pattern that is orthogonal to all others.
 * We use shifted Hadamard-like rows: mode k has value
 * +1 in pixels where (pixel_index / stride) is even,
 * -1 where odd, with stride = 2^k. This gives pairwise
 * orthogonal rows for k < log2(npix).
 * ------------------------------------------------------ */

static void build_orthogonal_modes(float *modes, int nmodes, int npix)
{
    for (int k = 0; k < nmodes; k++)
    {
        int stride = 1 << k; /* 1, 2, 4, 8, 16 */
        for (int p = 0; p < npix; p++)
        {
            int block           = p / stride;
            modes[k * npix + p] = (block % 2 == 0) ? 1.0f : -1.0f;
        }
    } // for k
}


/* --------------------------------------------------------
 * Helper: compute mode norms for verification
 * ------------------------------------------------------ */

static void compute_mode_norms(const float *modes, float *norms, int nmodes, int npix)
{
    for (int k = 0; k < nmodes; k++)
    {
        float sum = 0.0f;
        for (int p = 0; p < npix; p++)
        {
            float v = modes[k * npix + p];
            sum += v * v;
        }
        norms[k] = sum;
    }
}


/* --------------------------------------------------------
 * Helper: verify coefficients match expected
 *
 * Returns 0 on success, 1 on failure.
 * ------------------------------------------------------ */

static int verify_coefficients(const float *outarray,
                               const float *expected,
                               const float *norms,
                               int          nmodes,
                               int          verbose,
                               const char  *label)
{
    int pass = 1;
    for (int k = 0; k < nmodes; k++)
    {
        float extracted = (norms != NULL) ? outarray[k] / norms[k] : outarray[k];
        float exp_val   = expected[k];
        float err       = fabsf(extracted - exp_val);
        float ref       = fabsf(exp_val) > 1.0e-6f ? fabsf(exp_val) : 1.0f;

        if (verbose)
        {
            printf("  [%s] mode %2d: expected=%+8.4f"
                   "  got=%+8.4f  err=%.2e\n",
                   label, k, exp_val, extracted, err);
        }

        if (err / ref > TOL)
        {
            printf("  [%s] FAIL mode %d: expected %f,"
                   " got %f (err=%e)\n",
                   label, k, exp_val, extracted, err);
            pass = 0;
        }
    }
    return pass ? 0 : 1;
}


/* ================================================================
 * CPU TESTS
 * ============================================================= */


/* --------------------------------------------------------
 * CPU Test 1: Basic round-trip
 * ------------------------------------------------------ */

static int test_cpu_roundtrip(int verbose)
{
    printf("[CPU 1] Round-trip coefficient extraction\n");

    float *modes = (float *) malloc(sizeof(float) * NMODES * NPIX);
    build_orthogonal_modes(modes, NMODES, NPIX);

    float coeff_in[NMODES] = { 1.0f, -2.5f, 3.0f, 0.5f, -1.0f };

    /* Synthesize input: I[p] = sum_k c[k] * M[k][p] */
    float *input = (float *) calloc(NPIX, sizeof(float));
    for (int k = 0; k < NMODES; k++)
    {
        for (int p = 0; p < NPIX; p++)
        {
            input[p] += coeff_in[k] * modes[k * NPIX + p];
        }
    }

    float *outarray = (float *) calloc(NMODES, sizeof(float));
    matrixMulCPU(modes, input, outarray, NMODES, NPIX);

    float norms[NMODES];
    compute_mode_norms(modes, norms, NMODES, NPIX);

    int ret = verify_coefficients(outarray, coeff_in, norms, NMODES, verbose, "CPU");

    free(outarray);
    free(input);
    free(modes);

    if (ret == 0)
    {
        printf("  PASS\n");
    }
    return ret;
}


/* --------------------------------------------------------
 * CPU Test 2: Zero input → zero output
 * ------------------------------------------------------ */

static int test_cpu_zero_input(int verbose)
{
    printf("[CPU 2] Zero input produces zero output\n");

    float *modes = (float *) malloc(sizeof(float) * NMODES * NPIX);
    build_orthogonal_modes(modes, NMODES, NPIX);

    float *input    = (float *) calloc(NPIX, sizeof(float));
    float *outarray = (float *) calloc(NMODES, sizeof(float));

    matrixMulCPU(modes, input, outarray, NMODES, NPIX);

    int pass = 1;
    for (int k = 0; k < NMODES; k++)
    {
        if (fabsf(outarray[k]) > 1.0e-7f)
        {
            printf("  FAIL mode %d: expected 0, got %e\n", k, outarray[k]);
            pass = 0;
        }
    }

    if (verbose && pass)
    {
        printf("  All output coefficients are zero\n");
    }

    free(outarray);
    free(input);
    free(modes);

    if (pass)
    {
        printf("  PASS\n");
    }
    return pass ? 0 : 1;
}


/* --------------------------------------------------------
 * CPU Test 3: Single-mode isolation (orthogonality)
 * ------------------------------------------------------ */

static int test_cpu_orthogonality(int verbose)
{
    printf("[CPU 3] Single-mode isolation "
           "(orthogonality)\n");

    float *modes = (float *) malloc(sizeof(float) * NMODES * NPIX);
    build_orthogonal_modes(modes, NMODES, NPIX);

    float norms[NMODES];
    compute_mode_norms(modes, norms, NMODES, NPIX);

    int pass = 1;
    for (int target = 0; target < NMODES; target++)
    {
        float *input = (float *) malloc(sizeof(float) * NPIX);
        memcpy(input, &modes[target * NPIX], sizeof(float) * NPIX);

        float *outarray = (float *) calloc(NMODES, sizeof(float));
        matrixMulCPU(modes, input, outarray, NMODES, NPIX);

        for (int k = 0; k < NMODES; k++)
        {
            float normalized = outarray[k] / norms[k];
            float expected   = (k == target) ? 1.0f : 0.0f;
            float err        = fabsf(normalized - expected);

            if (err > TOL)
            {
                printf("  FAIL target=%d, mode %d: "
                       "expected %f, got %f\n",
                       target, k, expected, normalized);
                pass = 0;
            }
        }

        if (verbose)
        {
            printf("  target mode %d: ", target);
            for (int k = 0; k < NMODES; k++)
            {
                printf("%+.3f ", outarray[k] / norms[k]);
            }
            printf("\n");
        }

        free(outarray);
        free(input);
    }

    free(modes);

    if (pass)
    {
        printf("  PASS\n");
    }
    return pass ? 0 : 1;
}


/* --------------------------------------------------------
 * CPU Test 4: SHM stream round-trip
 * ------------------------------------------------------ */

static int test_cpu_shm_roundtrip(int verbose)
{
    printf("[CPU 4] SHM stream data path\n");

    IMAGE img_modes;
    IMAGE img_input;

    {
        uint32_t sz[3] = { NPIX_X, NPIX_Y, NMODES };
        errno_t  ret =
            ImageStreamIO_createIm(&img_modes, SNAME_MODES, 3, sz, _DATATYPE_FLOAT, 1, 10, 0);
        if (ret != 0)
        {
            printf("  SKIP: cannot create SHM stream\n");
            return 0;
        }
    }

    {
        uint32_t sz[2] = { NPIX_X, NPIX_Y };
        errno_t  ret =
            ImageStreamIO_createIm(&img_input, SNAME_INPUT, 2, sz, _DATATYPE_FLOAT, 1, 10, 0);
        if (ret != 0)
        {
            printf("  SKIP: cannot create SHM stream\n");
            ImageStreamIO_destroyIm(&img_modes);
            return 0;
        }
    }

    build_orthogonal_modes(img_modes.array.F, NMODES, NPIX);

    float coeff_in[NMODES] = { 1.5f, -0.5f, 2.0f, -1.0f, 0.25f };
    memset(img_input.array.F, 0, sizeof(float) * NPIX);
    for (int k = 0; k < NMODES; k++)
    {
        for (int p = 0; p < NPIX; p++)
        {
            img_input.array.F[p] += coeff_in[k] * img_modes.array.F[k * NPIX + p];
        }
    }

    float *outarray = (float *) calloc(NMODES, sizeof(float));
    matrixMulCPU(img_modes.array.F, img_input.array.F, outarray, NMODES, NPIX);

    float norms[NMODES];
    compute_mode_norms(img_modes.array.F, norms, NMODES, NPIX);

    int ret = verify_coefficients(outarray, coeff_in, norms, NMODES, verbose, "CPU-SHM");

    free(outarray);
    ImageStreamIO_destroyIm(&img_input);
    ImageStreamIO_destroyIm(&img_modes);

    if (ret == 0)
    {
        printf("  PASS\n");
    }
    return ret;
}


/* --------------------------------------------------------
 * CPU Test 5: Stress test (64×64, 20 delta modes)
 * ------------------------------------------------------ */

#define STRESS_NX 64
#define STRESS_NY 64
#define STRESS_NPIX (STRESS_NX * STRESS_NY)
#define STRESS_NMODES 20

static int test_cpu_stress(int verbose)
{
    printf("[CPU 5] Stress test (%dx%d, %d modes)\n", STRESS_NX, STRESS_NY, STRESS_NMODES);

    long   total = (long) STRESS_NMODES * STRESS_NPIX;
    float *modes = (float *) calloc(total, sizeof(float));
    for (int k = 0; k < STRESS_NMODES; k++)
    {
        modes[k * STRESS_NPIX + k] = 1.0f;
    }

    float coeff_in[STRESS_NMODES];
    for (int k = 0; k < STRESS_NMODES; k++)
    {
        coeff_in[k] = sinf((float) (k + 1) * 1.7f) * 10.0f;
    }

    float *input = (float *) calloc(STRESS_NPIX, sizeof(float));
    for (int k = 0; k < STRESS_NMODES; k++)
    {
        for (int p = 0; p < STRESS_NPIX; p++)
        {
            input[p] += coeff_in[k] * modes[k * STRESS_NPIX + p];
        }
    }

    float *outarray = (float *) calloc(STRESS_NMODES, sizeof(float));
    matrixMulCPU(modes, input, outarray, STRESS_NMODES, STRESS_NPIX);

    /* Delta modes have norm=1, no normalization */
    int ret = verify_coefficients(outarray, coeff_in, NULL, STRESS_NMODES, verbose, "CPU");

    free(outarray);
    free(input);
    free(modes);

    if (ret == 0)
    {
        printf("  PASS\n");
    }
    return ret;
}


/* ================================================================
 * GPU TESTS
 * ============================================================= */

#ifdef HAVE_CUDA

/* --------------------------------------------------------
 * Helper: run MVM on GPU via cuBLAS
 *
 * Mirrors the GPU path in MVMextractModes.c:
 * cublasSgemv(handle, CUBLAS_OP_T, npix, nmodes,
 *             &alpha, d_modes, npix, d_in, 1,
 *             &beta, d_out, 1)
 *
 * Returns 0 on success, -1 on failure.
 * ------------------------------------------------------ */

static int gpu_mvm(const float *h_modes,
                   const float *h_input,
                   float       *h_output,
                   int          nmodes,
                   int          npix)
{
    cublasHandle_t handle;
    cublasStatus_t stat;
    cudaError_t    cerr;
    float         *d_modes  = NULL;
    float         *d_input  = NULL;
    float         *d_output = NULL;
    int            ret      = -1;

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        printf("  cuBLAS init failed\n");
        return -1;
    }

    /* Allocate device memory */
    cerr = cudaMalloc((void **) &d_modes, sizeof(float) * nmodes * npix);
    if (cerr != cudaSuccess)
    {
        goto cleanup;
    }

    cerr = cudaMalloc((void **) &d_input, sizeof(float) * npix);
    if (cerr != cudaSuccess)
    {
        goto cleanup;
    }

    cerr = cudaMalloc((void **) &d_output, sizeof(float) * nmodes);
    if (cerr != cudaSuccess)
    {
        goto cleanup;
    }

    /* Copy data to GPU */
    cerr = cudaMemcpy(d_modes, h_modes, sizeof(float) * nmodes * npix, cudaMemcpyHostToDevice);
    if (cerr != cudaSuccess)
    {
        goto cleanup;
    }

    cerr = cudaMemcpy(d_input, h_input, sizeof(float) * npix, cudaMemcpyHostToDevice);
    if (cerr != cudaSuccess)
    {
        goto cleanup;
    }

    /* cuBLAS sgemv:
     * The modes matrix is row-major nmodes×npix.
     * cuBLAS uses column-major, so we treat it as
     * column-major npix×nmodes and use CUBLAS_OP_T:
     *   out = alpha * A^T * x + beta * y
     * This computes out[k] = sum_p modes[k*npix+p] * in[p]
     */
    {
        float alpha = 1.0f;
        float beta  = 0.0f;

        stat = cublasSgemv(handle, CUBLAS_OP_T, npix, nmodes, &alpha, d_modes, npix, d_input, 1,
                           &beta, d_output, 1);
        if (stat != CUBLAS_STATUS_SUCCESS)
        {
            printf("  cublasSgemv failed: %d\n", stat);
            goto cleanup;
        }
    }

    /* Copy result back */
    cerr = cudaMemcpy(h_output, d_output, sizeof(float) * nmodes, cudaMemcpyDeviceToHost);
    if (cerr != cudaSuccess)
    {
        goto cleanup;
    }

    ret = 0;

cleanup:
    if (d_output != NULL)
    {
        cudaFree(d_output);
    }
    if (d_input != NULL)
    {
        cudaFree(d_input);
    }
    if (d_modes != NULL)
    {
        cudaFree(d_modes);
    }
    cublasDestroy(handle);
    return ret;
}


/* --------------------------------------------------------
 * GPU Test 1: Round-trip coefficient extraction
 * ------------------------------------------------------ */

static int test_gpu_roundtrip(int verbose)
{
    printf("[GPU 1] Round-trip coefficient extraction\n");

    float *modes = (float *) malloc(sizeof(float) * NMODES * NPIX);
    build_orthogonal_modes(modes, NMODES, NPIX);

    float coeff_in[NMODES] = { 1.0f, -2.5f, 3.0f, 0.5f, -1.0f };

    float *input = (float *) calloc(NPIX, sizeof(float));
    for (int k = 0; k < NMODES; k++)
    {
        for (int p = 0; p < NPIX; p++)
        {
            input[p] += coeff_in[k] * modes[k * NPIX + p];
        }
    }

    float *outarray = (float *) calloc(NMODES, sizeof(float));

    if (gpu_mvm(modes, input, outarray, NMODES, NPIX) != 0)
    {
        printf("  SKIP: GPU MVM failed\n");
        free(outarray);
        free(input);
        free(modes);
        return 0;
    }

    float norms[NMODES];
    compute_mode_norms(modes, norms, NMODES, NPIX);

    int ret = verify_coefficients(outarray, coeff_in, norms, NMODES, verbose, "GPU");

    free(outarray);
    free(input);
    free(modes);

    if (ret == 0)
    {
        printf("  PASS\n");
    }
    return ret;
}


/* --------------------------------------------------------
 * GPU Test 2: Single-mode isolation (orthogonality)
 * ------------------------------------------------------ */

static int test_gpu_orthogonality(int verbose)
{
    printf("[GPU 2] Single-mode isolation "
           "(orthogonality)\n");

    float *modes = (float *) malloc(sizeof(float) * NMODES * NPIX);
    build_orthogonal_modes(modes, NMODES, NPIX);

    float norms[NMODES];
    compute_mode_norms(modes, norms, NMODES, NPIX);

    int pass = 1;
    for (int target = 0; target < NMODES; target++)
    {
        float *input = (float *) malloc(sizeof(float) * NPIX);
        memcpy(input, &modes[target * NPIX], sizeof(float) * NPIX);

        float *outarray = (float *) calloc(NMODES, sizeof(float));

        if (gpu_mvm(modes, input, outarray, NMODES, NPIX) != 0)
        {
            printf("  SKIP: GPU MVM failed\n");
            free(outarray);
            free(input);
            free(modes);
            return 0;
        }

        for (int k = 0; k < NMODES; k++)
        {
            float normalized = outarray[k] / norms[k];
            float expected   = (k == target) ? 1.0f : 0.0f;
            float err        = fabsf(normalized - expected);

            if (err > TOL)
            {
                printf("  FAIL target=%d, mode %d: "
                       "expected %f, got %f\n",
                       target, k, expected, normalized);
                pass = 0;
            }
        }

        if (verbose)
        {
            printf("  target mode %d: ", target);
            for (int k = 0; k < NMODES; k++)
            {
                printf("%+.3f ", outarray[k] / norms[k]);
            }
            printf("\n");
        }

        free(outarray);
        free(input);
    }

    free(modes);

    if (pass)
    {
        printf("  PASS\n");
    }
    return pass ? 0 : 1;
}


/* --------------------------------------------------------
 * GPU Test 3: Stress test (64×64, 20 modes)
 * ------------------------------------------------------ */

static int test_gpu_stress(int verbose)
{
    printf("[GPU 3] Stress test (%dx%d, %d modes)\n", STRESS_NX, STRESS_NY, STRESS_NMODES);

    long   total = (long) STRESS_NMODES * STRESS_NPIX;
    float *modes = (float *) calloc(total, sizeof(float));
    for (int k = 0; k < STRESS_NMODES; k++)
    {
        modes[k * STRESS_NPIX + k] = 1.0f;
    }

    float coeff_in[STRESS_NMODES];
    for (int k = 0; k < STRESS_NMODES; k++)
    {
        coeff_in[k] = sinf((float) (k + 1) * 1.7f) * 10.0f;
    }

    float *input = (float *) calloc(STRESS_NPIX, sizeof(float));
    for (int k = 0; k < STRESS_NMODES; k++)
    {
        for (int p = 0; p < STRESS_NPIX; p++)
        {
            input[p] += coeff_in[k] * modes[k * STRESS_NPIX + p];
        }
    }

    float *outarray = (float *) calloc(STRESS_NMODES, sizeof(float));

    if (gpu_mvm(modes, input, outarray, STRESS_NMODES, STRESS_NPIX) != 0)
    {
        printf("  SKIP: GPU MVM failed\n");
        free(outarray);
        free(input);
        free(modes);
        return 0;
    }

    int ret = verify_coefficients(outarray, coeff_in, NULL, STRESS_NMODES, verbose, "GPU");

    free(outarray);
    free(input);
    free(modes);

    if (ret == 0)
    {
        printf("  PASS\n");
    }
    return ret;
}


/* --------------------------------------------------------
 * GPU Test 4: CPU vs GPU cross-validation
 *
 * Run the same MVM on both CPU and GPU, verify that
 * both produce the same result (within tolerance).
 * ------------------------------------------------------ */

static int test_cpu_gpu_match(int verbose)
{
    printf("[GPU 4] CPU vs GPU cross-validation\n");

    float *modes = (float *) malloc(sizeof(float) * NMODES * NPIX);
    build_orthogonal_modes(modes, NMODES, NPIX);

    float coeff_in[NMODES] = { 3.14f, -1.41f, 2.72f, -0.58f, 1.62f };

    float *input = (float *) calloc(NPIX, sizeof(float));
    for (int k = 0; k < NMODES; k++)
    {
        for (int p = 0; p < NPIX; p++)
        {
            input[p] += coeff_in[k] * modes[k * NPIX + p];
        }
    }

    /* CPU result */
    float *cpu_out = (float *) calloc(NMODES, sizeof(float));
    matrixMulCPU(modes, input, cpu_out, NMODES, NPIX);

    /* GPU result */
    float *gpu_out = (float *) calloc(NMODES, sizeof(float));
    if (gpu_mvm(modes, input, gpu_out, NMODES, NPIX) != 0)
    {
        printf("  SKIP: GPU MVM failed\n");
        free(gpu_out);
        free(cpu_out);
        free(input);
        free(modes);
        return 0;
    }

    /* Compare CPU vs GPU */
    int pass = 1;
    for (int k = 0; k < NMODES; k++)
    {
        float diff = fabsf(cpu_out[k] - gpu_out[k]);
        float ref  = fabsf(cpu_out[k]) > 1.0e-6f ? fabsf(cpu_out[k]) : 1.0f;

        if (verbose)
        {
            printf("  mode %d: cpu=%+.6f  gpu=%+.6f"
                   "  diff=%.2e\n",
                   k, cpu_out[k], gpu_out[k], diff);
        }

        if (diff / ref > TOL)
        {
            printf("  FAIL mode %d: cpu=%f gpu=%f"
                   " diff=%e\n",
                   k, cpu_out[k], gpu_out[k], diff);
            pass = 0;
        }
    }

    free(gpu_out);
    free(cpu_out);
    free(input);
    free(modes);

    if (pass)
    {
        printf("  PASS\n");
    }
    return pass ? 0 : 1;
}

#endif /* HAVE_CUDA */


/* ================================================================
 * Help
 * ============================================================= */

static const char *APP_DESCRIPTION = "MVM mode extraction correctness tests";

static const char *APP_DESCRIPTION_LONG = "Round-trip correctness test for matrix-vector\n"
                                          "multiply (MVM) mode extraction. Creates\n"
                                          "orthogonal modes, synthesizes an input image\n"
                                          "from known coefficients, runs the MVM on CPU\n"
                                          "and/or GPU, and verifies extracted coefficients\n"
                                          "match the originals within tolerance.";


static void print_help(const char *prog, int mh_color)
{
    milk_help_banner(prog, APP_DESCRIPTION, mh_color);

    /* Usage */
    milk_help_section("Usage", mh_color);
    if (mh_color)
    {
        printf("  " MH_CMD "%s" MH_RST " [" MH_OPT "options" MH_RST "]\n\n", prog);
    }
    else
    {
        printf("  %s [options]\n\n", prog);
    }

    /* Description */
    milk_help_section("Description", mh_color);
    printf("  %s\n\n", APP_DESCRIPTION_LONG);

    /* Options */
    milk_help_section("Options", mh_color);
    printf("  %s         Show this help\n", MH(MH_OPT, "-h, --help"));
    printf("  %s              One-line description\n", MH(MH_OPT, "-h1"));
    printf("  %s              Verbose description\n", MH(MH_OPT, "-h2"));
    printf("  %s              Monochrome help\n", MH(MH_OPT, "-hm"));
    printf("  %s      Per-mode expected/actual values\n", MH(MH_OPT, "-v, --verbose"));
    printf("  %s          Run CPU tests only\n", MH(MH_OPT, "-c, --cpu"));
    printf("  %s          Run GPU tests only\n", MH(MH_OPT, "-g, --gpu"));
    printf("  (default)          "
           "Run both CPU and GPU\n\n");

    /* Examples */
    milk_help_section("Examples", mh_color);
    printf("  %s %s %s"
           "          # CPU tests, verbose\n",
           MH(MH_DIM, "$"), MH(MH_CMD, "test_MVMextract"), MH(MH_OPT, "-v -c"));
    printf("  %s %s %s"
           "          # GPU tests, verbose\n",
           MH(MH_DIM, "$"), MH(MH_CMD, "test_MVMextract"), MH(MH_OPT, "-v -g"));
    printf("  %s %s %s"
           "             # both, verbose\n",
           MH(MH_DIM, "$"), MH(MH_CMD, "test_MVMextract"), MH(MH_OPT, "-v"));
    printf("  %s %s %s %s\n\n", MH(MH_DIM, "$"), MH(MH_CMD, "ctest"), MH(MH_OPT, "-R MVMextract"),
           MH(MH_OPT, "--output-on-failure"));

    /* See Also */
    const char *seealso[] = {
        "milk-fpsexec-linalg-MVMextract:"
        "MVM mode extraction compute unit",
        "ctest:run registered CTest tests",
    };
    milk_help_see_also(seealso, 2, mh_color);
}


/* ================================================================
 * Main
 * ============================================================= */

int main(int argc, char *argv[])
{
    /* --- Handle help flags (before getopt) --- */
    int action = milk_help_init(argc, argv, APP_DESCRIPTION, APP_DESCRIPTION_LONG);

    if (action == MH_ACTION_H1 || action == MH_ACTION_H2)
    {
        return 0;
    }

    int mh_color = (action == MH_ACTION_HELP);
    if (action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_help(argv[0], mh_color);
        return 0;
    }

    /* --- Parse test-specific flags --- */
    int verbose            = 0;
    int run_cpu            = 1;
    int run_gpu            = 1;
    int explicit_selection = 0;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0)
        {
            verbose = 1;
        }
        else if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--cpu") == 0)
        {
            explicit_selection = 1;
            run_cpu            = 1;
            run_gpu            = 0;
        }
        else if (strcmp(argv[i], "-g") == 0 || strcmp(argv[i], "--gpu") == 0)
        {
            explicit_selection = 1;
            run_cpu            = 0;
            run_gpu            = 1;
        }
    } // for args

    /* Allow -c and -g together */
    if (explicit_selection == 0)
    {
        run_cpu = 1;
        run_gpu = 1;
    }

    printf("=== MVM Extract Modes Test Suite ===\n");
    printf("Image size: %dx%d (%d pixels), %d modes\n", NPIX_X, NPIX_Y, NPIX, NMODES);

#ifdef HAVE_CUDA
    {
        int dev_count = 0;
        cudaGetDeviceCount(&dev_count);
        printf("CUDA devices: %d\n", dev_count);
        if (dev_count == 0)
        {
            run_gpu = 0;
        }
    }
#else
    if (run_gpu)
    {
        printf("CUDA: not compiled in\n");
        run_gpu = 0;
    }
#endif

    printf("Running: %s%s%s\n\n", run_cpu ? "CPU" : "", (run_cpu && run_gpu) ? " + " : "",
           run_gpu ? "GPU" : "");

    int failures = 0;

    /* --- CPU tests --- */
    if (run_cpu)
    {
        failures += test_cpu_roundtrip(verbose);
        failures += test_cpu_zero_input(verbose);
        failures += test_cpu_orthogonality(verbose);
        failures += test_cpu_shm_roundtrip(verbose);
        failures += test_cpu_stress(verbose);
    }

    /* --- GPU tests --- */
#ifdef HAVE_CUDA
    if (run_gpu)
    {
        failures += test_gpu_roundtrip(verbose);
        failures += test_gpu_orthogonality(verbose);
        failures += test_gpu_stress(verbose);
        failures += test_cpu_gpu_match(verbose);
    }
#endif

    printf("\n=== Results: %d test(s) failed ===\n", failures);

    return failures > 0 ? 1 : 0;
}
