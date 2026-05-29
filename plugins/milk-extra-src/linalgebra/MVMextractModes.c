/**
 * @file MVMextractModes.c
 * @brief Mvmextractmodes module
 */

#include "ImageStreamIO/ImageStruct.h"
#ifdef HAVE_CUDA
#    include <cublas_v2.h>
#    include <cuda_runtime.h>
#    include <cuda_runtime_api.h>
#    include <cusolverDn.h>
#    include <device_types.h>
#endif


#include <pthread.h>


// Use MKL if available
// Otherwise use openBLAS
//
#ifdef HAVE_MKL
#    include "mkl.h"
#    define BLASLIB "IntelMKL"
#else
#    ifdef HAVE_OPENBLAS
#        include <cblas.h>
#        include <lapacke.h>
#        define BLASLIB "OpenBLAS"
#    endif
#endif


#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "libmilkcommon/pixel_dispatch.h"
#include "timeutils.h"

#include "MVM_CPU.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "MVMmextrmodes",
    .cmdkey           = "MVMmextrmodes",
    .description      = "extract modes by MVM",
    .description_long = "Extract modal coefficients from a wavefront by matrix-vector "
                        "multiplication. Projects the input onto a pre-computed mode basis."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static int32_t                          *GPUindex                                  = NULL;
static uint32_t *__attribute__((unused)) mmax                                      = NULL;
static uint32_t *__attribute__((unused)) nmax                                      = NULL;
static char                              insname[FUNCTION_PARAMETER_STRMAXLEN]     = "";
static char                              inmasksname[FUNCTION_PARAMETER_STRMAXLEN] = "";
static char                              immodes[FUNCTION_PARAMETER_STRMAXLEN]     = "";
static char                              outcoeff[FUNCTION_PARAMETER_STRMAXLEN]    = "";
static int64_t *__attribute__((unused))  outinit                                   = NULL;
static uint32_t                         *axmode                                    = NULL;
static int64_t                          *PROCESS                                   = NULL;
static int64_t                          *TRACEMODE                                 = NULL;
static int64_t                          *MODENORM                                  = NULL;
static char *__attribute__((unused))     intot_stream                              = NULL;
static char                              inrefsname[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static char                              outrefsname[FUNCTION_PARAMETER_STRMAXLEN] = "";
static uint64_t *__attribute__((unused)) twait                                     = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                           \
    X(".GPUindex", &GPUindex, FPTYPE_INT32, 1, FPFLAG_DEFAULT_INPUT, "GPU index, 99 for CPU")   \
    X(".insname", insname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input stream name")     \
    X(".inmasksname", inmasksname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,                  \
      "nput mask stream name")                                                                  \
    X(".immodes", immodes, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "modes stream name")     \
    X(".outcoeff", outcoeff, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "output coefficients") \
    X(".option.sname_refin", inrefsname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,            \
      "optional input reference to be subtracted stream")                                       \
    X(".option.sname_refout", outrefsname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,          \
      "optional output reference to be subtracted stream")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)

static MILK_HOT errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    int MODEVALCOMPUTE = 1; // 1 if compute, 0 if import


#ifdef HAVE_CUDA
    cublasHandle_t        cublasH       = NULL;
    cublasStatus_t        cublas_status = CUBLAS_STATUS_SUCCESS;
    cudaError_t           cudaStat      = cudaSuccess;
    struct cudaDeviceProp deviceProp;
#endif

    float *d_modes __attribute__((unused))   = NULL; // linear memory of GPU
    float *d_in __attribute__((unused))      = NULL;
    float *d_modeval __attribute__((unused)) = NULL;


    // each step is 2x longer average than previous step
    uint32_t NBaveSTEP = 10;

    int initref __attribute__((unused)) = 0; // 1 when reference has been processed

    // CONNECT TO INPUT STREAM
    IMGID imgin = imgid_make_from_name(insname);
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);
    printf("Input stream size : %u %u\n", imgin.md->size[0], imgin.md->size[1]);
    if (imgin.ID == -1)
    {
        return RETURN_FAILURE;
    }
    long m = imgin.md->size[0] * imgin.md->size[1];

    // CONNECT TO MASK STREAM
    int       use_mask   = 0;    //flag indicating that the mask is being used
    uint32_t  mask_npix  = 0;    //The number of 1 pixels in the mask
    uint32_t *mask_idx   = NULL; //Array holding the indices of the 1 pixels
    float    *masked_pix = NULL; //Array to hold the pixel values

    IMGID imgmask = imgid_make_from_name(inmasksname);
    if (resolveIMGID(&imgmask, ERRMODE_WARN, dcimg, dcnimg) != -1)
    {
        printf("Mask stream size : %u %u\n", imgmask.md->size[0], imgmask.md->size[1]);
        if (imgmask.md->size[0] == imgin.md->size[0] && imgmask.md->size[1] == imgin.md->size[1])
        {
            use_mask = 1;
        }
    }

    printf("USE MASK = %d\n", use_mask);

    //use_mask = 0; //for testing

    //setup the mask
    //
    if (use_mask)
    {
        for (long n = 0; n < imgmask.md->size[0] * imgmask.md->size[1]; ++n)
        {
            if (imgmask.im->array.F[n] == 1)
            {
                ++mask_npix;
            }
        }

        mask_idx   = (uint32_t *) malloc(mask_npix * sizeof(long));
        masked_pix = (float *) malloc(mask_npix * sizeof(float));
        long nn    = 0;
        for (long n = 0; n < imgmask.md->size[0] * imgmask.md->size[1]; ++n)
        {
            if (imgmask.im->array.F[n] == 1)
            {
                mask_idx[nn] = n;
                ++nn;
            }
        }

        printf("Mask has : %u pixels (%f%%)\n", mask_npix,
               (100.0 * mask_npix) / (imgmask.md->size[0] * imgmask.md->size[1]));
    }
    else
    {
        //Just use full image
        mask_npix = imgin.md->size[0] * imgin.md->size[1];
        printf("No mask using : %u pixels (%f%%)\n", mask_npix,
               (100.0 * mask_npix) / (imgin.md->size[0] * imgin.md->size[1]));
    }


    /* // This was probaly never implemented at all.
    // NORMALIZATION
    // CONNECT TO TOTAL FLUX STREAM
    imageID IDintot;
    IDintot = image_ID(intot_stream, dcimg, dcnimg);
    int INNORMMODE = 0; // 1 if input normalized

    if(IDintot == -1)
    {
        INNORMMODE = 0;
        create_2Dimage_ID("intot_tmp", 1, 1, &IDintot);
        dcimg[IDintot].array.F[0] = 1.0f;
    }
    else
    {
        INNORMMODE = 1;
    }
    */


    // CONNECT TO OPTIONAL INPUT REFERENCE STREAM
    imageID IDinref  = -1;
    IMGID   imginref = imgid_make_from_name(inrefsname);
    resolveIMGID(&imginref, ERRMODE_WARN, dcimg, dcnimg);
    if (imginref.ID == -1)
    {
        IMGID imgref = imgid_make_from_name_2D("_tmprefin", imgin.md->size[0], imgin.md->size[1]);
        imgref.mdt->shared = 0;
        imgref.im          = (IMAGE *) calloc(1, sizeof(IMAGE));
        imgid_mkimage(&imgref);
        /* calloc zeros the array */
        IDinref = imgref.ID;
    }
    else
    {
        IDinref = imginref.ID;
    }


    // CONNECT TO OPTIONAL OUTPUT REFERENCE STREAM
    IMGID imgoutref = imgid_make_from_name(outrefsname);
    resolveIMGID(&imgoutref, ERRMODE_WARN, dcimg, dcnimg);


    // CONNECT TO MODES STREAM
    IMGID imgmodes = imgid_make_from_name(immodes);
    resolveIMGID(&imgmodes, ERRMODE_WARN, dcimg, dcnimg);

    // Could this be imgid_compare?
    if (imgmodes.md->datatype != _DATATYPE_FLOAT)
    {
        PRINT_ERROR("Cannot operate with modes other than FP32!!!s");
        if (imgmodes.ID == -1)
        {
            return RETURN_FAILURE;
        }
        abort();
    }

    printf("Modes stream size : %u %u\n", imgmodes.md->size[0], imgmodes.md->size[1]);


    long    n;
    long    NBmodes                         = 1;
    imageID IDmodes __attribute__((unused)) = -1;


    if ((*axmode) == 0)
    {
        //
        // Extract modes.
        // This is the default geometry, no need to remap
        //
        n       = imgmodes.md->size[2];
        IDmodes = imgmodes.ID;
        NBmodes = n;
        printf("NBmodes = %ld\n", NBmodes);
        fflush(stdout);


        // make col-major storage
    }
    else
    {
        //
        // Expand
        // Remap to new matrix tmpmodes
        //

        NBmodes = imgmodes.md->size[0] * imgmodes.md->size[1];
        n       = NBmodes;
        printf("NBmodes = %ld\n", NBmodes);
        fflush(stdout);

        printf("creating _tmpmodes  %ld %ld %ld\n", (long) imgin.md->size[0],
               (long) imgin.md->size[1], NBmodes);
        fflush(stdout);

        IMGID imgtmp =
            imgid_make_from_name_3D("_tmpmodes", imgin.md->size[0], imgin.md->size[1], NBmodes);
        imgtmp.mdt->shared = 0;
        imgtmp.im          = (IMAGE *) calloc(1, sizeof(IMAGE));
        imgid_mkimage(&imgtmp);
        IDmodes = imgtmp.ID;

        for (uint32_t ii = 0; ii < imgin.md->size[0]; ii++)
        {
            for (uint32_t jj = 0; jj < imgin.md->size[1]; jj++)
            {
                for (long kk = 0; kk < NBmodes; kk++)
                {
                    imgtmp.im->array.F[kk * imgin.md->size[0] * imgin.md->size[1] +
                                       jj * imgin.md->size[0] + ii] =
                        imgmodes.im->array.F[NBmodes * (jj * imgin.md->size[0] + ii) + kk];
                }
            }
        }

        // save_fits("_tmpmodes", "_test_tmpmodes.fits");
    }

    float *normcoeff = (float *) malloc(sizeof(float) * NBmodes);

    if ((*MODENORM) == 1)
    {
        // In this mode, the input modes are normalized to unity (vector 2-norm)
        // norm is computed here


        // compute normalization coeffs
        for (long k = 0; k < NBmodes; k++)
        {
            normcoeff[k] = 0.0;
            for (long ii = 0; ii < m; ii++)
            {
                normcoeff[k] += imgmodes.im->array.F[k * m + ii] * imgmodes.im->array.F[k * m + ii];
            }
        }
    }
    else
    {
        // or set them to 1
        for (long k = 0; k < NBmodes; k++)
        {
            normcoeff[k] = 1.0;
        }
    }

    float *modevalarray    = (float *) malloc(sizeof(float) * n);
    float *modevalarrayref = (float *) malloc(sizeof(float) * n);

    uint32_t *arraytmp = (uint32_t *) malloc(sizeof(uint32_t) * 2);

    // IDrefout = image_ID(IDrefout_name, dcimg, dcnimg);
    imageID IDrefout = -1; // TODO handle this
    if (IDrefout == -1)
    {
        if ((*axmode) == 0)
        {
            arraytmp[0] = NBmodes;
            arraytmp[1] = 1;
        }
        else
        {
            arraytmp[0] = imgmodes.md->size[0];
            arraytmp[1] = imgmodes.md->size[1];
        }
    }
    else
    {
        arraytmp[0] = dcimg[IDrefout].md->size[0];
        arraytmp[1] = dcimg[IDrefout].md->size[1];
    }


    // CONNNECT TO OR CREATE OUTPUT STREAM
    IMGID imgout = stream_connect_create_2Df32(outcoeff, arraytmp[0], arraytmp[1]);

    // Local working copy of output
    float *outarray = (float *) malloc(sizeof(float) * arraytmp[0] * arraytmp[1]);


    MODEVALCOMPUTE = 1;

    free(arraytmp);


    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT;


    if (MODEVALCOMPUTE == 1)
    {
        if (((*GPUindex) >= 0) && ((*GPUindex) != 99))
        {
#ifdef HAVE_CUDA
            int deviceCount;
            int devicecntMax = 100;

            cudaGetDeviceCount(&deviceCount);
            printf("%d devices found\n", deviceCount);
            fflush(stdout);

            processinfo_WriteMessage_fmt(processinfo, "CUDA : %d devices", deviceCount);

            if (deviceCount > devicecntMax)
            {
                deviceCount = 0;
            }
            if (deviceCount < 0)
            {
                deviceCount = 0;
            }

            printf("\n");

            for (int k = 0; k < deviceCount; k++)
            {
                cudaGetDeviceProperties(&deviceProp, k);

                int clockRate;
                cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, k);

                printf("Device %d / %d [ %20s ]  has compute capability %d.%d.\n", k + 1,
                       deviceCount, deviceProp.name, deviceProp.major, deviceProp.minor);
                printf("  Total amount of global memory:                 %.0f MBytes "
                       "(%llu bytes)\n",
                       (float) deviceProp.totalGlobalMem / 1048576.0f,
                       (unsigned long long) deviceProp.totalGlobalMem);
                printf("  (%2d) Multiprocessors\n", deviceProp.multiProcessorCount);
                printf("  GPU Clock rate:                                %.0f MHz "
                       "(%0.2f GHz)\n",
                       clockRate * 1e-3f, clockRate * 1e-6f);
                printf("\n");
            }

            if ((*GPUindex) < deviceCount)
            {
                cudaSetDevice(*GPUindex);
            }
            else
            {
                printf("Invalid Device : %d / %d\n", *GPUindex, deviceCount);
                processinfo_WriteMessage_fmt(processinfo, "Invalid GPU device %d", *GPUindex);
                exit(0);
            }

            printf("Create cublas handle ...");
            fflush(stdout);
            cublas_status = cublasCreate(&cublasH);
            if (cublas_status != CUBLAS_STATUS_SUCCESS)
            {
                printf("CUBLAS initialization failed\n");
                return EXIT_FAILURE;
            }
            printf(" done\n");
            fflush(stdout);

            long   matsz;
            float *modesmat;

            if (use_mask)
            {
                //reformat the matrix using the mask
                matsz    = mask_npix * NBmodes;
                modesmat = (float *) malloc(sizeof(float) * mask_npix * dcimg[IDmodes].md->size[2]);

                uint32_t nrows = dcimg[IDmodes].md->size[2];
                uint32_t ncols = dcimg[IDmodes].md->size[0] * dcimg[IDmodes].md->size[1];

                for (uint32_t rr = 0; rr < nrows; ++rr)
                {
                    for (uint32_t cc = 0; cc < mask_npix; ++cc)
                    {
                        modesmat[rr * mask_npix + cc] =
                            dcimg[IDmodes].array.F[rr * ncols + mask_idx[cc]];
                    }
                }
            }
            else
            {
                matsz    = m * NBmodes;
                modesmat = dcimg[IDmodes].array.F;
            }

            // load modes to GPU
            cudaStat = cudaMalloc((void **) &d_modes, sizeof(float) * matsz);
            if (cudaStat != cudaSuccess)
            {
                printf("cudaMalloc d_modes returned error code %d, line %d\n", cudaStat, __LINE__);
                exit(EXIT_FAILURE);
            }

            cudaStat = cudaMemcpy(d_modes, modesmat, sizeof(float) * matsz, cudaMemcpyHostToDevice);
            // cudaStat = cudaMemcpy(d_modes, imgmodes.im->array.F, sizeof(float) * m * NBmodes, cudaMemcpyHostToDevice);

            if (use_mask)
            {
                free(modesmat);
            }

            if (cudaStat != cudaSuccess)
            {
                printf("cudaMemcpy returned error code %d, line %d\n", cudaStat, __LINE__);
                exit(EXIT_FAILURE);
            }


            // create d_in
            cudaStat = cudaMalloc((void **) &d_in, sizeof(float) * m);
            if (cudaStat != cudaSuccess)
            {
                printf("cudaMalloc d_in returned error code %d, line %d\n", cudaStat, __LINE__);
                exit(EXIT_FAILURE);
            }

            // create d_modeval
            cudaStat = cudaMalloc((void **) &d_modeval, sizeof(float) * NBmodes);
            if (cudaStat != cudaSuccess)
            {
                printf("cudaMalloc d_modeval returned error code %d, line %d\n", cudaStat,
                       __LINE__);
                exit(EXIT_FAILURE);
            }
#else
            processinfo_WriteMessage(processinfo, "NO CUDA - CPU fallback");
            *GPUindex = 99;
#endif
        }
    }

    if ((*TRACEMODE) == 1)
    {
        char    traceim_name[STRINGMAXLEN_IMGNAME];
        long    TRACEsize = 2000;
        imageID IDtrace __attribute__((unused));

        uint32_t *sizearraytmp = (uint32_t *) malloc(sizeof(uint32_t) * 2);

        {
            int slen = snprintf(traceim_name, STRINGMAXLEN_IMGNAME, "%s_trace", outcoeff);
            if (slen < 1)
            {
                PRINT_ERROR("snprintf wrote <1 char");
                abort(); // can't handle this error any other way
            }
            if (slen >= STRINGMAXLEN_IMGNAME)
            {
                PRINT_ERROR("snprintf string truncation");
                abort(); // can't handle this error any other way
            }
        }

        IMGID imgtrace = imgid_make_from_name(traceim_name);
        resolveIMGID(&imgtrace, ERRMODE_NULL, dcimg, dcnimg);
        int imOK = 1;
        if (imgtrace.ID == -1)
        {
            imOK = 0;
        }
        else
        {
            if ((imgtrace.md->size[0] != TRACEsize) || (imgtrace.md->size[1] != NBmodes))
            {
                imOK = 0;
                delete_image_ID(traceim_name, DELETE_IMAGE_ERRMODE_WARNING);
            }
        }
        if (imOK == 0)
        {
            imgtrace.mdt->naxis    = 2;
            imgtrace.mdt->size[0]  = TRACEsize;
            imgtrace.mdt->size[1]  = NBmodes;
            imgtrace.mdt->datatype = _DATATYPE_FLOAT;
            imgtrace.mdt->shared   = 1;
            imgtrace.im            = (IMAGE *) calloc(1, sizeof(IMAGE));
            imgid_mkimage(&imgtrace);
        }
        IDtrace = imgtrace.ID;
        free(sizearraytmp);
    }

    if ((*PROCESS) == 1)
    {
        char    process_ave_name[STRINGMAXLEN_IMGNAME];
        char    process_rms_name[STRINGMAXLEN_IMGNAME];
        imageID IDprocave __attribute__((unused));
        imageID IDprocrms __attribute__((unused));

        uint32_t *sizearraytmp = (uint32_t *) malloc(sizeof(uint32_t) * 2);

        {
            int slen = snprintf(process_ave_name, STRINGMAXLEN_IMGNAME, "%s_ave", outcoeff);
            if (slen < 1)
            {
                PRINT_ERROR("snprintf wrote <1 char");
                abort(); // can't handle this error any other way
            }
            if (slen >= STRINGMAXLEN_IMGNAME)
            {
                PRINT_ERROR("snprintf string truncation");
                abort(); // can't handle this error any other way
            }
        }

        sizearraytmp[0]  = NBmodes;
        sizearraytmp[1]  = NBaveSTEP;
        IMGID imgprocave = imgid_make_from_name(process_ave_name);
        resolveIMGID(&imgprocave, ERRMODE_NULL, dcimg, dcnimg);
        int imOK = 1;
        if (imgprocave.ID == -1)
        {
            imOK = 0;
        }
        else
        {
            if ((imgprocave.md->size[0] != NBmodes) || (imgprocave.md->size[1] != NBaveSTEP))
            {
                imOK = 0;
                delete_image_ID(process_ave_name, DELETE_IMAGE_ERRMODE_WARNING);
            }
        }
        if (imOK == 0)
        {
            imgprocave.mdt->naxis    = 2;
            imgprocave.mdt->size[0]  = NBmodes;
            imgprocave.mdt->size[1]  = NBaveSTEP;
            imgprocave.mdt->datatype = _DATATYPE_FLOAT;
            imgprocave.mdt->shared   = 1;
            imgprocave.im            = (IMAGE *) calloc(1, sizeof(IMAGE));
            imgid_mkimage(&imgprocave);
        }
        IDprocave = imgprocave.ID;
        free(sizearraytmp);

        sizearraytmp = (uint32_t *) malloc(sizeof(uint32_t) * 2);

        {
            int slen = snprintf(process_rms_name, STRINGMAXLEN_IMGNAME, "%s_rms", outcoeff);
            if (slen < 1)
            {
                PRINT_ERROR("snprintf wrote <1 char");
                abort(); // can't handle this error any other way
            }
            if (slen >= STRINGMAXLEN_IMGNAME)
            {
                PRINT_ERROR("snprintf string truncation");
                abort(); // can't handle this error any other way
            }
        }

        sizearraytmp[0]  = NBmodes;
        sizearraytmp[1]  = NBaveSTEP;
        IMGID imgprocrms = imgid_make_from_name(process_rms_name);
        resolveIMGID(&imgprocrms, ERRMODE_NULL, dcimg, dcnimg);
        imOK = 1;
        if (imgprocrms.ID == -1)
        {
            imOK = 0;
        }
        else
        {
            if ((imgprocrms.md->size[0] != NBmodes) || (imgprocrms.md->size[1] != NBaveSTEP))
            {
                imOK = 0;
                delete_image_ID(process_rms_name, DELETE_IMAGE_ERRMODE_WARNING);
            }
        }
        if (imOK == 0)
        {
            imgprocrms.mdt->naxis    = 2;
            imgprocrms.mdt->size[0]  = NBmodes;
            imgprocrms.mdt->size[1]  = NBaveSTEP;
            imgprocrms.mdt->datatype = _DATATYPE_FLOAT;
            imgprocrms.mdt->shared   = 1;
            imgprocrms.im            = (IMAGE *) calloc(1, sizeof(IMAGE));
            imgid_mkimage(&imgprocrms);
        }
        IDprocrms = imgprocrms.ID;
        free(sizearraytmp);
    }

    initref = 0; // 1 when reference has been processed

    // long twait1 = *twait;

    printf("LOOP START   MODEVALCOMPUTE = %d\n", MODEVALCOMPUTE);
    fflush(stdout);

    if (MODEVALCOMPUTE == 0)
    {
        printf("\n");
        printf("This function is NOT computing mode values\n");
        printf("Pre-existing stream %s was detected\n", outcoeff);
        printf("\n");
    }
    else
    {
        char msgstring[STRINGMAXLEN_PROCESSINFO_STATUSMSG];

        {
            int slen = snprintf(msgstring, STRINGMAXLEN_PROCESSINFO_STATUSMSG, "Running on GPU %d",
                                (*GPUindex));
            if (slen < 1)
            {
                PRINT_ERROR("snprintf wrote <1 char");
                abort(); // can't handle this error any other way
            }
            if (slen >= STRINGMAXLEN_PROCESSINFO_STATUSMSG)
            {
                PRINT_ERROR("snprintf string truncation");
                abort(); // can't handle this error any other way
            }
        }
    }

    printf(" m       = %u\n", mask_npix);
    printf(" n       = %ld\n", n);
    printf(" NBmodes = %ld\n", NBmodes);

    float    alpha __attribute__((unused)) = 1.0;
    float    beta __attribute__((unused))  = 0.0;
    uint64_t refindex                      = 0;

#ifdef HAVE_OPENBLAS
    printf("OpenBLASS  YES\n");
#else
    printf("OpenBLASS  NO\n");
#endif

#ifdef HAVE_MKL
    printf("MKL        YES\n");
#else
    printf("MKL        NO\n");
#endif


#ifdef HAVE_CUDA
    printf("CUDA       YES\n");
#else
    printf("CUDA       NO\n");
#endif


    float *ColMajorMatrix = (float *) malloc(sizeof(float) * m * n);
    if (*axmode == 0)
    {
        for (int ii = 0; ii < m; ii++)
        {
            for (int jj = 0; jj < n; jj++)
            {
                ColMajorMatrix[ii * n + jj] = imgmodes.im->array.F[jj * m + ii];
            }
        }
    }
    else
    {
        memcpy(ColMajorMatrix, imgmodes.im->array.F, sizeof(float) * m * n);
    }


    float *imginfloatptr = NULL;


    if (imgin.md->datatype == _DATATYPE_FLOAT)
    {
        imginfloatptr = imgin.im->array.F;
        printf("INPUT type = FLOAT  - no type conversion required\n");
    }
    else
    {
        imginfloatptr = (float *) malloc(sizeof(float) * imgin.md->size[0] * imgin.md->size[1]);
        printf("INPUT not float -> type conversion to float enabled\n");
    }


    printf(">>> START MVM loop\n");

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        // Are we computing a new reference ?
        // if yes, set initref to 0 (reference is NOT initialized)
        //
        if (refindex != dcimg[IDinref].md->cnt0)
        {
            initref  = 0;
            refindex = dcimg[IDinref].md->cnt0;
        }


        if (((*GPUindex) < 0) || (*GPUindex == 99))
        {
            // using CPU

#ifdef BLASLIB
            struct timespec t0, t1;
            clock_gettime(CLOCK_MILK, &t0);
            processinfo_WriteMessage_fmt(processinfo, "imgout %s ID %d", imgout.md->name,
                                         imgout.ID);
            {
                float beta = 0.0;

                if (imgoutref.ID != -1)
                {
                    beta = 1.0;
                    memcpy(outarray, imgoutref.im->array.F, sizeof(float) * n);
                }


                if (imgin.md->datatype != _DATATYPE_FLOAT)
                {
                    // type conversion to float
                    uint64_t npix = (uint64_t) imgin.md->size[0] * imgin.md->size[1];

#    define _MVM_CONV_CASE(DT, ACC, CTYPE)                         \
    case DT:                                                       \
        for (uint64_t _ii = 0; _ii < npix; _ii++)                  \
        {                                                          \
            imginfloatptr[_ii] = (float) imgin.im->array.ACC[_ii]; \
        }                                                          \
        break;

                    switch (imgin.md->datatype)
                    {
                        FOREACH_REAL_DATATYPE(_MVM_CONV_CASE)
                    default:
                        break;
                    }
#    undef _MVM_CONV_CASE
                }


                if (*axmode == 1)
                {
                    cblas_sgemv(CblasColMajor, CblasNoTrans, (int) n, (int) m, 1.0, ColMajorMatrix,
                                (int) n, imginfloatptr, 1, beta, outarray, 1);
                }
                else
                {
                    cblas_sgemv(CblasColMajor, CblasNoTrans, (int) n, (int) m, 1.0, ColMajorMatrix,
                                (int) n, imginfloatptr, 1, beta, outarray, 1);
                }

                clock_gettime(CLOCK_MILK, &t1);
                struct timespec tdiff;
                tdiff       = timespec_diff(t0, t1);
                double t01d = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
                processinfo_WriteMessage_fmt(processinfo, "%s %dx%d MVM %.3f us", BLASLIB, n, m,
                                             t01d * 1e6);
            }
#else
            // Run on CPU without lib
            int mmax1 = (*mmax);
            if (mmax1 > m)
            {
                mmax1 = m;
            }

            int nmax1 = (*nmax);
            if (nmax1 > n)
            {
                nmax1 = n;
            }

            for (int jj = 0; jj < n; jj++)
            {
                outarray[jj] = 0.0;
            }


            for (int ii = 0; ii < m; ii++)
            {
                for (int jj = 0; jj < n; jj++)
                {
                    int index = ii * n + jj;
                    outarray[jj] += imgmodes.im->array.F[index] * imginfloatptr[ii];
                }
            }

#endif

            // update output
            dcimg[imgout.ID].md->write = 1;
            for (int jj = 0; jj < n; jj++)
            {
                imgout.im->array.F[jj] = outarray[jj] / normcoeff[jj];
            }
            //            memcpy(imgout.im->array.F, outarray, sizeof(float)*n);
            processinfo_update_output_stream(processinfo, imgout.im, NULL);
        }
        else
        {
            // running on GPU
#ifdef HAVE_CUDA

            struct timespec t0, t1;
            clock_gettime(CLOCK_MILK, &t0);

            // load in_stream to GPU
            if (initref == 0)
            {
                if (use_mask == 1)
                {
                    for (uint32_t cc = 0; cc < mask_npix; ++cc)
                    {
                        masked_pix[cc] = dcimg[IDinref].array.F[mask_idx[cc]];
                    }
                }
                else
                {
                    masked_pix = dcimg[IDinref].array.F;
                }
                cudaStat =
                    cudaMemcpy(d_in, masked_pix, sizeof(float) * mask_npix, cudaMemcpyHostToDevice);
            }
            else
            {
                if (use_mask == 1)
                {
                    for (uint32_t cc = 0; cc < mask_npix; ++cc)
                    {
                        masked_pix[cc] = imginfloatptr[mask_idx[cc]];
                    }
                }
                else
                {
                    masked_pix = imginfloatptr;
                }
                cudaStat =
                    cudaMemcpy(d_in, masked_pix, sizeof(float) * mask_npix, cudaMemcpyHostToDevice);
            }

            if (cudaStat != cudaSuccess)
            {
                printf("initref = %d    %ld  %ld\n", initref, IDinref, imgin.ID);
                printf("cudaMemcpy returned error code %d, line %d\n", cudaStat, __LINE__);
                exit(EXIT_FAILURE);
            }

            // compute
            cublas_status = cublasSgemv(cublasH, CUBLAS_OP_T, mask_npix, NBmodes, &alpha, d_modes,
                                        mask_npix, d_in, 1, &beta, d_modeval, 1);
            if (cublas_status != CUBLAS_STATUS_SUCCESS)
            {
                printf("cublasSgemv returned error code %d, line(%d)\n", cublas_status, __LINE__);
                fflush(stdout);
                if (cublas_status == CUBLAS_STATUS_NOT_INITIALIZED)
                {
                    printf("   CUBLAS_STATUS_NOT_INITIALIZED\n");
                }
                if (cublas_status == CUBLAS_STATUS_INVALID_VALUE)
                {
                    printf("   CUBLAS_STATUS_INVALID_VALUE\n");
                }
                if (cublas_status == CUBLAS_STATUS_ARCH_MISMATCH)
                {
                    printf("   CUBLAS_STATUS_ARCH_MISMATCH\n");
                }
                if (cublas_status == CUBLAS_STATUS_EXECUTION_FAILED)
                {
                    printf("   CUBLAS_STATUS_EXECUTION_FAILED\n");
                }

                printf("GPU index                           = %d\n", (*GPUindex));

                printf("CUBLAS_OP                           = %d\n", CUBLAS_OP_T);
                printf("alpha                               = %f\n", alpha);
                printf("alpha                               = %f\n", beta);
                printf("m                                   = %d\n", (int) m);
                printf("NBmodes                             = %d\n", (int) NBmodes);
                fflush(stdout);
                exit(EXIT_FAILURE);
            }

            // copy result
            imgout.md->write = 1;

            if (initref == 0)
            {
                // construct reference to be subtracted
                printf("... reference compute\n");
                cudaStat = cudaMemcpy(modevalarrayref, d_modeval, sizeof(float) * NBmodes,
                                      cudaMemcpyDeviceToHost);

                IDrefout = image_ID(outrefsname, dcimg, dcnimg);
                if (IDrefout != -1)
                {
                    for (long k = 0; k < NBmodes; k++)
                    {
                        modevalarrayref[k] -= dcimg[IDrefout].array.F[k];
                    }
                }
            }
            else
            {
                cudaStat = cudaMemcpy(modevalarray, d_modeval, sizeof(float) * NBmodes,
                                      cudaMemcpyDeviceToHost);


                for (long k = 0; k < NBmodes; k++)
                {
                    imgout.im->array.F[k] = (modevalarray[k] - modevalarrayref[k]) / normcoeff[k];
                    // Renorm was never implemented
                    // (modevalarray[k] / dcimg[IDintot].array.F[0] - modevalarrayref[k]) / normcoeff[k];
                }


                clock_gettime(CLOCK_MILK, &t1);
                struct timespec tdiff;
                tdiff       = timespec_diff(t0, t1);
                double t01d = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
                processinfo_WriteMessage_fmt(processinfo, "GPU%d %dx%d MVM %.3f us", *GPUindex, n,
                                             m, t01d * 1e6);


                processinfo_update_output_stream(processinfo, imgout.im, NULL);
            }
#endif
        }

        initref = 1;
    }

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    free(outarray);

    free(ColMajorMatrix);

    free(normcoeff);
    free(modevalarray);
    free(modevalarrayref);


    if (imgin.md->datatype != _DATATYPE_FLOAT)
    {
        free(imginfloatptr);
    }


    if (use_mask)
    {
        free(masked_pix);
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#ifndef FPS_STANDALONE
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_linalgebra__MVMextractModes()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)
#endif
