// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file MVMextractModes.c
 * @brief Mvmextractmodes module
 */

// MILK_CMAKE_REQUEST_CUDA
// MILK_CMAKE_REQUEST_BLAS

#include "ImageStreamIO/ImageStruct.h"

#ifdef HAVE_CUDA
#    include <cublas_v2.h>
#    include <cuda_runtime.h>
#    include <cuda_runtime_api.h>
#    include <cusolverDn.h>
#    include <device_types.h>
#endif

#include <memory> // unique_ptr
#include "mvm_auxiliaries.hpp"


extern "C"
{

#include "milk_blas_lapacke.h"

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "libmilkcommon/pixel_dispatch.h"
#include "timeutils.h"
#include "linalgebra.h"

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

static int32_t  GPUindex                                  = 0;
static char     insname[FUNCTION_PARAMETER_STRMAXLEN]     = "";
static char     inmasksname[FUNCTION_PARAMETER_STRMAXLEN] = "";
static char     immodes[FUNCTION_PARAMETER_STRMAXLEN]     = "";
static char     outcoeff[FUNCTION_PARAMETER_STRMAXLEN]    = "";
static uint32_t axis_mode                                 = 1;
static int64_t  opt_modenorm                              = 1;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                           \
    X(".GPUindex", &GPUindex, FPTYPE_INT32, 1, FPFLAG_DEFAULT_INPUT,                            \
      "GPU index, 99 for CPU [BLAS if avail.], 98 for plain CPU [OMP if avail.]")               \
    X(".insname", insname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input stream name")     \
    X(".inmasksname", inmasksname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,                  \
      "spatial mask stream name")                                                               \
    X(".immodes", immodes, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "modes stream name")     \
    X(".outcoeff", outcoeff, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "output coefficients") \
    X(".axmode", &axis_mode, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,                            \
      "axis mode: 0=extract, 1=expand")                                                         \
    X(".option.MODENORM", &opt_modenorm, FPTYPE_ONOFF, 0, FPFLAG_DEFAULT_INPUT,                 \
      "normalize modes to unit 2-norm")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)


enum class ComputeMode : int
{
    OMP_OR_PLAIN = 0,
    BLAS         = 1,
    CUDA         = 2
};

errno_t cuda_attempt_init(PROCESSINFO *processinfo)
{
#ifdef HAVE_CUDA
    int                   deviceCount;
    struct cudaDeviceProp deviceProp;

    cudaGetDeviceCount(&deviceCount);
    printf("%d devices found\n", deviceCount);
    fflush(stdout);
    processinfo_WriteMessage_fmt(processinfo, "CUDA : %d devices", deviceCount);

    if (deviceCount < 0 || deviceCount > 100)
    {
        return RETURN_FAILURE;
    }


    for (int k = 0; k < deviceCount; k++)
    {
        cudaGetDeviceProperties(&deviceProp, k);

        int clockRate;
        cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, k);

        printf("Device %d / %d [ %20s ]  has compute capability %d.%d.\n", k + 1, deviceCount,
               deviceProp.name, deviceProp.major, deviceProp.minor);
        printf("  Total amount of global memory:                 %.0f MBytes "
               "(%llu bytes)\n",
               (float) deviceProp.totalGlobalMem / 1048576.0f,
               (unsigned long long) deviceProp.totalGlobalMem);
        printf("  (%2d) Multiprocessors\n", deviceProp.multiProcessorCount);
        printf("  GPU Clock rate:                                %.0f MHz "
               "(%0.2f GHz)\n\n",
               clockRate * 1e-3f, clockRate * 1e-6f);
    }

    if (GPUindex < deviceCount)
    {
        cudaSetDevice(GPUindex);
    }
    else
    {
        printf("Invalid Device : %d / %d\n", GPUindex, deviceCount);
        fflush(stdout);
        processinfo_WriteMessage_fmt(processinfo, "Invalid GPU device %d", GPUindex);
        return RETURN_FAILURE;
    }

    return RETURN_SUCCESS;
#else // HAVE_CUDA
    return RETURN_FAILURE;
#endif
}

ComputeMode _compute_mode_determine(PROCESSINFO *processinfo)
{
    if (GPUindex >= 0 && GPUindex != 99 && GPUindex != 98 &&
        RETURN_SUCCESS == cuda_attempt_init(processinfo))
    {
        processinfo_WriteMessage_fmt(processinfo, "Successful CUDA init - GPU %d", GPUindex);
        printf("-------------\nBACKEND: CUDA\n-------------\n");
        return ComputeMode::CUDA;
    }
#ifdef BLASLIB
    if (GPUindex == 99)
    {
        printf("-------------\nBACKEND: BLAS [%s]\n-------------\n", BLASLIB);
        return ComputeMode::BLAS;
    }
#endif

#ifdef _OPENMP
    printf("-------------\nBACKEND: CPU [OPENMP]\n-------------\n");
#else
    printf("-------------\nBACKEND: CPU [BASIC]\n-------------\n");
#endif
    return ComputeMode::OMP_OR_PLAIN;
}

void initializer_new_data_imgmodes(float    *modes_copy,
                                   float    *norm_coeffs,
                                   uint32_t *mask_idx,
                                   uint64_t  mask_npix,
                                   IMAGE    *im_modes)
{
    auto     s     = im_modes->md->size;
    uint64_t n_pix = (uint64_t) s[0] * s[1];
    float   *arr   = im_modes->array.F;
    if (mask_idx != NULL)
    {
        if (opt_modenorm == 1)
        {
            for (uint32_t nn = 0; nn < s[2]; nn++)
            {
                for (uint64_t pp = 0; pp < mask_npix; pp++)
                {
                    norm_coeffs[nn] +=
                        arr[nn * n_pix + mask_idx[pp]] * arr[nn * n_pix + mask_idx[pp]];
                }
                for (uint64_t pp = 0; pp < mask_npix; pp++)
                {
                    modes_copy[nn * mask_npix + pp] =
                        arr[nn * s[0] * s[1] + mask_idx[pp]] / norm_coeffs[nn];
                }
            }
        }
        else
        {
            for (uint32_t nn = 0; nn < s[2]; nn++)
            {
                norm_coeffs[nn] = 1.0f;
                for (uint64_t pp = 0; pp < mask_npix; pp++)
                {
                    modes_copy[nn * mask_npix + pp] = arr[nn * s[0] * s[1] + mask_idx[pp]];
                }
            }
        }
    }
    else
    {
        if (opt_modenorm == 1)
        {
            for (uint32_t nn = 0; nn < s[2]; nn++)
            {
                for (uint64_t pp = 0; pp < n_pix; pp++)
                {
                    norm_coeffs[nn] += arr[nn * n_pix + pp] * arr[nn * n_pix + pp];
                }
                for (uint64_t pp = 0; pp < n_pix; pp++)
                {
                    modes_copy[nn * n_pix + pp] = arr[nn * n_pix + pp] / norm_coeffs[nn];
                }
            }
        }
        else
        {
            for (uint32_t nn = 0; nn < s[2]; nn++)
            {
                norm_coeffs[nn] = 1.0f;
            }
            memcpy(modes_copy, arr, im_modes->md->imdatamemsize);
        }
    }
}

void cast_to_float(float *target, IMAGE *source_image, uint64_t n)
{
#define _MVM_CONV_CASE(DT, ACC, CTYPE)                          \
    case DT:                                                    \
        for (uint64_t _ii = 0; _ii < n; _ii++)                  \
        {                                                       \
            target[_ii] = (float) source_image->array.ACC[_ii]; \
        }                                                       \
        break;

    switch (source_image->md->datatype)
    {
        FOREACH_REAL_DATATYPE(_MVM_CONV_CASE)
    default:
        break;
    }
#undef _MVM_CONV_CASE
}

static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();

    // CONNECT TO INPUT STREAM
    IMGID imgid_in = imgid_make_from_name(insname);
    resolveIMGID(&imgid_in, ERRMODE_WARN, dcimg, dcnimg);
    printf("Input stream size : %u %u\n", imgid_in.md->size[0], imgid_in.md->size[1]);
    if (imgid_in.ID == -1)
    {
        return RETURN_FAILURE;
    }
    uint64_t nb_modes; // later

    // CONNECT TO MASK STREAM
    int       use_mask  = 0;    //flag indicating that the mask is being used
    uint64_t  mask_npix = 0;    //The number of 1 pixels in the mask
    uint32_t *mask_idx  = NULL; //Array holding the indices of the 1 pixels

    // CONNECT TO MODES STREAM
    IMGID imgid_modes         = imgid_make_from_name(immodes);
    imgid_modes.mdt->datatype = _DATATYPE_FLOAT;
    resolveIMGID(&imgid_modes, ERRMODE_FAIL, dcimg, dcnimg);

    auto     modes_size            = imgid_modes.md->size;
    uint64_t n_pixels_spatial_side = modes_size[0] * modes_size[1];
    printf("Modes stream size : %u %u %u\n", modes_size[0], modes_size[1], modes_size[2]);
    nb_modes = modes_size[2];

    // CONNECT TO MASK, INITIALIZE MASKING
    IMGID imgid_mask = imgid_make_from_name(inmasksname);
    if (resolveIMGID(&imgid_mask, ERRMODE_WARN, dcimg, dcnimg) != -1)
    {
        printf("Mask stream size : %u %u\n", imgid_mask.md->size[0], imgid_mask.md->size[1]);
        use_mask =
            (imgid_mask.md->size[0] == modes_size[0] && imgid_mask.md->size[1] == modes_size[1] &&
             imgid_mask.md->datatype == _DATATYPE_FLOAT);
    }
    printf("USE MASK = %d\n", use_mask);


    //setup the mask
    if (use_mask)
    {
        for (uint64_t n = 0; n < n_pixels_spatial_side; ++n)
        {
            if (imgid_mask.im->array.F[n] == 1)
            {
                ++mask_npix;
            }
        }

        mask_idx    = (uint32_t *) malloc(mask_npix * sizeof(uint32_t));
        uint64_t nn = 0;
        for (uint64_t pp = 0; pp < n_pixels_spatial_side; ++pp)
        {
            if (imgid_mask.im->array.F[pp] == 1.0f) // TODO why mask not integer???
            {
                mask_idx[nn] = (uint32_t) pp;
                ++nn;
            }
        }

        printf("Mask has : %lu pixels (%f%%)\n", mask_npix,
               (100.0 * mask_npix) / n_pixels_spatial_side);
    }
    else
    {
        // Just use full image
        mask_npix = n_pixels_spatial_side;
        printf("No mask using : %lu pixels (%f%%)\n", mask_npix,
               (100.0 * mask_npix) / n_pixels_spatial_side);
    }


    // NORMALIZATION AND POST-INIT FOR MODES MATRIX
    float *normcoeff         = (float *) calloc(nb_modes, sizeof(float));
    float *masked_modes_copy = (float *) malloc(mask_npix * nb_modes * SIZEOF_DATATYPE_FLOAT);
    // TODO what WAS the meaning of normalization when axis_mode = 1 ????
    initializer_new_data_imgmodes(masked_modes_copy, normcoeff, mask_idx, mask_npix,
                                  imgid_modes.im);

    // CONNNECT TO OR CREATE OUTPUT STREAM
    IMGID imgid_out = stream_connect_create_2Df32(
        outcoeff, axis_mode == 0 ? nb_modes : modes_size[0], axis_mode == 0 ? 1 : modes_size[1]);
    memset(imgid_out.im->array.F, 0, imgid_out.md->imdatamemsize);

    float *imgin_float_casted_ptr = NULL;
    if (imgid_in.md->datatype == _DATATYPE_FLOAT)
    {
        imgin_float_casted_ptr = imgid_in.im->array.F;
        printf("INPUT is FLOAT  -> no type conversion required\n");
    }
    else
    {
        imgin_float_casted_ptr = (float *) malloc(sizeof(float) * n_pixels_spatial_side);
        printf("INPUT NOT float -> type conversion to float enabled\n");
    }


    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT;
    /* Runtime backend selector */
    ComputeMode                 compute_mode = _compute_mode_determine(processinfo);
    std::unique_ptr<MVMBackend> backend      = nullptr;


    if (compute_mode == ComputeMode::BLAS)
    {
#if defined(HAVE_MKL) || defined(HAVE_OPENBLAS)
        backend = std::make_unique<MVMBackendBLAS>(imgin_float_casted_ptr, imgid_out.im->array.F,
                                                   n_pixels_spatial_side, nb_modes, axis_mode);
#endif
    }
    else if (compute_mode == ComputeMode::OMP_OR_PLAIN)
    {
        backend = std::make_unique<MVMBackendCPU>(imgin_float_casted_ptr, imgid_out.im->array.F,
                                                  n_pixels_spatial_side, nb_modes, axis_mode);
    }
    else if (compute_mode == ComputeMode::CUDA)
    {
#ifdef HAVE_CUDA
        backend = std::make_unique<MVMBackendCUBLAS>(imgin_float_casted_ptr, imgid_out.im->array.F,
                                                     n_pixels_spatial_side, nb_modes, axis_mode);
#endif
    }

    if (backend == nullptr)
    {
        // TODO print a very explicit error message and fail
        // enum error? exit(0)
    }

    if (use_mask)
    {
        backend->enable_masking(mask_idx, mask_npix);
    }

    backend->load_matrix(masked_modes_copy, n_pixels_spatial_side, nb_modes);

    printf("LOOP START\n");
    fflush(stdout);

    printf("axmode     = %d [%s]\n", axis_mode,
           axis_mode == 0 ? "modal extraction" : "modal expansion");
    if (axis_mode == 0)
    {
        printf("in_shape       = %u x %u\n", modes_size[0], modes_size[1]);
        printf("out_shape       = %u\n", modes_size[2]);
    }
    else
    {
        printf(" in_shape       = %u x 1\n", modes_size[2]);
        printf("out_shape       = %u x %u\n", modes_size[0], modes_size[1]);
    }

    processinfo_WriteMessage_fmt(processinfo, "Backend: %s",
                                 compute_mode == ComputeMode::CUDA   ? "cuBLAS"
                                 : compute_mode == ComputeMode::BLAS ? "BLAS"
                                                                     : "CPU");


    printf(">>> START MVM loop\n");

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    /*
        Summary of steps:
        - float cast [on CPU, common]
        - input masking (axmode == 0 + masking)
        - MVM
        - output de-masking  (axmode == 1 + masking)
        */
    {
        /* Type conversion to float (all backends) */
        if (imgid_in.md->datatype != _DATATYPE_FLOAT)
        {
            cast_to_float(imgin_float_casted_ptr, imgid_in.im, n_pixels_spatial_side);
        }

        imgid_out.md->write =
            1; // We don't really know at which point the backend workflow will begin writes, so we flag early.

        backend->matrixMul(); // Here we MVM
        // We're done
        processinfo_update_output_stream(processinfo, imgid_out.im, imgid_in.im);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    free(normcoeff);
    free(masked_modes_copy);

    if (imgid_in.md->datatype != _DATATYPE_FLOAT)
    {
        free(imgin_float_casted_ptr);
    }


    if (use_mask)
    {
        free(mask_idx);
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* =============================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
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
} // extern "C"
