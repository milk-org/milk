// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "compilertest.h"

// MILK_CMAKE_REQUEST_CUDA
#ifdef HAVE_CUDA
#    include <cublas_v2.h>
#endif

#include <assert.h>

static char some_string[FUNCTION_PARAMETER_STRMAXLEN] = "string";

#define REQCUDA_PARAMS(X)                                                             \
    X(".some_string", some_string, FPTYPE_STRING_NOT_STREAM, 1, FPFLAG_DEFAULT_INPUT, \
      "Unused string")

MILK_PARAMS_DEFINE(reqcuda, REQCUDA_PARAMS);

static errno_t reqcuda_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    MILK_PROCINFO_INIT(self)

#ifdef HAVE_CUDA
    // Allocate 2 x 1MB on GPU.
    void *cuda_ptr_in;
    void *cuda_ptr_out;
    cudaMalloc(&cuda_ptr_in, 1024 * 1024);
    cudaMalloc(&cuda_ptr_out, 1024 * 1024);
    cublasHandle_t cublas_handle;
    assert(CUBLAS_STATUS_SUCCESS == cublasCreate(&cublas_handle));
    float alpha = 1.0f;
#endif

    MILK_PROCINFO_LOOPSTART(self)
    {
#ifdef HAVE_CUDA
        printf("CublasSaxpy...\n");
        cublasSaxpy(cublas_handle, 1024 * 256, &alpha, cuda_ptr_in, 1, cuda_ptr_out, 1);
#else
        printf("Doing nothing...\n");
#endif
    }
    MILK_PROCINFO_END(self)

#ifdef HAVE_CUDA
    cudaFree(cuda_ptr_in);
    cudaFree(cuda_ptr_out);
#endif

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_ct_requestcuda = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "reqcuda",
                 .cmdkey           = "reqcuda",
                 .description      = "Dumb test for MILK_CMAKE_REQUEST_CUDA test",
                 .description_long = "Dumb test for MILK_CMAKE_REQUEST_CUDA test" },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(reqcuda),
    .compute = reqcuda_compute,
};
EXPORT_FPSEXEC(milk, "compilertest-requestcuda", runnable_ct_requestcuda);
