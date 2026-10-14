// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

// MILK_CMAKE_MANDATE_CUDA
#include <cublas_v2.h>
#include <assert.h>

static char some_string[FUNCTION_PARAMETER_STRMAXLEN] = "string";

#define MANDCUDA_PARAMS(X)                                                            \
    X(".some_string", some_string, FPTYPE_STRING_NOT_STREAM, 1, FPFLAG_DEFAULT_INPUT, \
      "Unused string")

MILK_PARAMS_DEFINE(mandcuda, MANDCUDA_PARAMS);

static errno_t mandcuda_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    MILK_PROCINFO_INIT(self)

    // Allocate 2 x 1MB on GPU.
    void *cuda_ptr_in;
    void *cuda_ptr_out;
    cudaMalloc(&cuda_ptr_in, 1024 * 1024);
    cudaMalloc(&cuda_ptr_out, 1024 * 1024);

    float alpha = 1.0f;

    cublasHandle_t cublas_handle;
    assert(CUBLAS_STATUS_SUCCESS == cublasCreate(&cublas_handle));

    MILK_PROCINFO_LOOPSTART(self)
    {
        printf("CublasSaxpy...\n");
        cublasSaxpy(cublas_handle, 1024 * 256, &alpha, cuda_ptr_in, 1, cuda_ptr_out, 1);
    }
    MILK_PROCINFO_END(self)

    cudaFree(cuda_ptr_in);
    cudaFree(cuda_ptr_out);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

// Not declared in compilertest.h: the module registry declares it weak (file may be dropped).
const MILK_RUNNABLE runnable_ct_mandatecuda = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "mandcuda",
                 .cmdkey           = "mandcuda",
                 .description      = "Dumb test for MILK_CMAKE_MANDATE_CUDA test",
                 .description_long = "Dumb test for MILK_CMAKE_MANDATE_CUDA test" },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(mandcuda),
    .compute = mandcuda_compute,
};
EXPORT_FPSEXEC(milk, "compilertest-mandatecuda", runnable_ct_mandatecuda);
