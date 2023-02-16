/**
 * @file    cudacomp.h
 * @brief   Function prototypes for CUDA/MAGMA wrapper
 *
 * Also uses MAGMA library
 *
 *
 * @bug Magma can hang on magma_dsyevd_gpu
 *
 */

#ifndef _CUDACOMP_H
#define _CUDACOMP_H

#ifdef HAVE_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_types.h>
#include <pthread.h>

#endif

#include "cudacomp/GPU_SVD_computeControlMatrix.h"
#include "cudacomp/GPU_loop_MultMat_execute.h"
#include "cudacomp/GPU_loop_MultMat_setup.h"
#include "cudacomp/GPUloadCmat.h"
#include "cudacomp/MatMatMult_testPseudoInverse.h"
#include "cudacomp/cudacomp_MVMextractModesLoop.h"
#include "cudacomp/cudacompinit.h"
#include "cudacomp/cudacomptest.h"
#include "cudacomp/magma_compute_SVDpseudoInverse.h"
#include "cudacomp/magma_compute_SVDpseudoInverse_SVD.h"
#include "cudacomp/printGPUMATMULTCONF.h"

#include "cudacomp/MVM_CPU.h"

void __attribute__((constructor)) libinit_cudacomp();

#endif
