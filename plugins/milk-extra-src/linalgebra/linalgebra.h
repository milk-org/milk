/**
 * @file    linalgebra.h
 * @brief   Function prototypes linalgebra
 *
 */

#ifndef _LINALGEBRA_H
#define _LINALGEBRA_H

#ifdef HAVE_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_types.h>
#include <pthread.h>

#endif

#include "linalgebra/GPU_SVD_computeControlMatrix.h"
#include "linalgebra/GPU_loop_MultMat_execute.h"
#include "linalgebra/GPU_loop_MultMat_setup.h"
#include "linalgebra/GPUloadCmat.h"
#include "linalgebra/magma_MatMatMult_testPseudoInverse.h"
#include "linalgebra/cublas_linalgebra_MVMextractModesLoop.h"
#include "linalgebra/linalgebrainit.h"
#include "linalgebra/cublas_linalgebratest.h"
#include "linalgebra/magma_compute_SVDpseudoInverse.h"
#include "linalgebra/magma_compute_SVDpseudoInverse_SVD.h"
#include "linalgebra/printGPUMATMULTCONF.h"

#include "linalgebra/MVM_CPU.h"

void __attribute__((constructor)) libinit_linalgebra();

#endif
