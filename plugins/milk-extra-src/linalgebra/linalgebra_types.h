// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file linalgebra_types.h
 * @brief Linalgebra types module
 */

#ifndef _LINALGEBRA_TYPES_H
#define _LINALGEBRA_TYPES_H

#ifdef HAVE_CUDA
#    include <cublas_v2.h>
#    include <cuda_runtime.h>
#    include <cuda_runtime_api.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#    include "computeunit.h"
#    include "ImageStreamIO/ImageStreamIO.h"
#endif

// Flags for SVD computations
#define COMPSVD_SKIP_BIGMAT 0x1UL
#define COMPSVD_COMP_PSINV 0x2UL
#define COMPSVD_COMP_CHECKPSINV 0x4UL
#define COMPSVD_COMP_RECONSTRUCT 0x8UL

// data passed to each thread
typedef struct
{
    int  thread_no;
    long numl0;
    int  cindex; // computation index
    int *status; // where to white status

    // timers
    struct timespec *t0;
    struct timespec *t1;
    struct timespec *t2;
    struct timespec *t3;
    struct timespec *t4;
    struct timespec *t5;

} LINALGEBRA_THDATA;

#ifdef HAVE_CUDA
/** \brief This structure holds the GPU computation setup for matrix multiplication
 *
 * By declaring an array of these structures,
 * several parallel computations can be executed
 *
 */

typedef struct
{
    int      init;       /**< 1 if initialized               */
    int     *refWFSinit; /**< reference init                 */
    int      alloc;      /**< 1 if memory has been allocated */
    imageID  CM_ID;
    uint64_t CM_cnt;
    long     timerID;

    uint32_t M;
    uint32_t N;

    /// synchronization
    int sem; /**< if sem = 1, wait for semaphore to perform computation */
    int gpuinit;

    /// one semaphore per thread
    sem_t **semptr1; /**< command to start matrix multiplication (input) */
    sem_t **semptr2; /**< memory transfer to device completed (output)   */
    sem_t **semptr3; /**< computation done (output)                      */
    sem_t **semptr4; /**< command to start transfer to host (input)      */
    sem_t **semptr5; /**< output transfer to host completed (output)     */

    // computer memory (host)
    float  *cMat;
    float **cMat_part;
    float  *wfsVec;
    float **wfsVec_part;
    float  *wfsRef;
    float **wfsRef_part;
    float  *dmVec;
    float  *dmVecTMP;
    float **dmVec_part;
    float **dmRef_part;

    // GPU memory (device)
    float **d_cMat;
    float **d_wfsVec;
    float **d_dmVec;
    float **d_wfsRef;
    float **d_dmRef;

    // threads
    LINALGEBRA_THDATA *thdata;
    int               *iret;
    pthread_t         *threadarray;
    int                NBstreams;
    cudaStream_t      *stream;
    cublasHandle_t    *handle;

    // splitting limits
    uint32_t *Nsize;
    uint32_t *Noffset;

    int *GPUdevice;

    int orientation;

    imageID IDout;

} GPUMATMULTCONF;
#endif

#endif
