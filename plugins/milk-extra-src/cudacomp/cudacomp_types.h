#ifndef _CUDACOMP_TYPES_H
#define _CUDACOMP_TYPES_H

#ifdef HAVE_CUDA
#define HAVE_CUBLAS
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#endif

#include "CommandLineInterface/CLIcore.h"

#ifdef HAVE_MAGMA

/******************* CPU memory */
#define TESTING_MALLOC_CPU(ptr, type, size)                                    \
    if (MAGMA_SUCCESS !=                                                       \
        magma_malloc_cpu((void **) &ptr, (size) * sizeof(type)))               \
    {                                                                          \
        fprintf(stderr, "!!!! magma_malloc_cpu failed for: %s\n", #ptr);       \
        magma_finalize();                                                      \
        exit(-1);                                                              \
    }

#define TESTING_DMALLOC_CPU(ptr, size)                                         \
    if (MAGMA_SUCCESS != magma_dmalloc_cpu(&ptr, (size_t) (size)))             \
    {                                                                          \
        fprintf(stderr, "!!!! magma_malloc_cpu failed for: %s\n", #ptr);       \
        magma_finalize();                                                      \
        exit(-1);                                                              \
    }

#define TESTING_SMALLOC_CPU(ptr, size)                                         \
    if (MAGMA_SUCCESS != magma_smalloc_cpu(&ptr, (size_t) (size)))             \
    {                                                                          \
        fprintf(stderr, "!!!! magma_fmalloc_cpu failed for: %s\n", #ptr);      \
        magma_finalize();                                                      \
        exit(-1);                                                              \
    }

#define TESTING_FREE_CPU(ptr) magma_free_cpu(ptr)

/******************* Pinned CPU memory */
#ifdef HAVE_CUBLAS
// In CUDA, this allocates pinned memory.
#define TESTING_MALLOC_PIN(ptr, type, size)                                    \
    if (MAGMA_SUCCESS !=                                                       \
        magma_malloc_pinned((void **) &ptr, (size) * sizeof(type)))            \
    {                                                                          \
        fprintf(stderr, "!!!! magma_malloc_pinned failed for: %s\n", #ptr);    \
        magma_finalize();                                                      \
        exit(-1);                                                              \
    }

#define TESTING_FREE_PIN(ptr) magma_free_pinned(ptr)
#else
// For OpenCL, we don't support pinned memory yet.
#define TESTING_MALLOC_PIN(ptr, type, size)                                    \
    if (MAGMA_SUCCESS !=                                                       \
        magma_malloc_cpu((void **) &ptr, (size) * sizeof(type)))               \
    {                                                                          \
        fprintf(stderr, "!!!! magma_malloc_cpu failed for: %s\n", #ptr);       \
        magma_finalize();                                                      \
        exit(-1);                                                              \
    }

#define TESTING_FREE_PIN(ptr) magma_free_cpu(ptr)
#endif

/******************* GPU memory */
#ifdef HAVE_CUBLAS
// In CUDA, this has (void**) cast.
#define TESTING_MALLOC_DEV(ptr, type, size)                                    \
    if (MAGMA_SUCCESS !=                                                       \
        magma_malloc((void **) &ptr, (size_t) sizeof(type) * size))            \
    {                                                                          \
        fprintf(                                                               \
            stderr,                                                            \
            "!!!! magma_malloc failed for: %s  size = %ld  typesize = %d\n",   \
            #ptr,                                                              \
            (long) size,                                                       \
            (int) sizeof(type));                                               \
        magma_finalize();                                                      \
        exit(-1);                                                              \
    }

#define TESTING_DMALLOC_DEV(ptr, size)                                         \
    if (MAGMA_SUCCESS != magma_dmalloc(&ptr, (size_t) (size)))                 \
    {                                                                          \
        fprintf(                                                               \
            stderr,                                                            \
            "!!!! magma_dmalloc failed for: %s  size = %ld  typesize = %d\n",  \
            #ptr,                                                              \
            (long) size,                                                       \
            (int) sizeof(double));                                             \
        magma_finalize();                                                      \
        exit(-1);                                                              \
    }

#define TESTING_SMALLOC_DEV(ptr, size)                                         \
    if (MAGMA_SUCCESS != magma_smalloc(&ptr, (size_t) (size)))                 \
    {                                                                          \
        fprintf(                                                               \
            stderr,                                                            \
            "!!!! magma_fmalloc failed for: %s  size = %ld  typesize = %d\n",  \
            #ptr,                                                              \
            (long) size,                                                       \
            (int) sizeof(float));                                              \
        magma_finalize();                                                      \
        exit(-1);                                                              \
    }

#else
// For OpenCL, ptr is cl_mem* and there is no cast.
#define TESTING_MALLOC_DEV(ptr, type, size)                                    \
    if (MAGMA_SUCCESS != magma_malloc(&ptr, (size) * sizeof(type)))            \
    {                                                                          \
        fprintf(stderr, "!!!! magma_malloc failed for: %s\n", #ptr);           \
        magma_finalize();                                                      \
        exit(-1);                                                              \
    }
#endif

#define TESTING_FREE_DEV(ptr) magma_free(ptr)

#endif

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

} CUDACOMP_THDATA;

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
    CUDACOMP_THDATA *thdata;
    int             *iret;
    pthread_t       *threadarray;
    int              NBstreams;
    cudaStream_t    *stream;
    cublasHandle_t  *handle;

    // splitting limits
    uint32_t *Nsize;
    uint32_t *Noffset;

    int *GPUdevice;

    int orientation;

    imageID IDout;

} GPUMATMULTCONF;
#endif

#endif
