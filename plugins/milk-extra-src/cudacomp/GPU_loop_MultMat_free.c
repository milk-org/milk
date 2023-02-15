/** @file GPU_loop_MultMat_free.c
 */

#ifdef HAVE_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "cudacomp_types.h"

extern GPUMATMULTCONF gpumatmultconf[20];

int GPU_loop_MultMat_free(int index)
{
    int device;

    cudaFree(gpumatmultconf[index].d_cMat);
    cudaFree(gpumatmultconf[index].d_dmVec);
    cudaFree(gpumatmultconf[index].d_wfsVec);
    cudaFree(gpumatmultconf[index].d_wfsRef);
    cudaFree(gpumatmultconf[index].d_dmRef);
    free(gpumatmultconf[index].stream);

    for(device = 0; device < gpumatmultconf[index].NBstreams; device++)
    {
        // free memory for stream
        cublasDestroy(gpumatmultconf[index].handle[device]);
        free(gpumatmultconf[index].cMat_part[device]);
        free(gpumatmultconf[index].wfsVec_part[device]);
        free(gpumatmultconf[index].dmVec_part[device]);
    }

    free(gpumatmultconf[index].cMat_part);
    free(gpumatmultconf[index].dmVec_part);
    free(gpumatmultconf[index].wfsVec_part);

    free(gpumatmultconf[index].Nsize);
    free(gpumatmultconf[index].Noffset);

    free(gpumatmultconf[index].iret);
    free(gpumatmultconf[index].threadarray);
    free(gpumatmultconf[index].thdata);

    free(gpumatmultconf[index].refWFSinit);

    free(gpumatmultconf[index].GPUdevice);

    return (0);
}

#endif
