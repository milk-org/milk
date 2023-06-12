/** @file GPUloadCmat.c
 */

#ifdef HAVE_CUDA

#include <cublas_v2.h>

#include "CommandLineInterface/CLIcore.h"
#include "linalgebra_types.h"

extern GPUMATMULTCONF gpumatmultconf[20];

errno_t GPUloadCmat(int index)
{

    printf("LOADING MATRIX TO GPU ... ");
    fflush(stdout);

    for(int device = 0; device < gpumatmultconf[index].NBstreams; device++)
    {
        for(unsigned int n = gpumatmultconf[index].Noffset[device];
                n < gpumatmultconf[index].Noffset[device] +
                gpumatmultconf[index].Nsize[device];
                n++)
        {
            if(gpumatmultconf[index].orientation == 0)
            {
                for(unsigned int m = 0; m < gpumatmultconf[index].M; m++)
                {
                    gpumatmultconf[index]
                    .cMat_part[device]
                    [(n - gpumatmultconf[index].Noffset[device]) *
                                                                 gpumatmultconf[index].M +
                                                                 m] =
                         gpumatmultconf[index]
                         .cMat[m * gpumatmultconf[index].N + n];
                }
            }
            else
            {
                for(unsigned int m = 0; m < gpumatmultconf[index].M; m++)
                {
                    gpumatmultconf[index]
                    .cMat_part[device]
                    [(n - gpumatmultconf[index].Noffset[device]) *
                                                                 gpumatmultconf[index].M +
                                                                 m] =
                         gpumatmultconf[index]
                         .cMat[n * gpumatmultconf[index].M + m];
                }
            }
        }
    }

    for(int device = 0; device < gpumatmultconf[index].NBstreams; device++)
    {
        cudaSetDevice(gpumatmultconf[index].GPUdevice[device]);
        cublasStatus_t error =
            cublasSetMatrix(gpumatmultconf[index].M,
                            gpumatmultconf[index].Nsize[device],
                            sizeof(float),
                            gpumatmultconf[index].cMat_part[device],
                            gpumatmultconf[index].M,
                            gpumatmultconf[index].d_cMat[device],
                            gpumatmultconf[index].M);

        if(error != CUBLAS_STATUS_SUCCESS)
        {
            printf("cudblasSetMatrix returned error code %d, line(%d)\n",
                   (int) error,
                   __LINE__);
            exit(EXIT_FAILURE);
        }
    }
    printf("done\n");
    fflush(stdout);

    return RETURN_SUCCESS;
}

#endif
