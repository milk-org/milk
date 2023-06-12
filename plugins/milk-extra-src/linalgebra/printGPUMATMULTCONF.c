/** @file printGPUMATMULTCONF.c
 */

#ifdef HAVE_CUDA

#include "CommandLineInterface/CLIcore.h"

#include "linalgebra_types.h"

extern GPUMATMULTCONF gpumatmultconf[20];

errno_t LINALGEBRA_printGPUMATMULTCONF(int index)
{
    printf("\n");
    printf("============= GPUMATMULTCONF %d ======================\n", index);
    printf(" init              = %20d\n", (int) gpumatmultconf[index].init);
    printf(" refWFSinit        = %p\n",
           (void *) gpumatmultconf[index].refWFSinit);

    if(gpumatmultconf[index].refWFSinit != NULL)
    {
        printf("     refWFSinit[0]     = %20d\n",
               (int) gpumatmultconf[index].refWFSinit[0]);
    }

    printf(" alloc             = %20d\n", (int) gpumatmultconf[index].alloc);
    printf(" CM_ID             = %20ld\n", gpumatmultconf[index].CM_ID);
    printf(" CM_cnt            = %20ld\n", gpumatmultconf[index].CM_cnt);
    printf(" timerID           = %20ld\n", gpumatmultconf[index].timerID);
    printf(" M                 = %20d\n", (int) gpumatmultconf[index].M);
    printf(" N                 = %20d\n", (int) gpumatmultconf[index].N);

    /// synchronization
    printf(" sem               = %20d\n", (int) gpumatmultconf[index].sem);
    printf(" gpuinit           = %20d\n", (int) gpumatmultconf[index].gpuinit);

    /// one semaphore per thread
    /*
        sem_t **semptr1;
        sem_t **semptr2;
        sem_t **semptr3;
        sem_t **semptr4;
        sem_t **semptr5;
    */

    printf(" cMat              = %20p\n", (void *) gpumatmultconf[index].cMat);
    printf(" cMat_part         = %20p\n",
           (void *) gpumatmultconf[index].cMat_part);
    printf(" wfsVec            = %20p\n",
           (void *) gpumatmultconf[index].wfsVec);
    printf(" wfsVec_part       = %20p\n",
           (void *) gpumatmultconf[index].wfsVec_part);
    printf(" wfsRef            = %20p\n",
           (void *) gpumatmultconf[index].wfsRef);
    printf(" wfsRef_part       = %20p\n",
           (void *) gpumatmultconf[index].wfsRef_part);
    printf(" dmVec             = %20p\n", (void *) gpumatmultconf[index].dmVec);
    printf(" dmVecTMP          = %20p\n",
           (void *) gpumatmultconf[index].dmVecTMP);
    printf(" dmVec_part        = %20p\n",
           (void *) gpumatmultconf[index].dmVec_part);
    printf(" dmRef_part        = %20p\n",
           (void *) gpumatmultconf[index].dmRef_part);

    printf(" d_cMat            = %20p\n",
           (void *) gpumatmultconf[index].d_cMat);
    printf(" d_wfsVec          = %20p\n",
           (void *) gpumatmultconf[index].d_wfsVec);
    printf(" d_dmVec           = %20p\n",
           (void *) gpumatmultconf[index].d_dmVec);
    printf(" d_wfsRef          = %20p\n",
           (void *) gpumatmultconf[index].d_wfsRef);
    printf(" d_dmRef           = %20p\n",
           (void *) gpumatmultconf[index].d_dmRef);

    // threads
    printf(" thdata            = %20p\n",
           (void *) gpumatmultconf[index].thdata);
    printf(" threadarray       = %20p\n",
           (void *) gpumatmultconf[index].threadarray);
    printf(" NBstreams         = %20d\n",
           (int) gpumatmultconf[index].NBstreams);
    printf(" stream            = %20p\n",
           (void *) gpumatmultconf[index].stream);
    printf(" handle            = %20p\n",
           (void *) gpumatmultconf[index].handle);

    printf(" Nsize             = %20p\n", (void *) gpumatmultconf[index].Nsize);
    printf(" Noffset           = %20p\n",
           (void *) gpumatmultconf[index].Noffset);
    printf(" GPUdevice         = %20p\n",
           (void *) gpumatmultconf[index].GPUdevice);

    printf(" orientation       = %20d\n",
           (int) gpumatmultconf[index].orientation);

    printf("======================================================\n");
    printf("\n");

    return RETURN_SUCCESS;
}

#endif
