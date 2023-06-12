/** @file GPU_loop_MultMat_setup.c
 */

#ifdef HAVE_CUDA

#include <cublas_v2.h>
#include <fcntl.h>
#include <pthread.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

#include "GPU_loop_MultMat_free.h"
#include "GPUloadCmat.h"
#include "linalgebra_types.h"
#include "linalgebrainit.h"

extern GPUMATMULTCONF gpumatmultconf[20];
extern imageID        IDtimerinit;
extern imageID        IDtiming;
extern int            cuda_deviceCount;

/**
 * ## Purpose
 *
 * Setup matrix multiplication using multiple GPUs
 *
 * ## Parameters
 *
 * @param[in]	index
 * 		Configuration index
 *
 * @param[in]   IDcontrM_name
 * 		Control matrix image name
 *
 * @param[in]	IDwfsim_name
 * 		Wavefront sensor image stream
 *
 * @param[out]	IDoutdmmodes_name
 * 		Output DM modes
 *
 *  IDoutdmmodes_name  = alpha * IDcontrM_name x IDwfsim_name
 *
 * ## Details
 *
 * upon setup, IDwfsim_name is the WFS ref and initWFSref = 0
 *
*/

errno_t GPU_loop_MultMat_setup(int         index,
                               const char *IDcontrM_name,
                               const char *IDwfsim_name,
                               const char *IDoutdmmodes_name,
                               long        NBGPUs,
                               int        *GPUdevice,
                               int         orientation,
                               int         USEsem,
                               int         initWFSref,
                               long        loopnb)
{
    //LINALGEBRA_printGPUMATMULTCONF(index);

    DEBUG_TRACE_FSTART();

    /// This function will not do anything if the initialization has already been performed

    if(gpumatmultconf[index].init == 0)
    {
        int pid;
        //struct cudaDeviceProp deviceProp;
        char sname[200];

        imageID IDwfsim;
        imageID IDwfsref;

        printf("STARTING SETUP of GPU MVM #%d .....\n", index);
        fflush(stdout);

        pid = getpid();

        /*if (IDtimerinit == 0)
        {
            char name[200];

            sprintf(name, "aol%ld_looptiming", loopnb);
            IDtiming = image_ID(name);

            if (IDtiming == -1)
            {
                create_2Dimage_ID(name, 50, 1, &IDtiming);
            }
        }*/

        if(gpumatmultconf[index].alloc == 1)
        {
            GPU_loop_MultMat_free(index);
            gpumatmultconf[index].alloc = 0;
        }

        if(USEsem == 1)
        {
            gpumatmultconf[index].sem = 1;
        }
        else
        {
            gpumatmultconf[index].sem = 0;
        }

        printf("USEsem = %d\n", USEsem);
        fflush(stdout);

        gpumatmultconf[index].orientation = orientation;



        // Load Control Matrix
        //
        printf("Using Matrix %s\n", IDcontrM_name);
        imageID IDcontrM            = image_ID(IDcontrM_name);
        gpumatmultconf[index].CM_ID = IDcontrM;
        printf("    size : [");
        for(int dim = 0; dim < data.image[IDcontrM].md->naxis; dim++)
        {
            printf(" %d", data.image[IDcontrM].md[0].size[dim]);
        }
        printf(" ]\n");


        gpumatmultconf[index].CM_cnt =
            data.image[gpumatmultconf[index].CM_ID].md[0].cnt0;


        if(orientation == 0)
        {
            if(data.image[IDcontrM].md[0].naxis == 3)
            {
                gpumatmultconf[index].M = data.image[IDcontrM].md[0].size[2];
                gpumatmultconf[index].N = data.image[IDcontrM].md[0].size[0] *
                                          data.image[IDcontrM].md[0].size[1];
                //   cmatdim = 3;
            }
            else
            {
                gpumatmultconf[index].M = data.image[IDcontrM].md[0].size[1];
                gpumatmultconf[index].N = data.image[IDcontrM].md[0].size[0];
                // cmatdim = 2;
            }
            printf("[0] [%ld] M = %d\n",
                   IDcontrM,
                   (int) gpumatmultconf[index].M);
            printf("[0] [%ld] N = %d\n",
                   IDcontrM,
                   (int) gpumatmultconf[index].N);
        }
        else
        {
            if(data.image[IDcontrM].md[0].naxis == 3)
            {
                gpumatmultconf[index].M = data.image[IDcontrM].md[0].size[0] *
                                          data.image[IDcontrM].md[0].size[1];
                gpumatmultconf[index].N = data.image[IDcontrM].md[0].size[2];
                //   cmatdim = 3;
            }
            else
            {
                gpumatmultconf[index].M = data.image[IDcontrM].md[0].size[0];
                gpumatmultconf[index].N = data.image[IDcontrM].md[0].size[1];
                //   cmatdim = 2;
            }

            printf("[1] [%ld] M = %d\n",
                   IDcontrM,
                   (int) gpumatmultconf[index].M);
            printf("[1] [%ld] N = %d\n",
                   IDcontrM,
                   (int) gpumatmultconf[index].N);
        }

        gpumatmultconf[index].cMat = data.image[IDcontrM].array.F;

        /// Load Input vectors
        IDwfsim                      = image_ID(IDwfsim_name);
        gpumatmultconf[index].wfsVec = data.image[IDwfsim].array.F;
        IDwfsref                     = image_ID(IDwfsim_name);
        gpumatmultconf[index].wfsRef = data.image[IDwfsref].array.F;

        if(orientation == 0)
        {
            printf("[0] Input vector size: %ld %ld\n",
                   (long) data.image[IDwfsim].md[0].size[0],
                   (long) data.image[IDwfsim].md[0].size[1]);

            if((uint32_t)(data.image[IDwfsim].md[0].size[0] *
                          data.image[IDwfsim].md[0].size[1]) !=
                    gpumatmultconf[index].N)
            {
                PRINT_ERROR(
                    "CONTRmat and WFSvec size not compatible: %ld vs %d\n",
                    (long)(data.image[IDwfsim].md[0].size[0] *
                           data.image[IDwfsim].md[0].size[1]),
                    (int) gpumatmultconf[index].N);
                fflush(stdout);
                DEBUG_TRACE_FEXIT();
                return (EXIT_FAILURE);
                exit(0);
            }
        }
        else
        {
            printf("[1] Input vector size: %ld \n",
                   (long) data.image[IDwfsim].md[0].size[0]);
            if(data.image[IDwfsim].md[0].size[0] != gpumatmultconf[index].N)
            {
                printf(
                    "ERROR: CONTRmat and WFSvec size not compatible: %ld %d\n",
                    (long) data.image[IDwfsim].md[0].size[0],
                    (int) gpumatmultconf[index].N);
                fflush(stdout);
                sleep(2);
                exit(0);
            }
        }

        printf("Setting up gpumatmultconf\n");
        fflush(stdout);

        if((gpumatmultconf[index].IDout = image_ID(IDoutdmmodes_name)) == -1)
        {
            uint32_t *sizearraytmp;

            sizearraytmp    = (uint32_t *) malloc(sizeof(uint32_t) * 2);
            sizearraytmp[0] = gpumatmultconf[index].M;
            sizearraytmp[1] = 1;
            create_image_ID(IDoutdmmodes_name,
                            2,
                            sizearraytmp,
                            _DATATYPE_FLOAT,
                            1,
                            10,
                            0,
                            &(gpumatmultconf[index].IDout));
            free(sizearraytmp);
        }
        else
        {
            if((uint32_t)(data.image[gpumatmultconf[index].IDout]
                          .md[0]
                          .size[0] *
                          data.image[gpumatmultconf[index].IDout]
                          .md[0]
                          .size[1]) != gpumatmultconf[index].M)
            {
                printf(
                    "ERROR: CONTRmat and WFSvec size not compatible: %ld %d\n",
                    (long)(data.image[gpumatmultconf[index].IDout]
                           .md[0]
                           .size[0] *
                           data.image[gpumatmultconf[index].IDout]
                           .md[0]
                           .size[1]),
                    (int) gpumatmultconf[index].M);
                printf("gpumatmultconf[index].IDout = %ld\n",
                       gpumatmultconf[index].IDout);
                list_image_ID();
                fflush(stdout);
                sleep(2);
                exit(0);
            }
        }

        gpumatmultconf[index].dmVecTMP =
            data.image[gpumatmultconf[index].IDout].array.F;

        // This section will create a thread

        pthread_t GPUscan_thread;

        pthread_create(&GPUscan_thread,
                       NULL,
                       GPU_scanDevices,
                       (void *) &cuda_deviceCount);
        if(pthread_join(GPUscan_thread, NULL))
        {
            fprintf(stderr, "Error joining thread\n");
            exit(0);
        }

        gpumatmultconf[index].NBstreams = cuda_deviceCount;
        if(NBGPUs < cuda_deviceCount)
        {
            gpumatmultconf[index].NBstreams = NBGPUs;
        }

        gpumatmultconf[index].Nsize =
            (uint32_t *) malloc(sizeof(long) * gpumatmultconf[index].NBstreams);
        gpumatmultconf[index].Noffset =
            (uint32_t *) malloc(sizeof(long) * gpumatmultconf[index].NBstreams);
        gpumatmultconf[index].Noffset[0] = 0;
        for(int device = 1; device < gpumatmultconf[index].NBstreams; device++)
        {
            gpumatmultconf[index].Noffset[device] =
                gpumatmultconf[index].Noffset[device - 1] +
                (long)(gpumatmultconf[index].N /
                       gpumatmultconf[index].NBstreams);
            gpumatmultconf[index].Nsize[device - 1] =
                gpumatmultconf[index].Noffset[device] -
                gpumatmultconf[index].Noffset[device - 1];
        }
        gpumatmultconf[index].Nsize[gpumatmultconf[index].NBstreams - 1] =
            gpumatmultconf[index].N -
            gpumatmultconf[index].Noffset[gpumatmultconf[index].NBstreams - 1];

        printf(
            "Allocating physical GPU(s) to stream(s) (index %d, NBGPU(s) = "
            "%ld)\n",
            index,
            NBGPUs);
        printf("%d stream(s)\n", gpumatmultconf[index].NBstreams);
        fflush(stdout);

        gpumatmultconf[index].GPUdevice = (int *) malloc(sizeof(int) * NBGPUs);

        printf("- - - - - - - - -\n");
        fflush(stdout);

        for(int device = 0; device < gpumatmultconf[index].NBstreams; device++)
        {
            printf("stream %2d  ->  GPU device %2d\n",
                   device,
                   GPUdevice[device]);
            fflush(stdout);
            gpumatmultconf[index].GPUdevice[device] = GPUdevice[device];
        }

        printf("-----------------------------------------------------\n");
        fflush(stdout);
        for(int device = 0; device < gpumatmultconf[index].NBstreams; device++)
        {
            printf("DEVICE %2d  [%2d]:  %5d -> %5d  (%d)\n",
                   device,
                   gpumatmultconf[index].GPUdevice[device],
                   (int) gpumatmultconf[index].Noffset[device],
                   (int)(gpumatmultconf[index].Noffset[device] +
                         gpumatmultconf[index].Nsize[device]),
                   (int) gpumatmultconf[index].Nsize[device]);
            fflush(stdout);
        }
        printf("-----------------------------------------------------\n");
        fflush(stdout);

        // device (GPU)
        gpumatmultconf[index].d_cMat = (float **) malloc(
                                           sizeof(float *) * gpumatmultconf[index].NBstreams);
        if(gpumatmultconf[index].d_cMat == NULL)
        {
            printf("malloc allocation error - %s %d\n", __FILE__, __LINE__);
            exit(0);
        }

        gpumatmultconf[index].d_wfsVec = (float **) malloc(
                                             sizeof(float *) * gpumatmultconf[index].NBstreams);
        if(gpumatmultconf[index].d_wfsVec == NULL)
        {
            printf("malloc allocation error - %s %d\n", __FILE__, __LINE__);
            exit(0);
        }

        gpumatmultconf[index].d_dmVec = (float **) malloc(
                                            sizeof(float *) * gpumatmultconf[index].NBstreams);
        if(gpumatmultconf[index].d_dmVec == NULL)
        {
            printf("malloc allocation error - %s %d\n", __FILE__, __LINE__);
            exit(0);
        }

        gpumatmultconf[index].d_wfsRef = (float **) malloc(
                                             sizeof(float *) * gpumatmultconf[index].NBstreams); // WFS reference
        if(gpumatmultconf[index].d_wfsRef == NULL)
        {
            printf("malloc allocation error - %s %d\n", __FILE__, __LINE__);
            exit(0);
        }

        gpumatmultconf[index].d_dmRef = (float **) malloc(
                                            sizeof(float *) * gpumatmultconf[index].NBstreams); // DM reference
        if(gpumatmultconf[index].d_dmRef == NULL)
        {
            printf("malloc allocation error - %s %d\n", __FILE__, __LINE__);
            exit(0);
        }

        gpumatmultconf[index].stream = (cudaStream_t *) malloc(
                                           sizeof(cudaStream_t) * gpumatmultconf[index].NBstreams);
        if(gpumatmultconf[index].stream == NULL)
        {
            printf("malloc allocation error - %s %d\n", __FILE__, __LINE__);
            exit(0);
        }

        gpumatmultconf[index].handle = (cublasHandle_t *) malloc(
                                           sizeof(cublasHandle_t) * gpumatmultconf[index].NBstreams);
        if(gpumatmultconf[index].handle == NULL)
        {
            printf("malloc allocation error - %s %d\n", __FILE__, __LINE__);
            exit(0);
        }

        // host (computer)
        gpumatmultconf[index].cMat_part = (float **) malloc(
                                              sizeof(float *) * gpumatmultconf[index].NBstreams);
        if(gpumatmultconf[index].cMat_part == NULL)
        {
            printf("malloc allocation error - %s %d\n", __FILE__, __LINE__);
            exit(0);
        }

        gpumatmultconf[index].wfsVec_part = (float **) malloc(
                                                sizeof(float *) * gpumatmultconf[index].NBstreams);
        if(gpumatmultconf[index].wfsVec_part == NULL)
        {
            printf("malloc allocation error - %s %d\n", __FILE__, __LINE__);
            exit(0);
        }

        gpumatmultconf[index].dmVec_part = (float **) malloc(
                                               sizeof(float *) * gpumatmultconf[index].NBstreams);
        if(gpumatmultconf[index].dmVec_part == NULL)
        {
            printf("malloc allocation error - %s %d\n", __FILE__, __LINE__);
            exit(0);
        }

        gpumatmultconf[index].wfsRef_part = (float **) malloc(
                                                sizeof(float *) * gpumatmultconf[index].NBstreams); // WFS reference
        if(gpumatmultconf[index].wfsRef_part == NULL)
        {
            printf("malloc allocation error - %s %d\n", __FILE__, __LINE__);
            exit(0);
        }

        gpumatmultconf[index].dmRef_part = (float **) malloc(
                                               sizeof(float *) *
                                               gpumatmultconf[index]
                                               .NBstreams); // DM reference (for checking only)
        if(gpumatmultconf[index].dmRef_part == NULL)
        {
            printf("malloc allocation error - %s %d\n", __FILE__, __LINE__);
            exit(0);
        }

        gpumatmultconf[index].refWFSinit =
            (int *) malloc(sizeof(int) * gpumatmultconf[index].NBstreams);
        if(gpumatmultconf[index].refWFSinit == NULL)
        {
            printf("malloc allocation error - %s %d\n", __FILE__, __LINE__);
            exit(0);
        }

        gpumatmultconf[index].semptr1 = (sem_t **) malloc(
                                            sizeof(sem_t *) * gpumatmultconf[index].NBstreams);
        if(gpumatmultconf[index].semptr1 == NULL)
        {
            printf("malloc allocation error - %s %d\n", __FILE__, __LINE__);
            exit(0);
        }

        gpumatmultconf[index].semptr2 = (sem_t **) malloc(
                                            sizeof(sem_t *) * gpumatmultconf[index].NBstreams);
        if(gpumatmultconf[index].semptr2 == NULL)
        {
            printf("malloc allocation error - %s %d\n", __FILE__, __LINE__);
            exit(0);
        }

        gpumatmultconf[index].semptr3 = (sem_t **) malloc(
                                            sizeof(sem_t *) * gpumatmultconf[index].NBstreams);
        if(gpumatmultconf[index].semptr3 == NULL)
        {
            printf("malloc allocation error - %s %d\n", __FILE__, __LINE__);
            exit(0);
        }

        gpumatmultconf[index].semptr4 = (sem_t **) malloc(
                                            sizeof(sem_t *) * gpumatmultconf[index].NBstreams);
        if(gpumatmultconf[index].semptr4 == NULL)
        {
            printf("malloc allocation error - %s %d\n", __FILE__, __LINE__);
            exit(0);
        }

        gpumatmultconf[index].semptr5 = (sem_t **) malloc(
                                            sizeof(sem_t *) * gpumatmultconf[index].NBstreams);
        if(gpumatmultconf[index].semptr5 == NULL)
        {
            printf("malloc allocation error - %s %d\n", __FILE__, __LINE__);
            exit(0);
        }

        for(int device = 0; device < gpumatmultconf[index].NBstreams; device++)
        {
            gpumatmultconf[index].cMat_part[device] =
                (float *) malloc(sizeof(float) * gpumatmultconf[index].M *
                                 gpumatmultconf[index].Nsize[device]);
            if(gpumatmultconf[index].cMat_part[device] == NULL)
            {
                printf("malloc allocation error - %s %d\n", __FILE__, __LINE__);
                exit(0);
            }

            gpumatmultconf[index].wfsVec_part[device] = (float *) malloc(
                        sizeof(float) * gpumatmultconf[index].Nsize[device]);
            if(gpumatmultconf[index].wfsVec_part[device] == NULL)
            {
                printf("malloc allocation error - %s %d\n", __FILE__, __LINE__);
                exit(0);
            }

            gpumatmultconf[index].wfsRef_part[device] = (float *) malloc(
                        sizeof(float) * gpumatmultconf[index].Nsize[device]);
            if(gpumatmultconf[index].wfsRef_part[device] == NULL)
            {
                printf("malloc allocation error - %s %d\n", __FILE__, __LINE__);
                exit(0);
            }

            gpumatmultconf[index].dmVec_part[device] =
                (float *) malloc(sizeof(float) * gpumatmultconf[index].M);
            if(gpumatmultconf[index].dmVec_part[device] == NULL)
            {
                printf("malloc allocation error - %s %d\n", __FILE__, __LINE__);
                exit(0);
            }

            gpumatmultconf[index].dmRef_part[device] =
                (float *) malloc(sizeof(float) * gpumatmultconf[index].M);
            if(gpumatmultconf[index].dmRef_part[device] == NULL)
            {
                printf("malloc allocation error - %s %d\n", __FILE__, __LINE__);
                exit(0);
            }

            sprintf(sname,
                    "loop%02ld_i%02d_gpu%02d_sem1_%06d",
                    loopnb,
                    index,
                    GPUdevice[device],
                    pid);
            if((gpumatmultconf[index].semptr1[device] =
                        sem_open(sname, O_CREAT, 0644, 1)) == SEM_FAILED)
            {
                perror("semaphore initilization");
                exit(0);
            }
            sem_init(gpumatmultconf[index].semptr1[device], 1, 0);

            sprintf(sname,
                    "loop%02ld_i%02d_gpu%02d_sem2_%06d",
                    loopnb,
                    index,
                    GPUdevice[device],
                    pid);
            if((gpumatmultconf[index].semptr2[device] =
                        sem_open(sname, O_CREAT, 0644, 1)) == SEM_FAILED)
            {
                perror("semaphore initilization");
                exit(0);
            }
            sem_init(gpumatmultconf[index].semptr2[device], 1, 0);

            sprintf(sname,
                    "loop%02ld_i%02d_gpu%02d_sem3_%06d",
                    loopnb,
                    index,
                    GPUdevice[device],
                    pid);
            if((gpumatmultconf[index].semptr3[device] =
                        sem_open(sname, O_CREAT, 0644, 1)) == SEM_FAILED)
            {
                perror("semaphore initilization");
                exit(0);
            }
            sem_init(gpumatmultconf[index].semptr3[device], 1, 0);

            sprintf(sname,
                    "loop%02ld_i%02d_gpu%02d_sem4_%06d",
                    loopnb,
                    index,
                    GPUdevice[device],
                    pid);
            if((gpumatmultconf[index].semptr4[device] =
                        sem_open(sname, O_CREAT, 0644, 1)) == SEM_FAILED)
            {
                perror("semaphore initilization");
                exit(0);
            }
            sem_init(gpumatmultconf[index].semptr4[device], 1, 0);

            sprintf(sname,
                    "loop%02ld_i%02d_gpu%02d_sem5_%06d",
                    loopnb,
                    index,
                    GPUdevice[device],
                    pid);
            if((gpumatmultconf[index].semptr5[device] =
                        sem_open(sname, O_CREAT, 0644, 1)) == SEM_FAILED)
            {
                perror("semaphore initilization");
                exit(0);
            }
            sem_init(gpumatmultconf[index].semptr5[device], 1, 0);
        }

        // this create two threads per device
        for(int device = 0; device < gpumatmultconf[index].NBstreams; device++)
        {
            cudaSetDevice(GPUdevice[device]);
            cudaStreamCreate(&gpumatmultconf[index].stream[device]);
        }

        for(int device = 0; device < gpumatmultconf[index].NBstreams; device++)
        {
            cudaSetDevice(GPUdevice[device]);

            // ALLOCATE MEMORY ON DEVICE

            cudaError_t error =
                cudaMalloc((void **) &gpumatmultconf[index].d_cMat[device],
                           sizeof(float) * gpumatmultconf[index].M *
                           gpumatmultconf[index].Nsize[device]);
            if(error != cudaSuccess)
            {
                printf("cudaMalloc d_cMat returned error code %d, line(%d)\n",
                       error,
                       __LINE__);
                exit(EXIT_FAILURE);
            }
            else
            {
                printf("ALLOCATED gpumatmultconf[%d].d_cMat[%d] size %d x %d\n",
                       index,
                       device,
                       (int) gpumatmultconf[index].M,
                       (int) gpumatmultconf[index].Nsize[device]);
            }

            error =
                cudaMalloc((void **) &gpumatmultconf[index].d_wfsVec[device],
                           sizeof(float) * gpumatmultconf[index].Nsize[device]);
            if(error != cudaSuccess)
            {
                printf("cudaMalloc d_wfsVec returned error code %d, line(%d)\n",
                       error,
                       __LINE__);
                exit(EXIT_FAILURE);
            }
            else
            {
                printf("ALLOCATED gpumatmultconf[%d].d_wfsVec[%d] size %d\n",
                       index,
                       device,
                       (int) gpumatmultconf[index].Nsize[device]);
            }

            error =
                cudaMalloc((void **) &gpumatmultconf[index].d_wfsRef[device],
                           sizeof(float) * gpumatmultconf[index].Nsize[device]);
            if(error != cudaSuccess)
            {
                printf("cudaMalloc d_wfsRef returned error code %d, line(%d)\n",
                       error,
                       __LINE__);
                exit(EXIT_FAILURE);
            }
            else
            {
                printf("ALLOCATED gpumatmultconf[%d].d_wfsRef[%d] size %d\n",
                       index,
                       device,
                       (int) gpumatmultconf[index].Nsize[device]);
            }

            error = cudaMalloc((void **) &gpumatmultconf[index].d_dmVec[device],
                               sizeof(float) * gpumatmultconf[index].M);
            if(error != cudaSuccess)
            {
                printf("cudaMalloc d_dmVec returned error code %d, line(%d)\n",
                       error,
                       __LINE__);
                exit(EXIT_FAILURE);
            }
            else
            {
                printf("ALLOCATED gpumatmultconf[%d].d_dmVec[%d] size %d\n",
                       index,
                       device,
                       (int) gpumatmultconf[index].M);
            }

            error = cudaMalloc((void **) &gpumatmultconf[index].d_dmRef[device],
                               sizeof(float) * gpumatmultconf[index].M);
            if(error != cudaSuccess)
            {
                printf("cudaMalloc d_dmVec returned error code %d, line(%d)\n",
                       error,
                       __LINE__);
                exit(EXIT_FAILURE);
            }
            else
            {
                printf("ALLOCATED gpumatmultconf[%d].d_dmRef[%d] size %d\n",
                       index,
                       device,
                       (int) gpumatmultconf[index].M);
            }

            cublasStatus_t stat =
                cublasCreate(&gpumatmultconf[index].handle[device]);
            printf("INITIALIZED CUBLAS handle index=%d device=%d\n",
                   index,
                   device);
            fflush(stdout);
            if(stat != CUBLAS_STATUS_SUCCESS)
            {
                printf("CUBLAS initialization failed\n");
                return EXIT_FAILURE;
            }
        }

        for(int device = 0; device < gpumatmultconf[index].NBstreams; device++)
            for(unsigned long n = gpumatmultconf[index].Noffset[device];
                    n < gpumatmultconf[index].Noffset[device] +
                    gpumatmultconf[index].Nsize[device];
                    n++)
            {
                gpumatmultconf[index]
                .wfsVec_part[device]
                [n - gpumatmultconf[index].Noffset[device]] =
                    gpumatmultconf[index].wfsVec[n];
                gpumatmultconf[index]
                .wfsRef_part[device]
                [n - gpumatmultconf[index].Noffset[device]] =
                    gpumatmultconf[index].wfsRef[n];
            }

        // copy memory to devices
        for(int device = 0; device < gpumatmultconf[index].NBstreams; device++)
        {
            cudaError_t error =
                cudaMemcpy(gpumatmultconf[index].d_wfsVec[device],
                           gpumatmultconf[index].wfsVec_part[device],
                           sizeof(float) * gpumatmultconf[index].Nsize[device],
                           cudaMemcpyHostToDevice);
            if(error != cudaSuccess)
            {
                printf(
                    "cudaMemcpy d_wfsVec wfsVec returned error code %d, "
                    "line(%d)\n",
                    error,
                    __LINE__);
                exit(EXIT_FAILURE);
            }

            printf("COPY wfsRef_part to d_wfsRef\n");
            fflush(stdout);
            error =
                cudaMemcpy(gpumatmultconf[index].d_wfsRef[device],
                           gpumatmultconf[index].wfsRef_part[device],
                           sizeof(float) * gpumatmultconf[index].Nsize[device],
                           cudaMemcpyHostToDevice);
            if(error != cudaSuccess)
            {
                printf(
                    "cudaMemcpy d_wfsRef wfsRef returned error code %d, "
                    "line(%d)\n",
                    error,
                    __LINE__);
                exit(EXIT_FAILURE);
            }
        }

        GPUloadCmat(index);

        printf("SETUP %d DONE, READY TO START COMPUTATIONS  ", index);
        fflush(stdout);

        gpumatmultconf[index].iret =
            (int *) malloc(sizeof(int) * gpumatmultconf[index].NBstreams);
        if(gpumatmultconf[index].iret == NULL)
        {
            printf("malloc allocation error - %s %d\n", __FILE__, __LINE__);
            exit(0);
        }

        // thread data
        gpumatmultconf[index].thdata = (LINALGEBRA_THDATA *) malloc(
                                           sizeof(LINALGEBRA_THDATA) * gpumatmultconf[index].NBstreams);
        if(gpumatmultconf[index].thdata == NULL)
        {
            printf("malloc allocation error - %s %d\n", __FILE__, __LINE__);
            exit(0);
        }

        gpumatmultconf[index].threadarray = (pthread_t *) malloc(
                                                sizeof(pthread_t) * gpumatmultconf[index].NBstreams);
        if(gpumatmultconf[index].threadarray == NULL)
        {
            printf("malloc allocation error - %s %d\n", __FILE__, __LINE__);
            exit(0);
        }

        for(uint32_t m = 0; m < gpumatmultconf[index].M; m++)
        {
            gpumatmultconf[index].dmVecTMP[m] = 0.0;
        }

        // cnt = 0;
        // iter = 0;
        gpumatmultconf[index].init = 1;

        printf(". . . \n");
        fflush(stdout);
    }

    for(int device = 0; device < gpumatmultconf[index].NBstreams; device++)
    {
        gpumatmultconf[index].refWFSinit[device] = initWFSref;
    }

    // printf("CONFIGURATION DONE \n");
    // fflush(stdout);

    //	LINALGEBRA_printGPUMATMULTCONF(index);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

#endif
