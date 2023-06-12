/** @file GPU_loop_MultMat_execute.c
 */

#ifdef HAVE_CUDA

#include <pthread.h>
#include <time.h>

#include <cublas_v2.h>

#include "CommandLineInterface/CLIcore.h"
#include "CommandLineInterface/timeutils.h"

#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"
#include "linalgebra_types.h"

#include "GPUloadCmat.h"

extern imageID        IDtiming;
extern float          cublasSgemv_alpha;
extern float          cublasSgemv_beta;
extern GPUMATMULTCONF gpumatmultconf[20];

static int FORCESEMINIT = 1;

/*
 *
 * sequence of events :
 *
 * wait semptr1              (wait for input image data)
 * transfer input CPU -> GPU
 * post semptr2
 * COMPUTE
 * post semptr3
 * wait semptr4
 *
 *
 *
 *
 */

void __attribute__((hot)) *GPUcomputeMVM_function(void *ptr)
{
    LINALGEBRA_THDATA *thdata;
    int              device;
    int              index;
    const char      *ptr0; // source
    //const char *ptr1; // dest
    //float      *ptr0f; // test
    int *ptrstat;
    //imageID     IDtest;
    //char        fname[200];
    long long iter;
    long long itermax = 1;
    //float       imtot;
    //float       alphatmp;
    //float       betatmp;
    int  semval;
    long cnt;
    //FILE        *fptest;

    float alpharef, betaref;

    struct timespec t00;

    int ComputeGPU_FLAG = 1; //TEST

    thdata = (LINALGEBRA_THDATA *) ptr;
    device = thdata->thread_no;
    index  = thdata->cindex;

    ptrstat = (int *)((char *) thdata->status +
                      sizeof(int) * device); // + sizeof(int)*10*index);  //TBR

    *ptrstat = 1;

    // LOG function start
    int  logfunc_level     = 0;
    int  logfunc_level_max = 1;
    char commentstring[200];
    sprintf(commentstring, "MVM compute on GPU");
    CORE_logFunctionCall(logfunc_level,
                         logfunc_level_max,
                         0,
                         __FILE__,
                         __func__,
                         __LINE__,
                         commentstring);

    ptr0 = (char *) gpumatmultconf[index].wfsVec;
    ptr0 += sizeof(float) * gpumatmultconf[index].Noffset[device];
    //ptr0f = (float*) ptr0;

    cudaSetDevice(gpumatmultconf[index].GPUdevice[device]);

    cublasSetStream(gpumatmultconf[index].handle[device],
                    gpumatmultconf[index].stream[device]);

    if(gpumatmultconf[index].sem == 1)
    {
        itermax = -1;
    }
    else
    {
        itermax = 1;
    }

    iter = 0;
    while(iter != itermax)
    {
        //printf("====================================== gpumatmultconf[index].M = %d\n", gpumatmultconf[index].M);
        //fflush(stdout);

        clock_gettime(CLOCK_MILK, &t00);

        // copy DM reference to output to prepare computation:   d_dmVec <- d_dmRef
        if(ComputeGPU_FLAG == 1)
        {
            cudaError_t error =
                cudaMemcpy(gpumatmultconf[index].d_dmVec[device],
                           gpumatmultconf[index].d_dmRef[device],
                           sizeof(float) * gpumatmultconf[index].M,
                           cudaMemcpyDeviceToDevice);

            if(error != cudaSuccess)
            {
                printf(
                    "cudaMemcpy d_wfsVec wfsVec returned error code %d, "
                    "line(%d)\n",
                    error,
                    __LINE__);
                fflush(stdout);
                exit(EXIT_FAILURE);
            }
        }

        *ptrstat = 2; // wait for image

        //
        // Wait for semaphore #1 to be posted to transfer from CPU to GPU
        //
        //printf("%s %d      index = %d  sem = %d\n", __FILE__, __LINE__, index, gpumatmultconf[index].sem);//TEST
        if(gpumatmultconf[index].sem == 1)
        {
            sem_wait(gpumatmultconf[index].semptr1[device]);

            if(FORCESEMINIT == 1)
            {
                sem_getvalue(gpumatmultconf[index].semptr1[device], &semval);
                for(cnt = 0; cnt < semval; cnt++)
                {
                    printf(
                        "WARNING %s %d  : sem_trywait on semptr1 index %d "
                        "device %d\n",
                        __FILE__,
                        __LINE__,
                        index,
                        device);
                    fflush(stdout);
                    sem_trywait(gpumatmultconf[index].semptr1[device]);
                }
            }
        }

        thdata->t0->tv_sec  = t00.tv_sec;
        thdata->t0->tv_nsec = t00.tv_nsec;
        clock_gettime(CLOCK_MILK, thdata->t1);

        *ptrstat = 3; // transfer: prt0 -> d_wfsVec
        if(ComputeGPU_FLAG == 1)
        {
            cublasStatus_t stat =
                cublasSetVector(gpumatmultconf[index].Nsize[device],
                                sizeof(float),
                                (float *) ptr0,
                                1,
                                gpumatmultconf[index].d_wfsVec[device],
                                1);
            if(stat != CUBLAS_STATUS_SUCCESS)
            {
                fprintf(stderr, "!! device access error (read C)\n");
                if(stat == CUBLAS_STATUS_NOT_INITIALIZED)
                {
                    printf("   CUBLAS_STATUS_NOT_INITIALIZED\n");
                }
                if(stat == CUBLAS_STATUS_INVALID_VALUE)
                {
                    printf("   CUBLAS_STATUS_INVALID_VALUE\n");
                }
                if(stat == CUBLAS_STATUS_MAPPING_ERROR)
                {
                    printf("   CUBLAS_STATUS_MAPPING_ERROR\n");
                }
                exit(EXIT_FAILURE);
            }
        }

        clock_gettime(CLOCK_MILK, thdata->t2);

        if(gpumatmultconf[index].refWFSinit[device] ==
                0) // compute DM reference (used when reference changes)
        {
            printf("DM reference changed -> recompute\n");
            fflush(stdout);

            *ptrstat = 4; // compute

            // enable this post if outside process needs to be notified of computation start
            /*            if(gpumatmultconf[index].sem==1)
                            sem_post(gpumatmultconf[index].semptr2[device]);*/

            //  printf("%d  device %d (GPU %d): compute reference product\n", index, device, gpumatmultconf[index].GPUdevice[device]);
            //  fflush(stdout);

            //            alphatmp = cublasSgemv_alpha;
            //            betatmp = cublasSgemv_beta;

            // MOVE THIS TO CPU AS A SEPARATE THREAD TO AVOID LOOP PAUSE ??
            //        cublasSgemv_alpha = 1.0;
            //        cublasSgemv_beta = 0.0;
            alpharef = 1.0;
            betaref  = 0.0;

            cublasStatus_t stat =
                cublasSgemv(gpumatmultconf[index].handle[device],
                            CUBLAS_OP_N,
                            gpumatmultconf[index].M,
                            gpumatmultconf[index].Nsize[device],
                            &alpharef,
                            gpumatmultconf[index].d_cMat[device],
                            gpumatmultconf[index].M,
                            gpumatmultconf[index].d_wfsVec[device],
                            1,
                            &betaref,
                            gpumatmultconf[index].d_dmRef[device],
                            1);

            if(stat != CUBLAS_STATUS_SUCCESS)
            {
                printf("cublasSgemv returned error code %d, line(%d)\n",
                       stat,
                       __LINE__);
                fflush(stdout);
                if(stat == CUBLAS_STATUS_NOT_INITIALIZED)
                {
                    printf("   CUBLAS_STATUS_NOT_INITIALIZED\n");
                }
                if(stat == CUBLAS_STATUS_INVALID_VALUE)
                {
                    printf("   CUBLAS_STATUS_INVALID_VALUE\n");
                }
                if(stat == CUBLAS_STATUS_ARCH_MISMATCH)
                {
                    printf("   CUBLAS_STATUS_ARCH_MISMATCH\n");
                }
                if(stat == CUBLAS_STATUS_EXECUTION_FAILED)
                {
                    printf("   CUBLAS_STATUS_EXECUTION_FAILED\n");
                }

                printf("device %d of index %d\n", device, index);
                printf("GPU device                          = %d\n",
                       gpumatmultconf[index].GPUdevice[device]);

                printf("CUBLAS_OP_N                         = %d\n",
                       CUBLAS_OP_N);
                printf("alpha                               = %f\n", alpharef);
                printf("beta                                = %f\n", betaref);
                printf("gpumatmultconf[index].M             = %d\n",
                       (int) gpumatmultconf[index].M);
                printf("gpumatmultconf[index].Nsize[device] = %d\n",
                       (int) gpumatmultconf[index].Nsize[device]);
                fflush(stdout);
                exit(EXIT_FAILURE);
            }

            //          cublasSgemv_alpha = alphatmp;
            //          cublasSgemv_beta = betatmp;

            gpumatmultconf[index].refWFSinit[device] = 1;

            // enable this post if outside process needs to be notified of computation start
            /*
            if(gpumatmultconf[index].sem==1)
                sem_post(gpumatmultconf[index].semptr3[device]);
            */

            *ptrstat = 5; // transfer result

            if(gpumatmultconf[index].sem == 1)
            {
                sem_wait(gpumatmultconf[index].semptr4[device]);
                if(FORCESEMINIT == 1)
                {
                    sem_getvalue(gpumatmultconf[index].semptr4[device],
                                 &semval);
                    for(cnt = 0; cnt < semval; cnt++)
                    {
                        printf(
                            "WARNING %s %d  : sem_trywait on semptr4 index %d "
                            "device %d\n",
                            __FILE__,
                            __LINE__,
                            index,
                            device);
                        fflush(stdout);
                        sem_trywait(gpumatmultconf[index].semptr4[device]);
                    }
                }
            }

            // copy d_dmRef -> dmRef_part
            stat = cublasGetVector(gpumatmultconf[index].M,
                                   sizeof(float),
                                   gpumatmultconf[index].d_dmRef[device],
                                   1,
                                   gpumatmultconf[index].dmRef_part[device],
                                   1);

            if(stat != CUBLAS_STATUS_SUCCESS)
            {
                fprintf(stderr, "!! device access error (read C)\n");
                if(stat == CUBLAS_STATUS_NOT_INITIALIZED)
                {
                    printf("   CUBLAS_STATUS_NOT_INITIALIZED\n");
                }
                if(stat == CUBLAS_STATUS_INVALID_VALUE)
                {
                    printf("   CUBLAS_STATUS_INVALID_VALUE\n");
                }
                if(stat == CUBLAS_STATUS_MAPPING_ERROR)
                {
                    printf("   CUBLAS_STATUS_MAPPING_ERROR\n");
                }
                exit(EXIT_FAILURE);
            }

            // TEST

            /*    sprintf(fname, "gputest%d.txt", device);
                if((fptest = fopen(fname, "w"))==NULL)
                {
                    printf("ERROR: cannot create file \"%s\"\n", fname);
                    exit(0);
                }
                printf("Writing test file \"%s\"\n", fname);
                fflush(stdout);
                for(ii=0; ii<gpumatmultconf[index].M; ii++)
                    fprintf(fptest, "%ld %f\n", ii, gpumatmultconf[index].dmRef_part[device][ii]);
                fclose(fptest);
            */
            if(gpumatmultconf[index].sem == 1)
            {
                sem_post(gpumatmultconf[index].semptr5[device]);
            }

            *ptrstat = 6;
        }
        else
        {
            *ptrstat = 4; // compute

            //
            // Post semaphore #2 when starting computation
            // Enable if listening to semptr2
            /*
            if(gpumatmultconf[index].sem==1)
                sem_post(gpumatmultconf[index].semptr2[device]);
                */

            if(ComputeGPU_FLAG == 1)
            {
                cublasStatus_t stat =
                    cublasSgemv(gpumatmultconf[index].handle[device],
                                CUBLAS_OP_N,
                                gpumatmultconf[index].M,
                                gpumatmultconf[index].Nsize[device],
                                &cublasSgemv_alpha,
                                gpumatmultconf[index].d_cMat[device],
                                gpumatmultconf[index].M,
                                gpumatmultconf[index].d_wfsVec[device],
                                1,
                                &cublasSgemv_beta,
                                gpumatmultconf[index].d_dmVec[device],
                                1);

                if(stat != CUBLAS_STATUS_SUCCESS)
                {
                    printf(
                        "cublasSgemv returned error code %d, line(%d), "
                        "index=%d\n",
                        stat,
                        __LINE__,
                        index);
                    fflush(stdout);
                    if(stat == CUBLAS_STATUS_NOT_INITIALIZED)
                    {
                        printf("   CUBLAS_STATUS_NOT_INITIALIZED\n");
                    }
                    if(stat == CUBLAS_STATUS_INVALID_VALUE)
                    {
                        printf("   CUBLAS_STATUS_INVALID_VALUE\n");
                    }
                    if(stat == CUBLAS_STATUS_ARCH_MISMATCH)
                    {
                        printf("   CUBLAS_STATUS_ARCH_MISMATCH\n");
                    }
                    if(stat == CUBLAS_STATUS_EXECUTION_FAILED)
                    {
                        printf("   CUBLAS_STATUS_EXECUTION_FAILED\n");
                    }

                    printf("device %d of index %d\n", device, index);
                    printf("GPU device                          = %d\n",
                           gpumatmultconf[index].GPUdevice[device]);
                    printf("CUBLAS_OP_N                         = %d\n",
                           CUBLAS_OP_N);
                    printf("alpha                               = %f\n",
                           cublasSgemv_alpha);
                    printf("alpha                               = %f\n",
                           cublasSgemv_beta);
                    printf("gpumatmultconf[index].M             = %d\n",
                           (int) gpumatmultconf[index].M);
                    printf("gpumatmultconf[index].Nsize[device] = %d\n",
                           (int) gpumatmultconf[index].Nsize[device]);
                    fflush(stdout);
                    exit(EXIT_FAILURE);
                }
            }
            clock_gettime(CLOCK_MILK, thdata->t3);

            //
            // When computation is done on GPU, post semaphore #3
            //
            /*
            if(gpumatmultconf[index].sem==1)
                sem_post(gpumatmultconf[index].semptr3[device]);
            */
            *ptrstat = 5; // transfer result

            //
            // Wait for semaphore #4 to be posted to transfer from GPU to CPU
            //
            if(gpumatmultconf[index].sem == 1)
            {
                sem_wait(gpumatmultconf[index].semptr4[device]);
                if(FORCESEMINIT == 1)
                {
                    sem_getvalue(gpumatmultconf[index].semptr4[device],
                                 &semval);
                    for(cnt = 0; cnt < semval; cnt++)
                    {
                        printf(
                            "WARNING %s %d  : sem_trywait on semptr4 index %d "
                            "device %d\n",
                            __FILE__,
                            __LINE__,
                            index,
                            device);
                        fflush(stdout);
                        sem_trywait(gpumatmultconf[index].semptr4[device]);
                    }
                }
            }

            clock_gettime(CLOCK_MILK, thdata->t4);

            //cudaMemcpy ( gpumatmultconf[index].dmVec_part[device], gpumatmultconf[index].d_dmVec[device], sizeof(float)*gpumatmultconf[index].M, cudaMemcpyDeviceToHost);
            // result is on gpumatmultconf[index].d_dmVec[device]

            if(ComputeGPU_FLAG == 1)
            {
                cublasStatus_t stat =
                    cublasGetVector(gpumatmultconf[index].M,
                                    sizeof(float),
                                    gpumatmultconf[index].d_dmVec[device],
                                    1,
                                    gpumatmultconf[index].dmVec_part[device],
                                    1);

                if(stat != CUBLAS_STATUS_SUCCESS)
                {
                    fprintf(stderr, "!! device access error (read C)\n");
                    if(stat == CUBLAS_STATUS_NOT_INITIALIZED)
                    {
                        printf("   CUBLAS_STATUS_NOT_INITIALIZED\n");
                    }
                    if(stat == CUBLAS_STATUS_INVALID_VALUE)
                    {
                        printf("   CUBLAS_STATUS_INVALID_VALUE\n");
                    }
                    if(stat == CUBLAS_STATUS_MAPPING_ERROR)
                    {
                        printf("   CUBLAS_STATUS_MAPPING_ERROR\n");
                    }
                    exit(EXIT_FAILURE);
                }
            }
        }

        clock_gettime(CLOCK_MILK, thdata->t5);
        //
        // When data is ready on CPU, post semaphore #5
        //
        if(gpumatmultconf[index].sem == 1)
        {
            sem_post(gpumatmultconf[index].semptr5[device]);
        }

        *ptrstat = 6;

        // START MODE VALUES COMPUTATION HERE

        iter++;
    }

    // LOG function / process end
    CORE_logFunctionCall(logfunc_level,
                         logfunc_level_max,
                         1,
                         __FILE__,
                         __func__,
                         __LINE__,
                         commentstring);

    pthread_exit(0);
}

//
// increments status by 4
//
int GPU_loop_MultMat_execute(int   index,
                             int  *status,
                             int  *GPUstatus,
                             float alpha,
                             float beta,
                             int   timing,
                             int   TimerOffsetIndex)
{
    int ptn;
    //int statustot;
    int  semval;
    long cnt;
    int  TimerIndex;

    struct timespec tdt0[10];
    struct timespec tdt1[10];
    struct timespec tdt2[10];
    struct timespec tdt3[10];
    struct timespec tdt4[10];
    struct timespec tdt5[10];

#ifdef _PRINT_TEST
    printf("[%s] [%d]  Start (index %d)\n", __FILE__, __LINE__, index);
    fflush(stdout);
#endif

    TimerIndex = TimerOffsetIndex;

    cublasSgemv_alpha = alpha;
    cublasSgemv_beta  = beta;

    // flush semaphores
    for(ptn = 0; ptn < gpumatmultconf[index].NBstreams; ptn++)
    {
        sem_getvalue(gpumatmultconf[index].semptr1[ptn], &semval);
        for(cnt = 0; cnt < semval; cnt++)
        {
            printf(
                "WARNING %s %d  : [%ld] sem_trywait on semptr1 index %d ptn "
                "%d\n",
                __FILE__,
                __LINE__,
                semval - cnt,
                index,
                ptn);
            fflush(stdout);
            sem_trywait(gpumatmultconf[index].semptr1[ptn]);
        }

        sem_getvalue(gpumatmultconf[index].semptr2[ptn], &semval);
        for(cnt = 0; cnt < semval; cnt++)
        {
            printf(
                "WARNING %s %d  : [%ld] sem_trywait on semptr2 index %d ptn "
                "%d\n",
                __FILE__,
                __LINE__,
                semval - cnt,
                index,
                ptn);
            fflush(stdout);
            sem_trywait(gpumatmultconf[index].semptr2[ptn]);
        }

        sem_getvalue(gpumatmultconf[index].semptr3[ptn], &semval);
        for(cnt = 0; cnt < semval; cnt++)
        {
            printf(
                "WARNING %s %d  : [%ld] sem_trywait on semptr3 index %d ptn "
                "%d\n",
                __FILE__,
                __LINE__,
                semval - cnt,
                index,
                ptn);
            fflush(stdout);
            sem_trywait(gpumatmultconf[index].semptr3[ptn]);
        }

        sem_getvalue(gpumatmultconf[index].semptr4[ptn], &semval);
        for(cnt = 0; cnt < semval; cnt++)
        {
            printf(
                "WARNING %s %d  : [%ld] sem_trywait on semptr4 index %d ptn "
                "%d\n",
                __FILE__,
                __LINE__,
                semval - cnt,
                index,
                ptn);
            fflush(stdout);
            sem_trywait(gpumatmultconf[index].semptr4[ptn]);
        }

        sem_getvalue(gpumatmultconf[index].semptr5[ptn], &semval);
        for(cnt = 0; cnt < semval; cnt++)
        {
            printf(
                "WARNING %s %d  : [%ld] sem_trywait on semptr5 index %d ptn "
                "%d\n",
                __FILE__,
                __LINE__,
                semval - cnt,
                index,
                ptn);
            fflush(stdout);
            sem_trywait(gpumatmultconf[index].semptr5[ptn]);
        }
    }

#ifdef _PRINT_TEST
    printf("[%s] [%d]  semaphores flushed\n", __FILE__, __LINE__);
    fflush(stdout);
#endif

    if(timing == 1)
    {
        struct timespec tnow;

        *status = *status + 1; // ->7
        clock_gettime(CLOCK_MILK, &tnow);
        double tdiffv =
            timespec_diff_double(data.image[IDtiming].md[0].atime, tnow);
        data.image[IDtiming].array.F[TimerIndex] = tdiffv; //25
        TimerIndex++;
    }

    //    if((index==0)||(index==2)) /// main CM multiplication loop
    //    {

    if(gpumatmultconf[index].CM_cnt !=
            data.image[gpumatmultconf[index].CM_ID].md[0].cnt0)
        if(data.image[gpumatmultconf[index].CM_ID].md[0].write == 0)
        {
            printf("New CM detected (cnt : %ld)\n",
                   data.image[gpumatmultconf[index].CM_ID].md[0].cnt0);
            GPUloadCmat(index);
            gpumatmultconf[index].CM_cnt =
                data.image[gpumatmultconf[index].CM_ID].md[0].cnt0;
        }
    //   }

    // index is the matrix multiplication index (unique to each matrix multiplication stream operation)
    // ptn is the thread number = GPU device number

    //    if((gpumatmultconf[index].sem==0)||

    if(gpumatmultconf[index].gpuinit == 0)
    {
        printf("GPU pthread create, index = %d    %d %d\n",
               index,
               gpumatmultconf[index].sem,
               gpumatmultconf[index].gpuinit); //TEST
        fflush(stdout);

        for(ptn = 0; ptn < gpumatmultconf[index].NBstreams; ptn++)
        {
            gpumatmultconf[index].thdata[ptn].thread_no = ptn;
            gpumatmultconf[index].thdata[ptn].numl0     = ptn * ptn;
            gpumatmultconf[index].thdata[ptn].cindex    = index;
            gpumatmultconf[index].thdata[ptn].status    = GPUstatus;
            gpumatmultconf[index].thdata[ptn].t0        = &tdt0[ptn];
            gpumatmultconf[index].thdata[ptn].t1        = &tdt1[ptn];
            gpumatmultconf[index].thdata[ptn].t2        = &tdt2[ptn];
            gpumatmultconf[index].thdata[ptn].t3        = &tdt3[ptn];
            gpumatmultconf[index].thdata[ptn].t4        = &tdt4[ptn];
            gpumatmultconf[index].thdata[ptn].t5        = &tdt5[ptn];
            gpumatmultconf[index].iret[ptn] =
                pthread_create(&gpumatmultconf[index].threadarray[ptn],
                               NULL,
                               GPUcomputeMVM_function,
                               (void *) &gpumatmultconf[index].thdata[ptn]);
            if(gpumatmultconf[index].iret[ptn])
            {
                fprintf(stderr,
                        "Error - pthread_create() return code: %d\n",
                        gpumatmultconf[index].iret[ptn]);
                exit(EXIT_FAILURE);
            }
        }
        gpumatmultconf[index].gpuinit = 1;
    }

    if(timing == 1)
    {
        struct timespec tnow;

        *status = *status + 1; // -> 8
        clock_gettime(CLOCK_MILK, &tnow);
        double tdiffv =
            timespec_diff_double(data.image[IDtiming].md[0].atime, tnow);
        data.image[IDtiming].array.F[TimerIndex] = tdiffv; //26
        TimerIndex++;
    }

#ifdef _PRINT_TEST
    printf("[%s] [%d] - START COMPUTATION   gpumatmultconf[%d].sem = %d\n",
           __FILE__,
           __LINE__,
           index,
           gpumatmultconf[index].sem);
    fflush(stdout);
#endif

    if(gpumatmultconf[index].sem == 0)
    {
#ifdef _PRINT_TEST
        printf("[%s] [%d] - pthread join     %d streams\n",
               __FILE__,
               __LINE__,
               gpumatmultconf[index].NBstreams);
        fflush(stdout);
#endif

        for(ptn = 0; ptn < gpumatmultconf[index].NBstreams; ptn++)
        {
            pthread_join(gpumatmultconf[index].threadarray[ptn], NULL);
        }
    }
    else
    {
        for(ptn = 0; ptn < gpumatmultconf[index].NBstreams; ptn++)
        {
            sem_post(gpumatmultconf[index].semptr1[ptn]); // START COMPUTATION
            sem_post(gpumatmultconf[index].semptr4[ptn]);
        }

#ifdef _PRINT_TEST
        printf("[%s] [%d] - posted input semaphores  ( %d streams )\n",
               __FILE__,
               __LINE__,
               gpumatmultconf[index].NBstreams);
        fflush(stdout);
#endif

        for(ptn = 0; ptn < gpumatmultconf[index].NBstreams; ptn++)
        {
            sem_wait(gpumatmultconf[index].semptr5[ptn]); // WAIT FOR RESULT
        }

#ifdef _PRINT_TEST
        printf("[%s] [%d] - output semaphores wait complete\n",
               __FILE__,
               __LINE__);
        fflush(stdout);
#endif

        // for safety, set semaphores to zerosem_getvalue(data.image[IDarray[i]].semptr[s], &semval);
        if(FORCESEMINIT == 1)
            for(ptn = 0; ptn < gpumatmultconf[index].NBstreams; ptn++)
            {
                sem_getvalue(gpumatmultconf[index].semptr5[ptn], &semval);
                for(cnt = 0; cnt < semval; cnt++)
                {
                    printf(
                        "WARNING %s %d  : sem_trywait on semptr5 index %d ptn "
                        "%d\n",
                        __FILE__,
                        __LINE__,
                        index,
                        ptn);
                    fflush(stdout);
                    sem_trywait(gpumatmultconf[index].semptr5[ptn]);
                }
            }
    }

#ifdef _PRINT_TEST
    printf("[%s] [%d] - \n", __FILE__, __LINE__);
    fflush(stdout);
#endif

    if(timing == 1)
    {
        double tdiffv = timespec_diff_double(tdt0[0], tdt1[0]);
        data.image[IDtiming].array.F[TimerIndex] = tdiffv; //27
        TimerIndex++;

        tdiffv = timespec_diff_double(tdt1[0], tdt2[0]);
        data.image[IDtiming].array.F[TimerIndex] = tdiffv; //28
        TimerIndex++;

        tdiffv = timespec_diff_double(tdt2[0], tdt3[0]);
        data.image[IDtiming].array.F[TimerIndex] = tdiffv; //29
        TimerIndex++;

        tdiffv = timespec_diff_double(tdt3[0], tdt4[0]);
        data.image[IDtiming].array.F[TimerIndex] = tdiffv; //30
        TimerIndex++;

        tdiffv = timespec_diff_double(tdt4[0], tdt5[0]);
        data.image[IDtiming].array.F[TimerIndex] = tdiffv; //31
        TimerIndex++;
    }

    // SUM RESULTS FROM SEPARATE GPUs
#ifdef _PRINT_TEST
    printf("[%s] [%d] - SUM RESULTS FROM SEPARATE GPUs\n", __FILE__, __LINE__);
    fflush(stdout);
#endif

    if(timing == 1)
    {
        struct timespec tnow;

        *status = *status + 1; // -> 9
        clock_gettime(CLOCK_MILK, &tnow);
        double tdiffv =
            timespec_diff_double(data.image[IDtiming].md[0].atime, tnow);
        data.image[IDtiming].array.F[TimerIndex] = tdiffv; //32
        TimerIndex++;
    }

    data.image[gpumatmultconf[index].IDout].md[0].write = 1;

    for(uint32_t m = 0; m < gpumatmultconf[index].M; m++)
    {
        gpumatmultconf[index].dmVecTMP[m] = 0.0;
    }

    for(ptn = 0; ptn < gpumatmultconf[index].NBstreams; ptn++)
    {
        for(uint32_t m = 0; m < gpumatmultconf[index].M; m++)
        {
            gpumatmultconf[index].dmVecTMP[m] +=
                gpumatmultconf[index].dmVec_part[ptn][m];
        }
    }

    if(timing == 1)
    {
        struct timespec tnow;

        data.image[gpumatmultconf[index].IDout].md[0].cnt1 =
            data.image[IDtiming].md[0].cnt1;

        *status = *status + 1; // -> 10
        clock_gettime(CLOCK_MILK, &tnow);
        double tdiffv =
            timespec_diff_double(data.image[IDtiming].md[0].atime, tnow);
        data.image[IDtiming].array.F[TimerIndex] = tdiffv; //33
        TimerIndex++;
    }

    data.image[gpumatmultconf[index].IDout].md[0].cnt0++;
    COREMOD_MEMORY_image_set_sempost_byID(gpumatmultconf[index].IDout, -1);
    data.image[gpumatmultconf[index].IDout].md[0].write = 0;

#ifdef _PRINT_TEST
    printf("[%s] [%d] - DONE\n", __FILE__, __LINE__);
    fflush(stdout);
#endif

    return 0;
}

#endif
