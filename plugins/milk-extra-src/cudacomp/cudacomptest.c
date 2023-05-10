/** @file cudacomptest.c
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

#include "GPU_SVD_computeControlMatrix.h"
#include "GPU_loop_MultMat_execute.h"
#include "GPU_loop_MultMat_setup.h"

#ifdef HAVE_CUDA

// ==========================================
// Forward declaration(s)
// ==========================================

errno_t GPUcomp_test(__attribute__((unused)) long NBact,
                     long                         NBmodes,
                     long                         WFSsize,
                     long                         GPUcnt);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t CUDACOMP_test_cli()
{
    if(CLI_checkarg(1, 2) + CLI_checkarg(2, 2) + CLI_checkarg(3, 2) +
            CLI_checkarg(4, 2) ==
            0)
    {
        GPUcomp_test(data.cmdargtoken[1].val.numl,
                     data.cmdargtoken[2].val.numl,
                     data.cmdargtoken[3].val.numl,
                     data.cmdargtoken[4].val.numl);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_ERROR;
    }
}

// ==========================================
// Register CLI command(s)
// ==========================================

errno_t cudacomptest_addCLIcmd()
{
    RegisterCLIcommand("cudacomptest",
                       __FILE__,
                       CUDACOMP_test_cli,
                       "test CUDA comp",
                       "<NB actuators [long]> <NB modes [long]> <NB pixels "
                       "[long]> <NB GPU [long]>",
                       "cudacomptest 1000 20 1000 1",
                       "int GPUcomp_test(long NBact, long NBmodes, long "
                       "WFSsize, long GPUcnt)");

    return RETURN_SUCCESS;
}

errno_t GPUcomp_test(__attribute__((unused)) long NBact,
                     long                         NBmodes,
                     long                         WFSsize,
                     long                         GPUcnt)
{
    imageID         ID_contrM;
    imageID         ID_WFS;
    imageID         ID_cmd_modes;
    uint32_t       *cmsize;
    uint32_t       *wfssize;
    uint32_t       *cmdmodessize;
    int             status;
    int             GPUstatus[100];
    long            iter;
    long            NBiter = 50000;
    double          time1sec, time2sec;
    struct timespec tnow;
    int            *GPUdevices;
    double          SVDeps = 0.1;

    //printf("Testing SVD on CPU\n");
    // linopt_compute_reconstructionMatrix("Rmat", "Cmat", SVDeps, "VTmat");

    create_2Dimage_ID("Rmat", WFSsize, WFSsize, NULL);

    printf("Testing SVD on GPU\n");
    GPU_SVD_computeControlMatrix(0, "Rmat", "Cmat", SVDeps, "VTmat");
    list_image_ID();
    printf("DONE ... ");
    fflush(stdout);

    // CHECK RESULT
    /*   arraysizetmp = (long*) malloc(sizeof(long)*3);
       ID_R = image_ID("Rmat");
       ID_C = image_ID("Cmat");

       if(data.image[ID_R].md[0].naxis==3)
       {
           m = data.image[ID_R].md[0].size[0]*data.image[ID_R].md[0].size[1];
           n = data.image[ID_R].md[0].size[2];
           printf("3D image -> %ld %ld\n", m, n);
           fflush(stdout);
       }
       else
       {
           m = data.image[ID_R].md[0].size[0];
           n = data.image[ID_R].md[0].size[1];
           printf("2D image -> %ld %ld\n", m, n);
           fflush(stdout);
       }


       printf("CHECKING RESULT ... ");
       fflush(stdout);

       ID = create_2Dimage_ID("SVDcheck", n, n);
       for(ii=0;ii<n;ii++)
           for(jj=0;jj<n;jj++)
               {
                   val = 0.0;
                   for(k=0;k<m;k++)
                       val += data.image[ID_C].array.F[ii*m+k] * data.image[ID_R].array.F[jj*m+k];
                   data.image[ID].array.F[jj*n+ii] = val;
               }
       save_fits("SVDcheck", "SVDcheck.fits");

    free(arraysizetmp);
       printf("DONE\n");
       fflush(stdout);*/

    printf("Testing GPU matrix multiplication speed, %ld GPUs\n", GPUcnt);

    GPUdevices = (int *) malloc(sizeof(int) * GPUcnt);
    for(int k = 0; k < GPUcnt; k++)
    {
        GPUdevices[k] = k + 8;
    }

    cmsize    = (uint32_t *) malloc(sizeof(uint32_t) * 3);
    cmsize[0] = WFSsize;
    cmsize[1] = WFSsize;
    cmsize[2] = NBmodes;
    create_image_ID("cudatestcm",
                    3,
                    cmsize,
                    _DATATYPE_FLOAT,
                    1,
                    0,
                    0,
                    &ID_contrM);

    wfssize    = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    wfssize[0] = WFSsize;
    wfssize[1] = WFSsize;
    create_image_ID("cudatestwfs",
                    2,
                    wfssize,
                    _DATATYPE_FLOAT,
                    1,
                    0,
                    0,
                    &ID_WFS);

    cmdmodessize    = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    cmdmodessize[0] = NBmodes;
    cmdmodessize[1] = 1;
    create_image_ID("cudatestcmd",
                    2,
                    cmdmodessize,
                    _DATATYPE_FLOAT,
                    1,
                    0,
                    0,
                    &ID_cmd_modes);

    GPU_loop_MultMat_setup(0,
                           data.image[ID_contrM].name,
                           data.image[ID_WFS].name,
                           data.image[ID_cmd_modes].name,
                           GPUcnt,
                           GPUdevices,
                           0,
                           1,
                           1,
                           0);

    clock_gettime(CLOCK_MILK, &tnow);
    time1sec = 1.0 * ((long) tnow.tv_sec) + 1.0e-9 * tnow.tv_nsec;

    for(iter = 0; iter < NBiter; iter++)
    {
        status = 0;
        GPU_loop_MultMat_execute(0, &status, &GPUstatus[0], 1.0, 0.0, 1, 0);
    }
    clock_gettime(CLOCK_MILK, &tnow);
    time2sec = 1.0 * ((long) tnow.tv_sec) + 1.0e-9 * tnow.tv_nsec;

    printf("Frequ = %12.3f Hz\n", 1.0 * NBiter / (time2sec - time1sec));

    printf("done\n");
    fflush(stdout);

    delete_image_ID("cudatestcm", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("cudatestwfs", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("cudatestcmd", DELETE_IMAGE_ERRMODE_WARNING);

    free(cmsize);
    free(wfssize);
    free(cmdmodessize);
    free(GPUdevices);

    return RETURN_SUCCESS;
}

#endif
