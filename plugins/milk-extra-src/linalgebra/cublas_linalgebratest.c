/**
 * @file cublas_linalgebratest.c
 * @brief Cublas linalgebratest module
 */

/** @file linalgebratest.c
 */

// MILK_CMAKE_MANDATE_CUDA

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
#    include "fps.h"
#    include "ImageStreamIO/ImageStreamIO.h"
#endif
#include "COREMOD_memory/COREMOD_memory.h"

#include "GPU_SVD_computeControlMatrix.h"
#include "GPU_loop_MultMat_execute.h"
#include "GPU_loop_MultMat_setup.h"

// ==========================================
// Forward declaration(s)
// ==========================================

errno_t GPUcomp_test(__attribute__((unused)) long NBact, long NBmodes, long WFSsize, long GPUcnt);

// ==========================================
// Gen 4 V2 CLI command: linalgebratest
// ==========================================

static int64_t      lt_nact         = 1000;
static int64_t      lt_nmod         = 20;
static int64_t      lt_wsz          = 1000;
static int64_t      lt_gpu          = 1;
static FPS_APP_INFO FPS_app_info_lt = {
    .fps_name         = "linalgebratest",
    .cmdkey           = "linalgebratest",
    .description      = "test CUDA comp",
    .description_long = "Test and benchmark cuBLAS linear algebra operations. Verifies GPU matrix "
                        "multiply correctness and measures throughput."
};
#define FPS_PARAMS_LT(X)                                                       \
    X(".nbact", &lt_nact, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "NB act")     \
    X(".nbmodes", &lt_nmod, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "NB modes") \
    X(".wfssize", &lt_wsz, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "NB pixels") \
    X(".gpucnt", &lt_gpu, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "NB GPU")
#include "fps.h"
static FPS_CLI_BINDING                   lt_b[]     = { FPS_PARAMS_LT(FPS_X_BINDING) };
static const int                         lt_nb      = sizeof(lt_b) / sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF                      farg[]     = { FPS_PARAMS_LT(FPS_X_FARG) };
static CLICMDDATA                        CLIcmddata = { "", "", CLICMD_FIELDS_DEFAULTS };
static CMDSETTINGS                       lt_cms     = { 0 };
static __attribute__((constructor)) void init_lt(void)
{
    strncpy(CLIcmddata.key, FPS_app_info_lt.cmdkey, sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description, FPS_app_info_lt.description,
            sizeof(CLIcmddata.description) - 1);
    CLIcmddata.nbarg         = sizeof(farg) / sizeof(CLICMDARGDEF);
    CLIcmddata.funcfpscliarg = farg;
    CLIcmddata.flags         = CLICMDFLAG_FPS;
    if (!CLIcmddata.cmdsettings)
    {
        CLIcmddata.cmdsettings = &lt_cms;
    }
}
static errno_t lt_compute(void)
{
    GPUcomp_test((long) lt_nact, (long) lt_nmod, (long) lt_wsz, (long) lt_gpu);
    return RETURN_SUCCESS;
}
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info_lt, farg, &CLIcmddata, lt_b, lt_nb,
                                        lt_compute);
}

errno_t linalgebratest_addCLIcmd()
{
    safe_fps_fill_farg_examples(farg, lt_b, lt_nb);
    INSERT_STD_CLIREGISTERFUNC;

    return RETURN_SUCCESS;
}

errno_t GPUcomp_test(__attribute__((unused)) long NBact, long NBmodes, long WFSsize, long GPUcnt)
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
       ID_R = image_ID("Rmat", dcimg, dcnimg);
       ID_C = image_ID("Cmat", dcimg, dcnimg);

       if(dcimg[ID_R].md[0].naxis==3)
       {
           m = dcimg[ID_R].md[0].size[0]*dcimg[ID_R].md[0].size[1];
           n = dcimg[ID_R].md[0].size[2];
           printf("3D image -> %ld %ld\n", m, n);
           fflush(stdout);
       }
       else
       {
           m = dcimg[ID_R].md[0].size[0];
           n = dcimg[ID_R].md[0].size[1];
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
                       val += dcimg[ID_C].array.F[ii*m+k] * dcimg[ID_R].array.F[jj*m+k];
                   dcimg[ID].array.F[jj*n+ii] = val;
               }
       save_fits("SVDcheck", "SVDcheck.fits");

    free(arraysizetmp);
       printf("DONE\n");
       fflush(stdout);*/

    printf("Testing GPU matrix multiplication speed, %ld GPUs\n", GPUcnt);

    GPUdevices = (int *) malloc(sizeof(int) * GPUcnt);
    for (int k = 0; k < GPUcnt; k++)
    {
        GPUdevices[k] = k + 8;
    }

    cmsize    = (uint32_t *) malloc(sizeof(uint32_t) * 3);
    cmsize[0] = WFSsize;
    cmsize[1] = WFSsize;
    cmsize[2] = NBmodes;
    {
        IMGID img_cm         = imgid_make_from_name("cudatestcm");
        img_cm.mdt->naxis    = 3;
        img_cm.mdt->size[0]  = cmsize[0];
        img_cm.mdt->size[1]  = cmsize[1];
        img_cm.mdt->size[2]  = cmsize[2];
        img_cm.mdt->datatype = _DATATYPE_FLOAT;
        img_cm.mdt->shared   = 1;
        img_cm.im            = (IMAGE *) calloc(1, sizeof(IMAGE));
        imgid_mkimage(&img_cm);
        ID_contrM = img_cm.ID;
    }

    wfssize    = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    wfssize[0] = WFSsize;
    wfssize[1] = WFSsize;
    {
        IMGID img_wfs         = imgid_make_from_name("cudatestwfs");
        img_wfs.mdt->naxis    = 2;
        img_wfs.mdt->size[0]  = wfssize[0];
        img_wfs.mdt->size[1]  = wfssize[1];
        img_wfs.mdt->datatype = _DATATYPE_FLOAT;
        img_wfs.mdt->shared   = 1;
        img_wfs.im            = (IMAGE *) calloc(1, sizeof(IMAGE));
        imgid_mkimage(&img_wfs);
        ID_WFS = img_wfs.ID;
    }

    cmdmodessize    = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    cmdmodessize[0] = NBmodes;
    cmdmodessize[1] = 1;
    {
        IMGID img_cmd         = imgid_make_from_name("cudatestcmd");
        img_cmd.mdt->naxis    = 2;
        img_cmd.mdt->size[0]  = cmdmodessize[0];
        img_cmd.mdt->size[1]  = cmdmodessize[1];
        img_cmd.mdt->datatype = _DATATYPE_FLOAT;
        img_cmd.mdt->shared   = 1;
        img_cmd.im            = (IMAGE *) calloc(1, sizeof(IMAGE));
        imgid_mkimage(&img_cmd);
        ID_cmd_modes = img_cmd.ID;
    }

    GPU_loop_MultMat_setup(0, dcimg[ID_contrM].name, dcimg[ID_WFS].name, dcimg[ID_cmd_modes].name,
                           GPUcnt, GPUdevices, 0, 1, 1, 0);

    clock_gettime(CLOCK_MILK, &tnow);
    time1sec = 1.0 * ((long) tnow.tv_sec) + 1.0e-9 * tnow.tv_nsec;

    for (iter = 0; iter < NBiter; iter++)
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
