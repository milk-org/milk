/**
 * @file    cudacomp_MVMextractModesLoop.c
 * @brief   CUDA functions wrapper
 *
 * Requires CUDA library
 *
 */

// include sem_timedwait
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

/* =============================================================================================== */
/* =============================================================================================== */
/*                                        HEADER FILES                                             */
/* =============================================================================================== */
/* =============================================================================================== */

#include <malloc.h>
#include <math.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <semaphore.h>

#include <time.h>

#include <sys/file.h>
#include <sys/mman.h>
#include <sys/types.h>

#ifdef HAVE_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <device_types.h>
#include <pthread.h>

#endif

#include "CommandLineInterface/CLIcore.h"
#include "CommandLineInterface/timeutils.h"

#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"

//#include "cudacomp/cudacomp.h"

#include "linopt_imtools/linopt_imtools.h" // for testing

#ifdef HAVE_CUDA

// ==========================================
// Forward declaration(s)
// ==========================================

errno_t CUDACOMP_MVMextractModesLoop_FPCONF();

errno_t __attribute__((hot)) CUDACOMP_MVMextractModesLoop_RUN();

int __attribute__((hot)) CUDACOMP_MVMextractModesLoop(
    const char *in_stream,     // input stream
    const char *intot_stream,  // [optional]   input normalization stream
    const char *IDmodes_name,  // Modes matrix
    const char *IDrefin_name,  // [optional] input reference  - to be subtracted
    const char *IDrefout_name, // [optional] output reference - to be added
    const char *IDmodes_val_name, // ouput stream
    int         GPUindex,         // GPU index
    int         PROCESS,          // 1 if postprocessing
    int         TRACEMODE,        // 1 if writing trace
    int         MODENORM,         // 1 if input modes should be normalized
    int         insem,            // input semaphore index
    int         axmode, // 0 for normal mode extraction, 1 for expansion
    long        twait,  // if >0, insert time wait [us] at each iteration
    int         semwarn // 1 if warning when input stream semaphore >1
);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t CUDACOMP_MVMextractModesLoop_cli()
{

    // try FPS implementation
    // set data.fpsname, providing default value as first arg, and set data.FPS_CMDCODE value
    // default FPS name will be used if CLI process has NOT been named
    // see code in function_parameter.c for detailed rules
    function_parameter_getFPSargs_from_CLIfunc("cudaMVM");

    if(data.FPS_CMDCODE != 0)  // use FPS implementation
    {
        // set pointers to CONF and RUN functions
        data.FPS_CONFfunc = CUDACOMP_MVMextractModesLoop_FPCONF;
        data.FPS_RUNfunc  = CUDACOMP_MVMextractModesLoop_RUN;
        function_parameter_execFPScmd();
        return RETURN_SUCCESS;
    }

    // non FPS implementation - all parameters specified at function launch
    if(CLI_checkarg(1, CLIARG_IMG) + CLI_checkarg(2, CLIARG_STR) +
            CLI_checkarg(3, CLIARG_IMG) + CLI_checkarg(4, CLIARG_STR) +
            CLI_checkarg(5, CLIARG_STR) + CLI_checkarg(6, CLIARG_STR) +
            CLI_checkarg(7, CLIARG_INT64) + CLI_checkarg(8, CLIARG_INT64) +
            CLI_checkarg(9, CLIARG_INT64) + CLI_checkarg(10, CLIARG_INT64) +
            CLI_checkarg(11, CLIARG_INT64) + CLI_checkarg(12, CLIARG_INT64) +
            CLI_checkarg(13, CLIARG_INT64) + CLI_checkarg(14, CLIARG_INT64) ==
            0)
    {
        CUDACOMP_MVMextractModesLoop(data.cmdargtoken[1].val.string,
                                     data.cmdargtoken[2].val.string,
                                     data.cmdargtoken[3].val.string,
                                     data.cmdargtoken[4].val.string,
                                     data.cmdargtoken[5].val.string,
                                     data.cmdargtoken[6].val.string,
                                     data.cmdargtoken[7].val.numl,
                                     data.cmdargtoken[8].val.numl,
                                     data.cmdargtoken[9].val.numl,
                                     data.cmdargtoken[10].val.numl,
                                     data.cmdargtoken[11].val.numl,
                                     data.cmdargtoken[12].val.numl,
                                     data.cmdargtoken[13].val.numl,
                                     data.cmdargtoken[14].val.numl);

        return RETURN_SUCCESS;
    }
    else
    {
        return RETURN_FAILURE;
    }
}

// ==========================================
// Register CLI command(s)
// ==========================================

errno_t cudacomp_MVMextractModesLoop_addCLIcmd()
{

    RegisterCLIcommand(
        "cudaextrmodes",
        __FILE__,
        CUDACOMP_MVMextractModesLoop_cli,
        "CUDA extract mode values loop. Note that intot and refout parameters "
        "can be NULL",
        "<inval stream> <intot stream> <modes> <refin val> <refout_val> "
        "<outmode vals> <GPU index [long]> <PROCESS "
        "flag> <TRACEMODE flag> <MODE norm flag> <input semaphore> <axis "
        "orientation> <twait [us]> <semwarn>",
        "cudaextrmodes inmap inmaptot modes imref imoutref modeval 3 1 1 1 3 0 "
        "0",
        "int CUDACOMP_MVMextractModesLoop(const char *in_stream, const char "
        "*intot_stream, const char *IDmodes_name, "
        "const char *IDrefin_name, const char *IDrefout_name, const char "
        "*IDmodes_val_name, int GPUindex, int PROCESS, "
        "int TRACEMODE, int MODENORM, int insem, int axmode, long twait, int "
        "semwarn)");

    return RETURN_SUCCESS;
}

//
// manages configuration parameters
// initializes configuration parameters structure
//

errno_t CUDACOMP_MVMextractModesLoop_FPCONF()
{

    FPS_SETUP_INIT(data.FPS_name, data.FPS_CMDCODE); // sets up fps

    FPS2PROCINFOMAP fps2procinfo;
    fps_add_processinfo_entries(&fps);

    // ===========================
    // ALLOCATE FPS ENTRIES
    // ===========================

    void *pNull = NULL;

    uint64_t FPFLAG;
    FPFLAG = FPFLAG_DEFAULT_INPUT;
    FPFLAG &= FPFLAG_WRITECONF;
    FPFLAG &= ~FPFLAG_WRITERUN;

    long GPUindex_default[4] = {0, 0, 9, 0};
    //long fp_GPUindex = 0;
    function_parameter_add_entry(&fps,
                                 ".GPUindex",
                                 "GPU index",
                                 FPTYPE_INT64,
                                 FPFLAG_DEFAULT_INPUT,
                                 &GPUindex_default,
                                 NULL);

    //long fp_streamname_in = 0;
    function_parameter_add_entry(&fps,
                                 ".sname_in",
                                 "input stream vector",
                                 FPTYPE_STREAMNAME,
                                 FPFLAG_DEFAULT_INPUT_STREAM,
                                 pNull,
                                 NULL);

    //long fp_streamname_modes = 0;
    function_parameter_add_entry(&fps,
                                 ".sname_modes",
                                 "input modes matrix",
                                 FPTYPE_STREAMNAME,
                                 FPFLAG_DEFAULT_INPUT_STREAM,
                                 pNull,
                                 NULL);

    FPFLAG = FPFLAG_DEFAULT_INPUT_STREAM;
    FPFLAG &= ~FPFLAG_STREAM_RUN_REQUIRED;
    //long fp_streamname_intot = 0;
    function_parameter_add_entry(&fps,
                                 ".option.sname_intot",
                                 "optional input normalization stream",
                                 FPTYPE_STREAMNAME,
                                 FPFLAG,
                                 pNull,
                                 NULL);

    //long fp_streamname_refin = 0;
    function_parameter_add_entry(
        &fps,
        ".option.sname_refin",
        "optional input reference to be subtracted stream",
        FPTYPE_STREAMNAME,
        FPFLAG,
        pNull,
        NULL);

    //long fp_streamname_refout = 0;
    function_parameter_add_entry(
        &fps,
        ".option.sname_refout",
        "optional output reference to be subtracted stream",
        FPTYPE_STREAMNAME,
        FPFLAG,
        pNull,
        NULL);

    //long fp_stream_outmodesval = 0;
    function_parameter_add_entry(&fps,
                                 ".sname_outmodesval",
                                 "output mode coefficients stream",
                                 FPTYPE_STREAMNAME,
                                 FPFLAG,
                                 pNull,
                                 NULL);

    //long fp_outinit = 0;
    function_parameter_add_entry(&fps,
                                 ".outinit",
                                 "output stream init mode",
                                 FPTYPE_ONOFF,
                                 FPFLAG,
                                 pNull,
                                 NULL);

    //long fp_PROCESS = 0;
    function_parameter_add_entry(&fps,
                                 ".option.PROCESS",
                                 "1 if processing",
                                 FPTYPE_ONOFF,
                                 FPFLAG_DEFAULT_INPUT,
                                 pNull,
                                 NULL);

    //long fp_TRACEMODE = 0;
    function_parameter_add_entry(&fps,
                                 ".option.TRACEMODE",
                                 "1 if writing trace",
                                 FPTYPE_ONOFF,
                                 FPFLAG_DEFAULT_INPUT,
                                 pNull,
                                 NULL);

    //long fp_MODENORM = 0;
    function_parameter_add_entry(&fps,
                                 ".option.MODENORM",
                                 "1 if input modes should be normalized",
                                 FPTYPE_ONOFF,
                                 FPFLAG_DEFAULT_INPUT,
                                 pNull,
                                 NULL);

    //long fp_insem = 0;
    function_parameter_add_entry(&fps,
                                 ".option.insem",
                                 "input semaphore index",
                                 FPTYPE_INT64,
                                 FPFLAG_DEFAULT_INPUT,
                                 pNull,
                                 NULL);

    //long fp_axmode = 0;
    function_parameter_add_entry(
        &fps,
        ".option.axmode",
        "0 for normal mode extraction, 1 for expansion",
        FPTYPE_INT64,
        FPFLAG_DEFAULT_INPUT,
        pNull,
        NULL);

    //long fp_twait = 0;
    function_parameter_add_entry(
        &fps,
        ".option.twait",
        "if >0, insert time wait [us] at each iteration",
        FPTYPE_INT64,
        FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,
        pNull,
        NULL);

    //long fp_semwarn = 0;
    function_parameter_add_entry(&fps,
                                 ".option.semwarn",
                                 "issue warning when input stream semaphore >1",
                                 FPTYPE_ONOFF,
                                 FPFLAG_DEFAULT_INPUT,
                                 pNull,
                                 NULL);

    // ==============================================
    // ======== START FPS CONF LOOP =================
    // ==============================================
    FPS_CONFLOOP_START // macro in function_parameter.h

    // ==============================================
    // ======== STOP FPS CONF LOOP ==================
    // ==============================================
    FPS_CONFLOOP_END // macro in function_parameter.h

    return RETURN_SUCCESS;
}

/**
 * @brief MVM, GPU-based
 *
 *
 * Used for AO application, single GPU
 * This is meant to be used as stand-alone MVM process managed by cacao
 *
 *
 *
 *
 * [axmode 0] Converting WFS image to modes
 * Input is 2D (WFS)
 * Output is 1D (modes)
 *
 * Matrix is 3D
 * (size[0], size[1]) = (sizeWFS[0], sizeWFS[1])
 * (size[2]) = NBmodes
 *
 *
 *
 *
 *
 * [axmode 1] Expanding DM vector to WFS vector
 * Input is 2D vector (DM)
 * Output is 2D vector (WFS)
 *
 * Matrix is 3D.
 * (size[0], size[1]) = (sizeWFS[0], sizeWFS[1])
 * (size[2]) = sizeDM[0] x sizeDM[1]
 *
 * Matrix internally remapped to :
 * (size[0], size[1]) = (sizeDM[0], sizeDM[1])
 * (size[2]) = sizeWFS[0] x sizeWFS[1]
 *
 *
 */

errno_t __attribute__((hot)) CUDACOMP_MVMextractModesLoop_RUN()
{
    imageID IDmodes;
    imageID ID;
    imageID ID_modeval;

    cublasHandle_t        cublasH       = NULL;
    cublasStatus_t        cublas_status = CUBLAS_STATUS_SUCCESS;
    cudaError_t           cudaStat      = cudaSuccess;
    struct cudaDeviceProp deviceProp;
    int                   m, n;
    int                   k;
    uint32_t             *arraytmp;

    float *d_modes   = NULL; // linear memory of GPU
    float *d_in      = NULL;
    float *d_modeval = NULL;

    float alpha = 1.0;
    float beta  = 0.0;

    //long scnt;
    int  semval;
    long ii, jj, kk;

    long   NBmodes;
    float *normcoeff;

    //imageID IDoutact;
    uint32_t *sizearraytmp;

    //imageID ID_modeval_mult;
    int imOK;

    char    traceim_name[STRINGMAXLEN_IMGNAME];
    long    TRACEsize  = 2000;
    long    TRACEindex = 0;
    imageID IDtrace;

    uint32_t NBaveSTEP =
        10; // each step is 2x longer average than previous step
    double  stepcoeff;
    double  stepcoeff0 = 0.3;
    char    process_ave_name[STRINGMAXLEN_IMGNAME];
    char    process_rms_name[STRINGMAXLEN_IMGNAME];
    imageID IDprocave;
    imageID IDprocrms;
    long    step;

    double tmpv;

    float *modevalarray;
    float *modevalarrayref;

    int     initref  = 0; // 1 when reference has been processed
    int     BETAMODE = 0;
    imageID IDrefout;

    uint32_t        refindex;
    long            twait1;
    struct timespec t0;

    struct timespec t00;
    struct timespec t01;
    struct timespec t02;
    struct timespec t03;
    struct timespec t04;
    struct timespec t05;
    struct timespec t06;

    struct timespec t1;

    int MODEVALCOMPUTE = 1; // 1 if compute, 0 if import

    int RT_priority = 91; //any number from 0-99

    int devicecntMax = 100;


    FPS_CONNECT(data.FPS_name, FPSCONNECT_RUN);

    // ===============================
    // GET FUNCTION PARAMETER VALUES
    // ===============================

    char in_stream[STRINGMAXLEN_IMGNAME];
    strncpy(in_stream,
            functionparameter_GetParamPtr_STRING(&fps, ".sname_in"),
            FUNCTION_PARAMETER_STRMAXLEN);

    char IDmodes_name[STRINGMAXLEN_IMGNAME];
    strncpy(IDmodes_name,
            functionparameter_GetParamPtr_STRING(&fps, ".sname_modes"),
            FUNCTION_PARAMETER_STRMAXLEN);

    char intot_stream[STRINGMAXLEN_IMGNAME];
    strncpy(intot_stream,
            functionparameter_GetParamPtr_STRING(&fps, ".option.sname_intot"),
            FUNCTION_PARAMETER_STRMAXLEN);

    char IDrefin_name[STRINGMAXLEN_IMGNAME];
    strncpy(IDrefin_name,
            functionparameter_GetParamPtr_STRING(&fps, ".option.sname_refin"),
            FUNCTION_PARAMETER_STRMAXLEN);

    char IDrefout_name[STRINGMAXLEN_IMGNAME];
    strncpy(IDrefout_name,
            functionparameter_GetParamPtr_STRING(&fps, ".option.sname_refout"),
            FUNCTION_PARAMETER_STRMAXLEN);

    char IDmodes_val_name[STRINGMAXLEN_IMGNAME];
    strncpy(IDmodes_val_name,
            functionparameter_GetParamPtr_STRING(&fps, ".sname_outmodesval"),
            FUNCTION_PARAMETER_STRMAXLEN);

    int outinit = functionparameter_GetParamValue_ONOFF(&fps, ".outinit");

    int GPUindex = functionparameter_GetParamValue_INT64(&fps, ".GPUindex");
    int PROCESS =
        functionparameter_GetParamValue_ONOFF(&fps, ".option.PROCESS");
    int TRACEMODE =
        functionparameter_GetParamValue_ONOFF(&fps, ".option.TRACEMODE");
    int MODENORM =
        functionparameter_GetParamValue_ONOFF(&fps, ".option.MODENORM");
    int insem   = functionparameter_GetParamValue_INT64(&fps, ".option.insem");
    int axmode  = functionparameter_GetParamValue_INT64(&fps, ".option.axmode");
    long *twait = functionparameter_GetParamPtr_INT64(&fps, ".option.twait");
    int   semwarn =
        functionparameter_GetParamValue_ONOFF(&fps, ".option.semwarn");

    // ===============================
    // Review input parameters
    // ===============================

    printf("\n");
    printf("in_stream        : %16s  input stream\n", in_stream);
    printf("intot_stream     : %16s  [optional] input normalization stream\n",
           intot_stream);
    printf("IDmodes_name     : %16s  Modes\n", IDmodes_name);
    printf(
        "IDrefin_name     : %16s  [optional] input reference  - to be "
        "subtracted\n",
        IDrefin_name);
    printf(
        "IDrefout_name    : %16s  [optional] output reference - to be added\n",
        IDrefout_name);
    printf("IDmodes_val_name : %16s  ouput stream\n", IDmodes_val_name);

    printf("GPUindex         : %16d  GPU index\n", GPUindex);
    printf("PROCESS          : %16d  1 if postprocessing\n", PROCESS);
    printf("TRACEMODE        : %16d  1 if writing trace\n", TRACEMODE);
    printf("MODENORM         : %16d  1 if input modes should be normalized\n",
           MODENORM);
    printf("insem            : %16d  input semaphore index\n", insem);
    printf(
        "axmode           : %16d  0 for normal mode extraction, 1 for "
        "expansion\n",
        axmode);
    printf(
        "twait            : %16ld  if >0, insert time wait [us] at each "
        "iteration\n",
        *twait);
    printf(
        "semwarn          : %16d  1 if warning when input stream semaphore "
        ">1\n",
        semwarn);
    printf("\n");

    // ===========================
    // processinfo support
    // ===========================
    char pinfoname[STRINGMAXLEN_PROCESSINFO_NAME];
    {
        int slen = snprintf(pinfoname,
                            STRINGMAXLEN_PROCESSINFO_NAME,
                            "cudaMVMextract-%s",
                            in_stream);
        if(slen < 1)
        {
            PRINT_ERROR("snprintf wrote <1 char");
            abort(); // can't handle this error any other way
        }
        if(slen >= STRINGMAXLEN_PROCESSINFO_NAME)
        {
            PRINT_ERROR("snprintf string truncation");
            abort(); // can't handle this error any other way
        }
    }

    char pinfodescr[STRINGMAXLEN_PROCESSINFO_DESCRIPTION];
    {
        int slen = snprintf(pinfodescr,
                            STRINGMAXLEN_PROCESSINFO_DESCRIPTION,
                            "%s->%s",
                            in_stream,
                            IDmodes_val_name);
        if(slen < 1)
        {
            PRINT_ERROR("snprintf wrote <1 char");
            abort(); // can't handle this error any other way
        }
        if(slen >= STRINGMAXLEN_PROCESSINFO_DESCRIPTION)
        {
            PRINT_ERROR("snprintf string truncation");
            abort(); // can't handle this error any other way
        }
    }

    char pinfomsg[STRINGMAXLEN_PROCESSINFO_STATUSMSG];
    {
        int slen =
            snprintf(pinfomsg, STRINGMAXLEN_PROCESSINFO_STATUSMSG, "Setup");
        if(slen < 1)
        {
            PRINT_ERROR("snprintf wrote <1 char");
            abort(); // can't handle this error any other way
        }
        if(slen >= STRINGMAXLEN_PROCESSINFO_STATUSMSG)
        {
            PRINT_ERROR("snprintf string truncation");
            abort(); // can't handle this error any other way
        }
    }

    PROCESSINFO *processinfo;
    processinfo = processinfo_setup(
                      pinfoname, // short name for the processinfo instance, no spaces, no dot, name should be human-readable
                      pinfodescr, // description
                      pinfomsg,   // message on startup
                      __FUNCTION__,
                      __FILE__,
                      __LINE__);

    // OPTIONAL SETTINGS
    processinfo->MeasureTiming = 1; // Measure timing
    processinfo->RT_priority =
        RT_priority; // RT_priority, 0-99. Larger number = higher priority. If <0, ignore

    int loopOK = 1;

    // ===========================
    // INITIALIZATIONS
    // ===========================

    // CONNECT TO INPUT STREAM
    long IDin;
    IDin = image_ID(in_stream);

    // ERROR HANDLING
    if(IDin == -1)
    {
        struct timespec errtime;
        struct tm      *errtm;

        clock_gettime(CLOCK_MILK, &errtime);
        errtm = gmtime(&errtime.tv_sec);

        fprintf(stderr,
                "%02d:%02d:%02d.%09ld  ERROR [%s %s %d] Input stream %s does "
                "not exist, cannot proceed\n",
                errtm->tm_hour,
                errtm->tm_min,
                errtm->tm_sec,
                errtime.tv_nsec,
                __FILE__,
                __FUNCTION__,
                __LINE__,
                in_stream);
        return 1;
    }

    m = data.image[IDin].md[0].size[0] * data.image[IDin].md[0].size[1];

    // NORMALIZATION
    // CONNECT TO TOTAL FLUX STREAM
    long IDintot;
    IDintot        = image_ID(intot_stream);
    int INNORMMODE = 0; // 1 if input normalized

    if(IDintot == -1)
    {
        INNORMMODE = 0;
        create_2Dimage_ID("intot_tmp", 1, 1, &IDintot);
        data.image[IDintot].array.F[0] = 1.0;
    }
    else
    {
        INNORMMODE = 1;
    }

    // CONNECT TO WFS REFERENCE STREAM
    long IDref;
    IDref = image_ID(IDrefin_name);
    if(IDref == -1)
    {
        create_2Dimage_ID("_tmprefin",
                          data.image[IDin].md[0].size[0],
                          data.image[IDin].md[0].size[1],
                          &IDref);

        for(ii = 0; ii < data.image[IDin].md[0].size[0] *
                data.image[IDin].md[0].size[1];
                ii++)
        {
            data.image[IDref].array.F[ii] = 0.0;
        }
    }

    if(axmode == 0)
    {
        //
        // Extract modes.
        // This is the default geometry, no need to remap
        //
        IDmodes = image_ID(IDmodes_name);
        n       = data.image[IDmodes].md[0].size[2];
        NBmodes = n;
    }
    else
    {
        //
        // Expand from DM to WFS
        // Remap to new matrix tmpmodes
        //
        ID = image_ID(IDmodes_name);
        printf("Modes: ID = %ld   %s\n", ID, IDmodes_name);
        fflush(stdout);

        NBmodes = data.image[ID].md[0].size[0] * data.image[ID].md[0].size[1];
        n       = NBmodes;
        printf("NBmodes = %ld\n", NBmodes);
        fflush(stdout);

        printf("creating _tmpmodes  %ld %ld %ld\n",
               (long) data.image[IDin].md[0].size[0],
               (long) data.image[IDin].md[0].size[1],
               NBmodes);
        fflush(stdout);

        create_3Dimage_ID("_tmpmodes",
                          data.image[IDin].md[0].size[0],
                          data.image[IDin].md[0].size[1],
                          NBmodes,
                          &IDmodes);

        for(ii = 0; ii < data.image[IDin].md[0].size[0]; ii++)
            for(jj = 0; jj < data.image[IDin].md[0].size[1]; jj++)
            {
                for(kk = 0; kk < NBmodes; kk++)
                {
                    data.image[IDmodes]
                    .array.F[kk * data.image[IDin].md[0].size[0] *
                                data.image[IDin].md[0].size[1] +
                                jj * data.image[IDin].md[0].size[0] + ii] =
                                 data.image[ID]
                                 .array
                                 .F[NBmodes *
                                            (jj * data.image[IDin].md[0].size[0] + ii) +
                                            kk];
                }
            }

        //save_fits("_tmpmodes", "_test_tmpmodes.fits");
    }

    normcoeff = (float *) malloc(sizeof(float) * NBmodes);

    if(MODENORM == 1)
    {
        for(k = 0; k < NBmodes; k++)
        {
            normcoeff[k] = 0.0;
            for(ii = 0; ii < m; ii++)
            {
                normcoeff[k] += data.image[IDmodes].array.F[k * m + ii] *
                                data.image[IDmodes].array.F[k * m + ii];
            }
        }
    }
    else
        for(k = 0; k < NBmodes; k++)
        {
            normcoeff[k] = 1.0;
        }

    modevalarray    = (float *) malloc(sizeof(float) * n);
    modevalarrayref = (float *) malloc(sizeof(float) * n);

    arraytmp = (uint32_t *) malloc(sizeof(uint32_t) * 2);

    IDrefout = image_ID(IDrefout_name);
    if(IDrefout == -1)
    {
        arraytmp[0] = NBmodes;
        arraytmp[1] = 1;
    }
    else
    {
        arraytmp[0] = data.image[IDrefout].md[0].size[0];
        arraytmp[1] = data.image[IDrefout].md[0].size[1];
    }

    // CONNNECT TO OUTPUT STREAM

    ID_modeval = image_ID(IDmodes_val_name);
    if(ID_modeval == -1)
    {
        // CREATE IT
        create_image_ID(IDmodes_val_name,
                        2,
                        arraytmp,
                        _DATATYPE_FLOAT,
                        1,
                        0,
                        0,
                        &ID_modeval);
        MODEVALCOMPUTE = 1;
    }
    else
    {
        // USE STREAM, DO NOT COMPUTE IT
        printf("======== Using pre-existing stream %s, insem = %d\n",
               IDmodes_val_name,
               insem);
        fflush(stdout);

        if(outinit == 0)
        {
            MODEVALCOMPUTE = 0;
        }
        else
        {
            MODEVALCOMPUTE = 1;
        }

        // drive semaphore to zero
        while(ImageStreamIO_semtrywait(data.image+ID_modeval, insem) == 0)
        {
            printf("WARNING %s %d  : sem_trywait on ID_modeval\n",
                   __FILE__,
                   __LINE__);
            fflush(stdout);
        }
    }

    free(arraytmp);

    printf("OUTPUT STREAM : %s  ID: %ld\n", IDmodes_val_name, ID_modeval);
    list_image_ID();

    if(MODEVALCOMPUTE == 1)
    {
        int deviceCount;

        cudaGetDeviceCount(&deviceCount);
        if(deviceCount > devicecntMax)
        {
            deviceCount = 0;
        }
        if(deviceCount < 0)
        {
            deviceCount = 0;
        }

        printf("%s: %d devices found\n", __func__, deviceCount);
        fflush(stdout);
        printf("\n");
        for(k = 0; k < deviceCount; k++)
        {
            cudaGetDeviceProperties(&deviceProp, k);
            printf("Device %d / %d [ %20s ]  has compute capability %d.%d.\n",
                   k,
                   deviceCount,
                   deviceProp.name,
                   deviceProp.major,
                   deviceProp.minor);
            printf(
                "  Total amount of global memory:                 %.0f MBytes "
                "(%llu bytes)\n",
                (float) deviceProp.totalGlobalMem / 1048576.0f,
                (unsigned long long) deviceProp.totalGlobalMem);
            printf("  (%2d) Multiprocessors\n", deviceProp.multiProcessorCount);
            printf(
                "  GPU Clock rate:                                %.0f MHz "
                "(%0.2f GHz)\n",
                deviceProp.clockRate * 1e-3f,
                deviceProp.clockRate * 1e-6f);
            printf("\n");
        }

        if(GPUindex < deviceCount)
        {
            cudaSetDevice(GPUindex);
        }
        else
        {
            printf("Invalid Device : %d / %d\n", GPUindex, deviceCount);
            exit(0);
        }

        printf("Create cublas handle ...");
        fflush(stdout);
        cublas_status = cublasCreate(&cublasH);
        if(cublas_status != CUBLAS_STATUS_SUCCESS)
        {
            printf("CUBLAS initialization failed\n");
            return EXIT_FAILURE;
        }
        printf(" done\n");
        fflush(stdout);

        // load modes to GPU
        cudaStat = cudaMalloc((void **) &d_modes, sizeof(float) * m * NBmodes);
        if(cudaStat != cudaSuccess)
        {
            printf("cudaMalloc d_modes returned error code %d, line %d\n",
                   cudaStat,
                   __LINE__);
            exit(EXIT_FAILURE);
        }
        cudaStat = cudaMemcpy(d_modes,
                              data.image[IDmodes].array.F,
                              sizeof(float) * m * NBmodes,
                              cudaMemcpyHostToDevice);
        if(cudaStat != cudaSuccess)
        {
            printf("cudaMemcpy returned error code %d, line %d\n",
                   cudaStat,
                   __LINE__);
            exit(EXIT_FAILURE);
        }

        // create d_in
        cudaStat = cudaMalloc((void **) &d_in, sizeof(float) * m);
        if(cudaStat != cudaSuccess)
        {
            printf("cudaMalloc d_in returned error code %d, line %d\n",
                   cudaStat,
                   __LINE__);
            exit(EXIT_FAILURE);
        }

        // create d_modeval
        cudaStat = cudaMalloc((void **) &d_modeval, sizeof(float) * NBmodes);
        if(cudaStat != cudaSuccess)
        {
            printf("cudaMalloc d_modeval returned error code %d, line %d\n",
                   cudaStat,
                   __LINE__);
            exit(EXIT_FAILURE);
        }
    }

    //loopcnt = 0;

    if(TRACEMODE == 1)
    {
        sizearraytmp = (uint32_t *) malloc(sizeof(uint32_t) * 2);

        {
            int slen = snprintf(traceim_name,
                                STRINGMAXLEN_IMGNAME,
                                "%s_trace",
                                IDmodes_val_name);
            if(slen < 1)
            {
                PRINT_ERROR("snprintf wrote <1 char");
                abort(); // can't handle this error any other way
            }
            if(slen >= STRINGMAXLEN_IMGNAME)
            {
                PRINT_ERROR("snprintf string truncation");
                abort(); // can't handle this error any other way
            }
        }

        sizearraytmp[0] = TRACEsize;
        sizearraytmp[1] = NBmodes;
        IDtrace         = image_ID(traceim_name);
        imOK            = 1;
        if(IDtrace == -1)
        {
            imOK = 0;
        }
        else
        {
            if((data.image[IDtrace].md[0].size[0] != TRACEsize) ||
                    (data.image[IDtrace].md[0].size[1] != NBmodes))
            {
                imOK = 0;
                delete_image_ID(traceim_name, DELETE_IMAGE_ERRMODE_WARNING);
            }
        }
        if(imOK == 0)
        {
            create_image_ID(traceim_name,
                            2,
                            sizearraytmp,
                            _DATATYPE_FLOAT,
                            1,
                            0,
                            0,
                            &IDtrace);
        }
        free(sizearraytmp);
    }

    if(PROCESS == 1)
    {
        sizearraytmp = (uint32_t *) malloc(sizeof(uint32_t) * 2);

        {
            int slen = snprintf(process_ave_name,
                                STRINGMAXLEN_IMGNAME,
                                "%s_ave",
                                IDmodes_val_name);
            if(slen < 1)
            {
                PRINT_ERROR("snprintf wrote <1 char");
                abort(); // can't handle this error any other way
            }
            if(slen >= STRINGMAXLEN_IMGNAME)
            {
                PRINT_ERROR("snprintf string truncation");
                abort(); // can't handle this error any other way
            }
        }

        sizearraytmp[0] = NBmodes;
        sizearraytmp[1] = NBaveSTEP;
        IDprocave       = image_ID(process_ave_name);
        imOK            = 1;
        if(IDprocave == -1)
        {
            imOK = 0;
        }
        else
        {
            if((data.image[IDprocave].md[0].size[0] != NBmodes) ||
                    (data.image[IDprocave].md[0].size[1] != NBaveSTEP))
            {
                imOK = 0;
                delete_image_ID(process_ave_name, DELETE_IMAGE_ERRMODE_WARNING);
            }
        }
        if(imOK == 0)
        {
            create_image_ID(process_ave_name,
                            2,
                            sizearraytmp,
                            _DATATYPE_FLOAT,
                            1,
                            0,
                            0,
                            &IDprocave);
        }
        free(sizearraytmp);

        sizearraytmp = (uint32_t *) malloc(sizeof(uint32_t) * 2);

        {
            int slen = snprintf(process_rms_name,
                                STRINGMAXLEN_IMGNAME,
                                "%s_rms",
                                IDmodes_val_name);
            if(slen < 1)
            {
                PRINT_ERROR("snprintf wrote <1 char");
                abort(); // can't handle this error any other way
            }
            if(slen >= STRINGMAXLEN_IMGNAME)
            {
                PRINT_ERROR("snprintf string truncation");
                abort(); // can't handle this error any other way
            }
        }

        sizearraytmp[0] = NBmodes;
        sizearraytmp[1] = NBaveSTEP;
        IDprocrms       = image_ID(process_rms_name);
        imOK            = 1;
        if(IDprocrms == -1)
        {
            imOK = 0;
        }
        else
        {
            if((data.image[IDprocrms].md[0].size[0] != NBmodes) ||
                    (data.image[IDprocrms].md[0].size[1] != NBaveSTEP))
            {
                imOK = 0;
                delete_image_ID(process_rms_name, DELETE_IMAGE_ERRMODE_WARNING);
            }
        }
        if(imOK == 0)
        {
            create_image_ID(process_rms_name,
                            2,
                            sizearraytmp,
                            _DATATYPE_FLOAT,
                            1,
                            0,
                            0,
                            &IDprocrms);
        }
        free(sizearraytmp);
    }

    initref = 0;

    twait1 = *twait;

    printf("LOOP START   MODEVALCOMPUTE = %d\n", MODEVALCOMPUTE);
    fflush(stdout);

    if(MODEVALCOMPUTE == 0)
    {
        printf("\n");
        printf("This function is NOT computing mode values\n");
        printf("Pre-existing stream %s was detected\n", IDmodes_val_name);
        printf("\n");
        if(data.processinfo == 1)
        {
            strcpy(processinfo->statusmsg, "Passing stream, no computation");
            //sprintf(processinfo->description, "passthrough, no comp");
        }
    }
    else
    {
        char msgstring[STRINGMAXLEN_PROCESSINFO_STATUSMSG];

        {
            int slen = snprintf(msgstring,
                                STRINGMAXLEN_PROCESSINFO_STATUSMSG,
                                "Running on GPU %d",
                                GPUindex);
            if(slen < 1)
            {
                PRINT_ERROR("snprintf wrote <1 char");
                abort(); // can't handle this error any other way
            }
            if(slen >= STRINGMAXLEN_PROCESSINFO_STATUSMSG)
            {
                PRINT_ERROR("snprintf string truncation");
                abort(); // can't handle this error any other way
            }
        }
        if(data.processinfo == 1)
        {
            strcpy(processinfo->statusmsg, msgstring);
        }
    }

    // ==================================
    // STARTING LOOP
    // ==================================
    processinfo_loopstart(
        processinfo); // Notify processinfo that we are entering loop

    if(MODEVALCOMPUTE == 1)
    {
        int slen = snprintf(pinfomsg,
                            STRINGMAXLEN_PROCESSINFO_STATUSMSG,
                            "MVM %s %s -> %s TRACE=%d PROC=%d",
                            IDmodes_name,
                            in_stream,
                            IDmodes_val_name,
                            TRACEMODE,
                            PROCESS);
        if(slen < 1)
        {
            PRINT_ERROR("snprintf wrote <1 char");
            abort(); // can't handle this error any other way
        }
        if(slen >= STRINGMAXLEN_PROCESSINFO_STATUSMSG)
        {
            PRINT_ERROR("snprintf string truncation");
            abort(); // can't handle this error any other way
        }
    }
    else
    {
        int slen = snprintf(pinfomsg,
                            STRINGMAXLEN_PROCESSINFO_STATUSMSG,
                            "passthrough %s TRACE=%d PROC=%d",
                            IDmodes_val_name,
                            TRACEMODE,
                            PROCESS);
        if(slen < 1)
        {
            PRINT_ERROR("snprintf wrote <1 char");
            abort(); // can't handle this error any other way
        }
        if(slen >= STRINGMAXLEN_PROCESSINFO_STATUSMSG)
        {
            PRINT_ERROR("snprintf string truncation");
            abort(); // can't handle this error any other way
        }
    }
    processinfo_WriteMessage(processinfo, pinfomsg);

    // set up input trigger stream
    processinfo_waitoninputstream_init(processinfo,
                                       IDin,
                                       PROCESSINFO_TRIGGERMODE_SEMAPHORE,
                                       5);

    while(loopOK == 1)
    {
        struct timespec tdiff;
        double          tdiffv;

        //int t00OK = 0;
        //int t01OK = 0;
        //int t02OK = 0;
        //int t03OK = 0;
        //int t04OK = 0;
        //int t05OK = 0;
        //int t06OK = 0;

        loopOK = processinfo_loopstep(processinfo);

        clock_gettime(CLOCK_MILK, &t0);

        // We either compute the result in this function (MODEVALCOMPUTE = 1)
        // or we read it from ID_modeval stream (MODEVALCOMPUTE = 0)

        if(MODEVALCOMPUTE == 1)
        {

            int doComputation = 0;

            // Are we computing a new reference ?
            // if yes, set initref to 0 (reference is NOT initialized)
            //
            if(refindex != data.image[IDref].md[0].cnt0)
            {
                initref  = 0;
                refindex = data.image[IDref].md[0].cnt0;
            }

            if(initref == 1)
            {
                // Reference is already initialized
                // wait for input stream to be changed to start computation
                //

                processinfo_waitoninputstream(processinfo);
                if(processinfo->triggerstatus ==
                        PROCESSINFO_TRIGGERSTATUS_RECEIVED)
                {
                    doComputation = 1;
                }
                else
                {
                    doComputation = 0;
                }
            }
            else
            {
                // compute response of reference immediately
                printf("COMPUTE NEW REFERENCE RESPONSE\n");
                doComputation = 1;
            }

            //t00OK = 1;
            clock_gettime(CLOCK_MILK, &t00);

            processinfo_exec_start(processinfo);

            if(doComputation == 1)
            {
                // load in_stream to GPU
                if(initref == 0)
                {
                    cudaStat = cudaMemcpy(d_in,
                                          data.image[IDref].array.F,
                                          sizeof(float) * m,
                                          cudaMemcpyHostToDevice);
                }
                else
                {
                    cudaStat = cudaMemcpy(d_in,
                                          data.image[IDin].array.F,
                                          sizeof(float) * m,
                                          cudaMemcpyHostToDevice);
                }

                if(cudaStat != cudaSuccess)
                {
                    printf("initref = %d    %ld  %ld\n", initref, IDref, IDin);
                    printf("cudaMemcpy returned error code %d, line %d\n",
                           cudaStat,
                           __LINE__);
                    exit(EXIT_FAILURE);
                }

                //t01OK = 1;
                clock_gettime(CLOCK_MILK, &t01);

                if(BETAMODE == 1)
                {
                    beta     = -1.0;
                    cudaStat = cudaMemcpy(d_modeval,
                                          modevalarrayref,
                                          sizeof(float) * NBmodes,
                                          cudaMemcpyHostToDevice);
                }

                //t02OK = 1;
                clock_gettime(CLOCK_MILK, &t02);

                // compute
                cublas_status = cublasSgemv(cublasH,
                                            CUBLAS_OP_T,
                                            m,
                                            NBmodes,
                                            &alpha,
                                            d_modes,
                                            m,
                                            d_in,
                                            1,
                                            &beta,
                                            d_modeval,
                                            1);
                if(cublas_status != CUBLAS_STATUS_SUCCESS)
                {
                    printf("cublasSgemv returned error code %d, line(%d)\n",
                           cublas_status,
                           __LINE__);
                    fflush(stdout);
                    if(cublas_status == CUBLAS_STATUS_NOT_INITIALIZED)
                    {
                        printf("   CUBLAS_STATUS_NOT_INITIALIZED\n");
                    }
                    if(cublas_status == CUBLAS_STATUS_INVALID_VALUE)
                    {
                        printf("   CUBLAS_STATUS_INVALID_VALUE\n");
                    }
                    if(cublas_status == CUBLAS_STATUS_ARCH_MISMATCH)
                    {
                        printf("   CUBLAS_STATUS_ARCH_MISMATCH\n");
                    }
                    if(cublas_status == CUBLAS_STATUS_EXECUTION_FAILED)
                    {
                        printf("   CUBLAS_STATUS_EXECUTION_FAILED\n");
                    }

                    printf("GPU index                           = %d\n",
                           GPUindex);

                    printf("CUBLAS_OP                           = %d\n",
                           CUBLAS_OP_T);
                    printf("alpha                               = %f\n", alpha);
                    printf("alpha                               = %f\n", beta);
                    printf("m                                   = %d\n",
                           (int) m);
                    printf("NBmodes                             = %d\n",
                           (int) NBmodes);
                    fflush(stdout);
                    exit(EXIT_FAILURE);
                }

                // copy result
                data.image[ID_modeval].md[0].write = 1;

                //t03OK = 1;
                clock_gettime(CLOCK_MILK, &t03);

                if(initref == 0)
                {
                    // construct reference to be subtracted
                    printf("... reference compute\n");
                    cudaStat = cudaMemcpy(modevalarrayref,
                                          d_modeval,
                                          sizeof(float) * NBmodes,
                                          cudaMemcpyDeviceToHost);

                    IDrefout = image_ID(IDrefout_name);
                    if(IDrefout != -1)
                        for(k = 0; k < NBmodes; k++)
                        {
                            modevalarrayref[k] -=
                                data.image[IDrefout].array.F[k];
                        }

                    if((INNORMMODE == 0) && (MODENORM == 0))
                    {
                        BETAMODE =
                            1; // include ref subtraction in GPU operation
                    }
                    else
                    {
                        BETAMODE = 0;
                    }
                }
                else
                {
                    cudaStat = cudaMemcpy(modevalarray,
                                          d_modeval,
                                          sizeof(float) * NBmodes,
                                          cudaMemcpyDeviceToHost);

                    if(BETAMODE == 0)
                    {
                        for(k = 0; k < NBmodes; k++)
                        {
                            data.image[ID_modeval].array.F[k] =
                                (modevalarray[k] /
                                 data.image[IDintot].array.F[0] -
                                 modevalarrayref[k]) /
                                normcoeff[k];
                        }
                    }
                    else
                        for(k = 0; k < NBmodes; k++)
                        {
                            data.image[ID_modeval].array.F[k] = modevalarray[k];
                        }

                    processinfo_update_output_stream(processinfo, ID_modeval);
                }
            }
        }
        else
        {
            // WAIT FOR NEW MODEVAL
            int rval;
            rval = ImageStreamIO_semwait(data.image+ID_modeval, insem);
            if(rval == -1)  // interrupt
            {
                loopOK = 0;
            }

            processinfo_exec_start(processinfo);
        }

        //t04OK = 1;
        clock_gettime(CLOCK_MILK, &t04);

        if(TRACEMODE == 1)
        {
            data.image[ID_modeval].md[0].write = 1;

            for(k = 0; k < NBmodes; k++)
            {
                data.image[IDtrace].array.F[k * TRACEsize + TRACEindex] =
                    data.image[ID_modeval].array.F[k];
            }
            data.image[IDtrace].md[0].cnt1 = TRACEindex;

            semval = ImageStreamIO_semvalue(data.image+IDtrace, 0);
            if(semval < SEMAPHORE_MAXVAL)
            {
                ImageStreamIO_sempost(data.image+IDtrace, 0);
            }
            semval = ImageStreamIO_semvalue(data.image+IDtrace, 1);
            if(semval < SEMAPHORE_MAXVAL)
            {
                ImageStreamIO_sempost(data.image+IDtrace, 1);
            }
            data.image[IDtrace].md[0].cnt0++;
            data.image[IDtrace].md[0].write = 0;

            TRACEindex++;
            if(TRACEindex >= TRACEsize)
            {
                TRACEindex = 0;
                // copy to tracef shared memory (frozen trace)
            }
        }

        //t05OK = 1;
        clock_gettime(CLOCK_MILK, &t05);

        if(PROCESS == 1)
        {
            stepcoeff                         = stepcoeff0;
            data.image[IDprocave].md[0].write = 1;
            for(step = 0; step < NBaveSTEP; step++)
            {
                for(k = 0; k < NBmodes; k++)
                {
                    data.image[IDprocave].array.F[NBmodes * step + k] =
                        (1.0 - stepcoeff) *
                        data.image[IDprocave].array.F[NBmodes * step + k] +
                        stepcoeff * data.image[ID_modeval].array.F[k];
                }
                stepcoeff *= stepcoeff0;
            }
            processinfo_update_output_stream(processinfo, IDprocave);

            stepcoeff                         = stepcoeff0;
            data.image[IDprocrms].md[0].write = 1;
            for(step = 0; step < NBaveSTEP; step++)
            {
                for(k = 0; k < NBmodes; k++)
                {
                    tmpv = data.image[ID_modeval].array.F[k] -
                           data.image[IDprocave].array.F[NBmodes * step + k];
                    tmpv = tmpv * tmpv;
                    data.image[IDprocrms].array.F[NBmodes * step + k] =
                        (1.0 - stepcoeff) *
                        data.image[IDprocrms].array.F[NBmodes * step + k] +
                        stepcoeff * tmpv;
                }
                stepcoeff *= stepcoeff0;
            }

            processinfo_update_output_stream(processinfo, IDprocrms);
        }

        //t06OK = 1;
        clock_gettime(CLOCK_MILK, &t06);

        processinfo_exec_end(processinfo);

        if(twait1 < 0)
        {
            twait1 = 0;
        }

        if(*twait > 0)
        {
            struct timespec treq, trem;

            treq.tv_sec  = (long)(twait1 / 1000000);
            treq.tv_nsec = (long)(1e9 * (twait1 - treq.tv_sec * 1000000));
            nanosleep(&treq, &trem);
        }

        clock_gettime(CLOCK_MILK, &t1);
        tdiff  = timespec_diff(t0, t1);
        tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

        if(tdiffv < 1.0e-6 * (*twait))
        {
            twait1++;
        }
        else
        {
            twait1--;
        }
        initref = 1;
    }

    processinfo_cleanExit(processinfo);
    function_parameter_RUNexit(&fps);

    if(MODEVALCOMPUTE == 1)
    {
        cudaFree(d_modes);
        cudaFree(d_in);
        cudaFree(d_modeval);

        if(cublasH)
        {
            cublasDestroy(cublasH);
        }
    }

    free(normcoeff);
    free(modevalarray);
    free(modevalarrayref);

    return RETURN_SUCCESS;
}

/**
 * @brief extract mode coefficients from data stream
 *
 * modes need to be orthogonal
 * single GPU computation
 *
 * @param[in]   in_stream            input stream
 * @param[in]   intot_stream         [optional]   input normalization stream
 * @param[in]   IDmodes_name         Modes
 * @param[in]   IDrefin_name         [optional] input reference  - to be subtracted
 * @param[in]   IDrefout_name        [optional] output reference - to be added
 * @param[out]  IDmodes_val_name     ouput stream
 * @param[in]   GPUindex             GPU index
 * @param[in]   PROCESS              1 if postprocessing
 * @param[in]   TRACEMODE            1 if writing trace
 * @param[in]   MODENORM             1 if input modes should be normalized
 * @param[in]   insem                input semaphore index
 * @param[in]   axmode               0 for normal mode extraction, 1 for expansion
 * @param[in]   twait		         if >0, insert time wait [us] at each iteration
 * @param[in]   semwarn              1 if warning when input stream semaphore >1
 *
 * @note IMPORTANT: if IDmodes_val_name exits, use it and do not compute it
 *
 * @note if IDrefout_name exists, match output image size to IDrefout_name
 */

int __attribute__((hot)) CUDACOMP_MVMextractModesLoop(
    const char *in_stream,     // input stream
    const char *intot_stream,  // [optional]   input normalization stream
    const char *IDmodes_name,  // Modes matrix
    const char *IDrefin_name,  // [optional] input reference  - to be subtracted
    const char *IDrefout_name, // [optional] output reference - to be added
    const char *IDmodes_val_name, // ouput stream
    int         GPUindex,         // GPU index
    int         PROCESS,          // 1 if postprocessing
    int         TRACEMODE,        // 1 if writing trace
    int         MODENORM,         // 1 if input modes should be normalized
    int         insem,            // input semaphore index
    int         axmode, // 0 for normal mode extraction, 1 for expansion
    long        twait,  // if >0, insert time wait [us] at each iteration
    int         semwarn // 1 if warning when input stream semaphore >1
)
{

    // ==================================
    // CREATE FPS AND START CONF
    // ==================================

    long pindex = (long)
                  getpid(); // index used to differentiate multiple calls to function
    // if we don't have anything more informative, we use PID

    FUNCTION_PARAMETER_STRUCT fps;

    {
        // write FPS name
        int slen = snprintf(data.FPS_name,
                            STRINGMAXLEN_FPS_NAME,
                            "cudaMVMextmodes-%06ld",
                            pindex);
        if(slen < 1)
        {
            PRINT_ERROR("snprintf wrote <1 char");
            abort(); // can't handle this error any other way
        }
        if(slen >= STRINGMAXLEN_FPS_NAME)
        {
            PRINT_ERROR("snprintf string truncation");
            abort(); // can't handle this error any other way
        }
    }

    data.FPS_CMDCODE = FPSCMDCODE_FPSINIT;
    CUDACOMP_MVMextractModesLoop_FPCONF();

    // ==================================
    // SET PARAMETER VALUES
    // ==================================
    //int SMfd = -1;
    function_parameter_struct_connect(data.FPS_name, &fps, FPSCONNECT_SIMPLE);

    functionparameter_SetParamValue_STRING(&fps, ".sname_in", in_stream);
    functionparameter_SetParamValue_STRING(&fps, ".sname_modes", IDmodes_name);
    functionparameter_SetParamValue_STRING(&fps,
                                           ".option.sname_intot",
                                           intot_stream);
    functionparameter_SetParamValue_STRING(&fps,
                                           ".option.sname_refin",
                                           IDrefin_name);
    functionparameter_SetParamValue_STRING(&fps,
                                           ".option.sname_refout",
                                           IDrefout_name);
    functionparameter_SetParamValue_STRING(&fps,
                                           ".sname_outmodesval",
                                           IDmodes_val_name);

    functionparameter_SetParamValue_INT64(&fps, ".GPUindex", GPUindex);
    functionparameter_SetParamValue_ONOFF(&fps, ".option.PROCESS", PROCESS);
    functionparameter_SetParamValue_ONOFF(&fps, ".option.TRACEMODE", TRACEMODE);
    functionparameter_SetParamValue_ONOFF(&fps, ".option.MODENORM", MODENORM);
    functionparameter_SetParamValue_INT64(&fps, ".option.insem", insem);
    functionparameter_SetParamValue_INT64(&fps, ".option.axmode", axmode);
    functionparameter_SetParamValue_INT64(&fps, ".option.twait", twait);
    functionparameter_SetParamValue_ONOFF(&fps, ".option.semwarn", semwarn);

    function_parameter_struct_disconnect(&fps);

    // ==================================
    // START RUN PROCESS
    // ==================================

    CUDACOMP_MVMextractModesLoop_RUN();

    return RETURN_SUCCESS;
}

#endif
