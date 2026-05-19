/**
 * @file    stream_sem.c
 * @brief   Stream semaphore operations
 *
 * CLI and standalone FPS commands for operating on
 * ImageStreamIO semaphores. Provides five commands:
 *
 *  - imseminfo     — print semaphore status
 *  - imsetsempost  — post a single semaphore
 *  - imsetsempostl — post in a timed loop (primary)
 *  - imsetsemwait  — wait on a semaphore
 *  - imsetsemflush — flush (drain) a semaphore
 *
 * Each command is registered with the FPS framework
 * using the V2 X-macro parameter binding pattern.
 * The primary compute function (imsetsempostl)
 * supports full procinfo lifecycle management.
 */

#include <pthread.h>

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "image_ID.h"
#include "list_image.h"
#include "read_shmim.h"

static pthread_t *thrarray_semwait;
static long       NB_thrarray_semwait;

/* forward decls */
imageID COREMOD_MEMORY_image_seminfo(
    const char *IDname);
imageID COREMOD_MEMORY_image_set_sempost(
    const char *IDname,
    long       index);
/**
 * @brief Post to all semaphores of a stream (by ID).
 */
imageID COREMOD_MEMORY_image_set_sempost_byID(
    imageID ID,
    long    index);
/**
 * @brief Post to all semaphores except one (by ID).
 */
imageID COREMOD_MEMORY_image_set_sempost_excl_byID(
    imageID ID,
    long    index);
imageID COREMOD_MEMORY_image_set_sempost_loop(
    const char *IDname,
    long       index,
    long       dtus);
imageID COREMOD_MEMORY_image_set_semwait(
    const char *IDname,
    long       index);
void *waitforsemID(void *ID);
errno_t COREMOD_MEMORY_image_set_semwait_OR_IDarray(
    imageID *IDarray,
    long    NB_ID);
errno_t COREMOD_MEMORY_image_set_semflush_IDarray(
    imageID *IDarray,
    long    NB_ID);
imageID COREMOD_MEMORY_image_set_semflush(
    const char *IDname,
    long       index);


/* ================================================================
 *  COMMON PARAMS (image + semindex)
 * ============================================================= */

static char p_imname[FUNCTION_PARAMETER_STRMAXLEN] = "im1";
static long long p_semindex = 0;

#define FPS_PARAMS_IMSEM(X) \
    X(".imname", p_imname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "image name") \
    X(".semindex", &p_semindex, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "semaphore index")

#define FPS_PARAMS_IMSEM_INFO(X) \
    X(".imname", p_imname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "image name")

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static FPS_CLI_BINDING bindings_imsem_info[] =
{
    FPS_PARAMS_IMSEM_INFO(FPS_X_BINDING)
};
static const int nb_bindings_imsem_info = sizeof(bindings_imsem_info) / sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF farg_imsem_info[] =
{
    FPS_PARAMS_IMSEM_INFO(FPS_X_FARG)
};
#define CLICMD_FIELDS_IMSEM_INFO \
    __FILE__, sizeof(farg_imsem_info) / sizeof(CLICMDARGDEF), farg_imsem_info, CLICMDFLAG_FPS, NULL, NULL, NULL

static FPS_CLI_BINDING bindings_imsem[] =
{
    FPS_PARAMS_IMSEM(FPS_X_BINDING)
};
static const int nb_bindings_imsem = sizeof(bindings_imsem) / sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF farg_imsem[] =
{
    FPS_PARAMS_IMSEM(FPS_X_FARG)
};
#define CLICMD_FIELDS_IMSEM \
    __FILE__, sizeof(farg_imsem) / sizeof(CLICMDARGDEF), farg_imsem, CLICMDFLAG_FPS, NULL, NULL, NULL
#else
#define CLICMD_FIELDS_IMSEM_INFO \
    __FILE__, 0, NULL, 0, NULL, NULL, NULL
#define CLICMD_FIELDS_IMSEM \
    __FILE__, 0, NULL, 0, NULL, NULL, NULL
#endif


/* ================================================================
 *  CMD 1: imseminfo (1 arg)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_seminfo =
{
    .fps_name    = "imseminfo",
    .cmdkey      = "imseminfo",
    .description =
    "display semaphore info",
    .description_long =
    "Manage semaphores on shared memory image streams. Supports posting, waiting, flushing, and monitoring semaphore state for inter-process synchronization."
};

static CLICMDDATA CLIcmddata_seminfo =
{
    "", "", CLICMD_FIELDS_IMSEM_INFO
};
FPS_CMDSETTINGS_INIT(cms1, CLIcmddata_seminfo, FPS_app_info_seminfo)

static errno_t __attribute__((unused)) compute_seminfo()
{
    COREMOD_MEMORY_image_seminfo(p_imname);
    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 2: imsetsempost (2 args)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_sempost =
{
    .fps_name    = "imsetsempost",
    .cmdkey      = "imsetsempost",
    .description =
    "post image semaphore",
    .description_long =
    "Manage semaphores on shared memory image streams. Supports posting, waiting, flushing, and monitoring semaphore state for inter-process synchronization."
};

static CLICMDDATA CLIcmddata_sempost =
{
    "", "", CLICMD_FIELDS_IMSEM
};
FPS_CMDSETTINGS_INIT(cms2, CLIcmddata_sempost, FPS_app_info_sempost)

static errno_t __attribute__((unused)) compute_sempost()
{
    COREMOD_MEMORY_image_set_sempost(p_imname, p_semindex);
    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 3: imsetsempostl (3 args, primary)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info =
{
    .fps_name    = "imsetsempostl",
    .cmdkey      = "imsetsempostl",
    .description =
    "post image semaphore loop",
    .description_long =
    "Manage semaphores on shared memory image streams. Supports posting, waiting, flushing, and monitoring semaphore state for inter-process synchronization."
};

static long long p_dtus = 1000;

#define FPS_PARAMS(X) \
    FPS_PARAMS_IMSEM(X) \
    X(".dtus", &p_dtus, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "time interval [us]")

static FPS_CLI_BINDING my_bindings[] =
{
    FPS_PARAMS(FPS_X_BINDING)
};

static const int __attribute__((unused)) nb_bindings =
    sizeof(my_bindings) /
    sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] =
{
    FPS_PARAMS(FPS_X_FARG)
};

static CLICMDDATA CLIcmddata =
{
    "", "", CLICMD_FIELDS_DEFAULTS
};

FPS_CMDSETTINGS_INIT(cms3, CLIcmddata, FPS_app_info)

static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    COREMOD_MEMORY_image_set_sempost_loop(p_imname, p_semindex, p_dtus);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 4: imsetsemwait (2 args)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_semwait =
{
    .fps_name    = "imsetsemwait",
    .cmdkey      = "imsetsemwait",
    .description =
    "wait image semaphore",
    .description_long =
    "Manage semaphores on shared memory image streams. Supports posting, waiting, flushing, and monitoring semaphore state for inter-process synchronization."
};

static CLICMDDATA CLIcmddata_semwait =
{
    "", "", CLICMD_FIELDS_IMSEM
};
FPS_CMDSETTINGS_INIT(cms4, CLIcmddata_semwait, FPS_app_info_semwait)

static errno_t __attribute__((unused)) compute_semwait()
{
    COREMOD_MEMORY_image_set_semwait(p_imname, p_semindex);
    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 5: imsetsemflush (2 args)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_semflush =
{
    .fps_name    = "imsetsemflush",
    .cmdkey      = "imsetsemflush",
    .description =
    "flush image semaphore",
    .description_long =
    "Manage semaphores on shared memory image streams. Supports posting, waiting, flushing, and monitoring semaphore state for inter-process synchronization."
};

static CLICMDDATA CLIcmddata_semflush =
{
    "", "", CLICMD_FIELDS_IMSEM
};
FPS_CMDSETTINGS_INIT(cms5, CLIcmddata_semflush, FPS_app_info_semflush)

static errno_t __attribute__((unused)) compute_semflush()
{
    COREMOD_MEMORY_image_set_semflush(p_imname, p_semindex);
    return RETURN_SUCCESS;
}


/* ================================================================
 *  REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)

static errno_t CLIfunction_seminfo(void)
{
    return safe_fps_generic_CLIfunction(
               &FPS_app_info_seminfo,
               farg_imsem_info, &CLIcmddata_seminfo,
               bindings_imsem_info, nb_bindings_imsem_info, compute_seminfo);
}

static errno_t CLIfunction_sempost(void)
{
    return safe_fps_generic_CLIfunction(
               &FPS_app_info_sempost,
               farg_imsem, &CLIcmddata_sempost,
               bindings_imsem, nb_bindings_imsem, compute_sempost);
}

static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
               &FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings, compute_function);
}

static errno_t CLIfunction_semwait(void)
{
    return safe_fps_generic_CLIfunction(
               &FPS_app_info_semwait,
               farg_imsem, &CLIcmddata_semwait,
               bindings_imsem, nb_bindings_imsem, compute_semwait);
}

static errno_t CLIfunction_semflush(void)
{
    return safe_fps_generic_CLIfunction(
               &FPS_app_info_semflush,
               farg_imsem, &CLIcmddata_semflush,
               bindings_imsem, nb_bindings_imsem, compute_semflush);
}

errno_t
CLIADDCMD_COREMOD_memory__stream_sem()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    safe_fps_fill_farg_examples(farg_imsem, bindings_imsem, nb_bindings_imsem);

    {
        int cmdi = RegisterCLIcmd(CLIcmddata_seminfo, CLIfunction_seminfo);
        CLIcmddata_seminfo.cmdsettings = &data.cmd[cmdi].cmdsettings;
    }
    {
        int cmdi = RegisterCLIcmd(CLIcmddata_sempost, CLIfunction_sempost);
        CLIcmddata_sempost.cmdsettings = &data.cmd[cmdi].cmdsettings;
    }
    {
        int cmdi = RegisterCLIcmd(CLIcmddata, CLIfunction);
        CLIcmddata.cmdsettings = &data.cmd[cmdi].cmdsettings;
    }
    {
        int cmdi = RegisterCLIcmd(CLIcmddata_semwait, CLIfunction_semwait);
        CLIcmddata_semwait.cmdsettings = &data.cmd[cmdi].cmdsettings;
    }
    {
        int cmdi = RegisterCLIcmd(CLIcmddata_semflush, CLIfunction_semflush);
        CLIcmddata_semflush.cmdsettings = &data.cmd[cmdi].cmdsettings;
    }

    return RETURN_SUCCESS;
}
#endif

/**
 * @brief Print semaphore status for an image
 *
 * Prints write/read PIDs and current values for
 * every semaphore attached to the image, plus
 * the semlog value.
 *
 * @param IDname  Image name to query
 * @return Image ID on success
 */
imageID COREMOD_MEMORY_image_seminfo(
    const char *IDname)
{
    IMGID img = imgid_make_from_name(IDname);
    resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg);
    imageID ID = img.ID;
    if(ID == -1)
    {
        PRINT_WARNING("image \"%s\" not found", IDname);
        return -1;
    }

    printf("  cnt0 = %ld \n", dcimg[ID].md->cnt0);
    printf("  cnt1 = %ld \n", dcimg[ID].md->cnt1);
    printf("  NB SEMAPHORES = %3d \n", dcimg[ID].md[0].sem);
    printf(" semWritePID at %p\n", (void *) dcimg[ID].semWritePID);
    printf(" semReadPID  at %p\n", (void *) dcimg[ID].semReadPID);
    printf("----------------------------------\n");
    printf(" sem    value   writePID   readPID\n");
    printf("----------------------------------\n");

    for(int s = 0; s < dcimg[ID].md[0].sem; s++)
    {
        int semval;

        semval = ImageStreamIO_semvalue(dcimg + ID, s);

        printf("  %2d   %6d   %8d  %8d\n",
               s, semval, (int) dcimg[ID].semWritePID[s], (int) dcimg[ID].semReadPID[s]);
    }
    printf("----------------------------------\n");
    int semval;
    sem_getvalue(dcimg[ID].semlog, &semval);
    printf(" semlog = %3d\n", semval);
    printf("----------------------------------\n");

    return ID;
}

/**
 * @brief Post a semaphore by image name
 *
 * Resolves image name (loading from shared memory
 * if needed), then posts the specified semaphore.
 *
 * @param IDname  Image name
 * @param index   Semaphore index to post
 * @return Image ID
 */
imageID COREMOD_MEMORY_image_set_sempost(
    const char *IDname,
    long       index)
{
    IMGID img = imgid_make_from_name(IDname);
    resolveIMGID(&img, ERRMODE_NULL, dcimg, dcnimg);
    imageID ID = img.ID;
    if(ID == -1)
    {
        ID = read_sharedmem_image(IDname, dcimg, dcnimg);
    }
    if(ID == -1)
    {
        PRINT_WARNING("image \"%s\" not found", IDname);
        return -1;
    }

    ImageStreamIO_sempost(&dcimg[ID], index);

    return ID;
}

/**
 * @brief Post a semaphore by image slot ID
 *
 * @param ID     Image slot index
 * @param index  Semaphore index to post
 * @return Image ID
 */
imageID COREMOD_MEMORY_image_set_sempost_byID(
    imageID ID,
    long    index)
{
    if(ID < 0 || ID >= dcnimg)
    {
        return -1;
    }
    ImageStreamIO_sempost(&dcimg[ID], index);

    return ID;
}

/**
 * @brief Post semaphore exclusively by ID
 *
 * @param ID     Image slot index
 * @param index  Semaphore index
 * @return Image ID
 */
imageID COREMOD_MEMORY_image_set_sempost_excl_byID(
    imageID ID,
    long    index)
{
    if(ID < 0 || ID >= dcnimg)
    {
        return -1;
    }
    ImageStreamIO_sempost_excl(&dcimg[ID], index);

    return ID;
}

/**
 * @brief Post semaphore in timed loop
 *
 * Continuously posts semaphore at specified
 * interval until cancelled. Used for rate-
 * limited triggering of consumer loops.
 *
 * @param IDname  Image name
 * @param index   Semaphore index
 * @param dtus    Interval in microseconds
 * @return Image ID
 */
imageID
COREMOD_MEMORY_image_set_sempost_loop(
    const char *IDname, long index,
    long       dtus)
{
    IMGID img = imgid_make_from_name(IDname);
    resolveIMGID(&img, ERRMODE_NULL, dcimg, dcnimg);
    imageID ID = img.ID;
    if(ID == -1)
    {
        ID = read_sharedmem_image(IDname, dcimg, dcnimg);
    }
    if(ID == -1)
    {
        PRINT_WARNING("image \"%s\" not found", IDname);
        return -1;
    }

    ImageStreamIO_sempost_loop(&dcimg[ID], index, dtus);

    return ID;
}

/**
 * @brief Wait on a semaphore by image name
 *
 * Blocks until the specified semaphore is posted.
 *
 * @param IDname  Image name
 * @param index   Semaphore index
 * @return Image ID
 */
imageID COREMOD_MEMORY_image_set_semwait(
    const char *IDname,
    long       index)
{
    IMGID img = imgid_make_from_name(IDname);
    resolveIMGID(&img, ERRMODE_NULL, dcimg, dcnimg);
    imageID ID = img.ID;
    if(ID == -1)
    {
        ID = read_sharedmem_image(IDname, dcimg, dcnimg);
    }
    if(ID == -1)
    {
        PRINT_WARNING("image \"%s\" not found", IDname);
        return -1;
    }

    ImageStreamIO_semwait(&dcimg[ID], index);

    return ID;
}

/**
 * @brief Thread func: wait on sem0, cancel peers
 *
 * Used by semwait_OR_IDarray to implement OR-wait.
 * Each thread waits on one stream's sem0; the first
 * to wake cancels all sibling threads.
 *
 * @param ID  Image slot index (cast from void*)
 * @return NULL (exits via pthread_exit)
 */
void *waitforsemID(void *ID)
{
    pthread_t tid;
    int       t;
    //    int semval;

    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
    tid = pthread_self();

    //    semval = ImageStreamIO_semvalue(dcimg+(long) ID, ?sem_index);
    //    printf("tid %u waiting for sem ID %ld   sem = %d   (%s)\n", (unsigned int) tid, (long) ID, semval, dcimg[(long) ID].name);
    //    fflush(stdout);
    ImageStreamIO_semwait(dcimg + (imageID) ID, 0);
    //    printf("tid %u sem ID %ld done\n", (unsigned int) tid, (long) ID);
    //    fflush(stdout);

    for(t = 0; t < NB_thrarray_semwait; t++)
    {
        if(tid != thrarray_semwait[t])
        {
            //            printf("tid %u cancel thread %d tid %u\n", (unsigned int) tid, t, (unsigned int) (thrarray_semwait[t]));
            //           fflush(stdout);
            pthread_cancel(thrarray_semwait[t]);
        }
    }

    pthread_exit(NULL);
}

/**
 * @brief Wait on any of N streams' sem0 (OR logic)
 *
 * Spawns one thread per stream, each waiting on
 * sem0. When any thread returns, it cancels all
 * others — implementing a multi-stream OR-wait.
 *
 * @param IDarray  Array of image IDs to wait on
 * @param NB_ID    Length of IDarray
 * @return RETURN_SUCCESS
 */
errno_t COREMOD_MEMORY_image_set_semwait_OR_IDarray(
    imageID *IDarray,
    long    NB_ID)
{

    //    int semval;

    //   printf("======== ENTER COREMOD_MEMORY_image_set_semwait_OR_IDarray [%ld] =======\n", NB_ID);
    //   fflush(stdout);

    thrarray_semwait    = (pthread_t *) malloc(sizeof(pthread_t) * NB_ID);
    NB_thrarray_semwait = NB_ID;

    for(int t = 0; t < NB_ID; t++)
    {
        //      printf("thread %d create, ID = %ld\n", t, IDarray[t]);
        //      fflush(stdout);
        pthread_create(&thrarray_semwait[t], NULL, waitforsemID, (void *) IDarray[t]);
    }

    for(int t = 0; t < NB_ID; t++)
    {
        //         printf("thread %d tid %u join waiting\n", t, (unsigned int) thrarray_semwait[t]);
        //fflush(stdout);
        pthread_join(thrarray_semwait[t], NULL);
        //    printf("thread %d tid %u joined\n", t, (unsigned int) thrarray_semwait[t]);
    }

    free(thrarray_semwait);
    // printf("======== EXIT COREMOD_MEMORY_image_set_semwait_OR_IDarray =======\n");
    //fflush(stdout);

    return RETURN_SUCCESS;
}

/**
 * @brief Flush semaphores on multiple images
 *
 * Iterates all semaphores on each image, draining
 * their values to zero via sem_trywait.
 *
 * @param IDarray  Array of image IDs
 * @param NB_ID    Length of IDarray
 * @return RETURN_SUCCESS
 */
errno_t COREMOD_MEMORY_image_set_semflush_IDarray(
    imageID *IDarray,
    long    NB_ID)
{

    int  semval;


    list_image_ID();
    for(long i = 0; i < NB_ID; i++)
    {
        for(int s = 0; s < dcimg[IDarray[i]].md[0].sem; s++)
        {
            semval = ImageStreamIO_semvalue(dcimg + IDarray[i], s);
            printf("sem %d/%d of %s [%ld] = %d\n",
                   s, dcimg[IDarray[i]].md[0].sem, dcimg[IDarray[i]].name, IDarray[i], semval);
            fflush(stdout);
            for(long cnt = 0; cnt < semval; cnt++)
            {
                ImageStreamIO_semtrywait(dcimg + IDarray[i], s);
            }
        }
    }

    return RETURN_SUCCESS;
}

/**
 * @brief Flush a single image's semaphore
 *
 * If index < 0, flushes all semaphores on the
 * image. Resolves from shared memory if needed.
 *
 * @param IDname  Image name
 * @param index   Semaphore index (or -1 for all)
 * @return Image ID
 */
imageID COREMOD_MEMORY_image_set_semflush(
    const char *IDname,
    long       index)
{
    IMGID img = imgid_make_from_name(IDname);
    resolveIMGID(&img, ERRMODE_NULL, dcimg, dcnimg);
    imageID ID = img.ID;
    if(ID == -1)
    {
        ID = read_sharedmem_image(IDname, dcimg, dcnimg);
    }
    if(ID == -1)
    {
        PRINT_WARNING("image \"%s\" not found", IDname);
        return -1;
    }

    ImageStreamIO_semflush(&dcimg[ID], index);

    return ID;
}
