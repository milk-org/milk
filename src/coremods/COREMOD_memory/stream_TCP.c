/**
 * @file    stream_TCP.c
 * @brief   TCP stream transfer
 *
 * Uses FPS V2 framework.
 */
#include "ImageStreamIO/ImageStruct.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sched.h>

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "create_image.h"
#include "delete_image.h"
#include "image_ID.h"
#include "list_image.h"
#include "read_shmim.h"
#include "stream_sem.h"
#include "processinfo_setup.h"
#include "stream_net_common.h"

// set to 1 if transfering keywords
static int TCPTRANSFERKW = 1;

typedef struct
{
    long magic;
    long cnt0;
    long cnt1;
} TCP_BUFFER_METADATA;

long FRAME_MD_MAGIC = 0x12341234ff;

/* forward decls */
errno_t COREMOD_MEMORY_testfunction_semaphore(
    const char *IDname,
    int        semtrig,
    int        testmode);

/**
 * @brief Transmit a stream over TCP.
 *
 * Sends frames from a shared memory stream to
 * a remote host.
 */
imageID COREMOD_MEMORY_image_NETWORKtransmit(
    const char *IDname,
    const char *IPaddr,
    int        port,
    int        mode,
    int        RT_priority);

/**
 * @brief Receive a stream over TCP.
 *
 * Listens for frames from a remote sender and
 * writes them to a local shared memory stream.
 */
imageID COREMOD_MEMORY_image_NETWORKreceive(
    int port,
    int mode,
    int RT_priority);


/* ================================================================
 *  PARAMS
 * ============================================================= */

static char p_imname[
     FUNCTION_PARAMETER_STRMAXLEN]
    = "im1";
static char p_ipaddr[
     FUNCTION_PARAMETER_STRMAXLEN]
    = "127.0.0.1";
static long long p_port = 8888;
static long long p_mode = 0;
static long long p_rtprio = 80;
static long long p_semtrig_tcp = 1;
static long long p_testmode = 0;


/* ================================================================
 *  CMD 1: testfuncsem (3 args)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_tsem =
{
    .fps_name    = "testfuncsem",
    .cmdkey      = "testfuncsem",
    .description =
    "test semaphore loop",
    .description_long =
    "Transmit or receive image stream data over a TCP network connection. Enables sharing shared memory streams between machines for distributed processing."
};

#define FPS_PARAMS_TSEM(X) \
    X(".imname", p_imname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "image name") \
    X(".semindex", &p_semtrig_tcp, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "sem index") \
    X(".testmode", &p_testmode, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "test mode")

static CLICMDDATA CLIcmddata_tsem =
{
    "", "", CLICMD_FIELDS_NOPARAM
};
FPS_CMDSETTINGS_INIT(tsem, CLIcmddata_tsem, FPS_app_info_tsem)

static errno_t __attribute__((unused)) compute_tsem()
{
    COREMOD_MEMORY_testfunction_semaphore(
        p_imname, p_semtrig_tcp,
        p_testmode);
    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 2: imnetwtransmit (5 args, primary)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info =
{
    .fps_name    = "imnetwtransmit",
    .cmdkey      = "imnetwtransmit",
    .description =
    "transmit image over network",
    .description_long =
    "Transmit or receive image stream data over a TCP network connection. Enables sharing shared memory streams between machines for distributed processing."
};

#define FPS_PARAMS(X) \
    X(".imname", p_imname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "image name") \
    X(".ipaddr", p_ipaddr, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "IP address") \
    X(".port", &p_port, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "port") \
    X(".mode", &p_mode, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "sync mode") \
    X(".rtprio", &p_rtprio, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "RT priority")

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

FPS_CMDSETTINGS_INIT(tx, CLIcmddata, FPS_app_info)

static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    COREMOD_MEMORY_image_NETWORKtransmit(
        p_imname, p_ipaddr,
        p_port, p_mode, p_rtprio);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 3: imnetwreceive (3 args)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_rx =
{
    .fps_name    = "imnetwreceive",
    .cmdkey      = "imnetwreceive",
    .description =
    "receive image(s) over network",
    .description_long =
    "Transmit or receive image stream data over a TCP network connection. Enables sharing shared memory streams between machines for distributed processing."
};

#define FPS_PARAMS_RX(X) \
    X(".port", &p_port, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "port") \
    X(".mode", &p_mode, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "mode (1=counter sync)") \
    X(".rtprio", &p_rtprio, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "RT priority")

static CLICMDDATA CLIcmddata_rx =
{
    "", "", CLICMD_FIELDS_NOPARAM
};
FPS_CMDSETTINGS_INIT(rx, CLIcmddata_rx, FPS_app_info_rx)

static errno_t __attribute__((unused)) compute_rx()
{
    COREMOD_MEMORY_image_NETWORKreceive(
        p_port, p_mode, p_rtprio);
    return RETURN_SUCCESS;
}


/* ================================================================
 *  REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)

static FPS_CLI_BINDING bindings_tsem[] =
{
    FPS_PARAMS_TSEM(FPS_X_BINDING)
};
static const int nb_bindings_tsem =
    sizeof(bindings_tsem) /
    sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF farg_tsem[] =
{
    FPS_PARAMS_TSEM(FPS_X_FARG)
};

static FPS_CLI_BINDING bindings_rx[] =
{
    FPS_PARAMS_RX(FPS_X_BINDING)
};
static const int nb_bindings_rx =
    sizeof(bindings_rx) /
    sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF farg_rx[] =
{
    FPS_PARAMS_RX(FPS_X_FARG)
};

static errno_t CLIfunction_tsem(void)
{
    return safe_fps_generic_CLIfunction(
               &FPS_app_info_tsem,
               farg_tsem, &CLIcmddata_tsem,
               bindings_tsem, nb_bindings_tsem,
               compute_tsem);
}

static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
               &FPS_app_info, farg, &CLIcmddata,
               my_bindings, nb_bindings,
               compute_function);
}

static errno_t CLIfunction_rx(void)
{
    return safe_fps_generic_CLIfunction(
               &FPS_app_info_rx,
               farg_rx, &CLIcmddata_rx,
               bindings_rx, nb_bindings_rx,
               compute_rx);
}

errno_t
CLIADDCMD_COREMOD_memory__stream_TCP()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    safe_fps_fill_farg_examples(
        farg_tsem, bindings_tsem,
        nb_bindings_tsem);
    safe_fps_fill_farg_examples(
        farg_rx, bindings_rx,
        nb_bindings_rx);

    {
        int cmdi = RegisterCLIcmd(
                       CLIcmddata_tsem,
                       CLIfunction_tsem);
        CLIcmddata_tsem.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }
    {
        int cmdi = RegisterCLIcmd(
                       CLIcmddata, CLIfunction);
        CLIcmddata.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }
    {
        int cmdi = RegisterCLIcmd(
                       CLIcmddata_rx,
                       CLIfunction_rx);
        CLIcmddata_rx.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }

    return RETURN_SUCCESS;
}
#endif
errno_t COREMOD_MEMORY_testfunction_semaphore(
    const char *IDname,
    int        semtrig,
    int        testmode)
{
    imageID ID;
    int     semval;
    int     rv;
    long    loopcnt = 0;

    {
        IMGID img = imgid_make_from_name(IDname);
        resolveIMGID(
            &img,  ERRMODE_ABORT,
            dcimg, dcnimg);
        ID = img.ID;
    }
    IMAGE *img_p = &dcimg[ID];

    char pinfomsg[200];

    // ===========================
    // Start loop
    // ===========================
    int loopOK = 1;
    while(loopOK == 1)
    {
        printf("\n");
        usleep(500);

        semval = ImageStreamIO_semvalue(img_p, semtrig);
        snprintf(pinfomsg,
                 200,
                 "%ld TEST 0 semtrig %d  ID %ld  %d",
                 loopcnt,
                 semtrig,
                 ID,
                 semval);
        printf("MSG: %s\n", pinfomsg);
        fflush(stdout);

        if(testmode == 0)
        {
            rv = ImageStreamIO_semwait(img_p, semtrig);
        }

        if(testmode == 1)
        {
            rv = ImageStreamIO_semtrywait(img_p, semtrig);
        }

        if(testmode == 2)
        {
            ImageStreamIO_sempost(img_p, semtrig);
            rv = ImageStreamIO_semwait(img_p, semtrig);
        }

        if(rv == -1)
        {
            switch(errno)
            {

            case EINTR:
                printf(
                    "    sem_wait call was interrupted by a signal "
                    "handler\n");
                break;

            case EINVAL:
                printf("    not a valid semaphore\n");
                break;

            case EAGAIN:
                printf(
                    "    The operation could not be performed "
                    "without blocking (i.e., the semaphore "
                    "currently has "
                    "the value zero)\n");
                break;

            default:
                printf("    ERROR: unknown code %d\n", rv);
                break;
            }
        }
        else
        {
            printf("    OK\n");
        }

        semval = ImageStreamIO_semvalue(img_p, semtrig);
        snprintf(pinfomsg,
                 200,
                 "%ld TEST 1 semtrig %d  ID %ld  %d",
                 loopcnt,
                 semtrig,
                 ID,
                 semval);
        printf("MSG: %s\n", pinfomsg);
        fflush(stdout);

        loopcnt++;
    }

    return RETURN_SUCCESS;
}

/** continuously transmits 2D image through TCP link
 * mode = 1, force counter to be used for synchronization, ignore semaphores if they exist
 */

imageID COREMOD_MEMORY_image_NETWORKtransmit(
    const char *IDname,
    const char *IPaddr,
    int        port,
    int        mode,
    int        RT_priority)
{
    imageID            ID;
    IMAGE             *img_p;
    struct sockaddr_in sock_server;
    int                fds_client;
    int                flag = 1;
    int                result;
    unsigned long long cnt  = 0;
    long long          iter = 0;
    long               framesize; // pixel data only
    uint32_t           xsize, ysize;
    char              *ptr0; // source
    char              *ptr1; // source - offset by slice
    int                rs;
    int             semr;
    int             slice, oldslice;
    int             NBslices;

    TCP_BUFFER_METADATA frame_md = {0};
    long                framesize1; // pixel data + metadata
    long  framesizeall; // total frame size : pixel data + metadata + kw
    char *buff;         // transmit buffer

    int semtrig = 6; // TODO - scan for available sem
    // IMPORTANT: do not use semtrig 0
    int UseSem = 1;

    char errmsg[200];

    printf("Transmit stream %s over IP %s port %d\n", IDname, IPaddr, port);
    fflush(stdout);

    DEBUG_TRACEPOINT(" ");

    // ===========================
    // processinfo support
    // ===========================
    PROCESSINFO *processinfo;

    char pinfoname[200];
    snprintf(pinfoname, 200, "ntw-tx-%s", IDname);

    char descr[200];
    snprintf(descr, 200, "%s->%s/%d", IDname, IPaddr, port);

    char pinfomsg[200];
    snprintf(pinfomsg, 200, "setup");

    printf("Setup processinfo ...");
    fflush(stdout);
    processinfo = processinfo_setup(pinfoname,
                                    descr,    // description
                                    pinfomsg, // message on startup
                                    __FUNCTION__,
                                    __FILE__,
                                    __LINE__);
    printf(" done\n");
    fflush(stdout);

    // OPTIONAL SETTINGS
    processinfo->MeasureTiming = 1; // Measure timing
    processinfo->RT_priority =
        RT_priority; // RT_priority, 0-99. Larger number = higher priority. If <0, ignore

    int loopOK = 1;

    {
        IMGID img = imgid_make_from_name(IDname);
        resolveIMGID(
            &img,  ERRMODE_ABORT,
            dcimg, dcnimg);
        ID = img.ID;
    }
    img_p = &dcimg[ID];

    if((fds_client = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP)) < 0)
    {
        PRINT_ERROR("creating socket");
        return -1;
    }

    result = setsockopt(fds_client,     /* socket affected */
                        IPPROTO_TCP,    /* set option at TCP level */
                        TCP_NODELAY,    /* name of option */
                        (char *) &flag, /* the cast is historical cruft */
                        sizeof(int));   /* length of option value */

    if(result < 0)
    {
        processinfo_error(processinfo, "ERROR: setsockopt() failed\n");
        loopOK = 0;
    }

    if(loopOK == 1)
    {
        memset((char *) &sock_server, 0, sizeof(sock_server));
        sock_server.sin_family      = AF_INET;
        sock_server.sin_port        = htons(port);
        sock_server.sin_addr.s_addr = inet_addr(IPaddr);

        if(connect(fds_client,
                   (struct sockaddr *) &sock_server,
                   sizeof(sock_server)) < 0)
        {
            PRINT_ERROR("Error  connect() failed: %s", strerror(errno));
            printf("port = %d\n", port);
            processinfo_error(processinfo, "ERROR: connect() failed\n");
            loopOK = 0;
        }
    }

    if(loopOK == 1)
    {
        if(send(fds_client,
                (void *) img_p->md,
                sizeof(IMAGE_METADATA),
                0) != sizeof(IMAGE_METADATA))
        {
            printf(
                "send() sent a different number of bytes than expected "
                "%ld\n",
                sizeof(IMAGE_METADATA));
            fflush(stdout);
            processinfo_error(processinfo,
                              "send() sent a different number of bytes "
                              "than expected");
            loopOK = 0;
        }
    }

    if(loopOK == 1)
    {
        xsize    = img_p->md->size[0];
        ysize    = img_p->md->size[1];
        NBslices = 1;
        if(img_p->md->naxis > 2)
            if(img_p->md->size[2] > 1)
            {
                NBslices = img_p->md->size[2];
            }
    }

    if(loopOK == 1)
    {
        framesize = ImageStreamIO_typesize(img_p->md->datatype) * xsize * ysize;

        printf("IMAGE FRAME SIZE = %ld\n", framesize);
        fflush(stdout);

        if(-1 == ImageStreamIO_checktype(img_p->md->datatype, 0))
        {
            PRINT_ERROR(
                "wrong data type %d",
                (int) img_p->md->datatype);
            snprintf(errmsg,
                     200,
                     "WRONG DATA TYPE data type = %d\n",
                     img_p->md->datatype);
            processinfo_error(processinfo, errmsg);
            loopOK = 0;
        }
    }

    if(loopOK == 1)
    {
        ptr0 = (char *) img_p->array.raw;
        framesize1 = framesize + sizeof(TCP_BUFFER_METADATA);

        if(TCPTRANSFERKW == 0)
        {
            framesizeall = framesize1;
        }
        else
        {
            framesizeall =
                framesize1 + img_p->md->NBkw * sizeof(IMAGE_KEYWORD);
        }

        buff = (char *) malloc(sizeof(char) * framesizeall);

        printf("transfer buffer size = %ld\n", framesizeall);
        fflush(stdout);

        oldslice = 0;
        //sockOK = 1;
        printf("sem = %d\n", img_p->md->sem);
        fflush(stdout);
    }

    UseSem = stream_net_decide_sync(
                 img_p->md->sem, mode, semtrig,
                 processinfo);

    long frameincr = 0;
    long cnt0previous = 0;

    // ===========================
    // Start loop
    // ===========================
    processinfo_loopstart(
        processinfo); // Notify processinfo that we are entering loop

    while(loopOK == 1)
    {
        loopOK = processinfo_loopstep(processinfo);

        if(UseSem == 0)  // use counter
        {
            while(img_p->md->cnt0 == cnt)  // test if new frame exists
            {
                usleep(5);
            }
            cnt  = img_p->md->cnt0;
            semr = 0;
        }
        else
        {
            semr = stream_net_sem_wait(
                       img_p, semtrig);

            stream_net_sem_drain(
                img_p, semtrig,
                &iter, processinfo);
        }

        processinfo_exec_start(processinfo);
        if(processinfo_compute_status(processinfo) == 1)
        {

            if(semr == 0)
            {
                frame_md.magic = FRAME_MD_MAGIC;
                frame_md.cnt0 = img_p->md->cnt0;
                frame_md.cnt1 = img_p->md->cnt1;

                slice = stream_net_clamp_slice(
                            img_p->md->cnt1,
                            oldslice, NBslices);

                frame_md.cnt1 = slice;

                ptr1 =
                    ptr0 +
                    framesize *
                    slice; //img_p->md->cnt1; // frame that was just written
                __builtin_memcpy(buff, ptr1, framesize);
                *(TCP_BUFFER_METADATA *)(buff + framesize) = frame_md;
                //memcpy(buff + framesize, &frame_md, sizeof());

                if(TCPTRANSFERKW == 1)
                {
                    __builtin_memcpy(buff + framesize1,
                                     (char *) dcimg[ID].kw,
                                     dcimg[ID].md[0].NBkw * sizeof(IMAGE_KEYWORD));
                }

                rs = send(fds_client, buff, framesizeall, 0);

                if(rs != framesizeall)
                {
                    PRINT_ERROR("socket send error: %s", strerror(errno));
                    snprintf(errmsg,
                             200,
                             "ERROR: send() sent a different "
                             "number of bytes (%d) than "
                             "expected %ld  %ld  %ld",
                             rs,
                             (long) framesize,
                             (long) framesizeall,
                             (long) sizeof(TCP_BUFFER_METADATA));
                    printf("%s\n", errmsg);
                    fflush(stdout);
                    processinfo_WriteMessage(processinfo, errmsg);
                    loopOK = 0;
                }
                oldslice = slice;

                frameincr = (long) img_p->md->cnt0 - cnt0previous;
                if(frameincr > 1)
                {
                    printf("Skipped %ld frame(s) at index %ld %ld\n",
                           frameincr - 1,
                           (long)(img_p->md->cnt0),
                           (long)(img_p->md->cnt1));
                }

                cnt0previous = img_p->md->cnt0;
            }
        }
        // process signals, increment loop counter
        processinfo_exec_end(processinfo);

        if(DCSIG_ANY_SET())
        {
            loopOK = 0;
        }
    }
    // ==================================
    // ENDING LOOP
    // ==================================
    processinfo_cleanExit(processinfo);

    free(buff);

    close(fds_client);
    printf("port %d closed\n", port);
    fflush(stdout);

    return ID;
}
