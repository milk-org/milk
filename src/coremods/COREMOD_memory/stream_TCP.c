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
    int semtrig, int testmode);

imageID COREMOD_MEMORY_image_NETWORKtransmit(
    const char *IDname,
    const char *IPaddr,
    int port, int mode,
    int RT_priority);

imageID COREMOD_MEMORY_image_NETWORKreceive(
    int port, int mode,
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

static FPS_APP_INFO FPS_app_info_tsem = {
    .fps_name    = "testfuncsem",
    .cmdkey      = "testfuncsem",
    .description =
        "test semaphore loop"
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

static CLICMDDATA CLIcmddata_tsem = {
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

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "imnetwtransmit",
    .cmdkey      = "imnetwtransmit",
    .description =
        "transmit image over network"
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

static FPS_CLI_BINDING my_bindings[] = {
    FPS_PARAMS(FPS_X_BINDING)
};

static const int __attribute__((unused)) nb_bindings =
    sizeof(my_bindings) /
    sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = {
    FPS_PARAMS(FPS_X_FARG)
};

static CLICMDDATA CLIcmddata = {
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

static FPS_APP_INFO FPS_app_info_rx = {
    .fps_name    = "imnetwreceive",
    .cmdkey      = "imnetwreceive",
    .description =
        "receive image(s) over network"
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

static CLICMDDATA CLIcmddata_rx = {
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

static FPS_CLI_BINDING bindings_tsem[] = {
    FPS_PARAMS_TSEM(FPS_X_BINDING)
};
static const int nb_bindings_tsem =
    sizeof(bindings_tsem) /
    sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF farg_tsem[] = {
    FPS_PARAMS_TSEM(FPS_X_FARG)
};

static FPS_CLI_BINDING bindings_rx[] = {
    FPS_PARAMS_RX(FPS_X_BINDING)
};
static const int nb_bindings_rx =
    sizeof(bindings_rx) /
    sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF farg_rx[] = {
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
errno_t COREMOD_MEMORY_testfunction_semaphore(const char *IDname,
        int         semtrig,
        int         testmode)
{
    imageID ID;
    int     semval;
    int     rv;
    long    loopcnt = 0;

    ID = image_ID(IDname, dcimg, dcnimg);
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
    const char *IDname, const char *IPaddr, int port, int mode, int RT_priority)
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

    struct timespec ts;
    long            scnt;
    int             semval;
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

    ID = image_ID(IDname, dcimg, dcnimg);
    img_p = &dcimg[ID];

    if((fds_client = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP)) < 0)
    {
        printf("ERROR creating socket\n");
        exit(0);
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
            perror("Error  connect() failed ");
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
            printf("ERROR: WRONG DATA TYPE\n");
            snprintf(errmsg,
                     200,
                     "WRONG DATA TYPE data type = %d\n",
                     img_p->md->datatype);
            printf("data type = %d\n", img_p->md->datatype);
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

    if((img_p->md->sem == 0) || (mode == 1))
    {
        processinfo_WriteMessage(processinfo, "sync using counter");
        UseSem = 0;
    }
    else
    {
        char msgstring[200];
        snprintf(msgstring, 200, "sync using semaphore %d", semtrig);
        processinfo_WriteMessage(processinfo, msgstring);
    }

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
            if(clock_gettime(CLOCK_MILK, &ts) == -1)
            {
                perror("clock_gettime");
                exit(EXIT_FAILURE);
            }
            ts.tv_sec += 2;

            semr = ImageStreamIO_semtimedwait(img_p, semtrig, &ts);

            if(iter == 0)
            {
                processinfo_WriteMessage(processinfo, "Driving sem to 0");
                printf("Driving semaphore to zero ... ");
                fflush(stdout);
                semval = ImageStreamIO_semvalue(img_p, semtrig);
                int semvalcnt = semval;
                for(scnt = 0; scnt < semvalcnt; scnt++)
                {
                    semval = ImageStreamIO_semvalue(img_p, semtrig);
                    printf("sem = %d\n", semval);
                    fflush(stdout);
                    ImageStreamIO_semtrywait(img_p, semtrig);
                }
                printf("done\n");
                fflush(stdout);

                semval = ImageStreamIO_semvalue(img_p, semtrig);
                printf("-> sem = %d\n", semval);
                fflush(stdout);

                iter++;
            }
        }

        processinfo_exec_start(processinfo);
        if(processinfo_compute_status(processinfo) == 1)
        {

            if(semr == 0)
            {
                frame_md.magic = FRAME_MD_MAGIC;
                frame_md.cnt0 = img_p->md->cnt0;
                frame_md.cnt1 = img_p->md->cnt1;

                slice = img_p->md->cnt1;
                if(slice > oldslice + 1)
                {
                    slice = oldslice + 1;
                }
                if(NBslices > 1)
                    if(oldslice == NBslices - 1)
                    {
                        slice = 0;
                    }
                if(slice > NBslices - 1)
                {
                    slice = 0;
                }

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
                    perror("socket send error ");
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

        if((dcsigINT == 1) || (dcsigTERM == 1) ||
                (dcsigABRT == 1) || (dcsigBUS == 1) ||
                (dcsigSEGV == 1) || (dcsigHUP == 1) ||
                (dcsigPIPE == 1))
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

/** continuously receives 2D image through TCP link
 * mode = 1, force counter to be used for synchronization, ignore semaphores if they exist
 */

imageID COREMOD_MEMORY_image_NETWORKreceive(int                         port,
        __attribute__((unused)) int mode,
        int RT_priority)
{
    struct sockaddr_in sock_server;
    struct sockaddr_in sock_client;
    int                fds_server;
    int                fds_client;
    socklen_t          slen_client;

    int  flag = 1;
    long recvsize;
    int  result;
    long totsize    = 0;
    int  MAXPENDING = 5;

    IMAGE_METADATA *imgmd;
    imageID         ID;
    IMAGE          *img_p;
    long            framesize;
    uint32_t        xsize;
    uint32_t        ysize;
    char           *ptr0; // source
    long            NBslices;
    int             socketOpen = 1; // 0 if socket is closed
    int             semval __attribute__((unused));
    int             semnb __attribute__((unused));
    int             OKim;
    int             axis;

    imgmd = (IMAGE_METADATA *) malloc(sizeof(IMAGE_METADATA));

    TCP_BUFFER_METADATA *frame_md_p;
    long                 framesize1;    // pixel data + metadata
    long                 framesizefull; // pixel data + metadata + kw
    char                 *buff;          // buffer

    //size_t flushsize;
    char *socket_flush_buff;

    struct sched_param schedpar;

    PROCESSINFO *processinfo;
    if(dcprocinfo == 1)
    {
        // CREATE PROCESSINFO ENTRY
        // see processtools.c in module CommandLineInterface for details
        //
        char pinfoname[200];
        snprintf(pinfoname, 200, "ntw-receive-%d", port);
        processinfo           = processinfo_shm_create(pinfoname, 0);
        processinfo->loopstat = 0; // loop initialization

        strcpy(processinfo->source_FUNCTION, __FUNCTION__);
        strcpy(processinfo->source_FILE, __FILE__);
        processinfo->source_LINE = __LINE__;

        char msgstring[200];
        snprintf(msgstring, 200, "Waiting for input stream");
        processinfo_WriteMessage(processinfo, msgstring);
    }

    // CATCH SIGNALS

    if(
        sigaction(SIGTERM, &dcsigact, NULL) == -1 ||
        sigaction(SIGINT, &dcsigact, NULL) == -1 ||
        sigaction(SIGABRT, &dcsigact, NULL) == -1 ||
        sigaction(SIGBUS, &dcsigact, NULL) == -1 ||
        sigaction(SIGSEGV, &dcsigact, NULL) == -1 ||
        sigaction(SIGHUP, &dcsigact, NULL) == -1 ||
        sigaction(SIGPIPE, &dcsigact, NULL) == -1
    )
    {
        printf("\nCan't catch a requested signal (TERM, INT, ABRT, BUS, SEGV, HUP, PIPE)\n");
    }

    schedpar.sched_priority = RT_priority;
    if(seteuid(dceuid) != 0)  //This goes up to maximum privileges
    {
        PRINT_ERROR("seteuid error");
    }
    sched_setscheduler(0,
                       SCHED_FIFO,
                       &schedpar); //other option is SCHED_RR, might be faster
    if(seteuid(dcruid) != 0)    //Go back to normal privileges
    {
        PRINT_ERROR("seteuid error");
    }

    // create TCP socket
    if((fds_server = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP)) == -1)
    {
        printf("ERROR creating socket\n");
        if(dcprocinfo == 1)
        {
            processinfo->loopstat = 4;
            processinfo_WriteMessage(processinfo, "ERROR creating socket");
        }
        exit(0);
    }

    memset((char *) &sock_server, 0, sizeof(sock_server));

    result = setsockopt(fds_server,     /* socket affected */
                        IPPROTO_TCP,    /* set option at TCP level */
                        TCP_NODELAY,    /* name of option */
                        (char *) &flag, /* the cast is historical cruft */
                        sizeof(flag));   /* length of option value */
    result -= setsockopt(fds_server, SOL_SOCKET, SO_REUSEADDR, (char *) & flag,
                         sizeof(flag));
    result -= setsockopt(fds_server, SOL_SOCKET, SO_REUSEPORT, (char *) & flag,
                         sizeof(flag));
    if(result < 0)
    {
        printf("ERROR setsockopt\n");
        if(dcprocinfo == 1)
        {
            processinfo->loopstat = 4;
            processinfo_WriteMessage(processinfo, "ERROR socketopt");
        }
        exit(0);
    }

    sock_server.sin_family      = AF_INET;
    sock_server.sin_port        = htons(port);
    sock_server.sin_addr.s_addr = htonl(INADDR_ANY);

    //bind socket to port
    if(bind(fds_server,
            (struct sockaddr *) &sock_server,
            sizeof(sock_server)) == -1)
    {
        char msgstring[200];

        snprintf(msgstring, 200, "ERROR binding socket, port %d", port);
        printf("%s\n", msgstring);

        if(dcprocinfo == 1)
        {
            processinfo->loopstat = 4;
            processinfo_WriteMessage(processinfo, msgstring);
        }
        exit(0);
    }

    if(listen(fds_server, MAXPENDING) < 0)
    {
        char msgstring[200];

        snprintf(msgstring, 200, "ERROR listen socket");
        printf("%s\n", msgstring);

        if(dcprocinfo == 1)
        {
            processinfo->loopstat = 4;
            processinfo_WriteMessage(processinfo, msgstring);
        }

        exit(0);
    }

    //    cnt = 0;

    /* Set the size of the in-out parameter */
    slen_client = sizeof(sock_client);

    /* Wait for a client to connect */
    if((fds_client = accept(fds_server,
                            (struct sockaddr *) &sock_client,
                            &slen_client)) == -1)
    {
        char msgstring[200];

        snprintf(msgstring, 200, "ERROR accept socket");
        printf("%s\n", msgstring);

        if(dcprocinfo == 1)
        {
            processinfo->loopstat = 4;
            processinfo_WriteMessage(processinfo, msgstring);
        }

        exit(0);
    }

    printf("Client connected\n");
    fflush(stdout);

    // listen for image metadata
    if((recvsize =
                recv(fds_client, imgmd, sizeof(IMAGE_METADATA), MSG_WAITALL)) < 0)
    {
        char msgstring[200];

        snprintf(msgstring, 200, "ERROR receiving image metadata");
        printf("%s\n", msgstring);

        if(dcprocinfo == 1)
        {
            processinfo->loopstat = 4;
            processinfo_WriteMessage(processinfo, msgstring);
        }

        exit(0);
    }

    if(dcprocinfo == 1)
    {
        char msgstring[200];
        snprintf(msgstring, 200, "Receiving stream %s", imgmd->name);
        processinfo_WriteMessage(processinfo, msgstring);
    }

    // is image already in memory ?
    OKim = 0;

    ID = image_ID(imgmd->name, dcimg, dcnimg);
    printf("ID: %ld\n", ID);

    if(ID == -1)
    {
        // is it in shared memory ?
        ID = read_sharedmem_image(imgmd->name, dcimg, dcnimg);
        printf("ID: %ld\n", ID);
    }

    // img_p = &dcimg[ID]; // Of course that doesn't fucking work if ID is -1.

    list_image_ID();

    if(ID == -1)
    {
        OKim = 0;
    }
    else
    {
        img_p = &dcimg[ID];
        OKim = 1;
        if(imgmd->naxis != img_p->md->naxis)
        {
            OKim = 0;
        }
        if(OKim == 1)
        {
            for(axis = 0; axis < imgmd->naxis; axis++)
                if(imgmd->size[axis] != img_p->md->size[axis])
                {
                    OKim = 0;
                }
        }
        if(imgmd->datatype != img_p->md->datatype)
        {
            OKim = 0;
        }

        if(TCPTRANSFERKW == 1 && imgmd->NBkw > img_p->md->NBkw)
        {
            OKim = 0;
        }

        if(OKim == 0)
        {
            // This actually nukes img_p, but leaves imgmd unscathed.
            delete_image_ID(imgmd->name, DELETE_IMAGE_ERRMODE_WARNING);
            ID = -1;
        }
    }

    if(OKim == 0)
    {
        printf("IMAGE %s HAS TO BE CREATED\n", imgmd->name);
        fflush(stdout);
        {
            IMGID imgrcv =
                imgid_make_from_name(
                    imgmd->name);
            imgrcv.mdt->naxis =
                imgmd->naxis;
            for(int a = 0;
                a < imgmd->naxis; a++)
            {
                imgrcv.mdt->size[a] =
                    imgmd->size[a];
            }
            imgrcv.mdt->datatype =
                imgmd->datatype;
            imgrcv.mdt->shared =
                imgmd->shared;
            imgrcv.mdt->NBkw =
                imgmd->NBkw;
            imgrcv.im =
                (IMAGE *) calloc(
                    1, sizeof(IMAGE));
            imgid_mkimage(&imgrcv);
            ID = imgrcv.ID;
        }
        printf("Created image stream %s - shared = %d\n",
               imgmd->name,
               imgmd->shared);
        printf("Size = %d,%d\n", imgmd->size[0], imgmd->size[1]);
        // OKim is now OK. Re-point img_p
        img_p = &dcimg[ID];
    }
    else
    {
        printf("REUSING EXISTING IMAGE %s\n", imgmd->name);
    }

    xsize    = img_p->md->size[0];
    ysize    = img_p->md->size[1];
    NBslices = 1;
    if(img_p->md->naxis > 2)
        if(img_p->md->size[2] > 1)
        {
            NBslices = img_p->md->size[2];
        }

    char typestring[8];

    if(ImageStreamIO_checktype(img_p->md->datatype, 0) == -1)
    {
        printf("ERROR: WRONG DATA TYPE\n");
        snprintf(typestring, 8, "%s", "ERR");
        exit(0);
    }
    framesize = ImageStreamIO_typesize(img_p->md->datatype) * xsize * ysize;
    printf("image frame size = %ld\n", framesize);

    snprintf(typestring, 8, "%s", ImageStreamIO_typename(img_p->md->datatype));

    ptr0 = (char *) img_p->array.raw;

    if(dcprocinfo == 1)
    {
        char msgstring[200];
        snprintf(msgstring,
                 200,
                 "<- %s [%d x %d x %ld] %s",
                 imgmd->name,
                 (int) xsize,
                 (int) ysize,
                 NBslices,
                 typestring);
        snprintf(processinfo->description,
                 200,
                 "%s %dx%dx%ld %s",
                 imgmd->name,
                 (int) xsize,
                 (int) ysize,
                 NBslices,
                 typestring);
        processinfo_WriteMessage(processinfo, msgstring);
    }

    // this line is not needed, as frame_md is declared below
    // frame_md = (TCP_BUFFER_METADATA*) malloc(sizeof(TCP_BUFFER_METADATA));

    framesize1 = framesize + sizeof(TCP_BUFFER_METADATA);
    if(TCPTRANSFERKW == 0)
    {
        framesizefull = framesize1;
    }
    else
    {
        // Warning img_p->md->NBkw may be > imgmd->NBkw.
        // Use the correct one.
        framesizefull = framesize1 + imgmd->NBkw * sizeof(IMAGE_KEYWORD);
    }
    printf("image frame size full (img + md + kw) = %ld\n", framesizefull);

    buff = (char *) malloc(sizeof(char) * framesizefull);

    frame_md_p = (TCP_BUFFER_METADATA *)(buff + framesize);

    if(dcprocinfo == 1)
    {
        processinfo->loopstat =
            1; //notify processinfo that we are entering loop
    }

    socketOpen   = 1;
    long loopcnt = 0;
    int  loopOK  = 1;

    // In-loop counter watch and debug prompts
    long frameincr;
    long minputcnt        = 0;
    long moutputcnt       = 0;
    long monitorinterval  = 10000;
    long monitorindex     = 0;
    long monitorloopindex = 0;
    long cnt0previous     = 0;

    {
        // Finally, just before we start, flush the TCP receive buffer. BUT we need to flush an integer number of frames, that's important,
        // or we end up losing sync.
        // This entire thing is kinda useless... it's legacy dating from ImageStreamIO version mismatches where headers could have different sizes
        // at either end...
        socket_flush_buff = (char *) malloc(framesizefull);
        long recv_bytes = framesizefull;
        while(recv_bytes == framesizefull)
        {
            recv_bytes = recv(fds_client,
                socket_flush_buff,
                framesizefull,
                MSG_DONTWAIT);
            printf("TCP recv buffer flush. %ld stray bytes.\n", recv_bytes);
        }
        if(recv_bytes >
                0)    // Will be -1 if we got 0 bytes at the last iteration above
        {
            recv_bytes = recv(fds_client, socket_flush_buff, framesizefull - recv_bytes,
                              MSG_WAITALL);
            printf("Buffer flush finalize. %ld extra bytes.\n", recv_bytes);
        }
    }

    while(loopOK == 1)
    {
        if(dcprocinfo == 1)
        {
            while(processinfo->CTRLval == 1)  // pause
            {
                usleep(50);
            }

            if(processinfo->CTRLval == 2)  // single iteration
            {
                processinfo->CTRLval = 1;
            }

            if(processinfo->CTRLval == 3)  // exit loop
            {
                loopOK = 0;
            }
        }

        if((recvsize = recv(fds_client, buff, framesizefull, MSG_WAITALL)) < 0)
        {
            printf("ERROR recv()\n");
            socketOpen = 0;
        }

        if((dcprocinfo == 1) && (processinfo->MeasureTiming == 1))
        {
            processinfo_exec_start(processinfo);
        }

        if(recvsize != 0)
        {
            totsize += recvsize;
        }
        else
        {
            socketOpen = 0;
        }

        if(socketOpen == 1)
        {
            frame_md_p = (TCP_BUFFER_METADATA *)(buff + framesize);
            if(frame_md_p->magic != FRAME_MD_MAGIC)
            {
                printf("Bad magic! Looping fast.\n");
                continue;
            }

            img_p->md->write = 1;
            img_p->md->cnt1 = frame_md_p->cnt1;

            // copy pixel data
            if(NBslices > 1)
            {
                __builtin_memcpy(ptr0 + framesize * frame_md_p->cnt1, buff, framesize);
            }
            else
            {
                __builtin_memcpy(ptr0, buff, framesize);
            }

            if(TCPTRANSFERKW == 1)
            {
                // copy kw
                __builtin_memcpy(img_p->kw,
                       (IMAGE_KEYWORD *)(buff + framesize1),
                       img_p->md->NBkw * sizeof(IMAGE_KEYWORD));
            }

            frameincr = (long) frame_md_p->cnt0 - cnt0previous;
            if(frameincr > 1)
            {
                printf("Skipped %ld frame(s) at index %ld %ld\n",
                       frameincr - 1,
                       (long)(frame_md_p->cnt0),
                       (long)(frame_md_p->cnt1));
            }

            cnt0previous = frame_md_p->cnt0;

            if(monitorindex == monitorinterval)
            {
                printf(
                    "[%5ld]  input %20ld (+ %8ld) output %20ld (+ "
                    "%8ld)\n",
                    monitorloopindex,
                    frame_md_p->cnt0,
                    frame_md_p->cnt0 - minputcnt,
                    img_p->md->cnt0,
                    img_p->md->cnt0 - moutputcnt);

                minputcnt  = frame_md_p->cnt0;
                moutputcnt = img_p->md->cnt0;

                monitorloopindex++;
                monitorindex = 0;
            }

            monitorindex++;

            // Carry cnt0 to streamproctrace
            img_p->streamproctrace[0].cnt0 = img_p->md->cnt0;
            processinfo_update_output_stream(processinfo, &dcimg[ID], NULL);
        }

        if(socketOpen == 0)
        {
            loopOK = 0;
        }

        if((dcprocinfo == 1) && (processinfo->MeasureTiming == 1))
        {
            processinfo_exec_end(processinfo);
        }

        // process signals
        if(dcsigTERM || dcsigINT || dcsigABRT || dcsigBUS ||
                dcsigSEGV || dcsigHUP || dcsigPIPE)
        {
            loopOK = 0;
            if(dcprocinfo == 1)
            {
                if(dcsigTERM)
                {
                    processinfo_SIGexit(processinfo, SIGTERM);
                }
                else if(dcsigINT)
                {
                    processinfo_SIGexit(processinfo, SIGINT);
                }
                else if(dcsigABRT)
                {
                    processinfo_SIGexit(processinfo, SIGABRT);
                }
                else if(dcsigBUS)
                {
                    processinfo_SIGexit(processinfo, SIGBUS);
                }
                else if(dcsigSEGV)
                {
                    processinfo_SIGexit(processinfo, SIGSEGV);
                }
                else if(dcsigHUP)
                {
                    processinfo_SIGexit(processinfo, SIGHUP);
                }
                else if(dcsigPIPE)
                {
                    processinfo_SIGexit(processinfo, SIGPIPE);
                }
            }
        }

        loopcnt++;
        if(dcprocinfo == 1)
        {
            processinfo->loopcnt = loopcnt;
        }
    }

    if(dcprocinfo == 1)
    {
        processinfo_cleanExit(processinfo);
    }

    free(socket_flush_buff);
    free(buff);

    close(fds_client);

    printf("port %d closed\n", port);
    fflush(stdout);

    free(imgmd);

    return ID;
}

