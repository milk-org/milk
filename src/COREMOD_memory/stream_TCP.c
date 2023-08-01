/**
 * @file    stream_TCP.c
 * @brief   TCP stream transfer
 */

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sched.h>

#include "CommandLineInterface/CLIcore.h"
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
    long cnt0;
    long cnt1;
} TCP_BUFFER_METADATA;

// ==========================================
// Forward declaration(s)
// ==========================================

errno_t COREMOD_MEMORY_testfunction_semaphore(const char *IDname,
        int         semtrig,
        int         testmode);

imageID COREMOD_MEMORY_image_NETWORKtransmit(const char *IDname,
        const char *IPaddr,
        int         port,
        int         mode,
        int         RT_priority);

imageID
COREMOD_MEMORY_image_NETWORKreceive(int port, int mode, int RT_priority);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t COREMOD_MEMORY_testfunction_semaphore__cli()
{
    if(0 + CLI_checkarg(1, CLIARG_IMG) + CLI_checkarg(2, CLIARG_INT64) +
            CLI_checkarg(3, CLIARG_INT64) ==
            0)
    {
        COREMOD_MEMORY_testfunction_semaphore(data.cmdargtoken[1].val.string,
                                              data.cmdargtoken[2].val.numl,
                                              data.cmdargtoken[3].val.numl);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

static errno_t COREMOD_MEMORY_image_NETWORKtransmit__cli()
{
    if(0 + CLI_checkarg(1, CLIARG_IMG) + CLI_checkarg(2, CLIARG_STR_NOT_IMG) +
            CLI_checkarg(3, CLIARG_INT64) + CLI_checkarg(4, CLIARG_INT64) +
            CLI_checkarg(5, CLIARG_INT64) ==
            0)
    {
        COREMOD_MEMORY_image_NETWORKtransmit(data.cmdargtoken[1].val.string,
                                             data.cmdargtoken[2].val.string,
                                             data.cmdargtoken[3].val.numl,
                                             data.cmdargtoken[4].val.numl,
                                             data.cmdargtoken[5].val.numl);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

static errno_t COREMOD_MEMORY_image_NETWORKreceive__cli()
{
    if(0 + CLI_checkarg(1, CLIARG_INT64) + CLI_checkarg(2, CLIARG_INT64) +
            CLI_checkarg(3, CLIARG_INT64) ==
            0)
    {
        COREMOD_MEMORY_image_NETWORKreceive(data.cmdargtoken[1].val.numl,
                                            data.cmdargtoken[2].val.numl,
                                            data.cmdargtoken[3].val.numl);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

// ==========================================
// Register CLI command(s)
// ==========================================

errno_t stream__TCP_addCLIcmd()
{
    RegisterCLIcommand("testfuncsem",
                       __FILE__,
                       COREMOD_MEMORY_testfunction_semaphore__cli,
                       "test semaphore loop",
                       "<image> <semindex> <testmode>",
                       "testfuncsem im1 1 0",
                       "int COREMOD_MEMORY_testfunction_semaphore(const char "
                       "*IDname, int semtrig, int testmode)");

    RegisterCLIcommand("imnetwtransmit",
                       __FILE__,
                       COREMOD_MEMORY_image_NETWORKtransmit__cli,
                       "transmit image over network",
                       "<image> <IP addr> <port [long]> <sync mode [int]>",
                       "imnetwtransmit im1 127.0.0.1 0 8888 0",
                       "long COREMOD_MEMORY_image_NETWORKtransmit(const char "
                       "*IDname, const char *IPaddr, int port, int mode)");

    RegisterCLIcommand("imnetwreceive",
                       __FILE__,
                       COREMOD_MEMORY_image_NETWORKreceive__cli,
                       "receive image(s) over network. mode=1 uses counter "
                       "instead of semaphore",
                       "<port [long]> <mode [int]> <RT priority>",
                       "imnetwreceive 8887 0 80",
                       "long COREMOD_MEMORY_image_NETWORKreceive(int port, int "
                       "mode, int RT_priority)");

    return RETURN_SUCCESS;
}

errno_t COREMOD_MEMORY_testfunction_semaphore(const char *IDname,
        int         semtrig,
        int         testmode)
{
    imageID ID;
    int     semval;
    int     rv;
    long    loopcnt = 0;

    ID = image_ID(IDname);
    IMAGE *img_p = &data.image[ID];

    char pinfomsg[200];

    // ===========================
    // Start loop
    // ===========================
    int loopOK = 1;
    while(loopOK == 1)
    {
        printf("\n");
        usleep(500);

        sem_getvalue(img_p->semptr[semtrig], &semval);
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
            rv = sem_wait(img_p->semptr[semtrig]);
        }

        if(testmode == 1)
        {
            rv = sem_trywait(img_p->semptr[semtrig]);
        }

        if(testmode == 2)
        {
            sem_post(img_p->semptr[semtrig]);
            rv = sem_wait(img_p->semptr[semtrig]);
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

        sem_getvalue(img_p->semptr[semtrig], &semval);
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

    TCP_BUFFER_METADATA *frame_md;
    long                 framesize1; // pixel data + metadata
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

    ID = image_ID(IDname);
    img_p = &data.image[ID];

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
        xsize    = img_p->md[0].size[0];
        ysize    = img_p->md[0].size[1];
        NBslices = 1;
        if(img_p->md[0].naxis > 2)
            if(img_p->md[0].size[2] > 1)
            {
                NBslices = img_p->md[0].size[2];
            }
    }

    if(loopOK == 1)
    {
        framesize = ImageStreamIO_typesize(img_p->md[0].datatype) * xsize * ysize;

        printf("IMAGE FRAME SIZE = %ld\n", framesize);
        fflush(stdout);

        if(-1 == ImageStreamIO_checktype(img_p->md[0].datatype, 0))
        {
            printf("ERROR: WRONG DATA TYPE\n");
            snprintf(errmsg,
                     200,
                     "WRONG DATA TYPE data type = %d\n",
                     img_p->md[0].datatype);
            printf("data type = %d\n", img_p->md[0].datatype);
            processinfo_error(processinfo, errmsg);
            loopOK = 0;
        }
    }

    if(loopOK == 1)
    {
        ptr0 = (char *) img_p->array.raw;

        frame_md = (TCP_BUFFER_METADATA *) malloc(sizeof(TCP_BUFFER_METADATA));
        framesize1 = framesize + sizeof(TCP_BUFFER_METADATA);

        if(TCPTRANSFERKW == 0)
        {
            framesizeall = framesize1;
        }
        else
        {
            framesizeall =
                framesize1 + img_p->md[0].NBkw * sizeof(IMAGE_KEYWORD);
        }

        buff = (char *) malloc(sizeof(char) * framesizeall);

        printf("transfer buffer size = %ld\n", framesizeall);
        fflush(stdout);

        oldslice = 0;
        //sockOK = 1;
        printf("sem = %d\n", img_p->md[0].sem);
        fflush(stdout);
    }

    if((img_p->md[0].sem == 0) || (mode == 1))
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
            while(img_p->md[0].cnt0 == cnt)  // test if new frame exists
            {
                usleep(5);
            }
            cnt  = img_p->md[0].cnt0;
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

            semr = sem_timedwait(img_p->semptr[semtrig], &ts);

            if(iter == 0)
            {
                processinfo_WriteMessage(processinfo, "Driving sem to 0");
                printf("Driving semaphore to zero ... ");
                fflush(stdout);
                sem_getvalue(img_p->semptr[semtrig], &semval);
                int semvalcnt = semval;
                for(scnt = 0; scnt < semvalcnt; scnt++)
                {
                    sem_getvalue(img_p->semptr[semtrig], &semval);
                    printf("sem = %d\n", semval);
                    fflush(stdout);
                    sem_trywait(img_p->semptr[semtrig]);
                }
                printf("done\n");
                fflush(stdout);

                sem_getvalue(img_p->semptr[semtrig], &semval);
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
                frame_md[0].cnt0 = img_p->md[0].cnt0;
                frame_md[0].cnt1 = img_p->md[0].cnt1;

                slice = img_p->md[0].cnt1;
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

                frame_md[0].cnt1 = slice;

                ptr1 =
                    ptr0 +
                    framesize *
                    slice; //img_p->md[0].cnt1; // frame that was just written
                memcpy(buff, ptr1, framesize);
                memcpy(buff + framesize, frame_md, sizeof(TCP_BUFFER_METADATA));

                if(TCPTRANSFERKW == 1)
                {
                    memcpy(buff + framesize1,
                           (char *) img_p->kw,
                           img_p->md[0].NBkw * sizeof(IMAGE_KEYWORD));
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
            }
        }
        // process signals, increment loop counter
        processinfo_exec_end(processinfo);

        if((data.signal_INT == 1) || (data.signal_TERM == 1) ||
                (data.signal_ABRT == 1) || (data.signal_BUS == 1) ||
                (data.signal_SEGV == 1) || (data.signal_HUP == 1) ||
                (data.signal_PIPE == 1))
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

    free(frame_md);

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
    int             semval;
    int             semnb;
    int             OKim;
    int             axis;

    imgmd = (IMAGE_METADATA *) malloc(sizeof(IMAGE_METADATA));

    TCP_BUFFER_METADATA *frame_md;
    long                 framesize1;    // pixel data + metadata
    long                 framesizefull; // pixel data + metadata + kw
    char                *buff;          // buffer

    //size_t flushsize;
    char *socket_flush_buff;



    struct sched_param schedpar;

    PROCESSINFO *processinfo;
    if(data.processinfo == 1)
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
        sigaction(SIGTERM, &data.sigact, NULL) == -1 ||
        sigaction(SIGINT, &data.sigact, NULL) == -1 ||
        sigaction(SIGABRT, &data.sigact, NULL) == -1 ||
        sigaction(SIGBUS, &data.sigact, NULL) == -1 ||
        sigaction(SIGSEGV, &data.sigact, NULL) == -1 ||
        sigaction(SIGHUP, &data.sigact, NULL) == -1 ||
        sigaction(SIGPIPE, &data.sigact, NULL) == -1
    )
    {
        printf("\nCan't catch a requested signal (TERM, INT, ABRT, BUS, SEGV, HUP, PIPE)\n");
    }

    schedpar.sched_priority = RT_priority;
    if(seteuid(data.euid) != 0)  //This goes up to maximum privileges
    {
        PRINT_ERROR("seteuid error");
    }
    sched_setscheduler(0,
                       SCHED_FIFO,
                       &schedpar); //other option is SCHED_RR, might be faster
    if(seteuid(data.ruid) != 0)    //Go back to normal privileges
    {
        PRINT_ERROR("seteuid error");
    }

    // create TCP socket
    if((fds_server = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP)) == -1)
    {
        printf("ERROR creating socket\n");
        if(data.processinfo == 1)
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
        if(data.processinfo == 1)
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

        if(data.processinfo == 1)
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

        if(data.processinfo == 1)
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

        if(data.processinfo == 1)
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

        if(data.processinfo == 1)
        {
            processinfo->loopstat = 4;
            processinfo_WriteMessage(processinfo, msgstring);
        }

        exit(0);
    }

    if(data.processinfo == 1)
    {
        char msgstring[200];
        snprintf(msgstring, 200, "Receiving stream %s", imgmd[0].name);
        processinfo_WriteMessage(processinfo, msgstring);
    }

    // is image already in memory ?
    OKim = 0;

    ID = image_ID(imgmd[0].name);
    if(ID == -1)
    {
        // is it in shared memory ?
        ID = read_sharedmem_image(imgmd[0].name);
    }

    img_p = &data.image[ID];

    list_image_ID();

    if(ID == -1)
    {
        OKim = 0;
    }
    else
    {
        OKim = 1;
        if(imgmd[0].naxis != img_p->md[0].naxis)
        {
            OKim = 0;
        }
        if(OKim == 1)
        {
            for(axis = 0; axis < imgmd[0].naxis; axis++)
                if(imgmd[0].size[axis] != img_p->md[0].size[axis])
                {
                    OKim = 0;
                }
        }
        if(imgmd[0].datatype != img_p->md[0].datatype)
        {
            OKim = 0;
        }

        if(OKim == 0)
        {
            delete_image_ID(imgmd[0].name, DELETE_IMAGE_ERRMODE_WARNING);
            ID = -1;
        }
    }

    int nbkw = 0;
    if(TCPTRANSFERKW == 1)
    {
        nbkw = imgmd[0].NBkw;
        if(imgmd[0].NBkw != img_p->md[0].NBkw)
        {
            OKim = 0;
        }
    }

    if(OKim == 0)
    {
        printf("IMAGE %s HAS TO BE CREATED\n", imgmd[0].name);
        create_image_ID(imgmd[0].name,
                        imgmd[0].naxis,
                        imgmd[0].size,
                        imgmd[0].datatype,
                        imgmd[0].shared,
                        nbkw,
                        0,
                        &ID);
        printf("Created image stream %s - shared = %d\n",
               imgmd[0].name,
               imgmd[0].shared);
        printf("Size = %d,%d\n", imgmd[0].size[0], imgmd[0].size[1]);
    }
    else
    {
        printf("REUSING EXISTING IMAGE %s\n", imgmd[0].name);
    }

    xsize    = img_p->md[0].size[0];
    ysize    = img_p->md[0].size[1];
    NBslices = 1;
    if(img_p->md[0].naxis > 2)
        if(img_p->md[0].size[2] > 1)
        {
            NBslices = img_p->md[0].size[2];
        }

    char typestring[8];

    if(ImageStreamIO_checktype(img_p->md[0].datatype, 0) == -1)
    {
        printf("ERROR: WRONG DATA TYPE\n");
        snprintf(typestring, 8, "%s", "ERR");
        exit(0);
    }
    framesize = ImageStreamIO_typesize(img_p->md[0].datatype) * xsize * ysize;
    printf("image frame size = %ld\n", framesize);

    snprintf(typestring, 8, "%s", ImageStreamIO_typename(img_p->md[0].datatype));

    ptr0 = (char *) img_p->array.raw;


    if(data.processinfo == 1)
    {
        char msgstring[200];
        snprintf(msgstring,
                 200,
                 "<- %s [%d x %d x %ld] %s",
                 imgmd[0].name,
                 (int) xsize,
                 (int) ysize,
                 NBslices,
                 typestring);
        snprintf(processinfo->description,
                 200,
                 "%s %dx%dx%ld %s",
                 imgmd[0].name,
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
        framesizefull = framesize1 + nbkw * sizeof(IMAGE_KEYWORD);
    }

    buff = (char *) malloc(sizeof(char) * framesizefull);

    frame_md = (TCP_BUFFER_METADATA *)(buff + framesize);

    if(data.processinfo == 1)
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
            recv_bytes = recv(fds_client, socket_flush_buff, framesizefull, MSG_DONTWAIT);
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
        if(data.processinfo == 1)
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

        if((data.processinfo == 1) && (processinfo->MeasureTiming == 1))
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
            frame_md = (TCP_BUFFER_METADATA *)(buff + framesize);

            img_p->md[0].cnt1 = frame_md[0].cnt1;

            // copy pixel data
            if(NBslices > 1)
            {
                memcpy(ptr0 + framesize * frame_md[0].cnt1, buff, framesize);
            }
            else
            {
                memcpy(ptr0, buff, framesize);
            }

            if(TCPTRANSFERKW == 1)
            {
                // copy kw
                memcpy(img_p->kw,
                       (IMAGE_KEYWORD *)(buff + framesize1),
                       nbkw * sizeof(IMAGE_KEYWORD));
            }

            frameincr = (long) frame_md[0].cnt0 - cnt0previous;
            if(frameincr > 1)
            {
                printf("Skipped %ld frame(s) at index %ld %ld\n",
                       frameincr - 1,
                       (long)(frame_md[0].cnt0),
                       (long)(frame_md[0].cnt1));
            }

            cnt0previous = frame_md[0].cnt0;

            if(monitorindex == monitorinterval)
            {
                printf(
                    "[%5ld]  input %20ld (+ %8ld) output %20ld (+ "
                    "%8ld)\n",
                    monitorloopindex,
                    frame_md[0].cnt0,
                    frame_md[0].cnt0 - minputcnt,
                    img_p->md[0].cnt0,
                    img_p->md[0].cnt0 - moutputcnt);

                minputcnt  = frame_md[0].cnt0;
                moutputcnt = img_p->md[0].cnt0;

                monitorloopindex++;
                monitorindex = 0;
            }

            monitorindex++;

            img_p->md[0].cnt0++;
            for(semnb = 0; semnb < img_p->md[0].sem; semnb++)
            {
                sem_getvalue(img_p->semptr[semnb], &semval);
                if(semval < SEMAPHORE_MAXVAL)
                {
                    sem_post(img_p->semptr[semnb]);
                }
            }

            sem_getvalue(img_p->semlog, &semval);
            if(semval < 2)
            {
                sem_post(img_p->semlog);
            }
        }

        if(socketOpen == 0)
        {
            loopOK = 0;
        }

        if((data.processinfo == 1) && (processinfo->MeasureTiming == 1))
        {
            processinfo_exec_end(processinfo);
        }

        // process signals
        if(data.signal_TERM || data.signal_INT || data.signal_ABRT || data.signal_BUS ||
                data.signal_SEGV || data.signal_HUP || data.signal_PIPE)
        {
            loopOK = 0;
            if(data.processinfo == 1)
            {
                if(data.signal_TERM)
                {
                    processinfo_SIGexit(processinfo, SIGTERM);
                }
                else if(data.signal_INT)
                {
                    processinfo_SIGexit(processinfo, SIGINT);
                }
                else if(data.signal_ABRT)
                {
                    processinfo_SIGexit(processinfo, SIGABRT);
                }
                else if(data.signal_BUS)
                {
                    processinfo_SIGexit(processinfo, SIGBUS);
                }
                else if(data.signal_SEGV)
                {
                    processinfo_SIGexit(processinfo, SIGSEGV);
                }
                else if(data.signal_HUP)
                {
                    processinfo_SIGexit(processinfo, SIGHUP);
                }
                else if(data.signal_PIPE)
                {
                    processinfo_SIGexit(processinfo, SIGPIPE);
                }
            }
        }

        loopcnt++;
        if(data.processinfo == 1)
        {
            processinfo->loopcnt = loopcnt;
        }
    }

    if(data.processinfo == 1)
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
