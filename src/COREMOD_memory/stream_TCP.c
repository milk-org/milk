/**
 * @file    stream_TCP.c
 * @brief   TCP stream transfer
 */


#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sched.h>


#include "CommandLineInterface/CLIcore.h"
#include "image_ID.h"
#include "list_image.h"
#include "create_image.h"
#include "delete_image.h"
#include "read_shmim.h"
#include "stream_sem.h"




typedef struct
{
    long cnt0;
    long cnt1;
} TCP_BUFFER_METADATA;






// ==========================================
// Forward declaration(s)
// ==========================================


errno_t COREMOD_MEMORY_testfunction_semaphore(
    const char *IDname,
    int         semtrig,
    int         testmode
);


imageID COREMOD_MEMORY_image_NETWORKtransmit(
    const char *IDname,
    const char *IPaddr,
    int         port,
    int         mode,
    int         RT_priority
);

imageID COREMOD_MEMORY_image_NETWORKreceive(
    int port,
    int mode,
    int RT_priority
);




// ==========================================
// Command line interface wrapper function(s)
// ==========================================


static errno_t COREMOD_MEMORY_testfunction_semaphore__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_LONG)
            + CLI_checkarg(3, CLIARG_LONG)
            == 0)
    {
        COREMOD_MEMORY_testfunction_semaphore(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.numl,
            data.cmdargtoken[3].val.numl
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}



static errno_t COREMOD_MEMORY_image_NETWORKtransmit__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_STR_NOT_IMG)
            + CLI_checkarg(3, CLIARG_LONG)
            + CLI_checkarg(4, CLIARG_LONG)
            + CLI_checkarg(5, CLIARG_LONG)
            == 0)
    {
        COREMOD_MEMORY_image_NETWORKtransmit(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.string,
            data.cmdargtoken[3].val.numl,
            data.cmdargtoken[4].val.numl,
            data.cmdargtoken[5].val.numl
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}


static errno_t COREMOD_MEMORY_image_NETWORKreceive__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_LONG)
            + CLI_checkarg(2, CLIARG_LONG)
            + CLI_checkarg(3, CLIARG_LONG)
            == 0)
    {
        COREMOD_MEMORY_image_NETWORKreceive(
            data.cmdargtoken[1].val.numl,
            data.cmdargtoken[2].val.numl,
            data.cmdargtoken[3].val.numl
        );
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
    RegisterCLIcommand(
        "testfuncsem",
        __FILE__,
        COREMOD_MEMORY_testfunction_semaphore__cli,
        "test semaphore loop",
        "<image> <semindex> <testmode>",
        "testfuncsem im1 1 0",
        "int COREMOD_MEMORY_testfunction_semaphore(const char *IDname, int semtrig, int testmode)");

    RegisterCLIcommand(
        "imnetwtransmit",
        __FILE__,
        COREMOD_MEMORY_image_NETWORKtransmit__cli,
        "transmit image over network",
        "<image> <IP addr> <port [long]> <sync mode [int]>",
        "imnetwtransmit im1 127.0.0.1 0 8888 0",
        "long COREMOD_MEMORY_image_NETWORKtransmit(const char *IDname, const char *IPaddr, int port, int mode)");

    RegisterCLIcommand(
        "imnetwreceive",
        __FILE__,
        COREMOD_MEMORY_image_NETWORKreceive__cli,
        "receive image(s) over network. mode=1 uses counter instead of semaphore",
        "<port [long]> <mode [int]> <RT priority>",
        "imnetwreceive 8887 0 80",
        "long COREMOD_MEMORY_image_NETWORKreceive(int port, int mode, int RT_priority)");	


    return RETURN_SUCCESS;
}

















errno_t COREMOD_MEMORY_testfunction_semaphore(
    const char *IDname,
    int         semtrig,
    int         testmode
)
{
    imageID ID;
    int     semval;
    int     rv;
    long    loopcnt = 0;

    ID = image_ID(IDname);

    char pinfomsg[200];


    // ===========================
    // Start loop
    // ===========================
    int loopOK = 1;
    while(loopOK == 1)
    {
        printf("\n");
        usleep(500);

        sem_getvalue(data.image[ID].semptr[semtrig], &semval);
        sprintf(pinfomsg, "%ld TEST 0 semtrig %d  ID %ld  %d", loopcnt, semtrig, ID,
                semval);
        printf("MSG: %s\n", pinfomsg);
        fflush(stdout);

        if(testmode == 0)
        {
            rv = sem_wait(data.image[ID].semptr[semtrig]);
        }

        if(testmode == 1)
        {
            rv = sem_trywait(data.image[ID].semptr[semtrig]);
        }

        if(testmode == 2)
        {
            sem_post(data.image[ID].semptr[semtrig]);
            rv = sem_wait(data.image[ID].semptr[semtrig]);
        }

        if(rv == -1)
        {
            switch(errno)
            {

                case EINTR:
                    printf("    sem_wait call was interrupted by a signal handler\n");
                    break;

                case EINVAL:
                    printf("    not a valid semaphore\n");
                    break;

                case EAGAIN:
                    printf("    The operation could not be performed without blocking (i.e., the semaphore currently has the value zero)\n");
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

        sem_getvalue(data.image[ID].semptr[semtrig], &semval);
        sprintf(pinfomsg, "%ld TEST 1 semtrig %d  ID %ld  %d", loopcnt, semtrig, ID,
                semval);
        printf("MSG: %s\n", pinfomsg);
        fflush(stdout);


        loopcnt ++;
    }


    return RETURN_SUCCESS;
}






/** continuously transmits 2D image through TCP link
 * mode = 1, force counter to be used for synchronization, ignore semaphores if they exist
 */


imageID COREMOD_MEMORY_image_NETWORKtransmit(
    const char *IDname,
    const char *IPaddr,
    int         port,
    int         mode,
    int         RT_priority
)
{
    imageID    ID;
    struct     sockaddr_in sock_server;
    int        fds_client;
    int        flag = 1;
    int        result;
    unsigned long long  cnt = 0;
    long long  iter = 0;
    long       framesize; // pixel data only
    uint32_t   xsize, ysize;
    char      *ptr0; // source
    char      *ptr1; // source - offset by slice
    int        rs;
//    int        sockOK;

    //struct     sched_param schedpar;
    struct     timespec ts;
    long       scnt;
    int        semval;
    int        semr;
    int        slice, oldslice;
    int        NBslices;

    TCP_BUFFER_METADATA *frame_md;
    long       framesize1; // pixel data + metadata
    char      *buff; // transmit buffer


    int        semtrig = 6; // TODO - scan for available sem
    // IMPORTANT: do not use semtrig 0
    int        UseSem = 1;

    char       errmsg[200];

    int        TMPDEBUG = 0; // set to 1 for debugging this function


    printf("Transmit stream %s over IP %s port %d\n", IDname, IPaddr, port);
    fflush(stdout);

    DEBUG_TRACEPOINT(" ");

    if(TMPDEBUG == 1)
    {
        COREMOD_MEMORY_testfunction_semaphore(IDname, 0, 0);
    }

    // ===========================
    // processinfo support
    // ===========================
    PROCESSINFO *processinfo;

    char pinfoname[200];
    sprintf(pinfoname, "ntw-tx-%s", IDname);

    char descr[200];
    sprintf(descr, "%s->%s/%d", IDname, IPaddr, port);

    char pinfomsg[200];
    sprintf(pinfomsg, "setup");


    printf("Setup processinfo ...");
    fflush(stdout);
    processinfo = processinfo_setup(
                      pinfoname,
                      descr,    // description
                      pinfomsg,  // message on startup
                      __FUNCTION__, __FILE__, __LINE__
                  );
    printf(" done\n");
    fflush(stdout);



    // OPTIONAL SETTINGS
    processinfo->MeasureTiming = 1; // Measure timing
    processinfo->RT_priority =
        RT_priority;  // RT_priority, 0-99. Larger number = higher priority. If <0, ignore

    int loopOK = 1;

    ID = image_ID(IDname);


    printf("TMPDEBUG = %d\n", TMPDEBUG);
    fflush(stdout);


    if(TMPDEBUG == 0)
    {

        if((fds_client = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP)) < 0)
        {
            printf("ERROR creating socket\n");
            exit(0);
        }

        result = setsockopt(fds_client,            /* socket affected */
                            IPPROTO_TCP,     /* set option at TCP level */
                            TCP_NODELAY,     /* name of option */
                            (char *) &flag,  /* the cast is historical cruft */
                            sizeof(int));    /* length of option value */


        if(result < 0)
        {
            processinfo_error(processinfo, "ERROR: setsockopt() failed\n");
            loopOK = 0;
        }

        if(loopOK == 1)
        {
            memset((char *) &sock_server, 0, sizeof(sock_server));
            sock_server.sin_family = AF_INET;
            sock_server.sin_port = htons(port);
            sock_server.sin_addr.s_addr = inet_addr(IPaddr);

            if(connect(fds_client, (struct sockaddr *) &sock_server,
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
            if(send(fds_client, (void *) data.image[ID].md, sizeof(IMAGE_METADATA),
                    0) != sizeof(IMAGE_METADATA))
            {
                printf("send() sent a different number of bytes than expected %ld\n",
                       sizeof(IMAGE_METADATA));
                fflush(stdout);
                processinfo_error(processinfo,
                                  "send() sent a different number of bytes than expected");
                loopOK = 0;
            }
        }


        if(loopOK == 1)
        {
            xsize = data.image[ID].md[0].size[0];
            ysize = data.image[ID].md[0].size[1];
            NBslices = 1;
            if(data.image[ID].md[0].naxis > 2)
                if(data.image[ID].md[0].size[2] > 1)
                {
                    NBslices = data.image[ID].md[0].size[2];
                }
        }

        if(loopOK == 1)
        {
            switch(data.image[ID].md[0].datatype)
            {

                case _DATATYPE_INT8:
                    framesize = SIZEOF_DATATYPE_INT8 * xsize * ysize;
                    break;
                case _DATATYPE_UINT8:
                    framesize = SIZEOF_DATATYPE_UINT8 * xsize * ysize;
                    break;

                case _DATATYPE_INT16:
                    framesize = SIZEOF_DATATYPE_INT16 * xsize * ysize;
                    break;
                case _DATATYPE_UINT16:
                    framesize = SIZEOF_DATATYPE_UINT16 * xsize * ysize;
                    break;

                case _DATATYPE_INT32:
                    framesize = SIZEOF_DATATYPE_INT32 * xsize * ysize;
                    break;
                case _DATATYPE_UINT32:
                    framesize = SIZEOF_DATATYPE_UINT32 * xsize * ysize;
                    break;

                case _DATATYPE_INT64:
                    framesize = SIZEOF_DATATYPE_INT64 * xsize * ysize;
                    break;
                case _DATATYPE_UINT64:
                    framesize = SIZEOF_DATATYPE_UINT64 * xsize * ysize;
                    break;

                case _DATATYPE_FLOAT:
                    framesize = SIZEOF_DATATYPE_FLOAT * xsize * ysize;
                    break;
                case _DATATYPE_DOUBLE:
                    framesize = SIZEOF_DATATYPE_DOUBLE * xsize * ysize;
                    break;


                default:
                    printf("ERROR: WRONG DATA TYPE\n");
                    sprintf(errmsg, "WRONG DATA TYPE data type = %d\n",
                            data.image[ID].md[0].datatype);
                    printf("data type = %d\n", data.image[ID].md[0].datatype);
                    processinfo_error(processinfo, errmsg);
                    loopOK = 0;
                    break;
            }


            printf("IMAGE FRAME SIZE = %ld\n", framesize);
            fflush(stdout);
        }

        if(loopOK == 1)
        {
            switch(data.image[ID].md[0].datatype)
            {

                case _DATATYPE_INT8:
                    ptr0 = (char *) data.image[ID].array.SI8;
                    break;
                case _DATATYPE_UINT8:
                    ptr0 = (char *) data.image[ID].array.UI8;
                    break;

                case _DATATYPE_INT16:
                    ptr0 = (char *) data.image[ID].array.SI16;
                    break;
                case _DATATYPE_UINT16:
                    ptr0 = (char *) data.image[ID].array.UI16;
                    break;

                case _DATATYPE_INT32:
                    ptr0 = (char *) data.image[ID].array.SI32;
                    break;
                case _DATATYPE_UINT32:
                    ptr0 = (char *) data.image[ID].array.UI32;
                    break;

                case _DATATYPE_INT64:
                    ptr0 = (char *) data.image[ID].array.SI64;
                    break;
                case _DATATYPE_UINT64:
                    ptr0 = (char *) data.image[ID].array.UI64;
                    break;

                case _DATATYPE_FLOAT:
                    ptr0 = (char *) data.image[ID].array.F;
                    break;
                case _DATATYPE_DOUBLE:
                    ptr0 = (char *) data.image[ID].array.D;
                    break;

                default:
                    printf("ERROR: WRONG DATA TYPE\n");
                    exit(0);
                    break;
            }




            frame_md = (TCP_BUFFER_METADATA *) malloc(sizeof(TCP_BUFFER_METADATA));
            framesize1 = framesize + sizeof(TCP_BUFFER_METADATA);
            buff = (char *) malloc(sizeof(char) * framesize1);

            oldslice = 0;
            //sockOK = 1;
            printf("sem = %d\n", data.image[ID].md[0].sem);
            fflush(stdout);
        }

        if((data.image[ID].md[0].sem == 0) || (mode == 1))
        {
            processinfo_WriteMessage(processinfo, "sync using counter");
            UseSem = 0;
        }
        else
        {
            char msgstring[200];
            sprintf(msgstring, "sync using semaphore %d", semtrig);
            processinfo_WriteMessage(processinfo, msgstring);
        }

    }
    // ===========================
    // Start loop
    // ===========================
    processinfo_loopstart(
        processinfo); // Notify processinfo that we are entering loop



    while(loopOK == 1)
    {
        loopOK = processinfo_loopstep(processinfo);


        if(TMPDEBUG == 1)
        {
            sem_getvalue(data.image[ID].semptr[semtrig], &semval);
            sprintf(pinfomsg, "%ld TEST 0 semtrig %d  ID %ld  %d", processinfo->loopcnt,
                    semtrig, ID, semval);
            printf("MSG: %s\n", pinfomsg);
            fflush(stdout);
            processinfo_WriteMessage(processinfo, pinfomsg);

            //     if(semval < 3) {
            //        usleep(2000000);
            //     }

            sem_wait(data.image[ID].semptr[semtrig]);

            sem_getvalue(data.image[ID].semptr[semtrig], &semval);
            sprintf(pinfomsg, "%ld TEST 1 semtrig %d  ID %ld  %d", processinfo->loopcnt,
                    semtrig, ID, semval);
            printf("MSG: %s\n", pinfomsg);
            fflush(stdout);
            processinfo_WriteMessage(processinfo, pinfomsg);
        }
        else
        {
            if(UseSem == 0)   // use counter
            {
                while(data.image[ID].md[0].cnt0 == cnt)   // test if new frame exists
                {
                    usleep(5);
                }
                cnt = data.image[ID].md[0].cnt0;
                semr = 0;
            }
            else
            {
                if(clock_gettime(CLOCK_REALTIME, &ts) == -1)
                {
                    perror("clock_gettime");
                    exit(EXIT_FAILURE);
                }
                ts.tv_sec += 2;

#ifndef __MACH__
                semr = sem_timedwait(data.image[ID].semptr[semtrig], &ts);
#else
                alarm(1);  // send SIGALRM to process in 1 sec - Will force sem_wait to proceed in 1 sec
                semr = sem_wait(data.image[ID].semptr[semtrig]);
#endif

                if(iter == 0)
                {
                    processinfo_WriteMessage(processinfo, "Driving sem to 0");
                    printf("Driving semaphore to zero ... ");
                    fflush(stdout);
                    sem_getvalue(data.image[ID].semptr[semtrig], &semval);
                    int semvalcnt = semval;
                    for(scnt = 0; scnt < semvalcnt; scnt++)
                    {
                        sem_getvalue(data.image[ID].semptr[semtrig], &semval);
                        printf("sem = %d\n", semval);
                        fflush(stdout);
                        sem_trywait(data.image[ID].semptr[semtrig]);
                    }
                    printf("done\n");
                    fflush(stdout);

                    sem_getvalue(data.image[ID].semptr[semtrig], &semval);
                    printf("-> sem = %d\n", semval);
                    fflush(stdout);

                    iter++;
                }
            }


        }





        processinfo_exec_start(processinfo);
        if(processinfo_compute_status(processinfo) == 1)
        {
            if(TMPDEBUG == 0)
            {
                if(semr == 0)
                {
                    frame_md[0].cnt0 = data.image[ID].md[0].cnt0;
                    frame_md[0].cnt1 = data.image[ID].md[0].cnt1;

                    slice = data.image[ID].md[0].cnt1;
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

                    ptr1 = ptr0 + framesize *
                           slice; //data.image[ID].md[0].cnt1; // frame that was just written
                    memcpy(buff, ptr1, framesize);

                    memcpy(buff + framesize, frame_md, sizeof(TCP_BUFFER_METADATA));

                    rs = send(fds_client, buff, framesize1, 0);

                    if(rs != framesize1)
                    {
                        perror("socket send error ");
                        sprintf(errmsg,
                                "ERROR: send() sent a different number of bytes (%d) than expected %ld  %ld  %ld",
                                rs, (long) framesize, (long) framesize1, (long) sizeof(TCP_BUFFER_METADATA));
                        printf("%s\n", errmsg);
                        fflush(stdout);
                        processinfo_WriteMessage(processinfo, errmsg);
                        loopOK = 0;
                    }
                    oldslice = slice;
                }
            }

        }
        // process signals, increment loop counter
        processinfo_exec_end(processinfo);

        if((data.signal_INT == 1) || \
                (data.signal_TERM == 1) || \
                (data.signal_ABRT == 1) || \
                (data.signal_BUS == 1) || \
                (data.signal_SEGV == 1) || \
                (data.signal_HUP == 1) || \
                (data.signal_PIPE == 1))
        {
            loopOK = 0;
        }
    }
    // ==================================
    // ENDING LOOP
    // ==================================
    processinfo_cleanExit(processinfo);

    if(TMPDEBUG == 0)
    {
        free(buff);

        close(fds_client);
        printf("port %d closed\n", port);
        fflush(stdout);

        free(frame_md);
    }

    return ID;
}






/** continuously receives 2D image through TCP link
 * mode = 1, force counter to be used for synchronization, ignore semaphores if they exist
 */


imageID COREMOD_MEMORY_image_NETWORKreceive(
    int port,
    __attribute__((unused)) int mode,
    int RT_priority
)
{
    struct sockaddr_in   sock_server;
    struct sockaddr_in   sock_client;
    int                  fds_server;
    int                  fds_client;
    socklen_t            slen_client;

    int             flag = 1;
    long            recvsize;
    int             result;
    long            totsize = 0;
    int             MAXPENDING = 5;


    IMAGE_METADATA *imgmd;
    imageID         ID;
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
    long framesize1; // pixel data + metadata
    char *buff; // buffer



    struct sched_param schedpar;




    PROCESSINFO *processinfo;
    if(data.processinfo == 1)
    {
        // CREATE PROCESSINFO ENTRY
        // see processtools.c in module CommandLineInterface for details
        //
        char pinfoname[200];
        sprintf(pinfoname, "ntw-receive-%d", port);
        processinfo = processinfo_shm_create(pinfoname, 0);
        processinfo->loopstat = 0; // loop initialization

        strcpy(processinfo->source_FUNCTION, __FUNCTION__);
        strcpy(processinfo->source_FILE,     __FILE__);
        processinfo->source_LINE = __LINE__;

        char msgstring[200];
        sprintf(msgstring, "Waiting for input stream");
        processinfo_WriteMessage(processinfo, msgstring);
    }

    // CATCH SIGNALS

    if(sigaction(SIGTERM, &data.sigact, NULL) == -1)
    {
        printf("\ncan't catch SIGTERM\n");
    }

    if(sigaction(SIGINT, &data.sigact, NULL) == -1)
    {
        printf("\ncan't catch SIGINT\n");
    }

    if(sigaction(SIGABRT, &data.sigact, NULL) == -1)
    {
        printf("\ncan't catch SIGABRT\n");
    }

    if(sigaction(SIGBUS, &data.sigact, NULL) == -1)
    {
        printf("\ncan't catch SIGBUS\n");
    }

    if(sigaction(SIGSEGV, &data.sigact, NULL) == -1)
    {
        printf("\ncan't catch SIGSEGV\n");
    }

    if(sigaction(SIGHUP, &data.sigact, NULL) == -1)
    {
        printf("\ncan't catch SIGHUP\n");
    }

    if(sigaction(SIGPIPE, &data.sigact, NULL) == -1)
    {
        printf("\ncan't catch SIGPIPE\n");
    }






    schedpar.sched_priority = RT_priority;
#ifndef __MACH__
    if(seteuid(data.euid) != 0)     //This goes up to maximum privileges
    {
        PRINT_ERROR("seteuid error");
    }
    sched_setscheduler(0, SCHED_FIFO,
                       &schedpar); //other option is SCHED_RR, might be faster
    if(seteuid(data.ruid) != 0)     //Go back to normal privileges
    {
        PRINT_ERROR("seteuid error");
    }
#endif

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

    result = setsockopt(fds_server,            /* socket affected */
                        IPPROTO_TCP,     /* set option at TCP level */
                        TCP_NODELAY,     /* name of option */
                        (char *) &flag,  /* the cast is historical cruft */
                        sizeof(int));    /* length of option value */
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


    sock_server.sin_family = AF_INET;
    sock_server.sin_port = htons(port);
    sock_server.sin_addr.s_addr = htonl(INADDR_ANY);

    //bind socket to port
    if(bind(fds_server, (struct sockaddr *)&sock_server,
            sizeof(sock_server)) == -1)
    {
        char msgstring[200];

        sprintf(msgstring, "ERROR binding socket, port %d", port);
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

        sprintf(msgstring, "ERROR listen socket");
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
    if((fds_client = accept(fds_server, (struct sockaddr *) &sock_client,
                            &slen_client)) == -1)
    {
        char msgstring[200];

        sprintf(msgstring, "ERROR accept socket");
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
    if((recvsize = recv(fds_client, imgmd, sizeof(IMAGE_METADATA),
                        MSG_WAITALL)) < 0)
    {
        char msgstring[200];

        sprintf(msgstring, "ERROR receiving image metadata");
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
        sprintf(msgstring, "Receiving stream %s", imgmd[0].name);
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

    list_image_ID();

    if(ID == -1)
    {
        OKim = 0;
    }
    else
    {
        OKim = 1;
        if(imgmd[0].naxis != data.image[ID].md[0].naxis)
        {
            OKim = 0;
        }
        if(OKim == 1)
        {
            for(axis = 0; axis < imgmd[0].naxis; axis++)
                if(imgmd[0].size[axis] != data.image[ID].md[0].size[axis])
                {
                    OKim = 0;
                }
        }
        if(imgmd[0].datatype != data.image[ID].md[0].datatype)
        {
            OKim = 0;
        }

        if(OKim == 0)
        {
            delete_image_ID(imgmd[0].name);
            ID = -1;
        }
    }



    if(OKim == 0)
    {
        printf("IMAGE %s HAS TO BE CREATED\n", imgmd[0].name);
        ID = create_image_ID(imgmd[0].name, imgmd[0].naxis, imgmd[0].size,
                             imgmd[0].datatype, imgmd[0].shared, 0);
        printf("Created image stream %s - shared = %d\n", imgmd[0].name,
               imgmd[0].shared);
    }
    else
    {
        printf("REUSING EXISTING IMAGE %s\n", imgmd[0].name);
    }





    COREMOD_MEMORY_image_set_createsem(imgmd[0].name, IMAGE_NB_SEMAPHORE);

    xsize = data.image[ID].md[0].size[0];
    ysize = data.image[ID].md[0].size[1];
    NBslices = 1;
    if(data.image[ID].md[0].naxis > 2)
        if(data.image[ID].md[0].size[2] > 1)
        {
            NBslices = data.image[ID].md[0].size[2];
        }


    char typestring[8];

    switch(data.image[ID].md[0].datatype)
    {

        case _DATATYPE_INT8:
            framesize = SIZEOF_DATATYPE_INT8 * xsize * ysize;
            sprintf(typestring, "INT8");
            break;

        case _DATATYPE_UINT8:
            framesize = SIZEOF_DATATYPE_UINT8 * xsize * ysize;
            sprintf(typestring, "UINT8");
            break;

        case _DATATYPE_INT16:
            framesize = SIZEOF_DATATYPE_INT16 * xsize * ysize;
            sprintf(typestring, "INT16");
            break;

        case _DATATYPE_UINT16:
            framesize = SIZEOF_DATATYPE_UINT16 * xsize * ysize;
            sprintf(typestring, "UINT16");
            break;

        case _DATATYPE_INT32:
            framesize = SIZEOF_DATATYPE_INT32 * xsize * ysize;
            sprintf(typestring, "INT32");
            break;

        case _DATATYPE_UINT32:
            framesize = SIZEOF_DATATYPE_UINT32 * xsize * ysize;
            sprintf(typestring, "UINT32");
            break;

        case _DATATYPE_INT64:
            framesize = SIZEOF_DATATYPE_INT64 * xsize * ysize;
            sprintf(typestring, "INT64");
            break;

        case _DATATYPE_UINT64:
            framesize = SIZEOF_DATATYPE_UINT64 * xsize * ysize;
            sprintf(typestring, "UINT64");
            break;

        case _DATATYPE_FLOAT:
            framesize = SIZEOF_DATATYPE_FLOAT * xsize * ysize;
            sprintf(typestring, "FLOAT");
            break;

        case _DATATYPE_DOUBLE:
            framesize = SIZEOF_DATATYPE_DOUBLE * xsize * ysize;
            sprintf(typestring, "DOUBLE");
            break;

        default:
            printf("ERROR: WRONG DATA TYPE\n");
            sprintf(typestring, "ERR");
            exit(0);
            break;
    }

    printf("image frame size = %ld\n", framesize);

    switch(data.image[ID].md[0].datatype)
    {

        case _DATATYPE_INT8:
            ptr0 = (char *) data.image[ID].array.SI8;
            break;
        case _DATATYPE_UINT8:
            ptr0 = (char *) data.image[ID].array.UI8;
            break;

        case _DATATYPE_INT16:
            ptr0 = (char *) data.image[ID].array.SI16;
            break;
        case _DATATYPE_UINT16:
            ptr0 = (char *) data.image[ID].array.UI16;
            break;

        case _DATATYPE_INT32:
            ptr0 = (char *) data.image[ID].array.SI32;
            break;
        case _DATATYPE_UINT32:
            ptr0 = (char *) data.image[ID].array.UI32;
            break;

        case _DATATYPE_INT64:
            ptr0 = (char *) data.image[ID].array.SI64;
            break;
        case _DATATYPE_UINT64:
            ptr0 = (char *) data.image[ID].array.UI64;
            break;

        case _DATATYPE_FLOAT:
            ptr0 = (char *) data.image[ID].array.F;
            break;
        case _DATATYPE_DOUBLE:
            ptr0 = (char *) data.image[ID].array.D;
            break;

        default:
            printf("ERROR: WRONG DATA TYPE\n");
            exit(0);
            break;
    }



    if(data.processinfo == 1)
    {
        char msgstring[200];
        sprintf(msgstring, "<- %s [%d x %d x %ld] %s", imgmd[0].name, (int) xsize,
                (int) ysize, NBslices, typestring);
        sprintf(processinfo->description, "%s %dx%dx%ld %s", imgmd[0].name, (int) xsize,
                (int) ysize, NBslices, typestring);
        processinfo_WriteMessage(processinfo, msgstring);
    }



    // this line is not needed, as frame_md is declared below
    // frame_md = (TCP_BUFFER_METADATA*) malloc(sizeof(TCP_BUFFER_METADATA));

    framesize1 = framesize + sizeof(TCP_BUFFER_METADATA);
    buff = (char *) malloc(sizeof(char) * framesize1);

    frame_md = (TCP_BUFFER_METADATA *)(buff + framesize);



    if(data.processinfo == 1)
    {
        processinfo->loopstat = 1;    //notify processinfo that we are entering loop
    }

    socketOpen = 1;
    long loopcnt = 0;
    int loopOK = 1;

    while(loopOK == 1)
    {
        if(data.processinfo == 1)
        {
            while(processinfo->CTRLval == 1)   // pause
            {
                usleep(50);
            }

            if(processinfo->CTRLval == 2)   // single iteration
            {
                processinfo->CTRLval = 1;
            }

            if(processinfo->CTRLval == 3)   // exit loop
            {
                loopOK = 0;
            }
        }


        if((recvsize = recv(fds_client, buff, framesize1, MSG_WAITALL)) < 0)
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


            data.image[ID].md[0].cnt1 = frame_md[0].cnt1;


            if(NBslices > 1)
            {
                memcpy(ptr0 + framesize * frame_md[0].cnt1, buff, framesize);
            }
            else
            {
                memcpy(ptr0, buff, framesize);
            }

            data.image[ID].md[0].cnt0++;
            for(semnb = 0; semnb < data.image[ID].md[0].sem ; semnb++)
            {
                sem_getvalue(data.image[ID].semptr[semnb], &semval);
                if(semval < SEMAPHORE_MAXVAL)
                {
                    sem_post(data.image[ID].semptr[semnb]);
                }
            }

            sem_getvalue(data.image[ID].semlog, &semval);
            if(semval < 2)
            {
                sem_post(data.image[ID].semlog);
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

        if(data.signal_TERM == 1)
        {
            loopOK = 0;
            if(data.processinfo == 1)
            {
                processinfo_SIGexit(processinfo, SIGTERM);
            }
        }

        if(data.signal_INT == 1)
        {
            loopOK = 0;
            if(data.processinfo == 1)
            {
                processinfo_SIGexit(processinfo, SIGINT);
            }
        }

        if(data.signal_ABRT == 1)
        {
            loopOK = 0;
            if(data.processinfo == 1)
            {
                processinfo_SIGexit(processinfo, SIGABRT);
            }
        }

        if(data.signal_BUS == 1)
        {
            loopOK = 0;
            if(data.processinfo == 1)
            {
                processinfo_SIGexit(processinfo, SIGBUS);
            }
        }

        if(data.signal_SEGV == 1)
        {
            loopOK = 0;
            if(data.processinfo == 1)
            {
                processinfo_SIGexit(processinfo, SIGSEGV);
            }
        }

        if(data.signal_HUP == 1)
        {
            loopOK = 0;
            if(data.processinfo == 1)
            {
                processinfo_SIGexit(processinfo, SIGHUP);
            }
        }

        if(data.signal_PIPE == 1)
        {
            loopOK = 0;
            if(data.processinfo == 1)
            {
                processinfo_SIGexit(processinfo, SIGPIPE);
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


    free(buff);

    close(fds_client);

    printf("port %d closed\n", port);
    fflush(stdout);

    free(imgmd);


    return ID;
}




