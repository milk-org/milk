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
static int MULTIGRAM_MAGIC = 0x3E; // Random magic to start datagrams with.
static int DGRAM_CHUNK_SIZE = 62 *
                              1024; // Max payload per datagram, just shy of the maximum 65507 bytes


// ==========================================
// Forward declaration(s)
// ==========================================

imageID COREMOD_MEMORY_image_NETUDPtransmit(const char *IDname,
        const char *IPaddr,
        int         port,
        int         do_counter_sync,
        int         RT_priority);

imageID COREMOD_MEMORY_image_NETUDPreceive(int port,
        int do_counter_sync,
        int RT_priority);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t COREMOD_MEMORY_image_NETUDPtransmit__cli()
{
    if(0 + CLI_checkarg(1, CLIARG_IMG) + CLI_checkarg(2, CLIARG_STR_NOT_IMG) +
            CLI_checkarg(3, CLIARG_INT64) + CLI_checkarg(4, CLIARG_INT64) +
            CLI_checkarg(5, CLIARG_INT64) ==
            0)
    {
        COREMOD_MEMORY_image_NETUDPtransmit(data.cmdargtoken[1].val.string,
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

static errno_t COREMOD_MEMORY_image_NETUDPreceive__cli()
{
    if(0 + CLI_checkarg(1, CLIARG_INT64) + CLI_checkarg(2, CLIARG_INT64) +
            CLI_checkarg(3, CLIARG_INT64) ==
            0)
    {
        COREMOD_MEMORY_image_NETUDPreceive(data.cmdargtoken[1].val.numl,
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

errno_t stream__UDP_addCLIcmd()
{

    RegisterCLIcommand(
        "imudptransmit",
        __FILE__,
        COREMOD_MEMORY_image_NETUDPtransmit__cli,
        "transmit image over network",
        "<image> <IP addr> <port [long]> <do_counter_sync [int]>",
        "imudptransmit im1 127.0.0.1 0 8888 0",
        "long COREMOD_MEMORY_image_NETWORKtransmit(const char "
        "*IDname, const char *IPaddr, int port, int do_counter_sync)");

    RegisterCLIcommand(
        "imudpreceive",
        __FILE__,
        COREMOD_MEMORY_image_NETUDPreceive__cli,
        "receive image(s) over network. do_counter_sync=1 uses counter "
        "instead of semaphore",
        "<port [long]> <do_counter_sync [int]> <RT priority>",
        "imupdreceive 8887 0 80",
        "long COREMOD_MEMORY_image_NETWORKreceive(int port, int "
        "do_counter_sync, int RT_priority)");

    return RETURN_SUCCESS;
}

/** continuously transmits 2D image through TCP link
 * do_counter_sync = 1, force counter to be used for synchronization, ignore semaphores if they exist
 */

imageID COREMOD_MEMORY_image_NETUDPtransmit(const char *IDname,
        const char *IPaddr,
        int         port,
        int         do_counter_sync,
        int         RT_priority)
{
    imageID            ID;
    struct sockaddr_in sock_server;
    int                fds_client;
    int                flag = 1;
    int                result;
    unsigned long long cnt  = 0;
    long long          iter = 0;
    long               framesize; // pixel data only
    uint32_t           xsize, ysize;
    char              *ptr_img_data; // source
    char              *ptr_img_data_slice; // source - offset by slice
    int                res; // Return status for socket ops
    int                byte_sock_count;

    struct timespec ts;
    long            scnt;
    int             semval;
    int             semr;
    int             slice, oldslice;
    int             NBslices;

    long            framesize1; // pixel data + metadata
    long            framesizeall; // total frame size : pixel data + metadata + kw

    char           *buff; // socket-side buffer (magic and metadata at beginning)
    char           *ptr_buff_metadata; // socket-side buffer at metadata offset
    char           *ptr_buff_data; // socket-side buffer at data offset
    char           *ptr_buff_keywords; // socket-side buffer at keyword offset

    // Datagrams
    long            n_udp_dgrams;
    long            last_dgram_chunk;
    char           *ptr_this_dgram;
    long            this_dgram_size;


    int semtrig = 6; // TODO - scan for available sem
    // IMPORTANT: do not use semtrig 0
    int use_sem = 1;

    char errmsg[200];

    printf("Transmit stream %s over UDP/IP %s port %d\n", IDname, IPaddr, port);
    fflush(stdout);

    DEBUG_TRACEPOINT(" ");

    // ===========================
    // processinfo support
    // ===========================
    PROCESSINFO *processinfo;

    char pinfoname[STRINGMAXLEN_FILENAME];
    snprintf(pinfoname, STRINGMAXLEN_FILENAME, "ntw-tx-%s", IDname);

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

    int loopOK = 1; // Master flag

    ID = image_ID(IDname);

    if((fds_client = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0)
    {
        printf("ERROR creating socket\n");
        exit(0);
    }

    setsockopt(fds_client, SOL_SOCKET, SO_REUSEADDR, (char *) & flag, sizeof(flag));
    setsockopt(fds_client, SOL_SOCKET, SO_REUSEPORT, (char *) & flag, sizeof(flag));

#ifdef SO_ATTACH_REUSEPORT_CBPF
    setsockopt(fds_client, SOL_SOCKET, SO_ATTACH_REUSEPORT_CBPF, (char *) & flag,
               sizeof(flag));
#endif

    if(loopOK == 1)
    {
        memset((char *) &sock_server, 0, sizeof(sock_server));
        sock_server.sin_family      = AF_INET;
        sock_server.sin_port        = htons(port);
        sock_server.sin_addr.s_addr = inet_addr(IPaddr);
    }

    if(loopOK == 1)
    {
        xsize    = data.image[ID].md[0].size[0];
        ysize    = data.image[ID].md[0].size[1];
        NBslices = 1;
        if(data.image[ID].md[0].naxis > 2 && data.image[ID].md[0].size[2] > 1)
        {
            NBslices = data.image[ID].md[0].size[2];
        }
    }

    if(loopOK == 1)
    {
        framesize = ImageStreamIO_typesize(data.image[ID].md[0].datatype) * xsize *
                    ysize;
        printf("IMAGE FRAME SIZE = %ld\n", framesize);
        fflush(stdout);
    }

    if(loopOK == 1)
    {
        ptr_img_data = (char *) ImageStreamIO_get_image_d_ptr(&data.image[ID]);

        framesize1 = framesize + sizeof(IMAGE_METADATA);

        if(TCPTRANSFERKW == 0)
        {
            framesizeall = framesize1;
        }
        else
        {
            framesizeall =
                framesize1 + data.image[ID].md[0].NBkw * sizeof(IMAGE_KEYWORD);
        }

        // Prepare segmentation into 62k datagrams
        n_udp_dgrams = framesizeall / DGRAM_CHUNK_SIZE + 1;
        last_dgram_chunk = framesizeall % DGRAM_CHUNK_SIZE;

        // Prepare transmit buffer - add two bytes for the magic + dgram number
        buff = (char *) malloc(sizeof(char) * framesizeall);
        ptr_buff_metadata = buff + 2;
        ptr_buff_data = ptr_buff_metadata + sizeof(IMAGE_METADATA);
        ptr_buff_keywords = ptr_buff_data + framesize;


        printf("Transfer buffer size = %ld\n", framesizeall);
        printf("Using %ld UDP datagrams\n", n_udp_dgrams);
        fflush(stdout);

        oldslice = 0;
        //sockOK = 1;
        printf("sem = %d\n", data.image[ID].md[0].sem);
        fflush(stdout);
    }

    if((data.image[ID].md[0].sem == 0) || (do_counter_sync == 1))
    {
        processinfo_WriteMessage(processinfo, "sync using counter");
        use_sem = 0;
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

        if(use_sem == 0)  // use counter
        {
            while(data.image[ID].md[0].cnt0 == cnt)  // test if new frame exists
            {
                usleep(5);
            }
            cnt  = data.image[ID].md[0].cnt0;
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

            semr = ImageStreamIO_semtimedwait(data.image+ID, semtrig, &ts);

            if(iter == 0)
            {
                processinfo_WriteMessage(processinfo, "Driving sem to 0");
                printf("Driving semaphore to zero ... ");
                fflush(stdout);
                semval = ImageStreamIO_semvalue(data.image+ID, semtrig);
                int semvalcnt = semval;
                for(scnt = 0; scnt < semvalcnt; scnt++)
                {
                    semval = ImageStreamIO_semvalue(data.image+ID, semtrig);
                    printf("sem = %d\n", semval);
                    fflush(stdout);
                    ImageStreamIO_semtrywait(data.image+ID, semtrig);
                }
                printf("done\n");
                fflush(stdout);

                semval = ImageStreamIO_semvalue(data.image+ID, semtrig);
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

                // Fill up the transmission buffer
                memcpy(ptr_buff_metadata, &data.image[ID].md[0], sizeof(IMAGE_METADATA));

                ptr_img_data_slice = ptr_img_data + framesize * slice;
                memcpy(ptr_buff_data, ptr_img_data_slice, framesize);

                if(TCPTRANSFERKW == 1)
                {
                    memcpy(ptr_buff_keywords,
                           (char *) data.image[ID].kw,
                           data.image[ID].md[0].NBkw * sizeof(IMAGE_KEYWORD));
                }

                // Send the datagrams
                byte_sock_count = 0;
                ptr_this_dgram = ptr_buff_metadata - 2;
                for(int dgram = 0; dgram < n_udp_dgrams; ++dgram)
                {
                    this_dgram_size = dgram == n_udp_dgrams - 1 ? last_dgram_chunk + 2 :
                                      DGRAM_CHUNK_SIZE + 2;
                    // Using the extra 2 bytes at the beginning for the first dgram
                    // Overwriting the 2 last bytes of previous dgrams for subsequent ones
                    ptr_this_dgram[0] = MULTIGRAM_MAGIC;
                    ptr_this_dgram[1] = dgram;

                    //printf("This dgram id: %d, size: %ld\n", dgram, this_dgram_size);
                    res = sendto(fds_client, ptr_this_dgram, this_dgram_size, 0,
                                 (const struct sockaddr *)&sock_server, sizeof(sock_server));
                    byte_sock_count += res;

                    ptr_this_dgram += DGRAM_CHUNK_SIZE; // Shift by 62k
                }

                if(byte_sock_count != framesizeall + 2 * n_udp_dgrams)
                {
                    perror("socket send error ");
                    snprintf(errmsg,
                             200,
                             "ERROR: send() sent a different "
                             "number of bytes (%d) than "
                             "expected %ld",
                             byte_sock_count,
                             framesizeall + 2 * n_udp_dgrams);
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

    return ID;
}

/** continuously receives 2D image through TCP link
 * do_counter_sync = 1, force counter to be used for synchronization, ignore semaphores if they exist
 */

imageID COREMOD_MEMORY_image_NETUDPreceive(
    int port,
    __attribute__((unused)) int do_counter_sync,
    int RT_priority)
{
    struct sockaddr_in sock_server;
    struct sockaddr_in sock_client;
    int                fds_server;
    int                fds_client;
    socklen_t          slen_client = (socklen_t) sizeof(sock_client);

    int  flag = 1;
    long recvsize;
    int  result;
    int  MAXPENDING = 5;

    IMAGE_METADATA *imgmd;
    IMAGE_METADATA *imgmd_remote;
    imageID         ID;
    long            framesize;
    uint32_t        xsize;
    uint32_t        ysize;

    char           *ptr_dest_data_root; // Dest ISIO data buffer
    char           *ptr_dest_data_sliceroot; // Dest ISIO data buffer
    char           *ptr_dest_data_current; // Dest ISIO data buffer

    char           *ptr_buff_metadata; // socket-side buffer at metadata offset
    char           *ptr_buff_data; // socket-side buffer at data offset
    char           *ptr_buff_keywords; // socket-side buffer at keyword offset

    char           *buff; // socket-side complete buffer
    char           *buff_udp; // socket-side datagram buffer
    buff_udp = (char *) malloc(sizeof(char) * DGRAM_CHUNK_SIZE + 2);

    // Datagrams
    long            n_udp_dgrams;
    long            last_dgram_chunk;

    long            NBslices;
    int             socketOpen = 1; // 0 if socket is closed
    int             semval;
    int             semnb;
    int             OKim;
    int             axis;

    imgmd = (IMAGE_METADATA *) malloc(sizeof(IMAGE_METADATA));

    long                 framesize1;    // pixel data + metadata
    long                 framesizefull; // pixel data + metadata + kw

    struct sched_param schedpar;

    PROCESSINFO *processinfo;
    if(data.processinfo == 1)
    {
        // CREATE PROCESSINFO ENTRY
        // see processtools.c in module CommandLineInterface for details
        //
        char pinfoname[STRINGMAXLEN_FILENAME];
        snprintf(pinfoname, STRINGMAXLEN_FILENAME, "ntw-receive-%d", port);
        processinfo           = processinfo_shm_create(pinfoname, 0);
        processinfo->loopstat = PROCESSINFO_LOOPSTAT_INIT;

        strcpy(processinfo->source_FUNCTION, __FUNCTION__);
        strcpy(processinfo->source_FILE, __FILE__);
        processinfo->source_LINE = __LINE__;

        processinfo_WriteMessage(processinfo, "Waiting for input stream");
    }

    // CATCH SIGNALS
    {
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
    }

    schedpar.sched_priority = RT_priority;
    if(seteuid(data.euid) != 0)  //This goes up to maximum privileges
    {
        PRINT_ERROR("seteuid error");
    }
    sched_setscheduler(0, SCHED_FIFO, &schedpar);
    if(seteuid(data.ruid) != 0)    //Go back to normal privileges
    {
        PRINT_ERROR("seteuid error");
    }

    // create UDP socket
    if((fds_server = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1)
    {
        printf("ERROR creating socket\n");
        if(data.processinfo == 1)
        {
            processinfo->loopstat = PROCESSINFO_LOOPSTAT_ERROR;
            processinfo_WriteMessage(processinfo, "ERROR creating socket");
        }
        exit(0);
    }

    memset((char *) &sock_server, 0, sizeof(sock_server));

    sock_server.sin_family      = AF_INET;
    sock_server.sin_port        = htons(port);
    sock_server.sin_addr.s_addr = htonl(INADDR_ANY);

    setsockopt(fds_server, SOL_SOCKET, SO_NO_CHECK, (char *) & flag, sizeof(flag));
    setsockopt(fds_server, SOL_SOCKET, SO_REUSEADDR, (char *) & flag, sizeof(flag));
    setsockopt(fds_server, SOL_SOCKET, SO_REUSEPORT, (char *) & flag, sizeof(flag));

#ifdef SO_ATTACH_REUSEPORT_CBPF
    setsockopt(fds_server, SOL_SOCKET, SO_ATTACH_REUSEPORT_CBPF, (char *) & flag,
               sizeof(flag));
#endif

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
            processinfo->loopstat = PROCESSINFO_LOOPSTAT_ERROR;
            processinfo_WriteMessage(processinfo, msgstring);
        }
        exit(0);
    }

    // Try and receive only the metadata
    // May have to go through several datagrams...
    int MAX_DATAGRAM_WAIT = 300;
    for(int n_dgram_wait = 0; n_dgram_wait < MAX_DATAGRAM_WAIT; ++n_dgram_wait)
    {
        recvsize =
            recvfrom(fds_server, buff_udp, sizeof(IMAGE_METADATA) + 2, 0,
                     (struct sockaddr *)&sock_client, &slen_client);
        if(recvsize < 0 || n_dgram_wait == MAX_DATAGRAM_WAIT - 1)
        {
            char msgstring[200];

            snprintf(msgstring,
                     200,
                     "ERROR receiving image metadata, recvsize = %ld, n_dgram_wait = %d",
                     recvsize, n_dgram_wait);
            printf("%s\n", msgstring);

            if(data.processinfo == 1)
            {
                processinfo->loopstat = PROCESSINFO_LOOPSTAT_ERROR;
                processinfo_WriteMessage(processinfo, msgstring);
            }

            exit(0);
        }

        // printf("Init phase: recvsize = %ld, buff_udp[0] = %d, buff_udp[1] = %d\n", recvsize, buff_udp[0], buff_udp[1]);

        // If this is a first datagram, we're having the metadata here:
        if(buff_udp[0] == MULTIGRAM_MAGIC && buff_udp[1] == 0)
        {
            memcpy(imgmd, buff_udp + 2, sizeof(IMAGE_METADATA));
            break;
        }
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
            delete_image_ID(imgmd[0].name, DELETE_IMAGE_ERRMODE_WARNING);
            ID = -1;
        }
    }

    int nbkw = 0;
    if(TCPTRANSFERKW == 1)
    {
        nbkw = imgmd[0].NBkw;
        if(imgmd[0].NBkw != data.image[ID].md[0].NBkw)
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

    xsize    = data.image[ID].md[0].size[0];
    ysize    = data.image[ID].md[0].size[1];
    NBslices = 1;
    if(data.image[ID].md[0].naxis > 2)
        if(data.image[ID].md[0].size[2] > 1)
        {
            NBslices = data.image[ID].md[0].size[2];
        }



    if(data.processinfo == 1)
    {
        char typestring[8];
        snprintf(typestring, 8, "%s",
                 ImageStreamIO_typename(data.image[ID].md[0].datatype));
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
                 STRINGMAXLEN_PROCESSINFO_DESCRIPTION,
                 "%s %dx%dx%ld %s",
                 imgmd[0].name,
                 (int) xsize,
                 (int) ysize,
                 NBslices,
                 typestring);
        processinfo_WriteMessage(processinfo, msgstring);
    }

    framesize =
        ImageStreamIO_typesize(data.image[ID].md[0].datatype) * xsize * ysize;
    printf("image frame size = %ld\n", framesize);

    ptr_dest_data_root = (char *) ImageStreamIO_get_image_d_ptr(&data.image[ID]);

    framesize1 = framesize + sizeof(IMAGE_METADATA);
    if(TCPTRANSFERKW == 0)
    {
        framesizefull = framesize1;
    }
    else
    {
        framesizefull = framesize1 + nbkw * sizeof(IMAGE_KEYWORD);
    }



    // TODO
    buff = (char *) malloc(sizeof(char) * framesizefull);
    ptr_buff_metadata = buff;
    ptr_buff_data = ptr_buff_metadata + sizeof(IMAGE_METADATA);
    ptr_buff_keywords = ptr_buff_data + framesize;

    n_udp_dgrams = framesizefull / DGRAM_CHUNK_SIZE + 1;
    last_dgram_chunk = framesizefull % DGRAM_CHUNK_SIZE;

    if(data.processinfo == 1)
    {
        //notify processinfo that we are entering loop
        processinfo->loopstat = PROCESSINFO_LOOPSTAT_ACTIVE;
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

    long first_dgram_bytes = n_udp_dgrams == 1 ? last_dgram_chunk + 2 :
                             DGRAM_CHUNK_SIZE + 2;
    long this_dgram_bytes;
    int abort_frame;


    while(loopOK == 1)
    {
        if(data.processinfo == 1)
        {
            while(processinfo->CTRLval == PROCESSINFO_CTRLVAL_PAUSE)
            {
                usleep(50);
            }

            if(processinfo->CTRLval == PROCESSINFO_CTRLVAL_INCR)
            {
                processinfo->CTRLval = PROCESSINFO_CTRLVAL_PAUSE;
            }

            if(processinfo->CTRLval == PROCESSINFO_CTRLVAL_EXIT)  // exit loop
            {
                loopOK = 0;
            }
        }

        // Resync to a 0-zth datagram if necessary
        abort_frame = 0;
        for(int n_dgram_wait = 0; n_dgram_wait < MAX_DATAGRAM_WAIT; ++n_dgram_wait)
        {
            recvsize = recvfrom(fds_server, buff_udp, first_dgram_bytes, 0,
                                (struct sockaddr *)&sock_client, &slen_client);
            if(recvsize < 0 || n_dgram_wait == MAX_DATAGRAM_WAIT - 1)
            {
                printf("ERROR recvfrom`()\n");
                socketOpen = 0;
                break;
            }

            if(buff_udp[0] == MULTIGRAM_MAGIC && buff_udp[1] == 0)
            {
                memcpy(buff, buff_udp + 2, first_dgram_bytes - 2);
                break;
            }
        }



        if((data.processinfo == 1) && (processinfo->MeasureTiming == 1))
        {
            processinfo_exec_start(processinfo);
        }

        if(recvsize == 0)
        {
            socketOpen = 0;
        }

        if(socketOpen == 1)
        {
            // We already have the first datagram.

            // Weak copy although we now have all the metadata in buff
            imgmd_remote = (IMAGE_METADATA *)(ptr_buff_metadata);

            data.image[ID].md[0].cnt1 =
                imgmd_remote[0].cnt1; // For multi-slice only, really.

            // copy pixel data. Watch that cnt1 == cnt0 for unsliced data, so need to ignore
            if(NBslices == 1)
            {
                ptr_dest_data_sliceroot = ptr_dest_data_root;
            }
            else
            {
                ptr_dest_data_sliceroot = ptr_dest_data_root + framesize * imgmd_remote[0].cnt1;
            }

            // Acquire and copy subsequent datagrams
            for(int k_dgram = 1; k_dgram < n_udp_dgrams ; ++k_dgram)
            {
                this_dgram_bytes = k_dgram == n_udp_dgrams - 1 ? last_dgram_chunk :
                                   DGRAM_CHUNK_SIZE;
                recvsize = recvfrom(fds_server, buff_udp, first_dgram_bytes, 0,
                                    (struct sockaddr *)&sock_client, &slen_client);

                if(recvsize < 0)
                {
                    printf("ERROR recvfrom`()\n");
                    socketOpen = 0;
                    break;
                }
                if(buff_udp[0] != MULTIGRAM_MAGIC || buff_udp[1] != k_dgram)
                {
                    printf("UDP datagram sequence error (magic: %d, seen: %d, expected: %d)\n",
                           buff_udp[0], buff_udp[1], k_dgram);
                    abort_frame = 1;
                    break;
                }
                memcpy(buff + k_dgram * DGRAM_CHUNK_SIZE, buff_udp + 2, this_dgram_bytes);
            }
        }
        if(socketOpen == 1 && abort_frame == 0)
        {

            // Copy the data !
            memcpy(ptr_dest_data_sliceroot, ptr_buff_data, framesize);

            if(TCPTRANSFERKW == 1)
            {
                // copy kw
                memcpy(data.image[ID].kw,
                       (IMAGE_KEYWORD *)(ptr_buff_keywords),
                       nbkw * sizeof(IMAGE_KEYWORD));
            }

            frameincr = (long) imgmd_remote[0].cnt0 - cnt0previous;

            if(frameincr > 1)
            {
                printf("Skipped %ld frame(s) at index %ld %ld\n",
                       frameincr - 1,
                       (long)(imgmd_remote[0].cnt0),
                       (long)(imgmd_remote[0].cnt1));
            }

            cnt0previous = imgmd_remote[0].cnt0;

            if(monitorindex == monitorinterval)
            {
                printf(
                    "[%5ld]  input %20ld (+ %8ld) output %20ld (+ "
                    "%8ld)\n",
                    monitorloopindex,
                    imgmd_remote[0].cnt0,
                    imgmd_remote[0].cnt0 - minputcnt,
                    data.image[ID].md[0].cnt0,
                    data.image[ID].md[0].cnt0 - moutputcnt);

                minputcnt  = imgmd_remote[0].cnt0;
                moutputcnt = data.image[ID].md[0].cnt0;

                monitorloopindex++;
                monitorindex = 0;
            }

            monitorindex++;

            data.image[ID].md[0].cnt0++;
            for(semnb = 0; semnb < data.image[ID].md[0].sem; semnb++)
            {
                semval = ImageStreamIO_semvalue(data.image+ID, semnb);
                if(semval < SEMAPHORE_MAXVAL)
                {
                    ImageStreamIO_sempost(data.image+ID, semnb);
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

        if((data.signal_TERM | data.signal_INT | data.signal_ABRT |
                data.signal_BUS | data.signal_SEGV | data.signal_HUP |
                data.signal_PIPE) != 0)
        {
            loopOK = 0;
            if(data.processinfo == 1)
            {
                if(data.signal_TERM == 1)
                {
                    processinfo_SIGexit(processinfo, SIGTERM);
                }
                else if(data.signal_INT == 1)
                {
                    processinfo_SIGexit(processinfo, SIGINT);
                }
                else if(data.signal_ABRT == 1)
                {
                    processinfo_SIGexit(processinfo, SIGABRT);
                }
                else if(data.signal_BUS == 1)
                {
                    processinfo_SIGexit(processinfo, SIGBUS);
                }
                else if(data.signal_SEGV == 1)
                {
                    processinfo_SIGexit(processinfo, SIGSEGV);
                }
                else if(data.signal_HUP == 1)
                {
                    processinfo_SIGexit(processinfo, SIGHUP);
                }
                else if(data.signal_PIPE == 1)
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

    free(buff);
    free(buff_udp);

    close(fds_client);

    printf("port %d closed\n", port);
    fflush(stdout);

    free(imgmd);

    return ID;
}
