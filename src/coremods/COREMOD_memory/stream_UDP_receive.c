/**
 * @file    stream_UDP_receive.c
 * @brief   UDP stream receive function
 *
 * Extracted from stream_UDP.c
 *
 * @see stream_UDP.c for transmit and FPS framework.
 */

#include <arpa/inet.h>
#include <netinet/in.h>
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
static int MULTIGRAM_MAGIC = 0x3E;
static int DGRAM_CHUNK_SIZE = 62 *
                              1024;

/** continuously receives 2D image through TCP link
 * do_counter_sync = 1, force counter to be used for synchronization, ignore semaphores if they exist
 */

imageID COREMOD_MEMORY_image_NETUDPreceive(
    int                         port,
    __attribute__((unused)) int do_counter_sync,
    int RT_priority)
{
    struct sockaddr_in sock_server;
    struct sockaddr_in sock_client;
    int                fds_server;
    //int                fds_client;
    socklen_t          slen_client = (socklen_t) sizeof(sock_client);

    int  flag = 1;
    long recvsize;
    //int  result;
    //int  MAXPENDING = 5;

    IMAGE_METADATA *imgmd;
    IMAGE_METADATA *imgmd_remote;
    imageID         ID;
    long            framesize;
    uint32_t        xsize;
    uint32_t        ysize;

    char           *ptr_dest_data_root; // Dest ISIO data buffer
    char           *ptr_dest_data_sliceroot; // Dest ISIO data buffer
//    char           *ptr_dest_data_current; // Dest ISIO data buffer

    char           *ptr_buff_metadata; // socket-side buffer at metadata offset
    char           *ptr_buff_data; // socket-side buffer at data offset
    char           *ptr_buff_keywords; // socket-side buffer at keyword offset

    char           *buff; // socket-side complete buffer
    char           *buff_udp; // socket-side datagram buffer
    buff_udp = (char *) malloc(sizeof(char) * DGRAM_CHUNK_SIZE + 2);
    char           *bigbuff_1MB; // socket-side datagram buffer
    bigbuff_1MB = (char *) malloc(sizeof(char) * 1024 * 1024);

    // Datagrams
    long            n_udp_dgrams;
    long            last_dgram_chunk;

    long            NBslices;
    int             socketOpen = 1; // 0 if socket is closed
    int             semval;
    int             semnb;
    int             OKim;


    imgmd = (IMAGE_METADATA *) malloc(sizeof(IMAGE_METADATA));

    long                 framesize1;    // pixel data + metadata
    long                 framesizefull; // pixel data + metadata + kw

    PROCESSINFO *processinfo;
    if(dcprocinfo == 1)
    {
        // CREATE PROCESSINFO ENTRY
        // see processtools.c in module CommandLineInterface for details
        //
        char pinfoname[STRINGMAXLEN_FILENAME];
        snprintf(pinfoname, STRINGMAXLEN_FILENAME, "ntw-receive-%d", port);

        PROCESSINFO_AUX_SETUP(processinfo, pinfoname, "", "Waiting for input stream");
    }

    // CATCH SIGNALS
    stream_net_signal_catch();

    stream_net_rt_sched_set(RT_priority);

    // create UDP socket
    if((fds_server = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0)
    {
        PRINT_ERROR("creating socket");
        if(dcprocinfo == 1)
        {
            processinfo->loopstat = PROCESSINFO_LOOPSTAT_ERROR;
            processinfo_WriteMessage(processinfo, "ERROR creating socket");
        }
        free(imgmd);
        free(buff_udp);
        free(bigbuff_1MB);
        return -1;
    }

    memset((char *) &sock_server, 0, sizeof(sock_server));

    sock_server.sin_family      = AF_INET;
    sock_server.sin_port        = htons(port);
    sock_server.sin_addr.s_addr = htonl(INADDR_ANY);

    setsockopt(fds_server,
               SOL_SOCKET,
               SO_NO_CHECK,
               (char *) & flag,
               sizeof(flag));
    setsockopt(fds_server,
               SOL_SOCKET,
               SO_REUSEADDR,
               (char *) & flag,
               sizeof(flag));
    setsockopt(fds_server,
               SOL_SOCKET,
               SO_REUSEPORT,
               (char *) & flag,
               sizeof(flag));

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
        PRINT_ERROR("%s", msgstring);

        if(dcprocinfo == 1)
        {
            processinfo->loopstat = PROCESSINFO_LOOPSTAT_ERROR;
            processinfo_WriteMessage(processinfo, msgstring);
        }
        close(fds_server);
        free(imgmd);
        free(buff_udp);
        free(bigbuff_1MB);
        return -1;
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
            PRINT_ERROR("%s", msgstring);

            if(dcprocinfo == 1)
            {
                processinfo->loopstat = PROCESSINFO_LOOPSTAT_ERROR;
                processinfo_WriteMessage(processinfo, msgstring);
            }

            close(fds_server);
            free(imgmd);
            free(buff_udp);
            free(bigbuff_1MB);
            return -1;
        }

        // printf("Init phase: recvsize = %ld, buff_udp[0] = %d, buff_udp[1] = %d\n", recvsize, buff_udp[0], buff_udp[1]);

        // If this is a first datagram, we're having the metadata here:
        if(buff_udp[0] == MULTIGRAM_MAGIC && buff_udp[1] == 0)
        {
            __builtin_memcpy(imgmd, buff_udp + 2, sizeof(IMAGE_METADATA));
            break;
        }
    }

    if(dcprocinfo == 1)
    {
        char msgstring[200];
        snprintf(msgstring, 200, "Receiving stream %s", imgmd[0].name);
        processinfo_WriteMessage(processinfo, msgstring);
    }

    // is image already in memory ?
    OKim = 0;

    {
        IMGID img = imgid_make_from_name(
                        imgmd[0].name);
        resolveIMGID(
            &img,  ERRMODE_NULL,
            dcimg, dcnimg);
        ID = img.ID;
    }
    if(ID == -1)
    {
        // is it in shared memory ?
        ID = read_sharedmem_image(
                 imgmd[0].name, dcimg, dcnimg);
    }

    list_image_ID();

    if(ID == -1)
    {
        OKim = 0;
    }
    else
    {
        OKim = 1;
        if(imgmd[0].naxis != dcimg[ID].md[0].naxis)
        {
            OKim = 0;
        }
        if(OKim == 1)
        {
            for(int axis = 0; axis < imgmd[0].naxis; axis++)
                if(imgmd[0].size[axis] != dcimg[ID].md[0].size[axis])
                {
                    OKim = 0;
                }
        }
        if(imgmd[0].datatype != dcimg[ID].md[0].datatype)
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
        if(imgmd[0].NBkw != dcimg[ID].md[0].NBkw)
        {
            OKim = 0;
        }
    }

    if(OKim == 0)
    {
        printf("IMAGE %s HAS TO BE CREATED\n", imgmd[0].name);
        {
            IMGID imgrcv =
                imgid_make_from_name(
                    imgmd[0].name);
            imgrcv.mdt->naxis =
                imgmd[0].naxis;
            for(int a = 0;
                    a < imgmd[0].naxis; a++)
            {
                imgrcv.mdt->size[a] =
                    imgmd[0].size[a];
            }
            imgrcv.mdt->datatype =
                imgmd[0].datatype;
            imgrcv.mdt->shared =
                imgmd[0].shared;
            imgrcv.mdt->NBkw = nbkw;
            imgrcv.im =
                (IMAGE *) calloc(
                    1, sizeof(IMAGE));
            imgid_mkimage(&imgrcv);
            ID = imgrcv.ID;
        }
        printf("Created image stream %s - shared = %d\n",
               imgmd[0].name,
               imgmd[0].shared);
        printf("Size = %d,%d\n", imgmd[0].size[0], imgmd[0].size[1]);
    }
    else
    {
        printf("REUSING EXISTING IMAGE %s\n", imgmd[0].name);
    }

    xsize    = dcimg[ID].md[0].size[0];
    ysize    = dcimg[ID].md[0].size[1];
    NBslices = 1;
    if(dcimg[ID].md[0].naxis > 2)
        if(dcimg[ID].md[0].size[2] > 1)
        {
            NBslices = dcimg[ID].md[0].size[2];
        }

    if(dcprocinfo == 1)
    {
        char typestring[8];
        snprintf(typestring, 8, "%s",
                 ImageStreamIO_typename(dcimg[ID].md[0].datatype));
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
        ImageStreamIO_typesize(dcimg[ID].md[0].datatype) * xsize * ysize;
    printf("image frame size = %ld\n", framesize);

    ptr_dest_data_root = (char *) ImageStreamIO_get_image_d_ptr(&dcimg[ID]);

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

    {
        int total_udp_size = 3 * ((n_udp_dgrams - 1) * (DGRAM_CHUNK_SIZE + 2) +
                                  last_dgram_chunk + 2);
        setsockopt(fds_server, SOL_SOCKET, SO_SNDBUF, &total_udp_size,
                   sizeof(total_udp_size));
    }

    if(dcprocinfo == 1)
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
    int abort_frame = 1; // Initial sync

    while(loopOK == 1)
    {
        if(dcprocinfo == 1)
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

        if((dcprocinfo == 1) && (processinfo->MeasureTiming == 1))
        {
            processinfo_exec_start(processinfo);
        }

        // recvfrom should return a zero-th datagram.

        if(abort_frame)
        {
            // Purge buffer and resync to a 0-th datagram if necessary
            // This while will terminate when MSG_DONTWAIT causes a
            // errno = EAGAIN / EWOULDBLOCK, meaning the queue is empty.
            int sz;
            while((sz = recvfrom(fds_server, bigbuff_1MB, 1024 * 1024, MSG_DONTWAIT,
                                 (struct sockaddr *)&sock_client, &slen_client)) >= 0)
            {
                printf("Urrrrgh. -- %d\n", sz);
                fflush(stdout);
            }
            // Now give ourselves a chance to grab a clean 0-th datagram.
            for(int n_dgram_wait = 0; n_dgram_wait < MAX_DATAGRAM_WAIT; ++n_dgram_wait)
            {
                recvsize = recvfrom(fds_server, buff_udp, first_dgram_bytes, 0,
                                    (struct sockaddr *)&sock_client, &slen_client);
                if(recvsize < 0 || n_dgram_wait == MAX_DATAGRAM_WAIT - 1)
                {
                    PRINT_ERROR(
                        "recvfrom() @ A [%d - %s]",
                        errno, strerror(errno));
                    loopOK = 0;
                    socketOpen = 0;
                    break;
                }

                if(buff_udp[0] == MULTIGRAM_MAGIC && buff_udp[1] == 0)
                {
                    printf("-- Resync achieved after %d datagrams.\n", n_dgram_wait);
                    abort_frame = 0;
                    break;
                }
                else
                {
                    printf("MMMMMMhhhhhh.\n");
                    fflush(stdout);
                }
            }

        }
        else
        {
            // Normal operation
            if(recvfrom(fds_server, buff_udp, first_dgram_bytes, 0,
                        (struct sockaddr *)&sock_client, &slen_client) < 0)
            {
                PRINT_ERROR(
                    "recvfrom() @ B [%d - %s]",
                    errno, strerror(errno));
                loopOK = 0;
                socketOpen = 0;
                break;
            }
        }

        if(socketOpen == 1 && loopOK == 1)
        {
            if(buff_udp[0] == MULTIGRAM_MAGIC && buff_udp[1] == 0)
            {
                // 0-th datagram is legit, memcpy it.
                __builtin_memcpy(buff, buff_udp + 2, first_dgram_bytes - 2);
            }
            else
            {
                // abort frame and go again
                abort_frame = 1;
                printf("Aborting frame at datagram 0.\n");
                continue;
            }

            // Now we have the first datagram.

            // Weak copy although we now have all the metadata in buff
            imgmd_remote = (IMAGE_METADATA *)(ptr_buff_metadata);

            dcimg[ID].md[0].cnt1 =
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
                    PRINT_ERROR("recvfrom()");
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
                __builtin_memcpy(buff + k_dgram * DGRAM_CHUNK_SIZE,
                                 buff_udp + 2,
                                 this_dgram_bytes);
            }
        }
        if(socketOpen == 1 && abort_frame == 0)
        {

            // Copy the data !
            __builtin_memcpy(ptr_dest_data_sliceroot, ptr_buff_data, framesize);

            if(TCPTRANSFERKW == 1)
            {
                // copy kw
                __builtin_memcpy(dcimg[ID].kw,
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
                    dcimg[ID].md[0].cnt0,
                    dcimg[ID].md[0].cnt0 - moutputcnt);

                minputcnt  = imgmd_remote[0].cnt0;
                moutputcnt = dcimg[ID].md[0].cnt0;

                monitorloopindex++;
                monitorindex = 0;
            }

            monitorindex++;

            dcimg[ID].md[0].cnt0++;
            for(semnb = 0; semnb < dcimg[ID].md[0].sem; semnb++)
            {
                semval = ImageStreamIO_semvalue(dcimg + ID, semnb);
                if(semval < SEMAPHORE_MAXVAL)
                {
                    ImageStreamIO_sempost(dcimg + ID, semnb);
                }
            }

            sem_getvalue(dcimg[ID].semlog, &semval);
            if(semval < 2)
            {
                sem_post(dcimg[ID].semlog);
            }
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

        if(DCSIG_ANY_SET())
        {
            loopOK = 0;
            if(dcprocinfo == 1)
            {
                DCSIG_PROCESS_EXIT(processinfo);
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

    free(buff);
    free(buff_udp);

    //close(fds_client);

    printf("port %d closed\n", port);
    fflush(stdout);

    free(imgmd);

    return ID;
}
