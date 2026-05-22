/**
 * @file    stream_TCP_receive.c
 * @brief   TCP stream receive function
 *
 * Extracted from stream_TCP.c
 *
 * @see stream_TCP.c for transmit and FPS framework.
 */

#include <arpa/inet.h>
#include <math.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sched.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
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

static int TCPTRANSFERKW = 1;

typedef struct
{
    long magic;
    long cnt0;
    long cnt1;
} TCP_BUFFER_METADATA;

extern long FRAME_MD_MAGIC;

/** continuously receives 2D image through TCP link
 * mode = 1, force counter to be used for synchronization, ignore semaphores if they exist
 */

imageID COREMOD_MEMORY_image_NETWORKreceive(int                         port,
                                            __attribute__((unused)) int mode,
                                            int                         RT_priority)
{
    int fds_server;
    int fds_client;

    long totsize = 0;

    IMAGE_METADATA *imgmd = (IMAGE_METADATA *) malloc(sizeof(IMAGE_METADATA));
    PROCESSINFO    *processinfo;
    if (dcprocinfo == 1)
    {
        // CREATE PROCESSINFO ENTRY
        // see processtools.c in module CommandLineInterface for details
        //
        char pinfoname[200];
        snprintf(pinfoname, 200, "ntw-receive-%d", port);

        char msgstring[200];
        snprintf(msgstring, 200, "Waiting for input stream");

        PROCESSINFO_AUX_SETUP(processinfo, pinfoname, "", msgstring);
    }

    // CATCH SIGNALS
    stream_net_signal_catch();

    stream_net_rt_sched_set(RT_priority);

    // create TCP socket
    {
        struct sockaddr_in sock_server;
        int                flag = 1;
        int                result;
        int                MAXPENDING = 5;

        if ((fds_server = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP)) == -1)
        {
            PRINT_ERROR("creating socket");
            if (dcprocinfo == 1)
            {
                processinfo->loopstat = 4;
                processinfo_WriteMessage(processinfo, "ERROR creating socket");
            }
            free(imgmd);
            return -1;
        }

        memset((char *) &sock_server, 0, sizeof(sock_server));

        result = setsockopt(fds_server,     /* socket affected */
                            IPPROTO_TCP,    /* set option at TCP level */
                            TCP_NODELAY,    /* name of option */
                            (char *) &flag, /* the cast is historical cruft */
                            sizeof(flag));  /* length of option value */
        result -= setsockopt(fds_server, SOL_SOCKET, SO_REUSEADDR, (char *) &flag, sizeof(flag));
        result -= setsockopt(fds_server, SOL_SOCKET, SO_REUSEPORT, (char *) &flag, sizeof(flag));
        if (result < 0)
        {
            PRINT_ERROR("setsockopt");
            if (dcprocinfo == 1)
            {
                processinfo->loopstat = 4;
                processinfo_WriteMessage(processinfo, "ERROR socketopt");
            }
            close(fds_server);
            free(imgmd);
            return -1;
        }

        sock_server.sin_family      = AF_INET;
        sock_server.sin_port        = htons(port);
        sock_server.sin_addr.s_addr = htonl(INADDR_ANY);

        //bind socket to port
        if (bind(fds_server, (struct sockaddr *) &sock_server, sizeof(sock_server)) == -1)
        {
            char msgstring[200];

            snprintf(msgstring, 200, "ERROR binding socket, port %d", port);
            PRINT_ERROR("%s", msgstring);

            if (dcprocinfo == 1)
            {
                processinfo->loopstat = 4;
                processinfo_WriteMessage(processinfo, msgstring);
            }
            close(fds_server);
            free(imgmd);
            return -1;
        }

        if (listen(fds_server, MAXPENDING) < 0)
        {
            char msgstring[200];

            snprintf(msgstring, 200, "ERROR listen socket");
            PRINT_ERROR("%s", msgstring);

            if (dcprocinfo == 1)
            {
                processinfo->loopstat = 4;
                processinfo_WriteMessage(processinfo, msgstring);
            }

            close(fds_server);
            free(imgmd);
            return -1;
        }

        //    cnt = 0;

        struct sockaddr_in sock_client;
        socklen_t          slen_client = sizeof(sock_client);

        /* Wait for a client to connect */
        if ((fds_client = accept(fds_server, (struct sockaddr *) &sock_client, &slen_client)) == -1)
        {
            char msgstring[200];

            snprintf(msgstring, 200, "ERROR accept socket");
            PRINT_ERROR("%s", msgstring);

            if (dcprocinfo == 1)
            {
                processinfo->loopstat = 4;
                processinfo_WriteMessage(processinfo, msgstring);
            }

            close(fds_server);
            free(imgmd);
            return -1;
        }

        printf("Client connected\n");
        fflush(stdout);
    }

    // listen for image metadata
    long recvsize;
    if ((recvsize = recv(fds_client, imgmd, sizeof(IMAGE_METADATA), MSG_WAITALL)) < 0)
    {
        char msgstring[200];

        snprintf(msgstring, 200, "ERROR receiving image metadata");
        PRINT_ERROR("%s", msgstring);

        if (dcprocinfo == 1)
        {
            processinfo->loopstat = 4;
            processinfo_WriteMessage(processinfo, msgstring);
        }

        close(fds_client);
        close(fds_server);
        free(imgmd);
        return -1;
    }

    if (dcprocinfo == 1)
    {
        char msgstring[200];
        snprintf(msgstring, 200, "Receiving stream %s", imgmd->name);
        processinfo_WriteMessage(processinfo, msgstring);
    }

    // is image already in memory ?
    int     OKim = 0;
    imageID ID   = -1;

    {
        IMGID img = imgid_make_from_name(imgmd->name);
        resolveIMGID(&img, ERRMODE_NULL, dcimg, dcnimg);
        ID = img.ID;
    }
    printf("ID: %ld\n", ID);

    if (ID == -1)
    {
        // is it in shared memory ?
        ID = read_sharedmem_image(imgmd->name, dcimg, dcnimg);
        printf("ID: %ld\n", ID);
    }

    // img_p = &dcimg[ID]; // Of course that doesn't fucking work if ID is -1.

    list_image_ID();

    IMAGE *img_p = NULL;
    if (ID == -1)
    {
        OKim = 0;
    }
    else
    {
        img_p = &dcimg[ID];
        OKim  = 1;
        if (imgmd->naxis != img_p->md->naxis)
        {
            OKim = 0;
        }
        if (OKim == 1)
        {
            for (int axis = 0; axis < imgmd->naxis; axis++)
            {
                if (imgmd->size[axis] != img_p->md->size[axis])
                {
                    OKim = 0;
                }
            }
        }
        if (imgmd->datatype != img_p->md->datatype)
        {
            OKim = 0;
        }

        if (TCPTRANSFERKW == 1 && imgmd->NBkw > img_p->md->NBkw)
        {
            OKim = 0;
        }

        if (OKim == 0)
        {
            // This actually nukes img_p, but leaves imgmd unscathed.
            delete_image_ID(imgmd->name, DELETE_IMAGE_ERRMODE_WARNING);
            ID = -1;
        }
    }

    if (OKim == 0)
    {
        printf("IMAGE %s HAS TO BE CREATED\n", imgmd->name);
        fflush(stdout);
        {
            IMGID imgrcv      = imgid_make_from_name(imgmd->name);
            imgrcv.mdt->naxis = imgmd->naxis;
            for (int a = 0; a < imgmd->naxis; a++)
            {
                imgrcv.mdt->size[a] = imgmd->size[a];
            }
            imgrcv.mdt->datatype = imgmd->datatype;
            imgrcv.mdt->shared   = imgmd->shared;
            imgrcv.mdt->NBkw     = imgmd->NBkw;
            imgrcv.im            = (IMAGE *) calloc(1, sizeof(IMAGE));
            imgid_mkimage(&imgrcv);
            ID = imgrcv.ID;
        }
        printf("Created image stream %s - shared = %d\n", imgmd->name, imgmd->shared);
        printf("Size = %d,%d\n", imgmd->size[0], imgmd->size[1]);
        // OKim is now OK. Re-point img_p
        img_p = &dcimg[ID];
    }
    else
    {
        printf("REUSING EXISTING IMAGE %s\n", imgmd->name);
    }

    uint32_t xsize    = img_p->md->size[0];
    uint32_t ysize    = img_p->md->size[1];
    long     NBslices = 1;
    if (img_p->md->naxis > 2)
    {
        if (img_p->md->size[2] > 1)
        {
            NBslices = img_p->md->size[2];
        }
    }

    char typestring[8];

    if (ImageStreamIO_checktype(img_p->md->datatype, 0) == -1)
    {
        PRINT_ERROR("wrong data type %d", (int) img_p->md->datatype);
        snprintf(typestring, 8, "%s", "ERR");
        close(fds_client);
        close(fds_server);
        free(imgmd);
        return -1;
    }
    long framesize = ImageStreamIO_typesize(img_p->md->datatype) * xsize * ysize;
    printf("image frame size = %ld\n", framesize);

    snprintf(typestring, 8, "%s", ImageStreamIO_typename(img_p->md->datatype));

    char *ptr0 = (char *) img_p->array.raw;

    if (dcprocinfo == 1)
    {
        char msgstring[200];
        snprintf(msgstring, 200, "<- %s [%d x %d x %ld] %s", imgmd->name, (int) xsize, (int) ysize,
                 NBslices, typestring);
        snprintf(processinfo->description, 200, "%s %dx%dx%ld %s", imgmd->name, (int) xsize,
                 (int) ysize, NBslices, typestring);
        processinfo_WriteMessage(processinfo, msgstring);
    }

    // this line is not needed, as frame_md is declared below
    // frame_md = (TCP_BUFFER_METADATA*) malloc(sizeof(TCP_BUFFER_METADATA));

    long framesize1 = framesize + sizeof(TCP_BUFFER_METADATA);
    long framesizefull;
    if (TCPTRANSFERKW == 0)
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

    char *buff = (char *) malloc(sizeof(char) * framesizefull);

    TCP_BUFFER_METADATA *frame_md_p = (TCP_BUFFER_METADATA *) (buff + framesize);

    if (dcprocinfo == 1)
    {
        processinfo->loopstat = 1; //notify processinfo that we are entering loop
    }

    {
        int  socketOpen = 1; // 0 if socket is closed
        long loopcnt    = 0;
        int  loopOK     = 1;

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
            char *socket_flush_buff = (char *) malloc(framesizefull);
            long  recv_bytes        = framesizefull;
            while (recv_bytes == framesizefull)
            {
                recv_bytes = recv(fds_client, socket_flush_buff, framesizefull, MSG_DONTWAIT);
                printf("TCP recv buffer flush. %ld stray bytes.\n", recv_bytes);
            }
            if (recv_bytes > 0) // Will be -1 if we got 0 bytes at the last iteration above
            {
                recv_bytes =
                    recv(fds_client, socket_flush_buff, framesizefull - recv_bytes, MSG_WAITALL);
                printf("Buffer flush finalize. %ld extra bytes.\n", recv_bytes);
            }
            free(socket_flush_buff);
        }

        while (loopOK == 1)
        {
            if (dcprocinfo == 1)
            {
                while (processinfo->CTRLval == 1) // pause
                {
                    usleep(50);
                }

                if (processinfo->CTRLval == 2) // single iteration
                {
                    processinfo->CTRLval = 1;
                }

                if (processinfo->CTRLval == 3) // exit loop
                {
                    loopOK = 0;
                }
            }

            if ((recvsize = recv(fds_client, buff, framesizefull, MSG_WAITALL)) < 0)
            {
                PRINT_ERROR("recv()");
                socketOpen = 0;
            }

            if ((dcprocinfo == 1) && (processinfo->MeasureTiming == 1))
            {
                processinfo_exec_start(processinfo);
            }

            if (recvsize != 0)
            {
                totsize += recvsize;
            }
            else
            {
                socketOpen = 0;
            }

            if (socketOpen == 1)
            {
                frame_md_p = (TCP_BUFFER_METADATA *) (buff + framesize);
                if (frame_md_p->magic != FRAME_MD_MAGIC)
                {
                    printf("Bad magic! Looping fast.\n");
                    continue;
                }

                img_p->md->write = 1;
                img_p->md->cnt1  = frame_md_p->cnt1;

                // copy pixel data
                if (NBslices > 1)
                {
                    __builtin_memcpy(ptr0 + framesize * frame_md_p->cnt1, buff, framesize);
                }
                else
                {
                    __builtin_memcpy(ptr0, buff, framesize);
                }

                if (TCPTRANSFERKW == 1)
                {
                    // copy kw
                    __builtin_memcpy(img_p->kw, (IMAGE_KEYWORD *) (buff + framesize1),
                                     img_p->md->NBkw * sizeof(IMAGE_KEYWORD));
                }

                frameincr = (long) frame_md_p->cnt0 - cnt0previous;
                if (frameincr > 1)
                {
                    printf("Skipped %ld frame(s) at index %ld %ld\n", frameincr - 1,
                           (long) (frame_md_p->cnt0), (long) (frame_md_p->cnt1));
                }

                cnt0previous = frame_md_p->cnt0;

                if (monitorindex == monitorinterval)
                {
                    printf("[%5ld]  input %20ld (+ %8ld) output %20ld (+ "
                           "%8ld)\n",
                           monitorloopindex, frame_md_p->cnt0, frame_md_p->cnt0 - minputcnt,
                           img_p->md->cnt0, img_p->md->cnt0 - moutputcnt);

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

            if (socketOpen == 0)
            {
                loopOK = 0;
            }

            if ((dcprocinfo == 1) && (processinfo->MeasureTiming == 1))
            {
                processinfo_exec_end(processinfo);
            }

            // process signals
            if (DCSIG_ANY_SET())
            {
                loopOK = 0;
                if (dcprocinfo == 1)
                {
                    DCSIG_PROCESS_EXIT(processinfo);
                }
            }

            loopcnt++;
            if (dcprocinfo == 1)
            {
                processinfo->loopcnt = loopcnt;
            }
        }

        if (dcprocinfo == 1)
        {
            processinfo_cleanExit(processinfo);
        }
    } // closes block around loop

    free(buff);

    close(fds_client);

    printf("port %d closed\n", port);
    fflush(stdout);

    free(imgmd);

    return ID;
}
