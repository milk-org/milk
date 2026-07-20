// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

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
#include "COREMOD_tools/mvprocCPUset.h"

// set to 1 if transfering keywords
static int SEND_KEYWORDS = 1;

typedef struct
{
    /*
    This struct contains everything from IMAGE_METADATA
    that is not memory-internal to the sender and receiver
    but inherently related to the data and needs to be carried along.
    */
    long magic;
    long cnt0;
    long slice;
    long nbkw;
} NETWORK_HEADER;

static long MAGIC_FRAME_METADATA = 0x12341234ff; // Doesn't matter.
static uint8_t MAGIC_UDP_MULTIGRAMS =
    0x3E; // Random magic to start datagrams with.
static int UDP_DGRAM_CHUNK_SIZE = 62 *
                                  1024; // Max payload per datagram, just shy of the maximum 65507 bytes

// ==========================================
// Forward declaration(s)
// ==========================================

errno_t COREMOD_MEMORY_testfunction_semaphore(const char *IDname,
        int semtrig,
        int testmode);

imageID COREMOD_MEMORY_image_NETWORKtransmit(const char *IDname,
        const char *IPaddr,
        int port,
        int mode,
        int RT_priority);

imageID COREMOD_MEMORY_image_NETWORKreceive(int port, int do_counter_sync,
        int RT_priority);

imageID COREMOD_MEMORY_image_NETUDPtransmit(const char *IDname,
        const char *IPaddr,
        int port,
        int do_counter_sync,
        int RT_priority);

imageID COREMOD_MEMORY_image_NETUDPreceive(int port,
        int do_counter_sync,
        int RT_priority);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

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

errno_t stream__NETW_addCLIcmd()
{
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

int initialize_endpoints_tcp(
    const char *IPaddr,
    const int port,
    int *p_fds_client,
    struct sockaddr_in *sock_server,
    PROCESSINFO *pinfo)
{
    int flag = 1;

    if((*p_fds_client = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP)) < 0)
    {
        printf("ERROR creating socket\n");
        exit(0);
    }

    int r = setsockopt(*p_fds_client, /* socket affected */
                       IPPROTO_TCP,   /* set option at TCP level */
                       TCP_NODELAY,   /* name of option */
                       (char *)&flag, /* the cast is historical cruft */
                       sizeof(int));  /* length of option value */

    if(r < 0)
    {
        processinfo_error(pinfo, "ERROR: setsockopt() failed\n");
        return -1;
    }

    memset((char *)sock_server, 0, sizeof(*sock_server));
    sock_server->sin_family = AF_INET;
    sock_server->sin_port = htons(port);
    sock_server->sin_addr.s_addr = inet_addr(IPaddr);

    if(connect(*p_fds_client,
               (struct sockaddr *)sock_server,
               sizeof(*sock_server)) < 0)
    {
        perror("Error  connect() failed ");
        printf("port = %d\n", port);
        processinfo_error(pinfo, "ERROR: connect() failed\n");
        return -1;
    }
    return 0;
}

int initialize_endpoints_udp(
    const char *IPaddr,
    const int port,
    int *p_fds_client,
    struct sockaddr_in *sock_server,
    PROCESSINFO *pinfo)
{
    int flag = 1;

    if((*p_fds_client = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0)
    {
        printf("ERROR creating UDP socket\n");
        processinfo_error(pinfo, "ERROR creating UDP socket\n");
        return -1;
    }

    int r = setsockopt(*p_fds_client, SOL_SOCKET, SO_REUSEADDR, (char *)&flag,
                       sizeof(flag));
    r -= setsockopt(*p_fds_client, SOL_SOCKET, SO_REUSEPORT, (char *)&flag,
                    sizeof(flag));

#ifdef SO_ATTACH_REUSEPORT_CBPF
    r -= setsockopt(*p_fds_client, SOL_SOCKET, SO_ATTACH_REUSEPORT_CBPF,
                    (char *)&flag,
                    sizeof(flag));
#endif
    if(r < 0)
    {
        processinfo_error(pinfo, "ERROR: setsockopt() failed\n");
        return -1;
    }

    memset((char *)sock_server, 0, sizeof(*sock_server));
    sock_server->sin_family = AF_INET;
    sock_server->sin_port = htons(port);
    sock_server->sin_addr.s_addr = inet_addr(IPaddr);

    if(connect(*p_fds_client,
               (struct sockaddr *)sock_server,
               sizeof(*sock_server)) < 0)
    {
        perror("Error UDP connect() failed ");
        printf("port = %d\n", port);
        processinfo_error(pinfo, "ERROR: UDP connect() failed\n");
        return -1;
    }
    return 0;
}

long send_TCP(int fds_client, int num_chunks, char **chunks_ptrs,
              uint64_t *chunk_sizes)
{
    // SEND
    long _sent_bytes = 0;

    // Find last chunk with non-zero size; if all are zero send nothing.
    int last_nonzeroindex = num_chunks - 1;
    while(last_nonzeroindex >= 0 && chunk_sizes[last_nonzeroindex] == 0)
    {
        last_nonzeroindex--;
    }

    for(int ii = 0; ii <= last_nonzeroindex; ++ii)
    {
        if(chunk_sizes[ii] > 0)
            _sent_bytes += send(fds_client, chunks_ptrs[ii], (size_t)chunk_sizes[ii],
                                ii == last_nonzeroindex ? 0 : MSG_MORE);
    }

    return _sent_bytes;
}

long send_UDP(int fds_client, struct sockaddr_in sock_server, int num_chunks,
              char **chunks_ptrs, uint64_t *chunk_sizes)
{
    long _sent_bytes = 0;

    int last_nonzeroindex = num_chunks - 1;
    while(last_nonzeroindex >= 0 && chunk_sizes[last_nonzeroindex] == 0)
    {
        last_nonzeroindex--;
    }

    uint64_t total_size = 0;
    for(int ii = 0; ii < num_chunks; ++ii)
    {
        total_size += chunk_sizes[ii];
    }

    // Prepare segmentation into 62k datagrams
    int new_dgram = 1;
    int dgram_count = 0;
    uint8_t dgram_header[2] = {(uint8_t)MAGIC_UDP_MULTIGRAMS, 0};
    uint64_t chunk_remaining_bytes = 0;
    uint64_t dgram_remaining_bytes = (uint64_t)UDP_DGRAM_CHUNK_SIZE;

    const struct sockaddr *pss = (const struct sockaddr *)&sock_server;
    socklen_t psss = sizeof(sock_server);

    for(int chunk = 0; chunk < num_chunks; ++chunk)
    {
        if(chunk_sizes[chunk] == 0)
        {
            continue;
        }

        chunk_remaining_bytes = chunk_sizes[chunk];
        const char *src = chunks_ptrs[chunk];

        while(chunk_remaining_bytes > 0)
        {
            if(new_dgram)
            {
                sendto(fds_client, dgram_header, 2, MSG_MORE, pss, psss);
                new_dgram = 0;
            }

            uint64_t to_send = chunk_remaining_bytes < dgram_remaining_bytes
                               ? chunk_remaining_bytes
                               : dgram_remaining_bytes;

            // Flush this datagram if it is now full, or if this is the last byte of all data.
            int closes_dgram = (to_send == dgram_remaining_bytes)
                               || ((uint64_t)_sent_bytes + to_send == total_size);

            sendto(fds_client, src, (size_t)to_send,
                   closes_dgram ? 0 : MSG_MORE, pss, psss);

            _sent_bytes += (long)to_send;
            src += to_send;
            chunk_remaining_bytes -= to_send;
            dgram_remaining_bytes -= to_send;

            if(closes_dgram)
            {
                dgram_count++;
                dgram_header[1] = (uint8_t)dgram_count;
                dgram_remaining_bytes = (uint64_t)UDP_DGRAM_CHUNK_SIZE;
                new_dgram = 1;
            }
        }
    }

    return _sent_bytes;
}

imageID common_NETWORKtransmit(
    const char *IDname, const char *IPaddr, int port, int do_counter_sync,
    int RT_priority,
    int use_TCP)
{
    imageID ID = -1;
    IMAGE *img_p;
    char errmsg[200];

    NETWORK_HEADER header = {MAGIC_FRAME_METADATA, 0, 0, 0};
    int fds_client;
    struct sockaddr_in sock_server;

    int nb_slices;

    int use_sem = 0;
    int iter = 0;
    int sem_trig_id = -1;

    DEBUG_TRACEPOINT(" ");

    // ===========================
    // processinfo support
    // ===========================
    PROCESSINFO *processinfo;

    char pinfoname[STRINGMAXLEN_FILENAME];
    snprintf(pinfoname, STRINGMAXLEN_FILENAME, "ntw-tx-%d-%s", use_TCP, IDname);

    char descr[200];
    snprintf(descr, 200, "%s->%s/%d", IDname, IPaddr, port);

    printf("Setup processinfo ...");
    fflush(stdout);
    processinfo = processinfo_setup(pinfoname,
                                    descr,   // description
                                    "setup", // message on startup
                                    __FUNCTION__,
                                    __FILE__,
                                    __LINE__);
    printf(" done\n");
    fflush(stdout);

    // OPTIONAL SETTINGS
    processinfo->MeasureTiming = 1; // Measure timing
    processinfo->RT_priority = RT_priority;

    int loopOK = 1; // Master flag

    ID = image_ID(IDname);
    img_p = &data.image[ID];

    if(use_TCP)
    {
        loopOK = (0 == initialize_endpoints_tcp(IPaddr, port, &fds_client, &sock_server,
                                                processinfo));
    }
    else
    {
        loopOK = (0 == initialize_endpoints_udp(IPaddr, port, &fds_client, &sock_server,
                                                processinfo));
    }
    if(!loopOK)
    {
        goto cleanup;
    }

    nb_slices = img_p->md->naxis > 2 ? img_p->md->size[2] : 1;

    uint64_t size_img_data = ImageStreamIO_typesize(img_p->md->datatype) *
                             img_p->md->size[0] * img_p->md->size[1];
    printf("IMAGE FRAME SIZE = %ld\n", size_img_data);
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
        goto cleanup;
    }

    // Prepare to send
    char *ptr_header = (char *)&header;
    uint64_t size_header = sizeof(NETWORK_HEADER);
    char *ptr_metadata = (char *)img_p->md;
    uint64_t size_metadata = sizeof(IMAGE_METADATA);
    char *ptr_img_data = (char *)ImageStreamIO_get_image_d_ptr(&data.image[ID]);
    // size_image_data above
    char *ptr_img_keywords = (char *)img_p->kw; // CAN BE NULL
    uint64_t size_img_keywords = SEND_KEYWORDS ? img_p->md->NBkw * sizeof(
                                     IMAGE_KEYWORD)
                                 : 0L;

    header.nbkw = SEND_KEYWORDS ? img_p->md->NBkw : 0;

    char *pointers_to_send[4] = {ptr_header, ptr_metadata, ptr_img_data, ptr_img_keywords};
    uint64_t sizes_to_send[4] = {size_header, size_metadata, size_img_data, size_img_keywords};

    uint64_t total_bytes_to_transfer = size_header + size_metadata + size_img_data +
                                       size_img_keywords;

    if((img_p->md->sem == 0) || (do_counter_sync == 1))
    {
        processinfo_WriteMessage(processinfo, "sync using counter");
        use_sem = 0;
    }
    else
    {
        sem_trig_id = ImageStreamIO_getsemwaitindex(img_p, -1);
        use_sem = 1;
        char msgstring[200];
        snprintf(msgstring, 200, "sync using semaphore %d", sem_trig_id);
        processinfo_WriteMessage(processinfo, msgstring);
    }
    // ===========================
    // Start loop
    // ===========================
    processinfo_loopstart(processinfo);

    long _last_sent_cnt0 = 0;
    long _last_sent_slice = 0;
    long _now_cnt0 = 0;
    long _sem_rval = 0;
    long _frame_incr = 0;
    long _sending_slice = 0;
    long _sent_bytes = 0;

    struct timespec ts = {0};

    while(loopOK == 1)
    {
        loopOK = processinfo_loopstep(processinfo);

        if(use_sem == 0)  // use counter
        {
            while(img_p->md->cnt0 == _last_sent_cnt0)  // test if new frame exists
            {
                usleep(5);
            }
            _last_sent_cnt0 = img_p->md->cnt0;
            _sem_rval = 0;
        }
        else
        {
            if(clock_gettime(CLOCK_MILK, &ts) == -1)
            {
                perror("clock_gettime");
                exit(EXIT_FAILURE);
            }
            ts.tv_sec += 2;

            _sem_rval = ImageStreamIO_semtimedwait(img_p, sem_trig_id, &ts);

            if(iter == 0)
            {
                processinfo_WriteMessage(processinfo, "Driving sem to 0");
                printf("Driving semaphore to zero ... ");
                fflush(stdout);
                ImageStreamIO_semflush(img_p, sem_trig_id);
                iter++;
            }
        }

        processinfo_exec_start(processinfo);
        if(processinfo_compute_status(processinfo) != 1 || _sem_rval != 0)
        {
            goto loop_cleanup;
        }

        // DO SOME PREPARATION.
        _now_cnt0 = img_p->md->cnt0;
        _frame_incr = _now_cnt0 - _last_sent_cnt0;
        _sending_slice = (_last_sent_slice + 1) %
                         nb_slices; // May cause stall / catch-up errors for nb_slices > 1

        header.cnt0 = _now_cnt0;
        header.slice = _sending_slice;

        pointers_to_send[2] = ptr_img_data + size_img_data * _sending_slice;

        if(use_TCP)
        {
            _sent_bytes = send_TCP(fds_client, 4, pointers_to_send, sizes_to_send);
        }
        else
        {
            _sent_bytes = send_UDP(fds_client, sock_server, 4, pointers_to_send,
                                   sizes_to_send);
        }

        if(_sent_bytes != total_bytes_to_transfer)
        {
            perror("socket send error ");
            snprintf(errmsg,
                     200,
                     "ERROR: send() sent a different "
                     "number of bytes (%ld) than expected %ld",
                     _sent_bytes,
                     (long)total_bytes_to_transfer);
            printf("%s\n", errmsg);
            fflush(stdout);
            processinfo_WriteMessage(processinfo, errmsg);
        }

        if(img_p->md->cnt0 != _now_cnt0)
        {
            printf("cnt0 incremented during send: %ld -> %ld\n", _now_cnt0,
                   img_p->md->cnt0);
        }

        _last_sent_cnt0 = img_p->md->cnt0;
        _last_sent_slice = _sending_slice;

loop_cleanup:
        // process signals, increment loop counter
        processinfo_exec_end(processinfo);

        if((data.signal_INT == 1) || (data.signal_TERM == 1) ||
                (data.signal_ABRT == 1) || (data.signal_BUS == 1) ||
                (data.signal_SEGV == 1) || (data.signal_HUP == 1) ||
                (data.signal_PIPE == 1))
        {
            loopOK = 0;
        }
    } // while loopOK

cleanup:
    // ==================================
    // ENDING LOOP
    // ==================================
    processinfo_cleanExit(processinfo);

    // free(buff);

    close(fds_client);
    printf("port %d closed\n", port);
    fflush(stdout);

    return ID;
}

imageID COREMOD_MEMORY_image_NETWORKtransmit(
    const char *IDname, const char *IPaddr, int port, int do_counter_sync,
    int RT_priority)
{
    printf("Transmit stream %s over TCP/IP %s port %d\n", IDname, IPaddr, port);
    fflush(stdout);
    return common_NETWORKtransmit(IDname, IPaddr, port, do_counter_sync,
                                  RT_priority, 1);
}

imageID COREMOD_MEMORY_image_NETUDPtransmit(const char *IDname,
        const char *IPaddr, int port,
        int do_counter_sync,
        int RT_priority)
{
    printf("Transmit stream %s over UDP/IP %s port %d\n", IDname, IPaddr, port);
    fflush(stdout);
    return common_NETWORKtransmit(IDname, IPaddr, port, do_counter_sync,
                                  RT_priority, 0);
}

// ==========================================
// Shared receive helpers
// ==========================================

/**
 * @brief Validate or (re)create the destination IMAGE from received metadata.
 *
 * Called on the first frame of every connection.  If the existing shared-memory
 * image is compatible with @p md it is reused; otherwise it is deleted and
 * recreated.  On return *p_img is valid and *p_framesize / *p_NBslices are set.
 *
 * @return 0 on success, -1 on unrecoverable error (bad datatype).
 */
static imageID ensure_recv_image(IMAGE_METADATA *md,
                                 long nbkw_wire)
{
    int axis;
    imageID ID = image_ID(md->name);

    if(ID == -1)
    {
        ID = read_sharedmem_image(md->name);
    }

    list_image_ID();

    int ok = (ID != -1);
    if(ok)
    {
        IMAGE *im = &data.image[ID];
        if(md->naxis != im->md->naxis)
        {
            ok = 0;
        }
        if(ok)
            for(axis = 0; axis < (int)md->naxis; axis++)
                if(md->size[axis] != im->md->size[axis])
                {
                    ok = 0;
                    break;
                }
        if(md->datatype != im->md->datatype)
        {
            ok = 0;
        }
        if(SEND_KEYWORDS && nbkw_wire > im->md->NBkw)
        {
            ok = 0;
        }

        if(!ok)
        {
            delete_image_ID(md->name, DELETE_IMAGE_ERRMODE_WARNING);
            ID = -1;
        }
    }

    if(!ok)
    {
        printf("IMAGE %s HAS TO BE CREATED\n", md->name);
        fflush(stdout);
        create_image_ID(md->name, md->naxis, md->size, md->datatype,
                        md->shared, (int)nbkw_wire, 0, &ID);
        printf("Created image stream %s  shared=%d  size=%d x %d\n",
               md->name, md->shared, md->size[0], md->size[1]);
    }
    else
    {
        printf("REUSING EXISTING IMAGE %s\n", md->name);
    }
    IMAGE *im = &data.image[ID];

    if(ImageStreamIO_checktype(im->md->datatype, 0) == -1)
    {
        printf("ERROR: WRONG DATA TYPE\n");
        return -1;
    }

    return ID;
}

// ==========================================
// Receiver socket helpers
// ==========================================

static int initialize_receiving_endpoints_tcp(int port,
        PROCESSINFO *processinfo, int *p_fds_client)
{
    int flag = 1;
    int result;
    int MAXPENDING = 5;
    struct sockaddr_in sock_server;
    struct sockaddr_in sock_client;
    socklen_t slen_client;
    int fds_server;
    int fds_client;

    if((fds_server = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP)) == -1)
    {
        processinfo_WriteMessage(processinfo, "ERROR creating socket");
        return -1;
    }

    memset((char *)&sock_server, 0, sizeof(sock_server));
    result = setsockopt(fds_server, IPPROTO_TCP, TCP_NODELAY,
                        (char *)&flag, sizeof(flag));
    result -= setsockopt(fds_server, SOL_SOCKET, SO_REUSEADDR,
                         (char *)&flag, sizeof(flag));
    result -= setsockopt(fds_server, SOL_SOCKET, SO_REUSEPORT,
                         (char *)&flag, sizeof(flag));
    if(result < 0)
    {
        processinfo_WriteMessage(processinfo, "ERROR setsockopt");
        return -1;
    }

    sock_server.sin_family = AF_INET;
    sock_server.sin_port = htons(port);
    sock_server.sin_addr.s_addr = htonl(INADDR_ANY);

    if(bind(fds_server, (struct sockaddr *)&sock_server, sizeof(sock_server)) == -1)
    {
        char m[200];
        snprintf(m, 200, "ERROR bind port %d", port);
        processinfo_WriteMessage(processinfo, m);
        return -1;
    }
    if(listen(fds_server, MAXPENDING) < 0)
    {
        processinfo_WriteMessage(processinfo, "ERROR listen");
        return -1;
    }

    slen_client = sizeof(sock_client);
    if((fds_client = accept(fds_server,
                            (struct sockaddr *)&sock_client, &slen_client)) == -1)
    {
        processinfo_WriteMessage(processinfo, "ERROR accept");
        return -1;
    }
    printf("TCP client connected on port %d\n", port);
    fflush(stdout);

    *p_fds_client = fds_client;
    return 0;
}

static int initialize_receiving_endpoints_udp(int port,
        PROCESSINFO *processinfo, int *p_fds_server)
{
    int flag = 1;
    struct sockaddr_in sock_server;
    int fds_server;

    if((fds_server = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0)
    {
        processinfo_WriteMessage(processinfo, "ERROR creating UDP socket");
        return -1;
    }

    memset(&sock_server, 0, sizeof(sock_server));
    sock_server.sin_family = AF_INET;
    sock_server.sin_port = htons(port);
    sock_server.sin_addr.s_addr = htonl(INADDR_ANY);

    setsockopt(fds_server, SOL_SOCKET, SO_REUSEADDR, &flag, sizeof(flag));
    setsockopt(fds_server, SOL_SOCKET, SO_REUSEPORT, &flag, sizeof(flag));
#ifdef SO_ATTACH_REUSEPORT_CBPF
    setsockopt(fds_server, SOL_SOCKET, SO_ATTACH_REUSEPORT_CBPF, &flag,
               sizeof(flag));
#endif

    // The sender delivers all datagrams of one frame nearly simultaneously via
    // MSG_MORE corking.  The default SO_RCVBUF (~208 KB on Linux) holds only ~3
    // datagrams of 62 KB each; anything beyond that is silently dropped by the
    // kernel before the application can drain them, regardless of frame rate.
    // Request a buffer large enough for at least 64 full datagrams (~4 MB).
    // The kernel silently caps this at net.core.rmem_max; use SO_RCVBUFFORCE
    // (requires CAP_NET_ADMIN) as a fallback to override that limit.
    {
        int rcvbuf = 64 * (UDP_DGRAM_CHUNK_SIZE + 2);
        if(setsockopt(fds_server, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf)) < 0)
        {
#ifdef SO_RCVBUFFORCE
            setsockopt(fds_server, SOL_SOCKET, SO_RCVBUFFORCE, &rcvbuf, sizeof(rcvbuf));
#endif
        }
    }

    if(bind(fds_server, (struct sockaddr *)&sock_server, sizeof(sock_server)) == -1)
    {
        char m[200];
        snprintf(m, 200, "ERROR bind UDP port %d", port);
        processinfo_WriteMessage(processinfo, m);
        return -1;
    }

    printf("UDP receiver listening on port %d\n", port);
    fflush(stdout);

    *p_fds_server = fds_server;
    return 0;
}

static imageID initial_receive_and_initialize_image_TCP(int fds_endpoint,
        int *nbkw_wire)
{

    uint8_t buffer[sizeof(NETWORK_HEADER) + sizeof(IMAGE_METADATA)] = {0};
    int expected_size = sizeof(NETWORK_HEADER) + sizeof(IMAGE_METADATA);

    NETWORK_HEADER *hdr_tmp = (NETWORK_HEADER *)buffer;
    IMAGE_METADATA *md_tmp = (IMAGE_METADATA *)(buffer + sizeof(NETWORK_HEADER));

    while(1)
    {
        int r = recv(fds_endpoint, buffer,
                     sizeof(NETWORK_HEADER) + sizeof(IMAGE_METADATA), MSG_WAITALL);
        if(r < 0)
        {
            printf("Connection closed during initial header recv\n");
            return -1;
        }
        if(hdr_tmp->magic != MAGIC_FRAME_METADATA)
        {
            while(recv(fds_endpoint, buffer, expected_size, MSG_DONTWAIT) > 0)
            {
            } // Purge buffers
        }
        else
        {
            break; // Receive is valid
        }
    }
    *nbkw_wire = hdr_tmp->nbkw;
    return ensure_recv_image(md_tmp, hdr_tmp->nbkw);
}

static imageID initial_receive_and_initialize_image_UDP(int fds_endpoint,
        int *nbkw_wire)
{
    // Buffer large enough for the 2-byte datagram header + NETWORK_HEADER + IMAGE_METADATA.
    static const size_t expected_size = 2 + sizeof(NETWORK_HEADER) + sizeof(
                                            IMAGE_METADATA);
    char buff[2 + sizeof(NETWORK_HEADER) + sizeof(IMAGE_METADATA)];
    struct sockaddr_in src_addr;
    socklen_t src_len = sizeof(src_addr);

    NETWORK_HEADER *hdr_tmp = (NETWORK_HEADER *)(buff + 2);
    IMAGE_METADATA *md_tmp = (IMAGE_METADATA *)(buff + 2 + sizeof(NETWORK_HEADER));

    // Block until we receive a valid seq-0 datagram carrying NETWORK_HEADER+IMAGE_METADATA.
    while(1)
    {
        long r = recvfrom(fds_endpoint, buff, sizeof(buff),
                          MSG_WAITALL, (struct sockaddr *)&src_addr, &src_len);
        if(r < 0)
        {
            printf("ERROR recvfrom during initial UDP recv\n");
            return -1;
        }

        // Must be a seq-0 datagram.
        if((uint8_t)buff[0] != MAGIC_UDP_MULTIGRAMS || (uint8_t)buff[1] != 0
                || hdr_tmp->magic != MAGIC_FRAME_METADATA)
            while(recv(fds_endpoint, buff, expected_size, MSG_DONTWAIT) > 0)
            {
            } // Purge buffers
        else
        {
            break;
        }
    }
    *nbkw_wire = hdr_tmp->nbkw;
    return ensure_recv_image(md_tmp, hdr_tmp->nbkw);
}

// ==========================================
// Frame receive helpers
// ==========================================

/**
 * @brief Receive one full TCP frame into @p buff.
 *
 * Loops until a frame with valid NETWORK_HEADER magic is received.
 * Purges the kernel buffer on magic mismatch before retrying.
 *
 * @return framesizefull on success, -1 if the connection is closed or errored.
 */
static long recv_buffer_tcp(int fds, char *buff, long framesizefull)
{
    NETWORK_HEADER *hdr = (NETWORK_HEADER *)buff;
    while(1)
    {
        long r = recv(fds, buff, (size_t)framesizefull, MSG_WAITALL);
        if(r <= 0)
        {
            return -1;
        }

        if(hdr->magic != MAGIC_FRAME_METADATA)
        {
            printf("TCP bad magic — purging kernel buffer\n");
            while(recv(fds, buff, (size_t)framesizefull, MSG_DONTWAIT) > 0)
            {
            }
            continue;
        }
        return r;
    }
}

/**
 * @brief Receive one full UDP frame into @p buff from datagrams.
 *
 * Reassembles n_dgrams datagrams of the form [magic(1)|seq(1)|payload(<=CHUNK)].
 * Uses recvmsg scatter-gather to write payload bytes directly into @p buff
 * at the correct offset without any intermediate allocation.
 * Retries on bad seq-0 datagram; returns -1 on socket error or mid-frame seq error.
 *
 * @return framesizefull on success, -1 on error.
 */
static long recv_buffer_udp(int fds, char *buff, char *dgram_buff,
                            long framesizefull)
{
    int n_dgrams = (int)((framesizefull + (long)UDP_DGRAM_CHUNK_SIZE - 1) /
                         (long)UDP_DGRAM_CHUNK_SIZE);
    struct sockaddr_in src;

    uint8_t hdr0[2];
    int has_continue = 0;
    while(1)  // retry until a valid seq-0 datagram with good magic is received
    {
        long payload0 = (n_dgrams == 1) ? framesizefull : (long)UDP_DGRAM_CHUNK_SIZE;
        struct iovec iov0[2] =
        {
            {.iov_base = hdr0, .iov_len = 2},
            {.iov_base = buff, .iov_len = (size_t)payload0}
        };
        struct msghdr msg0 =
        {
            .msg_name = &src,
            .msg_namelen = sizeof(src),
            .msg_iov = iov0,
            .msg_iovlen = 2
        };
        if(recvmsg(fds, &msg0, 0) < 0)
        {
            return -1;
        }

        if(hdr0[0] != (uint8_t)MAGIC_UDP_MULTIGRAMS || hdr0[1] != 0)
        {
            continue;
        }

        NETWORK_HEADER *hdr = (NETWORK_HEADER *)buff;
        if(hdr->magic != MAGIC_FRAME_METADATA)
        {
            continue;
        }

        // Receive datagrams 1..n_dgrams-1 directly into buff
        for(int k = 1; k < n_dgrams; ++k)
        {
            uint8_t hdrk[2];
            long expect = (k == n_dgrams - 1)
                          ? (framesizefull - (long)k * UDP_DGRAM_CHUNK_SIZE)
                          : (long)UDP_DGRAM_CHUNK_SIZE;
            struct iovec iovk[2] =
            {
                {.iov_base = hdrk, .iov_len = 2},
                {.iov_base = buff + (long)k * UDP_DGRAM_CHUNK_SIZE, .iov_len = (size_t)expect}
            };
            struct msghdr msgk =
            {
                .msg_name = &src,
                .msg_namelen = sizeof(src),
                .msg_iov = iovk,
                .msg_iovlen = 2
            };
            if(recvmsg(fds, &msgk, 0) < 0)
            {
                return -1;
            }

            if(hdrk[0] != (uint8_t)MAGIC_UDP_MULTIGRAMS || hdrk[1] != (uint8_t)k)
            {
                printf("UDP datagram %d bad header (magic=%02x seq=%d)\n",
                       k, hdrk[0], hdrk[1]);
                has_continue = 1;
                break;
            }
        }
        if(has_continue)
        {
            has_continue = 0;
            continue;
        }
        return framesizefull;
    }
}

// ==========================================
// TCP receiver
// ==========================================

/** continuously receives 2D image through TCP link */
imageID common_image_NETWORKreceive(int port,
                                    __attribute__((unused)) int do_counter_sync,
                                    int RT_priority,
                                    int use_TCP)
{
    int fds_receive_point = -1;

    imageID ID = -1;
    IMAGE *img_p = NULL;

    struct sched_param schedpar;

    // ===========================
    // processinfo
    // ===========================
    PROCESSINFO *processinfo;
    {
        char pinfoname[200];
        snprintf(pinfoname, 200, "ntw-receive-%d-%d", use_TCP, port);
        processinfo = processinfo_shm_create(pinfoname, 0);
        processinfo->loopstat = 0;
        strcpy(processinfo->source_FUNCTION, __FUNCTION__);
        strcpy(processinfo->source_FILE, __FILE__);
        processinfo->source_LINE = __LINE__;
        processinfo_WriteMessage(processinfo, "Waiting for input stream");
    }

    // ===========================
    // Signals
    // ===========================
    if(sigaction(SIGTERM, &data.sigact, NULL) == -1 ||
            sigaction(SIGINT, &data.sigact, NULL) == -1 ||
            sigaction(SIGABRT, &data.sigact, NULL) == -1 ||
            sigaction(SIGBUS, &data.sigact, NULL) == -1 ||
            sigaction(SIGSEGV, &data.sigact, NULL) == -1 ||
            sigaction(SIGHUP, &data.sigact, NULL) == -1 ||
            sigaction(SIGPIPE, &data.sigact, NULL) == -1)
    {
        printf("\nCan't catch a requested signal\n");
    }

    // ===========================
    // RT priority
    // ===========================
    schedpar.sched_priority = RT_priority;
    if(seteuid(data.euid) != 0)
    {
        PRINT_ERROR("seteuid error");
    }
    sched_setscheduler(0, SCHED_FIFO, &schedpar);
    if(seteuid(data.ruid) != 0)
    {
        PRINT_ERROR("seteuid error");
    }

    // ===========================
    // Server socket
    // ===========================
    if(use_TCP)
    {
        if(initialize_receiving_endpoints_tcp(port, processinfo,
                                              &fds_receive_point) != 0)
        {
            goto cleanup;
        }
    }
    else
    {
        if(initialize_receiving_endpoints_udp(port, processinfo,
                                              &fds_receive_point) != 0)
        {
            goto cleanup;
        }
    }

    int nbkw_wire = 0;
    if(use_TCP)
    {
        ID = initial_receive_and_initialize_image_TCP(fds_receive_point, &nbkw_wire);
    }
    else
    {
        ID = initial_receive_and_initialize_image_UDP(fds_receive_point, &nbkw_wire);
    }
    if(ID < 0)
    {
        goto cleanup;
    }

    img_p = &data.image[ID];

    // Compute and allocate buffer
    ssize_t framesize = img_p->md->size[0] * img_p->md->size[1] *
                        ImageStreamIO_typesize(img_p->md->datatype);
    ssize_t recv_size_full = sizeof(NETWORK_HEADER) +
                             sizeof(IMAGE_METADATA) +
                             framesize +
                             nbkw_wire * sizeof(IMAGE_KEYWORD);
    long n_slices = img_p->md->naxis > 2 ? img_p->md->size[2] : 1;

    // Log stream info
    {
        char typestring[8];
        snprintf(typestring, 8, "%s",
                 ImageStreamIO_typename(img_p->md->datatype));
        char m[200];
        snprintf(m, 200, "<- %s [%d x %d x %ld] %s",
                 img_p->md->name, img_p->md->size[0], img_p->md->size[1],
                 n_slices, typestring);
        snprintf(processinfo->description, 200,
                 "%s %dx%dx%ld %s", img_p->md->name,
                 img_p->md->size[0], img_p->md->size[1], n_slices, typestring);
        processinfo_WriteMessage(processinfo, m);
    }

    char *buffer_full_data = malloc(recv_size_full);
    char *buffer_datagrams = malloc(UDP_DGRAM_CHUNK_SIZE);

    // ===========================
    // Main loop
    // ===========================
    processinfo->loopstat = 1;
    int loopOK = 1;
    int has_kw = (nbkw_wire > 0);

    long cnt0previous = 0;
    long _monitor_index = 0;
    long _monitor_loop_index = 0;
    long _m_input_cnt0 = 0;
    long _m_output_cnt0 = 0;
    long _monitor_interval = 10000;

    // Housekeeping pointers on the receive buffer
    NETWORK_HEADER *hdr = (NETWORK_HEADER *)buffer_full_data;
    IMAGE_METADATA *recv_md = (IMAGE_METADATA *)((char *)hdr + sizeof(
                                  NETWORK_HEADER));
    char *pixel_ptr = (char *)recv_md + sizeof(IMAGE_METADATA);
    char *kw_ptr = pixel_ptr + framesize;

    // Housekeeping buffer to the output image (slice 0)
    char *ptr0 = (char *)img_p->array.raw;

    long recvsize;

    while(loopOK == 1)
    {
        // --- processinfo control ---
        if(data.processinfo == 1)
        {
            while(processinfo->CTRLval == 1)
            {
                usleep(50);    // pause
            }
            if(processinfo->CTRLval == 2)
            {
                processinfo->CTRLval = 1;    // single step
            }
            if(processinfo->CTRLval == 3)
            {
                loopOK = 0;    // exit
            }
        }

        // Normal path: blocking recv of one full frame.
        if(use_TCP)
        {
            recvsize = recv_buffer_tcp(fds_receive_point, buffer_full_data, recv_size_full);
        }
        else
        {
            recvsize = recv_buffer_udp(fds_receive_point, buffer_full_data,
                                       buffer_datagrams, recv_size_full);
        }
        if(recvsize <= 0)
        {
            printf("Connection closed (recvsize=%ld)\n", recvsize);
            loopOK = 0;
            goto loop_cleanup;
        }

        if((data.processinfo == 1) && (processinfo->MeasureTiming == 1))
        {
            processinfo_exec_start(processinfo);
        }

        long recv_slice = hdr->slice;

        img_p->md->write = 1;
        img_p->md->cnt1 = (uint64_t)recv_slice;
        img_p->md->atime = recv_md->atime; // TODO

        if(n_slices > 1)
        {
            memcpy(ptr0 + framesize * recv_slice, pixel_ptr, (size_t)framesize);
        }
        else
        {
            memcpy(ptr0, pixel_ptr, (size_t)framesize);
        }

        if(has_kw)
            memcpy(img_p->kw, (IMAGE_KEYWORD *)kw_ptr,
                   (size_t)(img_p->md->NBkw * (long)sizeof(IMAGE_KEYWORD)));

        long frameincr = (long)hdr->cnt0 - cnt0previous;
        if(frameincr > 1)
            printf("Skipped %ld frame(s) cnt0=%ld slice=%ld\n",
                   frameincr - 1, (long)hdr->cnt0, recv_slice);
        cnt0previous = hdr->cnt0;

        if(_monitor_index == _monitor_interval)
        {
            printf("[%5ld]  input %20ld (+ %8ld)  output %20ld (+ %8ld)\n",
                   _monitor_loop_index,
                   hdr->cnt0, hdr->cnt0 - _m_input_cnt0,
                   img_p->md->cnt0, img_p->md->cnt0 - _m_output_cnt0);
            _m_input_cnt0 = hdr->cnt0;
            _m_output_cnt0 = img_p->md->cnt0;
            _monitor_loop_index++;
            _monitor_index = 0;
        }
        _monitor_index++;

        img_p->streamproctrace[0].cnt0 = img_p->md->cnt0;
        processinfo_update_output_stream_atime(processinfo, ID,
                                               &img_p->md->atime); // TODO

loop_cleanup:
        if((data.processinfo == 1) && (processinfo->MeasureTiming == 1))
        {
            processinfo_exec_end(processinfo);
        }

        if(data.processinfo == 1)
        {
            processinfo->loopcnt++;
        }

        // Signals
        if((data.signal_TERM || data.signal_INT || data.signal_ABRT ||
                data.signal_BUS || data.signal_SEGV || data.signal_HUP ||
                data.signal_PIPE) &&
                data.processinfo)
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
            break;
        }
    } // while loopOK

cleanup:
    processinfo_cleanExit(processinfo);
    free(buffer_full_data);
    free(buffer_datagrams);
    close(fds_receive_point);
    printf("TCP port %d closed\n", port);
    fflush(stdout);

    return ID;
}

imageID COREMOD_MEMORY_image_NETWORKreceive(int port,
        __attribute__((unused)) int do_counter_sync,
        int RT_priority)
{
    return common_image_NETWORKreceive(port, do_counter_sync, RT_priority, 1);
}

/** continuously receives 2D image through UDP */
imageID COREMOD_MEMORY_image_NETUDPreceive(int port,
        __attribute__((unused)) int do_counter_sync,
        int RT_priority)
{
    return common_image_NETWORKreceive(port, do_counter_sync, RT_priority, 0);
}
