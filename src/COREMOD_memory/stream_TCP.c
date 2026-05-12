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
    0x3E;          // Random magic to start datagrams with.
static int UDP_DGRAM_CHUNK_SIZE = 62 *
                                  1024;     // Max payload per datagram, just shy of the maximum 65507 bytes

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
    int last_nonzeroindex = num_chunks;
    while(chunk_sizes[last_nonzeroindex] == 0 && last_nonzeroindex >= 0)
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

    int last_nonzeroindex = num_chunks;
    while(chunk_sizes[last_nonzeroindex] == 0 && last_nonzeroindex >= 0)
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

            _sent_bytes           += (long)to_send;
            src                   += to_send;
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
                                     IMAGE_KEYWORD) : 0L;

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

/** continuously receives 2D image through TCP link
 * mode = 1, force counter to be used for synchronization, ignore semaphores if they exist
 */

imageID COREMOD_MEMORY_image_NETWORKreceive(int port,
        __attribute__((unused)) int mode,
        int RT_priority)
{
    struct sockaddr_in sock_server;
    struct sockaddr_in sock_client;
    int fds_server;
    int fds_client;
    socklen_t slen_client;

    int flag = 1;
    long recvsize;
    int result;
    long totsize = 0;
    int MAXPENDING = 5;

    IMAGE_METADATA *imgmd;
    imageID ID;
    IMAGE *img_p;
    long framesize;
    uint32_t xsize;
    uint32_t ysize;
    char *ptr0; // source
    long NBslices;
    int socketOpen = 1; // 0 if socket is closed
    int semval;
    int semnb;
    int OKim;
    int axis;

    imgmd = (IMAGE_METADATA *)malloc(sizeof(IMAGE_METADATA));

    NETWORK_HEADER *frame_md_p;
    long framesize1;    // pixel data + metadata
    long framesizefull; // pixel data + metadata + kw
    char *buff;         // buffer

    // size_t flushsize;
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
        processinfo = processinfo_shm_create(pinfoname, 0);
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
        sigaction(SIGPIPE, &data.sigact, NULL) == -1)
    {
        printf("\nCan't catch a requested signal (TERM, INT, ABRT, BUS, SEGV, HUP, PIPE)\n");
    }

    schedpar.sched_priority = RT_priority;
    if(seteuid(data.euid) != 0)  // This goes up to maximum privileges
    {
        PRINT_ERROR("seteuid error");
    }
    sched_setscheduler(0,
                       SCHED_FIFO,
                       &schedpar); // other option is SCHED_RR, might be faster
    if(seteuid(data.ruid) != 0)    // Go back to normal privileges
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

    memset((char *)&sock_server, 0, sizeof(sock_server));

    result = setsockopt(fds_server,    /* socket affected */
                        IPPROTO_TCP,   /* set option at TCP level */
                        TCP_NODELAY,   /* name of option */
                        (char *)&flag, /* the cast is historical cruft */
                        sizeof(flag)); /* length of option value */
    result -= setsockopt(fds_server, SOL_SOCKET, SO_REUSEADDR, (char *)&flag,
                         sizeof(flag));
    result -= setsockopt(fds_server, SOL_SOCKET, SO_REUSEPORT, (char *)&flag,
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

    sock_server.sin_family = AF_INET;
    sock_server.sin_port = htons(port);
    sock_server.sin_addr.s_addr = htonl(INADDR_ANY);

    // bind socket to port
    if(bind(fds_server,
            (struct sockaddr *)&sock_server,
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
                            (struct sockaddr *)&sock_client,
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

    // Receive initial NETWORK_HEADER for stream setup
    NETWORK_HEADER initial_hdr;
    if(recv(fds_client, &initial_hdr, sizeof(NETWORK_HEADER), MSG_WAITALL) < 0)
    {
        char msgstring[200];
        snprintf(msgstring, 200, "ERROR receiving initial frame header");
        printf("%s\n", msgstring);
        if(data.processinfo == 1)
        {
            processinfo->loopstat = 4;
            processinfo_WriteMessage(processinfo, msgstring);
        }
        exit(0);
    }
    if(initial_hdr.magic != MAGIC_FRAME_METADATA)
    {
        char msgstring[200];
        snprintf(msgstring, 200, "ERROR: bad magic in initial frame header");
        printf("%s\n", msgstring);
        if(data.processinfo == 1)
        {
            processinfo->loopstat = 4;
            processinfo_WriteMessage(processinfo, msgstring);
        }
        exit(0);
    }
    // Receive initial IMAGE_METADATA for stream setup
    if(recv(fds_client, imgmd, sizeof(IMAGE_METADATA), MSG_WAITALL) < 0)
    {
        char msgstring[200];
        snprintf(msgstring, 200, "ERROR receiving initial image metadata");
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
        snprintf(msgstring, 200, "Receiving stream %s", imgmd->name);
        processinfo_WriteMessage(processinfo, msgstring);
    }

    // is image already in memory ?
    OKim = 0;

    ID = image_ID(imgmd->name);
    printf("ID: %ld\n", ID);

    if(ID == -1)
    {
        // is it in shared memory ?
        ID = read_sharedmem_image(imgmd->name);
        printf("ID: %ld\n", ID);
    }

    // img_p = &data.image[ID]; // Of course that doesn't fucking work if ID is -1.

    list_image_ID();

    if(ID == -1)
    {
        OKim = 0;
    }
    else
    {
        img_p = &data.image[ID];
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

        if(SEND_KEYWORDS == 1 && imgmd->NBkw > img_p->md->NBkw)
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
        create_image_ID(imgmd->name,
                        imgmd->naxis,
                        imgmd->size,
                        imgmd->datatype,
                        imgmd->shared,
                        imgmd->NBkw,
                        0,
                        &ID);
        printf("Created image stream %s - shared = %d\n",
               imgmd->name,
               imgmd->shared);
        printf("Size = %d,%d\n", imgmd->size[0], imgmd->size[1]);
        // OKim is now OK. Re-point img_p
        img_p = &data.image[ID];
    }
    else
    {
        printf("REUSING EXISTING IMAGE %s\n", imgmd->name);
    }

    xsize = img_p->md->size[0];
    ysize = img_p->md->size[1];
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

    ptr0 = (char *)img_p->array.raw;

    if(data.processinfo == 1)
    {
        char msgstring[200];
        snprintf(msgstring,
                 200,
                 "<- %s [%d x %d x %ld] %s",
                 imgmd->name,
                 (int)xsize,
                 (int)ysize,
                 NBslices,
                 typestring);
        snprintf(processinfo->description,
                 200,
                 "%s %dx%dx%ld %s",
                 imgmd->name,
                 (int)xsize,
                 (int)ysize,
                 NBslices,
                 typestring);
        processinfo_WriteMessage(processinfo, msgstring);
    }

    // Wire layout: [NETWORK_HEADER | IMAGE_METADATA | keywords (opt) | pixel slice]
    int has_kw = (SEND_KEYWORDS > 0) && (initial_hdr.nbkw > 0);
    framesize1 = sizeof(NETWORK_HEADER) + sizeof(IMAGE_METADATA);
    framesizefull = framesize1 + (has_kw ? (long)(initial_hdr.nbkw * sizeof(
                                      IMAGE_KEYWORD)) : 0L) + framesize;
    printf("image frame size full (hdr + md + kw + img) = %ld\n", framesizefull);

    buff = (char *)malloc(sizeof(char) * framesizefull);

    frame_md_p = (NETWORK_HEADER *)buff;

    if(data.processinfo == 1)
    {
        processinfo->loopstat =
            1; // notify processinfo that we are entering loop
    }

    socketOpen = 1;
    long loopcnt = 0;
    int loopOK = 1;

    // In-loop counter watch and debug prompts
    long frameincr;
    long minputcnt = 0;
    long moutputcnt = 0;
    long monitorinterval = 10000;
    long monitorindex = 0;
    long monitorloopindex = 0;
    long cnt0previous = 0;

    {
        // Finally, just before we start, flush the TCP receive buffer. BUT we need to flush an integer number of frames, that's important,
        // or we end up losing sync.
        // This entire thing is kinda useless... it's legacy dating from ImageStreamIO version mismatches where headers could have different sizes
        // at either end...
        socket_flush_buff = (char *)malloc(framesizefull);
        long recv_bytes = framesizefull;
        while(recv_bytes == framesizefull)
        {
            recv_bytes = recv(fds_client, socket_flush_buff, framesizefull, MSG_DONTWAIT);
            printf("TCP recv buffer flush. %ld stray bytes.\n", recv_bytes);
        }
        if(recv_bytes >
                0) // Will be -1 if we got 0 bytes at the last iteration above
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
            frame_md_p = (NETWORK_HEADER *)buff;
            if(frame_md_p->magic != MAGIC_FRAME_METADATA)
            {
                printf("Bad magic! Looping fast.\n");
                continue;
            }

            long recv_slice = frame_md_p->slice;
            IMAGE_METADATA *recv_md_p = (IMAGE_METADATA *)(buff + sizeof(NETWORK_HEADER));
            char *kw_ptr = buff + sizeof(NETWORK_HEADER) + sizeof(IMAGE_METADATA);
            char *pixel_ptr = kw_ptr + (has_kw ? (long)(initial_hdr.nbkw * sizeof(
                                            IMAGE_KEYWORD)) : 0L);

            img_p->md->write = 1;
            img_p->md->cnt1 = (uint64_t)recv_slice;
            img_p->md->atime = recv_md_p->atime;

            // copy pixel data
            if(NBslices > 1)
            {
                memcpy(ptr0 + framesize * recv_slice, pixel_ptr, framesize);
            }
            else
            {
                memcpy(ptr0, pixel_ptr, framesize);
            }

            if(has_kw)
            {
                // copy kw
                memcpy(img_p->kw,
                       (IMAGE_KEYWORD *)kw_ptr,
                       img_p->md->NBkw * sizeof(IMAGE_KEYWORD));
            }

            frameincr = (long)frame_md_p->cnt0 - cnt0previous;
            if(frameincr > 1)
            {
                printf("Skipped %ld frame(s) at index cnt0=%ld | slice=%ld\n",
                       frameincr - 1,
                       (long)(frame_md_p->cnt0),
                       recv_slice);
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

                minputcnt = frame_md_p->cnt0;
                moutputcnt = img_p->md->cnt0;

                monitorloopindex++;
                monitorindex = 0;
            }

            monitorindex++;

            // Carry cnt0 to streamproctrace
            img_p->streamproctrace[0].cnt0 = img_p->md->cnt0;
            processinfo_update_output_stream(processinfo, ID);
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
