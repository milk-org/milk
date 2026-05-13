/**
 * @file    stream_UDP.c
 * @brief   UDP stream transfer
 *
 * Uses FPS V2 framework.
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

/* forward decls */
imageID COREMOD_MEMORY_image_NETUDPtransmit(
    const char *IDname,
    const char *IPaddr,
    int port, int do_counter_sync,
    int RT_priority);

imageID COREMOD_MEMORY_image_NETUDPreceive(
    int port, int do_counter_sync,
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
static long long p_csync = 0;
static long long p_rtprio = 80;


/* ================================================================
 *  CMD 1: imudptransmit (5 args, primary)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "imudptransmit",
    .cmdkey      = "imudptransmit",
    .description =
        "transmit image over UDP network",
    .description_long =
        "Transmit or receive image stream data over UDP for low-latency network streaming. Suitable for real-time telemetry where occasional packet loss is acceptable."
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
    X(".csync", &p_csync, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "counter sync") \
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
    COREMOD_MEMORY_image_NETUDPtransmit(
        p_imname, p_ipaddr,
        p_port, p_csync, p_rtprio);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 2: imudpreceive (3 args)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_rx = {
    .fps_name    = "imudpreceive",
    .cmdkey      = "imudpreceive",
    .description =
        "receive image(s) over UDP "
        "network",
    .description_long =
        "Transmit or receive image stream data over UDP for low-latency network streaming. Suitable for real-time telemetry where occasional packet loss is acceptable."
};

#define FPS_PARAMS_RX(X) \
    X(".port", &p_port, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "port") \
    X(".csync", &p_csync, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "counter sync") \
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
    COREMOD_MEMORY_image_NETUDPreceive(
        p_port, p_csync, p_rtprio);
    return RETURN_SUCCESS;
}


/* ================================================================
 *  REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)

static FPS_CLI_BINDING bindings_rx[] = {
    FPS_PARAMS_RX(FPS_X_BINDING)
};
static const int nb_bindings_rx =
    sizeof(bindings_rx) /
    sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF farg_rx[] = {
    FPS_PARAMS_RX(FPS_X_FARG)
};

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
CLIADDCMD_COREMOD_memory__stream_UDP()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    safe_fps_fill_farg_examples(
        farg_rx, bindings_rx,
        nb_bindings_rx);

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
    //int                result;
    unsigned long long cnt  = 0;
    long long          iter = 0;
    long               framesize; // pixel data only
    uint32_t           xsize, ysize;
    char              *ptr_img_data; // source
    char              *ptr_img_data_slice; // source - offset by slice
    int                res; // Return status for socket ops
    int                byte_sock_count;
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

    {
        IMGID img = imgid_make_from_name(IDname);
        resolveIMGID(
            &img, ERRMODE_ABORT,
            dcimg, dcnimg);
        ID = img.ID;
    }

    if((fds_client = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0)
    {
        PRINT_ERROR("creating UDP socket");
        return -1;
    }

    setsockopt(fds_client,
        SOL_SOCKET,
        SO_REUSEADDR,
        (char *) & flag,
        sizeof(flag));
    setsockopt(fds_client,
        SOL_SOCKET,
        SO_REUSEPORT,
        (char *) & flag,
        sizeof(flag));

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
        xsize    = dcimg[ID].md[0].size[0];
        ysize    = dcimg[ID].md[0].size[1];
        NBslices = 1;
        if(dcimg[ID].md[0].naxis > 2 && dcimg[ID].md[0].size[2] > 1)
        {
            NBslices = dcimg[ID].md[0].size[2];
        }
    }

    if(loopOK == 1)
    {
        framesize = ImageStreamIO_typesize(dcimg[ID].md[0].datatype) * xsize *
                    ysize;
        printf("IMAGE FRAME SIZE = %ld\n", framesize);
        fflush(stdout);
    }

    if(loopOK == 1)
    {
        ptr_img_data = (char *) ImageStreamIO_get_image_d_ptr(&dcimg[ID]);

        framesize1 = framesize + sizeof(IMAGE_METADATA);

        if(TCPTRANSFERKW == 0)
        {
            framesizeall = framesize1;
        }
        else
        {
            framesizeall =
                framesize1 + dcimg[ID].md[0].NBkw * sizeof(IMAGE_KEYWORD);
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
        printf("sem = %d\n", dcimg[ID].md[0].sem);
        fflush(stdout);
    }

    use_sem = stream_net_decide_sync(
        dcimg[ID].md[0].sem, do_counter_sync,
        semtrig, processinfo);

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
            while(dcimg[ID].md[0].cnt0 == cnt)  // test if new frame exists
            {
                usleep(5);
            }
            cnt  = dcimg[ID].md[0].cnt0;
            semr = 0;
        }
        else
        {
            semr = stream_net_sem_wait(
                dcimg + ID, semtrig);

            stream_net_sem_drain(
                dcimg + ID, semtrig,
                &iter, processinfo);
        }

        processinfo_exec_start(processinfo);
        if(processinfo_compute_status(processinfo) == 1)
        {

            if(semr == 0)
            {

                slice = stream_net_clamp_slice(
                    dcimg[ID].md[0].cnt1,
                    oldslice, NBslices);

                // Fill up the transmission buffer
                __builtin_memcpy(ptr_buff_metadata,
                    &dcimg[ID].md[0],
                    sizeof(IMAGE_METADATA));

                ptr_img_data_slice = ptr_img_data + framesize * slice;
                __builtin_memcpy(ptr_buff_data, ptr_img_data_slice, framesize);

                if(TCPTRANSFERKW == 1)
                {
                    __builtin_memcpy(ptr_buff_keywords,
                           (char *) dcimg[ID].kw,
                           dcimg[ID].md[0].NBkw * sizeof(IMAGE_KEYWORD));
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
                    PRINT_ERROR("socket send error: %s", strerror(errno));
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

