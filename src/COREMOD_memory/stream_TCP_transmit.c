/**
 * @file    stream_TCPtransmit.c
 * @brief   transmit stream over TCP
 */

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>


#include "CommandLineInterface/CLIcore.h"
#include "CommandLineInterface/timeutils.h"

#include "image_ID.h"
#include "stream_sem.h"

#include "COREMOD_tools/COREMOD_tools.h"



typedef struct
{
    long cnt0;
    long cnt1;
} TCP_BUFFER_METADATA;

// set to 1 if transfering keywords
static int TCPTRANSFERKW = 1;


// variables local to this translation unit
static char *inimname;
static char *IPaddr;

static uint32_t *port;


static CLICMDARGDEF farg[] = {
    {
        CLIARG_IMG,
        ".in_sname",
        "input stream",
        "ims1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimname,
        NULL
    },
    {
        CLIARG_STR,
        ".IPaddr",
        "IP address",
        "1.1.1.1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &IPaddr,
        NULL
    },
    {
        CLIARG_UINT32,
        ".port",
        "port number",
        "31020",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &port,
        NULL
    }
};

static CLICMDDATA CLIcmddata = {"shmimTCPtransmit",
                                "transmit stream over TCP",
                                CLICMD_FIELDS_DEFAULTS
                               };

// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}



// Wrapper function, used by all CLI calls
static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID img = mkIMGID_from_name(inimname);
    resolveIMGID(&img, ERRMODE_ABORT);


    int fds_client;
    if ((fds_client = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP)) < 0)
    {
        printf("ERROR creating socket\n");
        exit(0);
    }

    int  flag = 1;
    int result = setsockopt(fds_client,     /* socket affected */
                            IPPROTO_TCP,    /* set option at TCP level */
                            TCP_NODELAY,    /* name of option */
                            (char *) &flag, /* the cast is historical cruft */
                            sizeof(int));   /* length of option value */

    if (result < 0)
    {
        DEBUG_TRACE_FEXIT();
        return RETURN_FAILURE;
    }


    struct sockaddr_in sock_server;
    memset((char *) &sock_server, 0, sizeof(sock_server));
    sock_server.sin_family      = AF_INET;
    sock_server.sin_port        = htons(port);
    sock_server.sin_addr.s_addr = inet_addr(IPaddr);

    if (connect(fds_client,
                (struct sockaddr *) &sock_server,
                sizeof(sock_server)) < 0)
    {
        DEBUG_TRACE_FEXIT();
        return RETURN_FAILURE;
    }

    if (send(fds_client,
             (void *) img.md,
             sizeof(IMAGE_METADATA),
             0) != sizeof(IMAGE_METADATA))
    {
        DEBUG_TRACE_FEXIT();
        return RETURN_FAILURE;
    }


    uint32_t xsize    = img.md->size[0];
    uint32_t ysize    = img.md->size[1];
    int NBslices = 1;
    if (img.md->naxis > 2)
    {
        if (img.md->size[2] > 1)
        {
            NBslices = img.md->size[2];
        }
    }


    long framesize; // pixel data only
    switch (img.md->datatype)
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
        DEBUG_TRACE_FEXIT();
        return RETURN_FAILURE;
        break;
    }


    char              *ptr0; // source
    switch (img.md->datatype)
    {

    case _DATATYPE_INT8:
        ptr0 = (char *) img.im->array.SI8;
        break;
    case _DATATYPE_UINT8:
        ptr0 = (char *) img.im->array.UI8;
        break;

    case _DATATYPE_INT16:
        ptr0 = (char *) img.im->array.SI16;
        break;
    case _DATATYPE_UINT16:
        ptr0 = (char *) img.im->array.UI16;
        break;

    case _DATATYPE_INT32:
        ptr0 = (char *) img.im->array.SI32;
        break;
    case _DATATYPE_UINT32:
        ptr0 = (char *) img.im->array.UI32;
        break;

    case _DATATYPE_INT64:
        ptr0 = (char *) img.im->array.SI64;
        break;
    case _DATATYPE_UINT64:
        ptr0 = (char *) img.im->array.UI64;
        break;

    case _DATATYPE_FLOAT:
        ptr0 = (char *) img.im->array.F;
        break;
    case _DATATYPE_DOUBLE:
        ptr0 = (char *) img.im->array.D;
        break;

    default:
        printf("ERROR: WRONG DATA TYPE\n");
        DEBUG_TRACE_FEXIT();
        return RETURN_FAILURE;
        break;
    }

    TCP_BUFFER_METADATA *frame_md;
    frame_md = (TCP_BUFFER_METADATA *) malloc(sizeof(TCP_BUFFER_METADATA));

    // pixel data + metadata
    long framesize1 = framesize + sizeof(TCP_BUFFER_METADATA);


    long  framesizeall; // total frame size : pixel data + metadata + kw
    if (TCPTRANSFERKW == 0)
    {
        framesizeall = framesize1;
    }
    else
    {
        framesizeall =
            framesize1 + img.md->NBkw * sizeof(IMAGE_KEYWORD);
    }

    char *buff;         // transmit buffer
    buff = (char *) malloc(sizeof(char) * framesizeall);

    printf("transfer buffer size = %ld\n", framesizeall);
    fflush(stdout);

    int oldslice = 0;
    //sockOK = 1;
    printf("sem = %d\n", img.md->sem);
    fflush(stdout);



    char              *ptr1; // source - offset by slice


    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    frame_md[0].cnt0 = img.md->cnt0;
    frame_md[0].cnt1 = img.md->cnt1;

    int slice = img.md->cnt1;
    if (slice > oldslice + 1)
    {
        slice = oldslice + 1;
    }
    if (NBslices > 1)
        if (oldslice == NBslices - 1)
        {
            slice = 0;
        }
    if (slice > NBslices - 1)
    {
        slice = 0;
    }

    frame_md[0].cnt1 = slice;

    ptr1 =
        ptr0 +
        framesize *
        slice; //data.image[ID].md[0].cnt1; // frame that was just written
    memcpy(buff, ptr1, framesize);
    memcpy(buff + framesize, frame_md, sizeof(TCP_BUFFER_METADATA));

    if (TCPTRANSFERKW == 1)
    {
        memcpy(buff + framesize1,
               (char *) data.image[img.ID].kw,
               img.md->NBkw * sizeof(IMAGE_KEYWORD));
    }

    int rs = send(fds_client, buff, framesizeall, 0);

    if (rs != framesizeall)
    {
        /*perror("socket send error ");
        sprintf(errmsg,
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
        loopOK = 0;*/
        DEBUG_TRACE_FEXIT();
        return RETURN_FAILURE;
    }
    oldslice = slice;


    //processinfo_update_output_stream(processinfo, img.ID);



    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    free(buff);

    close(fds_client);
    printf("port %d closed\n", port);
    fflush(stdout);

    free(frame_md);


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t
CLIADDCMD_COREMOD_memory__stream_TCPtransmit()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
