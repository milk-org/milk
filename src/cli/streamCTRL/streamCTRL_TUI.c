
/**
 * @file streamCTRL.c
 * @brief Data streams control panel
 *
 * Manages data streams
 *
 *
 */


#ifndef __STDC_LIB_EXT1__
typedef int errno_t;
#endif

#include <sys/stat.h>
#include <sys/select.h>
#include <fcntl.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

#include "streamCTRL_defs.h"
#include "streamCTRL_ansi.h"
#include "streamCTRL_TUIcompat.h"

// default location of file mapped semaphores, can be over-ridden by env variable MILK_SHM_DIR
#ifndef SHAREDSHMDIR
#    define SHAREDSHMDIR "/dev/shm"
#endif

#include "streamCTRL_TUI_internal.h"

#include "streamCTRL_find_streams.h"
#include "streamCTRL_print_inode.h"
#include "streamCTRL_print_procpid.h"
#include "streamCTRL_print_trace.h"
#include "streamCTRL_scan.h"
#include "streamCTRL_utilfuncs.h"


/* Tab definitions: (display_mode, key_label, tab_label)
 * Used by the tab-bar renderer via RENDER_ONE_TAB and to derive TAB_COUNT
 * for CTRL+L/R wrap-around.
 *
 * To add a tab, you must also:
 *   1. Add a DISPLAY_MODE_* constant in streamCTRL_TUI.h.
 *   2. Add a case handler (case KEY_F(n) or case 'c') in
 *      streamCTRL_keyinput_process() in this file.
 *   3. Add a help entry in the DISPLAY_MODE_HELP section below.
 */


short unsigned int wrow, wcol;


// current streamCTRL TUI status

struct streamCTRL_TUI_parameters sTUIparam;


/**
 * @brief Clean up streamCTRL TUI resources on exit.
 */
static void streamCTRL_TUI_exit()
{
}


STREAMINFO *g_streaminfo_qsort = NULL;
IMAGE      *g_sort_images      = NULL;
int         g_sort_col         = 0;
int         g_sort_dir         = 0;

/**
 * cmp_stream_col - unified column comparator for streams
 *
 * Compares two streams by the column specified in
 * g_sort_col. Uses g_sort_dir to flip order.
 * Elements a,b are pointers to long (sindex values).
 */
int cmp_stream_col(const void *a, const void *b)
{
    long idxA = *(const long *) a;
    long idxB = *(const long *) b;
    int  res  = 0;

    STREAMINFO *siA = &g_streaminfo_qsort[idxA];
    STREAMINFO *siB = &g_streaminfo_qsort[idxB];

    imageID IDA = siA->ID;
    imageID IDB = siB->ID;

    /* Streams with no valid ID sort to the bottom */
    if (IDA < 0 && IDB < 0)
    {
        return 0;
    }
    if (IDA < 0)
    {
        return 1;
    }
    if (IDB < 0)
    {
        return -1;
    }

    IMAGE_METADATA *mdA = g_sort_images[IDA].md;
    IMAGE_METADATA *mdB = g_sort_images[IDB].md;

    if (mdA == NULL && mdB == NULL)
    {
        return 0;
    }
    if (mdA == NULL)
    {
        return 1;
    }
    if (mdB == NULL)
    {
        return -1;
    }

    switch (g_sort_col)
    {
    case STREAM_SORT_NAME:
        res = strcmp(siA->sname, siB->sname);
        break;

    case STREAM_SORT_TYPE:
        res = (siA->datatype < siB->datatype) ? -1 : (siA->datatype > siB->datatype) ? 1 : 0;
        break;

    case STREAM_SORT_SIZE:
    {
        uint64_t neA = mdA->nelement;
        uint64_t neB = mdB->nelement;
        res          = (neA < neB) ? -1 : (neA > neB) ? 1 : 0;
        break;
    }

    case STREAM_SORT_CNT0:
    {
        uint64_t c0A = mdA->cnt0;
        uint64_t c0B = mdB->cnt0;
        res          = (c0A < c0B) ? -1 : (c0A > c0B) ? 1 : 0;
        break;
    }

    case STREAM_SORT_CPID:
        res = (mdA->creatorPID < mdB->creatorPID)   ? -1
              : (mdA->creatorPID > mdB->creatorPID) ? 1
                                                    : 0;
        break;

    case STREAM_SORT_OPID:
        res = (mdA->ownerPID < mdB->ownerPID) ? -1 : (mdA->ownerPID > mdB->ownerPID) ? 1 : 0;
        break;

    case STREAM_SORT_FREQ:
        res = (siA->frequ_disp < siB->frequ_disp)   ? -1
              : (siA->frequ_disp > siB->frequ_disp) ? 1
                                                    : 0;
        break;
    }

    if (g_sort_dir == 1)
    {
        res = -res;
    }
    return res;
}

/**
 * @brief Control screen for stream structures
 *
 * @return errno_t
 */
errno_t streamCTRL_CTRLscreen(void)
{
    // Ensure we track sigINT
    extern volatile sig_atomic_t sc_sigINT;
    DEBUG_TRACE_FSTART();

    // initialize sCTRLTUIparams
    sTUIparam.loopOK             = 1;
    sTUIparam.dindexSelected     = 0;
    sTUIparam.DisplayDetailLevel = 0;
    sTUIparam.DisplayMode        = DISPLAY_MODE_SUMMARY;
    sTUIparam.NBsindex           = 0;
    sTUIparam.SORTING            = 0;
    sTUIparam.DISPLAY_ALL_SEMS   = 1; // Display all semaphores / just the first 2.
    sTUIparam.fuserScan          = 0;
    sTUIparam.SORT_TOGGLE        = 0;
    sTUIparam.sort_col           = STREAM_SORT_NONE;
    sTUIparam.sort_dir           = 0;
    sTUIparam.frequ              = 10.0; // Hz


    // Display fields
    STREAMINFO    *streaminfo;
    STREAMINFOPROC streaminfoproc;

    DEBUG_TRACEPOINT("function start ");

    pthread_t threadscan;

    struct streamCTRL_TUI_state state;
    state.DispName_NBchar = 36;
    state.DispSize_NBchar = 20;
    state.Dispcnt0_NBchar = 16;
    state.Dispfreq_NBchar = 8;
    state.DispPID_NBchar  = 8;
    state.doffsetindex    = 0;
    state.inodeselected   = 0;
    state.monstrlen       = 200;
    state.monstring       = (char *) malloc(state.monstrlen);

    state.PIDmax = get_PIDmax();
    DEBUG_TRACEPOINT("PID max = %d ", state.PIDmax);

    state.PIDname_array = (char **) malloc(sizeof(char *) * state.PIDmax);
    for (int pidi = 0; pidi < state.PIDmax; pidi++)
    {
        state.PIDname_array[pidi] = NULL;
    }

    streaminfoproc.WriteFlistToFile = 0;
    streaminfoproc.loopcnt          = 0;
    streaminfoproc.fuserUpdate      = 0;

    streaminfo = (STREAMINFO *) malloc(sizeof(STREAMINFO) * streamNBID_MAX);
    for (int sindex = 0; sindex < streamNBID_MAX; sindex++)
    {
        streaminfo[sindex].ID                   = -1;
        streaminfo[sindex].ISIOretval           = IMAGESTREAMIO_FILEOPEN;
        streaminfo[sindex].updatevalue          = 0.0;
        streaminfo[sindex].updatevalue_frozen   = 0.0;
        streaminfo[sindex].cnt0                 = 0;
        streaminfo[sindex].streamOpenPID_status = 0;
        streaminfo[sindex].erased               = 0;
        streaminfo[sindex].frequ_disp           = 0.0;
        streaminfo[sindex].t_avg_start          = 0.0;
        streaminfo[sindex].cnt0_avg_start       = 0;
    }
    streaminfoproc.PIDtable = state.PIDname_array;

    IMAGE *streamCTRLimages = (IMAGE *) malloc(sizeof(IMAGE) * streamNBID_MAX);
    for (imageID imID = 0; imID < streamNBID_MAX; imID++)
    {
        streamCTRLimages[imID].used    = 0;
        streamCTRLimages[imID].shmfd   = -1;
        streamCTRLimages[imID].memsize = 0;
        streamCTRLimages[imID].semptr  = NULL;
        streamCTRLimages[imID].semlog  = NULL;
    }

    streamCTRLarg_struct streamCTRLdata;
    streamCTRLdata.sinfo          = streaminfo;
    streamCTRLdata.streaminfoproc = &streaminfoproc;
    streamCTRLdata.images         = streamCTRLimages;


    // catch signals (CTRL-C etc)
    //


    // default: use TUI
    TUI_set_screenprintmode(SCREENPRINT_NCURSES);

    if (getenv("MILK_TUIPRINT_STDIO"))
    {
        // use stdio instead of TUI
        TUI_set_screenprintmode(SCREENPRINT_STDIO);
    }

    if (getenv("MILK_TUIPRINT_NONE"))
    {
        TUI_set_screenprintmode(SCREENPRINT_NONE);
    }

    DEBUG_TRACEPOINT("Initialize terminal");
    TUI_init_terminal(&wrow, &wcol);
    atexit(streamCTRL_TUI_exit);


    streaminfoproc.filter       = 0;
    streaminfoproc.NBstream     = 0;
    streaminfoproc.twaitus      = 100000; // 10 Hz
    streaminfoproc.fuserUpdate0 = 1;      //update on first instance

    // inodes that are upstream of current selection
    state.NBupstreaminodeMAX = 100;
    state.NBupstreaminode    = 0;
    state.upstreaminode      = (ino_t *) malloc(sizeof(ino_t) * state.NBupstreaminodeMAX);

    // processes that are upstream of current selection
    state.NBupstreamprocMAX = 100;
    state.NBupstreamproc    = 0;
    state.upstreamproc      = (pid_t *) malloc(sizeof(pid_t) * state.NBupstreamprocMAX);

    TUI_clearscreen(0, 0);
    DEBUG_TRACEPOINT(" ");

    // redirect stderr to /dev/null

    /* Redirect stderr to a per-PID log file for the TUI session.
     * backstderr and newstderrfname are used post-loop for cleanup. */
    int  backstderr;
    char newstderrfname[STRINGMAXLEN_FULLFILENAME];

    fflush(stderr);
    backstderr = dup(STDERR_FILENO);
    WRITE_FULLFILENAME(newstderrfname, "%s/stderr.cli.%d.txt", SHAREDSHMDIR, (int) getpid());
    {
        umask(0);
        int newstderr = open(newstderrfname, O_WRONLY | O_CREAT, FILEMODE);
        dup2(newstderr, STDERR_FILENO);
        close(newstderr);
    }

    DEBUG_TRACEPOINT("Start scan thread");
    streaminfoproc.loop = 1;
    pthread_create(&threadscan, NULL, streamCTRL_scan, (void *) &streamCTRLdata);

    DEBUG_TRACEPOINT("Scan thread started");


    state.loopcnt = 0;

    DEBUG_TRACEPOINT("get terminal size");
    TUI_init_terminal(&wrow, &wcol);
    //        TUI_get_terminal_size(&wrow, &wcol);

    state.body_start_row = 0;
    while (sc_sigINT == 0 && sc_sigTERM == 0 && sTUIparam.loopOK == 1)
    {
        DEBUG_TRACEPOINT("loop start");

        /* Rough estimate used only for the "Currently displaying" status
         * string in the header. The accurate value is computed below
         * via sc_cursor_row after all headers have been drawn. */
        int NBsinfodisp = wrow - 6;
        if (NBsinfodisp < 1)
        {
            NBsinfodisp = 1;
        }

        if (streaminfoproc.loopcnt == 1)
        {
            sTUIparam.SORTING     = 2;
            sTUIparam.SORT_TOGGLE = 1;
        }
        DEBUG_TRACEPOINT(" ");

        //if(fuserUpdate != 1) // don't wait if ongoing fuser scan

        {
            struct timeval tv;
            tv.tv_sec  = 0;
            tv.tv_usec = (long) (1000000.0 / sTUIparam.frequ);

            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(STDIN_FILENO, &fds);

            select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv);
        }
        DEBUG_TRACEPOINT(" ");

        int ch;
        while ((ch = ansi_get_key()) != ANSI_KEY_NONE)
        {
            DEBUG_TRACEPOINT("Process input character");
            streamCTRL_keyinput_process(ch, &streamCTRLdata, &state);
            DEBUG_TRACEPOINT("Input character processed");
        }

        sTUIparam.NBsindex = streaminfoproc.NBstream;
        DEBUG_TRACEPOINT(" ");

        TUI_clearscreen(&wrow, &wcol);

        streamCTRL_render_screen(&streamCTRLdata, &state);

        DEBUG_TRACEPOINT(" ");

        state.loopcnt++;
    }

    streaminfoproc.loop = 0;
    pthread_join(threadscan, NULL);

    for (int pidi = 0; pidi < state.PIDmax; pidi++)
    {
        if (state.PIDname_array[pidi] != NULL)
        {
            free(state.PIDname_array[pidi]);
        }
    }
    free(state.PIDname_array);

    for (imageID ID = 0; ID < streamNBID_MAX; ID++)
    {
        if (streamCTRLimages[ID].used == 1)
        {
            ImageStreamIO_closeIm(&streamCTRLimages[ID]);
        }
    }

    free(streamCTRLimages);
    free(streaminfo);
    free(state.upstreaminode);
    free(state.upstreamproc);
    free(state.monstring);

    fflush(stderr);
    dup2(backstderr, STDERR_FILENO);
    close(backstderr);

    remove(newstderrfname);

    DEBUG_TRACEPOINT(" ");

    return EXIT_SUCCESS;
}
