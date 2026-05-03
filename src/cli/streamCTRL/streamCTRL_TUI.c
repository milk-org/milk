
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
#define SHAREDSHMDIR  "/dev/shm"
#endif

#include "streamCTRL_TUI.h"

#include "streamCTRL_find_streams.h"
#include "streamCTRL_print_inode.h"
#include "streamCTRL_print_procpid.h"
#include "streamCTRL_print_trace.h"
#include "streamCTRL_scan.h"
#include "streamCTRL_utilfuncs.h"

static inline void streamCTRL_set_sem_color(int val) {
    if (val == 0) {
        screenprint_setcolor(2); // green
    } else if (val >= 10) {
        screenprint_setcolor(4); // red
    } else {
        ansi_detect_color_level();
        if (ansi__color_level >= 3) {
            int r = 150 + (val - 1) * (255 - 150) / 9;
            int g = 100 - (val - 1) * 100 / 9;
            int b = 0;
            SC_APPEND("\033[38;2;%d;%d;%dm", r, g, b);
        } else if (ansi__color_level == 2) {
            if (val < 4) SC_APPEND("\033[38;5;130m");
            else if (val < 7) SC_APPEND("\033[38;5;166m");
            else SC_APPEND("\033[38;5;196m");
        } else {
            screenprint_setcolor(3);
        }
    }
}

/**
 * streamCTRL_render_active_bg - print cnt0 field with active bg
 * @str:         the string to print
 * @len:         number of characters in str
 * @color_level: terminal color capability (2=256, 3=TrueColor)
 *
 * Sets a solid green-tinted background to highlight that the
 * stream counter is actively updating.  One escape per row.
 */
static inline void streamCTRL_render_active_bg(
    const char *str,
    int         len,
    int         color_level
) {
    if(color_level >= 3)
    {
        SC_APPEND("\033[48;2;0;50;30m");
    }
    else
    {
        SC_APPEND("\033[48;5;22m");
    }

    for(int i = 0; i < len; i++)
    {
        if(sc_cursor_col < sc_term_cols &&
           sc_framebuf_pos < SC_FRAMEBUF_SIZE - 1)
        {
            sc_framebuf[sc_framebuf_pos++] = str[i];
            sc_cursor_col++;
        }
    }
}

/**
 * streamCTRL_print_frequ_field - print frequency with colored bg
 * @frequ:       current stream frequency in Hz
 * @wave_age:    seconds since last cnt0 update
 * @color_level: terminal color capability (0-3)
 *
 * Background brightness is proportional to log10(frequ).
 * Visible only while the stream is active (wave_age <= 1s).
 */
static inline void streamCTRL_print_frequ_field(
    double frequ,
    double wave_age,
    int    color_level
) {
    char fbuf[32];

    if(frequ < 0.005) {
        snprintf(fbuf, sizeof(fbuf), " %7s Hz", "---");
    } else {
        snprintf(fbuf, sizeof(fbuf), " %7.2f Hz", frequ);
    }

    /* No background when inactive or terminal too basic. */
    if(color_level < 2 || wave_age > 1.0) {
        TUI_printfw("%s", fbuf);
        return;
    }

    double log_br = 0.0;
    if(frequ >= 1.0) {
        log_br = log10(frequ) / log10(9999.0);
    }
    if(log_br > 1.0) log_br = 1.0;

    if(color_level >= 3) {
        int r = (int)(  10.0 * log_br);
        int g = (int)( 180.0 * log_br);
        int b = (int)(  80.0 * log_br);
        SC_APPEND("\033[48;2;%d;%d;%dm", r, g, b);
    } else {
        int idx = (int)(5.0 * log_br);
        static const int ramp[6] = {
            17, 23, 29, 35, 41, 47
        };
        SC_APPEND("\033[48;5;%dm",
                  ramp[idx < 6 ? idx : 5]);
    }

    TUI_printfw("%s", fbuf);
    SC_APPEND("\033[0m");
}




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
#define TAB_LIST(X) \
    X(DISPLAY_MODE_HELP,    "h",  "Help") \
    X(DISPLAY_MODE_SUMMARY, "F2", "summary") \
    X(DISPLAY_MODE_WRITE,   "F3", "write PIDs") \
    X(DISPLAY_MODE_READ,    "F4", "read PIDs") \
    X(DISPLAY_MODE_SPTRACE, "F5", \
      "process traces") \
    X(DISPLAY_MODE_FUSER,   "F6", "access")

/* Number of tabs — derived from TAB_LIST so CTRL+L/R wrap stays in sync */
#define TAB_COUNT_ONE(mode, key, label) +1
#define TAB_COUNT (0 TAB_LIST(TAB_COUNT_ONE))


static short unsigned int wrow, wcol;


// current streamCTRL TUI status

struct streamCTRL_TUI_parameters
{
    int loopOK;
    int dindexSelected;
    int DisplayDetailLevel;
    int DisplayMode;
    int NBsindex;
    int SORTING;
    int DISPLAY_ALL_SEMS;
    struct tm *uttime_lastScan;
    int fuserScan;
    int SORT_TOGGLE;
    float frequ; // Hz
    int sort_col; // 0=none, 1..7=column id
    int sort_dir; // 0=ascending, 1=descending
    long ssindex[streamNBID_MAX]; // sorted index array
} sTUIparam;


static void streamCTRL_TUI_exit()
{
}

static errno_t streamCTRL_keyinput_process(
    int ch,
    streamCTRLarg_struct *streamCTRLdata
)
{
    char c; // for user input
    int  stringindex;
    time_t  rawtime;
    long sindex;

    switch(ch)
    {
    case ANSI_KEY_CTRL_LEFT: // CTRL+LEFT
        sTUIparam.DisplayMode--;
        if (sTUIparam.DisplayMode < DISPLAY_MODE_HELP)
            sTUIparam.DisplayMode = TAB_COUNT;
        break;

    case ANSI_KEY_CTRL_RIGHT: // CTRL+RIGHT
        sTUIparam.DisplayMode++;
        if (sTUIparam.DisplayMode > TAB_COUNT)
            sTUIparam.DisplayMode = DISPLAY_MODE_HELP;
        break;

    case 3:   // Ctrl+C
    case 'x': // Exit control screen
        sTUIparam.loopOK = 0;
        break;

    case ANSI_KEY_UP:
        sTUIparam.dindexSelected--;
        if(sTUIparam.dindexSelected < 0)
        {
            sTUIparam.dindexSelected = 0;
        }
        break;

    case ANSI_KEY_DOWN:
        sTUIparam.dindexSelected++;
        if(sTUIparam.dindexSelected > sTUIparam.NBsindex - 1)
        {
            sTUIparam.dindexSelected = sTUIparam.NBsindex - 1;
        }
        break;

    case ANSI_KEY_PGUP:
        sTUIparam.dindexSelected -= 10;
        if(sTUIparam.dindexSelected < 0)
        {
            sTUIparam.dindexSelected = 0;
        }
        break;

    case ANSI_KEY_LEFT:
        sTUIparam.DisplayDetailLevel = 0;
        break;

    case ANSI_KEY_RIGHT:
        sTUIparam.DisplayDetailLevel = 1;
        break;

    case ANSI_KEY_PGDN:
        sTUIparam.dindexSelected += 10;
        if(sTUIparam.dindexSelected > sTUIparam.NBsindex - 1)
        {
            sTUIparam.dindexSelected = sTUIparam.NBsindex - 1;
        }
        break;

    // ============ SCREENS

    case 'h': // help
        sTUIparam.DisplayMode = DISPLAY_MODE_HELP;
        break;

    case ANSI_KEY_F2: // semvals
        sTUIparam.DisplayMode = DISPLAY_MODE_SUMMARY;
        break;

    case ANSI_KEY_F3: // write PIDs
        sTUIparam.DisplayMode = DISPLAY_MODE_WRITE;
        break;

    case ANSI_KEY_F4: // read PIDs
        sTUIparam.DisplayMode = DISPLAY_MODE_READ;
        break;

    case ANSI_KEY_F5: // read PIDs
        sTUIparam.DisplayMode = DISPLAY_MODE_SPTRACE;
        break;

    case ANSI_KEY_F6: // open files
        if((sTUIparam.DisplayMode == DISPLAY_MODE_FUSER) ||
                (streamCTRLdata->streaminfoproc->fuserUpdate0 == 1))
        {
            streamCTRLdata->streaminfoproc->fuserUpdate = 1;
            time(&rawtime);
            sTUIparam.uttime_lastScan           = gmtime(&rawtime);
            sTUIparam.fuserScan                 = 1;
        }
        sTUIparam.DisplayMode = DISPLAY_MODE_FUSER;
        break;

    // ============ ACTIONS

    case ctrl('e'): // erase stream
        if(sTUIparam.dindexSelected >= 0)
        {
            sindex =
                sTUIparam.ssindex[
                    sTUIparam.dindexSelected];
            // Flag for removal by scan thread
            // Actual destroy happens in scan
            // thread to avoid race condition
            streamCTRLdata->sinfo[sindex].erased = 1;
        }
        break;

    // ============ SCANNING

    case '{': // slower scan update
        streamCTRLdata->streaminfoproc->twaitus = (int)(1.2 *
                streamCTRLdata->streaminfoproc->twaitus);
        if(streamCTRLdata->streaminfoproc->twaitus > 1000000)
        {
            streamCTRLdata->streaminfoproc->twaitus = 1000000;
        }
        break;

    case '}': // faster scan update
        streamCTRLdata->streaminfoproc->twaitus =
            (int)(0.83333333333333333333 * streamCTRLdata->streaminfoproc->twaitus);
        if(streamCTRLdata->streaminfoproc->twaitus < 1000)
        {
            streamCTRLdata->streaminfoproc->twaitus = 1000;
        }
        break;

    case 'o': // output next scan to file
        streamCTRLdata->streaminfoproc->WriteFlistToFile = 1;
        break;

    // ============ DISPLAY

    case '-': // slower display update
        sTUIparam.frequ *= 0.5;
        if(sTUIparam.frequ < 1.0)
        {
            sTUIparam.frequ = 1.0;
        }
        if(sTUIparam.frequ > 64.0)
        {
            sTUIparam.frequ = 64.0;
        }
        break;

    case '+': // faster display update
        sTUIparam.frequ *= 2.0;
        if(sTUIparam.frequ < 1.0)
        {
            sTUIparam.frequ = 1.0;
        }
        if(sTUIparam.frequ > 64.0)
        {
            sTUIparam.frequ = 64.0;
        }
        break;

    case '1': // shortcut: sort by stream name
        sTUIparam.sort_col = STREAM_SORT_NAME;
        sTUIparam.sort_dir = 0;
        sTUIparam.SORTING = 1;
        break;

    case '2': // shortcut: sort by update recency
        sTUIparam.sort_col = STREAM_SORT_NONE;
        sTUIparam.sort_dir = 0;
        sTUIparam.SORTING     = 2;
        sTUIparam.SORT_TOGGLE = 1;
        break;

    case '3': // shortcut: sort by process access
        sTUIparam.sort_col = STREAM_SORT_NONE;
        sTUIparam.sort_dir = 0;
        sTUIparam.SORTING     = 3;
        sTUIparam.SORT_TOGGLE = 1;
        break;

    case '4': // shortcut: sort by frequency
        sTUIparam.sort_col = STREAM_SORT_FREQ;
        sTUIparam.sort_dir = 1; // descending
        sTUIparam.SORTING = 0;
        break;

    case ']': // next sort column
        sTUIparam.sort_col++;
        if (sTUIparam.sort_col > STREAM_NB_SORT_COLS)
            sTUIparam.sort_col = STREAM_SORT_NONE;
        sTUIparam.sort_dir = 0;
        // Disable legacy sort modes
        sTUIparam.SORTING = 0;
        break;

    case '[': // toggle sort direction
        if (sTUIparam.sort_col > STREAM_SORT_NONE)
            sTUIparam.sort_dir = !sTUIparam.sort_dir;
        break;

    case 'f': // stream name filter toggle
        if(streamCTRLdata->streaminfoproc->filter == 0)
        {
            streamCTRLdata->streaminfoproc->filter = 1;
        }
        else
        {
            streamCTRLdata->streaminfoproc->filter = 0;
        }
        break;

    case 'F': // set stream name filter string
        // TUI_exit();
        EXECUTE_SYSTEM_COMMAND("clear");
        printf("Enter string: ");
        fflush(stdout);
        stringindex = 0;
        while(((c = getchar()) != '\n') &&
                (stringindex < STRINGLENMAX - 2))
        {
            streamCTRLdata->streaminfoproc->namefilter[stringindex] = c;
            if(c == 127)  // delete key
            {
                putchar(0x8);
                putchar(' ');
                putchar(0x8);
                stringindex--;
            }
            else
            {
                //printf("[%d]", (int) c);
                putchar(c); // echo on screen
                stringindex++;
            }
        }
        printf("string entered\n");
        streamCTRLdata->streaminfoproc->namefilter[stringindex] = '\0';
        TUI_init_terminal(&wrow, &wcol);
        break;

    case 's': // toggle all sems / 2 sems
        sTUIparam.DISPLAY_ALL_SEMS = !sTUIparam.DISPLAY_ALL_SEMS;
        break;

    case 'r': // force full screen redraw
        if(write(STDOUT_FILENO, "\033[2J", 4) < 0) {}
        break;
    }
    return EXIT_SUCCESS;
}

static STREAMINFO *g_streaminfo_qsort = NULL;
static IMAGE *g_sort_images = NULL;
static int g_sort_col = 0;
static int g_sort_dir = 0;

/**
 * cmp_stream_col - unified column comparator for streams
 *
 * Compares two streams by the column specified in
 * g_sort_col. Uses g_sort_dir to flip order.
 * Elements a,b are pointers to long (sindex values).
 */
static int cmp_stream_col(const void *a, const void *b)
{
    long idxA = *(const long *)a;
    long idxB = *(const long *)b;
    int res = 0;

    STREAMINFO *siA = &g_streaminfo_qsort[idxA];
    STREAMINFO *siB = &g_streaminfo_qsort[idxB];

    imageID IDA = siA->ID;
    imageID IDB = siB->ID;

    /* Streams with no valid ID sort to the bottom */
    if (IDA < 0 && IDB < 0) return 0;
    if (IDA < 0) return 1;
    if (IDB < 0) return -1;

    IMAGE_METADATA *mdA = g_sort_images[IDA].md;
    IMAGE_METADATA *mdB = g_sort_images[IDB].md;

    if (mdA == NULL && mdB == NULL) return 0;
    if (mdA == NULL) return 1;
    if (mdB == NULL) return -1;

    switch (g_sort_col)
    {
    case STREAM_SORT_NAME:
        res = strcmp(siA->sname, siB->sname);
        break;

    case STREAM_SORT_TYPE:
        res = (siA->datatype < siB->datatype)
              ? -1
              : (siA->datatype > siB->datatype)
                ? 1 : 0;
        break;

    case STREAM_SORT_SIZE:
    {
        uint64_t neA = mdA->nelement;
        uint64_t neB = mdB->nelement;
        res = (neA < neB) ? -1
              : (neA > neB) ? 1 : 0;
        break;
    }

    case STREAM_SORT_CNT0:
    {
        uint64_t c0A = mdA->cnt0;
        uint64_t c0B = mdB->cnt0;
        res = (c0A < c0B) ? -1
              : (c0A > c0B) ? 1 : 0;
        break;
    }

    case STREAM_SORT_CPID:
        res = (mdA->creatorPID < mdB->creatorPID)
              ? -1
              : (mdA->creatorPID > mdB->creatorPID)
                ? 1 : 0;
        break;

    case STREAM_SORT_OPID:
        res = (mdA->ownerPID < mdB->ownerPID)
              ? -1
              : (mdA->ownerPID > mdB->ownerPID)
                ? 1 : 0;
        break;

    case STREAM_SORT_FREQ:
        res = (siA->frequ_disp < siB->frequ_disp)
              ? -1
              : (siA->frequ_disp > siB->frequ_disp)
                ? 1 : 0;
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
    sTUIparam.loopOK = 1;
    sTUIparam.dindexSelected = 0;
    sTUIparam.DisplayDetailLevel = 0;
    sTUIparam.DisplayMode      = DISPLAY_MODE_SUMMARY;
    sTUIparam.NBsindex = 0;
    sTUIparam.SORTING     = 0;
    sTUIparam.DISPLAY_ALL_SEMS = 1; // Display all semaphores / just the first 2.
    sTUIparam.fuserScan = 0;
    sTUIparam.SORT_TOGGLE = 0;
    sTUIparam.sort_col = STREAM_SORT_NONE;
    sTUIparam.sort_dir = 0;
    sTUIparam.frequ = 10.0; // Hz


    int stringmaxlen = 300;

    // Display fields
    STREAMINFO    *streaminfo;
    STREAMINFOPROC streaminfoproc;


//    long dindex;           // display index
    long doffsetindex = 0; // offset index if more entries than can be displayed


    int monstrlen = 200;
    char  monstring[monstrlen];


    DEBUG_TRACEPOINT("function start ");

    pthread_t threadscan;

    // display
    int DispName_NBchar = 36;
    int DispSize_NBchar = 20;
    int Dispcnt0_NBchar = 16;
    int Dispfreq_NBchar = 8;
    int DispPID_NBchar  = 8;

    // create PID name table
    char **PIDname_array;
    int    PIDmax;

    PIDmax = get_PIDmax();

    DEBUG_TRACEPOINT("PID max = %d ", PIDmax);

    PIDname_array = (char **) malloc(sizeof(char *) * PIDmax);
    for(int pidi = 0; pidi < PIDmax; pidi++)
    {
        PIDname_array[pidi] = NULL;
    }

    streaminfoproc.WriteFlistToFile = 0;
    streaminfoproc.loopcnt          = 0;
    streaminfoproc.fuserUpdate      = 0;

    streaminfo = (STREAMINFO *) malloc(sizeof(STREAMINFO) * streamNBID_MAX);
    for(int sindex = 0; sindex < streamNBID_MAX; sindex++)
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
    streaminfoproc.PIDtable = PIDname_array;

    IMAGE *streamCTRLimages = (IMAGE *) malloc(sizeof(IMAGE) * streamNBID_MAX);
    for(imageID imID = 0; imID < streamNBID_MAX; imID++)
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

    if(getenv("MILK_TUIPRINT_STDIO"))
    {
        // use stdio instead of TUI
        TUI_set_screenprintmode(SCREENPRINT_STDIO);
    }

    if(getenv("MILK_TUIPRINT_NONE"))
    {
        TUI_set_screenprintmode(SCREENPRINT_NONE);
    }

    DEBUG_TRACEPOINT("Initialize terminal");
    TUI_init_terminal(&wrow, &wcol);
    atexit(streamCTRL_TUI_exit);


    long long loopcnt  = 0;


    streaminfoproc.filter       = 0;
    streaminfoproc.NBstream     = 0;
    streaminfoproc.twaitus      = 100000; // 10 Hz
    streaminfoproc.fuserUpdate0 = 1;     //update on first instance

    // inodes that are upstream of current selection
    int    NBupstreaminodeMAX = 100;
    ino_t *upstreaminode;
    int    NBupstreaminode = 0;
    upstreaminode = (ino_t *) malloc(sizeof(ino_t) * NBupstreaminodeMAX);

    // processes that are upstream of current selection
    int    NBupstreamprocMAX = 100;
    pid_t *upstreamproc;
    int    NBupstreamproc = 0;
    upstreamproc          = (pid_t *) malloc(sizeof(pid_t) * NBupstreamprocMAX);

    TUI_clearscreen(0, 0);
    DEBUG_TRACEPOINT(" ");

    // redirect stderr to /dev/null

    int  backstderr;
    int  newstderr;
    char newstderrfname[STRINGMAXLEN_FULLFILENAME];

    fflush(stderr);
    backstderr = dup(STDERR_FILENO);
    WRITE_FULLFILENAME(newstderrfname,
                       "%s/stderr.cli.%d.txt",
                       SHAREDSHMDIR,
                       (int)getpid());

    umask(0);
    newstderr = open(newstderrfname, O_WRONLY | O_CREAT, FILEMODE);
    dup2(newstderr, STDERR_FILENO);
    close(newstderr);

    DEBUG_TRACEPOINT("Start scan thread");
    streaminfoproc.loop = 1;
    pthread_create(&threadscan,
                   NULL,
                   streamCTRL_scan,
                   (void *) &streamCTRLdata);

    DEBUG_TRACEPOINT("Scan thread started");


    loopcnt = 0;

    DEBUG_TRACEPOINT("get terminal size");
    TUI_init_terminal(&wrow, &wcol);
    //        TUI_get_terminal_size(&wrow, &wcol);

    ino_t inodeselected      = 0;

    while(sc_sigINT == 0 && sc_sigTERM == 0 && sTUIparam.loopOK == 1)
    {
        DEBUG_TRACEPOINT("loop start");

        /* Rough estimate used only for the "Currently displaying" status
         * string in the header. The accurate value is computed below
         * via sc_cursor_row after all headers have been drawn. */
        int NBsinfodisp = wrow - 6;
        if(NBsinfodisp < 1)
        {
            NBsinfodisp = 1;
        }

        if(streaminfoproc.loopcnt == 1)
        {
            sTUIparam.SORTING     = 2;
            sTUIparam.SORT_TOGGLE = 1;
        }
        DEBUG_TRACEPOINT(" ");

        //if(fuserUpdate != 1) // don't wait if ongoing fuser scan

        {
            struct timeval tv;
            tv.tv_sec = 0;
            tv.tv_usec = (long)(1000000.0 / sTUIparam.frequ);

            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(STDIN_FILENO, &fds);

            select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv);
        }
        DEBUG_TRACEPOINT(" ");

        int ch;
        while((ch = ansi_get_key()) != ANSI_KEY_NONE)
        {
            DEBUG_TRACEPOINT("Process input character");
            streamCTRL_keyinput_process(ch, &streamCTRLdata);
            DEBUG_TRACEPOINT("Input character processed");
        }

        sTUIparam.NBsindex = streaminfoproc.NBstream;
        DEBUG_TRACEPOINT(" ");

        TUI_clearscreen(&wrow, &wcol);

        if(sTUIparam.dindexSelected < 0)
        {
            sTUIparam.dindexSelected = 0;
        }
        if(sTUIparam.dindexSelected > sTUIparam.NBsindex - 1)
        {
            sTUIparam.dindexSelected = sTUIparam.NBsindex - 1;
        }

        DEBUG_TRACEPOINT("Erase screen");

        //attron(A_BOLD);
        screenprint_setbold();
        snprintf(monstring,
                 monstrlen,
                 "[%d x %d] [PID %d] STREAM MONITOR: PRESS (x) TO STOP, (h) "
                 "FOR HELP",
                 wrow,
                 wcol,
                 getpid());
        //streamCTRL__print_header(monstring, '-');
        DEBUG_TRACEPOINT("Print header");
        screenprint_setcolor(12);
        TUI_print_header(monstring, '-');
        screenprint_unsetcolor(12);
        //attroff(A_BOLD);
        screenprint_unsetbold();

        DEBUG_TRACEPOINT("Start display");

        if(sTUIparam.DisplayMode == DISPLAY_MODE_HELP)  // help
        {
            //int attrval = A_BOLD;

            DEBUG_TRACEPOINT(" ");


            print_help_entry("x", "Exit");

            TUI_newline();
            TUI_printfw("============ SCREENS");
            TUI_newline();
            print_help_entry("h", "help");
            print_help_entry("F2", "semaphore values");
            print_help_entry("F3", "semaphore read  PIDs");
            print_help_entry("F4", "semaphore write PIDs");
            print_help_entry("F5", "stream process trace");
            print_help_entry("F6", "stream open by processes ...");
            print_help_entry("CTRL+L/R", "cycle between tabs");

            TUI_newline();
            TUI_printfw("============ ACTIONS");
            TUI_newline();
            print_help_entry("CTRL+e", "Erase stream");

            TUI_newline();
            TUI_printfw("============ SCANNING");
            TUI_newline();
            print_help_entry("}", "Increase scan frequency");
            print_help_entry("{", "Decrease scan frequency");
            print_help_entry("o", "output next scan to file");

            TUI_newline();
            TUI_printfw("============ DISPLAY");
            TUI_newline();
            print_help_entry("+/-", "Increase/decrease display frequency");
            print_help_entry("]", "Cycle sort column (name,type,size...)");
            print_help_entry("[", "Toggle sort direction (asc/desc)");
            print_help_entry("1", "Sort by stream name (alphabetical)");
            print_help_entry("2", "Sort by recently updated");
            print_help_entry("3", "Sort by process access");
            print_help_entry("4", "Sort by frequency (descending)");
            print_help_entry("s", "Show 3 semaphores / all semaphores");
            print_help_entry("r", "Force full screen redraw");
            print_help_entry("F", "Set match string pattern");
            print_help_entry("f", "Toggle apply match string to stream");
        }
        else
        {
            DEBUG_TRACEPOINT(" ");
            /* Tab bar — rendered from TAB_LIST */
#define RENDER_ONE_TAB(mode, key, label) \
    { \
        screenprint_setcolor(7); \
        if (sTUIparam.DisplayMode == (mode)) \
            screenprint_setreverse(); \
        TUI_printfw("[%s] %s", (key), (label)); \
        if (sTUIparam.DisplayMode == (mode)) \
            screenprint_unsetreverse(); \
        screenprint_unsetcolor(7); \
        TUI_printfw("   "); \
    }

            TAB_LIST(RENDER_ONE_TAB)
#undef RENDER_ONE_TAB
            TUI_newline();

            TUI_printfw(
                "PIDmax = %d    Update frequ = %2d Hz  fscan=%5.2f Hz "
                "( %5.2f Hz %5.2f %% busy ) ",
                PIDmax,
                (int)(sTUIparam.frequ + 0.5),
                1.0 / streaminfoproc.dtscan,
                1000000.0 / streaminfoproc.twaitus,
                100.0 *
                (streaminfoproc.dtscan - 1.0e-6 * streaminfoproc.twaitus) /
                streaminfoproc.dtscan);

            if(streaminfoproc.fuserUpdate == 1)
            {
                screenprint_setcolor(9);
                TUI_printfw("fuser scan ongoing  %4d  / %4d   ",
                            streaminfoproc.sindexscan,
                            sTUIparam.NBsindex);
                screenprint_unsetcolor(9);
            }
            if(sTUIparam.DisplayMode == DISPLAY_MODE_FUSER)
            {
                if(sTUIparam.fuserScan == 1)
                {
                    TUI_printfw(
                        "Last scan on  %02d:%02d:%02d  - Press "
                        "F6 again to re-scan    C-c to stop "
                        "scan",
                        sTUIparam.uttime_lastScan->tm_hour,
                        sTUIparam.uttime_lastScan->tm_min,
                        sTUIparam.uttime_lastScan->tm_sec);
                    TUI_newline();
                }
                else
                {
                    TUI_printfw(
                        "Last scan on  XX:XX:XX  - Press F6 "
                        "again to scan             C-c to stop "
                        "scan");
                    TUI_newline();
                }
            }
            else
            {
                /* Sort status indicator */
                static const char *sort_col_names[] = {
                    "", "NAME", "TYPE", "SIZE",
                    "CNT0", "CPID", "OPID", "FREQ"
                };

                if (sTUIparam.sort_col > 0
                    && sTUIparam.sort_col
                       <= STREAM_NB_SORT_COLS)
                {
                    screenprint_setcolor(6);
                    TUI_printfw(
                        "[SORT: %s %s]  ",
                        sort_col_names[
                            sTUIparam.sort_col],
                        sTUIparam.sort_dir
                            ? "DESC" : "ASC");
                    screenprint_unsetcolor(6);
                }
                else if (sTUIparam.SORTING > 0)
                {
                    screenprint_setcolor(6);
                    TUI_printfw(
                        "[SORT: mode %d]  ",
                        sTUIparam.SORTING);
                    screenprint_unsetcolor(6);
                }

                screenprint_setbold();
                TUI_printfw("]");
                screenprint_unsetbold();
                TUI_printfw(" cycle col  ");
                screenprint_setbold();
                TUI_printfw("[");
                screenprint_unsetbold();
                TUI_printfw(" flip dir");
                TUI_newline();
            }

            int lastindex;
            lastindex = doffsetindex + NBsinfodisp;
            if(lastindex > sTUIparam.NBsindex - 1)
            {
                lastindex = sTUIparam.NBsindex - 1;
            }

            if(lastindex < 0)
            {
                lastindex = 0;
            }

            {
                int ssIDselected = -1;
                if(sTUIparam.dindexSelected >= 0)
                {
                    ssIDselected = sTUIparam.ssindex[sTUIparam.dindexSelected];
                }

                TUI_printfw(
                    "%4d streams    Currently displaying %4ld-%4ld   "
                    "Selected %d  ID = %d  inode = %d",
                    sTUIparam.NBsindex,
                    (long)doffsetindex,
                    (long)lastindex,
                    sTUIparam.dindexSelected,
                    ssIDselected,
                    (int) inodeselected);
            }

            if(streaminfoproc.filter == 1)
            {
                screenprint_setcolor(9);
                TUI_printfw("  Filter = \"%s\"", streaminfoproc.namefilter);
                screenprint_unsetcolor(9);
            }

            TUI_newline();

            // attron(A_BOLD);

            TUI_printfw("%*s  %-*s  %-*s  %*s   %*s %*s %*s",
                        9,
                        "inode",
                        DispName_NBchar,
                        "name",
                        DispSize_NBchar,
                        "type",
                        Dispcnt0_NBchar,
                        "cnt0",
                        DispPID_NBchar,
                        "creaPID",
                        DispPID_NBchar,
                        "ownPID",
                        Dispfreq_NBchar,
                        "   frequ ");

            switch(sTUIparam.DisplayMode)
            {
            case DISPLAY_MODE_SUMMARY:
                TUI_printfw("     Semaphore values ....");
                TUI_newline();
                break;

            case DISPLAY_MODE_WRITE:
                TUI_printfw("     write PIDs ....");
                TUI_newline();
                break;

            case DISPLAY_MODE_READ:
                TUI_printfw("     read PIDs ....");
                TUI_newline();
                break;

            case DISPLAY_MODE_SPTRACE:
                TUI_printfw(
                    "     stream process traces:   \"(INODE "
                    "TYPE/SEM PID)>\"");
                TUI_newline();
                break;

            case DISPLAY_MODE_FUSER:
                TUI_printfw("     connected processes");
                TUI_newline();
                break;

            default:
                TUI_newline();
                break;
            }

            screenprint_unsetbold();

            /* Recompute exact list height now that all header rows have
             * been drawn.  sc_cursor_row is the next row to be written;
             * we want list rows from here to (wrow - 1), keeping the
             * very last row (wrow) for the footer bar. */
            NBsinfodisp = (int)wrow - sc_cursor_row - 1;
            if(NBsinfodisp < 1)
            {
                NBsinfodisp = 1;
            }

            /* Recompute lastindex with accurate NBsinfodisp. */
            lastindex = doffsetindex + NBsinfodisp;
            if(lastindex > sTUIparam.NBsindex - 1)
            {
                lastindex = sTUIparam.NBsindex - 1;
            }
            if(lastindex < 0)
            {
                lastindex = 0;
            }

            // SORT

            // build active streams array
            sTUIparam.NBsindex = 0;
            for(int sindex = 0; sindex < streaminfoproc.NBstream; sindex++)
            {
                if(streaminfo[sindex].erased == 1) continue;
                imageID ID = streaminfo[sindex].ID;
                /* ID == -1 means discovered but not yet mmap'd.
                 * Include it so names appear immediately. */
                if(ID >= 0 && streamCTRLimages[ID].used == 0) continue;

                sTUIparam.ssindex[sTUIparam.NBsindex] = sindex;
                sTUIparam.NBsindex++;
            }

            DEBUG_TRACEPOINT(" ");

            // compute dynamic lengths
            int max_name_len = 10;
            for(int dindex = 0; dindex < sTUIparam.NBsindex; dindex++)
            {
                int len = strlen(streaminfo[sTUIparam.ssindex[dindex]].sname);
                if (len > max_name_len) max_name_len = len;
            }
            DispName_NBchar = max_name_len + 2;

            /* ---- Column-based sort (new) ---- */
            if (sTUIparam.sort_col > STREAM_SORT_NONE
                && sTUIparam.SORTING == 0)
            {
                g_streaminfo_qsort = streaminfo;
                g_sort_images = streamCTRLimages;
                g_sort_col = sTUIparam.sort_col;
                g_sort_dir = sTUIparam.sort_dir;

                qsort(sTUIparam.ssindex,
                      sTUIparam.NBsindex,
                      sizeof(long),
                      cmp_stream_col);
            }

            /* ---- Legacy sort mode 1: alphabetical ---- */
            if (sTUIparam.SORTING == 1)
            {
                g_streaminfo_qsort = streaminfo;
                g_sort_images = streamCTRLimages;
                g_sort_col = STREAM_SORT_NAME;
                g_sort_dir = 0;

                qsort(sTUIparam.ssindex,
                      sTUIparam.NBsindex,
                      sizeof(long),
                      cmp_stream_col);
            }

            /* ---- Legacy sort modes 2/3: update recency ---- */
            if ((sTUIparam.SORTING == 2) ||
                    (sTUIparam.SORTING == 3))
            {
                long   *larray;
                double *varray;
                larray = (long *) malloc(
                    sizeof(long) * sTUIparam.NBsindex);
                varray = (double *) malloc(
                    sizeof(double) * sTUIparam.NBsindex);

                if (sTUIparam.SORT_TOGGLE == 1)
                {
                    for (long i = 0;
                         i < sTUIparam.NBsindex; i++)
                    {
                        long si = sTUIparam.ssindex[i];
                        streaminfo[si]
                            .updatevalue_frozen =
                            streaminfo[si].updatevalue;
                    }

                    if (sTUIparam.SORTING == 3)
                    {
                        for (long i = 0;
                             i < sTUIparam.NBsindex; i++)
                        {
                            long si =
                                sTUIparam.ssindex[i];
                            streaminfo[si]
                                .updatevalue_frozen +=
                                10000.0 *
                                streaminfo[si]
                                    .streamOpenPID_cnt1;
                        }
                    }

                    sTUIparam.SORT_TOGGLE = 0;
                }

                for (long i = 0;
                     i < sTUIparam.NBsindex; i++)
                {
                    long si = sTUIparam.ssindex[i];
                    larray[i] = si;
                    varray[i] =
                        streaminfo[si]
                            .updatevalue_frozen;
                }

                if (sTUIparam.NBsindex > 1)
                {
                    quick_sort2l(varray, larray,
                                 sTUIparam.NBsindex);
                }

                for (long i = 0;
                     i < sTUIparam.NBsindex; i++)
                {
                    sTUIparam.ssindex[
                        sTUIparam.NBsindex - i - 1] =
                        larray[i];
                }

                free(larray);
                free(varray);
            }

            DEBUG_TRACEPOINT(" ");

            // compute doffsetindex
            // Clamp scroll margins for small terminals
            {
                int margin_dn = 5;
                int margin_up = 10;

                if(margin_dn >= NBsinfodisp)
                {
                    margin_dn = NBsinfodisp - 1;
                }
                if(margin_dn < 0)
                {
                    margin_dn = 0;
                }
                if(margin_up >= NBsinfodisp)
                {
                    margin_up = NBsinfodisp - 1;
                }
                if(margin_up < 0)
                {
                    margin_up = 0;
                }

                while(sTUIparam.dindexSelected - doffsetindex >
                        NBsinfodisp - 1 - margin_dn)
                {
                    doffsetindex++;
                }

                while(sTUIparam.dindexSelected <
                        doffsetindex + margin_up)
                {
                    doffsetindex--;
                }
            }

            // Ensure selected item is always visible
            if(sTUIparam.dindexSelected < doffsetindex)
            {
                doffsetindex = sTUIparam.dindexSelected;
            }
            if(sTUIparam.dindexSelected >=
                    doffsetindex + NBsinfodisp)
            {
                doffsetindex =
                    sTUIparam.dindexSelected - NBsinfodisp + 1;
            }

            {
                long max_doffsetindex = sTUIparam.NBsindex - NBsinfodisp;
                if(max_doffsetindex < 0)
                {
                    max_doffsetindex = 0;
                }
                if(doffsetindex > max_doffsetindex)
                {
                    doffsetindex = max_doffsetindex;
                }
            }

            if(doffsetindex < 0)
            {
                doffsetindex = 0;
            }


            // DISPLAY
            //
            //

            /* Column header row with sort indicators */
            if (sTUIparam.DisplayMode < DISPLAY_MODE_FUSER)
            {
                /* Column labels and their sort IDs */
                struct {
                    const char *label;
                    int col_id;
                    int width;
                } cols[] = {
                    {"NAME",  STREAM_SORT_NAME, DispName_NBchar},
                    {"TYPE",  STREAM_SORT_TYPE, 4},
                    {"SIZE",  STREAM_SORT_SIZE, DispSize_NBchar},
                    {"CNT0",  STREAM_SORT_CNT0, Dispcnt0_NBchar+2},
                    {"CPID",  STREAM_SORT_CPID, 9},
                    {"OPID",  STREAM_SORT_OPID, 9},
                    {"FREQ",  STREAM_SORT_FREQ, 9},
                };
                int ncols = 7;

                /* inode placeholder */
                TUI_printfw("          ");

                for (int ci = 0; ci < ncols; ci++)
                {
                    int is_active =
                        (cols[ci].col_id ==
                         sTUIparam.sort_col);

                    if (is_active)
                    {
                        screenprint_setbold();
                        screenprint_setcolor(6);
                    }

                    char arrow = ' ';
                    if (is_active)
                    {
                        arrow = sTUIparam.sort_dir
                                ? '\x19' : '\x18';
                    }

                    TUI_printfw("%-*.*s%c",
                                cols[ci].width - 1,
                                cols[ci].width - 1,
                                cols[ci].label,
                                arrow);

                    if (is_active)
                    {
                        screenprint_unsetcolor(6);
                        screenprint_unsetbold();
                    }
                }

                TUI_newline();
            }

            int DisplayFlag = 0;

            int print_pid_mode = PRINT_PID_DEFAULT;

            /* Hoist time and color-level detection out of the per-stream loop
             * to avoid N system calls and repeated env-var checks per frame. */
            struct timespec frame_ts;
            clock_gettime(CLOCK_MONOTONIC, &frame_ts);
            double frame_t_sec = frame_ts.tv_sec + frame_ts.tv_nsec * 1e-9;

            ansi_detect_color_level();
            int frame_color_level = ansi__color_level;


            for(int dindex = 0; dindex < sTUIparam.NBsindex; dindex++)
            {
                imageID ID;
                int sindex = sTUIparam.ssindex[dindex];
                ID     = streaminfo[sindex].ID;

                int downstreammin = NO_DOWNSTREAM_INDEX;
                // minumum downstream index
                // looks for inodeselected in the list of upstream inodes
                // picks the smallest corresponding index
                // for example, if equal to 3, the current inode is a 3-rd gen children of selected inode
                // default initial value 100 is a placeholder indicating it is not a child

                DEBUG_TRACEPOINT(" ");

                if((dindex >= doffsetindex) &&
                        (dindex < NBsinfodisp + doffsetindex))
                {
                    DisplayFlag = 1;
                }
                else
                {
                    DisplayFlag = 0;
                }

                if(sTUIparam.DisplayDetailLevel == 1)
                {
                    if(dindex == sTUIparam.dindexSelected)
                    {
                        DisplayFlag = 1;
                    }
                    else
                    {
                        DisplayFlag = 0;
                    }
                }

                DEBUG_TRACEPOINT(" ");

                /* Stream name discovered but SHM not yet opened. Show
                 * just the name in dim style until connection is ready. */
                if(ID < 0)
                {
                    if(DisplayFlag == 1)
                    {
                        if(dindex == sTUIparam.dindexSelected)
                        {
                            screenprint_setreverse();
                        }

                        screenprint_setcolor(4);
                        TUI_printfw("          %-*.*s  ...",
                                    DispName_NBchar,
                                    DispName_NBchar,
                                    streaminfo[sindex].sname);
                        screenprint_unsetcolor(4);

                        if(dindex == sTUIparam.dindexSelected)
                        {
                            screenprint_unsetreverse();
                        }
                        
                        TUI_newline();
                    }
                    continue;
                }

                // Stream is guaranteed active and not erased


                if(streaminfo[sindex].ISIOretval != IMAGESTREAMIO_SUCCESS)
                {
                    if(DisplayFlag == 1)
                    {
                        TUI_printfw("          ");


                        if((dindex == sTUIparam.dindexSelected) && (sTUIparam.DisplayDetailLevel == 0))
                        {
                            screenprint_setreverse();
                        }

                        TUI_printfw("%-*.*s",
                                    DispName_NBchar,
                                    DispName_NBchar,
                                    streaminfo[sindex].sname);


                        screenprint_setcolor(4);
                        TUI_printfw("ERROR:");
                        screenprint_unsetcolor(4);
                        TUI_printfw("  ");

                        switch(streaminfo[sindex].ISIOretval)
                        {
                        case IMAGESTREAMIO_FILEOPEN :
                            TUI_printfw("cannot open file");
                            break;

                        case IMAGESTREAMIO_VERSION :
                            TUI_printfw("incompatible ISIO version");
                            break;

                        case IMAGESTREAMIO_FAILURE:
                            TUI_printfw("failed verification");
                            break;
                        }


                        if(dindex == sTUIparam.dindexSelected)
                        {
                            screenprint_unsetreverse();
                        }

                        TUI_newline();
                    }

                }
                else
                {
                    if(dindex == sTUIparam.dindexSelected)
                    {
                        DEBUG_TRACEPOINT(
                            "dindex %d %d",
                            dindex,
                            streamCTRLimages[streaminfo[sindex].ID].used);

                        // currently selected inode
                        inodeselected =
                            streamCTRLimages[streaminfo[sindex].ID].md->inode;

                        DEBUG_TRACEPOINT(
                            "inode %lu %s",
                            inodeselected,
                            streamCTRLimages[streaminfo[sindex].ID].md->name);

                        // identify upstream inodes
                        NBupstreaminode = 0;
                        for(int spti = 0;
                                spti < streamCTRLimages[ID].md[0].NBproctrace;
                                spti++)
                        {
                            if(NBupstreaminode < NBupstreaminodeMAX)
                            {
                                ino_t inode = streamCTRLimages[ID]
                                              .streamproctrace[spti]
                                              .trigger_inode;
                                if(inode != 0)
                                {
                                    upstreaminode[NBupstreaminode] = inode;
                                    NBupstreaminode++;
                                }
                            }
                        }

                        DEBUG_TRACEPOINT(" ");

                        // identify upstream processes
                        print_pid_mode = PRINT_PID_FORCE_NOUPSTREAM;
                        NBupstreamproc = 0;
                        for(int spti = 0;
                                spti < streamCTRLimages[ID].md[0].NBproctrace;
                                spti++)
                        {
                            if(NBupstreamproc < NBupstreamprocMAX)
                            {
                                ino_t procpid = streamCTRLimages[ID]
                                                .streamproctrace[spti]
                                                .procwrite_PID;
                                if(procpid > 0)
                                {
                                    upstreamproc[NBupstreamproc] = procpid;
                                    NBupstreamproc++;
                                }
                            }

                            DEBUG_TRACEPOINT(" ");
                        }
                    }
                    else
                    {
                        if(DisplayFlag == 1)
                        {
                            DEBUG_TRACEPOINT("%d, %s, ID = %ld, used = %d, name= %s, ISIOcode= %d (OK = %d)",
                                             sindex,
                                             streaminfo[sindex].sname,
                                             streaminfo[sindex].ID,
                                             streamCTRLimages[ID].used,
                                             streamCTRLimages[ID].name,
                                             streaminfo[sindex].ISIOretval,
                                             IMAGESTREAMIO_SUCCESS);

                            print_pid_mode = PRINT_PID_DEFAULT;
                            if(streamCTRLimages[ID].used == 1)
                            {
                                for(int spti = 0;
                                        spti < streamCTRLimages[ID].md->NBproctrace;
                                        spti++)
                                {
                                    ino_t inode = streamCTRLimages[ID]
                                                  .streamproctrace[spti]
                                                  .trigger_inode;
                                    if(inode == inodeselected)
                                    {
                                        if(spti < downstreammin)
                                        {
                                            downstreammin = spti;
                                        }
                                    }
                                }
                            }
                            DEBUG_TRACEPOINT(" ");
                        }
                    }

                    DEBUG_TRACEPOINT(" ");

                    int stringlen = 200;
                    char string[stringlen];

                    if(DisplayFlag == 1)
                    {
                        // print file inode
                        if(streamCTRLimages[ID].used == 1)
                        {
                            streamCTRL_print_inode(streamCTRLimages[ID].md[0].inode,
                                                   upstreaminode,
                                                   NBupstreaminode,
                                                   downstreammin);
                        }
                        TUI_printfw(" ");
                    }

                    if((dindex == sTUIparam.dindexSelected) && (sTUIparam.DisplayDetailLevel == 0))
                    {
                        screenprint_setreverse();
                    }

                    DEBUG_TRACEPOINT(" ");

                    if(DisplayFlag == 1)
                    {
                        if(streaminfo[sindex].SymLink == 1)
                        {
                            char namestring[stringmaxlen];

                            snprintf(namestring,
                                     stringmaxlen,
                                     "%s->%s",
                                     streaminfo[sindex].sname,
                                     streaminfo[sindex].linkname);

                            screenprint_setbold();
                            screenprint_setcolor(5);
                            TUI_printfw("%-*.*s",
                                        DispName_NBchar,
                                        DispName_NBchar,
                                        namestring);
                            screenprint_unsetcolor(5);
                            screenprint_unsetbold();
                        }
                        else
                        {
                            screenprint_setbold();
                            TUI_printfw("%-*.*s",
                                        DispName_NBchar,
                                        DispName_NBchar,
                                        streaminfo[sindex].sname);
                            screenprint_unsetbold();
                        }

                        /*if((int) strlen(streaminfo[sindex].sname) > DispName_NBchar)
                        {
                            attron(COLOR_PAIR(9));
                            TUI_printfw("+");
                            attroff(COLOR_PAIR(9));
                        }
                        else
                        {
                            TUI_printfw(" ");
                        }*/
                    }

                    DEBUG_TRACEPOINT(" ");

                    if((sTUIparam.DisplayMode < DISPLAY_MODE_FUSER) && (DisplayFlag == 1))
                    {
                        char str[STRINGMAXLEN_DEFAULT];
                        char str1[STRINGMAXLEN_DEFAULT];
                        int  j;

                        if(streamCTRLimages[streaminfo[sindex].ID].md == NULL)
                        {
                            snprintf(string, stringlen, " ???");
                        }
                        else
                        {
                            snprintf(string, stringlen, "%s",
                                     ImageStreamIO_typename_short(streaminfo[sindex].datatype));
                        }
                        TUI_printfw("%s", string);

                        DEBUG_TRACEPOINT(" ");
                        if(streamCTRLimages[streaminfo[sindex].ID].md == NULL)
                        {
                            snprintf(str, stringlen, "???");
                        }
                        else
                        {
                            snprintf(str,
                                     stringlen,
                                     " [%3ld",
                                     (long) streamCTRLimages[ID].md[0].size[0]);

                            for(j = 1; j < streamCTRLimages[ID].md[0].naxis; j++)
                            {
                                {
                                    int slen = snprintf(
                                                   str1,
                                                   STRINGMAXLEN_DEFAULT,
                                                   "%sx%3ld",
                                                   str,
                                                   (long) streamCTRLimages[ID].md[0].size[j]);
                                    if(slen < 1)
                                    {
                                        PRINT_ERROR(
                                            "snprintf "
                                            "wrote <1 "
                                            "char");
                                        abort(); // can't handle this error any other way
                                    }
                                    if(slen >= STRINGMAXLEN_DEFAULT)
                                    {
                                        PRINT_ERROR(
                                            "snprintf "
                                            "string "
                                            "truncatio"
                                            "n");
                                        abort(); // can't handle this error any other way
                                    }
                                }
                                snprintf(str,
                                         STRINGMAXLEN_DEFAULT,
                                         "%s", str1);
                            }
                            {
                                int slen = snprintf(str1,
                                                    STRINGMAXLEN_DEFAULT,
                                                    "%s]",
                                                    str);
                                if(slen < 1)
                                {
                                    PRINT_ERROR(
                                        "snprintf wrote <1 "
                                        "char");
                                    abort(); // can't handle this error any other way
                                }
                                if(slen >= STRINGMAXLEN_DEFAULT)
                                {
                                    PRINT_ERROR(
                                        "snprintf string "
                                        "truncation");
                                    abort(); // can't handle this error any other way
                                }
                            }

                            snprintf(str,
                                     STRINGMAXLEN_DEFAULT,
                                     "%s", str1);
                        }

                        DEBUG_TRACEPOINT(" ");

                        snprintf(string,
                                 stringlen,
                                 "%-*.*s ",
                                 DispSize_NBchar,
                                 DispSize_NBchar,
                                 str);
                        TUI_printfw("%s", string);

                        if(streamCTRLimages[streaminfo[sindex].ID].md == NULL)
                        {
                            snprintf(string, stringlen, " %*s ", Dispcnt0_NBchar, "???");
                        }
                        else
                        {

                            snprintf(string,
                                     stringlen,
                                     " %*ld ",
                                     Dispcnt0_NBchar,
                                     streamCTRLimages[ID].md[0].cnt0);
                        }
                        
                        double t_sec = frame_t_sec;

                        /* Update holdoff timestamp whenever the stream is active. */
                        if(streaminfo[sindex].deltacnt0 != 0)
                        {
                            streaminfo[sindex].last_wave_t = t_sec;
                        }

                        double wave_age = t_sec - streaminfo[sindex].last_wave_t;

                        /* 1-second block average of cnt0 update frequency.
                         * When the 1-s window expires, compute freq from the
                         * accumulated Δcnt0 and reset the window. */
                        if(streamCTRLimages[ID].md != NULL)
                        {
                            uint64_t cnt0now = streamCTRLimages[ID].md[0].cnt0;
                            double   dt_avg  = t_sec
                                               - streaminfo[sindex].t_avg_start;

                            if(dt_avg >= 1.0)
                            {
                                uint64_t dcnt = cnt0now
                                                - streaminfo[sindex].cnt0_avg_start;
                                streaminfo[sindex].frequ_disp     =
                                    (double) dcnt / dt_avg;
                                streaminfo[sindex].cnt0_avg_start = cnt0now;
                                streaminfo[sindex].t_avg_start    = t_sec;
                            }
                        }

                        /* Highlight cnt0 field when counter is
                         * actively changing (within 1s holdoff). */
                        if(wave_age <= 1.0 && frame_color_level >= 2)
                        {
                            int len_cnt = strlen(string);
                            screenprint_setcolor(2);
                            streamCTRL_render_active_bg(
                                string, len_cnt,
                                frame_color_level);
                            SC_APPEND("\033[0m");

                            if((dindex ==
                                sTUIparam.dindexSelected) &&
                               (sTUIparam.DisplayDetailLevel
                                == 0))
                            {
                                screenprint_setreverse();
                            }
                        }
                        else
                        {
                            TUI_printfw("%s", string);
                        }



                        // creatorPID
                        // ownerPID
                        if(streamCTRLimages[streaminfo[sindex].ID].md == NULL)
                        {
                            snprintf(string, stringlen, "???");
                        }
                        else
                        {
                            pid_t cpid; // creator PID
                            pid_t opid; // owner PID

                            cpid = streamCTRLimages[ID].md[0].creatorPID;
                            opid = streamCTRLimages[ID].md[0].ownerPID;

                            streamCTRL_print_procpid(8,
                                                     cpid,
                                                     upstreamproc,
                                                     NBupstreamproc,
                                                     print_pid_mode);
                            TUI_printfw(" ");
                            streamCTRL_print_procpid(8,
                                                     opid,
                                                     upstreamproc,
                                                     NBupstreamproc,
                                                     print_pid_mode);
                            TUI_printfw(" ");
                        }

                        // stream update frequency
                        //
                        if(streamCTRLimages[streaminfo[sindex].ID].md == NULL)
                        {
                            snprintf(string, stringlen, "???");
                            TUI_printfw("%s", string);
                        }
                        else
                        {
                            streamCTRL_print_frequ_field(
                                streaminfo[sindex].frequ_disp,
                                wave_age,
                                frame_color_level);
                        }
                    }

                    DEBUG_TRACEPOINT(" ");

                    if(streamCTRLimages[streaminfo[sindex].ID].md != NULL)
                    {
                        if((sTUIparam.DisplayMode == DISPLAY_MODE_SUMMARY) &&
                                (DisplayFlag == 1)) // sem vals
                        {

                            int s;
                            int max_s = sTUIparam.DISPLAY_ALL_SEMS
                                        ? streamCTRLimages[ID].md[0].sem
                                        : 3;
                            TUI_printfw(" ");
                            for(s = 0; s < max_s; s++)
                            {
                                int semval = ImageStreamIO_semvalue(streamCTRLimages + ID, s);
                                if (s > 0) {
                                    TUI_printfw(":");
                                }
                                streamCTRL_set_sem_color(semval);
                                snprintf(string, stringlen, "%02d", semval);
                                TUI_printfw("%s", string);
                                screenprint_unsetcolor(0);
                            }
                        }
                    }

                    DEBUG_TRACEPOINT(" ");

                    if(streamCTRLimages[streaminfo[sindex].ID].md != NULL)
                    {
                        if((sTUIparam.DisplayMode == DISPLAY_MODE_WRITE) &&
                                (DisplayFlag == 1)) // sem write PIDs
                        {
                            {
                                pid_t pid = streamCTRLimages[ID].semWritePID[0];
                                TUI_printfw(" ");
                                streamCTRL_print_procpid(8,
                                                         pid,
                                                         upstreamproc,
                                                         NBupstreamproc,
                                                         print_pid_mode);
                            }

                            if(sTUIparam.DisplayDetailLevel == 1)
                            {
#ifdef IMAGESTRUCT_WRITEHISTORY
                                TUI_newline();
                                TUI_printfw("WRITE timings :");
                                TUI_newline();
                                int windexref = streamCTRLimages[ID].md->wCBindex;
                                double tdouble0 = 0.0;

                                double *dtarray = (double *) malloc(sizeof(double) *
                                                                    (IMAGESTRUCT_FRAMEWRITEMDSIZE - 2));

                                double tdoubleprev = 0.0;
                                double deltatsum = 0.0;
                                double deltatsum2 = 0.0;
                                for(int wioffset = 0; wioffset < IMAGESTRUCT_FRAMEWRITEMDSIZE - 1; wioffset ++)
                                {
                                    int windex = windexref - wioffset;
                                    if(windex < 0)
                                    {
                                        windex += IMAGESTRUCT_FRAMEWRITEMDSIZE;
                                    }
                                    double tdouble = 1.0 * streamCTRLimages[ID].writehist[windex].writetime.tv_sec
                                                     + 1.0e-9 * streamCTRLimages[ID].writehist[windex].writetime.tv_nsec;
                                    double deltat = 0.0;

                                    if(wioffset == 0)
                                    {
                                        tdouble0 = tdouble;
                                        deltat = 0.0;
                                    }
                                    else
                                    {
                                        deltat = tdoubleprev - tdouble;
                                        dtarray[wioffset - 1] = deltat;
                                        deltatsum += deltat;
                                        deltatsum2 += deltat * deltat;
                                    }

                                    if(wioffset < 10)
                                    {
                                        TUI_printfw("%4d  cnt0 %8d  PID %6d  ts %9ld.%09ld   %.9f s ago  delta = %9.3f us",
                                                    wioffset,
                                                    streamCTRLimages[ID].writehist[windex].cnt0,
                                                    streamCTRLimages[ID].writehist[windex].wpid,
                                                    streamCTRLimages[ID].writehist[windex].writetime.tv_sec,
                                                    streamCTRLimages[ID].writehist[windex].writetime.tv_nsec,
                                                    tdouble0 - tdouble,
                                                    1.0e6 * (deltat));
                                        TUI_newline();
                                    }
                                    tdoubleprev = tdouble;
                                }

                                quick_sort_double(dtarray, IMAGESTRUCT_FRAMEWRITEMDSIZE - 2);

                                TUI_newline();

                                TUI_printfw("delta time (nbsample = %d):", IMAGESTRUCT_FRAMEWRITEMDSIZE);
                                TUI_newline();

                                double tave = 1.0e6 * deltatsum / (IMAGESTRUCT_FRAMEWRITEMDSIZE - 2);
                                TUI_printfw("AVERAGE =        %9.3f us", tave);
                                TUI_newline();

                                double trms = deltatsum2 - deltatsum * deltatsum / (IMAGESTRUCT_FRAMEWRITEMDSIZE
                                              - 2);
                                trms = 1.0e6 * sqrt(trms / (IMAGESTRUCT_FRAMEWRITEMDSIZE - 2));
                                TUI_printfw("RMS     =        %9.3f us  ( %8.3f %% )", trms,
                                            100.0 * trms / tave);
                                TUI_newline();

                                double p0us = 1.0e6 * dtarray[0];
                                TUI_printfw("  min          : %9.3f us    %9.3f us", p0us, p0us - tave);
                                TUI_newline();

                                double p10us = 1.0e6 * dtarray[(int)(0.1 * (IMAGESTRUCT_FRAMEWRITEMDSIZE - 2))];
                                TUI_printfw("  p10          : %9.3f us    %9.3f us", p10us, p10us - tave);
                                TUI_newline();

                                double p50us = 1.0e6 * dtarray[(IMAGESTRUCT_FRAMEWRITEMDSIZE - 2) / 2];
                                TUI_printfw("  p50 (median) : %9.3f us    %9.3f us", p50us, p50us - tave);
                                TUI_newline();

                                double p90us = 1.0e6 * dtarray[(int)(0.9 * (IMAGESTRUCT_FRAMEWRITEMDSIZE - 2))];
                                TUI_printfw("  p90          : %9.3f us    %9.3f us", p90us, p90us - tave);
                                TUI_newline();

                                double p100us = 1.0e6 * dtarray[IMAGESTRUCT_FRAMEWRITEMDSIZE - 3];
                                TUI_printfw("  max          : %9.3f us    %9.3f us", p100us, p100us - tave);
                                TUI_newline();


                                free(dtarray);
#endif
                            }
                        }
                    }

                    DEBUG_TRACEPOINT(" ");

                    if(streamCTRLimages[streaminfo[sindex].ID].md != NULL)
                    {
                        if((sTUIparam.DisplayMode == DISPLAY_MODE_READ) &&
                                (DisplayFlag == 1)) // sem read PIDs
                        {
                            int s;
                            int max_s = sTUIparam.DISPLAY_ALL_SEMS
                                        ? streamCTRLimages[ID].md[0].sem
                                        : 3;
                            TUI_printfw(" ");
                            for(s = 0; s < max_s; s++)
                            {
                                pid_t pid = streamCTRLimages[ID].semReadPID[s];
                                if (s > 0) TUI_printfw(":");
                                streamCTRL_print_procpid(0, // 0 for minimal width
                                                         pid,
                                                         upstreamproc,
                                                         NBupstreamproc,
                                                         print_pid_mode);
                            }
                        }
                    }

                    DEBUG_TRACEPOINT(" ");

                    if(streamCTRLimages[streaminfo[sindex].ID].md != NULL)
                    {
                        if((sTUIparam.DisplayMode == DISPLAY_MODE_SPTRACE) &&
                                (DisplayFlag == 1))
                        {
                            DEBUG_TRACEPOINT("show stream process trace");
                            DEBUG_TRACEPOINT("NBproctrace = %d", streamCTRLimages[ID].md->NBproctrace);

                            snprintf(string,
                                     stringlen,
                                     " %2d ",
                                     streamCTRLimages[ID].md->NBproctrace);
                            TUI_printfw("%s", string);

                            for(int spti = 0;
                                    spti < streamCTRLimages[ID].md->NBproctrace;
                                    spti++)
                            {
                                DEBUG_TRACEPOINT("stream process trace step %d", spti);
                                ino_t inode = streamCTRLimages[ID]
                                              .streamproctrace[spti]
                                              .trigger_inode;
                                int sem = streamCTRLimages[ID]
                                          .streamproctrace[spti]
                                          .trigsemindex;
                                pid_t pid = streamCTRLimages[ID]
                                            .streamproctrace[spti]
                                            .procwrite_PID;


                                DEBUG_TRACEPOINT("stream process trace step %d: triggermode", spti);

                                switch(streamCTRLimages[ID]
                                        .streamproctrace[spti]
                                        .triggermode)
                                {
                                case PROCESSINFO_TRIGGERMODE_IMMEDIATE:
                                    snprintf(string, stringlen, "(%7lu IM ", inode);
                                    break;

                                case PROCESSINFO_TRIGGERMODE_CNT0:
                                    snprintf(string, stringlen, "(%7lu C0 ", inode);
                                    break;

                                case PROCESSINFO_TRIGGERMODE_CNT1:
                                    snprintf(string, stringlen, "(%7lu C1 ", inode);
                                    break;

                                case PROCESSINFO_TRIGGERMODE_CNT2:
                                    snprintf(string, stringlen, "(%7lu C2 ", inode);
                                    break;

                                case PROCESSINFO_TRIGGERMODE_SEMAPHORE:
                                    snprintf(string, stringlen, "(%7lu %02d ", inode, sem);
                                    break;

                                case PROCESSINFO_TRIGGERMODE_DELAY:
                                    snprintf(string, stringlen, "(%7lu DL ", inode);
                                    break;

                                default:
                                    snprintf(string, stringlen, "(%7lu ?? ", inode);
                                    break;
                                }
                                TUI_printfw("%s", string);

                                DEBUG_TRACEPOINT(" ");

                                streamCTRL_print_procpid(8,
                                                         pid,
                                                         upstreamproc,
                                                         NBupstreamproc,
                                                         print_pid_mode);
                                TUI_printfw(")> ");
                                DEBUG_TRACEPOINT(" ");
                            }

                            if(sTUIparam.DisplayDetailLevel == 1)
                            {
                                DEBUG_TRACEPOINT(" ");
                                TUI_newline();
                                streamCTRL_print_SPTRACE_details(streamCTRLimages,
                                                                 ID,
                                                                 upstreamproc,
                                                                 NBupstreamproc,
                                                                 PRINT_PID_DEFAULT);
                                DEBUG_TRACEPOINT(" ");
                            }
                        }


                        DEBUG_TRACEPOINT(" ");
                        if((sTUIparam.DisplayMode == DISPLAY_MODE_SUMMARY) &&
                                (DisplayFlag == 1))
                        {
                            if(sTUIparam.DisplayDetailLevel == 1)
                            {
                                TUI_newline();
                                TUI_newline();
                                TUI_printfw("name            %10s", streamCTRLimages[ID].name);
                                TUI_newline();
                                TUI_printfw("createcnt       %10ld", streamCTRLimages[ID].createcnt);
                                TUI_newline();
                                TUI_printfw("shmfd           %10d", streamCTRLimages[ID].shmfd);
                                TUI_newline();
                                TUI_printfw("memsize         %10lu", streamCTRLimages[ID].memsize);
                                TUI_newline();
                                TUI_printfw("md.version      %10s", streamCTRLimages[ID].md->version);
                                TUI_newline();
                                TUI_printfw("md.name         %10s", streamCTRLimages[ID].md->name);
                                TUI_newline();
                                TUI_printfw("md.naxis        %10d", (int) streamCTRLimages[ID].md->naxis);
                                TUI_newline();
                                for(int axis = 0; axis < streamCTRLimages[ID].md->naxis; axis++)
                                {
                                    TUI_printfw("   md.size[%d]   %10d", axis,
                                                (int) streamCTRLimages[ID].md->size[axis]);
                                    TUI_newline();
                                }
                                TUI_printfw("md.nelement         %10lu", streamCTRLimages[ID].md->nelement);
                                TUI_newline();
                                TUI_printfw("md.datatype         %10d",
                                            (int) streamCTRLimages[ID].md->datatype);
                                TUI_newline();
                                TUI_printfw("md.creationtime     %10ld.%09ld",
                                            streamCTRLimages[ID].md->creationtime.tv_sec,
                                            (long)streamCTRLimages[ID].md->creationtime.tv_nsec);
                                TUI_newline();
                                TUI_printfw("md.lastaccesstime   %10ld.%09ld",
                                            streamCTRLimages[ID].md->lastaccesstime.tv_sec,
                                            (long)streamCTRLimages[ID].md->lastaccesstime.tv_nsec);
                                TUI_newline();
                                TUI_printfw("md.atime            %10ld.%09ld",
                                            streamCTRLimages[ID].md->atime.tv_sec, (long)streamCTRLimages[ID].md->atime.tv_nsec);
                                TUI_newline();
                                TUI_printfw("md.writetime        %10ld.%09ld",
                                            streamCTRLimages[ID].md->writetime.tv_sec,
                                            (long)streamCTRLimages[ID].md->writetime.tv_nsec);
                                TUI_newline();
                                TUI_printfw("md.creatorPID       %10ld",
                                            (long) streamCTRLimages[ID].md->creatorPID);
                                TUI_newline();
                                TUI_printfw("md.ownerPID         %10ld",
                                            (long) streamCTRLimages[ID].md->ownerPID);
                                TUI_newline();
                                TUI_printfw("md.shared           %10d",
                                            (int) streamCTRLimages[ID].md->shared);
                                TUI_newline();
                                TUI_printfw("md.inode            %10lu",
                                            (unsigned long) streamCTRLimages[ID].md->inode);
                                TUI_newline();
                                TUI_newline();
                                TUI_printfw("md.sem              %10d", (int) streamCTRLimages[ID].md->sem);
                                TUI_newline();
                            }
                        }
                        DEBUG_TRACEPOINT(" ");
                    }

                    DEBUG_TRACEPOINT(" ");

                    if((sTUIparam.DisplayMode == DISPLAY_MODE_FUSER) &&
                            (DisplayFlag ==
                             1)) // list processes that are accessing streams
                    {
                        if(streaminfoproc.fuserUpdate == 2)
                        {
                            streaminfo[sindex].streamOpenPID_status =
                                0; // not scanned
                        }

                        DEBUG_TRACEPOINT(" ");

                        int pidIndex;

                        switch(streaminfo[sindex].streamOpenPID_status)
                        {

                        case 1:
                            streaminfo[sindex].streamOpenPID_cnt1 = 0;
                            for(pidIndex = 0;
                                    pidIndex < streaminfo[sindex].streamOpenPID_cnt;
                                    pidIndex++)
                            {
                                pid_t pid =
                                    streaminfo[sindex].streamOpenPID[pidIndex];
                                streamCTRL_print_procpid(8,
                                                         pid,
                                                         upstreamproc,
                                                         NBupstreamproc,
                                                         print_pid_mode);

                                if((getpgid(pid) >= 0) && (pid != getpid()))
                                {

                                    snprintf(string,
                                             stringlen,
                                             ":%-*.*s",
                                             PIDnameStringLen,
                                             PIDnameStringLen,
                                             PIDname_array[pid]);
                                    TUI_printfw("%s", string);

                                    streaminfo[sindex].streamOpenPID_cnt1++;
                                }
                            }
                            break;

                        case 2:
                            snprintf(string, stringlen, "FAILED");
                            TUI_printfw("%s", string);
                            break;

                        default:
                            snprintf(string, stringlen, "NOT SCANNED");
                            TUI_printfw("%s", string);
                            break;
                        }
                    }

                    DEBUG_TRACEPOINT(" ");

                    if(DisplayFlag == 1)
                    {
                        if(dindex == sTUIparam.dindexSelected)
                        {
                            screenprint_unsetreverse();
                        }

                        TUI_newline();
                    }
                }

                DEBUG_TRACEPOINT(" ");

                if(streaminfoproc.fuserUpdate == 1)
                {
                    //      refresh();
                    if(sc_sigINT == 1)  // stop scan
                    {
                        // complete loop without scan
                        streaminfoproc.fuserUpdate = 2;

                        sc_sigINT = 0; // reset
                    } // complete loop without scan
                }

                DEBUG_TRACEPOINT(" ");
            }
        }

        DEBUG_TRACEPOINT(" ");

        /* ---- Scroll indicator footer ---- */
        if(sTUIparam.DisplayMode != DISPLAY_MODE_HELP)
        {
            int above = doffsetindex;
            int below = sTUIparam.NBsindex - (doffsetindex + NBsinfodisp);
            if(below < 0)
            {
                below = 0;
            }

            if(above > 0 || below > 0)
            {
                screenprint_setdim();
                if(above > 0)
                {
                    screenprint_setcolor(3); /* yellow */
                    TUI_printfw(" \033[1m\xe2\x86\x91\033[22m %d above ",
                                above);
                    screenprint_unsetcolor(3);
                }
                else
                {
                    TUI_printfw(" -- top -- ");
                }

                TUI_printfw("|");

                if(below > 0)
                {
                    screenprint_setcolor(3); /* yellow */
                    TUI_printfw(" \033[1m\xe2\x86\x93\033[22m %d below ",
                                below);
                    screenprint_unsetcolor(3);
                }
                else
                {
                    TUI_printfw(" -- end -- ");
                }
                screenprint_unsetdim();
            } /* if above > 0 || below > 0 */
            /* No trailing TUI_newline(): TUI_cleartobottom() clears
             * the rest of the footer row without risking a scroll. */
        } /* scroll indicator footer */

        TUI_cleartobottom();
        sc_frame_flush();

        DEBUG_TRACEPOINT(" ");

        loopcnt++;
    }

    streaminfoproc.loop = 0;
    pthread_join(threadscan, NULL);

    for(int pidi = 0; pidi < PIDmax; pidi++)
    {
        if(PIDname_array[pidi] != NULL)
        {
            free(PIDname_array[pidi]);
        }
    }
    free(PIDname_array);

    for(imageID ID = 0; ID < streamNBID_MAX; ID++)
    {
        if(streamCTRLimages[ID].used == 1)
        {
            ImageStreamIO_closeIm(&streamCTRLimages[ID]);
        }
    }

    free(streamCTRLimages);
    free(streaminfo);
    free(upstreaminode);
    free(upstreamproc);

    fflush(stderr);
    dup2(backstderr, STDERR_FILENO);
    close(backstderr);

    remove(newstderrfname);

    DEBUG_TRACEPOINT(" ");

    return EXIT_SUCCESS;
}

