
/**
 * @file streamCTRL.c
 * @brief Data streams control panel
 *
 * Manages data streams
 *
 *
 */

#define _GNU_SOURCE



#ifndef __STDC_LIB_EXT1__
typedef int errno_t;
#endif

#include <sys/file.h>
#include <sys/stat.h>
#include <ncurses.h>
#include <pthread.h>


#include "CommandLineInterface/timeutils.h"

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"


// default location of file mapped semaphores, can be over-ridden by env variable MILK_SHM_DIR
#define SHAREDSHMDIR  data.shmdir

#include "streamCTRL_TUI.h"

#include "TUItools.h"

#include "streamCTRL_find_streams.h"
#include "streamCTRL_print_inode.h"
#include "streamCTRL_print_procpid.h"
#include "streamCTRL_print_trace.h"
#include "streamCTRL_scan.h"
#include "streamCTRL_utilfuncs.h"


#define ctrl(x) ((x) &0x1f)


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
    long ssindex[streamNBID_MAX]; // sorted index array
} sTUIparam;








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
    case 'x': // Exit control screen
        sTUIparam.loopOK = 0;
        break;

    case KEY_UP:
        sTUIparam.dindexSelected--;
        if(sTUIparam.dindexSelected < 0)
        {
            sTUIparam.dindexSelected = 0;
        }
        break;

    case KEY_DOWN:
        sTUIparam.dindexSelected++;
        if(sTUIparam.dindexSelected > sTUIparam.NBsindex - 1)
        {
            sTUIparam.dindexSelected = sTUIparam.NBsindex - 1;
        }
        break;

    case KEY_PPAGE:
        sTUIparam.dindexSelected -= 10;
        if(sTUIparam.dindexSelected < 0)
        {
            sTUIparam.dindexSelected = 0;
        }
        break;

    case KEY_LEFT:
        sTUIparam.DisplayDetailLevel = 0;
        break;

    case KEY_RIGHT:
        sTUIparam.DisplayDetailLevel = 1;
        break;

    case KEY_NPAGE:
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

    case KEY_F(2): // semvals
        sTUIparam.DisplayMode = DISPLAY_MODE_SUMMARY;
        break;

    case KEY_F(3): // write PIDs
        sTUIparam.DisplayMode = DISPLAY_MODE_WRITE;
        break;

    case KEY_F(4): // read PIDs
        sTUIparam.DisplayMode = DISPLAY_MODE_READ;
        break;

    case KEY_F(5): // read PIDs
        sTUIparam.DisplayMode = DISPLAY_MODE_SPTRACE;
        break;

    case KEY_F(6): // open files
        if((sTUIparam.DisplayMode == DISPLAY_MODE_FUSER) ||
                (streamCTRLdata->streaminfoproc->fuserUpdate0 == 1))
        {
            streamCTRLdata->streaminfoproc->fuserUpdate = 1;
            time(&rawtime);
            sTUIparam.uttime_lastScan           = gmtime(&rawtime);
            sTUIparam.fuserScan                 = 1;
            streamCTRLdata->streaminfoproc->sindexscan = 0;
        }

        sTUIparam.DisplayMode = DISPLAY_MODE_FUSER;
        //erase();
        //TUI_printfw("SCANNING PROCESSES AND FILESYSTEM: PLEASE WAIT ...\n");
        //refresh();
        break;

    // ============ ACTIONS

    case ctrl('e'): // erase stream
        if(sTUIparam.dindexSelected >= 0)
        {
            sindex = sTUIparam.ssindex[sTUIparam.dindexSelected];
            DEBUG_TRACEPOINT("removing stream sindex = %ld", sindex);
            DEBUG_TRACEPOINT("removing stream ID = %ld", streaminfo[sindex].ID);

            ImageStreamIO_destroyIm(&streamCTRLdata->images[streamCTRLdata->sinfo[sindex].ID]);

            DEBUG_TRACEPOINT("%d", sTUIparam.dindexSelected);
        }
        break;

    // ============ SCANNING

    case '{': // slower scan update
        streamCTRLdata->streaminfoproc->twaitus = (int)(1.2 * streamCTRLdata->streaminfoproc->twaitus);
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

    case '1': // sorting by stream name
        sTUIparam.SORTING = 1;
        break;

    case '2': // sorting by update freq (default)
        sTUIparam.SORTING     = 2;
        sTUIparam.SORT_TOGGLE = 1;
        break;

    case '3': // sort by number of processes accessing
        sTUIparam.SORTING     = 3;
        sTUIparam.SORT_TOGGLE = 1;
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
        TUI_exit();
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
    }
    return EXIT_SUCCESS;
}










/**
 * @brief Control screen for stream structures
 *
 * @return errno_t
 */
errno_t streamCTRL_CTRLscreen()
{

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
    sTUIparam.frequ = 32.0; // Hz



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
    int Dispcnt0_NBchar = 10;
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
        streaminfo[sindex].updatevalue          = 0.0;
        streaminfo[sindex].updatevalue_frozen   = 0.0;
        streaminfo[sindex].cnt0                 = 0;
        streaminfo[sindex].streamOpenPID_status = 0;
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
    set_signal_catch();

    // default: use ncurses
    TUI_set_screenprintmode(SCREENPRINT_NCURSES);

    if(getenv("MILK_TUIPRINT_STDIO"))
    {
        // use stdio instead of ncurses
        TUI_set_screenprintmode(SCREENPRINT_STDIO);
    }

    if(getenv("MILK_TUIPRINT_NONE"))
    {
        TUI_set_screenprintmode(SCREENPRINT_NONE);
    }

    DEBUG_TRACEPOINT("Initialize terminal");
    TUI_init_terminal(&wrow, &wcol);


    long long loopcnt  = 0;




    streaminfoproc.filter       = 0;
    streaminfoproc.NBstream     = 0;
    streaminfoproc.twaitus      = 50000; // 20 Hz
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

    clear();
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
                       CLIPID);

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

    while(sTUIparam.loopOK == 1)
    {

        int NBsinfodisp = wrow - 7;

        if(streaminfoproc.loopcnt == 1)
        {
            sTUIparam.SORTING     = 2;
            sTUIparam.SORT_TOGGLE = 1;
        }

        //if(fuserUpdate != 1) // don't wait if ongoing fuser scan

        usleep((long)(1000000.0 / sTUIparam.frequ));
        //int ch = getch();
        int ch = get_singlechar_nonblock();

        sTUIparam.NBsindex = streaminfoproc.NBstream;

        TUI_clearscreen(&wrow, &wcol);

        TUI_ncurses_erase();

        DEBUG_TRACEPOINT("Process input character");
        //int selectedOK = 0; // goes to 1 if at least one process is selected

        streamCTRL_keyinput_process(ch, &streamCTRLdata);



        DEBUG_TRACEPOINT("Input character processed");

        if(sTUIparam.dindexSelected < 0)
        {
            sTUIparam.dindexSelected = 0;
        }
        if(sTUIparam.dindexSelected > sTUIparam.NBsindex - 1)
        {
            sTUIparam.dindexSelected = sTUIparam.NBsindex - 1;
        }

        DEBUG_TRACEPOINT("Erase screen");
        erase();

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
        TUI_print_header(monstring, '-');
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
            print_help_entry("1", "Sort by stream name (alphabetical)");
            print_help_entry("2", "Sort by recently updated");
            print_help_entry("3", "Sort by process access");
            print_help_entry("s", "Show 3 semaphores / all semaphores");
            print_help_entry("F", "Set match string pattern");
            print_help_entry("f", "Toggle apply match string to stream");
        }
        else
        {
            DEBUG_TRACEPOINT(" ");
            if(sTUIparam.DisplayMode == DISPLAY_MODE_HELP)  // Inaccessible.
            {
                screenprint_setreverse();
                TUI_printfw("[h] Help");
                screenprint_unsetreverse();
            }
            else
            {
                TUI_printfw("[h] Help");
            }
            TUI_printfw("   ");

            if(sTUIparam.DisplayMode == DISPLAY_MODE_SUMMARY)
            {
                screenprint_setreverse();
                TUI_printfw("[F2] summary");
                screenprint_unsetreverse();
            }
            else
            {
                TUI_printfw("[F2] summary");
            }
            TUI_printfw("   ");

            if(sTUIparam.DisplayMode == DISPLAY_MODE_WRITE)
            {
                screenprint_setreverse();
                TUI_printfw("[F3] write PIDs");
                screenprint_unsetreverse();
            }
            else
            {
                TUI_printfw("[F3] write PIDs");
            }
            TUI_printfw("   ");

            if(sTUIparam.DisplayMode == DISPLAY_MODE_READ)
            {
                screenprint_setreverse();
                TUI_printfw("[F4] read PIDs");
                screenprint_unsetreverse();
            }
            else
            {
                TUI_printfw("[F4] read PIDs");
            }
            TUI_printfw("   ");

            if(sTUIparam.DisplayMode == DISPLAY_MODE_SPTRACE)
            {
                screenprint_setreverse();
                TUI_printfw("[F5] process traces");
                screenprint_unsetreverse();
            }
            else
            {
                TUI_printfw("[F5] process traces");
            }
            TUI_printfw("   ");

            if(sTUIparam.DisplayMode == DISPLAY_MODE_FUSER)
            {
                screenprint_setreverse();
                TUI_printfw("[F6] access");
                screenprint_unsetreverse();
            }
            else
            {
                TUI_printfw("[F6] access");
            }
            TUI_printfw("   ");
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
                //attron(COLOR_PAIR(9));
                screenprint_setcolor(9);
                TUI_printfw("fuser scan ongoing  %4d  / %4d   ",
                            streaminfoproc.sindexscan,
                            sTUIparam.NBsindex);
                //attroff(COLOR_PAIR(9));
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
                    "%4d streams    Currently displaying %4d-%4d   "
                    "Selected %d  ID = %d  inode = %d",
                    sTUIparam.NBsindex,
                    doffsetindex,
                    lastindex,
                    sTUIparam.dindexSelected,
                    ssIDselected,
                    (int) inodeselected);
            }

            if(streaminfoproc.filter == 1)
            {
                //attron(COLOR_PAIR(9));
                screenprint_setcolor(9);
                TUI_printfw("  Filter = \"%s\"", streaminfoproc.namefilter);
                //attroff(COLOR_PAIR(9));
                screenprint_unsetcolor(9);
            }

            TUI_newline();

            attron(A_BOLD);

            TUI_printfw("%*s  %-*s  %-*s  %*s   %*s %*s %*s %8s",
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
                        "   frequ ",
                        "#sem");

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
            //attroff(A_BOLD);

            DEBUG_TRACEPOINT(" ");

            // SORT

            // default : no sorting
            for(int dindex = 0; dindex < sTUIparam.NBsindex; dindex++)
            {
                sTUIparam.ssindex[dindex] = dindex;
            }

            DEBUG_TRACEPOINT(" ");

            if(sTUIparam.SORTING == 1)  // alphabetical sorting
            {
                long *larray;
                larray = (long *) malloc(sizeof(long) * sTUIparam.NBsindex);
                for(long sindex = 0; sindex < sTUIparam.NBsindex; sindex++)
                {
                    larray[sindex] = sindex;
                }

                for(int sindex0 = 0; sindex0 < sTUIparam.NBsindex - 1; sindex0++)
                {
                    for(int sindex1 = sindex0 + 1; sindex1 < sTUIparam.NBsindex; sindex1++)
                    {
                        if(strcmp(streaminfo[larray[sindex0]].sname,
                                  streaminfo[larray[sindex1]].sname) > 0)
                        {
                            int tmpindex    = larray[sindex0];
                            larray[sindex0] = larray[sindex1];
                            larray[sindex1] = tmpindex;
                        }
                    }
                }

                for(long dindex = 0; dindex < sTUIparam.NBsindex; dindex++)
                {
                    sTUIparam.ssindex[dindex] = larray[dindex];
                }
                free(larray);
            }

            DEBUG_TRACEPOINT(" ");

            if((sTUIparam.SORTING == 2) ||
                    (sTUIparam.SORTING == 3)) // recent update and process access
            {
                long   *larray;
                double *varray;
                larray = (long *) malloc(sizeof(long) * sTUIparam.NBsindex);
                varray = (double *) malloc(sizeof(double) * sTUIparam.NBsindex);

                if(sTUIparam.SORT_TOGGLE == 1)
                {
                    for(long sindex = 0; sindex < sTUIparam.NBsindex; sindex++)
                    {
                        streaminfo[sindex].updatevalue_frozen =
                            streaminfo[sindex].updatevalue;
                    }

                    if(sTUIparam.SORTING == 3)
                    {
                        for(long sindex = 0; sindex < sTUIparam.NBsindex; sindex++)
                        {
                            streaminfo[sindex].updatevalue_frozen +=
                                10000.0 * streaminfo[sindex].streamOpenPID_cnt1;
                        }
                    }

                    sTUIparam.SORT_TOGGLE = 0;
                }

                for(long sindex = 0; sindex < sTUIparam.NBsindex; sindex++)
                {
                    larray[sindex] = sindex;
                    varray[sindex] = streaminfo[sindex].updatevalue_frozen;
                }

                if(sTUIparam.NBsindex > 1)
                {
                    quick_sort2l(varray, larray, sTUIparam.NBsindex);
                }

                for(long dindex = 0; dindex < sTUIparam.NBsindex; dindex++)
                {
                    sTUIparam.ssindex[sTUIparam.NBsindex - dindex - 1] = larray[dindex];
                }

                free(larray);
                free(varray);
            }

            DEBUG_TRACEPOINT(" ");

            // compute doffsetindex

            while(sTUIparam.dindexSelected - doffsetindex >
                    NBsinfodisp - 5) // scroll down
            {
                doffsetindex++;
            }

            while(sTUIparam.dindexSelected - doffsetindex <
                    NBsinfodisp - 10) // scroll up
            {
                doffsetindex--;
            }

            if(doffsetindex < 0)
            {
                doffsetindex = 0;
            }

            // DISPLAY

            int DisplayFlag = 0;

            int print_pid_mode = PRINT_PID_DEFAULT;
            for(int dindex = 0; dindex < sTUIparam.NBsindex; dindex++)
            {
                imageID ID;
                int sindex = sTUIparam.ssindex[dindex];
                ID     = streaminfo[sindex].ID;

                while((streamCTRLimages[streaminfo[sindex].ID].used == 0) &&
                        (dindex < sTUIparam.NBsindex))
                {
                    // skip this entry, as it is no longer in use
                    dindex++;
                    sindex = sTUIparam.ssindex[dindex];
                    ID     = streaminfo[sindex].ID;
                }

                int downstreammin =
                    NO_DOWNSTREAM_INDEX; // minumum downstream index
                // looks for inodeselected in the list of upstream inodes
                // picks the smallest corresponding index
                // for example, if equal to 3, the current inode is a 3-rd gen children of selected inode
                // default initial value 100 is a placeholder indicating it is not a child

                DEBUG_TRACEPOINT(" ");

                if((dindex > doffsetindex - 1) &&
                        (dindex < NBsinfodisp - 1 + doffsetindex))
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

                if(dindex == sTUIparam.dindexSelected)
                {
                    DEBUG_TRACEPOINT(
                        "dindex %ld %d",
                        dindex,
                        streamCTRLimages[streaminfo[sindex].ID].used);

                    // currently selected inode
                    inodeselected =
                        streamCTRLimages[streaminfo[sindex].ID].md[0].inode;

                    DEBUG_TRACEPOINT(
                        "inode %lu %s",
                        inodeselected,
                        streamCTRLimages[streaminfo[sindex].ID].md[0].name);

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
                    DEBUG_TRACEPOINT("ID = %ld", ID);
                    DEBUG_TRACEPOINT("used = %d", streamCTRLimages[ID].used);
                    print_pid_mode = PRINT_PID_DEFAULT;
                    if(streamCTRLimages[ID].used == 1)
                    {
                        for(int spti = 0;
                                spti < streamCTRLimages[ID].md[0].NBproctrace;
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
                    //attron(A_REVERSE);
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

                        //attron(COLOR_PAIR(5));
                        screenprint_setcolor(5);
                        TUI_printfw("%-*.*s",
                                    DispName_NBchar,
                                    DispName_NBchar,
                                    namestring);
                        //attroff(COLOR_PAIR(5));
                        screenprint_unsetcolor(5);
                    }
                    else
                    {
                        TUI_printfw("%-*.*s",
                                    DispName_NBchar,
                                    DispName_NBchar,
                                    streaminfo[sindex].sname);
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
                    TUI_printfw(string);

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
                            strcpy(str, str1);
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

                        strcpy(str, str1);
                    }

                    DEBUG_TRACEPOINT(" ");

                    snprintf(string,
                             stringlen,
                             "%-*.*s ",
                             DispSize_NBchar,
                             DispSize_NBchar,
                             str);
                    TUI_printfw(string);

                    if(streamCTRLimages[streaminfo[sindex].ID].md == NULL)
                    {
                        snprintf(string, stringlen, "???");
                    }
                    else
                    {

                        snprintf(string,
                                 stringlen,
                                 " %*ld ",
                                 Dispcnt0_NBchar,
                                 streamCTRLimages[ID].md[0].cnt0);
                    }
                    if(streaminfo[sindex].deltacnt0 == 0)
                    {
                        TUI_printfw(string);
                    }
                    else
                    {
                        //attron(COLOR_PAIR(2));
                        screenprint_setcolor(2);
                        TUI_printfw(string);
                        //attroff(COLOR_PAIR(2));
                        screenprint_unsetcolor(2);
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
                    }
                    else
                    {
                        snprintf(string,
                                 stringlen,
                                 " %*.2f Hz",
                                 Dispfreq_NBchar,
                                 streaminfo[sindex].updatevalue);
                    }
                    TUI_printfw(string);
                }

                DEBUG_TRACEPOINT(" ");

                if(streamCTRLimages[streaminfo[sindex].ID].md != NULL)
                {
                    if((sTUIparam.DisplayMode == DISPLAY_MODE_SUMMARY) &&
                            (DisplayFlag == 1)) // sem vals
                    {

                        snprintf(string,
                                 stringlen,
                                 " %3d sems ",
                                 streamCTRLimages[ID].md[0].sem);
                        TUI_printfw(string);

                        int s;
                        int max_s = sTUIparam.DISPLAY_ALL_SEMS
                                    ? streamCTRLimages[ID].md[0].sem
                                    : 3;
                        for(s = 0; s < max_s; s++)
                        {
                            int semval;
                            sem_getvalue(streamCTRLimages[ID].semptr[s],
                                         &semval);
                            snprintf(string, stringlen, " %7d", semval);
                            TUI_printfw(string);
                        }
                    }
                }

                DEBUG_TRACEPOINT(" ");
                if(streamCTRLimages[streaminfo[sindex].ID].md != NULL)
                {
                    if((sTUIparam.DisplayMode == DISPLAY_MODE_WRITE) &&
                            (DisplayFlag == 1)) // sem write PIDs
                    {
                        snprintf(string,
                                 stringlen,
                                 " %3d sems ",
                                 streamCTRLimages[ID].md[0].sem);
                        TUI_printfw(string);

                        {
                            int s;
                            int max_s = sTUIparam.DISPLAY_ALL_SEMS
                                        ? streamCTRLimages[ID].md[0].sem
                                        : 3;
                            for(s = 0; s < max_s; s++)
                            {
                                pid_t pid = streamCTRLimages[ID].semWritePID[s];
                                streamCTRL_print_procpid(8,
                                                         pid,
                                                         upstreamproc,
                                                         NBupstreamproc,
                                                         print_pid_mode);
                                TUI_printfw(" ");
                            }
                        }

                        if(sTUIparam.DisplayDetailLevel == 1)
                        {
#ifdef IMAGESTRUCT_WRITEHISTORY
                            TUI_newline();
                            TUI_printfw("WRITE timings :\n");
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
                                    TUI_printfw("%4d  cnt0 %8d  PID %6d  ts %9ld.%09ld   %.9f s ago  delta = %9.3f us\n",
                                                wioffset,
                                                streamCTRLimages[ID].writehist[windex].cnt0,
                                                streamCTRLimages[ID].writehist[windex].wpid,
                                                streamCTRLimages[ID].writehist[windex].writetime.tv_sec,
                                                streamCTRLimages[ID].writehist[windex].writetime.tv_nsec,
                                                tdouble0 - tdouble,
                                                1.0e6 * (deltat));
                                }
                                tdoubleprev = tdouble;
                            }

                            quick_sort_double(dtarray, IMAGESTRUCT_FRAMEWRITEMDSIZE - 2);

                            TUI_newline();

                            TUI_printfw("delta time (nbsample = %d):\n", IMAGESTRUCT_FRAMEWRITEMDSIZE);

                            double tave = 1.0e6 * deltatsum / (IMAGESTRUCT_FRAMEWRITEMDSIZE - 2);
                            TUI_printfw("AVERAGE =        %9.3f us\n", tave);

                            double trms = deltatsum2 - deltatsum * deltatsum / (IMAGESTRUCT_FRAMEWRITEMDSIZE
                                          - 2);
                            trms = 1.0e6 * sqrt(trms / (IMAGESTRUCT_FRAMEWRITEMDSIZE - 2));
                            TUI_printfw("RMS     =        %9.3f us  ( %8.3f %% )\n", trms,
                                        100.0 * trms / tave);

                            double p0us = 1.0e6 * dtarray[0];
                            TUI_printfw("  min          : %9.3f us    %9.3f us\n", p0us, p0us - tave);

                            double p10us = 1.0e6 * dtarray[(int)(0.1 * (IMAGESTRUCT_FRAMEWRITEMDSIZE - 2))];
                            TUI_printfw("  p10          : %9.3f us    %9.3f us\n", p10us, p10us - tave);

                            double p50us = 1.0e6 * dtarray[(IMAGESTRUCT_FRAMEWRITEMDSIZE - 2) / 2];
                            TUI_printfw("  p50 (median) : %9.3f us    %9.3f us\n", p50us, p50us - tave);

                            double p90us = 1.0e6 * dtarray[(int)(0.9 * (IMAGESTRUCT_FRAMEWRITEMDSIZE - 2))];
                            TUI_printfw("  p90          : %9.3f us    %9.3f us\n", p90us, p90us - tave);

                            double p100us = 1.0e6 * dtarray[IMAGESTRUCT_FRAMEWRITEMDSIZE - 3];
                            TUI_printfw("  max          : %9.3f us    %9.3f us\n", p100us, p100us - tave);


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
                        snprintf(string,
                                 stringlen,
                                 " %3d sems ",
                                 streamCTRLimages[ID].md[0].sem);
                        TUI_printfw(string);

                        int s;
                        int max_s = sTUIparam.DISPLAY_ALL_SEMS
                                    ? streamCTRLimages[ID].md[0].sem
                                    : 3;
                        for(s = 0; s < max_s; s++)
                        {
                            pid_t pid = streamCTRLimages[ID].semReadPID[s];
                            streamCTRL_print_procpid(8,
                                                     pid,
                                                     upstreamproc,
                                                     NBupstreamproc,
                                                     print_pid_mode);
                            TUI_printfw(" ");
                        }
                    }
                }

                if(streamCTRLimages[streaminfo[sindex].ID].md != NULL)
                {
                    if((sTUIparam.DisplayMode == DISPLAY_MODE_SPTRACE) &&
                            (DisplayFlag == 1))
                    {
                        snprintf(string,
                                 stringlen,
                                 " %2d ",
                                 streamCTRLimages[ID].md[0].NBproctrace);
                        TUI_printfw(string);

                        for(int spti = 0;
                                spti < streamCTRLimages[ID].md[0].NBproctrace;
                                spti++)
                        {
                            ino_t inode = streamCTRLimages[ID]
                                          .streamproctrace[spti]
                                          .trigger_inode;
                            int sem = streamCTRLimages[ID]
                                      .streamproctrace[spti]
                                      .trigsemindex;
                            pid_t pid = streamCTRLimages[ID]
                                        .streamproctrace[spti]
                                        .procwrite_PID;

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
                            TUI_printfw(string);
                            streamCTRL_print_procpid(8,
                                                     pid,
                                                     upstreamproc,
                                                     NBupstreamproc,
                                                     print_pid_mode);
                            TUI_printfw(")> ");
                        }

                        if(sTUIparam.DisplayDetailLevel == 1)
                        {
                            TUI_newline();
                            streamCTRL_print_SPTRACE_details(streamCTRLimages,
                                                             ID,
                                                             upstreamproc,
                                                             NBupstreamproc,
                                                             PRINT_PID_DEFAULT);
                        }
                    }
                    if((sTUIparam.DisplayMode == DISPLAY_MODE_SUMMARY) &&
                            (DisplayFlag == 1))
                    {
                        if(sTUIparam.DisplayDetailLevel == 1)
                        {
                            TUI_newline();
                            TUI_newline();
                            TUI_printfw("name            %10s\n", streamCTRLimages[ID].name);
                            TUI_printfw("createcnt       %10ld\n", streamCTRLimages[ID].createcnt);
                            TUI_printfw("shmfd           %10d\n", streamCTRLimages[ID].shmfd);
                            TUI_printfw("memsize         %10lu\n", streamCTRLimages[ID].memsize);
                            TUI_printfw("md.version      %10s\n", streamCTRLimages[ID].md->version);
                            TUI_printfw("md.name         %10s\n", streamCTRLimages[ID].md->name);
                            TUI_printfw("md.naxis        %10d\n", (int) streamCTRLimages[ID].md->naxis);
                            for(int axis = 0; axis < streamCTRLimages[ID].md->naxis; axis++)
                            {
                                TUI_printfw("   md.size[%d]   %10d\n", axis,
                                            (int) streamCTRLimages[ID].md->size[axis]);
                            }
                            TUI_printfw("md.nelement         %10lu\n", streamCTRLimages[ID].md->nelement);
                            TUI_printfw("md.datatype         %10d\n",
                                        (int) streamCTRLimages[ID].md->datatype);
                            TUI_printfw("md.creationtime     %10ld.%09d\n",
                                        streamCTRLimages[ID].md->creationtime.tv_sec,
                                        streamCTRLimages[ID].md->creationtime.tv_nsec);
                            TUI_printfw("md.lastaccesstime   %10ld.%09d\n",
                                        streamCTRLimages[ID].md->lastaccesstime.tv_sec,
                                        streamCTRLimages[ID].md->lastaccesstime.tv_nsec);
                            TUI_printfw("md.atime            %10ld.%09d\n",
                                        streamCTRLimages[ID].md->atime.tv_sec, streamCTRLimages[ID].md->atime.tv_nsec);
                            TUI_printfw("md.writetime        %10ld.%09d\n",
                                        streamCTRLimages[ID].md->writetime.tv_sec,
                                        streamCTRLimages[ID].md->writetime.tv_nsec);
                            TUI_printfw("md.creatorPID       %10ld\n",
                                        (long) streamCTRLimages[ID].md->creatorPID);
                            TUI_printfw("md.ownerPID         %10ld\n",
                                        (long) streamCTRLimages[ID].md->ownerPID);
                            TUI_printfw("md.shared           %10d\n",
                                        (int) streamCTRLimages[ID].md->shared);
                            TUI_printfw("md.inode            %10lu\n",
                                        (int) streamCTRLimages[ID].md->inode);
                            TUI_newline();
                            TUI_printfw("md.sem              %10d\n", (int) streamCTRLimages[ID].md->sem);
                        }
                    }
                }

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
                                TUI_printfw(string);

                                streaminfo[sindex].streamOpenPID_cnt1++;
                            }
                        }
                        break;

                    case 2:
                        snprintf(string, stringlen, "FAILED");
                        TUI_printfw(string);
                        break;

                    default:
                        snprintf(string, stringlen, "NOT SCANNED");
                        TUI_printfw(string);
                        break;
                    }
                }

                DEBUG_TRACEPOINT(" ");

                if(DisplayFlag == 1)
                {
                    if(dindex == sTUIparam.dindexSelected)
                    {
                        screenprint_unsetreverse();
                        //attroff(A_REVERSE);
                    }

                    /*attron(COLOR_PAIR(9));
                    TUI_printfw("+");
                    attroff(COLOR_PAIR(9));*/
                    TUI_newline();
                }

                DEBUG_TRACEPOINT(" ");

                if(streaminfoproc.fuserUpdate == 1)
                {
                    //      refresh();
                    if(data.signal_INT == 1)  // stop scan
                    {
                        streaminfoproc.fuserUpdate =
                            2;               // complete loop without scan
                        data.signal_INT = 0; // reset
                    }
                }

                DEBUG_TRACEPOINT(" ");
            }
        }

        DEBUG_TRACEPOINT(" ");

        refresh();

        DEBUG_TRACEPOINT(" ");

        loopcnt++;
        if((data.signal_TERM == 1) || (data.signal_INT == 1) ||
                (data.signal_ABRT == 1) || (data.signal_BUS == 1) ||
                (data.signal_SEGV == 1) || (data.signal_HUP == 1) ||
                (data.signal_PIPE == 1))
        {
            sTUIparam.loopOK = 0;
        }

        DEBUG_TRACEPOINT(" ");
    }

    endwin();

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
