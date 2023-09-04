#include <stdint.h>
#include <sys/types.h>

#ifndef __STDC_LIB_EXT1__
typedef int errno_t;
#endif

#include "CLIcore.h"
#include "TUItools.h"
#include "streamCTRL_TUI.h"
#include "streamCTRL_print_procpid.h"




errno_t streamCTRL_print_SPTRACE_details(
    IMAGE   *streamCTRLimages,
    imageID  ID,
    pid_t   *upstreamproc,
    int      NBupstreamproc,
    uint32_t print_pid_mode
)
{
    int Disp_inode_NBchar = 8;

    int Disp_sname_NBchar = 16;

    int Disp_cnt0_NBchar     = 12;
    int Disp_PID_NBchar      = 8;
    int Disp_type_NBchar     = 8;
    int Disp_trigstat_NBchar = 12;

    // suppress unused parameter warning
    (void) print_pid_mode;

    DEBUG_TRACEPOINT(" ");

    TUI_newline();
    TUI_printfw("   %*s %*s %*s",
                Disp_inode_NBchar,
                "inode",
                Disp_sname_NBchar,
                "stream",
                Disp_cnt0_NBchar,
                "cnt0",
                Disp_PID_NBchar,
                "PID",
                Disp_type_NBchar,
                "type");
    TUI_newline();

    DEBUG_TRACEPOINT(" ");

    for(int spti = 0; spti < streamCTRLimages[ID].md->NBproctrace; spti++)
    {
        DEBUG_TRACEPOINT("spti %d", spti);
        ino_t inode = streamCTRLimages[ID].streamproctrace[spti].trigger_inode;
        int   sem   = streamCTRLimages[ID].streamproctrace[spti].trigsemindex;
        pid_t pid   = streamCTRLimages[ID].streamproctrace[spti].procwrite_PID;

        uint64_t cnt0 = streamCTRLimages[ID].streamproctrace[spti].cnt0;

        TUI_printfw("%02d", spti);

        TUI_printfw(" %*lu", Disp_inode_NBchar, inode);

        DEBUG_TRACEPOINT("spti %d", spti);

        // look for ID corresponding to inode
        int IDscan  = 0;
        int IDfound = -1;
        while((IDfound == -1) && (IDscan < streamNBID_MAX))
        {
            if( (streamCTRLimages[IDscan].used == 1) && (streamCTRLimages[IDscan].md != NULL) )
            {
                if(streamCTRLimages[IDscan].md->inode == inode)
                {
                    IDfound = IDscan;
                }
            }
            IDscan++;
        }

        DEBUG_TRACEPOINT("spti %d", spti);


        if(IDfound == -1)
        {
            TUI_printfw(" %*s", Disp_sname_NBchar, "???");
        }
        else
        {
            TUI_printfw(" %*s",
                        Disp_sname_NBchar,
                        streamCTRLimages[IDfound].name);
        }

        TUI_printfw(" %*llu", Disp_cnt0_NBchar, cnt0);
        TUI_printfw(" ");

        Disp_PID_NBchar = streamCTRL_print_procpid(8,
                          pid,
                          upstreamproc,
                          NBupstreamproc,
                          PRINT_PID_FORCE_NOUPSTREAM);

        TUI_printfw(" ");

        switch(streamCTRLimages[ID].streamproctrace[spti].triggermode)
        {
        case PROCESSINFO_TRIGGERMODE_IMMEDIATE:
            TUI_printfw("%d%*s",
                        streamCTRLimages[ID].streamproctrace[spti].triggermode,
                        Disp_type_NBchar - 1,
                        "IMME");
            break;

        case PROCESSINFO_TRIGGERMODE_CNT0:
            TUI_printfw("%d%*s",
                        streamCTRLimages[ID].streamproctrace[spti].triggermode,
                        Disp_type_NBchar - 1,
                        "CNT0");
            break;

        case PROCESSINFO_TRIGGERMODE_CNT1:
            TUI_printfw("%d%*s",
                        streamCTRLimages[ID].streamproctrace[spti].triggermode,
                        Disp_type_NBchar - 1,
                        "CNT1");
            break;

        case PROCESSINFO_TRIGGERMODE_SEMAPHORE:
            TUI_printfw("%d%*s",
                        streamCTRLimages[ID].streamproctrace[spti].triggermode,
                        Disp_type_NBchar - 4,
                        "SM");
            TUI_printfw(" %2d", sem);
            break;

        case PROCESSINFO_TRIGGERMODE_DELAY:
            TUI_printfw("%d%*s",
                        streamCTRLimages[ID].streamproctrace[spti].triggermode,
                        Disp_type_NBchar - 1,
                        "DELA");
            break;

        default:
            TUI_printfw("%d%*s",
                        streamCTRLimages[ID].streamproctrace[spti].triggermode,
                        Disp_type_NBchar - 1,
                        "UNKN");
            break;
        }
        TUI_printfw(" ");

        int print_timing = 0;
        switch(streamCTRLimages[ID].streamproctrace[spti].triggerstatus)
        {
        case PROCESSINFO_TRIGGERSTATUS_WAITING:
            TUI_printfw("%*s", Disp_trigstat_NBchar, "WAITING");
            break;

        case PROCESSINFO_TRIGGERSTATUS_RECEIVED:
            screenprint_setcolor(2);
            //attron(COLOR_PAIR(2));
            TUI_printfw("%*s", Disp_trigstat_NBchar, "RECEIVED");
            screenprint_unsetcolor(2);
            //attroff(COLOR_PAIR(2));
            print_timing = 1;
            break;

        case PROCESSINFO_TRIGGERSTATUS_TIMEDOUT:
            //attron(COLOR_PAIR(3));
            screenprint_setcolor(3);
            TUI_printfw("%*s", Disp_trigstat_NBchar, "TIMEOUT");
            //attroff(COLOR_PAIR(3));
            screenprint_unsetcolor(3);
            print_timing = 1;
            break;

        default:
            TUI_printfw("%*s", Disp_trigstat_NBchar, "unknown");
            break;
        }

        // trigger time
        if(print_timing == 1)
        {
            TUI_printfw(
                " at %ld.%09ld s",
                streamCTRLimages[ID].streamproctrace[spti].ts_procstart.tv_sec,
                streamCTRLimages[ID]
                .streamproctrace[spti]
                .ts_procstart.tv_nsec);

            struct timespec tnow;
            clock_gettime(CLOCK_MILK, &tnow);
            struct timespec tdiff;

            tdiff = timespec_diff(
                        streamCTRLimages[ID].streamproctrace[spti].ts_procstart,
                        tnow);
            double tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

            TUI_printfw("  %12.3f us ago", tdiffv * 1.0e6);
        }

        TUI_newline();
    }

    DEBUG_TRACEPOINT(" ");

    return RETURN_SUCCESS;
}


