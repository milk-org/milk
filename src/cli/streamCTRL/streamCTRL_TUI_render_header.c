#include "streamCTRL_TUI_render_internal.h"

/**
 * @brief Render the streamCTRL column headers.
 */
void streamCTRL__render_header_streams(
    streamCTRLarg_struct        *streamCTRLdata,
    struct streamCTRL_TUI_state *state,
    int                         *NBsinfodisp_out,
    int                         *lastindex_out,
    double                      *frame_t_sec_out,
    int                         *frame_color_level_out)
{
    int NBsinfodisp = 10;
    double frame_t_sec = 0.0;
    int frame_color_level = 0;
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
        100.0 * (streaminfoproc.dtscan - 1.0e-6 * streaminfoproc.twaitus) / streaminfoproc.dtscan);

    if(streaminfoproc.fuserUpdate == 1)
    {
        screenprint_setcolor(9);
        TUI_printfw("fuser scan ongoing  %4d  / %4d   ",
                    streaminfoproc.sindexscan, sTUIparam.NBsindex);
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
                sTUIparam.uttime_lastScan->tm_min, sTUIparam.uttime_lastScan->tm_sec);
            TUI_newline();
        }
        else
        {
            TUI_printfw(
                "Last scan on  XX:XX:XX  - Press F6 "
                "again to scan             C-c to stop " "scan");
            TUI_newline();
        }
    }
    else
    {
        /* Sort status indicator */
        static const char *sort_col_names[] =
        {
            "", "NAME", "TYPE", "SIZE",
            "CNT0", "CPID", "OPID", "FREQ"
        };

        if(sTUIparam.sort_col > 0
                && sTUIparam.sort_col
                <= STREAM_NB_SORT_COLS)
        {
            screenprint_setcolor(6);
            TUI_printfw(
                "[SORT: %s %s]  ",
                sort_col_names[sTUIparam.sort_col], sTUIparam.sort_dir ? "DESC" : "ASC");
            screenprint_unsetcolor(6);
        }
        else if(sTUIparam.SORTING > 0)
        {
            screenprint_setcolor(6);
            TUI_printfw("[SORT: mode %d]  ", sTUIparam.SORTING);
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

    int lastindex = 0;
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
            (long)lastindex, sTUIparam.dindexSelected, ssIDselected, (int) inodeselected);
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
                DispPID_NBchar, "creaPID", DispPID_NBchar, "ownPID", Dispfreq_NBchar, "   frequ ");

    switch(sTUIparam.DisplayMode)
    {
    case DISPLAY_MODE_SUMMARY: TUI_printfw("     Semaphore values ....");
        TUI_newline();
        break;

    case DISPLAY_MODE_WRITE: TUI_printfw("     write PIDs ....");
        TUI_newline();
        break;

    case DISPLAY_MODE_READ: TUI_printfw("     read PIDs ....");
        TUI_newline();
        break;

    case DISPLAY_MODE_SPTRACE:
        TUI_printfw("     stream process traces:   \"(INODE " "TYPE/SEM PID)>\"");
        TUI_newline();
        break;

    case DISPLAY_MODE_FUSER: TUI_printfw("     connected processes");
        TUI_newline();
        break;

    default: TUI_newline();
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
        if(streaminfo[sindex].erased == 1)
        {
            continue;
        }
        imageID ID = streaminfo[sindex].ID;
        /* ID == -1 means discovered but not yet mmap'd.
         * Include it so names appear immediately. */
        if(ID >= 0 && streamCTRLimages[ID].used == 0)
        {
            continue;
        }

        sTUIparam.ssindex[sTUIparam.NBsindex] = sindex;
        sTUIparam.NBsindex++;
    }

    DEBUG_TRACEPOINT(" ");

    // compute dynamic lengths
    int max_name_len = 10;
    for(int dindex = 0; dindex < sTUIparam.NBsindex; dindex++)
    {
        int len = strlen(streaminfo[sTUIparam.ssindex[dindex]].sname);
        if(len > max_name_len)
        {
            max_name_len = len;
        }
    }
    DispName_NBchar = max_name_len + 2;

    /* ---- Column-based sort (new) ---- */
    if(sTUIparam.sort_col > STREAM_SORT_NONE
            && sTUIparam.SORTING == 0)
    {
        g_streaminfo_qsort = streaminfo;
        g_sort_images = streamCTRLimages;
        g_sort_col = sTUIparam.sort_col;
        g_sort_dir = sTUIparam.sort_dir;

        qsort(sTUIparam.ssindex, sTUIparam.NBsindex, sizeof(long), cmp_stream_col);
    }

    /* ---- Legacy sort mode 1: alphabetical ---- */
    if(sTUIparam.SORTING == 1)
    {
        g_streaminfo_qsort = streaminfo;
        g_sort_images = streamCTRLimages;
        g_sort_col = STREAM_SORT_NAME;
        g_sort_dir = 0;

        qsort(sTUIparam.ssindex, sTUIparam.NBsindex, sizeof(long), cmp_stream_col);
    }

    /* ---- Legacy sort modes 2/3: update recency ---- */
    if((sTUIparam.SORTING == 2) ||
            (sTUIparam.SORTING == 3))
    {
        long   *larray;
        double *varray;
        larray = (long *) malloc(sizeof(long) * sTUIparam.NBsindex);
        varray = (double *) malloc(sizeof(double) * sTUIparam.NBsindex);

        if(sTUIparam.SORT_TOGGLE == 1)
        {
            for(long i = 0;
                    i < sTUIparam.NBsindex; i++)
            {
                long si = sTUIparam.ssindex[i];
                streaminfo[si] .updatevalue_frozen = streaminfo[si].updatevalue;
            }

            if(sTUIparam.SORTING == 3)
            {
                for(long i = 0;
                        i < sTUIparam.NBsindex; i++)
                {
                    long si = sTUIparam.ssindex[i];
                    streaminfo[si]
                    .updatevalue_frozen += 10000.0 * streaminfo[si] .streamOpenPID_cnt1;
                }
            }

            sTUIparam.SORT_TOGGLE = 0;
        }

        for(long i = 0;
                i < sTUIparam.NBsindex; i++)
        {
            long si = sTUIparam.ssindex[i];
            larray[i] = si;
            varray[i] = streaminfo[si] .updatevalue_frozen;
        }

        if(sTUIparam.NBsindex > 1)
        {
            quick_sort2l(varray, larray, sTUIparam.NBsindex);
        }

        for(long i = 0;
                i < sTUIparam.NBsindex; i++)
        {
            sTUIparam.ssindex[sTUIparam.NBsindex - i - 1] = larray[i];
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
        doffsetindex = sTUIparam.dindexSelected - NBsinfodisp + 1;
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
    if(sTUIparam.DisplayMode < DISPLAY_MODE_FUSER)
    {
        /* Column labels and their sort IDs */
        struct
        {
            const char *label;
            int col_id;
            int width;
        } cols[] =
        {
            {"NAME",  STREAM_SORT_NAME, DispName_NBchar},
            {"TYPE",  STREAM_SORT_TYPE, 4},
            {"SIZE",  STREAM_SORT_SIZE, DispSize_NBchar},
            {"CNT0",  STREAM_SORT_CNT0, Dispcnt0_NBchar + 2},
            {"CPID",  STREAM_SORT_CPID, 9},
            {"OPID",  STREAM_SORT_OPID, 9},
            {"FREQ",  STREAM_SORT_FREQ, 9},
        };
        int ncols = 7;

        /* inode placeholder */
        TUI_printfw("          ");

        for(int ci = 0; ci < ncols; ci++)
        {
            int is_active = (cols[ci].col_id == sTUIparam.sort_col);

            if(is_active)
            {
                screenprint_setbold();
                screenprint_setcolor(6);
            }

            char arrow = ' ';
            if(is_active)
            {
                arrow = sTUIparam.sort_dir ? '\x19' : '\x18';
            }

            TUI_printfw("%-*.*s%c", cols[ci].width - 1, cols[ci].width - 1, cols[ci].label, arrow);

            if(is_active)
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
    frame_t_sec = frame_ts.tv_sec + frame_ts.tv_nsec * 1e-9;

    ansi_detect_color_level();
    frame_color_level = ansi__color_level;

    /* Record where data rows start for mouse click mapping.
     * sc_cursor_row is now past all header/column-header rows. */
    state->body_start_row = sc_cursor_row;

    if(NBsinfodisp_out)
    {
        *NBsinfodisp_out = NBsinfodisp;
    }
    if(lastindex_out)
    {
        *lastindex_out = lastindex;
    }
    if(frame_t_sec_out)
    {
        *frame_t_sec_out = frame_t_sec;
    }
    if(frame_color_level_out)
    {
        *frame_color_level_out = frame_color_level;
    }
}
