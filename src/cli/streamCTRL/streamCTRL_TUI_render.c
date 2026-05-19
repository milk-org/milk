#include "streamCTRL_TUI_render_internal.h"

extern int cmp_stream_col(
    const void *a,
    const void *b);

/**
 * @brief Render one complete streamCTRL TUI frame.
 *
 * Composes header, stream rows, and footer.
 */
void streamCTRL_render_screen(
    streamCTRLarg_struct        *streamCTRLdata,
    struct streamCTRL_TUI_state *state)
{
    int NBsinfodisp = 10;
    int lastindex = 0;
    double frame_t_sec = 0.0;
    int frame_color_level = 0;

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

    screenprint_setbold();
    snprintf(monstring, monstrlen, "[%d x %d] [PID %d] STREAM MONITOR: PRESS (x) TO STOP, (h) FOR HELP",
             wrow, wcol, getpid());
    DEBUG_TRACEPOINT("Print header");
    screenprint_setcolor(12);
    TUI_print_header(monstring, '-');
    screenprint_unsetcolor(12);
    screenprint_unsetbold();

    DEBUG_TRACEPOINT("Start display");

    if(sTUIparam.DisplayMode == DISPLAY_MODE_HELP)
    {
        streamCTRL__render_header_help(streamCTRLdata, state);
    }
    else
    {
        streamCTRL__render_header_streams(streamCTRLdata, state, &NBsinfodisp, &lastindex, &frame_t_sec,
                                          &frame_color_level);
        streamCTRL__render_stream_rows(streamCTRLdata, state, NBsinfodisp, frame_t_sec, frame_color_level);
    }

    streamCTRL__render_footer(streamCTRLdata, state, NBsinfodisp);
}
