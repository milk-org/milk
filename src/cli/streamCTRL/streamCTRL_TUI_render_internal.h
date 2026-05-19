#ifndef _STREAMCTRL_TUI_RENDER_INTERNAL_H
#define _STREAMCTRL_TUI_RENDER_INTERNAL_H

#include "streamCTRL_TUI_internal.h"

// Expose the macros for the split files
#define doffsetindex state->doffsetindex
#define monstrlen state->monstrlen
#define monstring state->monstring
#define DispName_NBchar state->DispName_NBchar
#define DispSize_NBchar state->DispSize_NBchar
#define Dispcnt0_NBchar state->Dispcnt0_NBchar
#define Dispfreq_NBchar state->Dispfreq_NBchar
#define DispPID_NBchar state->DispPID_NBchar
#define PIDmax state->PIDmax
#define PIDname_array state->PIDname_array
#define inodeselected state->inodeselected
#define NBupstreaminodeMAX state->NBupstreaminodeMAX
#define upstreaminode state->upstreaminode
#define NBupstreaminode state->NBupstreaminode
#define NBupstreamprocMAX state->NBupstreamprocMAX
#define upstreamproc state->upstreamproc
#define NBupstreamproc state->NBupstreamproc
#define loopcnt state->loopcnt
#define streaminfoproc (*streamCTRLdata->streaminfoproc)
#define streaminfo (streamCTRLdata->sinfo)
#define streamCTRLimages (streamCTRLdata->images)

// Inline static helpers from original file
static inline void streamCTRL_set_sem_color(int val)
{
    if(val == 0)
    {
        screenprint_setcolor(2); // green
    }
    else if(val >= 10)
    {
        screenprint_setcolor(4); // red
    }
    else
    {
        ansi_detect_color_level();
        if(ansi__color_level >= 3)
        {
            int r = 150 + (val - 1) * (255 - 150) / 9;
            int g = 100 - (val - 1) * 100 / 9;
            int b = 0;
            SC_APPEND("\033[38;2;%d;%d;%dm", r, g, b);
        }
        else if(ansi__color_level == 2)
        {
            if(val < 4)
            {
                SC_APPEND("\033[38;5;130m");
            }
            else if(val < 7)
            {
                SC_APPEND("\033[38;5;166m");
            }
            else
            {
                SC_APPEND("\033[38;5;196m");
            }
        }
        else
        {
            screenprint_setcolor(3);
        }
    }
}

static inline void streamCTRL_render_active_bg(
    const char *str,
    int        len,
    int        color_level)
{
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

static inline void streamCTRL_print_frequ_field(
    double frequ,
    double wave_age,
    int    color_level)
{
    char fbuf[32];

    if(frequ < 0.005)
    {
        snprintf(fbuf, sizeof(fbuf), " %7s Hz", "---");
    }
    else
    {
        snprintf(fbuf, sizeof(fbuf), " %7.2f Hz", frequ);
    }

    if(color_level < 2 || wave_age > 1.0)
    {
        TUI_printfw("%s", fbuf);
        return;
    }

    double log_br = 0.0;
    if(frequ >= 1.0)
    {
        log_br = log10(frequ) / log10(9999.0);
    }
    if(log_br > 1.0)
    {
        log_br = 1.0;
    }

    if(color_level >= 3)
    {
        int r = (int)(10.0 * log_br);
        int g = (int)(180.0 * log_br);
        int b = (int)(80.0 * log_br);
        SC_APPEND("\033[48;2;%d;%d;%dm", r, g, b);
    }
    else
    {
        int idx = (int)(5.0 * log_br);
        static const int ramp[6] =
        {
            17, 23, 29, 35, 41, 47
        };
        SC_APPEND("\033[48;5;%dm",
                  ramp[idx < 6 ? idx : 5]);
    }

    TUI_printfw("%s", fbuf);
    SC_APPEND("\033[0m");
}

void streamCTRL__render_header_help(
    streamCTRLarg_struct        *streamCTRLdata,
    struct streamCTRL_TUI_state *state);
void streamCTRL__render_header_streams(
    streamCTRLarg_struct        *streamCTRLdata,
    struct streamCTRL_TUI_state *state,
    int                         *NBsinfodisp_out,
    int                         *lastindex_out,
    double                      *frame_t_sec_out,
    int                         *frame_color_level_out);
void streamCTRL__render_stream_rows(
    streamCTRLarg_struct        *streamCTRLdata,
    struct streamCTRL_TUI_state *state,
    int                         NBsinfodisp,
    double                      frame_t_sec,
    int                         frame_color_level);
void streamCTRL__render_footer(
    streamCTRLarg_struct        *streamCTRLdata,
    struct streamCTRL_TUI_state *state,
    int                         NBsinfodisp);

extern int cmp_stream_col(
    const void *a,
    const void *b);

#endif
