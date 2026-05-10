/**
 * @file    overview_render_help.c
 */

#include "overview_render_internal.h"

void ov_render_help(const OV_LAYOUT *lay)
{
    /* ---- Content table ---- */
    static const struct
    {
        const char *text;
        int         color_row; /* 1 = render as colour-legend row */
    } lines[] =
    {
        { "Navigation",                              0 },
        { "  F2-F6 / ^Left/^Right  Switch views",    0 },
        { "  TAB      Cycle panel focus",            0 },
        { "  UP/DOWN  Navigate list",                0 },
        { "  Left/Right  Panel focus / scroll",      0 },
        { "  PgUp/Dn  Scroll page",                  0 },
        { "  Home/End  Jump to top/bottom",          0 },
        { "Sorting",                                 0 },
        { "  </>      Change sort column",           0 },
        { "  [        Toggle sort direction",        0 },
        { "  S        Sort by activity (Hz)",        0 },
        { "  s        Sort by name",                 0 },
        { "Display",                                 0 },
        { "  +/-      Adjust scan rate",             0 },
        { "  D        Toggle detail pane",           0 },
        { "  L        Toggle lineage mode",          0 },
        { "  p        Pause/resume display",         0 },
        { "  SPACE    Freeze selection highlight",   0 },
        { "  /        Filter (regex search)",        0 },
        { "  W        Export snapshot to file",      0 },
        { "  h        Toggle this help",             0 },
        { "  q        Quit",                         0 },
        { "Control mode  (c to toggle)",             0 },
        { "  FPS:  r=run  s=conf",                   0 },
        { "  STRM: d=delete stream",                 0 },
        { "Process signals  (PROCS & FPS panels)",   0 },
        { "  k  graceful kill (SIGTERM)",            0 },
        { "  K  immediate kill (SIGKILL)",           0 },
        { "  x  pause/resume (SIGSTOP/SIGCONT)",     0 },
        { "Detail View (ENTER or D)",                0 },
        { "  Toggles detail pane for selected item", 0 },
        { "Columns",                                 0 },
        { "  STRM: MB/s throughput, total in footer",0 },
        { "  PROC: DUTY% (exec/iter), CPU%, MEM",    0 },
        { "  FPS:  MEM (RSS)",                       0 },
        { "Mouse",                                     0 },
        { "  Click=select  DblClick=detail",           0 },
        { "  Scroll wheel=navigate list",              0 },
        { "",                                          0 },
        { "Colors:  ",                                 1 },
    };
    static const int NL = (int)(sizeof(lines) / sizeof(lines[0]));

    /* Compute exact box dimensions from content */
    int content_w = 0;
    for (int i = 0; i < NL; i++)
    {
        int l = (int) strlen(lines[i].text);
        if (l > content_w)
        {
            content_w = l;
        }
    }
    /* Add " stream proc fps" for the color legend row */
    int legend_extra = (int) strlen(" stream proc fps");
    int last_text_w  =
        (int) strlen(lines[NL - 1].text) + legend_extra;
    if (last_text_w > content_w)
    {
        content_w = last_text_w;
    }

    /* Box: 2-char left pad + content + 2-char right pad + 2 borders */
    int pw = content_w + 4 + 2;
    /* Box height: 2 borders + 1 top margin + NL lines + 1 bottom margin */
    int ph = NL + 4;

    int W  = lay->term_cols;
    int H  = lay->term_rows;

    int pr = (H - ph) / 2;
    int pc = (W - pw) / 2;
    if (pr < 1) { pr = 1; }
    if (pc < 1) { pc = 1; }

    /* ---- Step 1: dim overlay for the whole terminal ---- */
    /* REMOVED: no longer overwriting the whole terminal so background
     * panels remain visible. */

    /* ---- Step 2: border + interior ---- */
    ov_draw_panel_border(pr, pc, ph, pw, "HELP", OV_FG_BRIGHT, 1);

    for (int r = pr + 1; r < pr + ph - 1; r++)
    {
        clear_row(r, pc + 1, pw - 2, OV_BG_PANEL);
    }

    /* ---- Step 3: text content ---- */
    int row = pr + 2;
    for (int i = 0; i < NL; i++)
    {
        ov_buf_pos(row, pc + 3);
        ov_theme_bg(OV_BG_PANEL);

        /* Section headers: non-empty lines that
         * don't start with a space */
        const char *t = lines[i].text;
        int is_section = (t[0] != '\0'
                          && t[0] != ' '
                          && !lines[i].color_row);
        if (is_section)
        {
            ov_theme_fg(OV_FG_TITLE);
            ov_buf_bold();
        }
        else
        {
            ov_theme_fg(OV_FG_TEXT);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
        }

        if (lines[i].color_row)
        {
            /* Color legend: print label then colored words */
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("%s", lines[i].text);
            ov_theme_fg(OV_FG_STREAM);
            ov_buf_printf("stream ");
            ov_theme_fg(OV_FG_PROC);
            ov_buf_printf("proc ");
            ov_theme_fg(OV_FG_FPS);
            ov_buf_printf("fps");
        }
        else
        {
            ov_buf_printf("%-*s", content_w, lines[i].text);
        }

        ov_buf_reset_attr();
        row++;
    }

    ov_buf_reset_attr();
}

