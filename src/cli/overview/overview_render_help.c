/**
 * @file    overview_render_help.c
 * @brief   Collapsible help overlay for milk-CTRL
 *
 * Organizes help into topic sections that the user
 * can navigate (UP/DOWN) and expand/collapse (ENTER).
 * Sections are stored as static data; expand state
 * lives in OV_LAYOUT.help_expand bitmask.
 */

#include "overview_render_internal.h"

/* ---- Help content definition ---- */

/** Flag values for the 'flags' field */
#define HF_SECTION  1   /* section header row     */
#define HF_CHILD    2   /* child of a section     */
#define HF_COLORS   4   /* render as color legend  */

typedef struct
{
    const char *text;
    int         flags;
    int         section;  /* index of parent section */
} help_line_t;

/*
 * Section indices (must match help_expand bitmask
 * positions).
 */
enum {
    HS_NAV = 0,
    HS_SORT,
    HS_DISPLAY,
    HS_CTRL,
    HS_SIGNALS,
    HS_DETAIL,
    HS_COLUMNS,
    HS_MOUSE,
    HS_COLORS,
    HS_COUNT
};

/* clang-format off */
static const help_line_t HELP[] =
{
    /* --- Navigation --- */
    { "Navigation",                             HF_SECTION, HS_NAV },
    { "  F2-F6 / ^Left/^Right  Switch views",  HF_CHILD,   HS_NAV },
    { "  TAB      Cycle panel focus",           HF_CHILD,   HS_NAV },
    { "  UP/DOWN  Navigate list",               HF_CHILD,   HS_NAV },
    { "  Left/Right  Panel focus / scroll",     HF_CHILD,   HS_NAV },
    { "  SHIFT+Up/Down Ancestry (Up/Downstream)", HF_CHILD,   HS_NAV },
    { "  PgUp/Dn  Scroll page",                 HF_CHILD,   HS_NAV },
    { "  Home/End  Jump to top/bottom",         HF_CHILD,   HS_NAV },
    /* --- Sorting --- */
    { "Sorting",                                HF_SECTION, HS_SORT },
    { "  </> or ,/.  Change sort column",       HF_CHILD,   HS_SORT },
    { "  [           Toggle sort direction",    HF_CHILD,   HS_SORT },
    { "  S           Sort by activity (Hz)",    HF_CHILD,   HS_SORT },
    { "  s           Sort by name",             HF_CHILD,   HS_SORT },
    /* --- Display --- */
    { "Display",                                HF_SECTION, HS_DISPLAY },
    { "  +/-      Adjust scan rate",            HF_CHILD,   HS_DISPLAY },
    { "  D        Toggle detail pane",          HF_CHILD,   HS_DISPLAY },
    { "  L        Toggle lineage mode",         HF_CHILD,   HS_DISPLAY },
    { "  F        Pause/resume display",        HF_CHILD,   HS_DISPLAY },
    { "  SPACE    Freeze selection highlight",  HF_CHILD,   HS_DISPLAY },
    { "  /        Filter (regex search)",       HF_CHILD,   HS_DISPLAY },
    { "  W        Export snapshot to file",     HF_CHILD,   HS_DISPLAY },
    { "  G        Toggle command log panel",    HF_CHILD,   HS_DISPLAY },
    { "  v/V      Resize command log panel",    HF_CHILD,   HS_DISPLAY },
    { "  h        Toggle this help",            HF_CHILD,   HS_DISPLAY },
    { "  q        Quit",                        HF_CHILD,   HS_DISPLAY },
    /* --- Control mode --- */
    { "Control Mode  (c to toggle)",            HF_SECTION, HS_CTRL },
    { "  FPS:  r=run  s=conf",                  HF_CHILD,   HS_CTRL },
    { "  STRM: Del=delete stream",              HF_CHILD,   HS_CTRL },
    /* --- Process signals --- */
    { "Process Signals & Entries",              HF_SECTION, HS_SIGNALS },
    { "  k   graceful kill (SIGTERM)",          HF_CHILD,   HS_SIGNALS },
    { "  K   immediate kill (SIGKILL)",         HF_CHILD,   HS_SIGNALS },
    { "  x   pause/resume (SIGSTOP/SIGCONT)",   HF_CHILD,   HS_SIGNALS },
    { "  p   pause/resume (Control Mode)",      HF_CHILD,   HS_SIGNALS },
    { "  ^s  step one iter (Control Mode)",     HF_CHILD,   HS_SIGNALS },
    { "  e   clean exit (Control Mode)",        HF_CHILD,   HS_SIGNALS },
    { "  z   zero counters (Control Mode)",     HF_CHILD,   HS_SIGNALS },
    { "  CTRL+e  erase entry (STRM/PROC/FPS)",    HF_CHILD,   HS_SIGNALS },
    /* --- Detail view --- */
    { "Detail View",                            HF_SECTION, HS_DETAIL },
    { "  ENTER or D  Toggle detail pane",       HF_CHILD,   HS_DETAIL },
    { "  Shows params/connections/lineage",     HF_CHILD,   HS_DETAIL },
    /* --- Columns --- */
    { "Columns",                                HF_SECTION, HS_COLUMNS },
    { "  STRM: MB/s throughput, total in footer", HF_CHILD, HS_COLUMNS },
    { "  PROC: DUTY% (exec/iter), CPU%, MEM",   HF_CHILD,   HS_COLUMNS },
    { "  FPS:  MEM (RSS)",                      HF_CHILD,   HS_COLUMNS },
    /* --- Mouse --- */
    { "Mouse",                                  HF_SECTION, HS_MOUSE },
    { "  Click=select  DblClick=detail",        HF_CHILD,   HS_MOUSE },
    { "  Scroll wheel=navigate list",           HF_CHILD,   HS_MOUSE },
    /* --- Colors --- */
    { "Colors",                                 HF_SECTION, HS_COLORS },
    { "",                                       HF_CHILD | HF_COLORS,
                                                            HS_COLORS },
};
/* clang-format on */

static const int HELP_TOTAL =
    (int)(sizeof(HELP) / sizeof(HELP[0]));

/**
 * help_is_expanded - check if a section is expanded.
 * @lay: layout state
 * @sec: section index (HS_NAV, etc.)
 */
static inline int help_is_expanded(
    const OV_LAYOUT *lay, int sec)
{
    return (lay->help_expand >> sec) & 1;
}

/**
 * help_nb_sections - return number of section headers.
 */
static int help_nb_sections(void)
{
    return HS_COUNT;
}

/**
 * help_section_header_idx - return HELP[] index of the
 * Nth section header.
 * @n: 0-based section ordinal
 */
static int help_section_header_idx(int n)
{
    int sec = 0;
    for (int i = 0; i < HELP_TOTAL; i++)
    {
        if (HELP[i].flags & HF_SECTION)
        {
            if (sec == n)
            {
                return i;
            }
            sec++;
        }
    }
    return 0;
}

/**
 * help_visible_rows - count how many rows are visible
 * with the current expand state, and build a mapping
 * from visible row → HELP[] index.
 *
 * @lay:   layout (for expand bitmask)
 * @map:   output array (caller must size ≥ HELP_TOTAL)
 *         map[visible_row] = HELP[] index
 *
 * Return: number of visible rows
 */
static int help_visible_rows(
    const OV_LAYOUT *lay, int *map)
{
    int vis = 0;
    for (int i = 0; i < HELP_TOTAL; i++)
    {
        if (HELP[i].flags & HF_SECTION)
        {
            map[vis++] = i;
        }
        else if (HELP[i].flags & HF_CHILD)
        {
            if (help_is_expanded(lay, HELP[i].section))
            {
                map[vis++] = i;
            }
        }
    }
    return vis;
}

/* ---- Render entry point ---- */

void ov_render_help(const OV_LAYOUT *lay)
{
    /* Build visible-row mapping */
    int map[128];
    int nvis = help_visible_rows(lay, map);

    /* Compute content width */
    int content_w = 0;
    for (int i = 0; i < HELP_TOTAL; i++)
    {
        int l = (int) strlen(HELP[i].text);
        /* Section headers get "▸ " or "▾ " prefix */
        if (HELP[i].flags & HF_SECTION)
        {
            l += 4; /* chevron + space + padding */
        }
        if (l > content_w)
        {
            content_w = l;
        }
    }
    /* Color legend row adds extra */
    int legend_extra = (int) strlen("  stream proc fps");
    if (legend_extra + 4 > content_w)
    {
        content_w = legend_extra + 4;
    }
    if (content_w < 44)
    {
        content_w = 44;
    }

    /* Box dimensions */
    int pw = content_w + 6;     /* 2 pad + 2 border */
    int ph = nvis + 5;          /* 2 border + title + spacer */

    int W = lay->term_cols;
    int H = lay->term_rows;

    /* Cap to terminal */
    if (ph > H - 2) { ph = H - 2; }
    if (pw > W - 2) { pw = W - 2; }

    int pr = (H - ph) / 2;
    int pc = (W - pw) / 2;
    if (pr < 1) { pr = 1; }
    if (pc < 1) { pc = 1; }

    /* Border */
    ov_draw_panel_border(
        pr, pc, ph, pw,
        "HELP  (↑↓ navigate  ENTER expand  h close)",
        OV_FG_BRIGHT, 1, 0);

    /* Clear interior */
    for (int r = pr + 1; r < pr + ph - 1; r++)
    {
        clear_row(r, pc + 1, pw - 2, OV_BG_PANEL);
    }

    /* Visible content area (inside border + title) */
    int body_top = pr + 2;
    int body_h   = ph - 4;
    if (body_h < 1) { body_h = 1; }

    /* Ensure sel is in range */
    int sel = lay->help_sel;
    if (sel < 0) { sel = 0; }
    if (sel >= nvis) { sel = nvis - 1; }

    /* Scroll so cursor is visible */
    int scroll = 0;
    if (sel >= body_h)
    {
        scroll = sel - body_h + 1;
    }

    /* Render visible rows */
    int inner_w = pw - 4;
    for (int vr = 0; vr < body_h && vr + scroll < nvis;
         vr++)
    {
        int idx = map[vr + scroll];
        const help_line_t *h = &HELP[idx];
        int row = body_top + vr;
        int is_sel = ((vr + scroll) == sel);

        ov_buf_pos(row, pc + 2);
        ov_theme_bg(OV_BG_PANEL);

        /* Highlight selected row */
        if (is_sel)
        {
            ov_buf_bg(40, 45, 70);
        }

        if (h->flags & HF_SECTION)
        {
            /* Section header with chevron */
            int expanded = help_is_expanded(
                               lay, h->section);
            const char *chev = expanded
                               ? "▾" : "▸";

            if (is_sel)
            {
                ov_buf_fg(255, 220, 100);
            }
            else
            {
                ov_theme_fg(OV_FG_TITLE);
            }
            ov_buf_bold();
            ov_buf_printf(" %s %s", chev, h->text);

            /* Pad remainder */
            int used = 3 + (int)strlen(h->text) + 2;
            int pad  = inner_w - used;
            if (pad > 0)
            {
                ov_buf_hline(' ', pad);
            }
        }
        else if (h->flags & HF_COLORS)
        {
            /* Color legend row */
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("  ");
            ov_theme_fg(OV_FG_STREAM);
            ov_buf_printf("stream ");
            ov_theme_fg(OV_FG_PROC);
            ov_buf_printf("proc ");
            ov_theme_fg(OV_FG_FPS);
            ov_buf_printf("fps");
            int used = 22;
            int pad  = inner_w - used;
            if (pad > 0)
            {
                ov_buf_hline(' ', pad);
            }
        }
        else
        {
            /* Normal child row */
            ov_theme_fg(OV_FG_TEXT);
            ov_buf_printf(" %-*s", inner_w - 1,
                          h->text);
        }

        ov_buf_reset_attr();
    }

    ov_buf_reset_attr();
}

/**
 * ov_help_nb_sections - return number of help sections.
 *
 * Used by ov_handle_key to validate help_sel range.
 */
int ov_help_nb_sections(void)
{
    return help_nb_sections();
}

/**
 * ov_help_visible_count - return number of visible rows
 * in the help panel with the current expand state.
 * @lay: layout state
 */
int ov_help_visible_count(const OV_LAYOUT *lay)
{
    int map[128];
    return help_visible_rows(lay, map);
}

/**
 * ov_help_toggle_at - toggle expand/collapse for the
 * visible row at index @vis_row.
 * @lay:     layout state (help_expand is modified)
 * @vis_row: 0-based visible row index
 *
 * If the row is a section header, toggles its expand
 * bit and returns the section index.  Otherwise
 * returns -1.
 */
int ov_help_toggle_at(OV_LAYOUT *lay, int vis_row)
{
    int map[128];
    int nvis = help_visible_rows(lay, map);

    if (vis_row < 0 || vis_row >= nvis)
    {
        return -1;
    }

    int idx = map[vis_row];
    if (!(HELP[idx].flags & HF_SECTION))
    {
        return -1;
    }

    int sec = HELP[idx].section;
    lay->help_expand ^= (1U << sec);
    return sec;
}

