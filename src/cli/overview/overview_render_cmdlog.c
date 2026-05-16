/**
 * @file overview_render_cmdlog.c
 * @brief Render the command log strip for milk-CTRL
 *
 * Draws the most recent N log entries in a strip
 * above the status bar.  Each entry shows a timestamp,
 * a color-coded status bullet, and the message text.
 */

#include "overview_render_internal.h"

/**
 * ov_render_cmdlog - render the command log panel.
 * @lay: layout state (contains cmdlog + r_cmdlog)
 *
 * Draws up to lay->r_cmdlog.height rows, most recent
 * entry at the bottom.  Skips rendering when
 * cmdlog_rows == 0.
 */
void ov_render_cmdlog(const OV_LAYOUT *lay)
{
    OV_RECT r = lay->r_cmdlog;
    if (r.height <= 0 || r.width <= 0)
    {
        return;
    }

    const OV_CMDLOG *log = &lay->cmdlog;

    /* Number of entries to show (capped by panel height) */
    int show = log->count;
    if (show > r.height)
    {
        show = r.height;
    }

    /* Starting index in ring buffer for oldest visible entry.
     * head points to next write, so the most recent entry
     * is at (head - 1), and the oldest visible is at
     * (head - show). */
    int start = (log->head - show + OV_CMDLOG_MAX)
                % OV_CMDLOG_MAX;

    /* Dark background for the log strip */
    ov_rgb_t bg = {20, 20, 30};

    /* Render each row */
    for (int row = 0; row < r.height; row++)
    {
        ov_buf_pos(r.row + row, r.col);
        ov_theme_bg(bg);

        if (row >= show)
        {
            /* Empty row — fill with background */
            ov_theme_fg(OV_FG_DIM);
            ov_buf_hline(' ', r.width);
            continue;
        }

        int idx = (start + row) % OV_CMDLOG_MAX;
        const OV_CMDLOG_ENTRY *e = &log->entries[idx];

        /* Format timestamp HH:MM:SS */
        struct tm tm_buf;
        localtime_r(&e->ts.tv_sec, &tm_buf);
        char tstr[12];
        snprintf(tstr, sizeof(tstr),
                 "%02d:%02d:%02d",
                 tm_buf.tm_hour,
                 tm_buf.tm_min,
                 tm_buf.tm_sec);

        /* Status bullet color */
        ov_rgb_t bullet_fg;
        const char *bullet;
        switch (e->level)
        {
        case OV_CMDLOG_OK:
            bullet_fg = (ov_rgb_t){80, 220, 80};
            bullet    = "●";
            break;
        case OV_CMDLOG_FAIL:
            bullet_fg = (ov_rgb_t){220, 60, 60};
            bullet    = "●";
            break;
        case OV_CMDLOG_WARN:
            bullet_fg = (ov_rgb_t){220, 180, 40};
            bullet    = "●";
            break;
        default: /* INFO */
            bullet_fg = (ov_rgb_t){100, 100, 120};
            bullet    = "·";
            break;
        }

        /* Dim timestamp */
        ov_buf_fg(80, 80, 100);
        int nw = snprintf(NULL, 0, " %s ", tstr);
        ov_buf_printf(" %s ", tstr);

        /* Status bullet */
        ov_buf_fg(bullet_fg.r, bullet_fg.g, bullet_fg.b);
        ov_buf_printf("%s ", bullet);
        nw += 4; /* bullet(1) + space + 2 for UTF-8 */

        /* Message text */
        ov_buf_fg(180, 180, 200);
        int msg_max = r.width - nw - 1;
        if (msg_max < 0)
        {
            msg_max = 0;
        }
        int msg_len = (int) strlen(e->msg);
        if (msg_len > msg_max)
        {
            msg_len = msg_max;
        }
        ov_buf_printf("%.*s", msg_len, e->msg);
        nw += msg_len;

        /* Pad remainder */
        int pad = r.width - nw;
        if (pad > 0)
        {
            ov_buf_hline(' ', pad);
        }
    }

    ov_buf_reset_attr();
}
