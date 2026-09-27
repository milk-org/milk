// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_detail_stream.c
 * @brief   Stream detail and data lineage rendering for milk-CTRL.
 */

#include "overview_render_detail_internal.h"
#include <stdio.h>
#include <string.h>

/**
 * render_lineage_group - Render one lineage group (ancestors or descendants).
 *
 * Shared helper to avoid duplicating the ancestor/descendant rendering
 * logic. The only differences are the label, entry array, count, and
 * the sign character used as a depth prefix.
 *
 * @lay:     layout state
 * @m:       overview model
 * @entries: array of SG_LINEAGE_ENTRY (ancestors or descendants)
 * @nb:      number of entries
 * @label:   section header label (e.g. "Ancestors [FPS] (3):")
 * @sign:    depth-prefix character, '-' for ancestors, '+' for descendants
 * @ri:      row render index (modified in place)
 * @line_idx: logical line counter (modified in place)
 * @r:       panel rect
 * @row:     absolute start row
 * @max_rows: rendering row limit
 */
static void render_lineage_group(OV_LAYOUT              *lay,
                                 const OV_MODEL         *m,
                                 const SG_LINEAGE_ENTRY *entries,
                                 int                     nb,
                                 const char             *label,
                                 char                    sign,
                                 int                    *ri,
                                 int                    *line_idx,
                                 OV_RECT                 r,
                                 int                     row,
                                 int                     max_rows)
{
    /* Local skip predicate: pointer-safe version of the file-level skip_draw.
     * skip_draw uses bare 'ri'/'line_idx' names which don't work with
     * pointer parameters, so we dereference explicitly here.
     */
#define _LG_SKIP (*line_idx < lay->scroll_detail || *ri >= max_rows)

    if (!_LG_SKIP)
    {
        ov_buf_pos(row + *ri, r.col + 1);
    }
    if (!_LG_SKIP)
    {
        ov_theme_bg(OV_BG_PANEL);
    }
    if (!_LG_SKIP)
    {
        ov_theme_fg(OV_FG_TITLE);
    }
    if (!_LG_SKIP)
    {
        ov_buf_bold();
    }
    int printed = snprintf(NULL, 0, "%s", label);
    if (!_LG_SKIP)
    {
        ov_buf_printf("%s", label);
    }
    if (!_LG_SKIP)
    {
        ov_buf_reset_attr();
    }
    if (!_LG_SKIP)
    {
        ov_theme_bg(OV_BG_PANEL);
    }

    for (int ii = 0; ii < nb; ii++)
    {
        const SG_LINEAGE_ENTRY *le = &entries[ii];
        const char             *sn = m->streams[le->stream_idx].name;

        int item_len = snprintf(NULL, 0, "  %c%d %s", sign, le->depth, sn);
        if (le->via_name[0] != '\0')
        {
            item_len += snprintf(NULL, 0, "(%s)", le->via_name);
        }

        if (printed + item_len >= r.width - 2)
        {
            if (!_LG_SKIP)
            {
                render_pad_spaces(printed, r.width);
            }
            if (!_LG_SKIP)
            {
                (*ri)++;
            }
            (*line_idx)++;
            if (*ri >= max_rows)
            {
                break;
            }
            if (!_LG_SKIP)
            {
                ov_buf_pos(row + *ri, r.col + 1);
            }
            if (!_LG_SKIP)
            {
                ov_theme_bg(OV_BG_PANEL);
            }
            printed  = 0;
            item_len = snprintf(NULL, 0, " %c%d %s", sign, le->depth, sn);
            if (le->via_name[0] != '\0')
            {
                item_len += snprintf(NULL, 0, "(%s)", le->via_name);
            }
        } // if line wrap needed

        if (!_LG_SKIP)
        {
            ov_theme_fg(le->depth == 1 ? OV_FG_STREAM : OV_FG_DIM);
        }

        if (printed == 0)
        {
            int p1 = snprintf(NULL, 0, " %c%d %s", sign, le->depth, sn);
            if (!_LG_SKIP)
            {
                ov_buf_printf(" %c%d %s", sign, le->depth, sn);
            }
            printed += p1;
        }
        else
        {
            int p1 = snprintf(NULL, 0, "  %c%d %s", sign, le->depth, sn);
            if (!_LG_SKIP)
            {
                ov_buf_printf("  %c%d %s", sign, le->depth, sn);
            }
            printed += p1;
        }

        if (le->via_name[0] != '\0')
        {
            if (!_LG_SKIP)
            {
                ov_theme_fg(OV_FG_PROC);
            }
            if (!_LG_SKIP)
            {
                ov_buf_printf("(%s)", le->via_name);
            }
            printed += snprintf(NULL, 0, "(%s)", le->via_name);
        }
    } // for lineage entries

    if (!_LG_SKIP)
    {
        render_pad_spaces(printed, r.width);
    }
    if (!_LG_SKIP)
    {
        (*ri)++;
    }
    (*line_idx)++;

#undef _LG_SKIP
} // render_lineage_group


/**
 * ov_fps__render_detail_stream_lineage - render stream lineage (ancestors and descendants).
 * @lay:      Pointer to layout structure.
 * @m:        Pointer to data model snapshot.
 * @ssel:     Selected stream index in model.
 * @r:        Bounding rectangle of detail panel.
 * @ri:       Current rendering row offset within panel.
 * @line_idx: Logical line counter for vertical scrolling.
 * @row:      Base row coordinate on terminal.
 * @max_rows: Maximum visible rows in panel.
 *
 * Return: 1 on success, 0 otherwise.
 */
static int ov_fps__render_detail_stream_lineage(OV_LAYOUT      *lay,
                                                const OV_MODEL *m,
                                                int             ssel,
                                                OV_RECT         r,
                                                int            *ri,
                                                int            *line_idx,
                                                int             row,
                                                int             max_rows)
{
    SG_LINEAGE lin;
    sg_compute_lineage(m, ssel, (sg_mode_t) lay->lineage_mode, &lin);

    int has_lineage = (lin.nb_ancestors > 0 || lin.nb_descendants > 0);
    if (!has_lineage)
    {
        return 0;
    }

    /* Blank separator line */
    clear_row(row + *ri, r.col + 1, r.width - 2, OV_BG_PANEL);
    if (!(*line_idx < lay->scroll_detail || *ri >= max_rows))
    {
        (*ri)++;
    }
    (*line_idx)++;

    const char *ml = sg_mode_label((sg_mode_t) lay->lineage_mode);
    char        label_buf[64];

    if (lin.nb_ancestors > 0)
    {
        snprintf(label_buf, sizeof(label_buf), " Ancestors [%s] (%d):", ml, lin.nb_ancestors);
        render_lineage_group(lay, m, lin.ancestors, lin.nb_ancestors, label_buf, '-', ri, line_idx,
                             r, row, max_rows);
    } // if ancestors

    if (lin.nb_descendants > 0)
    {
        snprintf(label_buf, sizeof(label_buf), " Descendants [%s] (%d):", ml, lin.nb_descendants);
        render_lineage_group(lay, m, lin.descendants, lin.nb_descendants, label_buf, '+', ri,
                             line_idx, r, row, max_rows);
    } // if descendants

    return 1;
} // ov_fps__render_detail_stream_lineage


/**
 * @brief Render detailed stream info in the panel.
 */
int ov_fps__render_detail_stream(OV_LAYOUT      *lay,
                                 const OV_MODEL *m,
                                 int             ssel,
                                 OV_RECT         r,
                                 int             max_rows,
                                 int             row)
{
    const OV_STREAM *s = &m->streams[ssel];

    const char *tabs[] = { "CONNECTIONS", "LOOPS", "DETAILS", "RESOURCES" };
    ov_draw_panel_tabs(r.row, r.col, r.height, r.width, tabs, 4, lay->graph_tab_mode, OV_FG_STREAM,
                       lay->focus == OV_FOCUS_GRAPH);

    int ri       = 0;
    int line_idx = 0;

    /* Name */
    {
        H_ov_buf_pos(row + ri, r.col + 1);
        H_ov_theme_bg(OV_BG_PANEL);
        H_ov_theme_fg(OV_FG_TITLE);
        H_ov_buf_bold();
        int n = snprintf(NULL, 0, " %s", s->name);
        H_ov_buf_printf(" %s", s->name);
        H_ov_buf_reset_attr();
        H_ov_theme_bg(OV_BG_PANEL);
        H_render_pad_spaces(n, r.width);
        if (!skip_draw)
        {
            ri++;
        }
        line_idx++;
    }

    /* Type + Size */
    {
        char szb[48];
        if (s->naxis == 1)
        {
            snprintf(szb, sizeof(szb), "%u", (unsigned) s->size[0]);
        }
        else if (s->naxis == 2)
        {
            snprintf(szb, sizeof(szb), "%ux%u", (unsigned) s->size[0], (unsigned) s->size[1]);
        }
        else
        {
            snprintf(szb, sizeof(szb), "%ux%ux%u", (unsigned) s->size[0], (unsigned) s->size[1],
                     (unsigned) s->size[2]);
        }
        H_ov_buf_pos(row + ri, r.col + 1);
        H_ov_theme_bg(OV_BG_PANEL);
        H_ov_theme_fg(OV_FG_DIM);
        int n = snprintf(NULL, 0, " Type: %s  Size: %s  Elements: %" PRIu64,
                         render_dtype(s->datatype), szb, (uint64_t) s->nelement);
        H_ov_buf_printf(" Type: %s  Size: %s  Elements: %" PRIu64, render_dtype(s->datatype), szb,
                        (uint64_t) s->nelement);
        H_render_pad_spaces(n, r.width);
        if (!skip_draw)
        {
            ri++;
        }
        line_idx++;
    }

    /* Counters */
    {
        H_ov_buf_pos(row + ri, r.col + 1);
        H_ov_theme_bg(OV_BG_PANEL);
        H_ov_theme_fg(OV_FG_DIM);
        H_ov_buf_printf(" cnt0: ");
        H_ov_theme_fg(s->cnt_active ? OV_FG_ACTIVE : OV_FG_DIM);
        H_ov_buf_printf("%" PRIu64, (uint64_t) s->cnt0);
        H_ov_theme_fg(OV_FG_DIM);
        H_ov_buf_printf("  Hz: ");
        H_ov_theme_fg(s->cnt_active ? OV_FG_ACTIVE : OV_FG_DIM);
        int n = snprintf(NULL, 0, " cnt0: %" PRIu64 "  Hz: %.1f", (uint64_t) s->cnt0, s->update_hz);
        H_ov_buf_printf("%.1f", s->update_hz);
        H_render_pad_spaces(n, r.width);
        if (!skip_draw)
        {
            ri++;
        }
        line_idx++;
    }

    /* PIDs */
    {
        H_ov_buf_pos(row + ri, r.col + 1);
        H_ov_theme_bg(OV_BG_PANEL);
        H_ov_theme_fg(OV_FG_DIM);
        int n = snprintf(NULL, 0, " Creator: %d  Owner: %d  Inode: %" PRIu64, (int) s->creatorPID,
                         (int) s->ownerPID, (uint64_t) s->inode);
        H_ov_buf_printf(" Creator: %d  Owner: %d  Inode: %" PRIu64, (int) s->creatorPID,
                        (int) s->ownerPID, (uint64_t) s->inode);
        H_render_pad_spaces(n, r.width);
        if (!skip_draw)
        {
            ri++;
        }
        line_idx++;
    }

    /* Semaphores */
    if (s->nb_sem > 0)
    {
        {
            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_bg(OV_BG_PANEL);
            H_ov_theme_fg(OV_FG_TITLE);
            H_ov_buf_bold();
            int n = snprintf(NULL, 0, " Semaphores (%d):", s->nb_sem);
            H_ov_buf_printf(" Semaphores (%d):", s->nb_sem);
            H_ov_buf_reset_attr();
            H_ov_theme_bg(OV_BG_PANEL);
            H_render_pad_spaces(n, r.width);
            if (!skip_draw)
            {
                ri++;
            }
            line_idx++;
        }
        for (int ii = 0; ii < s->nb_sem; ii++)
        {
            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_bg(OV_BG_PANEL);
            int val = s->semval[ii];
            H_ov_theme_fg(val > 0 ? OV_FG_WARN : OV_FG_TEXT);
            char rpid_str[32] = "";
            if (s->read_pids[ii] > 0)
            {
                snprintf(rpid_str, sizeof(rpid_str), "reader:%d", (int) s->read_pids[ii]);
            }
            int n = snprintf(NULL, 0, "  [%d] val:%d  %s", ii, val, rpid_str);
            H_ov_buf_printf("  [%d] val:%d  %s", ii, val, rpid_str);
            H_render_pad_spaces(n, r.width);
            if (!skip_draw)
            {
                ri++;
            }
            line_idx++;
        } // for semaphores
    } // if nb_sem > 0
    /* Proctrace entries */
    if (s->nb_proctrace > 0)
    {
        H_ov_buf_pos(row + ri, r.col + 1);
        H_ov_theme_bg(OV_BG_PANEL);
        H_ov_theme_fg(OV_FG_TITLE);
        H_ov_buf_bold();
        int n = snprintf(NULL, 0, " Process trace (%d):", s->nb_proctrace);
        H_ov_buf_printf(" Process trace (%d):", s->nb_proctrace);
        H_ov_buf_reset_attr();
        H_ov_theme_bg(OV_BG_PANEL);
        H_render_pad_spaces(n, r.width);
        if (!skip_draw)
        {
            ri++;
        }
        line_idx++;

        for (int tt = 0; tt < s->nb_proctrace; tt++)
        {
            /* Find proc name by PID */
            const char *pname = "???";
            for (int pp = 0; pp < m->nb_procs; pp++)
            {
                if (m->procs[pp].PID == s->proctrace_pid[tt])
                {
                    pname = m->procs[pp].name;
                    break;
                }
            } // for procs
            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_bg(OV_BG_PANEL);
            H_ov_theme_fg(ov_pid_color(s->proctrace_pid[tt]));
            int n2 = snprintf(NULL, 0, "  PID %d (%s)  mode:%s", (int) s->proctrace_pid[tt], pname,
                              render_trigmode_label(s->proctrace_trigmode[tt]));
            H_ov_buf_printf("  PID %d (%s)  mode:%s", (int) s->proctrace_pid[tt], pname,
                            render_trigmode_label(s->proctrace_trigmode[tt]));
            H_render_pad_spaces(n2, r.width);
            if (!skip_draw)
            {
                ri++;
            }
            line_idx++;
        } // for proctrace
    } // if nb_proctrace > 0

    /* Stream lineage (ancestors + descendants) */
    ov_fps__render_detail_stream_lineage(lay, m, ssel, r, &ri, &line_idx, row, max_rows);

    /* Clear remaining rows */
    lay->detail_total_lines = line_idx;
    for (; ri < max_rows; ri++)
    {
        clear_row(row + ri, r.col + 1, r.width - 2, OV_BG_PANEL);
    }
    H_ov_buf_reset_attr();
    return 1;
} // ov_fps__render_detail_stream
