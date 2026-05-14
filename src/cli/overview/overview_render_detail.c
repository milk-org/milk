#include "overview_render_internal.h"

#define skip_draw (line_idx < lay->scroll_detail || ri >= max_rows)

#define H_ov_buf_pos(r, c) if (!(skip_draw)) ov_buf_pos(r, c)
#define H_ov_theme_bg(c) if (!(skip_draw)) ov_theme_bg(c)
#define H_ov_theme_fg(c) if (!(skip_draw)) ov_theme_fg(c)
#define H_ov_buf_bold() if (!(skip_draw)) ov_buf_bold()
#define H_ov_buf_reset_attr() if (!(skip_draw)) ov_buf_reset_attr()
#define H_ov_buf_printf(...) if (!(skip_draw)) ov_buf_printf(__VA_ARGS__)
#define H_render_pad_spaces(n, w) if (!(skip_draw)) render_pad_spaces(n, w)
static int ov_fps__render_detail_stream(
    OV_LAYOUT       *lay,
    const OV_MODEL *m,
    int ssel,
    OV_RECT r,
    int max_rows,
    int row)
{

        const OV_STREAM *s =
            &m->streams[ssel];

        const char *tabs[] = {"CONNECTIONS", "DETAILS", "RESOURCES"};
        ov_draw_panel_tabs(
            r.row, r.col, r.height, r.width,
            tabs, 3, lay->graph_tab_mode, OV_FG_STREAM, lay->focus == OV_FOCUS_GRAPH);

        int ri = 0;
    int line_idx = 0;

        /* Name */
        if (1)
        {
            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_bg(OV_BG_PANEL);
            H_ov_theme_fg(OV_FG_TITLE);
            H_ov_buf_bold();
            int n = snprintf(NULL, 0,
                " %s", s->name);
            H_ov_buf_printf(" %s", s->name);
            H_ov_buf_reset_attr();
            H_ov_theme_bg(OV_BG_PANEL);
            H_render_pad_spaces(n, r.width);
            if (!skip_draw) { ri++; }
            line_idx++;
        }
        /* Type + Size */
        if (1)
        {
            char szb[48];
            if (s->naxis == 1)
            {
                snprintf(szb, sizeof(szb),
                    "%u", (unsigned) s->size[0]);
            }
            else if (s->naxis == 2)
            {
                snprintf(szb, sizeof(szb),
                    "%ux%u",
                    (unsigned) s->size[0],
                    (unsigned) s->size[1]);
            }
            else
            {
                snprintf(szb, sizeof(szb),
                    "%ux%ux%u",
                    (unsigned) s->size[0],
                    (unsigned) s->size[1],
                    (unsigned) s->size[2]);
            }
            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_bg(OV_BG_PANEL);
            H_ov_theme_fg(OV_FG_DIM);
            int n = snprintf(NULL, 0,
                " Type: %s  Size: %s"
                "  Elements: %" PRIu64,
                render_dtype(s->datatype), szb,
                (uint64_t) s->nelement);
            H_ov_buf_printf(
                " Type: %s  Size: %s"
                "  Elements: %" PRIu64,
                render_dtype(s->datatype), szb,
                (uint64_t) s->nelement);
            H_render_pad_spaces(n, r.width);
            if (!skip_draw) { ri++; }
            line_idx++;
        }
        /* Counters */
        if (1)
        {
            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_bg(OV_BG_PANEL);
            H_ov_theme_fg(OV_FG_DIM);
            H_ov_buf_printf(" cnt0: ");
            H_ov_theme_fg(s->cnt_active
                ? OV_FG_ACTIVE : OV_FG_DIM);
            H_ov_buf_printf("%" PRIu64, (uint64_t) s->cnt0);
            H_ov_theme_fg(OV_FG_DIM);
            H_ov_buf_printf("  Hz: ");
            H_ov_theme_fg(s->cnt_active
                ? OV_FG_ACTIVE : OV_FG_DIM);
            int n = snprintf(NULL, 0,
                " cnt0: %" PRIu64 "  Hz: %.1f",
                (uint64_t) s->cnt0,
                s->update_hz);
            H_ov_buf_printf("%.1f", s->update_hz);
            H_render_pad_spaces(n, r.width);
            if (!skip_draw) { ri++; }
            line_idx++;
        }
        /* PIDs */
        if (1)
        {
            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_bg(OV_BG_PANEL);
            H_ov_theme_fg(OV_FG_DIM);
            int n = snprintf(NULL, 0,
                " Creator: %d  Owner: %d"
                "  Inode: %" PRIu64,
                (int) s->creatorPID,
                (int) s->ownerPID,
                (uint64_t) s->inode);
            H_ov_buf_printf(
                " Creator: %d  Owner: %d"
                "  Inode: %" PRIu64,
                (int) s->creatorPID,
                (int) s->ownerPID,
                (uint64_t) s->inode);
            H_render_pad_spaces(n, r.width);
            if (!skip_draw) { ri++; }
            line_idx++;
        }
        /* Semaphores */
        if (s->nb_sem > 0)
        {
            if (1)
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
                if (!skip_draw) { ri++; }
            line_idx++;
            }
            for (int i = 0; i < s->nb_sem; i++)
            {
                H_ov_buf_pos(row + ri, r.col + 1);
                H_ov_theme_bg(OV_BG_PANEL);
                
                int val = s->semval[i];
                if (val > 0)
                {
                    H_ov_theme_fg(OV_FG_WARN);
                }
                else
                {
                    H_ov_theme_fg(OV_FG_TEXT);
                }
                
                char rpid_str[32] = "";
                if (s->read_pids[i] > 0) {
                    snprintf(rpid_str, sizeof(rpid_str), "reader:%d", (int)s->read_pids[i]);
                }
                
                int n = snprintf(NULL, 0,
                    "  [%d] val:%d  %s",
                    i, val, rpid_str);
                H_ov_buf_printf(
                    "  [%d] val:%d  %s",
                    i, val, rpid_str);
                H_render_pad_spaces(n, r.width);
                if (!skip_draw) { ri++; }
            line_idx++;
            }
        }
        /* Proctrace entries */
        if (s->nb_proctrace > 0)
        {
            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_bg(OV_BG_PANEL);
            H_ov_theme_fg(OV_FG_TITLE);
            H_ov_buf_bold();
            int n = snprintf(NULL, 0,
                " Process trace (%d):",
                s->nb_proctrace);
            H_ov_buf_printf(
                " Process trace (%d):",
                s->nb_proctrace);
            H_ov_buf_reset_attr();
            H_ov_theme_bg(OV_BG_PANEL);
            H_render_pad_spaces(n, r.width);
            if (!skip_draw) { ri++; }
            line_idx++;

            for (int t = 0;
                 t < s->nb_proctrace
                ; t++)
            {
                /* Find proc name by PID */
                const char *pname = "???";
                for (int pp = 0;
                     pp < m->nb_procs; pp++)
                {
                    if (m->procs[pp].PID
                        == s->proctrace_pid[t])
                    {
                        pname = m->procs[pp].name;
                        break;
                    }
                }
                H_ov_buf_pos(
                    row + ri, r.col + 1);
                H_ov_theme_bg(OV_BG_PANEL);
                H_ov_theme_fg(
                    ov_pid_color(
                        s->proctrace_pid[t]));
                int n2 = snprintf(NULL, 0,
                    "  PID %d (%s)"
                    "  mode:%s",
                    (int) s->proctrace_pid[t],
                    pname,
                    render_trigmode_label(
                        s->proctrace_trigmode[t]));
                H_ov_buf_printf(
                    "  PID %d (%s)"
                    "  mode:%s",
                    (int) s->proctrace_pid[t],
                    pname,
                    render_trigmode_label(
                        s->proctrace_trigmode[t]));
                H_render_pad_spaces(
                    n2, r.width);
                if (!skip_draw) { ri++; }
            line_idx++;
            }
        }
        /* ---- Stream lineage ---- */
        {
            SG_LINEAGE lin;
            sg_compute_lineage(
                m, ssel,
                (sg_mode_t) lay->lineage_mode,
                &lin);

            int has_lineage =
                (lin.nb_ancestors > 0
                 || lin.nb_descendants > 0);

            if (has_lineage)
            {
                /* blank separator line */
                clear_row(
                    row + ri, r.col + 1,
                    r.width - 2, OV_BG_PANEL);
                if (!skip_draw) { ri++; }
            line_idx++;
            }

            /* Ancestors (upstream) */
            if (lin.nb_ancestors > 0
               )
            {
                H_ov_buf_pos(row + ri, r.col + 1);
                H_ov_theme_bg(OV_BG_PANEL);
                H_ov_theme_fg(OV_FG_TITLE);
                H_ov_buf_bold();
                const char *ml = sg_mode_label(
                    (sg_mode_t) lay->lineage_mode);
                int printed = snprintf(
                    NULL, 0,
                    " Ancestors [%s] (%d):",
                    ml, lin.nb_ancestors);
                H_ov_buf_printf(
                    " Ancestors [%s] (%d):",
                    ml, lin.nb_ancestors);
                H_ov_buf_reset_attr();
                H_ov_theme_bg(OV_BG_PANEL);

                for (int a = 0; a < lin.nb_ancestors; a++)
                {
                    const SG_LINEAGE_ENTRY *le = &lin.ancestors[a];
                    const char *sn = m->streams[le->stream_idx].name;

                    int item_len = snprintf(NULL, 0, "  -%d %s", le->depth, sn);
                    if (le->via_name[0] != '\0')
                    {
                        item_len += snprintf(NULL, 0, "(%s)", le->via_name);
                    }

                    if (printed + item_len >= r.width - 2)
                    {
                        H_render_pad_spaces(printed, r.width);
                        if (!skip_draw) { ri++; }
            line_idx++;
                        if (ri >= max_rows) break;
                        H_ov_buf_pos(row + ri, r.col + 1);
                        H_ov_theme_bg(OV_BG_PANEL);
                        printed = 0;
                        item_len = snprintf(NULL, 0, " -%d %s", le->depth, sn);
                        if (le->via_name[0] != '\0') {
                            item_len += snprintf(NULL, 0, "(%s)", le->via_name);
                        }
                    }

                    if (le->depth == 1)
                        H_ov_theme_fg(OV_FG_STREAM);
                    else
                        H_ov_theme_fg(OV_FG_DIM);

                    int p1 = snprintf(NULL, 0, "  -%d %s", le->depth, sn);
                    if (printed == 0) p1 = snprintf(NULL, 0, " -%d %s", le->depth, sn);
                    
                    if (printed == 0) H_ov_buf_printf(" -%d %s", le->depth, sn);
                    else H_ov_buf_printf("  -%d %s", le->depth, sn);
                    printed += p1;

                    if (le->via_name[0] != '\0')
                    {
                        H_ov_theme_fg(OV_FG_PROC);
                        H_ov_buf_printf("(%s)", le->via_name);
                        printed += snprintf(NULL, 0, "(%s)", le->via_name);
                    }
                }
                if (1)
                {
                    H_render_pad_spaces(printed, r.width);
                    if (!skip_draw) { ri++; }
            line_idx++;
                }
            } /* ancestors */

            /* Descendants (downstream) */
            if (lin.nb_descendants > 0
               )
            {
                H_ov_buf_pos(row + ri, r.col + 1);
                H_ov_theme_bg(OV_BG_PANEL);
                H_ov_theme_fg(OV_FG_TITLE);
                H_ov_buf_bold();
                const char *mld = sg_mode_label(
                    (sg_mode_t) lay->lineage_mode);
                int printed = snprintf(
                    NULL, 0,
                    " Descendants [%s] (%d):",
                    mld, lin.nb_descendants);
                H_ov_buf_printf(
                    " Descendants [%s] (%d):",
                    mld, lin.nb_descendants);
                H_ov_buf_reset_attr();
                H_ov_theme_bg(OV_BG_PANEL);

                for (int d = 0; d < lin.nb_descendants; d++)
                {
                    const SG_LINEAGE_ENTRY *le = &lin.descendants[d];
                    const char *sn = m->streams[le->stream_idx].name;

                    int item_len = snprintf(NULL, 0, "  +%d %s", le->depth, sn);
                    if (le->via_name[0] != '\0')
                    {
                        item_len += snprintf(NULL, 0, "(%s)", le->via_name);
                    }

                    if (printed + item_len >= r.width - 2)
                    {
                        H_render_pad_spaces(printed, r.width);
                        if (!skip_draw) { ri++; }
            line_idx++;
                        if (ri >= max_rows) break;
                        H_ov_buf_pos(row + ri, r.col + 1);
                        H_ov_theme_bg(OV_BG_PANEL);
                        printed = 0;
                        item_len = snprintf(NULL, 0, " +%d %s", le->depth, sn);
                        if (le->via_name[0] != '\0') {
                            item_len += snprintf(NULL, 0, "(%s)", le->via_name);
                        }
                    }

                    if (le->depth == 1)
                        H_ov_theme_fg(OV_FG_STREAM);
                    else
                        H_ov_theme_fg(OV_FG_DIM);

                    int p1 = snprintf(NULL, 0, "  +%d %s", le->depth, sn);
                    if (printed == 0) p1 = snprintf(NULL, 0, " +%d %s", le->depth, sn);
                    
                    if (printed == 0) H_ov_buf_printf(" +%d %s", le->depth, sn);
                    else H_ov_buf_printf("  +%d %s", le->depth, sn);
                    printed += p1;

                    if (le->via_name[0] != '\0')
                    {
                        H_ov_theme_fg(OV_FG_PROC);
                        H_ov_buf_printf("(%s)", le->via_name);
                        printed += snprintf(NULL, 0, "(%s)", le->via_name);
                    }
                }
                if (1)
                {
                    H_render_pad_spaces(printed, r.width);
                    if (!skip_draw) { ri++; }
            line_idx++;
                }
            } /* descendants */
        } /* lineage */
        /* Clear remaining rows */
        lay->detail_total_lines = line_idx;
        for (; ri < max_rows; ri++)
        {
            clear_row(
                row + ri, r.col + 1,
                r.width - 2, OV_BG_PANEL);
        }
        H_ov_buf_reset_attr();
        return 1;
    
}

static int ov_fps__render_detail_proc(
    OV_LAYOUT       *lay,
    const OV_MODEL *m,
    int psel,
    OV_RECT r,
    int max_rows,
    int row)
{

        const OV_PROC *p =
            &m->procs[psel];

        const char *tabs[] = {"CONNECTIONS", "DETAILS", "RESOURCES"};
        ov_draw_panel_tabs(
            r.row, r.col, r.height, r.width,
            tabs, 3, lay->graph_tab_mode, OV_FG_PROC, lay->focus == OV_FOCUS_GRAPH);

        int ri = 0;
    int line_idx = 0;

        /* Name + PID */
        if (1)
        {
            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_bg(OV_BG_PANEL);
            H_ov_theme_fg(OV_FG_TITLE);
            H_ov_buf_bold();
            int n = snprintf(NULL, 0,
                " %s  (PID %d)",
                p->name, (int) p->PID);
            H_ov_buf_printf(" %s  (PID %d)",
                p->name, (int) p->PID);
            H_ov_buf_reset_attr();
            H_ov_theme_bg(OV_BG_PANEL);
            H_render_pad_spaces(n, r.width);
            if (!skip_draw) { ri++; }
            line_idx++;
        }
        /* Status + loop info */
        if (1)
        {
            const char *sl;
            switch (p->loopstat)
            {
            case 0:  sl = "IDLE"; break;
            case 1:  sl = "RUNNING"; break;
            case 2:  sl = "PAUSED"; break;
            case 3:  sl = "TERMINATING"; break;
            case 4:  sl = "ERROR"; break;
            default: sl = "UNKNOWN"; break;
            }
            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_bg(OV_BG_PANEL);
            H_ov_theme_fg(OV_FG_TEXT);
            H_ov_buf_printf(" Status: %s  Loops: ", sl);
            H_ov_theme_fg(p->cnt_active
                ? OV_FG_ACTIVE : OV_FG_DIM);
            H_ov_buf_printf("%" PRId64, (int64_t) p->loopcnt);
            H_ov_theme_fg(OV_FG_DIM);
            H_ov_buf_printf("  Hz: ");
            H_ov_theme_fg(p->cnt_active
                ? OV_FG_ACTIVE : OV_FG_DIM);
            H_ov_buf_printf("%.1f", p->loop_hz);
            int n = snprintf(NULL, 0,
                " Status: %s  Loops: %" PRId64
                "  Hz: %.1f",
                sl, (int64_t) p->loopcnt,
                p->loop_hz);
            H_render_pad_spaces(n, r.width);
            if (!skip_draw) { ri++; }
            line_idx++;
        }
        /* CPU and Mem */
        if (1)
        {
            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_bg(OV_BG_PANEL);
            H_ov_theme_fg(OV_FG_TEXT);
            int n = snprintf(NULL, 0,
                " CPU: %5.1f%%  Mem: %" PRId64 " KB",
                p->cpu_used, (int64_t)p->mem_rss_kb);
            H_ov_buf_printf(
                " CPU: %5.1f%%  Mem: %" PRId64 " KB",
                p->cpu_used, (int64_t)p->mem_rss_kb);
            H_render_pad_spaces(n, r.width);
            if (!skip_draw) { ri++; }
            line_idx++;
        }
        /* Trigger info */
        if (1)
        {
            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_bg(OV_BG_PANEL);
            H_ov_theme_fg(OV_FG_CONN);
            int n = snprintf(NULL, 0,
                " Trigger: %s  stream: %s"
                "  sem: %d",
                render_trigmode_label(
                    p->triggermode),
                (p->trigstreamname[0] != '\0')
                    ? p->trigstreamname : "-",
                p->triggersem);
            H_ov_buf_printf(
                " Trigger: %s  stream: %s"
                "  sem: %d",
                render_trigmode_label(
                    p->triggermode),
                (p->trigstreamname[0] != '\0')
                    ? p->trigstreamname : "-",
                p->triggersem);
            H_render_pad_spaces(n, r.width);
            if (!skip_draw) { ri++; }
            line_idx++;
        }
        /* Missed frames */
        if (1)
        {
            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_bg(OV_BG_PANEL);
            if (p->triggermissed > 0)
            {
                H_ov_theme_fg(OV_FG_WARN);
            }
            else
            {
                H_ov_theme_fg(OV_FG_DIM);
            }
            int n = snprintf(NULL, 0,
                " Missed: %d (cumul: %" PRIu64 ")",
                p->triggermissed,
                (uint64_t)
                    p->triggermissed_cumul);
            H_ov_buf_printf(
                " Missed: %d (cumul: %" PRIu64 ")",
                p->triggermissed,
                (uint64_t)
                    p->triggermissed_cumul);
            H_render_pad_spaces(n, r.width);
            if (!skip_draw) { ri++; }
            line_idx++;
        }
        /* Exec time + overhead */
        if (1)
        {
            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_bg(OV_BG_PANEL);
            if (p->MeasureTiming
                && p->dtmedian_exec_ns > 0)
            {
                double exec_ms = 1.0e-6
                    * (double) p->dtmedian_exec_ns;
                double overhead = 0.0;
                if (p->dtmedian_iter_ns > 0)
                {
                    overhead = 100.0
                        * (double) p->dtmedian_exec_ns
                        / (double) p->dtmedian_iter_ns;
                }
                H_ov_theme_fg(OV_FG_TEXT);
                int n = snprintf(NULL, 0,
                    " Exec: %.3f ms  Load: %.1f%%"
                    "  RT: %d",
                    exec_ms, overhead,
                    p->rt_priority);
                H_ov_buf_printf(
                    " Exec: %.3f ms  Load: %.1f%%"
                    "  RT: %d",
                    exec_ms, overhead,
                    p->rt_priority);
                H_render_pad_spaces(n, r.width);
            }
            else
            {
                H_ov_theme_fg(OV_FG_DIM);
                int n = snprintf(NULL, 0,
                    " Timing: disabled  RT: %d",
                    p->rt_priority);
                H_ov_buf_printf(
                    " Timing: disabled  RT: %d",
                    p->rt_priority);
                H_render_pad_spaces(n, r.width);
            }
            if (!skip_draw) { ri++; }
            line_idx++;
        }
        lay->detail_total_lines = line_idx;
        for (; ri < max_rows; ri++)
        {
            clear_row(
                row + ri, r.col + 1,
                r.width - 2, OV_BG_PANEL);
        }
        H_ov_buf_reset_attr();
        return 1;
    
}

static int ov_fps__render_detail_fps(
    OV_LAYOUT       *lay,
    const OV_MODEL *m,
    int fsel,
    OV_RECT r,
    int max_rows,
    int row)
{

        const OV_FPS *f =
            &m->fps[fsel];

        const char *tabs[] = {"CONNECTIONS", "DETAILS", "RESOURCES"};
        ov_draw_panel_tabs(
            r.row, r.col, r.height, r.width,
            tabs, 3, lay->graph_tab_mode, OV_FG_FPS, lay->focus == OV_FOCUS_GRAPH);

        int ri = 0;
    int line_idx = 0;

        /* Name */
        if (1)
        {
            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_bg(OV_BG_PANEL);
            H_ov_theme_fg(OV_FG_TITLE);
            H_ov_buf_bold();
            int n = snprintf(NULL, 0,
                " %s", f->name);
            H_ov_buf_printf(" %s", f->name);
            H_ov_buf_reset_attr();
            H_ov_theme_bg(OV_BG_PANEL);
            H_render_pad_spaces(n, r.width);
            if (!skip_draw) { ri++; }
            line_idx++;
        }
        /* Description */
        if (f->description[0] != '\0')
        {
            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_bg(OV_BG_PANEL);
            H_ov_theme_fg(OV_FG_TEXT);
            int n = snprintf(NULL, 0,
                " %s", f->description);
            H_ov_buf_printf(" %s", f->description);
            H_render_pad_spaces(n, r.width);
            if (!skip_draw) { ri++; }
            line_idx++;
        }
        /* Conf / Run status */
        if (1)
        {
            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_bg(OV_BG_PANEL);
            H_ov_theme_fg(OV_FG_DIM);
            /* PID status labels */
            const char *cst, *rst;
            ov_pid_status_t cs =
                pid_get_status(f->confpid);
            ov_pid_status_t rs =
                pid_get_status(f->runpid);
            cst = (cs == OV_PID_ALIVE) ? "ALIVE"
                : (cs == OV_PID_ZOMBIE) ? "ZOMB"
                : "dead";
            rst = (rs == OV_PID_ALIVE) ? "ALIVE"
                : (rs == OV_PID_ZOMBIE) ? "ZOMB"
                : "dead";
            int n = snprintf(NULL, 0,
                " Conf: %s (PID %d)"
                "  Run: %s (PID %d)",
                cst, (int) f->confpid,
                rst, (int) f->runpid);
            H_ov_buf_printf(
                " Conf: %s (PID %d)"
                "  Run: %s (PID %d)",
                cst, (int) f->confpid,
                rst, (int) f->runpid);
            H_render_pad_spaces(n, r.width);
            if (!skip_draw) { ri++; }
            line_idx++;
        }
        /* Parameters */
        if (f->nb_disp_params > 0)
        {
            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_bg(OV_BG_PANEL);
            H_ov_theme_fg(OV_FG_TITLE);
            H_ov_buf_bold();
            int n = snprintf(NULL, 0,
                " Parameters (%d):",
                f->nb_disp_params);
            H_ov_buf_printf(
                " Parameters (%d):",
                f->nb_disp_params);
            H_ov_buf_reset_attr();
            H_ov_theme_bg(OV_BG_PANEL);
            H_render_pad_spaces(n, r.width);
            if (!skip_draw) { ri++; }
            line_idx++;

            for (int dp = 0; dp < f->nb_disp_params; dp++)
            {
                H_ov_buf_pos(row + ri, r.col + 1);
                H_ov_theme_bg(OV_BG_PANEL);
                H_ov_theme_fg(OV_FG_DIM);
                H_ov_buf_printf("  ");
                H_ov_theme_fg(OV_FG_CONN);
                H_ov_buf_printf("%-24.24s", f->disp_param_name[dp]);
                H_ov_theme_fg(OV_FG_TEXT);
                int n2 = snprintf(NULL, 0, " = %s", f->disp_param_value[dp]);
                H_ov_buf_printf(" = %s", f->disp_param_value[dp]);
                H_render_pad_spaces(2 + 24 + n2, r.width);
                if (!skip_draw) { ri++; }
            line_idx++;
            }
        }
        lay->detail_total_lines = line_idx;
        for (; ri < max_rows; ri++)
        {
            clear_row(
                row + ri, r.col + 1,
                r.width - 2, OV_BG_PANEL);
        }
        H_ov_buf_reset_attr();
        return 1;
    
}

int ov_render_detail_panel(
    OV_LAYOUT       *lay,
    const OV_MODEL  *m)
{
    OV_RECT r = lay->r_graph;
    int max_rows = r.height - 2;
    int row = r.row + 1;

    ov_focus_t focus = lay->freeze ? lay->freeze_focus : lay->focus;
    int ssel = lay->freeze ? lay->freeze_sel_stream : lay->sel_stream;
    int psel = lay->freeze ? lay->freeze_sel_proc   : lay->sel_proc;
    int fsel = lay->freeze ? lay->freeze_sel_fps    : lay->sel_fps;

    if (focus == OV_FOCUS_STREAMS && ssel >= 0 && ssel < m->nb_streams)
    {
        return ov_fps__render_detail_stream(lay, m, ssel, r, max_rows, row);
    }
    if (focus == OV_FOCUS_PROCS && psel >= 0 && psel < m->nb_procs)
    {
        return ov_fps__render_detail_proc(lay, m, psel, r, max_rows, row);
    }
    if (focus == OV_FOCUS_FPS && fsel >= 0 && fsel < m->nb_fps)
    {
        return ov_fps__render_detail_fps(lay, m, fsel, r, max_rows, row);
    }

    return 0;
}

int ov_render_resources_panel(
    const OV_LAYOUT *lay,
    const OV_MODEL  *m)
{
    OV_RECT r = lay->r_graph;
    int max_rows = r.height - 2;
    int row = r.row + 1;

    ov_focus_t focus = lay->freeze ? lay->freeze_focus : lay->focus;
    int ssel = lay->freeze ? lay->freeze_sel_stream : lay->sel_stream;
    int psel = lay->freeze ? lay->freeze_sel_proc   : lay->sel_proc;
    int fsel = lay->freeze ? lay->freeze_sel_fps    : lay->sel_fps;

    pid_t target_pid = 0;
    const char *target_name = "UNKNOWN";
    ov_rgb_t target_color = OV_FG_DIM;

    if (focus == OV_FOCUS_STREAMS && ssel >= 0 && ssel < m->nb_streams)
    {
        target_pid = m->streams[ssel].ownerPID;
        target_name = m->streams[ssel].name;
        target_color = OV_FG_STREAM;
    }
    else if (focus == OV_FOCUS_PROCS && psel >= 0 && psel < m->nb_procs)
    {
        target_pid = m->procs[psel].PID;
        target_name = m->procs[psel].name;
        target_color = OV_FG_PROC;
    }
    else if (focus == OV_FOCUS_FPS && fsel >= 0 && fsel < m->nb_fps)
    {
        target_pid = m->fps[fsel].runpid;
        if (target_pid == 0) target_pid = m->fps[fsel].confpid;
        target_name = m->fps[fsel].name;
        target_color = OV_FG_FPS;
    }

    const char *tabs[] = {"CONNECTIONS", "DETAILS", "RESOURCES"};
    ov_draw_panel_tabs(
        r.row, r.col, r.height, r.width,
        tabs, 3, lay->graph_tab_mode, target_color, lay->focus == OV_FOCUS_GRAPH);

    int ri = 0;
    int line_idx = 0;

    if (target_pid <= 0)
    {
        H_ov_buf_pos(row + ri, r.col + 1);
        H_ov_theme_bg(OV_BG_PANEL);
        H_ov_theme_fg(OV_FG_DIM);
        H_ov_buf_printf(" No active process for %s", target_name);
        H_render_pad_spaces(25 + strlen(target_name), r.width);
        if (!skip_draw) { ri++; }
            line_idx++;
    }
    else
    {
        /* Query core utilization */
        int active_cores[128];
        int num_active = pid_get_core_utilization(target_pid, active_cores, 128);
        uint64_t core_mask = 0;
        for (int i = 0; i < num_active; i++) {
            if (active_cores[i] >= 0 && active_cores[i] < 64) {
                core_mask |= (1ULL << active_cores[i]);
            }
        }
        char stat_path[256];
        snprintf(stat_path, sizeof(stat_path), "/proc/%d/statm", (int)target_pid);
        FILE *f = fopen(stat_path, "r");
        uint64_t vm_size = 0, vm_rss = 0;
        if (f) {
            if (fscanf(f, "%" SCNu64 " %" SCNu64, &vm_size, &vm_rss) == 2) {
                // scale to MB (assuming 4KB pages)
                vm_size = (vm_size * 4) / 1024;
                vm_rss = (vm_rss * 4) / 1024;
            }
            fclose(f);
        }

        ov_advanced_stats_t adv_stats;
        int has_adv = (pid_get_advanced_stats(target_pid, &adv_stats) == 0);
        
        int64_t target_loopcnt = 0;
        for (int i = 0; i < m->nb_procs; i++) {
            if (m->procs[i].PID == target_pid && m->procs[i].active) {
                target_loopcnt = m->procs[i].loopcnt;
                break;
            }
        }
        
        ov_perf_counters_t perf_cnt;
        int has_perf = (pid_read_perf_counters(target_pid, target_loopcnt, &perf_cnt) == 0);

        H_ov_buf_pos(row + ri, r.col + 1);
        H_ov_theme_bg(OV_BG_PANEL);
        H_ov_theme_fg(OV_FG_TITLE);
        H_ov_buf_bold();
        int n = snprintf(NULL, 0, " %s  (PID %d)", target_name, (int)target_pid);
        H_ov_buf_printf(" %s  (PID %d)", target_name, (int)target_pid);
        H_ov_buf_reset_attr();
        H_ov_theme_bg(OV_BG_PANEL);
        H_render_pad_spaces(n, r.width);
        if (!skip_draw) { ri++; }
            line_idx++;
        
        if (1)
        {
            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_fg(OV_FG_DIM);
            H_ov_buf_printf(" Memory Usage:");
            H_render_pad_spaces(14, r.width);
            if (!skip_draw) { ri++; }
            line_idx++;
        }
        
        if (1)
        {
            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_fg(OV_FG_TEXT);
            char buf[64];
            int nb = snprintf(buf, sizeof(buf), "   RSS: %4" PRIu64 " MB   VIRT: %4" PRIu64 " MB", vm_rss, vm_size);
            H_ov_buf_printf("%s", buf);
            H_render_pad_spaces(nb, r.width);
            if (!skip_draw) { ri++; }
            line_idx++;
        }
        
        if (1)
        {
            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_fg(OV_FG_DIM);
            H_ov_buf_printf(" CPU Core Activity:");
            H_render_pad_spaces(19, r.width);
            if (!skip_draw) { ri++; }
            line_idx++;
        }
        
        /* Draw core mask */
        if (1)
        {
            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_fg(OV_FG_TEXT);
            H_ov_buf_printf("   [");
            int chars_written = 4;
            long num_cores = sysconf(_SC_NPROCESSORS_ONLN);
            if (num_cores <= 0 || num_cores > 64) num_cores = 64;
            
            for (int i = 0; i < num_cores; i++) {
                if (core_mask & (1ULL << i)) {
                    H_ov_theme_fg(OV_FG_PROC);
                    H_ov_buf_printf("■");
                } else {
                    H_ov_theme_fg(OV_FG_DIM);
                    H_ov_buf_printf("-");
                }
                chars_written++;
            }
            H_ov_theme_fg(OV_FG_TEXT);
            H_ov_buf_printf("]");
            chars_written++;
            
            H_render_pad_spaces(chars_written, r.width);
            if (!skip_draw) { ri++; }
            line_idx++;
        }
        
        if (has_adv) {
            if (1) {
                H_ov_buf_pos(row + ri, r.col + 1);
                H_ov_theme_fg(OV_FG_DIM);
                H_render_pad_spaces(0, r.width);
                if (!skip_draw) { ri++; }
            line_idx++;
            }
            if (1) {
                H_ov_buf_pos(row + ri, r.col + 1);
                H_ov_theme_fg(OV_FG_DIM);
                H_ov_buf_printf(" Scheduling & Memory:");
                H_render_pad_spaces(21, r.width);
                if (!skip_draw) { ri++; }
            line_idx++;
            }
            if (1) {
                H_ov_buf_pos(row + ri, r.col + 1);
                H_ov_theme_fg(OV_FG_TEXT);
                char buf[128];
                int nb = snprintf(buf, sizeof(buf), "   Threads: %4" PRIu64 "    Migrations: %4" PRIu64, adv_stats.threads, adv_stats.migrations);
                H_ov_buf_printf("%s", buf);
                H_render_pad_spaces(nb, r.width);
                if (!skip_draw) { ri++; }
            line_idx++;
            }
            if (1) {
                H_ov_buf_pos(row + ri, r.col + 1);
                H_ov_theme_fg(OV_FG_TEXT);
                char buf[128];
                int nb = snprintf(buf, sizeof(buf), "   Ctx Sw:  %4" PRIu64 " (Vol) / %4" PRIu64 " (Invol)", adv_stats.vol_ctxt, adv_stats.nonvol_ctxt);
                H_ov_buf_printf("%s", buf);
                H_render_pad_spaces(nb, r.width);
                if (!skip_draw) { ri++; }
            line_idx++;
            }
            if (1) {
                H_ov_buf_pos(row + ri, r.col + 1);
                H_ov_theme_fg(OV_FG_TEXT);
                char buf[128];
                int nb = snprintf(buf, sizeof(buf), "   Faults:  %4" PRIu64 " (Min) / %4" PRIu64 " (Maj)", adv_stats.minflt, adv_stats.majflt);
                H_ov_buf_printf("%s", buf);
                H_render_pad_spaces(nb, r.width);
                if (!skip_draw) { ri++; }
            line_idx++;
            }
        }
        
        if (1) {
            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_fg(OV_FG_DIM);
            H_render_pad_spaces(0, r.width);
            if (!skip_draw) { ri++; }
            line_idx++;
        }
        
        if (1) {
            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_fg(OV_FG_DIM);
            H_ov_buf_printf(" Hardware Counters:");
            H_render_pad_spaces(19, r.width);
            if (!skip_draw) { ri++; }
            line_idx++;
        }
        
        if (1) {
            H_ov_buf_pos(row + ri, r.col + 1);
            if (has_perf) {
                H_ov_theme_fg(OV_FG_TEXT);
                char buf[128];
                int nb = snprintf(buf, sizeof(buf), "   Inst: %8" PRIu64 "    Cache Miss: %8" PRIu64 "", 
                    (uint64_t)perf_cnt.instructions, 
                    (uint64_t)perf_cnt.cache_misses);
                H_ov_buf_printf("%s", buf);
                H_render_pad_spaces(nb, r.width);
                if (!skip_draw) { ri++; }
            line_idx++;
                
                if (target_loopcnt > 0) {
                    if (1) {
                        H_ov_buf_pos(row + ri, r.col + 1);
                        nb = snprintf(buf, sizeof(buf), "   Inst/Iter:  %.1f    Cache Miss/Iter: %.1f", 
                            perf_cnt.inst_per_loop, perf_cnt.cache_miss_per_loop);
                        H_ov_buf_printf("%s", buf);
                        H_render_pad_spaces(nb, r.width);
                        if (!skip_draw) { ri++; }
            line_idx++;
                    }
                    if (1) {
                        H_ov_buf_pos(row + ri, r.col + 1);
                        H_ov_theme_fg(OV_FG_DIM);
                        nb = snprintf(buf, sizeof(buf), "   Miss/Iter Breakdown:");
                        H_ov_buf_printf("%s", buf);
                        H_render_pad_spaces(nb, r.width);
                        if (!skip_draw) { ri++; }
            line_idx++;
                    }
                    if (1) {
                        H_ov_buf_pos(row + ri, r.col + 1);
                        H_ov_theme_fg(OV_FG_TEXT);
                        nb = snprintf(buf, sizeof(buf), "     L1D: %.1f   LLC: %.1f   dTLB: %.1f   Branch: %.1f", 
                            perf_cnt.l1d_miss_per_loop, perf_cnt.llc_miss_per_loop, perf_cnt.dtlb_miss_per_loop, perf_cnt.branch_miss_per_loop);
                        H_ov_buf_printf("%s", buf);
                        H_render_pad_spaces(nb, r.width);
                        if (!skip_draw) { ri++; }
            line_idx++;
                    }
                }
            } else {
                H_ov_theme_fg(OV_FG_WARN);
                H_ov_buf_printf("   [Requires Privileges / CAP_PERFMON]");
                H_render_pad_spaces(38, r.width);
                if (!skip_draw) { ri++; }
            line_idx++;
                if (1) {
                    H_ov_buf_pos(row + ri, r.col + 1);
                    H_ov_buf_printf("   [Run: milk-setup-caps]");
                    H_render_pad_spaces(25, r.width);
                    if (!skip_draw) { ri++; }
            line_idx++;
                }
            }
        }
    }

    for (; ri < max_rows; ri++)
    {
        clear_row(row + ri, r.col + 1, r.width - 2, OV_BG_PANEL);
    }
    H_ov_buf_reset_attr();
    return 1;
}
