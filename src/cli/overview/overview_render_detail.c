#include "overview_render_internal.h"
static int ov_fps__render_detail_stream(
    const OV_LAYOUT *lay,
    const OV_MODEL *m,
    int ssel,
    OV_RECT r,
    int max_rows,
    int row)
{

        const OV_STREAM *s =
            &m->streams[ssel];

        ov_draw_panel_border(
            r.row, r.col, r.height, r.width,
            "STREAM DETAIL", OV_FG_STREAM, 0);

        int ri = 0;

        /* Name */
        if (ri < max_rows)
        {
            ov_buf_pos(row + ri, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_TITLE);
            ov_buf_bold();
            int n = snprintf(NULL, 0,
                " %s", s->name);
            ov_buf_printf(" %s", s->name);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
            render_pad_spaces(n, r.width);
            ri++;
        }
        /* Type + Size */
        if (ri < max_rows)
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
            ov_buf_pos(row + ri, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_DIM);
            int n = snprintf(NULL, 0,
                " Type: %s  Size: %s"
                "  Elements: %lu",
                render_dtype(s->datatype), szb,
                (unsigned long) s->nelement);
            ov_buf_printf(
                " Type: %s  Size: %s"
                "  Elements: %lu",
                render_dtype(s->datatype), szb,
                (unsigned long) s->nelement);
            render_pad_spaces(n, r.width);
            ri++;
        }
        /* Counters */
        if (ri < max_rows)
        {
            ov_buf_pos(row + ri, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf(" cnt0: ");
            ov_theme_fg(s->cnt_active
                ? OV_FG_ACTIVE : OV_FG_DIM);
            ov_buf_printf("%lu", (unsigned long) s->cnt0);
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("  Hz: ");
            ov_theme_fg(s->cnt_active
                ? OV_FG_ACTIVE : OV_FG_DIM);
            int n = snprintf(NULL, 0,
                " cnt0: %lu  Hz: %.1f",
                (unsigned long) s->cnt0,
                s->update_hz);
            ov_buf_printf("%.1f", s->update_hz);
            render_pad_spaces(n, r.width);
            ri++;
        }
        /* PIDs */
        if (ri < max_rows)
        {
            ov_buf_pos(row + ri, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_DIM);
            int n = snprintf(NULL, 0,
                " Creator: %d  Owner: %d"
                "  Inode: %lu",
                (int) s->creatorPID,
                (int) s->ownerPID,
                (unsigned long) s->inode);
            ov_buf_printf(
                " Creator: %d  Owner: %d"
                "  Inode: %lu",
                (int) s->creatorPID,
                (int) s->ownerPID,
                (unsigned long) s->inode);
            render_pad_spaces(n, r.width);
            ri++;
        }
        /* Semaphores */
        if (s->nb_sem > 0)
        {
            if (ri < max_rows)
            {
                ov_buf_pos(row + ri, r.col + 1);
                ov_theme_bg(OV_BG_PANEL);
                ov_theme_fg(OV_FG_TITLE);
                ov_buf_bold();
                int n = snprintf(NULL, 0, " Semaphores (%d):", s->nb_sem);
                ov_buf_printf(" Semaphores (%d):", s->nb_sem);
                ov_buf_reset_attr();
                ov_theme_bg(OV_BG_PANEL);
                render_pad_spaces(n, r.width);
                ri++;
            }
            for (int i = 0; i < s->nb_sem && ri < max_rows; i++)
            {
                ov_buf_pos(row + ri, r.col + 1);
                ov_theme_bg(OV_BG_PANEL);
                
                int val = s->semval[i];
                if (val > 0)
                {
                    ov_theme_fg(OV_FG_WARN);
                }
                else
                {
                    ov_theme_fg(OV_FG_TEXT);
                }
                
                char rpid_str[32] = "";
                if (s->read_pids[i] > 0) {
                    snprintf(rpid_str, sizeof(rpid_str), "reader:%d", (int)s->read_pids[i]);
                }
                
                int n = snprintf(NULL, 0,
                    "  [%d] val:%d  %s",
                    i, val, rpid_str);
                ov_buf_printf(
                    "  [%d] val:%d  %s",
                    i, val, rpid_str);
                render_pad_spaces(n, r.width);
                ri++;
            }
        }
        /* Proctrace entries */
        if (ri < max_rows && s->nb_proctrace > 0)
        {
            ov_buf_pos(row + ri, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_TITLE);
            ov_buf_bold();
            int n = snprintf(NULL, 0,
                " Process trace (%d):",
                s->nb_proctrace);
            ov_buf_printf(
                " Process trace (%d):",
                s->nb_proctrace);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
            render_pad_spaces(n, r.width);
            ri++;

            for (int t = 0;
                 t < s->nb_proctrace
                 && ri < max_rows; t++)
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
                ov_buf_pos(
                    row + ri, r.col + 1);
                ov_theme_bg(OV_BG_PANEL);
                ov_theme_fg(
                    ov_pid_color(
                        s->proctrace_pid[t]));
                int n2 = snprintf(NULL, 0,
                    "  PID %d (%s)"
                    "  mode:%s",
                    (int) s->proctrace_pid[t],
                    pname,
                    render_trigmode_label(
                        s->proctrace_trigmode[t]));
                ov_buf_printf(
                    "  PID %d (%s)"
                    "  mode:%s",
                    (int) s->proctrace_pid[t],
                    pname,
                    render_trigmode_label(
                        s->proctrace_trigmode[t]));
                render_pad_spaces(
                    n2, r.width);
                ri++;
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

            if (has_lineage && ri < max_rows)
            {
                /* blank separator line */
                clear_row(
                    row + ri, r.col + 1,
                    r.width - 2, OV_BG_PANEL);
                ri++;
            }

            /* Ancestors (upstream) */
            if (lin.nb_ancestors > 0
                && ri < max_rows)
            {
                ov_buf_pos(row + ri, r.col + 1);
                ov_theme_bg(OV_BG_PANEL);
                ov_theme_fg(OV_FG_TITLE);
                ov_buf_bold();
                const char *ml = sg_mode_label(
                    (sg_mode_t) lay->lineage_mode);
                int printed = snprintf(
                    NULL, 0,
                    " Ancestors [%s] (%d):",
                    ml, lin.nb_ancestors);
                ov_buf_printf(
                    " Ancestors [%s] (%d):",
                    ml, lin.nb_ancestors);
                ov_buf_reset_attr();
                ov_theme_bg(OV_BG_PANEL);

                for (int a = 0; a < lin.nb_ancestors && ri < max_rows; a++)
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
                        render_pad_spaces(printed, r.width);
                        ri++;
                        if (ri >= max_rows) break;
                        ov_buf_pos(row + ri, r.col + 1);
                        ov_theme_bg(OV_BG_PANEL);
                        printed = 0;
                        item_len = snprintf(NULL, 0, " -%d %s", le->depth, sn);
                        if (le->via_name[0] != '\0') {
                            item_len += snprintf(NULL, 0, "(%s)", le->via_name);
                        }
                    }

                    if (le->depth == 1)
                        ov_theme_fg(OV_FG_STREAM);
                    else
                        ov_theme_fg(OV_FG_DIM);

                    int p1 = snprintf(NULL, 0, "  -%d %s", le->depth, sn);
                    if (printed == 0) p1 = snprintf(NULL, 0, " -%d %s", le->depth, sn);
                    
                    if (printed == 0) ov_buf_printf(" -%d %s", le->depth, sn);
                    else ov_buf_printf("  -%d %s", le->depth, sn);
                    printed += p1;

                    if (le->via_name[0] != '\0')
                    {
                        ov_theme_fg(OV_FG_PROC);
                        ov_buf_printf("(%s)", le->via_name);
                        printed += snprintf(NULL, 0, "(%s)", le->via_name);
                    }
                }
                if (ri < max_rows)
                {
                    render_pad_spaces(printed, r.width);
                    ri++;
                }
            } /* ancestors */

            /* Descendants (downstream) */
            if (lin.nb_descendants > 0
                && ri < max_rows)
            {
                ov_buf_pos(row + ri, r.col + 1);
                ov_theme_bg(OV_BG_PANEL);
                ov_theme_fg(OV_FG_TITLE);
                ov_buf_bold();
                const char *mld = sg_mode_label(
                    (sg_mode_t) lay->lineage_mode);
                int printed = snprintf(
                    NULL, 0,
                    " Descendants [%s] (%d):",
                    mld, lin.nb_descendants);
                ov_buf_printf(
                    " Descendants [%s] (%d):",
                    mld, lin.nb_descendants);
                ov_buf_reset_attr();
                ov_theme_bg(OV_BG_PANEL);

                for (int d = 0; d < lin.nb_descendants && ri < max_rows; d++)
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
                        render_pad_spaces(printed, r.width);
                        ri++;
                        if (ri >= max_rows) break;
                        ov_buf_pos(row + ri, r.col + 1);
                        ov_theme_bg(OV_BG_PANEL);
                        printed = 0;
                        item_len = snprintf(NULL, 0, " +%d %s", le->depth, sn);
                        if (le->via_name[0] != '\0') {
                            item_len += snprintf(NULL, 0, "(%s)", le->via_name);
                        }
                    }

                    if (le->depth == 1)
                        ov_theme_fg(OV_FG_STREAM);
                    else
                        ov_theme_fg(OV_FG_DIM);

                    int p1 = snprintf(NULL, 0, "  +%d %s", le->depth, sn);
                    if (printed == 0) p1 = snprintf(NULL, 0, " +%d %s", le->depth, sn);
                    
                    if (printed == 0) ov_buf_printf(" +%d %s", le->depth, sn);
                    else ov_buf_printf("  +%d %s", le->depth, sn);
                    printed += p1;

                    if (le->via_name[0] != '\0')
                    {
                        ov_theme_fg(OV_FG_PROC);
                        ov_buf_printf("(%s)", le->via_name);
                        printed += snprintf(NULL, 0, "(%s)", le->via_name);
                    }
                }
                if (ri < max_rows)
                {
                    render_pad_spaces(printed, r.width);
                    ri++;
                }
            } /* descendants */
        } /* lineage */
        /* Clear remaining rows */
        for (; ri < max_rows; ri++)
        {
            clear_row(
                row + ri, r.col + 1,
                r.width - 2, OV_BG_PANEL);
        }
        ov_buf_reset_attr();
        return 1;
    
}

static int ov_fps__render_detail_proc(
    const OV_LAYOUT *lay,
    const OV_MODEL *m,
    int psel,
    OV_RECT r,
    int max_rows,
    int row)
{

        const OV_PROC *p =
            &m->procs[psel];

        ov_draw_panel_border(
            r.row, r.col, r.height, r.width,
            "PROCESS DETAIL", OV_FG_PROC, 0);

        int ri = 0;

        /* Name + PID */
        if (ri < max_rows)
        {
            ov_buf_pos(row + ri, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_TITLE);
            ov_buf_bold();
            int n = snprintf(NULL, 0,
                " %s  (PID %d)",
                p->name, (int) p->PID);
            ov_buf_printf(" %s  (PID %d)",
                p->name, (int) p->PID);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
            render_pad_spaces(n, r.width);
            ri++;
        }
        /* Status + loop info */
        if (ri < max_rows)
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
            ov_buf_pos(row + ri, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_TEXT);
            ov_buf_printf(" Status: %s  Loops: ", sl);
            ov_theme_fg(p->cnt_active
                ? OV_FG_ACTIVE : OV_FG_DIM);
            ov_buf_printf("%ld", (long) p->loopcnt);
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("  Hz: ");
            ov_theme_fg(p->cnt_active
                ? OV_FG_ACTIVE : OV_FG_DIM);
            ov_buf_printf("%.1f", p->loop_hz);
            int n = snprintf(NULL, 0,
                " Status: %s  Loops: %ld"
                "  Hz: %.1f",
                sl, (long) p->loopcnt,
                p->loop_hz);
            render_pad_spaces(n, r.width);
            ri++;
        }
        /* CPU and Mem */
        if (ri < max_rows)
        {
            ov_buf_pos(row + ri, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_TEXT);
            int n = snprintf(NULL, 0,
                " CPU: %5.1f%%  Mem: %ld KB",
                p->cpu_used, p->mem_rss_kb);
            ov_buf_printf(
                " CPU: %5.1f%%  Mem: %ld KB",
                p->cpu_used, p->mem_rss_kb);
            render_pad_spaces(n, r.width);
            ri++;
        }
        /* Trigger info */
        if (ri < max_rows)
        {
            ov_buf_pos(row + ri, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_CONN);
            int n = snprintf(NULL, 0,
                " Trigger: %s  stream: %s"
                "  sem: %d",
                render_trigmode_label(
                    p->triggermode),
                (p->trigstreamname[0] != '\0')
                    ? p->trigstreamname : "-",
                p->triggersem);
            ov_buf_printf(
                " Trigger: %s  stream: %s"
                "  sem: %d",
                render_trigmode_label(
                    p->triggermode),
                (p->trigstreamname[0] != '\0')
                    ? p->trigstreamname : "-",
                p->triggersem);
            render_pad_spaces(n, r.width);
            ri++;
        }
        /* Missed frames */
        if (ri < max_rows)
        {
            ov_buf_pos(row + ri, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
            if (p->triggermissed > 0)
            {
                ov_theme_fg(OV_FG_WARN);
            }
            else
            {
                ov_theme_fg(OV_FG_DIM);
            }
            int n = snprintf(NULL, 0,
                " Missed: %d (cumul: %lu)",
                p->triggermissed,
                (unsigned long)
                    p->triggermissed_cumul);
            ov_buf_printf(
                " Missed: %d (cumul: %lu)",
                p->triggermissed,
                (unsigned long)
                    p->triggermissed_cumul);
            render_pad_spaces(n, r.width);
            ri++;
        }
        /* Exec time + overhead */
        if (ri < max_rows)
        {
            ov_buf_pos(row + ri, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
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
                ov_theme_fg(OV_FG_TEXT);
                int n = snprintf(NULL, 0,
                    " Exec: %.3f ms  Load: %.1f%%"
                    "  RT: %d",
                    exec_ms, overhead,
                    p->rt_priority);
                ov_buf_printf(
                    " Exec: %.3f ms  Load: %.1f%%"
                    "  RT: %d",
                    exec_ms, overhead,
                    p->rt_priority);
                render_pad_spaces(n, r.width);
            }
            else
            {
                ov_theme_fg(OV_FG_DIM);
                int n = snprintf(NULL, 0,
                    " Timing: disabled  RT: %d",
                    p->rt_priority);
                ov_buf_printf(
                    " Timing: disabled  RT: %d",
                    p->rt_priority);
                render_pad_spaces(n, r.width);
            }
            ri++;
        }
        for (; ri < max_rows; ri++)
        {
            clear_row(
                row + ri, r.col + 1,
                r.width - 2, OV_BG_PANEL);
        }
        ov_buf_reset_attr();
        return 1;
    
}

static int ov_fps__render_detail_fps(
    const OV_LAYOUT *lay,
    const OV_MODEL *m,
    int fsel,
    OV_RECT r,
    int max_rows,
    int row)
{

        const OV_FPS *f =
            &m->fps[fsel];

        ov_draw_panel_border(
            r.row, r.col, r.height, r.width,
            "FPS DETAIL", OV_FG_FPS, 0);

        int ri = 0;

        /* Name */
        if (ri < max_rows)
        {
            ov_buf_pos(row + ri, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_TITLE);
            ov_buf_bold();
            int n = snprintf(NULL, 0,
                " %s", f->name);
            ov_buf_printf(" %s", f->name);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
            render_pad_spaces(n, r.width);
            ri++;
        }
        /* Description */
        if (ri < max_rows
            && f->description[0] != '\0')
        {
            ov_buf_pos(row + ri, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_TEXT);
            int n = snprintf(NULL, 0,
                " %s", f->description);
            ov_buf_printf(" %s", f->description);
            render_pad_spaces(n, r.width);
            ri++;
        }
        /* Conf / Run status */
        if (ri < max_rows)
        {
            ov_buf_pos(row + ri, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_DIM);
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
            ov_buf_printf(
                " Conf: %s (PID %d)"
                "  Run: %s (PID %d)",
                cst, (int) f->confpid,
                rst, (int) f->runpid);
            render_pad_spaces(n, r.width);
            ri++;
        }
        /* Parameters */
        if (ri < max_rows && f->nb_disp_params > 0)
        {
            ov_buf_pos(row + ri, r.col + 1);
            ov_theme_bg(OV_BG_PANEL);
            ov_theme_fg(OV_FG_TITLE);
            ov_buf_bold();
            int n = snprintf(NULL, 0,
                " Parameters (%d):",
                f->nb_disp_params);
            ov_buf_printf(
                " Parameters (%d):",
                f->nb_disp_params);
            ov_buf_reset_attr();
            ov_theme_bg(OV_BG_PANEL);
            render_pad_spaces(n, r.width);
            ri++;

            for (int dp = 0; dp < f->nb_disp_params && ri < max_rows; dp++)
            {
                ov_buf_pos(row + ri, r.col + 1);
                ov_theme_bg(OV_BG_PANEL);
                ov_theme_fg(OV_FG_DIM);
                ov_buf_printf("  ");
                ov_theme_fg(OV_FG_CONN);
                ov_buf_printf("%-24.24s", f->disp_param_name[dp]);
                ov_theme_fg(OV_FG_TEXT);
                int n2 = snprintf(NULL, 0, " = %s", f->disp_param_value[dp]);
                ov_buf_printf(" = %s", f->disp_param_value[dp]);
                render_pad_spaces(2 + 24 + n2, r.width);
                ri++;
            }
        }
        for (; ri < max_rows; ri++)
        {
            clear_row(
                row + ri, r.col + 1,
                r.width - 2, OV_BG_PANEL);
        }
        ov_buf_reset_attr();
        return 1;
    
}

int ov_render_detail_panel(
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
