/**
 * @file    overview_render_internal.h
 * @brief   Shared declarations for split render files
 *
 * Not part of the public API. Shared by
 * overview_render.c, overview_render_streams.c,
 * overview_render_procs.c, overview_render_fps.c.
 */

#ifndef OVERVIEW_RENDER_INTERNAL_H
#define OVERVIEW_RENDER_INTERNAL_H

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <regex.h>
#include <sys/resource.h>
#include <sys/time.h>

#include "stream_graph.h"
#include "overview_defs.h"
#include "overview_ansi.h"
#include "overview_theme.h"
#include "overview_data.h"
#include "overview_layout.h"

/* ---- Bitset helpers for related-item tracking ---- */

#define BITS_PER_WORD 64
#define OV_BSET_WORDS(n) \
    (((n) + BITS_PER_WORD - 1) / BITS_PER_WORD)

#define OV_STREAM_WORDS OV_BSET_WORDS(OV_MAX_STREAMS)
#define OV_FPS_WORDS    OV_BSET_WORDS(OV_MAX_FPS)
#define OV_PROC_WORDS   OV_BSET_WORDS(OV_MAX_PROCS)

typedef struct
{
    uint64_t streams[OV_STREAM_WORDS];
    uint64_t stream_written[OV_STREAM_WORDS];
    uint64_t fps[OV_FPS_WORDS];
    uint64_t fps_writes[OV_FPS_WORDS];
    uint64_t procs[OV_PROC_WORDS];
    uint64_t proc_writes[OV_PROC_WORDS];
    uint32_t fps_param_mask[OV_MAX_FPS];
    pid_t    sel_pid;

    /** Signed BFS depth for each stream:
     *  negative = upstream (ancestor),
     *  positive = downstream (descendant),
     *  0 = not in lineage (or is the root). */
    int8_t   stream_depth[OV_MAX_STREAMS];
} OV_RELATED;

/* ---- Shared render utilities ---- */

void render_pad_spaces(int chars_written, int panel_width);
int ov_render_header_text(const char *text, int hs, int max_vis_width);

void render_scroll_indicators(
    OV_RECT  r,
    int      scroll,
    int      max_rows,
    int      total,
    ov_rgb_t accent);

void clear_row(
    int row,
    int col,
    int width,
    ov_rgb_t bg);

int ov_filter_build(
    const char *filter,
    const char *names[],
    int         count,
    int        *out_idx,
    int         max_out);

void bset(uint64_t *words, int idx);
int  bget(const uint64_t *words, int idx);

const char *render_dtype(uint8_t dt);
int         dtype_bytesize(uint8_t dt);

void ov_compute_related(
    const OV_LAYOUT  *lay,
    const OV_MODEL   *m,
    OV_RELATED       *rel);

void render_highlighted_name(
    const char *name,
    int         max_len,
    regex_t    *re,
    int         has_re,
    ov_rgb_t    normal_fg,
    ov_rgb_t    row_bg);

/* Inline trigger-mode short label */
static inline const char *render_trigmode_label(
    int mode)
{
    switch (mode)
    {
    case 0:  return "IMM";
    case 1:  return "CN0";
    case 2:  return "CN1";
    case 3:  return "SEM";
    case 4:  return "DLY";
    case 5:  return "SMP";
    case 6:  return "CN2";
    default: return " - ";
    }
}

/* Inline memory-size formatter */
static inline void format_mem_kb(
    char *buf,
    size_t sz,
    long kb)
{
    if (kb <= 0) {
        snprintf(buf, sz, "   -");
    } else if (kb >= 1024 * 1024) {
        snprintf(buf, sz, "%4.1fG",
            (double)kb / (1024.0 * 1024.0));
    } else if (kb >= 1024) {
        snprintf(buf, sz, "%4ldM", kb / 1024);
    } else {
        snprintf(buf, sz, "%4ldK", kb);
    }
}

/* Inline sort column label builder */
static inline int sort_col_label(
    char       *buf,
    int         bufsz,
    const char *label,
    int         col_key,
    int         cur_key,
    int         desc,
    int         visual_width)
{
    if (col_key == cur_key)
    {
        snprintf(buf, bufsz, "\x01%s%s\x02",
                 label,
                 desc ? "\xe2\x96\xbc"
                      : "\xe2\x96\xb2");
        return visual_width + 2;
    }
    else
    {
        snprintf(buf, bufsz, "%s", label);
        return visual_width;
    }
}

/* Inline semaphore color helper */
static inline ov_rgb_t ov_get_sem_color(int val)
{
    if (val == 0) {
        return (ov_rgb_t){0, 150, 0};
    }
    if (val >= 10) {
        return (ov_rgb_t){160, 90, 30};
    }
    int r = 100 + (val - 1) * (180 - 100) / 9;
    int g = 120 - (val - 1) * (120 - 40) / 9;
    int b = 30;
    return (ov_rgb_t){r, g, b};
}
/* ---- Panel functions (split files) ---- */

void ov_render_streams_panel(
    const OV_LAYOUT  *lay,
    const OV_MODEL   *m,
    const OV_RELATED *rel);

void ov_render_procs_panel(
    const OV_LAYOUT  *lay,
    const OV_MODEL   *m,
    const OV_RELATED *rel);

void ov_render_fps_panel(
    const OV_LAYOUT  *lay,
    const OV_MODEL   *m,
    const OV_RELATED *rel);

int ov_render_detail_panel(
    const OV_LAYOUT *lay,
    const OV_MODEL  *m);

int ov_render_resources_panel(
    const OV_LAYOUT *lay,
    const OV_MODEL  *m);

void ov_render_graph_panel(
    const OV_LAYOUT *lay,
    const OV_MODEL  *m);

void ov_render_status(
    const OV_LAYOUT *lay,
    const OV_MODEL  *m);

void ov_render_cmdlog(const OV_LAYOUT *lay);

void ov_render_help(const OV_LAYOUT *lay);
void ov_render_preview_line(
    const OV_LAYOUT *lay,
    const OV_MODEL  *m);

/* Help panel utilities */
int ov_help_nb_sections(void);
int ov_help_visible_count(const OV_LAYOUT *lay);
int ov_help_toggle_at(OV_LAYOUT *lay, int vis_row);

extern float ov_scan_get_interval(void);


#endif /* OVERVIEW_RENDER_INTERNAL_H */
