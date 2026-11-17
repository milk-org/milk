// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    tlm_timing.c
 * @brief   Fast fixed-point reader and writer for milk telemetry timing files
 */

#include "COREMOD_memory/tlm_timing.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define TLM_PEEK_CHUNK 4096U

/* ============================================================================================== */
/* Fixed-Point ASCII Scanners                                                                     */
/* ============================================================================================== */

static inline const char *skip_hspace(const char *p, const char *end)
{
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\r'))
    {
        p++;
    }
    return p;
}

static inline int is_digit_ch(char c)
{
    return (c >= '0' && c <= '9');
}

static int parse_i64_tok(const char **pp, const char *end, int64_t *val_out)
{
    const char *p   = skip_hspace(*pp, end);
    int         neg = 0;
    if (p < end && (*p == '-' || *p == '+'))
    {
        neg = (*p == '-');
        p++;
    }
    if (p >= end || !is_digit_ch(*p))
    {
        return -1;
    }
    uint64_t acc = 0;
    while (p < end && is_digit_ch(*p))
    {
        acc = acc * 10ULL + (uint64_t) (*p - '0');
        p++;
    }
    *val_out = neg ? -(int64_t) acc : (int64_t) acc;
    *pp      = p;
    return 0;
}

static int parse_u64_tok(const char **pp, const char *end, uint64_t *val_out)
{
    const char *p = skip_hspace(*pp, end);
    if (p < end && *p == '+')
    {
        p++;
    }
    if (p >= end || !is_digit_ch(*p))
    {
        return -1;
    }
    uint64_t acc = 0;
    while (p < end && is_digit_ch(*p))
    {
        acc = acc * 10ULL + (uint64_t) (*p - '0');
        p++;
    }
    *val_out = acc;
    *pp      = p;
    return 0;
}

static int parse_sec_ns_tok(const char **pp, const char *end, int64_t *ns_out)
{
    const char *p   = skip_hspace(*pp, end);
    int         neg = 0;
    if (p < end && (*p == '-' || *p == '+'))
    {
        neg = (*p == '-');
        p++;
    }
    if (p >= end || (!is_digit_ch(*p) && *p != '.'))
    {
        return -1;
    }
    int64_t sec = 0;
    while (p < end && is_digit_ch(*p))
    {
        sec = sec * 10LL + (int64_t) (*p - '0');
        p++;
    }
    int64_t frac_ns = 0;
    if (p < end && *p == '.')
    {
        p++;
        int64_t scale = 100000000LL;
        while (p < end && is_digit_ch(*p))
        {
            if (scale > 0)
            {
                frac_ns += (int64_t) (*p - '0') * scale;
                scale /= 10;
            }
            p++;
        }
    }
    int64_t total = sec * 1000000000LL + frac_ns;
    *ns_out       = neg ? -total : total;
    *pp           = p;
    return 0;
}

static int skip_tok(const char **pp, const char *end)
{
    const char *p = skip_hspace(*pp, end);
    if (p >= end)
    {
        return -1;
    }
    while (p < end && *p != ' ' && *p != '\t' && *p != '\r')
    {
        p++;
    }
    *pp = p;
    return 0;
}

static int count_line_fields(const char *line, const char *end)
{
    const char *p     = line;
    int         count = 0;
    while (p < end)
    {
        p = skip_hspace(p, end);
        if (p >= end)
        {
            break;
        }
        count++;
        while (p < end && *p != ' ' && *p != '\t' && *p != '\r')
        {
            p++;
        }
    }
    return count;
}

static int is_comment_or_blank(const char *line, const char *end)
{
    const char *p = skip_hspace(line, end);
    return (p >= end || *p == '#');
}

static int parse_row(const char *line, const char *end, int ncols, MILK_TLM_ROW *row_out)
{
    const char *p = line;
    if (ncols == 7)
    {
        int64_t  frame_idx = 0;
        uint64_t main_idx  = 0;
        int64_t  t_log_ns  = 0;
        int64_t  t_acq_ns  = 0;
        uint64_t cnt0      = 0;
        uint64_t cnt1      = 0;
        if (parse_i64_tok(&p, end, &frame_idx) != 0 || parse_u64_tok(&p, end, &main_idx) != 0 ||
            skip_tok(&p, end) != 0 || parse_sec_ns_tok(&p, end, &t_log_ns) != 0 ||
            parse_sec_ns_tok(&p, end, &t_acq_ns) != 0 || parse_u64_tok(&p, end, &cnt0) != 0 ||
            parse_u64_tok(&p, end, &cnt1) != 0)
        {
            return -1;
        }
        p = skip_hspace(p, end);
        if (p < end)
        {
            return -1;
        }
        row_out->frame_idx = frame_idx;
        row_out->main_idx  = main_idx;
        row_out->t_log_ns  = t_log_ns;
        row_out->t_acq_ns  = t_acq_ns;
        row_out->cnt0      = cnt0;
        row_out->cnt1      = cnt1;
        return 0;
    }
    if (ncols == 4)
    {
        int64_t  frame_idx = 0;
        uint64_t cnt0      = 0;
        int64_t  wall_ns   = 0;
        if (parse_i64_tok(&p, end, &frame_idx) != 0 || parse_u64_tok(&p, end, &cnt0) != 0 ||
            parse_sec_ns_tok(&p, end, &wall_ns) != 0 || skip_tok(&p, end) != 0)
        {
            return -1;
        }
        p = skip_hspace(p, end);
        if (p < end)
        {
            return -1;
        }
        row_out->frame_idx = frame_idx;
        row_out->main_idx  = cnt0;
        row_out->t_log_ns  = wall_ns;
        row_out->t_acq_ns  = 0;
        row_out->cnt0      = cnt0;
        row_out->cnt1      = 0;
        return 0;
    }
    return -1;
}

static errno_t read_fd_fallback(int fd, size_t size_hint, char **buf_out, size_t *len_out)
{
    size_t cap = (size_hint > 0) ? size_hint : 4096U;
    char  *buf = (char *) malloc(cap);
    if (buf == NULL)
    {
        return RETURN_FAILURE;
    }
    size_t len = 0;
    for (;;)
    {
        if (len == cap)
        {
            size_t new_cap = cap * 2U;
            char  *nb      = (char *) realloc(buf, new_cap);
            if (nb == NULL)
            {
                free(buf);
                return RETURN_FAILURE;
            }
            buf = nb;
            cap = new_cap;
        }
        ssize_t n = read(fd, buf + len, cap - len);
        if (n < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            free(buf);
            return RETURN_FAILURE;
        }
        if (n == 0)
        {
            break;
        }
        len += (size_t) n;
    }
    *buf_out = buf;
    *len_out = len;
    return RETURN_SUCCESS;
}

static size_t pread_chunk(int fd, char *dst, size_t cap, off_t off)
{
    size_t total = 0;
    while (total < cap)
    {
        ssize_t n = pread(fd, dst + total, cap - total, off + (off_t) total);
        if (n < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            break;
        }
        if (n == 0)
        {
            break;
        }
        total += (size_t) n;
    }
    return total;
}

/* ============================================================================================== */
/* Public API                                                                                     */
/* ============================================================================================== */

errno_t milk_tlm_read_file(const char *path, MILK_TLM_TABLE *out)
{
    if (path == NULL || out == NULL)
    {
        return RETURN_FAILURE;
    }
    memset(out, 0, sizeof(*out));

    int fd = open(path, O_RDONLY);
    if (fd < 0)
    {
        PRINT_ERROR("cannot open timing file \"%s\"", path);
        return RETURN_FAILURE;
    }

    struct stat st;
    if (fstat(fd, &st) != 0)
    {
        close(fd);
        PRINT_ERROR("fstat failed on \"%s\"", path);
        return RETURN_FAILURE;
    }

    size_t      len      = (st.st_size > 0) ? (size_t) st.st_size : 0U;
    const char *data     = NULL;
    char       *heap_buf = NULL;
    int         is_mmap  = 0;

    if (len > 0)
    {
        void *mapped = mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped != MAP_FAILED)
        {
            data    = (const char *) mapped;
            is_mmap = 1;
        }
        else
        {
            if (read_fd_fallback(fd, len, &heap_buf, &len) != RETURN_SUCCESS)
            {
                close(fd);
                PRINT_ERROR("cannot read timing file \"%s\"", path);
                return RETURN_FAILURE;
            }
            data = heap_buf;
        }
    }
    else if (st.st_size == 0)
    {
        if (read_fd_fallback(fd, 0, &heap_buf, &len) != RETURN_SUCCESS)
        {
            close(fd);
            return RETURN_FAILURE;
        }
        data = heap_buf;
    }
    close(fd);

    if (len == 0)
    {
        free(heap_buf);
        return RETURN_SUCCESS;
    }

    size_t est_rows = (len / 80U) + 16U;
    out->rows       = (MILK_TLM_ROW *) malloc(est_rows * sizeof(MILK_TLM_ROW));
    if (out->rows == NULL)
    {
        if (is_mmap)
        {
            munmap((void *) data, len);
        }
        free(heap_buf);
        PRINT_ERROR("out of memory reading \"%s\"", path);
        return RETURN_FAILURE;
    }
    out->cap = est_rows;

    const char *p   = data;
    const char *end = data + len;
    while (p < end)
    {
        const char *nl       = (const char *) memchr(p, '\n', (size_t) (end - p));
        const char *line_end = (nl != NULL) ? nl : end;
        const char *next     = (nl != NULL) ? nl + 1 : end;

        if (!is_comment_or_blank(p, line_end))
        {
            if (out->ncols == 0)
            {
                int nf = count_line_fields(p, line_end);
                if (nf != 7 && nf != 4)
                {
                    milk_tlm_free(out);
                    if (is_mmap)
                    {
                        munmap((void *) data, len);
                    }
                    free(heap_buf);
                    PRINT_ERROR("unsupported timing column count %d in \"%s\"", nf, path);
                    return RETURN_FAILURE;
                }
                out->ncols = nf;
            }
            if (out->nrows == out->cap)
            {
                size_t        new_cap = out->cap * 2U;
                MILK_TLM_ROW *nr =
                    (MILK_TLM_ROW *) realloc(out->rows, new_cap * sizeof(MILK_TLM_ROW));
                if (nr == NULL)
                {
                    milk_tlm_free(out);
                    if (is_mmap)
                    {
                        munmap((void *) data, len);
                    }
                    free(heap_buf);
                    PRINT_ERROR("out of memory growing timing table for \"%s\"", path);
                    return RETURN_FAILURE;
                }
                out->rows = nr;
                out->cap  = new_cap;
            }
            if (parse_row(p, line_end, out->ncols, &out->rows[out->nrows]) != 0)
            {
                milk_tlm_free(out);
                if (is_mmap)
                {
                    munmap((void *) data, len);
                }
                free(heap_buf);
                PRINT_ERROR("malformed timing row in \"%s\"", path);
                return RETURN_FAILURE;
            }
            out->nrows++;
        }
        p = next;
    }

    if (is_mmap)
    {
        munmap((void *) data, len);
    }
    free(heap_buf);
    return RETURN_SUCCESS;
}

errno_t milk_tlm_peek_span(const char   *path,
                           MILK_TLM_ROW *first_out,
                           MILK_TLM_ROW *last_out,
                           size_t       *approx_nrows)
{
    if (path == NULL || first_out == NULL || last_out == NULL)
    {
        return RETURN_FAILURE;
    }
    memset(first_out, 0, sizeof(*first_out));
    memset(last_out, 0, sizeof(*last_out));
    if (approx_nrows != NULL)
    {
        *approx_nrows = 0;
    }

    int fd = open(path, O_RDONLY);
    if (fd < 0)
    {
        return RETURN_FAILURE;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0)
    {
        close(fd);
        return RETURN_FAILURE;
    }
    size_t fsize = (size_t) st.st_size;

    char   head[TLM_PEEK_CHUNK];
    size_t head_n    = pread_chunk(fd, head, (fsize < sizeof(head)) ? fsize : sizeof(head), 0);
    int    ncols     = 0;
    int    got_first = 0;
    size_t hdr_bytes = 0;

    const char *p   = head;
    const char *end = head + head_n;
    while (p < end)
    {
        const char *nl       = (const char *) memchr(p, '\n', (size_t) (end - p));
        const char *line_end = (nl != NULL) ? nl : end;
        if (nl == NULL && head_n < fsize)
        {
            break;
        }
        if (!is_comment_or_blank(p, line_end))
        {
            ncols = count_line_fields(p, line_end);
            if ((ncols == 7 || ncols == 4) && parse_row(p, line_end, ncols, first_out) == 0)
            {
                got_first = 1;
                hdr_bytes = (size_t) (p - head);
            }
            break;
        }
        p = (nl != NULL) ? nl + 1 : end;
    }

    if (!got_first)
    {
        close(fd);
        return RETURN_FAILURE;
    }

    char        tail[TLM_PEEK_CHUNK];
    const char *tbuf  = head;
    size_t      tlen  = head_n;
    int         start_midline = 0;
    if (fsize > sizeof(head))
    {
        off_t off     = (off_t) (fsize - sizeof(tail));
        tlen          = pread_chunk(fd, tail, sizeof(tail), off);
        tbuf          = tail;
        start_midline = 1;
    }
    close(fd);

    const char *tp   = tbuf;
    const char *tend = tbuf + tlen;
    if (start_midline)
    {
        const char *nl = (const char *) memchr(tp, '\n', (size_t) (tend - tp));
        tp             = (nl != NULL) ? (nl + 1) : tend;
    }

    int    got_last   = 0;
    size_t line_bytes = 0;
    size_t line_cnt   = 0;
    while (tp < tend)
    {
        const char *nl       = (const char *) memchr(tp, '\n', (size_t) (tend - tp));
        const char *line_end = (nl != NULL) ? nl : tend;
        if (!is_comment_or_blank(tp, line_end))
        {
            MILK_TLM_ROW row;
            if (parse_row(tp, line_end, ncols, &row) == 0)
            {
                *last_out = row;
                got_last  = 1;
                if (nl != NULL)
                {
                    line_bytes += (size_t) ((nl + 1) - tp);
                    line_cnt++;
                }
            }
        }
        tp = (nl != NULL) ? nl + 1 : tend;
    }

    if (!got_last)
    {
        *last_out = *first_out;
    }

    if (approx_nrows != NULL)
    {
        if (last_out->frame_idx >= first_out->frame_idx && first_out->frame_idx == 0)
        {
            *approx_nrows = (size_t) (last_out->frame_idx + 1);
        }
        else if (line_cnt > 0 && fsize > hdr_bytes)
        {
            double avg    = (double) line_bytes / (double) line_cnt;
            *approx_nrows = (size_t) ((double) (fsize - hdr_bytes) / avg + 0.5);
        }
        else
        {
            *approx_nrows = 1;
        }
    }
    return RETURN_SUCCESS;
}

errno_t milk_tlm_write_file(const char *path, const MILK_TLM_ROW *rows, size_t nrows)
{
    if (path == NULL || (nrows > 0 && rows == NULL))
    {
        return RETURN_FAILURE;
    }
    FILE *fp = fopen(path, "w");
    if (fp == NULL)
    {
        PRINT_ERROR("cannot create timing file \"%s\"", path);
        return RETURN_FAILURE;
    }

    fprintf(fp, "# Telemetry stream timing data \n");
    fprintf(fp, "# File written by function %s in file %s\n", __func__, __FILE__);
    fprintf(fp, "# \n");
    fprintf(fp, "# col1 : datacube frame index\n");
    fprintf(fp, "# col2 : Main index\n");
    fprintf(fp, "# col3 : Time since cube origin (logging)\n");
    fprintf(fp, "# col4 : Absolute time (logging)\n");
    fprintf(fp, "# col5 : Absolute time (acquisition)\n");
    fprintf(fp, "# col6 : stream cnt0 index\n");
    fprintf(fp, "# col7 : stream cnt1 index\n");
    fprintf(fp, "# \n");

    int64_t t0_ns = (nrows > 0) ? rows[0].t_log_ns : 0;
    for (size_t k = 0; k < nrows; k++)
    {
        double  dt     = (double) (rows[k].t_log_ns - t0_ns) * 1.0e-9;
        int64_t lsec   = rows[k].t_log_ns / 1000000000LL;
        int64_t lrem   = rows[k].t_log_ns % 1000000000LL;
        int64_t asec   = rows[k].t_acq_ns / 1000000000LL;
        int64_t arem   = rows[k].t_acq_ns % 1000000000LL;
        if (lrem < 0)
        {
            lsec -= 1;
            lrem += 1000000000LL;
        }
        if (arem < 0)
        {
            asec -= 1;
            arem += 1000000000LL;
        }
        if (rows[k].t_acq_ns == 0)
        {
            fprintf(fp,
                    "%10ld  %10lu  %15.9lf   %10lld.%09lld  %17.6lf   %10lu   %10lu\n",
                    (long) rows[k].frame_idx, (unsigned long) rows[k].main_idx, dt,
                    (long long) lsec, (long long) lrem, 0.0, (unsigned long) rows[k].cnt0,
                    (unsigned long) rows[k].cnt1);
        }
        else
        {
            fprintf(fp,
                    "%10ld  %10lu  %15.9lf   %10lld.%09lld  %10lld.%09lld   %10lu   %10lu\n",
                    (long) rows[k].frame_idx, (unsigned long) rows[k].main_idx, dt,
                    (long long) lsec, (long long) lrem, (long long) asec, (long long) arem,
                    (unsigned long) rows[k].cnt0, (unsigned long) rows[k].cnt1);
        }
    }

    int rc = ferror(fp);
    if (fclose(fp) != 0 || rc != 0)
    {
        PRINT_ERROR("write error on timing file \"%s\"", path);
        return RETURN_FAILURE;
    }
    return RETURN_SUCCESS;
}

void milk_tlm_free(MILK_TLM_TABLE *tbl)
{
    if (tbl != NULL)
    {
        free(tbl->rows);
        memset(tbl, 0, sizeof(*tbl));
    }
}
