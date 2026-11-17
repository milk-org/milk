// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    tlm_timing.h
 * @brief   Fast fixed-point reader and writer for milk telemetry timing files
 *
 * Parses and writes companion `.txt` timing files produced by `milk-logshim`
 * (7 columns) as well as `@grab --timestamps` files (4 columns), storing all
 * timestamps as exact 64-bit integer nanoseconds.
 */

#ifndef COREMOD_MEMORY_TLM_TIMING_H
#define COREMOD_MEMORY_TLM_TIMING_H

#include <stddef.h>
#include <stdint.h>

#include "libmilkcommon/milkDebugTools.h"

#ifdef __cplusplus
extern "C"
{
#endif

/**
     * struct MILK_TLM_ROW - One frame record from a telemetry timing file
     * @frame_idx: Zero-based frame index inside the companion FITS cube
     * @main_idx:  Main stream index (`cnt0` for `@grab --timestamps` files)
     * @t_log_ns:  Absolute logging epoch time in nanoseconds
     * @t_acq_ns:  Absolute acquisition epoch time in nanoseconds (`0` when unset)
     * @cnt0:      Stream `cnt0` counter
     * @cnt1:      Stream `cnt1` counter (`0` when absent)
     */
typedef struct
{
    int64_t  frame_idx;
    uint64_t main_idx;
    int64_t  t_log_ns;
    int64_t  t_acq_ns;
    uint64_t cnt0;
    uint64_t cnt1;
} MILK_TLM_ROW;

/**
     * struct MILK_TLM_TABLE - Parsed frame timing array from one timing file
     * @rows:  Heap-allocated array of @nrows frame records
     * @nrows: Number of parsed rows
     * @cap:   Allocated capacity in rows
     * @ncols: Detected column count (`7` = logshmim, `4` = `@grab --timestamps`, `0` = empty)
     */
typedef struct
{
    MILK_TLM_ROW *rows;
    size_t        nrows;
    size_t        cap;
    int           ncols;
} MILK_TLM_TABLE;

/**
     * milk_tlm_read_file - Parse a complete telemetry timing `.txt` file
     * @path: Path to the ASCII timing file
     * @out:  Output table struct (zero-initialized internally; caller frees via milk_tlm_free())
     *
     * Return: `RETURN_SUCCESS` on success, `RETURN_FAILURE` on I/O or format error
     */
errno_t milk_tlm_read_file(const char *path, MILK_TLM_TABLE *out);

/**
     * milk_tlm_peek_span - Read only the first and last data rows of a timing file
     * @path:         Path to the ASCII timing file
     * @first_out:    Output record for the first frame row
     * @last_out:     Output record for the last frame row
     * @approx_nrows: Optional output receiving exact or estimated row count (may be NULL)
     *
     * Reads at most 4 KiB from the head and 4 KiB from the tail of @path without
     * scanning the full file.
     *
     * Return: `RETURN_SUCCESS` on success, `RETURN_FAILURE` on I/O or format error
     */
errno_t milk_tlm_peek_span(const char   *path,
                           MILK_TLM_ROW *first_out,
                           MILK_TLM_ROW *last_out,
                           size_t       *approx_nrows);

/**
     * milk_tlm_write_file - Write frame timing rows in standard 7-column logshmim ASCII format
     * @path:  Output file path
     * @rows:  Input array of @nrows timing records (may be NULL if @nrows == 0)
     * @nrows: Number of records to write
     *
     * Return: `RETURN_SUCCESS` on success, `RETURN_FAILURE` on I/O error
     */
errno_t milk_tlm_write_file(const char *path, const MILK_TLM_ROW *rows, size_t nrows);

/**
     * milk_tlm_free - Release memory held by a `MILK_TLM_TABLE` and reset fields to zero
     * @tbl: Table to free (safe if NULL)
     */
void milk_tlm_free(MILK_TLM_TABLE *tbl);

#ifdef __cplusplus
}
#endif

#endif
