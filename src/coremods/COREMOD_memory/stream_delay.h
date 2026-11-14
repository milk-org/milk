// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    stream_delay.h
 * @brief   Stream delay, fractional resampling, and causal temporal filtering engine
 */

#ifndef COREMOD_MEMORY_STREAM_DELAY_H
#define COREMOD_MEMORY_STREAM_DELAY_H

#include <stddef.h>
#include <stdint.h>
#include <time.h>

#include "milk_runnable.h"

/* Delay and window unit modes */
#define MILK_SDELAY_UNIT_TIME 0
#define MILK_SDELAY_UNIT_FRAME 1

/* Causal temporal window filter modes */
#define MILK_SDELAY_AVE_OFF 0
#define MILK_SDELAY_AVE_BOXCAR 1
#define MILK_SDELAY_AVE_LINEAR 2
#define MILK_SDELAY_AVE_EXP 3
#define MILK_SDELAY_AVE_GAUSS 4

/**
 * struct MILK_STREAM_DELAY_CFG - Per-step delay and temporal filter configuration
 * @delay:      Target lag in seconds (`unit_mode == 0`) or frames (`unit_mode == 1`)
 * @window:     Window width or FIR tap spacing in selected unit (`0.0` = 1 frame per tap)
 * @unit_mode:  `MILK_SDELAY_UNIT_TIME` (`0`) or `MILK_SDELAY_UNIT_FRAME` (`1`)
 * @interp:     `0` for discrete/causal-nearest frame selection, `1` for linear interpolation
 * @ave_mode:   Window filter mode (`MILK_SDELAY_AVE_*`, used when `weights_im == NULL`)
 * @naive_mode: `1` to busy-wait sleep without ring-buffer history, `0` for ring buffer
 * @force_push: `1` to push `in_im` on every step regardless of `cnt0`, `0` to push on `cnt0` change
 * @allow_skip: `1` to skip output write during warmup or duplicate discrete frames, `0` to always write
 */
typedef struct
{
    double delay;
    double window;
    int    unit_mode;
    int    interp;
    int    ave_mode;
    int    naive_mode;
    int    force_push;
    int    allow_skip;
} MILK_STREAM_DELAY_CFG;

/**
 * struct MILK_STREAM_DELAY_STATE - Persistent ring-buffer state for stream delaying/filtering
 * @ring_buf:        64-byte-aligned circular frame buffer (`buff_size * slot_stride` bytes)
 * @ts_arr:          Per-slot monotonic `CLOCK_ISIO` timestamps
 * @seq_arr:         Per-slot monotonic push sequence numbers
 * @slot_stamp:      Per-slot generation stamp for O(1) sparse weight accumulation
 * @slot_pos:        Per-slot index into `active_slots` / `active_weights`
 * @active_slots:    Scratch list of ring-buffer slot indices contributing to current output
 * @active_weights:  Scratch list of linear-combination weights for `active_slots`
 * @buff_size:       Allocated ring-buffer capacity in frames (`>= 2`)
 * @nelement:        Number of pixels/elements per frame
 * @frame_bytes:     Exact byte size of one frame (`nelement * typesize(datatype)`)
 * @slot_stride:     64-byte-padded byte stride between ring-buffer slots
 * @head_idx:        Next ring-buffer slot index to write (`0 .. buff_size - 1`)
 * @valid_cnt:       Number of frames currently stored in the ring buffer (`0 .. buff_size`)
 * @push_seq:        Total number of frames pushed since initialization
 * @last_cnt0:       Last observed `in_im->md[0].cnt0` when `force_push == 0`
 * @has_cnt0:        `1` once at least one input frame has been observed
 * @step_stamp:      Monotonic stamp incremented on each weight-assembly pass
 * @last_emitted_seq: Sequence number of last emitted discrete single-slot frame
 * @last_delay:      Delay value at last discrete single-slot emission
 * @last_unit:       Unit mode at last discrete single-slot emission
 * @naxis:           Input frame axis count (`1..3`)
 * @size:            Input frame dimensions (`size[0..2]`)
 * @datatype:        Input frame `_DATATYPE_*` code
 * @status_framelag: Telemetry: age in frames of primary selected history slot
 * @status_kkin:     Telemetry: most recently written ring-buffer slot index
 * @status_kkout:    Telemetry: primary read ring-buffer slot index
 */
typedef struct
{
    uint8_t         *ring_buf;
    struct timespec *ts_arr;
    uint64_t        *seq_arr;
    uint64_t        *slot_stamp;
    uint32_t        *slot_pos;
    uint64_t        *active_slots;
    double          *active_weights;
    uint64_t         buff_size;
    uint64_t         nelement;
    size_t           frame_bytes;
    size_t           slot_stride;
    uint64_t         head_idx;
    uint64_t         valid_cnt;
    uint64_t         push_seq;
    uint64_t         last_cnt0;
    int              has_cnt0;
    uint64_t         step_stamp;
    uint64_t         last_emitted_seq;
    double           last_delay;
    int              last_unit;
    uint8_t          naxis;
    uint32_t         size[3];
    uint8_t          datatype;
    uint64_t         status_framelag;
    uint64_t         status_kkin;
    uint64_t         status_kkout;
} MILK_STREAM_DELAY_STATE;

/**
 * milk_stream_delay_init - Allocate and initialize a `MILK_STREAM_DELAY_STATE` ring buffer
 * @state:     State structure to initialize (zeroed on entry)
 * @in_im:     Template input `IMAGE` defining geometry and datatype
 * @buff_size: Ring-buffer capacity in frames (clamped to `>= 2`)
 *
 * Pre-allocates all 64-byte-aligned frame slots and weight-assembly scratch arrays so
 * `milk_stream_delay_step()` performs zero heap allocations in the real-time loop.
 *
 * Return: `RETURN_SUCCESS` on success, `RETURN_FAILURE` on invalid geometry or allocation failure
 */
errno_t milk_stream_delay_init(MILK_STREAM_DELAY_STATE *state,
                               const IMAGE             *in_im,
                               uint64_t                 buff_size);

/**
 * milk_stream_delay_step - Push @in_im and compute the delayed/filtered frame into @out_im
 * @state:       Initialized `MILK_STREAM_DELAY_STATE`
 * @in_im:       Current input `IMAGE`
 * @weights_im:  Optional 1D FIR tap-weight `IMAGE` (`NULL` when unused)
 * @cfg:         Per-step delay, unit, interpolation, and window filter parameters
 * @out_im:      Destination `IMAGE` matching @in_im geometry (`_DATATYPE_FLOAT` or same datatype)
 * @out_updated: Output flag set to `1` if @out_im was written, `0` if skipped
 *
 * Return: `RETURN_SUCCESS` on success, `RETURN_FAILURE` on mismatched dimensions or null pointers
 */
errno_t milk_stream_delay_step(MILK_STREAM_DELAY_STATE     *state,
                               const IMAGE                 *in_im,
                               const IMAGE                 *weights_im,
                               const MILK_STREAM_DELAY_CFG *cfg,
                               IMAGE                       *out_im,
                               int                         *out_updated);

/**
 * milk_stream_delay_free - Release all memory owned by @state and reset it to zero
 * @state: State structure to release (safe on NULL or zeroed struct)
 */
void milk_stream_delay_free(MILK_STREAM_DELAY_STATE *state);

extern const MILK_RUNNABLE runnable_COREMOD_memory_streamdelay;

#endif // COREMOD_MEMORY_STREAM_DELAY_H
