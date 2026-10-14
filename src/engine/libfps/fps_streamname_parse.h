// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    fps_streamname_parse.h
 * @brief   Parse @X: modifier prefixes from stream names
 *
 * Stream name strings may carry an optional prefix of
 * the form "@XY:name" where X,Y are modifier letters:
 *
 *   L  Local memory only (no SHM)
 *   S  Force shared memory
 *   F  Load from ./conf/shmim.<name>.fits
 *   E  Must exist (error if not found)
 *   N  Must not exist (error if found)
 *   T<type>  Pixel data type:
 *            TU8, TU16, TU32, TU64 (unsigned int)
 *            TI8, TI16, TI32, TI64 (signed int)
 *            TF16, TF32, TF64 (float16/32/64)
 *            TC64, TC128 (complex float32/64)
 *
 * Letters are stackable: "@LTF64:myimg" means local +
 * float64.  E and N are mutually exclusive.
 */

#ifndef FPS_STREAMNAME_PARSE_H
#define FPS_STREAMNAME_PARSE_H

#include <stdint.h>

/**
 * @brief Parsed stream name with modifier flags.
 */
typedef struct
{
    /** Location modifier: 'L','S','F', or '\0' */
    char loc;
    /** 1 if 'E' modifier present */
    int must_exist;
    /** 1 if 'N' modifier present */
    int must_new;
    /** Pixel datatype (_DATATYPE_*), or 0 if unspecified */
    uint8_t datatype;
    /** Pointer to bare name (past prefix) */
    const char *name;
    /** 1 if parse error (e.g. unknown modifier or E+N conflict) */
    int error;
} FPS_STREAMNAME_PARSED;

/**
 * @brief Parse a raw stream name string.
 *
 * If raw starts with '@' and contains ':', the
 * characters between '@' and ':' are scanned for
 * modifier letters.  Unknown letters or conflicting
 * modifiers set error = 1.
 *
 * @param raw  Raw stream name (possibly prefixed)
 * @return     Parsed result
 */
FPS_STREAMNAME_PARSED fps_streamname_parse(const char *raw);

/**
 * @brief Human-readable label for a modifier set.
 *
 * Writes a short tag like "[L]", "[SE]", "[TF32]", or ""
 * into the provided buffer.
 *
 * @param p      Parsed stream name
 * @param buf    Output buffer (>= 16 bytes)
 * @param bufsz  Size of buf
 */
void fps_streamname_modifier_label(const FPS_STREAMNAME_PARSED *p, char *buf, int bufsz);

/**
 * @brief Determine shared flag from stream name.
 *
 * Returns 0 if '@L:' prefix (local only).
 * Returns 1 otherwise (default = shared).
 *
 * @param raw  Raw stream name (possibly prefixed)
 * @return     1 for shared, 0 for local
 */
int fps_streamname_is_shared(const char *raw);

#endif
