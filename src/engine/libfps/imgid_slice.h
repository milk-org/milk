/**
 * @file    imgid_slice.h
 * @brief   Array slice descriptor and parser
 *
 * Adds FITSIO-like bracket syntax to stream names:
 *
 *   im[0:19,10:29]      crop to x=[0,19], y=[10,29]
 *   im[*,*,-*]          flip the third axis
 *   im[0:99:2,*]        take every 2nd column
 *   im[0:99:2b,*]       2x binning (average) on x
 *   im[-8:-1,*]         last 8 columns
 *
 * Indices are 0-based, inclusive on both ends.
 * Negative indices count from the end of the axis.
 *
 * The parser produces an IMGID_SLICE descriptor
 * that is carried inside the IMGID struct.
 * Materialization (data copy) is deferred until
 * imgid_get_image() is called.
 */

#ifndef IMGID_SLICE_H
#define IMGID_SLICE_H

#include <stdint.h>

#define IMGID_SLICE_MAXAXIS 3

/**
 * @brief Array slice descriptor.
 *
 * Populated by imgid_slice_parse().
 * Carried inside IMGID to describe a sub-view
 * of the source stream.
 */
typedef struct
{
    /** 1 if a slice was parsed, 0 otherwise. */
    uint8_t has_slice;

    /** Number of axes with slice specs. */
    uint8_t naxis;

    /** Start index per axis (inclusive, 0-based).
     *  Negative values mean offset from end. */
    int32_t start[IMGID_SLICE_MAXAXIS];

    /** End index per axis (inclusive, 0-based).
     *  Negative values mean offset from end. */
    int32_t end[IMGID_SLICE_MAXAXIS];

    /** Stride per axis.
     *  1 = every element (default).
     *  N = every Nth element.
     *  Negative = reversed traversal. */
    int32_t step[IMGID_SLICE_MAXAXIS];

    /** Per-axis binning flag.
     *  0 = stride/skip mode (default).
     *  1 = bin/average mode. */
    uint8_t bin[IMGID_SLICE_MAXAXIS];

    /** 1 if the parser encountered an error. */
    uint8_t error;

    /** Human-readable error message (if any). */
    char errmsg[80];

} IMGID_SLICE;


/**
 * @brief Parse a bracket slice string.
 *
 * Input: the text inside the brackets, e.g.
 * "0:19,10:29" or "*,*,-*".
 * Does NOT include the surrounding [ ].
 *
 * @param bracket_str  Text between [ and ]
 * @return Populated IMGID_SLICE descriptor
 */
IMGID_SLICE imgid_slice_parse(
    const char *bracket_str
);


/**
 * @brief Compute output dimensions for a slice.
 *
 * Resolves negative indices against the source
 * sizes and computes the resulting axis lengths.
 *
 * @param s         Slice descriptor (modified:
 *                  negative indices resolved)
 * @param src_naxis Source dimensionality
 * @param src_size  Source axis sizes
 * @param out_size  Output axis sizes (written)
 * @return 0 on success, 1 on error
 */
int imgid_slice_output_size(
    IMGID_SLICE *s,
    int          src_naxis,
    const uint32_t *src_size,
    uint32_t    *out_size
);


/**
 * @brief Format a slice descriptor as a string.
 *
 * Writes e.g. "[0:19,10:29]" into buf.
 * Used for generating deterministic SHM names
 * for shared materialized slices.
 *
 * @param s     Slice descriptor
 * @param buf   Output buffer
 * @param bufsz Size of output buffer
 */
void imgid_slice_format(
    const IMGID_SLICE *s,
    char              *buf,
    int                bufsz
);


/**
 * @brief Generate a deterministic SHM name.
 *
 * Combines the source stream name and slice
 * descriptor into a unique, filesystem-safe name,
 * e.g. "wfs__s0_19_10_29".
 *
 * @param srcname  Source stream bare name
 * @param s        Slice descriptor
 * @param buf      Output buffer
 * @param bufsz    Size of output buffer
 */
void imgid_slice_shmname(
    const char        *srcname,
    const IMGID_SLICE *s,
    char              *buf,
    int                bufsz
);


/**
 * @brief Split a stream name into bare name
 *        and bracket string.
 *
 * Given "im[0:19,10:29]", writes "im" into
 * name_buf and "0:19,10:29" into slice_buf.
 * If no brackets found, copy full name and
 * set slice_buf to "".
 *
 * @param raw       Full stream name string
 * @param name_buf  Output: bare name
 * @param name_sz   Size of name_buf
 * @param slice_buf Output: bracket contents
 * @param slice_sz  Size of slice_buf
 * @return 1 if brackets found, 0 otherwise
 */
int imgid_slice_split_name(
    const char *raw,
    char       *name_buf,
    int         name_sz,
    char       *slice_buf,
    int         slice_sz
);


#endif
