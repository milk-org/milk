/**
 * @file    imgid_slice.c
 * @brief   Array slice parser and utilities
 *
 * Parses bracket-based slice specifications
 * (e.g. "0:19,10:29,-*") into IMGID_SLICE
 * descriptors. All string manipulation, no
 * image data dependencies.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "imgid_slice.h"


/**
 * @brief Parse one axis specification.
 *
 * Accepted forms:
 *   "*"          full axis, step +1
 *   "-*"         full axis, step -1 (flip)
 *   "start:end"  inclusive range, step +1
 *   "start:end:step"   with stride
 *   "start:end:stepb"  with binning
 *   ""           empty = full axis
 *
 * Negative start/end values are kept as-is;
 * they are resolved against the actual axis
 * size later by imgid_slice_output_size().
 *
 * @param spec  Single axis spec string
 * @param s     Slice descriptor to populate
 * @param axis  Axis index (0, 1, or 2)
 * @return 0 on success, 1 on error
 */
static int parse_one_axis(
    const char  *spec,
    IMGID_SLICE *s,
    int          axis
)
{
    /* Default: full axis, step +1, no bin */
    s->start[axis] = 0;
    s->end[axis]   = -1; /* sentinel: full axis */
    s->step[axis]  = 1;
    s->bin[axis]   = 0;

    if (spec == NULL || spec[0] == '\0')
    {
        /* Empty = full axis */
        return 0;
    }

    /* Full axis wildcard */
    if (strcmp(spec, "*") == 0)
    {
        return 0;
    }

    /* Flipped full axis */
    if (strcmp(spec, "-*") == 0)
    {
        s->step[axis] = -1;
        return 0;
    }

    /* Parse start:end or start:end:step[b] */
    char buf[64];
    strncpy(buf, spec, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    /* Count colons */
    int ncolon = 0;
    for (int ii = 0; buf[ii] != '\0'; ii++)
    {
        if (buf[ii] == ':')
        {
            ncolon++;
        }
    }

    if (ncolon == 0)
    {
        /* Single index: "N" -> start=N, end=N */
        int32_t idx = (int32_t) strtol(buf, NULL, 10);
        s->start[axis] = idx;
        s->end[axis]   = idx;
        s->step[axis]  = 1;
        return 0;
    }

    if (ncolon == 1)
    {
        /* "start:end" */
        char *colon = strchr(buf, ':');
        *colon = '\0';
        const char *sstart = buf;
        const char *send   = colon + 1;

        if (sstart[0] != '\0')
        {
            s->start[axis] =
                (int32_t) strtol(sstart, NULL, 10);
        }
        /* else: start=0 (default) */

        if (send[0] != '\0')
        {
            s->end[axis] =
                (int32_t) strtol(send, NULL, 10);
        }
        /* else: end=-1 (full) */

        return 0;
    }

    if (ncolon == 2)
    {
        /* "start:end:step[b]" */
        char *c1 = strchr(buf, ':');
        *c1 = '\0';
        char *c2 = strchr(c1 + 1, ':');
        *c2 = '\0';

        const char *sstart = buf;
        const char *send   = c1 + 1;
        const char *sstep  = c2 + 1;

        if (sstart[0] != '\0')
        {
            s->start[axis] =
                (int32_t) strtol(sstart, NULL, 10);
        }

        if (send[0] != '\0')
        {
            s->end[axis] =
                (int32_t) strtol(send, NULL, 10);
        }

        /* Check for 'b' suffix (binning mode) */
        int slen = (int) strlen(sstep);
        if (slen > 0
            && (sstep[slen - 1] == 'b'
                || sstep[slen - 1] == 'B'))
        {
            s->bin[axis] = 1;
            /* Parse number before 'b' */
            char stepnum[32];
            strncpy(stepnum, sstep, slen - 1);
            stepnum[slen - 1] = '\0';
            s->step[axis] =
                (int32_t) strtol(stepnum, NULL, 10);
        }
        else
        {
            s->step[axis] =
                (int32_t) strtol(sstep, NULL, 10);
        }

        /* Step = 0 is invalid */
        if (s->step[axis] == 0)
        {
            return 1;
        }

        return 0;
    }

    /* Too many colons */
    return 1;
}


/**
 * @brief Parse a bracket slice string.
 *
 * Splits the comma-separated axis specs and
 * delegates to parse_one_axis() for each.
 *
 * @param bracket_str  Text between [ and ]
 * @return Populated IMGID_SLICE descriptor
 */
IMGID_SLICE imgid_slice_parse(
    const char *bracket_str
)
{
    IMGID_SLICE s;
    memset(&s, 0, sizeof(s));

    if (bracket_str == NULL
        || bracket_str[0] == '\0')
    {
        s.has_slice = 0;
        return s;
    }

    s.has_slice = 1;

    /* Copy to mutable buffer */
    char buf[256];
    strncpy(buf, bracket_str, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    /* Split on commas */
    char *axes[IMGID_SLICE_MAXAXIS];
    int naxis = 0;

    char *tok = strtok(buf, ",");
    while (tok != NULL
           && naxis < IMGID_SLICE_MAXAXIS)
    {
        axes[naxis] = tok;
        naxis++;
        tok = strtok(NULL, ",");
    }

    if (naxis == 0)
    {
        s.error = 1;
        snprintf(s.errmsg, sizeof(s.errmsg),
                 "empty slice specification");
        return s;
    }

    s.naxis = (uint8_t) naxis;

    /* Initialize all axes to defaults */
    for (int aa = 0; aa < IMGID_SLICE_MAXAXIS; aa++)
    {
        s.start[aa] = 0;
        s.end[aa]   = -1;
        s.step[aa]  = 1;
        s.bin[aa]   = 0;
    }

    /* Parse each axis */
    for (int aa = 0; aa < naxis; aa++)
    {
        if (parse_one_axis(axes[aa], &s, aa) != 0)
        {
            s.error = 1;
            snprintf(s.errmsg,
                     sizeof(s.errmsg),
                     "invalid axis %d spec: %s",
                     aa, axes[aa]);
            return s;
        }
    }

    return s;
}


/**
 * @brief Resolve negative indices and compute
 *        output dimensions.
 *
 * Negative start/end values are converted to
 * positive offsets from the axis size. The
 * sentinel end=-1 from parse is resolved to
 * size-1 (full axis). Output sizes account for
 * stride and binning.
 *
 * @param s         Slice (modified in place)
 * @param src_naxis Source dimensionality
 * @param src_size  Source axis sizes
 * @param out_size  Output axis sizes (written)
 * @return 0 on success, 1 on error
 */
int imgid_slice_output_size(
    IMGID_SLICE    *s,
    int             src_naxis,
    const uint32_t *src_size,
    uint32_t       *out_size
)
{
    if (!s->has_slice)
    {
        for (int aa = 0; aa < src_naxis; aa++)
        {
            out_size[aa] = src_size[aa];
        }
        return 0;
    }

    for (int aa = 0; aa < s->naxis; aa++)
    {
        if (aa >= src_naxis)
        {
            snprintf(s->errmsg,
                     sizeof(s->errmsg),
                     "slice axis %d > naxis %d",
                     aa, src_naxis);
            s->error = 1;
            return 1;
        }

        int32_t sz = (int32_t) src_size[aa];

        /* Resolve negative indices */
        if (s->start[aa] < 0)
        {
            s->start[aa] = sz + s->start[aa];
        }
        if (s->end[aa] < 0)
        {
            /* Sentinel -1 from parser = full axis
             * Other negatives = count from end */
            s->end[aa] = sz + s->end[aa];
        }

        /* Clamp to valid range */
        if (s->start[aa] < 0)
        {
            s->start[aa] = 0;
        }
        if (s->start[aa] >= sz)
        {
            snprintf(s->errmsg,
                     sizeof(s->errmsg),
                     "axis %d start %d >= size %d",
                     aa, s->start[aa], sz);
            s->error = 1;
            return 1;
        }
        if (s->end[aa] >= sz)
        {
            snprintf(s->errmsg,
                     sizeof(s->errmsg),
                     "axis %d end %d >= size %d",
                     aa, s->end[aa], sz);
            s->error = 1;
            return 1;
        }
        if (s->end[aa] < 0)
        {
            s->end[aa] = 0;
        }

        /* Compute axis output length */
        int32_t absstep = abs(s->step[aa]);
        if (absstep == 0)
        {
            s->error = 1;
            snprintf(s->errmsg,
                     sizeof(s->errmsg),
                     "axis %d step is 0", aa);
            return 1;
        }

        int32_t range;
        if (s->step[aa] > 0)
        {
            range = s->end[aa] - s->start[aa] + 1;
        }
        else
        {
            /* Reversed: start > end */
            range = s->start[aa] - s->end[aa] + 1;
        }

        if (range <= 0)
        {
            snprintf(s->errmsg,
                     sizeof(s->errmsg),
                     "axis %d empty range "
                     "[%d:%d] step %d",
                     aa, s->start[aa],
                     s->end[aa], s->step[aa]);
            s->error = 1;
            return 1;
        }

        if (s->bin[aa])
        {
            /* Binning: output = range / step */
            out_size[aa] =
                (uint32_t)(range / absstep);
        }
        else
        {
            /* Stride: output = ceil(range/step) */
            out_size[aa] =
                (uint32_t)((range + absstep - 1)
                           / absstep);
        }

        if (out_size[aa] == 0)
        {
            out_size[aa] = 1;
        }
    }

    /* Axes beyond naxis: copy source size */
    for (int aa = s->naxis; aa < src_naxis; aa++)
    {
        out_size[aa] = src_size[aa];
    }

    return 0;
}


/**
 * @brief Format a slice descriptor as a string.
 *
 * Produces e.g. "[0:19,10:29]" for display.
 *
 * @param s     Slice descriptor
 * @param buf   Output buffer
 * @param bufsz Size of output buffer
 */
void imgid_slice_format(
    const IMGID_SLICE *s,
    char              *buf,
    int                bufsz
)
{
    if (!s->has_slice)
    {
        buf[0] = '\0';
        return;
    }

    int pos = 0;
    pos += snprintf(buf + pos, bufsz - pos, "[");

    for (int aa = 0; aa < s->naxis; aa++)
    {
        if (aa > 0)
        {
            pos += snprintf(buf + pos,
                            bufsz - pos, ",");
        }

        if (s->step[aa] == 1
            && s->start[aa] == 0
            && s->end[aa] == -1)
        {
            pos += snprintf(buf + pos,
                            bufsz - pos, "*");
        }
        else if (s->step[aa] == -1
                 && s->start[aa] == 0
                 && s->end[aa] == -1)
        {
            pos += snprintf(buf + pos,
                            bufsz - pos, "-*");
        }
        else if (s->step[aa] == 1)
        {
            pos += snprintf(buf + pos,
                            bufsz - pos,
                            "%d:%d",
                            s->start[aa],
                            s->end[aa]);
        }
        else
        {
            pos += snprintf(buf + pos,
                            bufsz - pos,
                            "%d:%d:%d%s",
                            s->start[aa],
                            s->end[aa],
                            s->step[aa],
                            s->bin[aa] ? "b" : "");
        }
    }

    snprintf(buf + pos, bufsz - pos, "]");
}


/**
 * @brief Generate a deterministic SHM name
 *        from source name and slice descriptor.
 *
 * Produces filesystem-safe name like
 * "wfs__s0_19_10_29" by replacing colons
 * and commas with underscores.
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
)
{
    char slicestr[128];
    imgid_slice_format(s, slicestr,
                       sizeof(slicestr));

    /* Replace [ ] : , with underscores */
    for (int ii = 0; slicestr[ii] != '\0'; ii++)
    {
        if (slicestr[ii] == '['
            || slicestr[ii] == ']'
            || slicestr[ii] == ':'
            || slicestr[ii] == ','
            || slicestr[ii] == '-')
        {
            slicestr[ii] = '_';
        }
    }

    snprintf(buf, bufsz, "%s__%s",
             srcname, slicestr);
}


/**
 * @brief Split a stream name into bare name
 *        and bracket contents.
 *
 * Given "im[0:19,10:29]", writes "im" into
 * name_buf and "0:19,10:29" into slice_buf.
 * Handles nested @X: prefixes -- the split
 * is done on the leftmost '['.
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
)
{
    const char *open = strchr(raw, '[');
    if (open == NULL)
    {
        strncpy(name_buf, raw, name_sz - 1);
        name_buf[name_sz - 1] = '\0';
        slice_buf[0] = '\0';
        return 0;
    }

    /* Copy bare name (before '[') */
    int nlen = (int)(open - raw);
    if (nlen >= name_sz)
    {
        nlen = name_sz - 1;
    }
    memcpy(name_buf, raw, nlen);
    name_buf[nlen] = '\0';

    /* Copy bracket contents (between [ and ]) */
    const char *close = strrchr(raw, ']');
    if (close == NULL || close <= open)
    {
        /* Malformed: no closing bracket */
        slice_buf[0] = '\0';
        return 0;
    }

    int slen = (int)(close - open - 1);
    if (slen >= slice_sz)
    {
        slen = slice_sz - 1;
    }
    if (slen > 0)
    {
        memcpy(slice_buf, open + 1, slen);
    }
    slice_buf[slen] = '\0';

    return 1;
}
