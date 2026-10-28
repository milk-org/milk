// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_slice.c
 * @brief N-D image slice bounds evaluation and in-place image/slice assignment.
 *
 * Role & Goal:
 *   Evaluates bracketed N-D image slice expressions (`im[x0:x1, y0:y1]`) and
 *   in-place whole-image or sub-region assignments (`=`, `+=`, `-=`, `*=`, `/=`)
 *   within calculator expressions. Supports dynamic arithmetic expressions in
 *   slice bounds, fast-path single-pixel scalar reads/writes, and SHM-safe
 *   in-place updates with `processinfo_update_output_stream()` notification.
 */

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "CLIcore_script.h"
#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_memory/stream_slice.h"
#include "cli_calc_internal.h"
#include "cli_calc_parser.h"
#include "libmilkdata/milk_type_dispatch.h"

/** is_int_datatype - Return 1 for the signed and unsigned integer datatypes. */
static inline int is_int_datatype(uint8_t dt)
{
    return dt == _DATATYPE_UINT8 || dt == _DATATYPE_INT8 || dt == _DATATYPE_UINT16 ||
           dt == _DATATYPE_INT16 || dt == _DATATYPE_UINT32 || dt == _DATATYPE_INT32 ||
           dt == _DATATYPE_UINT64 || dt == _DATATYPE_INT64;
}

/** read_image_pixel - Read pixel @ii of @im as `VAL_LONG` for integer images or `VAL_DOUBLE`. */
static val_t read_image_pixel(const IMAGE *im, uint64_t ii)
{
    uint8_t dt = im->md[0].datatype;
    if (!is_int_datatype(dt))
    {
        return mk_double(cli_calc_read_im_pix(im, dt, ii));
    }
#define READ_INT_BODY(MBR) return mk_long((long) im->array.MBR[ii]);
    MILK_FOR_EACH_DATATYPE(dt, READ_INT_BODY, READ_INT_BODY(D))
#undef READ_INT_BODY
    return mk_long(0);
}

/** write_image_pixel - Store @lval (if @use_long) or @dval in pixel @ii of image @im. */
static void write_image_pixel(IMAGE *im, uint64_t ii, double dval, long lval, int use_long)
{
    uint8_t dt = im->md[0].datatype;
#define WRITE_PIX_BODY(MBR)                                                                       \
    {                                                                                             \
        im->array.MBR[ii] =                                                                       \
            use_long ? (__typeof__(im->array.MBR[0])) lval : (__typeof__(im->array.MBR[0])) dval; \
        return;                                                                                   \
    }
    MILK_FOR_EACH_DATATYPE(dt, WRITE_PIX_BODY, WRITE_PIX_BODY(D))
#undef WRITE_PIX_BODY
}

/** apply_compound_scalar - Combine an existing pixel value @old with @rhs under @aop. */
static inline int apply_compound_scalar(cli_token_type aop, double old, double rhs, double *out_val)
{
    switch (aop)
    {
    case TOK_OP_PLUS_EQ:
        *out_val = old + rhs;
        return 0;
    case TOK_OP_MINUS_EQ:
        *out_val = old - rhs;
        return 0;
    case TOK_OP_STAR_EQ:
        *out_val = old * rhs;
        return 0;
    case TOK_OP_SLASH_EQ:
        if (rhs == 0.0)
        {
            parse_errmsg("Division by zero");
            return -1;
        }
        *out_val = old / rhs;
        return 0;
    default:
        *out_val = rhs;
        return 0;
    }
}

/** apply_scalar_to_image_inplace - Apply `im aop val` on every pixel of @im. */
static int apply_scalar_to_image_inplace(IMAGE *im, cli_token_type aop, double val)
{
    return cli_calc_inplace(im, aop, NULL, val);
}

/** apply_image_to_image_inplace - Apply `dstim aop srcim` pixel by pixel. */
static int apply_image_to_image_inplace(IMAGE *dstim, const IMAGE *srcim, cli_token_type aop)
{
    if (aop == TOK_EQUAL && dstim->md[0].datatype == srcim->md[0].datatype)
    {
        int esize = ImageStreamIO_typesize(dstim->md[0].datatype);
        if (esize > 0)
        {
            __builtin_memcpy(dstim->array.raw, srcim->array.raw,
                             (size_t) esize * dstim->md[0].nelement);
            return 0;
        }
    }
    return cli_calc_inplace(dstim, aop, srcim, 0.0);
}

/** append_slice_str - Append string @s to `out_btext`, clamping `*out_len` to `out_sz - 1`. */
static void append_slice_str(char *out_btext, size_t out_sz, size_t *out_len, const char *s)
{
    if (*out_len + 1 < out_sz)
    {
        int w = snprintf(out_btext + *out_len, out_sz - *out_len, "%s", s);
        if (w > 0)
        {
            *out_len = (*out_len + (size_t) w < out_sz) ? (*out_len + (size_t) w) : (out_sz - 1);
        }
    }
}

/** eval_single_slice_bound_part - Resolve a single colon-delimited slice bound sub-expression. */
static int eval_single_slice_bound_part(char   *part_start,
                                        int     part_i,
                                        int     has_more_colon,
                                        char   *out_btext,
                                        size_t  out_sz,
                                        size_t *out_len,
                                        int    *out_is_single_pixel)
{
    while (*part_start == ' ' || *part_start == '\t')
    {
        part_start++;
    }
    char *part_end = part_start + strlen(part_start);
    while (part_end > part_start && (*(part_end - 1) == ' ' || *(part_end - 1) == '\t'))
    {
        *(--part_end) = '\0';
    }

    if (part_start[0] == '\0' || strcmp(part_start, "*") == 0 || strcmp(part_start, "-*") == 0)
    {
        if (part_i == 0 && !has_more_colon)
        {
            *out_is_single_pixel = 0;
        }
        append_slice_str(out_btext, out_sz, out_len, part_start);
        return 0;
    }

    int    has_bin = 0;
    size_t plen    = strlen(part_start);
    if (part_i >= 2 && plen > 1 && (part_start[plen - 1] == 'b' || part_start[plen - 1] == 'B') &&
        variable_ID(part_start) == -1 && cli_var_get(part_start) == NULL)
    {
        has_bin              = 1;
        part_start[plen - 1] = '\0';
    }

    char *endp = NULL;
    long  ival = strtol(part_start, &endp, 10);
    if (endp != part_start && *endp == '\0')
    {
        char num[64];
        snprintf(num, sizeof(num), "%ld%s", ival, has_bin ? "b" : "");
        append_slice_str(out_btext, out_sz, out_len, num);
        return 0;
    }

    int    mtype = 0;
    long   mlval = 0;
    double mdval = 0.0;
    if (!cli_calc_eval_math_to_val(part_start, &mtype, &mlval, &mdval))
    {
        char msg[256];
        snprintf(msg, sizeof(msg), "Invalid slice bound expression '%s'", part_start);
        parse_errmsg(msg);
        return -1;
    }
    long resolved = (mtype == 1) ? mlval : (long) llround(mdval);
    char num[64];
    snprintf(num, sizeof(num), "%ld%s", resolved, has_bin ? "b" : "");
    append_slice_str(out_btext, out_sz, out_len, num);
    return 0;
}

/** eval_slice_bounds - Evaluate variable or arithmetic expression bounds inside a slice string. */
static int eval_slice_bounds(const char *btext,
                             char       *out_btext,
                             size_t      out_sz,
                             int        *out_is_single_pixel)
{
    *out_is_single_pixel = 1;
    out_btext[0]         = '\0';
    size_t out_len       = 0;

    char buf[256];
    strncpy(buf, btext, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    char *p    = buf;
    int   axis = 0;
    while (*p != '\0')
    {
        char *axis_start = p;
        int   pdepth     = 0;
        while (*p != '\0' && !(*p == ',' && pdepth == 0))
        {
            if (*p == '(')
            {
                pdepth++;
            }
            else if (*p == ')')
            {
                pdepth--;
            }
            p++;
        }
        if (*p == ',')
        {
            *p = '\0';
            p++;
        }

        if (axis > 0)
        {
            append_slice_str(out_btext, out_sz, &out_len, ",");
        }
        axis++;

        char *ap     = axis_start;
        int   part_i = 0;
        while (1)
        {
            char *part_start = ap;
            int   sub_depth  = 0;
            while (*ap != '\0' && !(*ap == ':' && sub_depth == 0))
            {
                if (*ap == '(')
                {
                    sub_depth++;
                }
                else if (*ap == ')')
                {
                    sub_depth--;
                }
                ap++;
            }
            int has_more_colon = (*ap == ':');
            if (has_more_colon)
            {
                *ap = '\0';
                ap++;
                *out_is_single_pixel = 0;
            }
            if (part_i > 0)
            {
                append_slice_str(out_btext, out_sz, &out_len, ":");
            }
            if (eval_single_slice_bound_part(part_start, part_i, has_more_colon, out_btext, out_sz,
                                             &out_len, out_is_single_pixel) != 0)
            {
                return -1;
            }
            part_i++;
            if (!has_more_colon)
            {
                break;
            }
        }
    } /* while (*p != '\0') */

    return 0;
}

/** validate_slice_neg_bounds - Verify that negative slice indices do not underflow axis dimensions. */
static int validate_slice_neg_bounds(const IMGID_SLICE *slc, int snax, const uint32_t *ssz)
{
    for (int aa = 0; aa < slc->naxis && aa < snax; aa++)
    {
        int32_t sz = (int32_t) ssz[aa];
        if (slc->start[aa] < 0 && sz + slc->start[aa] < 0)
        {
            char msg[128];
            snprintf(msg, sizeof(msg), "axis %d start %d out of bounds for size %d", aa,
                     slc->start[aa], sz);
            parse_errmsg(msg);
            return -1;
        }
        if (slc->end[aa] < 0 && sz + slc->end[aa] < 0)
        {
            char msg[128];
            snprintf(msg, sizeof(msg), "axis %d end %d out of bounds for size %d", aa, slc->end[aa],
                     sz);
            parse_errmsg(msg);
            return -1;
        }
    }
    return 0;
}

/** Resolved slice metadata for an image token `bare[btext]`. */
typedef struct
{
    char        bare[200];
    imageID     srcid;
    IMAGE      *srcim;
    IMGID_SLICE slc;
    int         snax;
    uint32_t    ssz[3];
    uint32_t    outsz[3];
    int         onax;
    int         is_single_pixel;
} resolved_slice_t;

/** resolve_image_slice - Parse and validate an image slice token `bare[btext]` into @out. */
static int resolve_image_slice(const char       *tok_sval,
                               const char       *missing_im_msg,
                               resolved_slice_t *out)
{
    memset(out, 0, sizeof(*out));
    char btext[200];
    char resolved_btext[256];

    if (!imgid_slice_split_name(tok_sval, out->bare, (int) sizeof(out->bare), btext,
                                (int) sizeof(btext)))
    {
        parse_errmsg("Malformed slice: missing closing ']'");
        return -1;
    }
    out->srcid = image_ID(out->bare, data.core.image, data.core.NB_MAX_IMAGE);
    if (out->srcid == -1)
    {
        parse_errmsg(missing_im_msg);
        return -1;
    }
    if (eval_slice_bounds(btext, resolved_btext, sizeof(resolved_btext), &out->is_single_pixel) !=
        0)
    {
        return -1;
    }
    out->srcim = &data.core.image[out->srcid];
    out->slc   = imgid_slice_parse(resolved_btext);
    if (out->slc.error)
    {
        parse_errmsg(out->slc.errmsg);
        return -1;
    }
    out->snax   = out->srcim->md[0].naxis;
    out->ssz[0] = 1;
    out->ssz[1] = 1;
    out->ssz[2] = 1;
    for (int a = 0; a < out->snax && a < 3; a++)
    {
        out->ssz[a] = out->srcim->md[0].size[a];
    }
    if (validate_slice_neg_bounds(&out->slc, out->snax, out->ssz) != 0)
    {
        return -1;
    }
    if (imgid_slice_output_size(&out->slc, out->snax, out->ssz, out->outsz) != 0)
    {
        parse_errmsg(out->slc.errmsg[0] ? out->slc.errmsg : "Bad slice dims");
        return -1;
    }
    out->onax = 0;
    for (int a = 0; a < 3; a++)
    {
        if (out->outsz[a] > 0)
        {
            out->onax = a + 1;
        }
    }
    if (out->onax == 0)
    {
        out->onax = 1;
    }
    return 0;
}

/** single_pixel_flat_index - Compute the 1D flat array index for a resolved single-pixel slice. */
static inline uint64_t single_pixel_flat_index(const resolved_slice_t *rs)
{
    uint64_t ii = (uint64_t) rs->slc.start[0];
    if (rs->snax >= 2)
    {
        ii += (uint64_t) rs->slc.start[1] * rs->ssz[0];
    }
    if (rs->snax >= 3)
    {
        ii += (uint64_t) rs->slc.start[2] * rs->ssz[0] * rs->ssz[1];
    }
    return ii;
}

/**
 * cli_calc_eval_image_slice_assign - Evaluate LHS slice or single-pixel assignment `im[...] = rhs`
 * @tok_sval: Sliced LHS image token
 * @aop:      Assignment operator
 * @rhs:      Evaluated RHS value
 *
 * Return: Assigned `val_t`
 */
val_t cli_calc_eval_image_slice_assign(const char *tok_sval, cli_token_type aop, val_t rhs)
{
    resolved_slice_t rs;
    if (resolve_image_slice(tok_sval, "Target image not found for slice assignment", &rs) != 0)
    {
        return mk_double(0);
    }

    if (rs.is_single_pixel && rs.slc.naxis == rs.snax && rhs.type != VAL_STRING &&
        !cli_calc_is_complex_dt(rs.srcim->md[0].datatype))
    {
        uint64_t ii = single_pixel_flat_index(&rs);
        double   nv = to_double(rhs);
        if (aop != TOK_EQUAL)
        {
            double old = cli_calc_read_im_pix(rs.srcim, rs.srcim->md[0].datatype, ii);
            if (apply_compound_scalar(aop, old, nv, &nv) != 0)
            {
                return mk_double(0);
            }
        }
        rs.srcim->md[0].write = 1;
        write_image_pixel(rs.srcim, ii, nv, rhs.lval,
                          rhs.type == VAL_LONG && aop == TOK_EQUAL &&
                              is_int_datatype(rs.srcim->md[0].datatype));
        cli_calc_post_output(rs.srcim);
        return (rhs.type == VAL_LONG && aop == TOK_EQUAL) ? mk_long(rhs.lval) : mk_double(nv);
    }

    const char *tmpn = alloc_tmpname();
    imageID     tid  = -1;
    cli_calc_tmp_image(tmpn, rs.onax, rs.outsz, rs.srcim->md[0].datatype, 1, &tid);
    if (tid == -1)
    {
        parse_errmsg("Failed to allocate slice buffer");
        return mk_double(0);
    }

    IMGID simg;
    memset(&simg, 0, sizeof(simg));
    simg.ID       = rs.srcid;
    simg.im       = rs.srcim;
    simg.md       = rs.srcim->md;
    simg.slice    = rs.slc;
    simg.slice_im = &data.core.image[tid];
    if (aop != TOK_EQUAL)
    {
        imgid_slice_materialize(&simg);
    }

    if (rhs.type != VAL_STRING)
    {
        if (apply_scalar_to_image_inplace(simg.slice_im, aop, to_double(rhs)) != 0)
        {
            return mk_double(0);
        }
    }
    else
    {
        imageID rhsid = image_ID(rhs.sval, data.core.image, data.core.NB_MAX_IMAGE);
        if (rhsid == -1)
        {
            parse_errmsg("Source image does not exist");
            return mk_double(0);
        }
        IMAGE *rhsim = &data.core.image[rhsid];
        if (rhsim->md[0].nelement != simg.slice_im->md[0].nelement)
        {
            parse_errmsg("Dimension mismatch in slice assignment");
            return mk_double(0);
        }
        if (apply_image_to_image_inplace(simg.slice_im, rhsim, aop) != 0)
        {
            return mk_double(0);
        }
    }

    rs.srcim->md[0].write = 1;
    imgid_slice_writeback(&simg);
    cli_calc_post_output(rs.srcim);
    return mk_string(rs.bare);
}

/**
 * cli_calc_eval_image_whole_assign - Evaluate whole-image assignment `im = rhs` or compound `op=`
 * @tok_sval: Target image name
 * @aop:      Assignment operator
 * @rhs:      Evaluated RHS value
 *
 * Return: Assigned `val_t`
 */
val_t cli_calc_eval_image_whole_assign(const char *tok_sval, cli_token_type aop, val_t rhs)
{
    imageID dstid = image_ID(tok_sval, data.core.image, data.core.NB_MAX_IMAGE);
    if (dstid == -1)
    {
        parse_errmsg("Target image not found");
        return mk_double(0);
    }
    IMAGE *dstim = &data.core.image[dstid];

    if (rhs.type != VAL_STRING)
    {
        dstim->md[0].write = 1;
        if (apply_scalar_to_image_inplace(dstim, aop, to_double(rhs)) != 0)
        {
            dstim->md[0].write = 0;
            return mk_double(0);
        }
        cli_calc_post_output(dstim); /* deferred to the end of the tick inside a unit program */
        return mk_string(tok_sval);
    }

    imageID srcid = image_ID(rhs.sval, data.core.image, data.core.NB_MAX_IMAGE);
    if (srcid == -1)
    {
        parse_errmsg("Source image does not exist");
        return mk_double(0);
    }
    IMAGE *srcim = &data.core.image[srcid];

    if (dstim->md[0].nelement == srcim->md[0].nelement &&
        (dstim->md[0].shared == 1 || aop != TOK_EQUAL ||
         (dstim->md[0].naxis == srcim->md[0].naxis &&
          dstim->md[0].datatype == srcim->md[0].datatype)))
    {
        if (dstid != srcid || aop != TOK_EQUAL)
        {
            if (aop == TOK_EQUAL && dstim->md[0].shared == 0 && srcim->md[0].shared == 0 &&
                dstim->md[0].datatype == srcim->md[0].datatype &&
                strncmp(rhs.sval, "_tmpcalc", 8) == 0)
            {
                void *tmp_raw    = dstim->array.raw;
                dstim->array.raw = srcim->array.raw;
                srcim->array.raw = tmp_raw;
                dstim->md[0].cnt0++;
                return mk_string(tok_sval);
            }
            dstim->md[0].write = 1;
            if (apply_image_to_image_inplace(dstim, srcim, aop) != 0)
            {
                dstim->md[0].write = 0;
                return mk_double(0);
            }
            cli_calc_post_output(dstim); /* deferred to the end of the tick inside a unit program */
        }
        return mk_string(tok_sval);
    }

    if (dstim->md[0].shared == 1 || aop != TOK_EQUAL)
    {
        parse_errmsg("Dimension mismatch in image assignment");
        return mk_double(0);
    }

    FPS_STREAMNAME_PARSED sp = fps_streamname_parse(tok_sval);
    const char           *dst_bare =
        (!sp.error && sp.name != NULL && sp.name[0] != '\0') ? sp.name : tok_sval;
    int want_shm = (sp.loc == 'S') ? 1 : 0;

    delete_image_ID(dst_bare, 1);
    if (strncmp(rhs.sval, "_tmpcalc", 8) == 0 && !want_shm)
    {
        chname_image_ID(rhs.sval, dst_bare);
    }
    else
    {
        copy_image_ID(rhs.sval, dst_bare, want_shm);
    }
    if (data.core.Debug > 0)
    {
        printf("changing name\n");
    }
    return mk_string(dst_bare);
}

/**
 * cli_calc_eval_image_slice_read - Evaluate RHS slice or single-pixel read `im[...]`
 * @tok_sval: Sliced image token string
 *
 * Return: Scalar `val_t` for single pixel, or `VAL_STRING` naming the temporary slice image
 */
val_t cli_calc_eval_image_slice_read(const char *tok_sval)
{
    resolved_slice_t rs;
    if (resolve_image_slice(tok_sval, "Source image not found", &rs) != 0)
    {
        return mk_double(0);
    }

    if (parse_mode != 0 && rs.is_single_pixel && rs.slc.naxis == rs.snax &&
        !cli_calc_is_complex_dt(
            rs.srcim->md[0].datatype)) /* a complex pixel stays a 1-pixel image */
    {
        uint64_t ii = single_pixel_flat_index(&rs);
        return read_image_pixel(rs.srcim, ii);
    }

    const char *tmpn = alloc_tmpname();
    imageID     tid  = -1;
    cli_calc_tmp_image(tmpn, rs.onax, rs.outsz, rs.srcim->md[0].datatype, 1, &tid);
    if (tid == -1)
    {
        parse_errmsg("Failed to allocate slice image");
        return mk_string("");
    }
    IMGID simg;
    memset(&simg, 0, sizeof(simg));
    simg.ID       = rs.srcid;
    simg.im       = rs.srcim;
    simg.md       = rs.srcim->md;
    simg.slice    = rs.slc;
    simg.slice_im = &data.core.image[tid];
    imgid_slice_materialize(&simg);
    return mk_string(tmpn);
}

/**
 * cli_calc_slice_view - Describe slice token @tok_sval (`im[...]`) as a strided view of its image
 * @tok_sval: The image token with its bracket text
 * @missing:  Error message when the image does not exist
 * @out_im:   The sliced image
 * @base:     Element of @out_im at the view's origin
 * @stride:   Elements of @out_im per step along each view axis (negative for a reversed axis)
 * @vshape:   Extent of the view per axis (a single index keeps its axis with extent 1)
 * @naxis:    Axes of the view (those of the image)
 * @dyn:      1 when a bound holds a variable or an expression
 *
 * Return: 0; 1 for a slice that is not a view (binning); -1 on an error, which is reported.
 */
int cli_calc_slice_view(const char *tok_sval,
                        const char *missing,
                        IMAGE     **out_im,
                        int64_t    *base,
                        int64_t     stride[3],
                        uint32_t    vshape[3],
                        int        *naxis,
                        int        *dyn)
{
    resolved_slice_t rs;
    char             bare[200];
    char             btext[200];

    if (resolve_image_slice(tok_sval, missing, &rs) != 0)
    {
        return -1;
    }
    if (rs.is_single_pixel && rs.slc.naxis == rs.snax)
    {
        return 1; /* one pixel: a scalar on either side of an assignment, as the copying paths treat it */
    }
    for (int a = 0; a < rs.slc.naxis && a < 3; a++)
    {
        if (rs.slc.bin[a])
        {
            return 1;
        }
    }
    const int64_t ax[3] = { 1, (int64_t) rs.ssz[0], (int64_t) rs.ssz[0] * rs.ssz[1] };
    *base               = 0;
    for (int a = 0; a < 3; a++)
    {
        int     in    = (a < rs.slc.naxis && rs.slc.has_slice);
        int64_t start = in ? rs.slc.start[a] : 0;
        int64_t step  = in ? rs.slc.step[a] : 1;
        *base += start * ax[a];
        stride[a] = step * ax[a];
        vshape[a] = (rs.outsz[a] > 0) ? rs.outsz[a] : rs.ssz[a];
    }
    *naxis  = rs.onax;
    *out_im = rs.srcim;
    *dyn    = 0;
    if (imgid_slice_split_name(tok_sval, bare, (int) sizeof(bare), btext, (int) sizeof(btext)))
    {
        for (const char *c = btext; *c != '\0'; c++)
        {
            if (strchr("0123456789:,-* \t", *c) == NULL)
            {
                *dyn = 1;
                break;
            }
        }
    }
    return 0;
}

/** trim_part() - strip the blanks around a slice bound part in place; its start. */
static char *trim_part(char *part)
{
    while (*part == ' ' || *part == '\t')
    {
        part++;
    }
    char *end = part + strlen(part);
    while (end > part && (end[-1] == ' ' || end[-1] == '\t'))
    {
        *(--end) = '\0';
    }
    return part;
}

/** split_top_level() - end the segment at the first unparenthesized @sep; the next segment, or NULL. */
static char *split_top_level(char *seg, char sep)
{
    int depth = 0;
    for (char *c = seg; *c != '\0'; c++)
    {
        depth += (*c == '(') - (*c == ')');
        if (*c == sep && depth == 0)
        {
            *c = '\0';
            return c + 1;
        }
    }
    return NULL;
}

/**
 * cli_calc_slice_template - Split slice token @tok into its literal skeleton and expression bounds
 * @tok:      Image token `bare[btext]`
 * @bare:     Output image name
 * @baresz:   Size of @bare
 * @tmpl:     Output bracket text with each expression bound replaced by `'\001'`
 * @tmplsz:   Size of @tmpl
 * @parts:    Output expression texts, in bracket order
 * @maxparts: Capacity of @parts
 * @nparts:   Output number of expression bounds
 *
 * Literal bounds (numbers, `*`, `-*`, a trailing `b`) stay in @tmpl, as `eval_slice_bounds()` keeps them.
 *
 * Return: 0 on success, -1 if @tok is malformed or has more than @maxparts expression bounds.
 */
int cli_calc_slice_template(const char *tok,
                            char       *bare,
                            size_t      baresz,
                            char       *tmpl,
                            size_t      tmplsz,
                            char        parts[][64],
                            int         maxparts,
                            int        *nparts)
{
    char   btext[200];
    char   buf[256];
    size_t out_len = 0;

    *nparts = 0;
    tmpl[0] = '\0';
    if (!imgid_slice_split_name(tok, bare, (int) baresz, btext, (int) sizeof(btext)))
    {
        return -1;
    }
    snprintf(buf, sizeof(buf), "%s", btext);
    char *axis_seg = buf;
    int   axis     = 0;
    while (axis_seg != NULL)
    {
        char *next_axis = split_top_level(axis_seg, ',');
        if (axis++ > 0)
        {
            append_slice_str(tmpl, tmplsz, &out_len, ",");
        }
        char *part = axis_seg;
        int   pi   = 0;
        while (part != NULL)
        {
            char *next_part = split_top_level(part, ':');
            if (pi > 0)
            {
                append_slice_str(tmpl, tmplsz, &out_len, ":");
            }
            char *t = trim_part(part);
            if (t[0] == '\0' || strcmp(t, "*") == 0 || strcmp(t, "-*") == 0)
            {
                append_slice_str(tmpl, tmplsz, &out_len, t);
            }
            else
            {
                size_t plen = strlen(t);
                int has_bin = (pi >= 2 && plen > 1 && (t[plen - 1] == 'b' || t[plen - 1] == 'B') &&
                               variable_ID(t) == -1 && cli_var_get(t) == NULL);
                if (has_bin)
                {
                    t[plen - 1] = '\0';
                }
                char *endp = NULL;
                long  ival = strtol(t, &endp, 10);
                if (endp != t && *endp == '\0')
                {
                    char num[64];
                    snprintf(num, sizeof(num), "%ld%s", ival, has_bin ? "b" : "");
                    append_slice_str(tmpl, tmplsz, &out_len, num);
                }
                else
                {
                    if (*nparts >= maxparts)
                    {
                        return -1;
                    }
                    snprintf(parts[*nparts], 64, "%s", t);
                    (*nparts)++;
                    append_slice_str(tmpl, tmplsz, &out_len, has_bin ? "\001b" : "\001");
                }
            }
            part = next_part;
            pi++;
        }
        axis_seg = next_axis;
    }
    return 0;
}

/**
 * cli_calc_slice_geometry - View geometry of numeric bracket text @btext over image @im
 * @im:     Source image
 * @btext:  Bracket text with numeric bounds only
 * @base:   Output source element of the view's origin
 * @stride: Output source elements per step along each axis
 * @vshape: Output view extent per axis
 * @naxis:  Output number of axes of the view
 *
 * The runtime half of `cli_calc_slice_view()`, without name lookup or expression evaluation.
 *
 * Return: 0 for a view, 1 if @btext is not a view (binned), -1 on a bad slice (message set).
 */
int cli_calc_slice_geometry(const IMAGE *im,
                            const char  *btext,
                            int64_t     *base,
                            int64_t      stride[3],
                            uint32_t     vshape[3],
                            int         *naxis)
{
    IMGID_SLICE slc = imgid_slice_parse(btext);
    if (slc.error)
    {
        parse_errmsg(slc.errmsg);
        return -1;
    }
    int      snax     = im->md[0].naxis;
    uint32_t ssz[3]   = { 1, 1, 1 };
    uint32_t outsz[3] = { 0, 0, 0 };
    for (int a = 0; a < snax && a < 3; a++)
    {
        ssz[a] = im->md[0].size[a];
    }
    if (validate_slice_neg_bounds(&slc, snax, ssz) != 0)
    {
        return -1;
    }
    if (imgid_slice_output_size(&slc, snax, ssz, outsz) != 0)
    {
        parse_errmsg(slc.errmsg[0] ? slc.errmsg : "Bad slice dims");
        return -1;
    }
    for (int a = 0; a < slc.naxis && a < 3; a++)
    {
        if (slc.bin[a])
        {
            return 1;
        }
    }
    const int64_t ax[3] = { 1, (int64_t) ssz[0], (int64_t) ssz[0] * ssz[1] };
    *base               = 0;
    *naxis              = 0;
    for (int a = 0; a < 3; a++)
    {
        int     in    = (a < slc.naxis && slc.has_slice);
        int64_t start = in ? slc.start[a] : 0;
        int64_t step  = in ? slc.step[a] : 1;
        *base += start * ax[a];
        stride[a] = step * ax[a];
        vshape[a] = (outsz[a] > 0) ? outsz[a] : ssz[a];
        if (outsz[a] > 0)
        {
            *naxis = a + 1;
        }
    }
    if (*naxis == 0)
    {
        *naxis = 1;
    }
    return 0;
}
