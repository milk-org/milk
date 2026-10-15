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

/**
 * @brief Read pixel @ii of @im as a `double`.
 */
static double read_image_pixel_as_double(const IMAGE *im, uint64_t ii)
{
    uint8_t dt = im->md[0].datatype;
#define READ_PIX_BODY(MBR) return (double) im->array.MBR[ii];
    MILK_FOR_EACH_DATATYPE(dt, READ_PIX_BODY, READ_PIX_BODY(D))
#undef READ_PIX_BODY
    return 0.0;
}

/**
 * @brief Write @val to pixel @ii of @im.
 */
static void write_image_pixel_from_double(IMAGE *im, uint64_t ii, double val)
{
    uint8_t dt = im->md[0].datatype;
#define WRITE_PIX_BODY(MBR)                                     \
    {                                                           \
        im->array.MBR[ii] = (__typeof__(im->array.MBR[0])) val; \
        return;                                                 \
    }
    MILK_FOR_EACH_DATATYPE(dt, WRITE_PIX_BODY, WRITE_PIX_BODY(D))
#undef WRITE_PIX_BODY
}

/**
 * @brief Combine an existing pixel value @old with @rhs under compound operator @aop.
 */
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

/**
 * @brief Apply scalar assignment/compound op (`=`, `+=`, `-=`, `*=`, `/=`) to all pixels of @im.
 */
static int apply_scalar_to_image_inplace(IMAGE *im, cli_token_type aop, double val)
{
    if (aop == TOK_OP_SLASH_EQ && val == 0.0)
    {
        parse_errmsg("Division by zero");
        return -1;
    }
    uint64_t nelement = im->md[0].nelement;
    for (uint64_t ii = 0; ii < nelement; ii++)
    {
        double v = val;
        if (aop != TOK_EQUAL)
        {
            double old = read_image_pixel_as_double(im, ii);
            if (apply_compound_scalar(aop, old, val, &v) != 0)
            {
                return -1;
            }
        }
        write_image_pixel_from_double(im, ii, v);
    }
    return 0;
}

/**
 * @brief Apply image assignment/compound op (`=`, `+=`, `-=`, `*=`, `/=`) from @srcim into @dstim.
 */
static int apply_image_to_image_inplace(IMAGE *dstim, const IMAGE *srcim, cli_token_type aop)
{
    uint64_t nelement = dstim->md[0].nelement;
    if (aop == TOK_EQUAL && dstim->md[0].datatype == srcim->md[0].datatype)
    {
        int esize = ImageStreamIO_typesize(dstim->md[0].datatype);
        if (esize > 0)
        {
            __builtin_memcpy(dstim->array.raw, srcim->array.raw, (size_t) esize * nelement);
            return 0;
        }
    }
    for (uint64_t ii = 0; ii < nelement; ii++)
    {
        double sv = read_image_pixel_as_double(srcim, ii);
        double v  = sv;
        if (aop != TOK_EQUAL)
        {
            double old = read_image_pixel_as_double(dstim, ii);
            if (apply_compound_scalar(aop, old, sv, &v) != 0)
            {
                return -1;
            }
        }
        write_image_pixel_from_double(dstim, ii, v);
    }
    return 0;
}

/**
 * @brief Resolve a single colon-delimited slice bound sub-expression into @out_btext.
 */
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
        *out_len += (size_t) snprintf(out_btext + *out_len, out_sz - *out_len, "%s", part_start);
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
        *out_len += (size_t) snprintf(out_btext + *out_len, out_sz - *out_len, "%ld%s", ival,
                                      has_bin ? "b" : "");
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
    *out_len += (size_t) snprintf(out_btext + *out_len, out_sz - *out_len, "%ld%s", resolved,
                                  has_bin ? "b" : "");
    return 0;
}

/**
 * @brief Evaluate variable or arithmetic expression bounds inside a slice string.
 *
 * Converts e.g. `"x0:x1, cx-r:cx+r"` into resolved integer bounds `"32:95,48:80"`.
 * Sets @out_is_single_pixel to 1 if all axes are single indices without `:` or `*`.
 *
 * @return 0 on success, -1 on error
 */
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
            out_len += (size_t) snprintf(out_btext + out_len, out_sz - out_len, ",");
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
                out_len += (size_t) snprintf(out_btext + out_len, out_sz - out_len, ":");
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
    } // while (*p != '\0')

    return 0;
}

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

/**
 * @brief Parse and validate an image slice token `bare[btext]` into @out.
 */
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
 * @brief Evaluate LHS slice or single-pixel assignment `im[...] = rhs` (or compound op).
 */
val_t cli_calc_eval_image_slice_assign(const char *tok_sval, cli_token_type aop, val_t rhs)
{
    resolved_slice_t rs;
    if (resolve_image_slice(tok_sval, "Target image not found for slice assignment", &rs) != 0)
    {
        return mk_double(0);
    }

    if (rs.is_single_pixel && rs.slc.naxis == rs.snax && rhs.type != VAL_STRING)
    {
        uint64_t ii = single_pixel_flat_index(&rs);
        double   nv = to_double(rhs);
        if (aop != TOK_EQUAL)
        {
            double old = read_image_pixel_as_double(rs.srcim, ii);
            if (apply_compound_scalar(aop, old, nv, &nv) != 0)
            {
                return mk_double(0);
            }
        }
        rs.srcim->md[0].write = 1;
        write_image_pixel_from_double(rs.srcim, ii, nv);
        processinfo_update_output_stream(NULL, rs.srcim, NULL);
        return (rhs.type == VAL_LONG && aop == TOK_EQUAL) ? mk_long(rhs.lval) : mk_double(nv);
    }

    const char *tmpn = alloc_tmpname();
    imageID     tid  = -1;
    create_image_ID(tmpn, rs.onax, rs.outsz, rs.srcim->md[0].datatype, 0, 10, 0, &tid);
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
    processinfo_update_output_stream(NULL, rs.srcim, NULL);
    return mk_string(rs.bare);
}

/**
 * @brief Evaluate whole-image assignment `im = rhs` or compound assignment `+=`, `-=`, `*=`, `/=`.
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
        processinfo_update_output_stream(NULL, dstim, NULL);
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
            dstim->md[0].write = 1;
            if (apply_image_to_image_inplace(dstim, srcim, aop) != 0)
            {
                dstim->md[0].write = 0;
                return mk_double(0);
            }
            processinfo_update_output_stream(NULL, dstim, NULL);
        }
        return mk_string(tok_sval);
    }

    if (dstim->md[0].shared == 1 || aop != TOK_EQUAL)
    {
        parse_errmsg("Dimension mismatch in image assignment");
        return mk_double(0);
    }

    delete_image_ID(tok_sval, 1);
    if (strncmp(rhs.sval, "_tmpcalc", 8) == 0)
    {
        chname_image_ID(rhs.sval, tok_sval);
    }
    else
    {
        copy_image_ID(rhs.sval, tok_sval, 0);
    }
    if (data.core.Debug > 0)
    {
        printf("changing name\n");
    }
    return mk_string(tok_sval);
}

/**
 * @brief Evaluate RHS slice or single-pixel read `im[...]` into a scalar or temporary image.
 */
val_t cli_calc_eval_image_slice_read(const char *tok_sval)
{
    resolved_slice_t rs;
    if (resolve_image_slice(tok_sval, "Source image not found", &rs) != 0)
    {
        return mk_double(0);
    }

    if (parse_mode != 0 && rs.is_single_pixel && rs.slc.naxis == rs.snax)
    {
        uint64_t ii = single_pixel_flat_index(&rs);
        return mk_double(read_image_pixel_as_double(rs.srcim, ii));
    }

    const char *tmpn = alloc_tmpname();
    imageID     tid  = -1;
    create_image_ID(tmpn, rs.onax, rs.outsz, rs.srcim->md[0].datatype, 0, 10, 0, &tid);
    if (tid != -1)
    {
        IMGID simg;
        memset(&simg, 0, sizeof(simg));
        simg.ID       = rs.srcid;
        simg.im       = rs.srcim;
        simg.md       = rs.srcim->md;
        simg.slice    = rs.slc;
        simg.slice_im = &data.core.image[tid];
        imgid_slice_materialize(&simg);
    }
    return mk_string(tmpn);
}
