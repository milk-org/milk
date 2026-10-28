// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    imfunctions.c
 * @brief   Block-vectorized function-pointer dispatch and 2D median filter for image arithmetic
 */

#include <math.h>
#include <stdlib.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#endif

#include "COREMOD_memory/COREMOD_memory.h"
#include "image_elem_block.h"
#include "image_stats.h"
#include "imfunctions.h"
#include "libmilkcommon/milk_compiler.h"
#include "mathfuncs.h"

#ifdef _OPENMP
#    include <omp.h>
#endif

#define FT double
#define FK(name) name##_d
#include "image_medfilt_block.h"
#undef FT
#undef FK

#define FT float
#define FK(name) name##_f
#include "image_medfilt_block.h"
#undef FT
#undef FK

#define RESOLVE_CALL_CLEANUP(ID_in, ID_out, CALL_EXPR)    \
    do                                                    \
    {                                                     \
        IMGID _in  = imgid_make_from_name(ID_in);         \
        IMGID _out = imgid_make_from_name(ID_out);        \
        resolveIMGID(&_in, ERRMODE_WARN, dcimg, dcnimg);  \
        resolveIMGID(&_out, ERRMODE_NULL, dcimg, dcnimg); \
        if (_in.ID == -1)                                 \
        {                                                 \
            return RETURN_FAILURE;                        \
        }                                                 \
        if (_out.ID == -1)                                \
        {                                                 \
            _out.mdt->shared = dcshareddft;               \
            _out.mdt->NBkw   = NB_KEYWNODE_MAX;           \
        }                                                 \
        errno_t _ret = (CALL_EXPR);                       \
        if (_out.ID == -1 && _out.im != NULL)             \
        {                                                 \
            RegisterIMGID(&_out, dcimg, dcnimg);          \
        }                                                 \
        imgid_free(&_in);                                 \
        imgid_free(&_out);                                \
        return _ret;                                      \
    } while (0)

/** Allocate or re-create @imgout with geometry from @imgin and `FLOAT`/`DOUBLE` precision. */
static void prep_unary_out(IMGID *imgin, IMGID *imgout)
{
    imgout->mdt->naxis = imgin->md->naxis;
    for (uint8_t i = 0; i < imgin->md->naxis; i++)
    {
        imgout->mdt->size[i] = imgin->md->size[i];
    }
    imgout->mdt->datatype =
        (imgin->md->datatype == _DATATYPE_DOUBLE) ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT;
    if (imgout->im == NULL)
    {
        imgout->im = (IMAGE *) calloc(1, sizeof(IMAGE));
    }
    else if (imgout->im->md && imgout->im->md->shared == 1)
    {
        ImageStreamIO_closeIm(imgout->im);
    }
    else
    {
        ImageStreamIO_destroyIm(imgout->im);
    }
    imgid_mkimage(imgout);
}

errno_t arith_image_function_1_1_IMGID(IMGID *imgin, IMGID *imgout, double (*pt2function)(double))
{
    if (imgin->im == NULL)
    {
        return RETURN_FAILURE;
    }
    prep_unary_out(imgin, imgout);
    uint64_t nelement = imgin->md->nelement;
    uint64_t nblk     = (nelement + ARITH_ELEM_BLK - 1) / ARITH_ELEM_BLK;
    int      nth      = milk_omp_team_size(nelement, 262144);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
#endif
    for (uint64_t b = 0; b < nblk; b++)
    {
        MILK_ALIGNED(64) double buf[ARITH_ELEM_BLK];
        uint64_t                off = b * ARITH_ELEM_BLK;
        int len = (int) ((nelement - off < ARITH_ELEM_BLK) ? (nelement - off) : ARITH_ELEM_BLK);
        arith_load_block_d_inline(imgin->im, off, len, buf);
        arith_elem_fn1_ptr_d_inline(buf, buf, pt2function, len);
        arith_store_block_d_inline(imgout->im, off, len, buf, 0, 0.0);
    }
    return RETURN_SUCCESS;
}

errno_t arith_image_function_1_1(const char *ID_name,
                                 const char *ID_out,
                                 double (*pt2function)(double))
{
    RESOLVE_CALL_CLEANUP(ID_name, ID_out, arith_image_function_1_1_IMGID(&_in, &_out, pt2function));
}

errno_t arith_image_function_1_1_inplace_IMGID(IMGID *imgin, double (*pt2function)(double))
{
    uint64_t nelement    = imgin->im->md->nelement;
    uint64_t nblk        = (nelement + ARITH_ELEM_BLK - 1) / ARITH_ELEM_BLK;
    int      nth         = milk_omp_team_size(nelement, 262144);
    imgin->im->md->write = 1;
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
#endif
    for (uint64_t b = 0; b < nblk; b++)
    {
        MILK_ALIGNED(64) double buf[ARITH_ELEM_BLK];
        uint64_t                off = b * ARITH_ELEM_BLK;
        int len = (int) ((nelement - off < ARITH_ELEM_BLK) ? (nelement - off) : ARITH_ELEM_BLK);
        arith_load_block_d_inline(imgin->im, off, len, buf);
        arith_elem_fn1_ptr_d_inline(buf, buf, pt2function, len);
        arith_store_block_d_inline(imgin->im, off, len, buf, 0, 0.0);
    }
    imgin->im->md->write = 0;
    imgin->im->md->cnt0++;
    return RETURN_SUCCESS;
}

errno_t arith_image_function_1_1_inplace(const char *ID_name, double (*pt2function)(double))
{
    IMGID img = imgid_make_from_name(ID_name);
    resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg);
    if (img.ID == -1)
    {
        return RETURN_FAILURE;
    }
    errno_t ret = arith_image_function_1_1_inplace_IMGID(&img, pt2function);
    imgid_free(&img);
    return ret;
}

errno_t arith_image_function_2_1_IMGID(IMGID *inimg1,
                                       IMGID *inimg2,
                                       IMGID *outimg,
                                       double (*pt2function)(double, double))
{
    DEBUG_TRACE_FSTART();
    if (inimg1->im == NULL || inimg2->im == NULL)
    {
        return RETURN_FAILURE;
    }
    if (outimg->im == NULL)
    {
        imgid_copy(inimg1, outimg);
    }

    outimg->mdt->naxis =
        (inimg2->md->naxis > inimg1->md->naxis) ? inimg2->md->naxis : inimg1->md->naxis;

    int      in1expand[3] = { 1, 1, 1 };
    int      in2expand[3] = { 1, 1, 1 };
    int      same_shape   = (inimg1->md->naxis == inimg2->md->naxis);
    uint32_t osz[3]       = { 1, 1, 1 };

    for (uint8_t axis = 0; axis < outimg->mdt->naxis; axis++)
    {
        uint32_t s1 = (axis < inimg1->md->naxis) ? inimg1->md->size[axis] : 1;
        uint32_t s2 = (axis < inimg2->md->naxis) ? inimg2->md->size[axis] : 1;
        if (s1 != s2)
        {
            same_shape = 0;
            if (s1 == 1)
            {
                in1expand[axis]         = 0;
                outimg->mdt->size[axis] = s2;
            }
            else if (s2 == 1)
            {
                in2expand[axis]         = 0;
                outimg->mdt->size[axis] = s1;
            }
            else
            {
                PRINT_ERROR("axis %d size %u and %u incompatible", axis, s1, s2);
                abort();
            }
        }
        osz[axis] = outimg->mdt->size[axis];
    }
    for (uint8_t axis = outimg->mdt->naxis; axis < 3; axis++)
    {
        outimg->mdt->size[axis] = 1;
    }

    outimg->mdt->datatype =
        (inimg1->md->datatype == _DATATYPE_DOUBLE || inimg2->md->datatype == _DATATYPE_DOUBLE)
            ? _DATATYPE_DOUBLE
            : _DATATYPE_FLOAT;

    if (outimg->im == NULL)
    {
        outimg->im = (IMAGE *) calloc(1, sizeof(IMAGE));
    }
    else if (outimg->im->md && outimg->im->md->shared == 1)
    {
        ImageStreamIO_closeIm(outimg->im);
    }
    else
    {
        ImageStreamIO_destroyIm(outimg->im);
    }
    imgid_mkimage(outimg);

    uint64_t nelement = outimg->md->nelement;
    uint8_t  dt1      = inimg1->md->datatype;
    uint8_t  dt2      = inimg2->md->datatype;
    int      nth      = milk_omp_team_size(nelement, 262144);

    if (same_shape)
    {
        uint64_t nblk = (nelement + ARITH_ELEM_BLK - 1) / ARITH_ELEM_BLK;
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
#endif
        for (uint64_t b = 0; b < nblk; b++)
        {
            MILK_ALIGNED(64) double buf1[ARITH_ELEM_BLK];
            MILK_ALIGNED(64) double buf2[ARITH_ELEM_BLK];
            uint64_t                off = b * ARITH_ELEM_BLK;
            int len = (int) ((nelement - off < ARITH_ELEM_BLK) ? (nelement - off) : ARITH_ELEM_BLK);
            arith_load_block_d_inline(inimg1->im, off, len, buf1);
            arith_load_block_d_inline(inimg2->im, off, len, buf2);
            arith_elem_fn2_ptr_d_inline(buf1, buf1, buf2, pt2function, len);
            arith_store_block_d_inline(outimg->im, off, len, buf1, 0, 0.0);
        }
        DEBUG_TRACE_FEXIT();
        return RETURN_SUCCESS;
    }

    uint64_t nx1  = inimg1->md->size[0];
    uint64_t nxy1 = nx1 * ((inimg1->md->naxis >= 2) ? inimg1->md->size[1] : 1);
    uint64_t nx2  = inimg2->md->size[0];
    uint64_t nxy2 = nx2 * ((inimg2->md->naxis >= 2) ? inimg2->md->size[1] : 1);

#ifdef _OPENMP
#    pragma omp parallel for collapse(2) schedule(static) num_threads(nth) if (nth > 1)
#endif
    for (uint32_t kk = 0; kk < osz[2]; kk++)
    {
        for (uint32_t jj = 0; jj < osz[1]; jj++)
        {
            MILK_ALIGNED(64) double buf1[ARITH_ELEM_BLK];
            MILK_ALIGNED(64) double buf2[ARITH_ELEM_BLK];
            uint64_t                z1 = (uint64_t) (kk * in1expand[2]) * nxy1;
            uint64_t                z2 = (uint64_t) (kk * in2expand[2]) * nxy2;
            uint64_t                zo = (uint64_t) kk * osz[1] * osz[0];
            uint64_t                y1 = z1 + (uint64_t) (jj * in1expand[1]) * nx1;
            uint64_t                y2 = z2 + (uint64_t) (jj * in2expand[1]) * nx2;
            uint64_t                yo = zo + (uint64_t) jj * osz[0];
            for (uint32_t x0 = 0; x0 < osz[0]; x0 += ARITH_ELEM_BLK)
            {
                int len = (int) ((osz[0] - x0 < ARITH_ELEM_BLK) ? (osz[0] - x0) : ARITH_ELEM_BLK);
                if (in1expand[0])
                {
                    arith_load_block_d_inline(inimg1->im, y1 + x0, len, buf1);
                }
                else
                {
                    double c1 = arith_read_im_pix(inimg1->im, dt1, y1);
                    _Pragma("omp simd") for (int j = 0; j < len; j++)
                    {
                        buf1[j] = c1;
                    }
                }
                if (in2expand[0])
                {
                    arith_load_block_d_inline(inimg2->im, y2 + x0, len, buf2);
                }
                else
                {
                    double c2 = arith_read_im_pix(inimg2->im, dt2, y2);
                    _Pragma("omp simd") for (int j = 0; j < len; j++)
                    {
                        buf2[j] = c2;
                    }
                }
                arith_elem_fn2_ptr_d_inline(buf1, buf1, buf2, pt2function, len);
                arith_store_block_d_inline(outimg->im, yo + x0, len, buf1, 0, 0.0);
            }
        }
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

errno_t arith_img_function_2_1(IMGID  inimg1,
                               IMGID  inimg2,
                               IMGID *outimg,
                               double (*pt2function)(double, double))
{
    return arith_image_function_2_1_IMGID(&inimg1, &inimg2, outimg, pt2function);
}

errno_t arith_image_function_2_1(const char *ID_name1,
                                 const char *ID_name2,
                                 const char *ID_out,
                                 double (*pt2function)(double, double))
{
    IMGID inimg1 = imgid_make_from_name(ID_name1);
    IMGID inimg2 = imgid_make_from_name(ID_name2);
    IMGID outimg = imgid_make_from_name(ID_out);

    resolveIMGID(&inimg1, ERRMODE_WARN, dcimg, dcnimg);
    resolveIMGID(&inimg2, ERRMODE_WARN, dcimg, dcnimg);
    if (inimg1.ID == -1 || inimg2.ID == -1)
    {
        return RETURN_FAILURE;
    }
    resolveIMGID(&outimg, ERRMODE_NULL, dcimg, dcnimg);
    if (outimg.ID == -1)
    {
        outimg.mdt->shared = dcshareddft;
        outimg.mdt->NBkw   = NB_KEYWNODE_MAX;
    }

    errno_t ret = arith_image_function_2_1_IMGID(&inimg1, &inimg2, &outimg, pt2function);
    if (outimg.ID == -1 && outimg.im != NULL)
    {
        RegisterIMGID(&outimg, dcimg, dcnimg);
    }
    imgid_free(&inimg1);
    imgid_free(&inimg2);
    imgid_free(&outimg);
    return ret;
}

errno_t arith_image_function_2_1_inplace(const char *ID_name1,
                                         const char *ID_name2,
                                         double (*pt2function)(double, double))
{
    IMGID img1 = imgid_make_from_name(ID_name1);
    IMGID img2 = imgid_make_from_name(ID_name2);

    resolveIMGID(&img1, ERRMODE_ABORT, dcimg, dcnimg);
    resolveIMGID(&img2, ERRMODE_ABORT, dcimg, dcnimg);
    if (img1.im == NULL || img2.im == NULL || img1.md->nelement != img2.md->nelement)
    {
        imgid_free(&img1);
        imgid_free(&img2);
        return RETURN_FAILURE;
    }

    uint64_t nelement  = img1.md->nelement;
    uint64_t nblk      = (nelement + ARITH_ELEM_BLK - 1) / ARITH_ELEM_BLK;
    int      nth       = milk_omp_team_size(nelement, 262144);
    img1.im->md->write = 1;

#ifdef _OPENMP
#    pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
#endif
    for (uint64_t b = 0; b < nblk; b++)
    {
        MILK_ALIGNED(64) double buf1[ARITH_ELEM_BLK];
        MILK_ALIGNED(64) double buf2[ARITH_ELEM_BLK];
        uint64_t                off = b * ARITH_ELEM_BLK;
        int len = (int) ((nelement - off < ARITH_ELEM_BLK) ? (nelement - off) : ARITH_ELEM_BLK);
        arith_load_block_d_inline(img1.im, off, len, buf1);
        arith_load_block_d_inline(img2.im, off, len, buf2);
        arith_elem_fn2_ptr_d_inline(buf1, buf1, buf2, pt2function, len);
        arith_store_block_d_inline(img1.im, off, len, buf1, 0, 0.0);
    }

    img1.im->md->write = 0;
    img1.im->md->cnt0++;
    imgid_free(&img1);
    imgid_free(&img2);
    return RETURN_SUCCESS;
}

int arith_image_function_1f_1_IMGID(IMGID *imgin,
                                    double f1,
                                    IMGID *imgout,
                                    double (*pt2function)(double, double))
{
    if (imgin->im == NULL)
    {
        return RETURN_FAILURE;
    }
    prep_unary_out(imgin, imgout);
    uint64_t nelement = imgin->md->nelement;
    uint64_t nblk     = (nelement + ARITH_ELEM_BLK - 1) / ARITH_ELEM_BLK;
    int      nth      = milk_omp_team_size(nelement, 262144);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
#endif
    for (uint64_t b = 0; b < nblk; b++)
    {
        MILK_ALIGNED(64) double buf[ARITH_ELEM_BLK];
        uint64_t                off = b * ARITH_ELEM_BLK;
        int len = (int) ((nelement - off < ARITH_ELEM_BLK) ? (nelement - off) : ARITH_ELEM_BLK);
        arith_load_block_d_inline(imgin->im, off, len, buf);
        arith_elem_fn1f_ptr_d_inline(buf, buf, f1, pt2function, len);
        arith_store_block_d_inline(imgout->im, off, len, buf, 0, 0.0);
    }
    return EXIT_SUCCESS;
}

int arith_image_function_1f_1(const char *ID_name,
                              double      f1,
                              const char *ID_out,
                              double (*pt2function)(double, double))
{
    RESOLVE_CALL_CLEANUP(ID_name, ID_out,
                         arith_image_function_1f_1_IMGID(&_in, f1, &_out, pt2function));
}

int arith_image_function_1f_1_inplace_IMGID(IMGID *imgin,
                                            double f1,
                                            double (*pt2function)(double, double))
{
    uint64_t nelement = imgin->im->md[0].nelement;
    uint64_t nblk     = (nelement + ARITH_ELEM_BLK - 1) / ARITH_ELEM_BLK;
    int      nth      = milk_omp_team_size(nelement, 262144);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
#endif
    for (uint64_t b = 0; b < nblk; b++)
    {
        MILK_ALIGNED(64) double buf[ARITH_ELEM_BLK];
        uint64_t                off = b * ARITH_ELEM_BLK;
        int len = (int) ((nelement - off < ARITH_ELEM_BLK) ? (nelement - off) : ARITH_ELEM_BLK);
        arith_load_block_d_inline(imgin->im, off, len, buf);
        arith_elem_fn1f_ptr_d_inline(buf, buf, f1, pt2function, len);
        arith_store_block_d_inline(imgin->im, off, len, buf, 0, 0.0);
    }
    return EXIT_SUCCESS;
}

int arith_image_function_1f_1_inplace(const char *ID_name,
                                      double      f1,
                                      double (*pt2function)(double, double))
{
    IMGID img = imgid_make_from_name(ID_name);
    resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg);
    if (img.ID == -1)
    {
        return RETURN_FAILURE;
    }
    int ret = arith_image_function_1f_1_inplace_IMGID(&img, f1, pt2function);
    imgid_free(&img);
    return ret;
}

int arith_image_function_1ff_1_IMGID(IMGID *imgin,
                                     double f1,
                                     double f2,
                                     IMGID *imgout,
                                     double (*pt2function)(double, double, double))
{
    if (imgin->im == NULL)
    {
        return RETURN_FAILURE;
    }
    prep_unary_out(imgin, imgout);
    uint64_t nelement = imgin->md->nelement;
    uint64_t nblk     = (nelement + ARITH_ELEM_BLK - 1) / ARITH_ELEM_BLK;
    int      nth      = milk_omp_team_size(nelement, 262144);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
#endif
    for (uint64_t b = 0; b < nblk; b++)
    {
        MILK_ALIGNED(64) double buf[ARITH_ELEM_BLK];
        uint64_t                off = b * ARITH_ELEM_BLK;
        int len = (int) ((nelement - off < ARITH_ELEM_BLK) ? (nelement - off) : ARITH_ELEM_BLK);
        arith_load_block_d_inline(imgin->im, off, len, buf);
        if (pt2function == &Ptrunc)
        {
            arith_elem_clamp_d_inline(buf, buf, f1, f2, len);
        }
        else
        {
            for (int j = 0; j < len; j++)
            {
                buf[j] = pt2function(buf[j], f1, f2);
            }
        }
        arith_store_block_d_inline(imgout->im, off, len, buf, 0, 0.0);
    }
    return 0;
}

int arith_image_function_1ff_1(const char *ID_name,
                               double      f1,
                               double      f2,
                               const char *ID_out,
                               double (*pt2function)(double, double, double))
{
    RESOLVE_CALL_CLEANUP(ID_name, ID_out,
                         arith_image_function_1ff_1_IMGID(&_in, f1, f2, &_out, pt2function));
}

int arith_image_function_1ff_1_inplace_IMGID(IMGID *imgin,
                                             double f1,
                                             double f2,
                                             double (*pt2function)(double, double, double))
{
    uint64_t nelement = imgin->im->md[0].nelement;
    uint64_t nblk     = (nelement + ARITH_ELEM_BLK - 1) / ARITH_ELEM_BLK;
    int      nth      = milk_omp_team_size(nelement, 262144);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
#endif
    for (uint64_t b = 0; b < nblk; b++)
    {
        MILK_ALIGNED(64) double buf[ARITH_ELEM_BLK];
        uint64_t                off = b * ARITH_ELEM_BLK;
        int len = (int) ((nelement - off < ARITH_ELEM_BLK) ? (nelement - off) : ARITH_ELEM_BLK);
        arith_load_block_d_inline(imgin->im, off, len, buf);
        if (pt2function == &Ptrunc)
        {
            arith_elem_clamp_d_inline(buf, buf, f1, f2, len);
        }
        else
        {
            for (int j = 0; j < len; j++)
            {
                buf[j] = pt2function(buf[j], f1, f2);
            }
        }
        arith_store_block_d_inline(imgin->im, off, len, buf, 0, 0.0);
    }
    return 0;
}

int arith_image_function_1ff_1_inplace(const char *ID_name,
                                       double      f1,
                                       double      f2,
                                       double (*pt2function)(double, double, double))
{
    IMGID img = imgid_make_from_name(ID_name);
    resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg);
    if (img.ID == -1)
    {
        return RETURN_FAILURE;
    }
    int ret = arith_image_function_1ff_1_inplace_IMGID(&img, f1, f2, pt2function);
    imgid_free(&img);
    return ret;
}

errno_t arith_image_medfilt2d_im(const IMAGE *src, IMAGE *dst, int filter_size)
{
    if (src == NULL || dst == NULL || filter_size < 0 || filter_size > 3)
    {
        return RETURN_FAILURE;
    }
    uint64_t nelement = src->md[0].nelement;
    if (filter_size == 0)
    {
        uint64_t nblk = (nelement + ARITH_ELEM_BLK - 1) / ARITH_ELEM_BLK;
        int      nth  = milk_omp_team_size(nelement, 262144);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) num_threads(nth) if (nth > 1)
#endif
        for (uint64_t b = 0; b < nblk; b++)
        {
            MILK_ALIGNED(64) double buf[ARITH_ELEM_BLK];
            uint64_t                off = b * ARITH_ELEM_BLK;
            int len = (int) ((nelement - off < ARITH_ELEM_BLK) ? (nelement - off) : ARITH_ELEM_BLK);
            arith_load_block_d_inline(src, off, len, buf);
            arith_store_block_d_inline(dst, off, len, buf, 0, 0.0);
        }
        return RETURN_SUCCESS;
    }

    const int64_t nx = (int64_t) src->md[0].size[0];
    const int64_t ny =
        (src->md[0].naxis > 1 && src->md[0].size[1] > 0) ? (int64_t) src->md[0].size[1] : 1;
    const int64_t nz =
        (src->md[0].naxis > 2 && src->md[0].size[2] > 0) ? (int64_t) src->md[0].size[2] : 1;
    const int K   = 2 * filter_size + 1;
    const int rad = filter_size;
    int       nth = milk_omp_team_size(nelement, 65536);

    if (src->md[0].datatype == _DATATYPE_FLOAT && dst->md[0].datatype == _DATATYPE_FLOAT)
    {
        const float *MILK_RESTRICT in_f  = src->array.F;
        float *MILK_RESTRICT       out_f = dst->array.F;
#ifdef _OPENMP
#    pragma omp parallel for collapse(2) schedule(static) num_threads(nth) if (nth > 1)
#endif
        for (int64_t z = 0; z < nz; z++)
        {
            for (int64_t y = 0; y < ny; y++)
            {
                MILK_ALIGNED(64) float col[7][ARITH_ELEM_BLK + 6];
                int64_t                row_off = (z * ny + y) * nx;
                for (int64_t x0 = 0; x0 < nx; x0 += ARITH_ELEM_BLK)
                {
                    int N = (int) ((nx - x0 < ARITH_ELEM_BLK) ? (nx - x0) : ARITH_ELEM_BLK);
                    int W = N + K - 1;
                    for (int r = 0; r < K; r++)
                    {
                        int64_t sy       = y + (r - rad);
                        sy               = (sy < 0) ? 0 : ((sy >= ny) ? ny - 1 : sy);
                        const float *row = in_f + (z * ny + sy) * nx;
                        float       *cr  = col[r];
                        for (int u = 0; u < W; u++)
                        {
                            int64_t sx = x0 + u - rad;
                            sx         = (sx < 0) ? 0 : ((sx >= nx) ? nx - 1 : sx);
                            cr[u]      = row[sx];
                        }
                    }
                    arith_medfilt_cols_f(K, N, col, out_f + row_off + x0);
                }
            }
        }
        return RETURN_SUCCESS;
    }

    uint8_t dt = src->md[0].datatype;
#ifdef _OPENMP
#    pragma omp parallel for collapse(2) schedule(static) num_threads(nth) if (nth > 1)
#endif
    for (int64_t z = 0; z < nz; z++)
    {
        for (int64_t y = 0; y < ny; y++)
        {
            MILK_ALIGNED(64) double col[7][ARITH_ELEM_BLK + 6];
            MILK_ALIGNED(64) double dbuf[ARITH_ELEM_BLK];
            int64_t                 row_off = (z * ny + y) * nx;
            for (int64_t x0 = 0; x0 < nx; x0 += ARITH_ELEM_BLK)
            {
                int N = (int) ((nx - x0 < ARITH_ELEM_BLK) ? (nx - x0) : ARITH_ELEM_BLK);
                int W = N + K - 1;
#define MEDFILT_ROW_BODY(MBR)                                                                     \
    {                                                                                             \
        for (int r = 0; r < K; r++)                                                               \
        {                                                                                         \
            int64_t sy                               = y + (r - rad);                             \
            sy                                       = (sy < 0) ? 0 : ((sy >= ny) ? ny - 1 : sy); \
            const __typeof__(src->array.MBR[0]) *row = src->array.MBR + (z * ny + sy) * nx;       \
            double                              *cr  = col[r];                                    \
            for (int u = 0; u < W; u++)                                                           \
            {                                                                                     \
                int64_t sx = x0 + u - rad;                                                        \
                sx         = (sx < 0) ? 0 : ((sx >= nx) ? nx - 1 : sx);                           \
                cr[u]      = (double) row[sx];                                                    \
            }                                                                                     \
        }                                                                                         \
    }
                MILK_FOR_EACH_DATATYPE(dt, MEDFILT_ROW_BODY, MEDFILT_ROW_BODY(D))
#undef MEDFILT_ROW_BODY
                arith_medfilt_cols_d(K, N, col, dbuf);
                arith_store_block_d_inline(dst, (uint64_t) (row_off + x0), N, dbuf, 0, 0.0);
            }
        }
    }
    return RETURN_SUCCESS;
}
