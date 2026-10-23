// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    imfunctions.c
 * @brief   Per-pixel function-pointer dispatch for image arithmetic
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
#include "image_stats.h"
#include "imfunctions.h"
#include "libmilkcommon/milk_compiler.h"

#ifdef _OPENMP
#    include <omp.h>
#    define OMP_NELEMENT_LIMIT 100000
#endif

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

/** Write @v into pixel @ii of `FLOAT` or `DOUBLE` image @im. */
static inline void write_out_pix(IMAGE *im, uint8_t dt, uint64_t ii, double v)
{
    if (dt == _DATATYPE_FLOAT)
    {
        im->array.F[ii] = (float) v;
    }
    else if (dt == _DATATYPE_DOUBLE)
    {
        im->array.D[ii] = v;
    }
}

errno_t arith_image_function_1_1_IMGID(IMGID *imgin, IMGID *imgout, double (*pt2function)(double))
{
    if (imgin->im == NULL)
    {
        return RETURN_FAILURE;
    }
    prep_unary_out(imgin, imgout);
    uint64_t nelement = imgin->md->nelement;
    uint8_t  idt      = imgin->md->datatype;
    uint8_t  odt      = imgout->mdt->datatype;
#ifdef _OPENMP
#    pragma omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)
#endif
    for (uint64_t ii = 0; ii < nelement; ii++)
    {
        write_out_pix(imgout->im, odt, ii, pt2function(arith_read_im_pix(imgin->im, idt, ii)));
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
    uint8_t  dt          = imgin->im->md->datatype;
    uint64_t nelement    = imgin->im->md->nelement;
    imgin->im->md->write = 0;
#ifdef _OPENMP
#    pragma omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)
#endif
    for (uint64_t ii = 0; ii < nelement; ii++)
    {
        double v = pt2function(arith_read_im_pix(imgin->im, dt, ii));
        if (dt == _DATATYPE_DOUBLE)
        {
            imgin->im->array.D[ii] = v;
        }
        else
        {
            imgin->im->array.F[ii] = (float) v;
        }
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
    uint8_t  odt      = outimg->mdt->datatype;
    uint8_t  dt1      = inimg1->md->datatype;
    uint8_t  dt2      = inimg2->md->datatype;

    if (same_shape)
    {
#ifdef _OPENMP
#    pragma omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)
#endif
        for (uint64_t ii = 0; ii < nelement; ii++)
        {
            double v1 = arith_read_im_pix(inimg1->im, dt1, ii);
            double v2 = arith_read_im_pix(inimg2->im, dt2, ii);
            write_out_pix(outimg->im, odt, ii, pt2function(v1, v2));
        }
        DEBUG_TRACE_FEXIT();
        return RETURN_SUCCESS;
    }

    uint64_t nx1  = inimg1->md->size[0];
    uint64_t nxy1 = nx1 * ((inimg1->md->naxis >= 2) ? inimg1->md->size[1] : 1);
    uint64_t nx2  = inimg2->md->size[0];
    uint64_t nxy2 = nx2 * ((inimg2->md->naxis >= 2) ? inimg2->md->size[1] : 1);

    for (uint32_t kk = 0; kk < osz[2]; kk++)
    {
        uint64_t z1 = (uint64_t) (kk * in1expand[2]) * nxy1;
        uint64_t z2 = (uint64_t) (kk * in2expand[2]) * nxy2;
        uint64_t zo = (uint64_t) kk * osz[1] * osz[0];
        for (uint32_t jj = 0; jj < osz[1]; jj++)
        {
            uint64_t y1 = z1 + (uint64_t) (jj * in1expand[1]) * nx1;
            uint64_t y2 = z2 + (uint64_t) (jj * in2expand[1]) * nx2;
            uint64_t yo = zo + (uint64_t) jj * osz[0];
            for (uint32_t ii = 0; ii < osz[0]; ii++)
            {
                double v1 = arith_read_im_pix(inimg1->im, dt1, y1 + ii * in1expand[0]);
                double v2 = arith_read_im_pix(inimg2->im, dt2, y2 + ii * in2expand[0]);
                write_out_pix(outimg->im, odt, yo + ii, pt2function(v1, v2));
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
    uint8_t  dt1       = img1.md->datatype;
    uint8_t  dt2       = img2.md->datatype;
    img1.im->md->write = 1;

#ifdef _OPENMP
#    pragma omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)
#endif
    for (uint64_t ii = 0; ii < nelement; ii++)
    {
        double v1 = arith_read_im_pix(img1.im, dt1, ii);
        double v2 = arith_read_im_pix(img2.im, dt2, ii);
        write_out_pix(img1.im, dt1, ii, pt2function(v1, v2));
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
    uint8_t  idt      = imgin->md->datatype;
    uint8_t  odt      = imgout->mdt->datatype;
#ifdef _OPENMP
#    pragma omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)
#endif
    for (uint64_t ii = 0; ii < nelement; ii++)
    {
        write_out_pix(imgout->im, odt, ii, pt2function(arith_read_im_pix(imgin->im, idt, ii), f1));
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
    uint8_t  dt       = imgin->im->md[0].datatype;
#ifdef _OPENMP
#    pragma omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)
#endif
    for (uint64_t ii = 0; ii < nelement; ii++)
    {
        write_out_pix(imgin->im, dt, ii, pt2function(arith_read_im_pix(imgin->im, dt, ii), f1));
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
    uint8_t  idt      = imgin->md->datatype;
    uint8_t  odt      = imgout->mdt->datatype;
#ifdef _OPENMP
#    pragma omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)
#endif
    for (uint64_t ii = 0; ii < nelement; ii++)
    {
        write_out_pix(imgout->im, odt, ii,
                      pt2function(arith_read_im_pix(imgin->im, idt, ii), f1, f2));
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
    uint8_t  dt       = imgin->im->md[0].datatype;
#ifdef _OPENMP
#    pragma omp parallel for simd if (nelement > OMP_NELEMENT_LIMIT)
#endif
    for (uint64_t ii = 0; ii < nelement; ii++)
    {
        write_out_pix(imgin->im, dt, ii, pt2function(arith_read_im_pix(imgin->im, dt, ii), f1, f2));
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
