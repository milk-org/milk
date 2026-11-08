// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mc_linalg.h
 * @brief Internal interface for linear algebra, FFT, and stacking in `libmilkcalc`.
 *
 * Declares shape/type resolution (`mc_resolve_linalg_call`) and runtime execution entry
 * points (`mc_exec_blas_call`, `mc_exec_lapack_call`, `mc_exec_fft_or_stack`,
 * `mc_exec_ext_ctor`, `mc_exec_multi_linalg_or_ext`) for
 * non-elementwise constructor steps.
 */

#ifndef MILK_MC_LINALG_H
#define MILK_MC_LINALG_H

#include <stddef.h>
#include <stdint.h>

#include "ImageStreamIO/ImageStruct.h"
#include "mc_plan.h"

/** mc_write_im_pix() - write scalar value @v to element @idx of @im with datatype @dt. */
void mc_write_im_pix(IMAGE *im, uint8_t dt, uint64_t idx, double v);

/** mc_resolve_or_eval_im() - resolve identifier or evaluate sub-expression @e into an image ID. */
int mc_resolve_or_eval_im(const mc_expr_t *e,
                          imageID          avoid_id,
                          int              in_unit,
                          imageID         *out_id,
                          int             *is_tmp,
                          char            *err,
                          size_t           errsz);

/** mc_select_ctor_out_dt() - choose output datatype for constructor slot @slot given @pref_dt. */
uint8_t mc_select_ctor_out_dt(const mc_slot_t *slot, uint8_t pref_dt);

/** mc_multi_lhs_is_image() - return 1 if LHS target @lhs_idx of multi-output call @rhs0 is image. */
int mc_multi_lhs_is_image(const mc_expr_t *rhs0, int lhs_idx);

/** mc_resolve_linalg_call() - validate and type-check a MC_FN_LINALG call expression @e. */
int mc_resolve_linalg_call(mc_expr_t *e,
                           int        nlhs,
                           int        any_im,
                           int        any_cplx,
                           int        any_dbl,
                           char      *err,
                           size_t     errsz);

/** mc_ext_resolve_call() - validate and type-check a registered extension call @e. */
int mc_ext_resolve_call(mc_expr_t *e, int nlhs, char *err, size_t errsz);

/** mc_is_blas_or_linalg_ctor() - return 1 if assignment (@target, @aop, @rhs) lowers to BLAS/ctor. */
int mc_is_blas_or_linalg_ctor(const mc_expr_t *rhs, const mc_expr_t *target, mc_tok_type_t aop);

/** mc_exec_linalg_or_ext() - execute a BLAS, LAPACK, FFT, stack, or extension constructor step. */
int mc_exec_linalg_or_ext(mcalc_prog_t   *prog,
                          mc_ctor_step_t *cs,
                          imageID         avoid_id,
                          int             in_unit,
                          char           *err,
                          size_t          errsz);

/** mc_exec_blas_call() - execute a direct or algebraic-lowered BLAS / einsum constructor step. */
int mc_exec_blas_call(mcalc_prog_t   *prog,
                      mc_ctor_step_t *cs,
                      imageID         avoid_id,
                      int             in_unit,
                      char           *err,
                      size_t          errsz);

/** mc_exec_lapack_call() - execute a LAPACK solver, inverse, SVD, or symmetric eigensolver step. */
int mc_exec_lapack_call(mcalc_prog_t   *prog,
                        mc_ctor_step_t *cs,
                        imageID         avoid_id,
                        int             in_unit,
                        char           *err,
                        size_t          errsz);

/** mc_exec_fft_or_stack() - execute fft, ifft, fftshift, hstack, vstack, dstack, cat, or rot90. */
int mc_exec_fft_or_stack(mcalc_prog_t   *prog,
                         mc_ctor_step_t *cs,
                         imageID         avoid_id,
                         int             in_unit,
                         char           *err,
                         size_t          errsz);

/** mc_exec_ext_ctor() - execute a Tier 1a/1b/1c/2 user or module extension step. */
int mc_exec_ext_ctor(mcalc_prog_t   *prog,
                     mc_ctor_step_t *cs,
                     imageID         avoid_id,
                     int             in_unit,
                     char           *err,
                     size_t          errsz);

/** mc_exec_multi_linalg_or_ext() - execute a multi-LHS svd, eigh, or MILK_EXT_OP statement @s. */
int mc_exec_multi_linalg_or_ext(mcalc_prog_t    *prog,
                                mc_ctor_step_t  *cs,
                                const mc_stmt_t *s,
                                int              in_unit,
                                mc_val_t        *out,
                                char            *err,
                                size_t           errsz);

#endif /* MILK_MC_LINALG_H */
