// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/** @file MatMatMult_testPseudoInverse.c
 */

#ifdef HAVE_CUDA

#    ifdef HAVE_MAGMA

errno_t MatMatMult_testPseudoInverse_addCLIcmd();

long CLINALGEBRA_MatMatMult_testPseudoInverse(const char *IDmatA_name,
                                              const char *IDmatAinv_name,
                                              const char *IDmatOut_name);

#    endif

#endif
