// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_arith__Cim_Cim__Cim.h
 *
 */

/* ------------------------------------------------------------------------- */
/* complex image, complex image  -> complex image                            */
/* ------------------------------------------------------------------------- */

/*
int arith_image_Cadd_byID(long ID1, long ID2, long IDout);
int arith_image_Csub_byID(long ID1, long ID2, long IDout);
int arith_image_Cmult_byID(long ID1, long ID2, long IDout);
int arith_image_Cdiv_byID(long ID1, long ID2, long IDout);
*/

int arith_image_Cadd(const char *ID1_name, const char *ID2_name, const char *ID_out);
int arith_image_Csub(const char *ID1_name, const char *ID2_name, const char *ID_out);
int arith_image_Cmult(const char *ID1_name, const char *ID2_name, const char *ID_out);
int arith_image_Cdiv(const char *ID1_name, const char *ID2_name, const char *ID_out);
