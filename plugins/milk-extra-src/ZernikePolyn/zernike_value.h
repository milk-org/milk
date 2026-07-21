// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file zernike_value.h
 * @brief Zernike value module
 */

/**
 * @file zernike_value.h
 */


#ifndef _ZERNIKEPOLYN_VALUE_H
#define _ZERNIKEPOLYN_VALUE_H

double fact(int n);

int zernike_init();

long Zernike_n(long i);

long Zernike_m(long i);

double Zernike_value(long j, double r, double PA);

#endif
