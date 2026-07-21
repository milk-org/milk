// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file processinfo_internal.h
 * @brief Processinfo internal module
 */

#ifndef PROCESSINFO_INTERNAL_H
#define PROCESSINFO_INTERNAL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "milkDebugTools.h"
#include "ImageStreamIO/ImageStruct.h"

#include "processinfo.h"

/* Global process-info list (defined in processinfo_globals.c) */
extern PROCESSINFOLIST *pinfolist;

#endif // PROCESSINFO_INTERNAL_H
