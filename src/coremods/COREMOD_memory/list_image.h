// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file list_image.h
 * @brief List images in memory
 */

#include "milk_runnable.h"

extern const MILK_RUNNABLE runnable_COREMOD_memory_listim;

errno_t memory_monitor(const char *termttyname);

errno_t init_list_image_ID_ncurses(const char *termttyname);

void close_list_image_ID_ncurses();

errno_t list_image_ID_ncurses();

errno_t list_image_ID_ofp(FILE *fo);

errno_t list_image_ID_ofp_simple(FILE *fo);

errno_t list_image_ID();

errno_t list_image_ID_with_opts(int         json_mode,
                                int         porcelain_mode,
                                int         quiet_mode,
                                const char *pattern);

errno_t list_image_ID_file(const char *fname);

errno_t list_variable_ID(const char *regexstr);

errno_t list_variable_ID_file(const char *fname);
