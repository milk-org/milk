// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef OVERVIEW_DATA_LOOPS_INTERNAL_H
#define OVERVIEW_DATA_LOOPS_INTERNAL_H

#include <inttypes.h>
#include <pthread.h>
#include "overview_data_loops.h"

#define OV_MAX_SAVED_NAMES 128

typedef struct
{
    uint64_t hash;
    char     name[OV_LOOP_NAME_LEN];
} ov_saved_loop_name_t;

extern pthread_mutex_t      s_loop_names_mutex;
extern ov_saved_loop_name_t s_saved_names[OV_MAX_SAVED_NAMES];
extern int                  s_nb_saved_names;
extern int                  s_names_loaded;

int is_valid_loop_edge(
    const OV_MODEL *m,
    const OV_EDGE  *e,
    sg_mode_t       mode);

int register_cycle(
    OV_MODEL  *model,
    const int *path,
    int        path_len);

#endif /* OVERVIEW_DATA_LOOPS_INTERNAL_H */
