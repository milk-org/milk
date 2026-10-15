// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_shm_prop_internal.h
 * @brief Internal namespace handlers for live milk shared-memory entities.
 *
 * Role & Goal:
 *   Declares the per-namespace (`stream`, `fps`, `proc`, `seq`) backend
 *   functions called by the top-level dispatcher in `milk_shm_prop.c`.
 *   Keeping each namespace in its own translation unit keeps file and
 *   function lengths small and easy to audit.
 */

#ifndef MILK_SHM_PROP_INTERNAL_H
#define MILK_SHM_PROP_INTERNAL_H

#include <stddef.h>

/* Stream / local image namespace (@s., @stream.) */
int milk_shm_prop_get_stream(const char *name, const char *prop, char *out, size_t outsz);

int milk_shm_prop_set_stream(const char *name,
                             const char *prop,
                             const char *val,
                             const char *comment);

int milk_shm_prop_exists_stream(const char *name);

/* Function Parameter Structure namespace (@f., @fps.) */
int milk_shm_prop_get_fps(const char *name, const char *prop, char *out, size_t outsz);

int milk_shm_prop_set_fps(const char *name, const char *prop, const char *val);

int milk_shm_prop_exists_fps(const char *name, int check_running);

/* Processinfo telemetry namespace (@p., @proc., @procinfo.) */
int milk_shm_prop_get_proc(const char *name, const char *prop, char *out, size_t outsz);

int milk_shm_prop_set_proc(const char *name, const char *prop, const char *val);

int milk_shm_prop_exists_proc(const char *name, int check_running);

/* FPS Sequencer namespace (@q., @seq., @fpsseq.) */
int milk_shm_prop_get_seq(const char *name, const char *prop, char *out, size_t outsz);

int milk_shm_prop_set_seq(const char *name, const char *prop, const char *val);

int milk_shm_prop_exists_seq(const char *name, int check_running);

#endif /* MILK_SHM_PROP_INTERNAL_H */
