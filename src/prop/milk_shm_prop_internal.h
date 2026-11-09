// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_shm_prop_internal.h
 * @brief The namespace handlers of libmilkprop, and the helpers they share.
 *
 * Purpose:     milk_shm_prop.c parses an entity path and hands the name and property to the
 *              handler of its namespace; each namespace has a file of its own.
 * Does:        declares get, set and exists of each namespace: milk_shm_prop_stream.c (`s.` and
 *              `l.`), milk_shm_prop_fps.c (`f.`), milk_shm_prop_proc.c (`p.`); the stream
 *              directory and the exact number format of milk_shm_prop.c; the trigger mode names
 *              of milk_shm_prop_proc.c.
 * Used by:     the files of libmilkprop; milk_waitfor.c (`@waitfor`: the stream directory, the
 *              trigger mode names).
 * See also:    milk_shm_prop.h (the API).
 */

#ifndef MILK_SHM_PROP_INTERNAL_H
#define MILK_SHM_PROP_INTERNAL_H

#include <stddef.h>

/* Helpers of all the namespaces (milk_shm_prop.c) */
const char *milk_shm_prop_shmdir(void);

void milk_shm_prop_fmt_double(char *out, size_t outsz, double v);

/* Stream / local image namespace (@s., @stream., @l., @local.) */
int milk_shm_prop_get_stream(const char *name, const char *prop, char *out, size_t outsz);

int milk_shm_prop_set_stream(const char *name,
                             const char *prop,
                             const char *val,
                             const char *comment);

int milk_shm_prop_exists_stream(const char *name);

int milk_shm_prop_get_local_stream(const char *name, const char *prop, char *out, size_t outsz);

int milk_shm_prop_set_local_stream(const char *name,
                                   const char *prop,
                                   const char *val,
                                   const char *comment);

int milk_shm_prop_exists_local_stream(const char *name);

/* Function Parameter Structure namespace (@f., @fps.) */
int milk_shm_prop_get_fps(const char *name, const char *prop, char *out, size_t outsz);

int milk_shm_prop_set_fps(const char *name, const char *prop, const char *val);

int milk_shm_prop_exists_fps(const char *name, int check_running);

/* Processinfo telemetry namespace (@p., @proc.) */
int milk_shm_prop_get_proc(const char *name, const char *prop, char *out, size_t outsz);

int milk_shm_prop_set_proc(const char *name, const char *prop, const char *val);

int milk_shm_prop_exists_proc(const char *name, int check_running);

/** milk_trigmode_name() - name of processinfo trigger mode @mode (`semaphore`, `delay`, ...), or NULL. */
const char *milk_trigmode_name(long mode);

/** milk_trigmode_parse() - processinfo trigger mode named or numbered by @s, or -1. */
int milk_trigmode_parse(const char *s);

#endif /* MILK_SHM_PROP_INTERNAL_H */
