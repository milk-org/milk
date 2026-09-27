// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_data_loops.c
 * @brief Directed cycle detection and loop analysis for milk-CTRL
 *
 * Implements global directed cycle detection across streams, processes, and FPS.
 * Computes canonical cycle signatures, evaluates shared and exclusive resource
 * overlaps, aggregates loop health/rate metrics, and provides persistent loop naming.
 */

#include "overview_data_loops.h"
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "overview_data_loops_internal.h"

pthread_mutex_t      s_loop_names_mutex = PTHREAD_MUTEX_INITIALIZER;
ov_saved_loop_name_t s_saved_names[OV_MAX_SAVED_NAMES];
int                  s_nb_saved_names = 0;
int                  s_names_loaded   = 0;

/**
 * get_config_filepath - Resolve config path (~/.milk_loop_names.conf).
 * @buf: Output buffer
 * @sz:  Buffer capacity
 */
static void get_config_filepath(char *buf, size_t sz)
{
    const char *home = getenv("HOME");
    if (home != NULL && home[0] != '\0')
    {
        snprintf(buf, sz, "%s/.milk_loop_names.conf", home);
        return;
    }

    snprintf(buf, sz, "/tmp/milk_loop_names.conf");
}

/**
 * @brief Load custom loop names from disk into memory table.
 * @param[in,out] model System model containing detected loops
 */
void ov_loop_names_load(OV_MODEL *model)
{
    pthread_mutex_lock(&s_loop_names_mutex);
    char path[256];
    get_config_filepath(path, sizeof(path));

    FILE *fp = fopen(path, "r");
    if (fp == NULL)
    {
        /* Fallback check if user previously had ~/.milk/milk-CTRL_loops.conf */
        const char *home = getenv("HOME");
        if (home != NULL && home[0] != '\0')
        {
            char legacy_path[256];
            snprintf(legacy_path, sizeof(legacy_path), "%s/.milk/milk-CTRL_loops.conf", home);
            fp = fopen(legacy_path, "r");
        }
    }

    if (fp != NULL)
    {
        s_nb_saved_names = 0;
        char line[512];
        while (fgets(line, sizeof(line), fp) != NULL)
        {
            if (line[0] == '#' || line[0] == '\n' || line[0] == '\r')
            {
                continue;
            }

            uint64_t h = 0;
            char     nm[OV_LOOP_NAME_LEN];
            nm[0] = '\0';

            /* Parse: <hash_hex> <name> */
            if (sscanf(line, "%" PRIx64 " %47[^\r\n]", &h, nm) == 2)
            {
                if (s_nb_saved_names < OV_MAX_SAVED_NAMES)
                {
                    s_saved_names[s_nb_saved_names].hash = h;
                    strncpy(s_saved_names[s_nb_saved_names].name, nm, OV_LOOP_NAME_LEN - 1);
                    s_saved_names[s_nb_saved_names].name[OV_LOOP_NAME_LEN - 1] = '\0';
                    s_nb_saved_names++;
                }
            }
        }
        fclose(fp);
    }
    s_names_loaded = 1;

    /* Apply loaded names to existing model loops */
    if (model != NULL)
    {
        for (int i = 0; i < model->nb_loops; i++)
        {
            OV_LOOP *lp = &model->loops[i];
            for (int k = 0; k < s_nb_saved_names; k++)
            {
                if (s_saved_names[k].hash == lp->signature_hash)
                {
                    strncpy(lp->custom_name, s_saved_names[k].name, sizeof(lp->custom_name) - 1);
                    lp->custom_name[sizeof(lp->custom_name) - 1] = '\0';
                    lp->has_custom_name                          = 1;
                    strncpy(lp->name, lp->custom_name, sizeof(lp->name) - 1);
                    lp->name[sizeof(lp->name) - 1] = '\0';
                    break;
                }
            }
        }
    }
    pthread_mutex_unlock(&s_loop_names_mutex);
}

/**
 * @brief Save persistent custom loop names to disk (must hold s_loop_names_mutex).
 */
static void ov_loop_names_save_locked(void)
{
    char path[256];
    get_config_filepath(path, sizeof(path));

    FILE *fp = fopen(path, "w");
    if (fp == NULL)
    {
        return;
    }

    fprintf(fp, "# milk-CTRL loop custom names configuration\n");
    fprintf(fp, "# Format: <canonical_signature_hash_hex> <custom_name>\n");
    for (int i = 0; i < s_nb_saved_names; i++)
    {
        fprintf(fp, "%016" PRIx64 " %s\n", s_saved_names[i].hash, s_saved_names[i].name);
    }
    fclose(fp);
}

/**
 * @brief Save persistent custom loop names to disk.
 * @param[in] model System model containing detected loops (unused)
 */
void ov_loop_names_save(const OV_MODEL *model)
{
    (void) model;
    pthread_mutex_lock(&s_loop_names_mutex);
    ov_loop_names_save_locked();
    pthread_mutex_unlock(&s_loop_names_mutex);
}

/**
 * @brief Set a custom name for a loop and persist it.
 * @param[in,out] model    System model
 * @param[in]     loop_idx Index in model->loops[] (0..nb_loops-1)
 * @param[in]     new_name New human-readable name string
 * @return 0 on success, non-zero on error.
 */
int ov_loop_rename(OV_MODEL *model, int loop_idx, const char *new_name)
{
    if (model == NULL || loop_idx < 0 || loop_idx >= model->nb_loops || new_name == NULL)
    {
        return -1;
    }

    pthread_mutex_lock(&s_loop_names_mutex);
    OV_LOOP *lp = &model->loops[loop_idx];
    if (new_name[0] == '\0')
    {
        /* Clear custom name, restore auto_name */
        lp->has_custom_name = 0;
        lp->custom_name[0]  = '\0';
        strncpy(lp->name, lp->auto_name, sizeof(lp->name) - 1);
        lp->name[sizeof(lp->name) - 1] = '\0';

        /* Remove from saved names */
        for (int i = 0; i < s_nb_saved_names; i++)
        {
            if (s_saved_names[i].hash == lp->signature_hash)
            {
                for (int j = i; j < s_nb_saved_names - 1; j++)
                {
                    s_saved_names[j] = s_saved_names[j + 1];
                }
                s_nb_saved_names--;
                break;
            }
        }
    }
    else
    {
        strncpy(lp->custom_name, new_name, sizeof(lp->custom_name) - 1);
        lp->custom_name[sizeof(lp->custom_name) - 1] = '\0';
        lp->has_custom_name                          = 1;
        strncpy(lp->name, lp->custom_name, sizeof(lp->name) - 1);
        lp->name[sizeof(lp->name) - 1] = '\0';

        /* Update or add in saved table */
        int found = 0;
        for (int i = 0; i < s_nb_saved_names; i++)
        {
            if (s_saved_names[i].hash == lp->signature_hash)
            {
                strncpy(s_saved_names[i].name, lp->custom_name, OV_LOOP_NAME_LEN - 1);
                s_saved_names[i].name[OV_LOOP_NAME_LEN - 1] = '\0';
                found                                       = 1;
                break;
            }
        }
        if (!found && s_nb_saved_names < OV_MAX_SAVED_NAMES)
        {
            s_saved_names[s_nb_saved_names].hash = lp->signature_hash;
            strncpy(s_saved_names[s_nb_saved_names].name, lp->custom_name, OV_LOOP_NAME_LEN - 1);
            s_saved_names[s_nb_saved_names].name[OV_LOOP_NAME_LEN - 1] = '\0';
            s_nb_saved_names++;
        }
    }

    ov_loop_names_save_locked();
    pthread_mutex_unlock(&s_loop_names_mutex);
    return 0;
}

/**
 * ov_get_loop_name - Retrieve display name for a loop ID.
 * @model:   System model
 * @loop_id: 1-based loop ID
 *
 * Return: Display name string or fallback label.
 */
const char *ov_get_loop_name(const OV_MODEL *model, int loop_id)
{
    if (model == NULL || loop_id <= 0)
    {
        return "";
    }
    int idx = loop_id - 1;
    if (idx >= 0 && idx < model->nb_loops)
    {
        return model->loops[idx].name;
    }
    return "";
}

/**
 * ov_find_loop_by_id - Find loop array index from 1-based loop ID.
 * @model:   System model
 * @loop_id: 1-based loop ID
 *
 * Return: Array index in model->loops[], or -1 if not found.
 */
int ov_find_loop_by_id(const OV_MODEL *model, int loop_id)
{
    if (model == NULL || loop_id <= 0)
    {
        return -1;
    }
    for (int i = 0; i < model->nb_loops; i++)
    {
        if (model->loops[i].loop_id == loop_id)
        {
            return i;
        }
    }
    return -1;
}
