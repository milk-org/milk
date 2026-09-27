// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_theme.c
 * @brief Theme definitions and palette management for milk-CTRL.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

#include "overview_theme_internal.h"
#include "overview_ansi.h"

static const ov_theme_t *const ov_themes[] = {
    &ov_theme_dark,    &ov_theme_night,   &ov_theme_accessible,     &ov_theme_light,
    &ov_theme_nordic,  &ov_theme_dracula, &ov_theme_solarized_dark, &ov_theme_solarized_light,
    &ov_theme_monokai, &ov_theme_matrix,
};

static const int ov_num_themes = (int) (sizeof(ov_themes) / sizeof(ov_themes[0]));

static int        ov_theme_active_idx = 0;
const ov_theme_t *ov_active_theme     = &ov_theme_dark;

/* Public API implementation */

/**
 * ov_theme_count - return total number of registered themes.
 *
 * Return: Number of themes available in ov_themes array.
 */
int ov_theme_count(void)
{
    return ov_num_themes;
}

/**
 * ov_theme_get - retrieve a theme by index.
 * @index: Theme index (0 to ov_theme_count() - 1).
 *
 * Return: Pointer to theme or NULL if invalid.
 */
const ov_theme_t *ov_theme_get(int index)
{
    if (index < 0 || index >= ov_num_themes)
    {
        return NULL;
    }
    return ov_themes[index];
}

/**
 * ov_theme_get_active - retrieve active theme.
 *
 * Return: Pointer to currently active theme.
 */
const ov_theme_t *ov_theme_get_active(void)
{
    return ov_active_theme;
}

/**
 * ov_theme_get_active_index - retrieve active theme index.
 *
 * Return: Index of active theme.
 */
int ov_theme_get_active_index(void)
{
    return ov_theme_active_idx;
}

/**
 * ov_theme_find_by_id - find a theme by identifier or name.
 * @id: Identifier string (e.g. "dark", "night", "light", "accessible", "nordic").
 *
 * Return: Theme index or -1 if not found.
 */
int ov_theme_find_by_id(const char *id)
{
    if (id == NULL || id[0] == '\0')
    {
        return -1;
    }

    for (int i = 0; i < ov_num_themes; i++)
    {
        if (strcasecmp(ov_themes[i]->id, id) == 0 || strcasecmp(ov_themes[i]->name, id) == 0)
        {
            return i;
        }
    }

    /* Aliases */
    if (strcasecmp(id, "red") == 0 || strcasecmp(id, "obs-red") == 0)
    {
        return 1; /* night */
    }
    if (strcasecmp(id, "cvd") == 0 || strcasecmp(id, "colorblind") == 0)
    {
        return 2; /* accessible */
    }
    if (strcasecmp(id, "paper") == 0 || strcasecmp(id, "day") == 0)
    {
        return 3; /* light */
    }
    if (strcasecmp(id, "nord") == 0)
    {
        return 4; /* nordic */
    }
    if (strcasecmp(id, "drac") == 0)
    {
        return 5; /* dracula */
    }
    if (strcasecmp(id, "sol-dark") == 0 || strcasecmp(id, "soldark") == 0)
    {
        return 6; /* solarized-dark */
    }
    if (strcasecmp(id, "sol-light") == 0 || strcasecmp(id, "sollight") == 0)
    {
        return 7; /* solarized-light */
    }
    if (strcasecmp(id, "monokai-pro") == 0)
    {
        return 8; /* monokai */
    }
    if (strcasecmp(id, "green") == 0 || strcasecmp(id, "cyber") == 0 ||
        strcasecmp(id, "hacker") == 0)
    {
        return 9; /* matrix */
    }

    return -1;
}

/**
 * ov_theme_set - set the active theme by index.
 * @index: Target theme index.
 */
void ov_theme_set(int index)
{
    if (index < 0 || index >= ov_num_themes)
    {
        return;
    }

    ov_theme_active_idx = index;
    ov_active_theme     = ov_themes[index];

    ov_rgb_t tbg = ov_active_theme->bg_terminal;
    ov__default_bg =
        OV_COLOR_TRUE | ((tbg.r & 0xFF) << 16) | ((tbg.g & 0xFF) << 8) | (tbg.b & 0xFF);

    /* Invalidate front delta cache so full screen repaints with new palette */
    ov_buf_force_clear();
}

/**
 * ov_theme_cycle - cycle to the next theme.
 */
void ov_theme_cycle(void)
{
    int next = (ov_theme_active_idx + 1) % ov_num_themes;
    ov_theme_set(next);
}

/**
 * ov_theme_init - initialize theme subsystem from CLI arg or environment.
 * @preferred_theme: CLI argument or NULL.
 *
 * Defaults to the standard dark color theme (index 0).
 */
void ov_theme_init(const char *preferred_theme)
{
    int idx = -1;

    /* 1. CLI argument override */
    if (preferred_theme != NULL && preferred_theme[0] != '\0')
    {
        idx = ov_theme_find_by_id(preferred_theme);
    }

    /* 2. Environment variable MILK_CTRL_THEME */
    if (idx < 0)
    {
        const char *env_theme = getenv("MILK_CTRL_THEME");
        if (env_theme != NULL && env_theme[0] != '\0')
        {
            idx = ov_theme_find_by_id(env_theme);
        }
    }

    /* 3. Default to index 0 (Default Dark) */
    if (idx < 0)
    {
        idx = 0;
    }

    ov_theme_active_idx = idx;
    ov_active_theme     = ov_themes[idx];

    ov_rgb_t tbg = ov_active_theme->bg_terminal;
    ov__default_bg =
        OV_COLOR_TRUE | ((tbg.r & 0xFF) << 16) | ((tbg.g & 0xFF) << 8) | (tbg.b & 0xFF);
}
