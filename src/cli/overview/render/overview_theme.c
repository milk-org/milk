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

/** ov_theme_pack - theme RGB as the cell colour the renderer stores. */
static uint32_t ov_theme_pack(ov_rgb_t c)
{
    return OV_COLOR_TRUE | ((c.r & 0xFFu) << 16) | ((c.g & 0xFFu) << 8) | (c.b & 0xFFu);
}

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
 * ov_theme_find_by_id - find a theme by identifier, display name or alias (case-insensitive).
 * @id: Identifier string (e.g. "dark", "night", "light", "accessible", "nordic").
 *
 * Return: Theme index or -1 if not found.
 */
int ov_theme_find_by_id(const char *id)
{
    static const struct
    {
        const char *alias;
        const char *id;
    } aliases[] = {
        { "red", "night" },
        { "obs-red", "night" },
        { "cvd", "accessible" },
        { "colorblind", "accessible" },
        { "paper", "light" },
        { "day", "light" },
        { "nord", "nordic" },
        { "drac", "dracula" },
        { "sol-dark", "solarized-dark" },
        { "soldark", "solarized-dark" },
        { "sol-light", "solarized-light" },
        { "sollight", "solarized-light" },
        { "monokai-pro", "monokai" },
        { "green", "matrix" },
        { "cyber", "matrix" },
        { "hacker", "matrix" },
    };

    if (id == NULL || id[0] == '\0')
    {
        return -1;
    }
    for (size_t a = 0; a < sizeof(aliases) / sizeof(aliases[0]); a++)
    {
        if (strcasecmp(id, aliases[a].alias) == 0)
        {
            id = aliases[a].id;
            break;
        }
    }
    for (int i = 0; i < ov_num_themes; i++)
    {
        if (strcasecmp(ov_themes[i]->id, id) == 0 || strcasecmp(ov_themes[i]->name, id) == 0)
        {
            return i;
        }
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

    ov__default_bg         = ov_theme_pack(ov_active_theme->bg_terminal);
    ov__mono_reverse_bg[0] = ov_theme_pack(ov_active_theme->bg_selected);
    ov__mono_reverse_bg[1] = ov_theme_pack(ov_active_theme->bg_frozen);

    /* Invalidate front delta cache so full screen repaints with new palette */
    ov_buf_force_clear();
}

/**
 * ov_theme_init - initialize theme subsystem from CLI arg or environment.
 * @preferred_theme: CLI argument or NULL.
 *
 * Defaults to the standard dark color theme (index 0).
 * Return: 0, or -1 when MILK_CTRL_THEME names no theme (the default is applied).
 */
int ov_theme_init(const char *preferred_theme)
{
    int idx    = -1;
    int env_rc = 0;

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
            idx    = ov_theme_find_by_id(env_theme);
            env_rc = (idx < 0) ? -1 : 0;
        }
    }

    /* 3. Default to index 0 (Default Dark) */
    if (idx < 0)
    {
        idx = 0;
    }

    ov_theme_set(idx);
    return env_rc;
}
