// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_catalog.c
 * @brief Reading the install-time catalog of module commands (see milk_catalog.h); libc only, as it is
 * part of the thin bash client.
 */

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "milk_catalog.h"

#define CATALOG_PATHMAX 4096

typedef struct
{
    char *key;
    char *load;
} cat_ent_t;

static cat_ent_t *g_cat; /* sorted by key */
static size_t     g_ncat;
static int        g_loaded;

static int cmp_cat(const void *a, const void *b)
{
    return strcmp(((const cat_ent_t *) a)->key, ((const cat_ent_t *) b)->key);
}

static void catalog_add(char *line)
{
    char      *tab = strchr(line, '\t');
    cat_ent_t *q;

    line[strcspn(line, "\r\n")] = '\0';
    if (tab == NULL || tab == line || (q = realloc(g_cat, (g_ncat + 1) * sizeof(*q))) == NULL)
    {
        return;
    }
    g_cat                           = q;
    *tab                            = '\0';
    tab[1 + strcspn(tab + 1, "\t")] = '\0';
    g_cat[g_ncat].key               = strdup(line);
    g_cat[g_ncat].load              = strdup(tab + 1);
    if (g_cat[g_ncat].key != NULL && g_cat[g_ncat].load != NULL)
    {
        g_ncat++;
    }
}

static void catalog_load(void)
{
    const char    *inst = getenv("MILK_INSTALLDIR");
    char           dir[CATALOG_PATHMAX];
    DIR           *d;
    struct dirent *de;

    if (g_loaded)
    {
        return;
    }
    g_loaded = 1;
    if (inst == NULL || inst[0] == '\0')
    {
        return;
    }
    snprintf(dir, sizeof(dir), "%s/share/milk/catalog.d", inst);
    if ((d = opendir(dir)) == NULL)
    {
        return;
    }
    while ((de = readdir(d)) != NULL)
    {
        size_t l = strlen(de->d_name);
        char   p[CATALOG_PATHMAX];
        char   line[512];
        FILE  *fp;

        if (l < 5 || strcmp(de->d_name + l - 4, ".tsv") != 0)
        {
            continue;
        }
        snprintf(p, sizeof(p), "%s/%s", dir, de->d_name);
        if ((fp = fopen(p, "r")) == NULL)
        {
            continue;
        }
        while (fgets(line, sizeof(line), fp) != NULL)
        {
            catalog_add(line);
        }
        fclose(fp);
    }
    closedir(d);
    qsort(g_cat, g_ncat, sizeof(*g_cat), cmp_cat);
}

static const cat_ent_t *find_key(const char *key)
{
    cat_ent_t k = { .key = (char *) key, .load = NULL };

    catalog_load();
    return (key != NULL && g_ncat > 0) ? bsearch(&k, g_cat, g_ncat, sizeof(*g_cat), cmp_cat) : NULL;
}

int milk_catalog_has(const char *key)
{
    return find_key(key) != NULL;
}

const char *milk_catalog_module(const char *key)
{
    const cat_ent_t *e = find_key(key);

    return (e != NULL) ? e->load : NULL;
}

size_t milk_catalog_count(void)
{
    catalog_load();
    return g_ncat;
}

const char *milk_catalog_load_at(size_t i)
{
    catalog_load();
    return (i < g_ncat) ? g_cat[i].load : NULL;
}

const char *milk_catalog_short(const char *shortname)
{
    const char *hit = NULL;

    catalog_load();
    for (size_t i = 0; i < g_ncat; i++)
    {
        const char *dot = strchr(g_cat[i].key, '.');

        if (dot != NULL && strcmp(dot + 1, shortname) == 0)
        {
            if (hit != NULL)
            {
                return "?";
            }
            hit = g_cat[i].key;
        }
    }
    return hit;
}
