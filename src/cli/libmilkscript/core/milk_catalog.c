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

/**
 * cmp_cat - Compare two catalog entries by command key for `qsort` and `bsearch`
 * @a: Pointer to first `cat_ent_t` element
 * @b: Pointer to second `cat_ent_t` element
 *
 * Orders catalog entries lexicographically by their `key` field.
 *
 * Return: Negative, zero, or positive integer as `a->key` is less than, equal to,
 *         or greater than `b->key`
 */
static int cmp_cat(const void *a, const void *b)
{
    return strcmp(((const cat_ent_t *) a)->key, ((const cat_ent_t *) b)->key);
}

/**
 * catalog_add - Parse a TSV catalog line and append its command entry to `g_cat`
 * @line: Writable line buffer (`<key>\t<module_loadname>[\t<description>]`)
 *
 * Splits @line at the first tab character and stores duplicated `key` and `load`
 * strings in the dynamically grown `g_cat` array.
 *
 * Return: void
 */
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

/**
 * catalog_load - Lazily load and sort all `.tsv` files from `$MILK_INSTALLDIR/share/milk/catalog.d`
 *
 * Reads every per-module `.tsv` file generated at install time into `g_cat` on first
 * invocation and sorts the array by command key for binary search lookups.
 *
 * Return: void
 */
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

/**
 * find_key - Look up a command key in the sorted install-time catalog
 * @key: Full `<module>.<cmd>` key to search for
 *
 * Ensures the catalog is loaded into memory and performs a binary search over `g_cat`.
 *
 * Return: Pointer to the matching `cat_ent_t`, or NULL if not found
 */
static const cat_ent_t *find_key(const char *key)
{
    cat_ent_t k = { .key = (char *) key, .load = NULL };

    catalog_load();
    return (key != NULL && g_ncat > 0) ? bsearch(&k, g_cat, g_ncat, sizeof(*g_cat), cmp_cat) : NULL;
}

/**
 * milk_catalog_has - Check whether a command key is present in the install-time catalog
 * @key: Full `<module>.<cmd>` command name to query
 *
 * Used by both the thin bash client and the engine to test whether a command belongs
 * to an installed module without loading shared libraries first.
 *
 * Return: 1 if @key exists in the catalog, 0 otherwise
 */
int milk_catalog_has(const char *key)
{
    return find_key(key) != NULL;
}

/**
 * milk_catalog_module - Return the module load name that provides a catalogued command
 * @key: Full `<module>.<cmd>` command name to query
 *
 * Looks up @key in the install-time catalog so the engine can autoload the owning
 * module on first use.
 *
 * Return: Module load name string (e.g. `"milkfft"`), or NULL if @key is not catalogued
 */
const char *milk_catalog_module(const char *key)
{
    const cat_ent_t *e = find_key(key);

    return (e != NULL) ? e->load : NULL;
}

/**
 * milk_catalog_count - Return the total number of commands in the install-time catalog
 *
 * Loads the catalog if not already loaded and returns the entry count.
 *
 * Return: Number of catalogued commands
 */
size_t milk_catalog_count(void)
{
    catalog_load();
    return g_ncat;
}

/**
 * milk_catalog_load_at - Return the module load name of the catalog entry at index @i
 * @i: Zero-based index into the sorted catalog array
 *
 * Allows callers (such as `@module list --all`) to iterate over all installed modules
 * represented in the catalog.
 *
 * Return: Module load name at index @i, or NULL if @i is out of bounds
 */
const char *milk_catalog_load_at(size_t i)
{
    catalog_load();
    return (i < g_ncat) ? g_cat[i].load : NULL;
}

/**
 * milk_catalog_short - Resolve an unqualified command short name against the catalog
 * @shortname: Command suffix after the module dot (e.g. `"loadfits"`)
 *
 * Scans the catalog for `<module>.<shortname>` matches to provide helpful diagnostics
 * when a user invokes a bare or legacy `@shortname` spelling.
 *
 * Return: Full `<module>.<shortname>` key if unique, `"?"` if ambiguous across modules,
 *         or NULL if no catalogued command has that short name
 */
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
