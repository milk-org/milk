// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_cmd_index.c
 * @brief Name rules, hash index of data.cmd[] and registry generation (see milk_cmd_index.h).
 */

#include <ctype.h>
#include <stdio.h>
#include <string.h>

#include "CLIcore.h"
#include "milk_catalog.h"
#include "milk_cmd_index.h"
#include "milk_entity_ns.h"
#include "milk_lexicon.h"

#define CMD_INDEX_SIZE 4096 /* power of two, > 2 * DATA_NB_MAX_COMMAND */

static int      g_slot[CMD_INDEX_SIZE]; /* data.cmd index + 1, 0 = empty */
static uint32_t g_gen          = 0;
static int      g_allow_milk   = 0;
static uint32_t g_indexed_upto = 0;

/** fnv1a() - FNV-1a hash of @s. */
static uint32_t fnv1a(const char *s)
{
    uint32_t h = 2166136261u;
    for (; *s != '\0'; s++)
    {
        h = (h ^ (unsigned char) *s) * 16777619u;
    }
    return h;
}

void milk_reserve_milk_shortname(int on)
{
    g_allow_milk = on;
}

int milk_shortname_check(const char *name, int quiet)
{
    const char *why = NULL;
    size_t      n   = (name != NULL) ? strlen(name) : 0;

    if (n < 2 || n > 16 || !islower((unsigned char) name[0]))
    {
        why = "must match [a-z][a-z0-9]{1,15}";
    }
    for (size_t i = 1; why == NULL && i < n; i++)
    {
        if (!islower((unsigned char) name[i]) && !isdigit((unsigned char) name[i]))
        {
            why = "must match [a-z][a-z0-9]{1,15}";
        }
    }
    if (why == NULL && milk_lexicon_keyword(name) != NULL)
    {
        why = "is a language keyword";
    }
    if (why == NULL && milk_entity_ns_lookup(name, n) != MILK_NS_NONE)
    {
        why = "is an entity namespace";
    }
    if (why == NULL && strcmp(name, "milk") == 0 && !g_allow_milk)
    {
        why = "is reserved for the built-in module";
    }
    if (why != NULL && !quiet)
    {
        fprintf(stderr, "milk: module short name '%s' %s; its commands are not registered\n",
                (name != NULL) ? name : "", why);
    }
    return (why == NULL) ? 0 : -1;
}

/** valid_part() - 1 if @s[0..n) is a non-empty [A-Za-z0-9_] word. */
static int valid_part(const char *s, size_t n)
{
    if (n == 0)
    {
        return 0;
    }
    for (size_t i = 0; i < n; i++)
    {
        if (!isalnum((unsigned char) s[i]) && s[i] != '_')
        {
            return 0;
        }
    }
    return 1;
}

int milk_cmd_key_check(const char *key, int midx)
{
    if (key == NULL || key[0] == '\0')
    {
        return -1;
    }
    if (midx >= 0)
    {
        const char *dot = strchr(key, '.');
        if (milk_shortname_check(data.module[midx].shortname, 1) != 0)
        {
            return -1; /* the module short name was already reported by RegisterModule */
        }
        if (dot == NULL || !valid_part(dot + 1, strlen(dot + 1)))
        {
            fprintf(stderr, "milk: command '%s' must be <module>.<name> with [A-Za-z0-9_] names\n",
                    key);
            return -1;
        }
    }
    int prev = milk_cmd_lookup(key);
    if (prev >= 0)
    {
        fprintf(stderr,
                "milk: command '%s' is already registered (module %s); ignoring the new one\n", key,
                data.cmd[prev].module);
        return -1;
    }
    return 0;
}

int milk_cmd_lookup(const char *key)
{
    if (key == NULL)
    {
        return -1;
    }
    /* commands registered without milk_cmd_index_add() (none expected) are picked up lazily */
    while (g_indexed_upto < data.NBcmd)
    {
        milk_cmd_index_add(g_indexed_upto);
    }
    for (uint32_t h = fnv1a(key) & (CMD_INDEX_SIZE - 1), probes = 0; probes < CMD_INDEX_SIZE;
         h = (h + 1) & (CMD_INDEX_SIZE - 1), probes++)
    {
        int c = g_slot[h] - 1;
        if (c < 0)
        {
            return -1;
        }
        if ((unsigned int) c < data.NBcmd && strcmp(data.cmd[c].key, key) == 0)
        {
            return c;
        }
    }
    return -1;
}

void milk_cmd_index_add(unsigned int cmdi)
{
    if (cmdi < g_indexed_upto)
    {
        return;
    }
    for (uint32_t h = fnv1a(data.cmd[cmdi].key) & (CMD_INDEX_SIZE - 1), probes = 0;
         probes < CMD_INDEX_SIZE; h = (h + 1) & (CMD_INDEX_SIZE - 1), probes++)
    {
        if (g_slot[h] == 0)
        {
            g_slot[h] = (int) cmdi + 1;
            break;
        }
    }
    g_indexed_upto = cmdi + 1;
    if (strchr(data.cmd[cmdi].key, '.') != NULL && !milk_catalog_has(data.cmd[cmdi].key))
    {
        g_gen++; /* shells define functions for the catalog only: they resync on change */
    }
}

uint32_t milk_registry_gen(void)
{
    return g_gen;
}
