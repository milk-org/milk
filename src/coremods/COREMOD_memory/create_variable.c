// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    create_variable.c
 * @brief   create variables
 */

#include <string.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#    include "COREMOD_memory/COREMOD_memory.h"
#else
#    include "libmilkdata/milkdata.h"
#endif
#include "image_ID.h"
#include "COREMOD_memory/imageID.h"
#include "variable_ID.h"
#include "libmilkdata/milkdata.h" /* milk_table_bump() */

/**
 * resolve_or_alloc_variable_ID() - slot of variable @name, allocating it if it does not exist yet
 * @name: Variable name
 *
 * An existing variable is returned at once. Only a new name is checked against image names (a
 * variable may not shadow an image) and given a free slot.
 *
 * Return: variable ID, or -1 (NULL name, name used by an image, or table full)
 */
static variableID resolve_or_alloc_variable_ID(const char *name)
{
    if (name == NULL)
    {
        return -1;
    }

    variableID ID = variable_ID(name);
    if (ID != -1)
    {
        return ID;
    }

    if (imgid_exists(name))
    {
        printf("ERROR: cannot create variable \"%s\": name already used as an "
               "image\n",
               name);
        return -1;
    }

    ID = next_avail_variable_ID();
    if (ID == -1)
    {
        printf("ERROR: cannot create variable \"%s\": variable table full (max %ld variables)\n",
               name, dcnvar);
        return -1;
    }

    dcvar[ID].used = 1;
    variable_ID_note_used(ID);
    milk_table_bump();
    snprintf(dcvar[ID].name, sizeof(dcvar[ID].name), "%s", name);
    return ID;
}

/* creates floating point variable */
variableID create_variable_ID(const char *name, double value)
{
    variableID ID = resolve_or_alloc_variable_ID(name);
    if (ID != -1)
    {
        dcvar[ID].type    = 0; /** floating point double */
        dcvar[ID].value.f = value;
    }
    return ID;
}

/* creates long variable */
variableID create_variable_long_ID(const char *name, long value)
{
    variableID ID = resolve_or_alloc_variable_ID(name);
    if (ID != -1)
    {
        dcvar[ID].type    = 1; /** long */
        dcvar[ID].value.l = value;
    }
    return ID;
}

/* creates string variable */
variableID create_variable_string_ID(const char *name, const char *value)
{
    variableID ID = resolve_or_alloc_variable_ID(name);
    if (ID != -1)
    {
        dcvar[ID].type = 2; /** string */
        snprintf(dcvar[ID].value.s, sizeof(dcvar[ID].value.s), "%s", (value != NULL) ? value : "");
    }
    return ID;
}
