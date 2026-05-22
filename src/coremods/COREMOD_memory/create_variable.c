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

/* creates floating point variable */
variableID create_variable_ID(const char *name, double value)
{
    variableID ID;
    long       i2;

    ID = -1;

    i2 = variable_ID(name);

    if (imgid_exists(name))
    {
        printf("ERROR: cannot create variable \"%s\": name already used as an "
               "image\n",
               name);
    }
    else
    {
        if (i2 != -1)
        {
            //	  printf("Warning : variable name \"%s\" is already in use\n",name);
            ID = i2;
        }
        else
        {
            ID = next_avail_variable_ID();
        }

        dcvar[ID].used = 1;
        dcvar[ID].type = 0; /** floating point double */
        snprintf(dcvar[ID].name, sizeof(dcvar[ID].name), "%s", name);
        dcvar[ID].value.f = value;
    }
    return ID;
}

/* creates long variable */
variableID create_variable_long_ID(const char *name, long value)
{
    variableID ID;
    long       i2;

    ID = -1;
    i2 = variable_ID(name);

    if (imgid_exists(name))
    {
        printf("ERROR: cannot create variable \"%s\": name already used as an "
               "image\n",
               name);
    }
    else
    {
        if (i2 != -1)
        {
            //	  printf("Warning : variable name \"%s\" is already in use\n",name);
            ID = i2;
        }
        else
        {
            ID = next_avail_variable_ID();
        }

        dcvar[ID].used = 1;
        dcvar[ID].type = 1; /** long */
        snprintf(dcvar[ID].name, sizeof(dcvar[ID].name), "%s", name);
        dcvar[ID].value.l = value;
    }

    return ID;
}

/* creates long variable */
variableID create_variable_string_ID(const char *name, const char *value)
{
    variableID ID;
    long       i2;

    ID = -1;
    i2 = variable_ID(name);

    if (imgid_exists(name))
    {
        printf("ERROR: cannot create variable \"%s\": name already used as an "
               "image\n",
               name);
    }
    else
    {
        if (i2 != -1)
        {
            //	  printf("Warning : variable name \"%s\" is already in use\n",name);
            ID = i2;
        }
        else
        {
            ID = next_avail_variable_ID();
        }

        dcvar[ID].used = 1;
        dcvar[ID].type = 2; /** string */
        snprintf(dcvar[ID].name, sizeof(dcvar[ID].name), "%s", name);
        snprintf(dcvar[ID].value.s, sizeof(dcvar[ID].value.s), "%s", value);
    }

    return ID;
}
