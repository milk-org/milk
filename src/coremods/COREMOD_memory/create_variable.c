/**
 * @file    create_variable.c
 * @brief   create variables
 */

#include <string.h>

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "libmilkdata/milkdata.h"
#endif
#include "image_ID.h"
#include "variable_ID.h"

/* creates floating point variable */
variableID create_variable_ID(const char *name, double value)
{
    variableID ID;
    long       i1, i2;


    ID = -1;

    i1 = image_ID(name, dcimg, dcnimg);

    i2 = variable_ID(name);

    if(i1 != -1)
    {
        printf(
            "ERROR: cannot create variable \"%s\": name already used as an "
            "image\n",
            name);
    }
    else
    {
        if(i2 != -1)
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
        strcpy(dcvar[ID].name, name);
        dcvar[ID].value.f = value;
    }
    return ID;
}

/* creates long variable */
variableID create_variable_long_ID(const char *name, long value)
{
    variableID ID;
    long       i1, i2;

    ID = -1;
    i1 = image_ID(name, dcimg, dcnimg);
    i2 = variable_ID(name);

    if(i1 != -1)
    {
        printf(
            "ERROR: cannot create variable \"%s\": name already used as an "
            "image\n",
            name);
    }
    else
    {
        if(i2 != -1)
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
        strcpy(dcvar[ID].name, name);
        dcvar[ID].value.l = value;
    }

    return ID;
}

/* creates long variable */
variableID create_variable_string_ID(const char *name, const char *value)
{
    variableID ID;
    long       i1, i2;

    ID = -1;
    i1 = image_ID(name, dcimg, dcnimg);
    i2 = variable_ID(name);

    if(i1 != -1)
    {
        printf(
            "ERROR: cannot create variable \"%s\": name already used as an "
            "image\n",
            name);
    }
    else
    {
        if(i2 != -1)
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
        strcpy(dcvar[ID].name, name);
        strcpy(dcvar[ID].value.s, value);
    }

    return ID;
}
