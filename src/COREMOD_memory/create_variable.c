/**
 * @file    create_variable.c
 * @brief   create variables 
 */


#include "CommandLineInterface/CLIcore.h"
#include "variable_ID.h"
#include "image_ID.h"





/* creates floating point variable */
variableID create_variable_ID(
    const char *name,
    double value
)
{
    variableID ID;
    long i1, i2;

    //printf("TEST   %s  %ld   %ld %ld ================== \n", __FILE__, __LINE__, data.NB_MAX_IMAGE, data.NB_MAX_VARIABLE);

    ID = -1;
    //printf("TEST   %s  %ld   %ld %ld ================== \n", __FILE__, __LINE__, data.NB_MAX_IMAGE, data.NB_MAX_VARIABLE);

    i1 = image_ID(name);
    //printf("TEST   %s  %ld   %ld %ld ================== \n", __FILE__, __LINE__, data.NB_MAX_IMAGE, data.NB_MAX_VARIABLE);


    i2 = variable_ID(name);
    //    printf("TEST   %s  %ld   %ld %ld ================== \n", __FILE__, __LINE__, data.NB_MAX_IMAGE, data.NB_MAX_VARIABLE);

    if(i1 != -1)
    {
        printf("ERROR: cannot create variable \"%s\": name already used as an image\n",
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

        data.variable[ID].used = 1;
        data.variable[ID].type = 0; /** floating point double */
        strcpy(data.variable[ID].name, name);
        data.variable[ID].value.f = value;

    }
    //    printf("TEST   %s  %ld   %ld %ld ================== \n", __FILE__, __LINE__, data.NB_MAX_IMAGE, data.NB_MAX_VARIABLE);
    return ID;
}



/* creates long variable */
variableID create_variable_long_ID(
    const char *name,
    long value
)
{
    variableID ID;
    long i1, i2;

    ID = -1;
    i1 = image_ID(name);
    i2 = variable_ID(name);

    if(i1 != -1)
    {
        printf("ERROR: cannot create variable \"%s\": name already used as an image\n",
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

        data.variable[ID].used = 1;
        data.variable[ID].type = 1; /** long */
        strcpy(data.variable[ID].name, name);
        data.variable[ID].value.l = value;

    }

    return ID;
}



/* creates long variable */
variableID create_variable_string_ID(
    const char *name,
    const char *value
)
{
    variableID ID;
    long i1, i2;

    ID = -1;
    i1 = image_ID(name);
    i2 = variable_ID(name);

    if(i1 != -1)
    {
        printf("ERROR: cannot create variable \"%s\": name already used as an image\n",
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

        data.variable[ID].used = 1;
        data.variable[ID].type = 2; /** string */
        strcpy(data.variable[ID].name, name);
        strcpy(data.variable[ID].value.s, value);
    }

    return ID;
}











