/** @file delete_variable.c
 */

#include "CommandLineInterface/CLIcore.h"
#include "variable_ID.h"

/* deletes a variable ID */
errno_t delete_variable_ID(const char *varname)
{
    imageID ID;

    ID = variable_ID(varname);
    if(ID != -1)
    {
        data.variable[ID].used = 0;
        /*      free(data.variable[ID].name);*/
    }
    else
        fprintf(stderr,
                "%c[%d;%dm WARNING: variable %s does not exist [ %s  %s  %d ] "
                "%c[%d;m\n",
                (char) 27,
                1,
                31,
                varname,
                __FILE__,
                __func__,
                __LINE__,
                (char) 27,
                0);

    return RETURN_SUCCESS;
}
