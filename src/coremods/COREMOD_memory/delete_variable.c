/**
 * @file delete_variable.c
 * @brief Delete variable module
 */

/** @file delete_variable.c
 */

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#    include "COREMOD_memory/COREMOD_memory.h"
#else
#    include "libmilkdata/milkdata.h"
#endif
#include "variable_ID.h"

/* deletes a variable ID */
errno_t delete_variable_ID(const char *varname)
{
    imageID ID;

    ID = variable_ID(varname);
    if (ID != -1)
    {
        dcvar[ID].used = 0;
        /*      free(dcvar[ID].name);*/
    }
    else
    {
        fprintf(stderr,
                "%c[%d;%dm WARNING: variable %s does not exist [ %s  %s  %d ] "
                "%c[%d;m\n",
                (char) 27, 1, 31, varname, __FILE__, __func__, __LINE__, (char) 27, 0);
    }

    return RETURN_SUCCESS;
}
