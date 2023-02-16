/** @file naninf2zero.c
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

/* set all nan and inf pixel values to zero */
int basic_naninf2zero(const char *ID_name)
{
    imageID  ID;
    uint32_t naxes[2];
    long     cnt = 0;

    ID       = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    for(uint32_t jj = 0; jj < naxes[1]; jj++)
        for(uint32_t ii = 0; ii < naxes[0]; ii++)
        {
            if(!(fabs(data.image[ID].array.F[jj * naxes[0] + ii]) < HUGE_VAL))
            {
                data.image[ID].array.F[jj * naxes[0] + ii] = 0.0;
                cnt++;
            }
        }

    printf("%ld values replaced\n", cnt);

    return (0);
}
