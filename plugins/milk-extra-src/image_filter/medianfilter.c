/** @file medianfilter.c
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"

imageID median_filter(const char *__restrict ID_name,
                      const char *__restrict out_name,
                      int filter_size)
{
    imageID ID;
    imageID IDout;
    float  *array;
    long    ii, jj;
    long    naxes[2];
    int     i, j;

    /*  printf("Median filter...");
        fflush(stdout);*/
    save_fl_fits(ID_name, "mf_in.fits");

    array = (float *) malloc((2 * filter_size + 1) * (2 * filter_size + 1) *
                             sizeof(float));
    if(array == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    ID       = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];
    printf("name = %s, ID = %ld, Size = %ld %ld (%d)\n",
           ID_name,
           ID,
           naxes[0],
           naxes[1],
           filter_size);
    fflush(stdout);
    copy_image_ID(ID_name, out_name, 0);
    IDout = image_ID(out_name);

    for(jj = filter_size; jj < naxes[1] - filter_size; jj++)
        for(ii = filter_size; ii < naxes[0] - filter_size; ii++)
        {
            for(i = 0; i < (2 * filter_size + 1); i++)
                for(j = 0; j < (2 * filter_size + 1); j++)
                {
                    array[i * (2 * filter_size + 1) + j] =
                        data.image[ID]
                        .array.F[(jj - filter_size + j) * naxes[0] +
                                                        (ii - filter_size + i)];
                }
            quick_sort_float(array,
                             (2 * filter_size + 1) * (2 * filter_size + 1));
            data.image[IDout].array.F[jj * naxes[0] + ii] =
                array[((2 * filter_size + 1) * (2 * filter_size + 1) - 1) / 2];
        }
    free(array);

    save_fl_fits(out_name, "mf_out.fits");

    /*  printf("Done\n");
        fflush(stdout);*/

    return IDout;
}
