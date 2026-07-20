// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    info.c
 * @brief   Information about images
 *
 * Computes information about images
 *
 *
 *
 */

#define MODULE_SHORTNAME_DEFAULT "info"
#define MODULE_DESCRIPTION       "Image information and statistics"

#include "CommandLineInterface/CLIcore.h"
#include "cubeMatchMatrix.h"
#include "cubestats.h"
#include "image_stats.h"
#include "imagemon.h"
#include "improfile.h"
#include "stream_monproc.h"

int infoscreen_wcol;
int infoscreen_wrow; // window size

INIT_MODULE_LIB(info)

static errno_t init_module_CLI()
{
    cubeMatchMatrix_addCLIcmd();
    cubestats_addCLIcmd();

    CLIADDCMD_info__imagemon();
    CLIADDCMD_info__stream_monproc();

    image_stats_addCLIcmd();
    improfile_addCLIcmd();

    return RETURN_SUCCESS;
}






/* uses the repartition function F of the normal distribution law */
/* F(0) = 0.5 */
/* F(-0.1 * sig) = 0.460172162723 */
/* F(-0.2 * sig) = 0.420740290562 */
/* F(-0.3 * sig) = 0.382088577811 */
/* F(-0.4 * sig) = 0.34457825839 */
/* F(-0.5 * sig) = 0.308537538726 */
/* F(-0.6 * sig) = 0.27425311775 */
/* F(-0.7 * sig) = 0.241963652223 */
/* F(-0.8 * sig) = 0.211855398584 */
/* F(-0.9 * sig) = 0.184060125347 */
/* F(-1.0 * sig) = 0.158655253931 */
/* F(-1.1 * sig) = 0.135666060946 */
/* F(-1.2 * sig) = 0.115069670222 */
/* F(-1.3 * sig) = 0.0968004845855 */
/*
double background_photon_noise(
    const char *ID_name
)
{
    imageID        ID;
    uint32_t       naxes[2];
    double         value1, value2, value3, value;
    double        *array;
    uint64_t       nelements;

    ID = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];
    nelements = naxes[0] * naxes[1];

    array = (double *) malloc(naxes[1] * naxes[0] * sizeof(double));
    for(unsigned long jj = 0; jj < naxes[1]; jj++)
        for(unsigned long ii = 0; ii < naxes[0]; ii++)
        {
            array[jj * naxes[0] + ii] = data.image[ID].array.F[jj * naxes[0] +
ii];
        }

    quick_sort_double(array, nelements);


    // calculation using F(-0.9*sig) and F(-1.3*sig)
    value1 = array[(long)(0.184060125347 * naxes[1] * naxes[0])] -
array[(long)( 0.0968004845855 * naxes[1] * naxes[0])]; value1 /= (1.3 - 0.9);
    printf("(-1.3 -0.9) %f\n", value1);

    // calculation using F(-0.6*sig) and F(-1.3*sig)
    value2 = array[(long)(0.27425311775 * naxes[1] * naxes[0])] - array[(long)(
                 0.0968004845855 * naxes[1] * naxes[0])];
    value2 /= (1.3 - 0.6);
    printf("(-1.3 -0.6) %f\n", value2);

    // calculation using F(-0.3*sig) and F(-1.3*sig)
    value3 = array[(long)(0.382088577811 * naxes[1] * naxes[0])] -
array[(long)( 0.0968004845855 * naxes[1] * naxes[0])]; value3 /= (1.3 - 0.3);
    printf("(-1.3 -0.3) %f\n", value3);

    value = value3;

    free(array);

    return(value);
}
*/
