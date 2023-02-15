/** @file extrapolate_nearestpixel.c
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

#include "imcontract.h"

imageID basic_2Dextrapolate_nearestpixel(const char *__restrict IDin_name,
        const char *__restrict IDmask_name,
        const char *__restrict IDout_name)
{
    DEBUG_TRACE_FSTART();

    imageID IDin, IDmask, IDout;
    long    ii, jj, ii1, jj1, k;
    double  bdist, dist;
    long    naxes[2];

    long *maskii = NULL;
    long *maskjj = NULL;
    long  NBmaskpts;

    long IDmask1;

    IDin   = image_ID(IDin_name);
    IDmask = image_ID(IDmask_name);

    list_image_ID();
    IDmask1 = image_ID("_mask1");
    if(IDmask1 != -1)
    {
        printf("USING MASK\n");
    }

    naxes[0] = data.image[IDin].md[0].size[0];
    naxes[1] = data.image[IDin].md[0].size[1];

    NBmaskpts = 0;
    for(ii = 0; ii < naxes[0]; ii++)
        for(jj = 0; jj < naxes[1]; jj++)
            if(data.image[IDmask].array.F[jj * naxes[0] + ii] > 0.5)
            {
                NBmaskpts++;
            }

    maskii = (long *) malloc(sizeof(long) * NBmaskpts);
    if(maskii == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc error");
        exit(0);
    }
    maskii[0] = 0; // avoids warning about unused maskii

    maskjj = (long *) malloc(sizeof(long) * NBmaskpts);
    if(maskjj == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc error");
        exit(0);
    }
    maskjj[0] = 0; // avoids warning about unused maskjj

    NBmaskpts = 0;
    for(ii = 0; ii < naxes[0]; ii++)
        for(jj = 0; jj < naxes[1]; jj++)
            if(data.image[IDmask].array.F[jj * naxes[0] + ii] > 0.5)
            {
                maskii[NBmaskpts] = ii;
                maskjj[NBmaskpts] = jj;
                NBmaskpts++;
            }

    create_2Dimage_ID(IDout_name, naxes[0], naxes[1], &IDout);
    printf("imout = %s\n", IDout_name);
    printf("\n");

    for(ii = 0; ii < naxes[0]; ii++)
    {
        printf("\r%ld / %ld  ", ii, naxes[0]);
        fflush(stdout);

        for(jj = 0; jj < naxes[1]; jj++)
        {
            /*if(IDmask1==-1)
                OKpix = 1;
            else
            {
                if(data.image[IDmask1].array.F[jj*naxes[1]+ii]>0.5)
                    OKpix = 1;
                else
                    OKpix = 0;
            }*/
            bdist = (double)(naxes[0] + naxes[1]);
            bdist = bdist * bdist;
            for(k = 0; k < NBmaskpts; k++)
            {
                ii1 = maskii[k];
                jj1 = maskjj[k];
                dist =
                    1.0 * ((ii1 - ii) * (ii1 - ii) + (jj1 - jj) * (jj1 - jj));
                if(dist < bdist)
                {
                    bdist = dist;
                    data.image[IDout].array.F[jj * naxes[0] + ii] =
                        data.image[IDin].array.F[jj1 * naxes[0] + ii1];
                }
            }
        }
    }

    printf("\n");

    free(maskii);
    free(maskjj);

    DEBUG_TRACE_FEXIT();
    return (IDout);
}
