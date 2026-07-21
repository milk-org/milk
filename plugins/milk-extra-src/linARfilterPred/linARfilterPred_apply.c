// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "linARfilterPred_internal.h"

/* =============================================================================================== */
/*                                                                                                 */
/* 4. APPLY PREDICTIVE FILTER                                                                      */
/*                                                                                                 */
/* =============================================================================================== */
/* =============================================================================================== */

//
// real-time apply predictive filter
//
// filter can be smaller than input telemetry but needs to include contiguous pixels at the beginning of the input telemetry
//
imageID LINARFILTERPRED_Apply_LinPredictor_RT(const char *IDfilt_name,
                                              const char *IDin_name,
                                              const char *IDout_name)
{
    imageID   IDout;
    imageID   IDin;
    imageID   IDfilt;
    long      PForder;
    long      NBpix_in;
    long      NBpix_out;
    uint32_t *imsizearray;
    int       semtrig = 7;

    float *inarray;
    float *outarray;

    //    long ii; // input index
    //    long jj; // output index
    //    long kk; // time step index

    IDfilt = image_ID(IDfilt_name, dcimg, dcnimg);
    IDin   = image_ID(IDin_name, dcimg, dcnimg);

    PForder   = dcimg[IDfilt].md[0].size[2];
    NBpix_in  = dcimg[IDfilt].md[0].size[0];
    NBpix_out = dcimg[IDfilt].md[0].size[1];

    list_image_ID();

    if (dcimg[IDin].md[0].size[0] * dcimg[IDin].md[0].size[1] != NBpix_in)
    {
        printf("ERROR: lin predictor engine: filter input size does not match "
               "input telemetry\n");
        exit(0);
    }

    printf("Create prediction output %s\n", IDout_name);
    fflush(stdout);
    imsizearray = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    if (imsizearray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    imsizearray[0] = NBpix_out;
    imsizearray[1] = 1;
    {
        IMGID imgout_tmp         = imgid_make_from_name(IDout_name);
        imgout_tmp.mdt->naxis    = 2;
        imgout_tmp.mdt->size[0]  = imsizearray[0];
        imgout_tmp.mdt->size[1]  = imsizearray[1];
        imgout_tmp.mdt->datatype = _DATATYPE_FLOAT;
        imgout_tmp.mdt->shared   = 1;
        imgout_tmp.mdt->NBkw     = 1;
        imgout_tmp.im            = (IMAGE *) calloc(1, sizeof(IMAGE));
        imgid_mkimage(&imgout_tmp);
        IDout = imgout_tmp.ID;
    }
    free(imsizearray);
    COREMOD_MEMORY_image_set_semflush(IDout_name, -1);
    printf("Done\n");
    fflush(stdout);

    inarray = (float *) malloc(sizeof(float) * NBpix_in * PForder);
    if (inarray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    outarray = (float *) malloc(sizeof(float) * NBpix_out);
    if (outarray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    while (ImageStreamIO_semtrywait(dcimg + IDin, semtrig) == 0)
    {
    }
    while (1)
    {
        // initialize output array to zero
        for (uint32_t jj = 0; jj < NBpix_out; jj++)
        {
            outarray[jj] = 0.0;
        }

        // shift input buffer entries back one time step
        for (uint32_t kk = PForder - 1; kk > 0; kk--)
        {
            for (uint32_t ii = 0; ii < NBpix_in; ii++)
            {
                inarray[kk * NBpix_in + ii] = inarray[(kk - 1) * NBpix_in + ii];
            }
        }

        // multiply input by prediction matrix .. except for measurement yet to come
        for (uint32_t jj = 0; jj < NBpix_out; jj++)
        {
            for (uint32_t ii = 0; ii < NBpix_in; ii++)
            {
                for (uint32_t kk = 1; kk < PForder; kk++)
                {
                    outarray[jj] +=
                        dcimg[IDfilt].array.F[kk * NBpix_in * NBpix_out + jj * NBpix_in + ii] *
                        inarray[kk * NBpix_in + ii];
                }
            }
        }

        ImageStreamIO_semwait(dcimg + IDin, semtrig);

        // write new input in inarray vector
        for (uint32_t ii = 0; ii < NBpix_in; ii++)
        {
            inarray[ii] = dcimg[IDin].array.F[ii];
        }

        // multiply input by prediction matrix
        for (uint32_t jj = 0; jj < NBpix_out; jj++)
        {
            for (uint32_t ii = 0; ii < NBpix_in; ii++)
            {
                outarray[jj] += dcimg[IDfilt].array.F[jj * NBpix_in + ii] * inarray[ii];
            }
        }

        dcimg[IDout].md[0].write = 1;
        for (uint32_t jj = 0; jj < NBpix_out; jj++)
        {
            dcimg[IDout].array.F[jj] = outarray[jj];
        }
        COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);
        dcimg[IDout].md[0].cnt0++;
        dcimg[IDout].md[0].write = 0;
    }

    free(inarray);
    free(outarray);

    return IDout;
}


//
// out : prediction
//
// ADDITIONAL OUTPUTS:
// outf : time-shifted measurement
//

imageID LINARFILTERPRED_Apply_LinPredictor(const char *IDfilt_name,
                                           const char *IDin_name,
                                           float       PFlag,
                                           const char *IDout_name)
{
    imageID  IDout;
    imageID  IDin;
    imageID  IDfilt;
    uint32_t xsize;
    uint32_t ysize;
    uint64_t xysize;

    long  nbspl;
    long  PForder;
    long  step;
    long  kk;
    float alpha;
    long  PFlagl;
    float valp, valf;

    imageID IDoutf;

    IDin   = image_ID(IDin_name, dcimg, dcnimg);
    IDfilt = image_ID(IDfilt_name, dcimg, dcnimg);

    switch (dcimg[IDin].md[0].naxis)
    {
    case 2:
        nbspl = dcimg[IDin].md[0].size[1];
        xsize = dcimg[IDin].md[0].size[0];
        ysize = 1;
        create_2Dimage_ID(IDout_name, xsize, nbspl, &IDout);
        create_2Dimage_ID("outf", xsize, nbspl, &IDoutf);
        break;

    case 3:
        nbspl = dcimg[IDin].md[0].size[2];
        xsize = dcimg[IDin].md[0].size[0];
        ysize = dcimg[IDin].md[0].size[1];
        create_3Dimage_ID(IDout_name, xsize, ysize, nbspl, &IDout);
        create_3Dimage_ID("outf", xsize, ysize, nbspl, &IDoutf);
        break;

    default:
        printf("Invalid image size\n");
        break;
    }
    xysize = xsize * ysize;

    PForder = dcimg[IDfilt].md[0].size[2];

    if ((dcimg[IDfilt].md[0].size[0] != xysize) || (dcimg[IDfilt].md[0].size[1] != xysize))
    {
        printf("ERROR: filter \"%s\" size is incorrect\n", IDfilt_name);
        exit(0);
    }

    alpha  = PFlag - ((long) PFlag);
    PFlagl = (long) PFlag;

    for (kk = PForder; kk < nbspl; kk++) // time step
    {
        for (uint32_t iip = 0; iip < xysize; iip++) // predicted variable
        {
            valp = 0.0; // prediction
            for (step = 0; step < PForder; step++)
            {
                for (uint32_t ii = 0; ii < xsize * ysize; ii++) // input variable
                {
                    valp += dcimg[IDfilt].array.F[xysize * xysize * step + iip * xysize + ii] *
                            dcimg[IDin].array.F[(kk - step) * xysize + ii];
                }
            }
            dcimg[IDout].array.F[kk * xysize + iip] = valp;

            valf = 0.0;
            if (kk + PFlag + 1 < nbspl)
            {
                valf = (1.0 - alpha) * dcimg[IDin].array.F[(kk + PFlagl) * xysize + iip] +
                       alpha * dcimg[IDin].array.F[(kk + PFlagl + 1) * xysize + iip];
            }
            dcimg[IDoutf].array.F[kk * xysize + iip] = valf;
        }
    }

    return IDout;
}
