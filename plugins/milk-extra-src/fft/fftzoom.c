/** @file fftzoom.c
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

#include "dofft.h"
#include "permut.h"

int fftczoom(const char *ID_name, const char *IDout_name, long factor)
{
    imageID  ID;
    imageID  ID1;
    uint32_t naxes[2];
    double   coeff;

    char tmpzname[STRINGMAXLEN_IMGNAME];
    char tmpz1name[STRINGMAXLEN_IMGNAME];

    ID = image_ID(ID_name);

    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    coeff = 1.0 / (factor * factor * naxes[0] * naxes[1]);
    permut(ID_name);

    WRITE_IMAGENAME(tmpzname, "_tmpz_%d", (int) getpid());
    do2dfft(ID_name, tmpzname);

    permut(ID_name);
    permut(tmpzname);
    ID = image_ID(tmpzname);

    WRITE_IMAGENAME(tmpz1name, "_tmpz1_%d", (int) getpid());

    create_2DCimage_ID(tmpz1name, factor * naxes[0], factor * naxes[1], &ID1);

    for(uint32_t ii = 0; ii < naxes[0]; ii++)
        for(uint32_t jj = 0; jj < naxes[1]; jj++)
        {
            data.image[ID1]
            .array
            .CF[(jj + factor * naxes[1] / 2 - naxes[1] / 2) * naxes[0] *
                                                            factor +
                                                            (ii + factor * naxes[0] / 2 - naxes[0] / 2)]
            .re = data.image[ID].array.CF[jj * naxes[0] + ii].re * coeff;
            data.image[ID1]
            .array
            .CF[(jj + factor * naxes[1] / 2 - naxes[1] / 2) * naxes[0] *
                                                            factor +
                                                            (ii + factor * naxes[0] / 2 - naxes[0] / 2)]
            .im = data.image[ID].array.CF[jj * naxes[0] + ii].im * coeff;
        }
    delete_image_ID(tmpzname, DELETE_IMAGE_ERRMODE_WARNING);

    permut(tmpz1name);
    do2dffti(tmpz1name, IDout_name);
    permut(IDout_name);
    delete_image_ID(tmpz1name, DELETE_IMAGE_ERRMODE_WARNING);

    return (0);
}

int fftzoom(const char *ID_name, const char *IDout_name, long factor)
{
    imageID  ID;
    imageID  ID1;
    uint32_t naxes[2];
    double   coeff;

    ID = image_ID(ID_name);

    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    coeff = 1.0 / (factor * factor * naxes[0] * naxes[1]);
    permut(ID_name);

    CREATE_IMAGENAME(tmpzname, "_tmpz_%d", (int) getpid());

    do2drfft(ID_name, tmpzname);

    permut(ID_name);
    permut(tmpzname);
    ID = image_ID(tmpzname);

    CREATE_IMAGENAME(tmpz1name, "_tmpz1_%d", (int) getpid());

    create_2DCimage_ID(tmpz1name, factor * naxes[0], factor * naxes[1], &ID1);

    for(uint32_t ii = 0; ii < naxes[0]; ii++)
        for(uint32_t jj = 0; jj < naxes[1]; jj++)
        {
            data.image[ID1]
            .array
            .CF[(jj + factor * naxes[1] / 2 - naxes[1] / 2) * naxes[0] *
                                                            factor +
                                                            (ii + factor * naxes[0] / 2 - naxes[0] / 2)]
            .re = data.image[ID].array.CF[jj * naxes[0] + ii].re * coeff;
            data.image[ID1]
            .array
            .CF[(jj + factor * naxes[1] / 2 - naxes[1] / 2) * naxes[0] *
                                                            factor +
                                                            (ii + factor * naxes[0] / 2 - naxes[0] / 2)]
            .im = data.image[ID].array.CF[jj * naxes[0] + ii].im * coeff;
        }
    delete_image_ID(tmpzname, DELETE_IMAGE_ERRMODE_WARNING);

    permut(tmpz1name);

    CREATE_IMAGENAME(tmpz2name, "_tmpz2_%d", (int) getpid());

    do2dffti(tmpz1name, tmpz2name);

    permut(tmpz2name);
    delete_image_ID(tmpz1name, DELETE_IMAGE_ERRMODE_WARNING);

    CREATE_IMAGENAME(tbename, "_tbe_%d", (int) getpid());

    mk_reim_from_complex(tmpz2name, IDout_name, tbename, 0);

    delete_image_ID(tbename, DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID(tmpz2name, DELETE_IMAGE_ERRMODE_WARNING);

    return (0);
}
