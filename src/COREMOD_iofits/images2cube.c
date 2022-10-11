/**
 * @file    images2cube.c
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

// ==========================================
// Forward declaration(s)
// ==========================================

errno_t images_to_cube(const char *restrict img_name,
                       long nbframes,
                       const char *restrict cube_name);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

errno_t images_to_cube_cli()
{
    /*  if(data.cmdargtoken[1].type != 4)
      {
        printf("Image %s does not exist\n", data.cmdargtoken[1].val.string);
        return -1;
        }*/

    if(data.cmdargtoken[2].type != 2)
    {
        printf("second argument has to be integer\n");
        return -1;
    }

    images_to_cube(data.cmdargtoken[1].val.string,
                   data.cmdargtoken[2].val.numl,
                   data.cmdargtoken[3].val.string);

    return CLICMD_SUCCESS;
}

// ==========================================
// Register CLI command(s)
// ==========================================

errno_t images2cube_addCLIcmd()
{

    RegisterCLIcommand(
        "imgs2cube",
        __FILE__,
        images_to_cube_cli,
        "combine individual images into cube, image name is prefix followed by "
        "5 digits",
        "<input image format> <max index> <output cube>",
        "imgs2cube im_ 100 imc",
        "int images_to_cube(char *img_name, long nbframes, char *cube_name)");

    return RETURN_SUCCESS;
}

errno_t images_to_cube(const char *restrict img_name,
                       long nbframes,
                       const char *restrict cube_name)
{
    DEBUG_TRACE_FSTART();
    imageID  ID;
    imageID  ID1;
    long     frame;
    uint32_t naxes[2];
    uint32_t xsize, ysize;

    frame = 0;

    CREATE_IMAGENAME(imname, "%s%05ld", img_name, frame);

    ID1 = image_ID(imname);
    if(ID1 == -1)
    {
        PRINT_ERROR("Image \"%s\" does not exist", imname);
        exit(0);
    }
    naxes[0] = data.image[ID1].md[0].size[0];
    naxes[1] = data.image[ID1].md[0].size[1];
    xsize    = naxes[0];
    ysize    = naxes[1];

    printf("SIZE = %ld %ld %ld\n",
           (long) naxes[0],
           (long) naxes[1],
           (long) nbframes);
    fflush(stdout);

    FUNC_CHECK_RETURN(
        create_3Dimage_ID(cube_name, naxes[0], naxes[1], nbframes, &ID));

    for(uint32_t ii = 0; ii < naxes[0]; ii++)
        for(uint32_t jj = 0; jj < naxes[1]; jj++)
        {
            data.image[ID]
            .array.F[frame * naxes[0] * naxes[1] + (jj * naxes[0] + ii)] =
                data.image[ID1].array.F[jj * naxes[0] + ii];
        }

    for(frame = 1; frame < nbframes; frame++)
    {
        WRITE_IMAGENAME(imname, "%s%05ld", img_name, frame);
        printf("Adding image %s -> %ld/%ld ... ", img_name, frame, nbframes);
        fflush(stdout);

        ID1 = image_ID(imname);
        if(ID1 == -1)
        {
            PRINT_ERROR("Image \"%s\" does not exist - skipping", imname);
        }
        else
        {
            naxes[0] = data.image[ID1].md[0].size[0];
            naxes[1] = data.image[ID1].md[0].size[1];
            if((xsize != naxes[0]) || (ysize != naxes[1]))
            {
                PRINT_ERROR("Image has wrong size");
                exit(0);
            }
            for(uint32_t ii = 0; ii < naxes[0]; ii++)
                for(uint32_t jj = 0; jj < naxes[1]; jj++)
                {
                    data.image[ID].array.F[frame * naxes[0] * naxes[1] +
                                           (jj * naxes[0] + ii)] =
                                               data.image[ID1].array.F[jj * naxes[0] + ii];
                }
        }
        printf("Done\n");
        fflush(stdout);
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
