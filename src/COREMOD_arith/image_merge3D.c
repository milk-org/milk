/**
 * @file    image_merge3D.c
 * @brief   merge 3D images
 *
 *
 */


#include "CommandLineInterface/CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"




// join two cubes
imageID arith_image_merge3D(
    const char *ID_name1,
    const char *ID_name2,
    const char *IDout_name
)
{
    imageID ID1;
    imageID ID2;
    imageID IDout;
    long xsize, ysize, zsize1, zsize2;
    void *mapv;

    ID1 = image_ID(ID_name1);
    ID2 = image_ID(ID_name2);

    xsize = data.image[ID1].md[0].size[0];
    ysize = data.image[ID1].md[0].size[1];

    if(data.image[ID1].md[0].naxis == 2)
    {
        zsize1 = 1;
    }
    else
    {
        zsize1 = data.image[ID1].md[0].size[2];
    }

    if(data.image[ID2].md[0].naxis == 2)
    {
        zsize2 = 1;
    }
    else
    {
        zsize2 = data.image[ID2].md[0].size[2];
    }



    if((xsize != data.image[ID2].md[0].size[0])
            || (ysize != data.image[ID2].md[0].size[1]))
    {
        printf("ERROR: input images must have same x y sizes\n");
        printf("%s :  %ld %ld\n", ID_name1, xsize, ysize);
        printf("%s :  %ld %ld\n", ID_name2, (long) data.image[ID2].md[0].size[0],
               (long) data.image[ID2].md[0].size[1]);
        exit(0);
    }

    IDout = create_3Dimage_ID(IDout_name, xsize, ysize, zsize1 + zsize2);

    mapv = (void *) data.image[IDout].array.F;

    memcpy(mapv, (void *) data.image[ID1].array.F,
           sizeof(float)*xsize * ysize * zsize1);

    mapv += sizeof(float) * xsize * ysize * zsize1;
    memcpy(mapv, data.image[ID2].array.F, sizeof(float)*xsize * ysize * zsize2);

    return IDout;
}
