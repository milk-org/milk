/**
 * @file    image_merge3D.c
 * @brief   merge 3D images
 *
 *
 */


#include "CommandLineInterface/CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"





// ==========================================
// Forward declaration(s)
// ==========================================

imageID arith_image_merge3D(
    const char *ID_name1,
    const char *ID_name2,
    const char *IDout_name
);



// ==========================================
// Command line interface wrapper function(s)
// ==========================================


static errno_t arith_image_merge3D_cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_IMG)
            + CLI_checkarg(3, CLIARG_STR_NOT_IMG)
            == 0)
    {
        arith_image_merge3D(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.string,
            data.cmdargtoken[3].val.string);

        return CLICMD_SUCCESS;
    }
    else
    {
        return 1;
    }
}




// ==========================================
// Register CLI command(s)
// ==========================================

errno_t image_merge3D_addCLIcmd()
{

    RegisterCLIcommand(
        "merge3d",
        __FILE__,
        arith_image_merge3D_cli,
        "merge two 3D cubes into one",
        "<input cube 1> <input cube 2> <output cube>",
        "merge3d imc1 imc2 imcout",
        "long arith_image_merge3D(const char *ID_name1, const char *ID_name2, const char *IDout_name)");


    return RETURN_SUCCESS;
}















// join two cubes
imageID arith_image_merge3D(
    const char *ID_name1,
    const char *ID_name2,
    const char *IDout_name
)
{
    DEBUG_TRACE_FSTART();

    imageID ID1;
    imageID ID2;
    imageID IDout;
    uint32_t xsize, ysize, zsize1, zsize2;
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
        printf("%s :  %u %u\n", ID_name1, xsize, ysize);
        printf("%s :  %ld %ld\n", ID_name2, (long) data.image[ID2].md[0].size[0],
               (long) data.image[ID2].md[0].size[1]);
        exit(0);
    }

    uint32_t * sizearray = (uint32_t *) malloc(sizeof(uint32_t)*3);
    sizearray[0] = xsize;
    sizearray[1] = xsize;
    sizearray[2] = zsize1+zsize2;

    uint8_t datatype = data.image[ID1].md[0].datatype;

    FUNC_CHECK_RETURN(
        create_image_ID(IDout_name, 3, sizearray, datatype, 0, 0, 0, &IDout)
    );


    switch (datatype)
    {
    case _DATATYPE_FLOAT :
        mapv = (void *) data.image[IDout].array.F;
        memcpy(mapv, (void *) data.image[ID1].array.F,
               sizeof(float)*xsize * ysize * zsize1);
        mapv += sizeof(float) * xsize * ysize * zsize1;
        memcpy(mapv, data.image[ID2].array.F, sizeof(float)*xsize * ysize * zsize2);
        break;

    case _DATATYPE_DOUBLE :
        mapv = (void *) data.image[IDout].array.D;
        memcpy(mapv, (void *) data.image[ID1].array.D,
               sizeof(double)*xsize * ysize * zsize1);
        mapv += sizeof(double) * xsize * ysize * zsize1;
        memcpy(mapv, data.image[ID2].array.D, sizeof(double)*xsize * ysize * zsize2);
        break;

    case _DATATYPE_INT16 :
        mapv = (void *) data.image[IDout].array.SI16;
        memcpy(mapv, (void *) data.image[ID1].array.SI16,
               sizeof(int16_t)*xsize * ysize * zsize1);
        mapv += sizeof(int16_t) * xsize * ysize * zsize1;
        memcpy(mapv, data.image[ID2].array.SI16, sizeof(int16_t)*xsize * ysize * zsize2);
        break;

    case _DATATYPE_UINT16 :
        mapv = (void *) data.image[IDout].array.UI16;
        memcpy(mapv, (void *) data.image[ID1].array.UI16,
               sizeof(uint16_t)*xsize * ysize * zsize1);
        mapv += sizeof(uint16_t) * xsize * ysize * zsize1;
        memcpy(mapv, data.image[ID2].array.UI16, sizeof(uint16_t)*xsize * ysize * zsize2);
        break;

    default:
        printf("dataypte not supported\n");
    }



    DEBUG_TRACE_FEXIT();
    return IDout;
}
