/** @file stream_ave.c
 */
 
 
 

#include "CommandLineInterface/CLIcore.h"
#include "image_ID.h"
#include "stream_sem.h"
#include "create_image.h"
#include "delete_image.h"





// ==========================================
// Forward declaration(s)
// ==========================================

imageID COREMOD_MEMORY_streamAve(
    const char *IDstream_name,
    int         NBave,
    int         mode,
    const char *IDout_name
);



// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t COREMOD_MEMORY_streamAve__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_LONG)
            + CLI_checkarg(3, CLIARG_LONG)
            + CLI_checkarg(4, 5)
            == 0)
    {
        COREMOD_MEMORY_streamAve(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.numl,
            data.cmdargtoken[3].val.numl,
            data.cmdargtoken[4].val.string
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}




// ==========================================
// Register CLI command(s)
// ==========================================

errno_t stream_ave_addCLIcmd()
{
    RegisterCLIcommand(
        "streamave",
        __FILE__,
        COREMOD_MEMORY_streamAve__cli,
        "averages stream",
        "<instream> <NBave> <mode, 1 for single local instance, 0 for loop> <outstream>",
        "streamave instream 100 0 outstream",
        "long COREMODE_MEMORY_streamAve(const char *IDstream_name, int NBave, int mode, const char *IDout_name)");

    return RETURN_SUCCESS;
}

















/** @brief Averages frames in stream
 *
 * @param[in]  IDstream_name        Input stream
 * @param[in]  NBave                Number of consecutive frames to be averaged together
 * @param[in]  mode                 1: Perform average once, exit when completed and write output to local image
 * 									2: Run forever, write output to shared mem stream
 * @param[out] IDout_name           output stream name
 *
 */


imageID COREMOD_MEMORY_streamAve(
    const char *IDstream_name,
    int         NBave,
    int         mode,
    const char *IDout_name
)
{
    imageID      IDout;
    imageID      IDout0;
    imageID      IDin;
    uint8_t      datatype;
    uint32_t     xsize;
    uint32_t     ysize;
    uint32_t     xysize;
    uint32_t    *imsize;
    int_fast8_t  OKloop;
    int          cntin = 0;
    long         dtus = 20;
    long         ii;
    long         cnt0;
    long         cnt0old;

    IDin = image_ID(IDstream_name);
    datatype = data.image[IDin].md[0].datatype;
    xsize = data.image[IDin].md[0].size[0];
    ysize = data.image[IDin].md[0].size[1];
    xysize = xsize * ysize;


    IDout0 = create_2Dimage_ID("_streamAve_tmp", xsize, ysize);

    if(mode == 1) // local image
    {
        IDout = create_2Dimage_ID(IDout_name, xsize, ysize);
    }
    else // shared memory
    {
        IDout = image_ID(IDout_name);
        if(IDout == -1) // CREATE IT
        {
            imsize = (uint32_t *) malloc(sizeof(uint32_t) * 2);
            imsize[0] = xsize;
            imsize[1] = ysize;
            create_image_ID(IDout_name, 2, imsize, _DATATYPE_FLOAT, 1, 0, 0, &IDout);
            COREMOD_MEMORY_image_set_createsem(IDout_name, IMAGE_NB_SEMAPHORE);
            free(imsize);
        }
    }


    cntin = 0;
    cnt0old = data.image[IDin].md[0].cnt0;

    for(ii = 0; ii < xysize; ii++)
    {
        data.image[IDout].array.F[ii] = 0.0;
    }

    OKloop = 1;
    while(OKloop == 1)
    {
        // has new frame arrived ?
        cnt0 = data.image[IDin].md[0].cnt0;
        if(cnt0 != cnt0old)
        {
            switch(datatype)
            {
                case _DATATYPE_UINT8 :
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout0].array.F[ii] += data.image[IDin].array.UI8[ii];
                    }
                    break;

                case _DATATYPE_INT8 :
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout0].array.F[ii] += data.image[IDin].array.SI8[ii];
                    }
                    break;

                case _DATATYPE_UINT16 :
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout0].array.F[ii] += data.image[IDin].array.UI16[ii];
                    }
                    break;

                case _DATATYPE_INT16 :
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout0].array.F[ii] += data.image[IDin].array.SI16[ii];
                    }
                    break;

                case _DATATYPE_UINT32 :
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout0].array.F[ii] += data.image[IDin].array.UI32[ii];
                    }
                    break;

                case _DATATYPE_INT32 :
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout0].array.F[ii] += data.image[IDin].array.SI32[ii];
                    }
                    break;

                case _DATATYPE_UINT64 :
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout0].array.F[ii] += data.image[IDin].array.UI64[ii];
                    }
                    break;

                case _DATATYPE_INT64 :
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout0].array.F[ii] += data.image[IDin].array.SI64[ii];
                    }
                    break;

                case _DATATYPE_FLOAT :
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout0].array.F[ii] += data.image[IDin].array.F[ii];
                    }
                    break;

                case _DATATYPE_DOUBLE :
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout0].array.F[ii] += data.image[IDin].array.D[ii];
                    }
                    break;
            }

            cntin++;
            if(cntin == NBave)
            {
                cntin = 0;
                data.image[IDout].md[0].write = 1;
                for(ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout].array.F[ii] = data.image[IDout0].array.F[ii] / NBave;
                }
                data.image[IDout].md[0].cnt0++;
                data.image[IDout].md[0].write = 0;
                COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);

                if(mode != 1)
                {
                    for(ii = 0; ii < xysize; ii++)
                    {
                        data.image[IDout].array.F[ii] = 0.0;
                    }
                }
                else
                {
                    OKloop = 0;
                }
            }
            cnt0old = cnt0;
        }
        usleep(dtus);
    }

    delete_image_ID("_streamAve_tmp", DELETE_IMAGE_ERRMODE_WARNING);

    return IDout;
}

