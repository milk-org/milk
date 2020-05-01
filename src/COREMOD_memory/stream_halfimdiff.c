/** 
 * @file stream_hlfimdiff.c
 * @brief difference between two halves of stream image
 */


#include "CommandLineInterface/CLIcore.h"
#include "image_ID.h"
#include "create_image.h"
#include "stream_sem.h"


//
// compute difference between two halves of an image stream
// triggers on instream
//
imageID COREMOD_MEMORY_stream_halfimDiff(
    const char *IDstream_name,
    const char *IDstreamout_name,
    long        semtrig
)
{
    imageID    ID0;
    imageID    IDout;
    uint32_t   xsizein;
    uint32_t   ysizein;
//    uint32_t   xysizein;
    uint32_t   xsize;
    uint32_t   ysize;
    uint32_t   xysize;
    long       ii;
    uint32_t  *arraysize;
    unsigned long long  cnt;
    uint8_t    datatype;
    uint8_t    datatypeout;


    ID0 = image_ID(IDstream_name);

    xsizein = data.image[ID0].md[0].size[0];
    ysizein = data.image[ID0].md[0].size[1];
//    xysizein = xsizein*ysizein;

    xsize = xsizein;
    ysize = ysizein / 2;
    xysize = xsize * ysize;


    arraysize = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    arraysize[0] = xsize;
    arraysize[1] = ysize;

    datatype = data.image[ID0].md[0].datatype;
    datatypeout = _DATATYPE_FLOAT;
    switch(datatype)
    {

        case _DATATYPE_UINT8:
            datatypeout = _DATATYPE_INT16;
            break;

        case _DATATYPE_UINT16:
            datatypeout = _DATATYPE_INT32;
            break;

        case _DATATYPE_UINT32:
            datatypeout = _DATATYPE_INT64;
            break;

        case _DATATYPE_UINT64:
            datatypeout = _DATATYPE_INT64;
            break;


        case _DATATYPE_INT8:
            datatypeout = _DATATYPE_INT16;
            break;

        case _DATATYPE_INT16:
            datatypeout = _DATATYPE_INT32;
            break;

        case _DATATYPE_INT32:
            datatypeout = _DATATYPE_INT64;
            break;

        case _DATATYPE_INT64:
            datatypeout = _DATATYPE_INT64;
            break;

        case _DATATYPE_DOUBLE:
            datatypeout = _DATATYPE_DOUBLE;
            break;
    }

    IDout = image_ID(IDstreamout_name);
    if(IDout == -1)
    {
        IDout = create_image_ID(IDstreamout_name, 2, arraysize, datatypeout, 1, 0);
        COREMOD_MEMORY_image_set_createsem(IDstreamout_name, IMAGE_NB_SEMAPHORE);
    }

    free(arraysize);



    while(1)
    {
        // has new frame arrived ?
        if(data.image[ID0].md[0].sem == 0)
        {
            while(cnt == data.image[ID0].md[0].cnt0) // test if new frame exists
            {
                usleep(5);
            }
            cnt = data.image[ID0].md[0].cnt0;
        }
        else
        {
            sem_wait(data.image[ID0].semptr[semtrig]);
        }

        data.image[IDout].md[0].write = 1;

        switch(datatype)
        {

            case _DATATYPE_UINT8:
                for(ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout].array.SI16[ii] = data.image[ID0].array.UI8[ii] -
                                                       data.image[ID0].array.UI8[xysize + ii];
                }
                break;

            case _DATATYPE_UINT16:
                for(ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout].array.SI32[ii] = data.image[ID0].array.UI16[ii] -
                                                       data.image[ID0].array.UI16[xysize + ii];
                }
                break;

            case _DATATYPE_UINT32:
                for(ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout].array.SI64[ii] = data.image[ID0].array.UI32[ii] -
                                                       data.image[ID0].array.UI32[xysize + ii];
                }
                break;

            case _DATATYPE_UINT64:
                for(ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout].array.SI64[ii] = data.image[ID0].array.UI64[ii] -
                                                       data.image[ID0].array.UI64[xysize + ii];
                }
                break;



            case _DATATYPE_INT8:
                for(ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout].array.SI16[ii] = data.image[ID0].array.SI8[ii] -
                                                       data.image[ID0].array.SI8[xysize + ii];
                }
                break;

            case _DATATYPE_INT16:
                for(ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout].array.SI32[ii] = data.image[ID0].array.SI16[ii] -
                                                       data.image[ID0].array.SI16[xysize + ii];
                }
                break;

            case _DATATYPE_INT32:
                for(ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout].array.SI64[ii] = data.image[ID0].array.SI32[ii] -
                                                       data.image[ID0].array.SI32[xysize + ii];
                }
                break;

            case _DATATYPE_INT64:
                for(ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout].array.SI64[ii] = data.image[ID0].array.SI64[ii] -
                                                       data.image[ID0].array.SI64[xysize + ii];
                }
                break;



            case _DATATYPE_FLOAT:
                for(ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout].array.F[ii] = data.image[ID0].array.F[ii] -
                                                    data.image[ID0].array.F[xysize + ii];
                }
                break;

            case _DATATYPE_DOUBLE:
                for(ii = 0; ii < xysize; ii++)
                {
                    data.image[IDout].array.D[ii] = data.image[ID0].array.D[ii] -
                                                    data.image[ID0].array.D[xysize + ii];
                }
                break;

        }

        COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);
        data.image[IDout].md[0].cnt0++;
        data.image[IDout].md[0].write = 0;
    }


    return IDout;
}

