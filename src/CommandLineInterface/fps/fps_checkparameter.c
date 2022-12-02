/**
 * @file    fps_checkparameters.c
 * @brief   check FPS entries
 */

#include <sys/stat.h> // fstat

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "fps_connect.h"
#include "fps_disconnect.h"
#include "fps_outlog.h"




int functionparameter_CheckParameter(FUNCTION_PARAMETER_STRUCT *fpsentry,
                                     int                        pindex)
{
    int err = 0;


    // if entry is not active or not used, no error reported
    //
    if((!(fpsentry->parray[pindex].fpflag & FPFLAG_ACTIVE)))
    {
        return 0;
    }
    else
    {
        char msg[STRINGMAXLEN_FPS_LOGMSG];
        SNPRINTF_CHECK(msg,
                       STRINGMAXLEN_FPS_LOGMSG,
                       "%s",
                       fpsentry->parray[pindex].keywordfull);
        functionparameter_outlog("CHECKPARAM", "%s", msg);
    }

    // if entry is not used, no error reported
    if(!(fpsentry->parray[pindex].fpflag & FPFLAG_USED))
    {
        return 0;
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_CHECKINIT)
        if(fpsentry->parray[pindex].cnt0 == 0)
        {
            fpsentry->md->msgpindex[fpsentry->md->msgcnt] = pindex;
            fpsentry->md->msgcode[fpsentry->md->msgcnt] =
                FPS_MSG_FLAG_NOTINITIALIZED | FPS_MSG_FLAG_ERROR;
            if(snprintf(fpsentry->md->message[fpsentry->md->msgcnt],
                        FUNCTION_PARAMETER_STRUCT_MSG_SIZE,
                        "Not initialized") < 0)
            {
                PRINT_ERROR("snprintf error");
            }
            fpsentry->md->msgcnt++;
            if(fpsentry->md->msgcnt > FPS_NB_MSG - 1)
            {
                // max number of msg reached, write to last one
                fpsentry->md->msgcnt = FPS_NB_MSG - 1;
            }
            fpsentry->md->conferrcnt++;
            err = 1;
        }



    if(err == 0)
    {
        // Check min value
        if(fpsentry->parray[pindex].type == FPTYPE_INT64)
            if(fpsentry->parray[pindex].fpflag & FPFLAG_MINLIMIT)
                if(fpsentry->parray[pindex].val.i64[0] <
                        fpsentry->parray[pindex].val.i64[1])
                {
                    fpsentry->md->msgpindex[fpsentry->md->msgcnt] = pindex;
                    fpsentry->md->msgcode[fpsentry->md->msgcnt] =
                        FPS_MSG_FLAG_BELOWMIN | FPS_MSG_FLAG_ERROR;
                    if(snprintf(fpsentry->md->message[fpsentry->md->msgcnt],
                                FUNCTION_PARAMETER_STRUCT_MSG_SIZE,
                                "int64 value %ld below min %ld",
                                fpsentry->parray[pindex].val.i64[0],
                                fpsentry->parray[pindex].val.i64[1]) < 0)
                    {
                        PRINT_ERROR("snprintf error");
                    }
                    fpsentry->md->msgcnt++;
                    if(fpsentry->md->msgcnt > FPS_NB_MSG - 1)
                    {
                        // max number of msg reached, write to last one
                        fpsentry->md->msgcnt = FPS_NB_MSG - 1;
                    }
                    fpsentry->md->conferrcnt++;
                    err = 1;
                }

        if(fpsentry->parray[pindex].type == FPTYPE_FLOAT64)
            if(fpsentry->parray[pindex].fpflag & FPFLAG_MINLIMIT)
                if(fpsentry->parray[pindex].val.f64[0] <
                        fpsentry->parray[pindex].val.f64[1])
                {
                    fpsentry->md->msgpindex[fpsentry->md->msgcnt] = pindex;
                    fpsentry->md->msgcode[fpsentry->md->msgcnt] =
                        FPS_MSG_FLAG_BELOWMIN | FPS_MSG_FLAG_ERROR;
                    if(snprintf(fpsentry->md->message[fpsentry->md->msgcnt],
                                FUNCTION_PARAMETER_STRUCT_MSG_SIZE,
                                "float64 value %lf below min %lf",
                                fpsentry->parray[pindex].val.f64[0],
                                fpsentry->parray[pindex].val.f64[1]) < 0)
                    {
                        PRINT_ERROR("snprintf error");
                    }
                    fpsentry->md->msgcnt++;
                    if(fpsentry->md->msgcnt > FPS_NB_MSG - 1)
                    {
                        // max number of msg reached, write to last one
                        fpsentry->md->msgcnt = FPS_NB_MSG - 1;
                    }
                    fpsentry->md->conferrcnt++;
                    err = 1;
                }

        if(fpsentry->parray[pindex].type == FPTYPE_FLOAT32)
            if(fpsentry->parray[pindex].fpflag & FPFLAG_MINLIMIT)
                if(fpsentry->parray[pindex].val.f32[0] <
                        fpsentry->parray[pindex].val.f32[1])
                {
                    fpsentry->md->msgpindex[fpsentry->md->msgcnt] = pindex;
                    fpsentry->md->msgcode[fpsentry->md->msgcnt] =
                        FPS_MSG_FLAG_BELOWMIN | FPS_MSG_FLAG_ERROR;
                    if(snprintf(fpsentry->md->message[fpsentry->md->msgcnt],
                                FUNCTION_PARAMETER_STRUCT_MSG_SIZE,
                                "float32 value %f below min %f",
                                fpsentry->parray[pindex].val.f32[0],
                                fpsentry->parray[pindex].val.f32[1]) < 0)
                    {
                        PRINT_ERROR("snprintf error");
                    }
                    fpsentry->md->msgcnt++;
                    if(fpsentry->md->msgcnt > FPS_NB_MSG - 1)
                    {
                        // max number of msg reached, write to last one
                        fpsentry->md->msgcnt = FPS_NB_MSG - 1;
                    }
                    fpsentry->md->conferrcnt++;
                    err = 1;
                }
    }

    if(err == 0)
    {
        // Check max value
        if(fpsentry->parray[pindex].type == FPTYPE_INT64)
            if(fpsentry->parray[pindex].fpflag & FPFLAG_MAXLIMIT)
                if(fpsentry->parray[pindex].val.i64[0] >
                        fpsentry->parray[pindex].val.i64[2])
                {
                    fpsentry->md->msgpindex[fpsentry->md->msgcnt] = pindex;
                    fpsentry->md->msgcode[fpsentry->md->msgcnt] =
                        FPS_MSG_FLAG_ABOVEMAX | FPS_MSG_FLAG_ERROR;
                    if(snprintf(fpsentry->md->message[fpsentry->md->msgcnt],
                                FUNCTION_PARAMETER_STRUCT_MSG_SIZE,
                                "int64 value %ld above max %ld",
                                fpsentry->parray[pindex].val.i64[0],
                                fpsentry->parray[pindex].val.i64[2]) < 0)
                    {
                        PRINT_ERROR("snprintf error");
                    }
                    fpsentry->md->msgcnt++;
                    if(fpsentry->md->msgcnt > FPS_NB_MSG - 1)
                    {
                        // max number of msg reached, write to last one
                        fpsentry->md->msgcnt = FPS_NB_MSG - 1;
                    }
                    fpsentry->md->conferrcnt++;
                    err = 1;
                }

        if(fpsentry->parray[pindex].type == FPTYPE_FLOAT64)
            if(fpsentry->parray[pindex].fpflag & FPFLAG_MAXLIMIT)
                if(fpsentry->parray[pindex].val.f64[0] >
                        fpsentry->parray[pindex].val.f64[2])
                {
                    fpsentry->md->msgpindex[fpsentry->md->msgcnt] = pindex;
                    fpsentry->md->msgcode[fpsentry->md->msgcnt] =
                        FPS_MSG_FLAG_ABOVEMAX | FPS_MSG_FLAG_ERROR;
                    if(snprintf(fpsentry->md->message[fpsentry->md->msgcnt],
                                FUNCTION_PARAMETER_STRUCT_MSG_SIZE,
                                "float64 value %lf above max %lf",
                                fpsentry->parray[pindex].val.f64[0],
                                fpsentry->parray[pindex].val.f64[2]) < 0)
                    {
                        PRINT_ERROR("snprintf error");
                    }
                    fpsentry->md->msgcnt++;
                    if(fpsentry->md->msgcnt > FPS_NB_MSG - 1)
                    {
                        // max number of msg reached, write to last one
                        fpsentry->md->msgcnt = FPS_NB_MSG - 1;
                    }
                    fpsentry->md->conferrcnt++;
                    err = 1;
                }

        if(fpsentry->parray[pindex].type == FPTYPE_FLOAT32)
            if(fpsentry->parray[pindex].fpflag & FPFLAG_MAXLIMIT)
                if(fpsentry->parray[pindex].val.f32[0] >
                        fpsentry->parray[pindex].val.f32[2])
                {
                    fpsentry->md->msgpindex[fpsentry->md->msgcnt] = pindex;
                    fpsentry->md->msgcode[fpsentry->md->msgcnt] =
                        FPS_MSG_FLAG_ABOVEMAX | FPS_MSG_FLAG_ERROR;
                    if(snprintf(fpsentry->md->message[fpsentry->md->msgcnt],
                                FUNCTION_PARAMETER_STRUCT_MSG_SIZE,
                                "float32 value %f above max %f",
                                fpsentry->parray[pindex].val.f32[0],
                                fpsentry->parray[pindex].val.f32[2]) < 0)
                    {
                        PRINT_ERROR("snprintf error");
                    }
                    fpsentry->md->msgcnt++;
                    if(fpsentry->md->msgcnt > FPS_NB_MSG - 1)
                    {
                        // max number of msg reached, write to last one
                        fpsentry->md->msgcnt = FPS_NB_MSG - 1;
                    }
                    fpsentry->md->conferrcnt++;
                    err = 1;
                }
    }

    if(fpsentry->parray[pindex].type == FPTYPE_FILENAME)
    {
        if(fpsentry->parray[pindex].fpflag & FPFLAG_FILE_RUN_REQUIRED)
        {
            if(file_exists(fpsentry->parray[pindex].val.string[0]) == 0)
            {
                fpsentry->md->msgpindex[fpsentry->md->msgcnt] = pindex;
                fpsentry->md->msgcode[fpsentry->md->msgcnt] =
                    FPS_MSG_FLAG_ERROR;
                if(snprintf(fpsentry->md->message[fpsentry->md->msgcnt],
                            FUNCTION_PARAMETER_STRUCT_MSG_SIZE,
                            "File %s does not exist",
                            fpsentry->parray[pindex].val.string[0]) < 0)
                {
                    PRINT_ERROR("snprintf error");
                }
                fpsentry->md->msgcnt++;
                if(fpsentry->md->msgcnt > FPS_NB_MSG - 1)
                {
                    // max number of msg reached, write to last one
                    fpsentry->md->msgcnt = FPS_NB_MSG - 1;
                }
                fpsentry->md->conferrcnt++;
                err = 1;
            }
        }
    }

    if(fpsentry->parray[pindex].type == FPTYPE_FITSFILENAME)
    {

        if(fpsentry->parray[pindex].fpflag & FPFLAG_FILE_RUN_REQUIRED)
        {
            if(is_fits_file(fpsentry->parray[pindex].val.string[0]) == 0)
            {
                fpsentry->md->msgpindex[fpsentry->md->msgcnt] = pindex;
                fpsentry->md->msgcode[fpsentry->md->msgcnt] =
                    FPS_MSG_FLAG_ERROR;
                if(snprintf(fpsentry->md->message[fpsentry->md->msgcnt],
                            FUNCTION_PARAMETER_STRUCT_MSG_SIZE,
                            "FITS file %s does not exist",
                            fpsentry->parray[pindex].val.string[0]) < 0)
                {
                    PRINT_ERROR("snprintf error msgcnt %ld file %s line %d", fpsentry->md->msgcnt,
                                __FILE__, __LINE__);
                }
                fpsentry->md->msgcnt++;

                if(fpsentry->md->msgcnt > FPS_NB_MSG - 1)
                {
                    // max number of msg reached, write to last one
                    fpsentry->md->msgcnt = FPS_NB_MSG - 1;
                }
                fpsentry->md->conferrcnt++;
                err = 1;
            }
        }

    }

    if(fpsentry->parray[pindex].type == FPTYPE_EXECFILENAME)
    {
        if(fpsentry->parray[pindex].fpflag & FPFLAG_FILE_RUN_REQUIRED)
        {
            struct stat sb;
            if(!(stat(fpsentry->parray[pindex].val.string[0], &sb) == 0 &&
                    sb.st_mode & S_IXUSR))
            {
                fpsentry->md->msgpindex[fpsentry->md->msgcnt] = pindex;
                fpsentry->md->msgcode[fpsentry->md->msgcnt] =
                    FPS_MSG_FLAG_ERROR;
                if(snprintf(fpsentry->md->message[fpsentry->md->msgcnt],
                            FUNCTION_PARAMETER_STRUCT_MSG_SIZE,
                            "File %s cannot be executed",
                            fpsentry->parray[pindex].val.string[0]) < 0)
                {
                    PRINT_ERROR("snprintf error");
                }
                fpsentry->md->msgcnt++;
                if(fpsentry->md->msgcnt > FPS_NB_MSG - 1)
                {
                    // max number of msg reached, write to last one
                    fpsentry->md->msgcnt = FPS_NB_MSG - 1;
                }
                fpsentry->md->conferrcnt++;
                err = 1;
            }
        }
    }

    if(fpsentry->parray[pindex].type == FPTYPE_FPSNAME)
    {
        FUNCTION_PARAMETER_STRUCT fpstest;
        fpstest.SMfd = -1; // initialize

        functionparameter_ConnectExternalFPS(fpsentry, pindex, &fpstest);

        long NBparamMAX = fpsentry->parray[pindex].info.fps.FPSNBparamMAX;
        printf("%s NBparamMAX = %ld\n",
               fpsentry->parray[pindex].val.string[0],
               NBparamMAX);


        if(fpsentry->parray[pindex].fpflag & FPFLAG_FPS_RUN_REQUIRED)
        {
            if(NBparamMAX < 1)
            {
                fpsentry->md->msgpindex[fpsentry->md->msgcnt] = pindex;
                fpsentry->md->msgcode[fpsentry->md->msgcnt] =
                    FPS_MSG_FLAG_ERROR;
                if(snprintf(fpsentry->md->message[fpsentry->md->msgcnt],
                            FUNCTION_PARAMETER_STRUCT_MSG_SIZE,
                            "FPS %s: no connection %lu",
                            fpsentry->parray[pindex].val.string[0],
                            fpsentry->parray[pindex].fpflag &
                            FPFLAG_FPS_RUN_REQUIRED) < 0)
                {
                    PRINT_ERROR("snprintf error");
                }
                fpsentry->md->msgcnt++;
                if(fpsentry->md->msgcnt > FPS_NB_MSG - 1)
                {
                    // max number of msg reached, write to last one
                    fpsentry->md->msgcnt = FPS_NB_MSG - 1;
                }
                fpsentry->md->conferrcnt++;
                err = 1;
            }
            function_parameter_struct_disconnect(&fpstest);
        }
    }

    // STREAM CHECK
    if((fpsentry->parray[pindex].type & FPTYPE_STREAMNAME))
    {
        uint32_t imLOC;
        long     ID =
            COREMOD_IOFITS_LoadMemStream(fpsentry->parray[pindex].val.string[0],
                                         &(fpsentry->parray[pindex].fpflag),
                                         &imLOC);
        fpsentry->parray[pindex].info.stream.streamID = ID;

        if(ID > -1)
        {
            fpsentry->parray[pindex].info.stream.stream_sourceLocation = imLOC;
            fpsentry->parray[pindex].info.stream.stream_atype =
                data.image[ID].md[0].datatype;

            fpsentry->parray[pindex].info.stream.stream_naxis[0] =
                data.image[ID].md[0].naxis;
            fpsentry->parray[pindex].info.stream.stream_xsize[0] =
                data.image[ID].md[0].size[0];

            if(fpsentry->parray[pindex].info.stream.stream_naxis[0] > 1)
            {
                fpsentry->parray[pindex].info.stream.stream_ysize[0] =
                    data.image[ID].md[0].size[1];
            }
            else
            {
                fpsentry->parray[pindex].info.stream.stream_ysize[0] = 1;
            }

            if(fpsentry->parray[pindex].info.stream.stream_naxis[0] > 2)
            {
                fpsentry->parray[pindex].info.stream.stream_zsize[0] =
                    data.image[ID].md[0].size[2];
            }
            else
            {
                fpsentry->parray[pindex].info.stream.stream_zsize[0] = 1;
            }
        }

        if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_RUN_REQUIRED)
        {
            char msg[200];
            sprintf(msg,
                    "Loading stream %s",
                    fpsentry->parray[pindex].val.string[0]);
            functionparameter_outlog("LOADMEMSTREAM", "%s", msg);

            if(imLOC == STREAM_LOAD_SOURCE_NOTFOUND)
            {
                fpsentry->md->msgpindex[fpsentry->md->msgcnt] = pindex;
                fpsentry->md->msgcode[fpsentry->md->msgcnt] =
                    FPS_MSG_FLAG_ERROR;
                if(snprintf(fpsentry->md->message[fpsentry->md->msgcnt],
                            FUNCTION_PARAMETER_STRUCT_MSG_SIZE,
                            "cannot load stream %s",
                            fpsentry->parray[pindex].val.string[0]) < 0)
                {
                    PRINT_ERROR("snprintf error file %s line %d", __FILE__, __LINE__);
                }
                fpsentry->md->msgcnt++;
                if(fpsentry->md->msgcnt > FPS_NB_MSG - 1)
                {
                    // max number of msg reached, write to last one
                    fpsentry->md->msgcnt = FPS_NB_MSG - 1;
                }
                fpsentry->md->conferrcnt++;
                err = 1;
            }

        }

    }

    if(err == 1)
    {
        fpsentry->parray[pindex].fpflag |= FPFLAG_ERROR;
    }
    else
    {
        fpsentry->parray[pindex].fpflag &= ~FPFLAG_ERROR;
    }

    return err;
}




int functionparameter_CheckParametersAll(FUNCTION_PARAMETER_STRUCT *fpsentry)
{
    long NBparamMAX;
    long pindex;
    int  errcnt = 0;

    char msg[200];
    sprintf(msg, "%s", fpsentry->md->name);
    functionparameter_outlog("CHECKPARAMALL", "%s", msg);

    strcpy(fpsentry->md->message[0], "\0");
    NBparamMAX = fpsentry->md->NBparamMAX;

    // Check if Value is OK
    fpsentry->md->msgcnt     = 0;
    fpsentry->md->conferrcnt = 0;
    //    printf("Checking %d parameter entries\n", NBparam);
    for(pindex = 0; pindex < NBparamMAX; pindex++)
    {
        errcnt += functionparameter_CheckParameter(fpsentry, pindex);
    }

    // number of configuration errors - should be zero for run process to start
    fpsentry->md->conferrcnt = errcnt;

    if(errcnt == 0)
    {
        fpsentry->md->status |= FUNCTION_PARAMETER_STRUCT_STATUS_CHECKOK;
    }
    else
    {
        fpsentry->md->status &= ~FUNCTION_PARAMETER_STRUCT_STATUS_CHECKOK;
    }

    // compute write status

    for(pindex = 0; pindex < NBparamMAX; pindex++)
    {
        int writeOK; // do we have write permission ?

        // by default, adopt FPFLAG_WRITE flag
        if(fpsentry->parray[pindex].fpflag & FPFLAG_WRITE)
        {
            writeOK = 1;
        }
        else
        {
            writeOK = 0;
        }

        // if CONF running
        if(fpsentry->md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CONF)
        {
            if(fpsentry->parray[pindex].fpflag & FPFLAG_WRITECONF)
            {
                writeOK = 1;
            }
            else
            {
                writeOK = 0;
            }
        }

        // if RUN running
        if(fpsentry->md->status & FUNCTION_PARAMETER_STRUCT_STATUS_RUN)
        {
            if(fpsentry->parray[pindex].fpflag & FPFLAG_WRITERUN)
            {
                writeOK = 1;
            }
            else
            {
                writeOK = 0;
            }
        }

        if(writeOK == 0)
        {
            fpsentry->parray[pindex].fpflag &= ~FPFLAG_WRITESTATUS;
        }
        else
        {
            fpsentry->parray[pindex].fpflag |= FPFLAG_WRITESTATUS;
        }
    }

    fpsentry->md->signal &= ~FUNCTION_PARAMETER_STRUCT_SIGNAL_CHECKED;

    return 0;
}
