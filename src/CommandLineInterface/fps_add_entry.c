/**
 * @file    fps_add_entry.c
 * @brief   add parameter entry to FPS
 */



#include "CommandLineInterface/CLIcore.h"



/** @brief Add parameter to database with default settings
 *
 * If entry already exists, do not modify it
 *
 */

errno_t function_parameter_add_entry(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char                *keywordstring,
    const char                *descriptionstring,
    uint64_t                   type,
    uint64_t                   fpflag,
    void                      *valueptr,
    long                      *pindexptr
)
{
    DEBUG_TRACE_FSTART("%s %s", keywordstring, descriptionstring);

    long pindex = 0;
    char *pch;
    char tmpstring[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN *
                                                        FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
    FUNCTION_PARAMETER *funcparamarray;

    funcparamarray = fps->parray;

    long NBparamMAX = -1;

    NBparamMAX = fps->md->NBparamMAX;





    // process keywordstring
    // if string starts with ".", insert fps name
    char keywordstringC[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN *
                                                             FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
    if(keywordstring[0] == '.')
    {
        //printf("--------------- keywstring \"%s\" starts with dot -> adding \"%s\"\n", keywordstring, fps->md->name);
        sprintf(keywordstringC, "%s%s", fps->md->name, keywordstring);
    }
    else
    {
        //printf("--------------- keywstring \"%s\" unchanged\n", keywordstring);
        strcpy(keywordstringC, keywordstring);
    }



    // scan for existing keyword
    int scanOK = 0;
    long pindexscan;
    for(pindexscan = 0; pindexscan < NBparamMAX; pindexscan++)
    {
        if(strcmp(keywordstringC, funcparamarray[pindexscan].keywordfull) == 0)
        {
            pindex = pindexscan;
            scanOK = 1;
        }
    }

    if(scanOK == 0) // not found
    {
        // scan for first available entry
        pindex = 0;
        while((funcparamarray[pindex].fpflag & FPFLAG_ACTIVE) && (pindex < NBparamMAX))
        {
            pindex++;
        }

        if(pindex == NBparamMAX)
        {
            printf("ERROR [%s line %d]: NBparamMAX %ld limit reached\n", __FILE__, __LINE__,
                   NBparamMAX);
            fflush(stdout);
            printf("STEP %s %d\n", __FILE__, __LINE__);
            fflush(stdout);
            exit(0);
        }







        funcparamarray[pindex].fpflag = fpflag;



        // break full keyword into keywords
        strncpy(funcparamarray[pindex].keywordfull, keywordstringC,
                FUNCTION_PARAMETER_KEYWORD_STRMAXLEN * FUNCTION_PARAMETER_KEYWORD_MAXLEVEL - 1);
        strncpy(tmpstring, keywordstringC,
                FUNCTION_PARAMETER_KEYWORD_STRMAXLEN * FUNCTION_PARAMETER_KEYWORD_MAXLEVEL - 1);
        funcparamarray[pindex].keywordlevel = 0;
        pch = strtok(tmpstring, ".");
        while(pch != NULL)
        {
            strncpy(funcparamarray[pindex].keyword[funcparamarray[pindex].keywordlevel],
                    pch, FUNCTION_PARAMETER_KEYWORD_STRMAXLEN - 1);
            funcparamarray[pindex].keywordlevel++;
            pch = strtok(NULL, ".");
        }


        // Write description
        strncpy(funcparamarray[pindex].description, descriptionstring,
                FUNCTION_PARAMETER_DESCR_STRMAXLEN - 1);

        // type
        funcparamarray[pindex].type = type;



        // Allocate value
        funcparamarray[pindex].cnt0 = 0; // not allocated

        // Default values
        switch(funcparamarray[pindex].type)
        {
            case FPTYPE_INT32 :
                funcparamarray[pindex].val.i32[0] = 0;
                funcparamarray[pindex].val.i32[1] = 0;
                funcparamarray[pindex].val.i32[2] = 0;
                funcparamarray[pindex].val.i32[3] = 0;
                break;

            case FPTYPE_UINT32 :
                funcparamarray[pindex].val.ui32[0] = 0;
                funcparamarray[pindex].val.ui32[1] = 0;
                funcparamarray[pindex].val.ui32[2] = 0;
                funcparamarray[pindex].val.ui32[3] = 0;
                break;

            case FPTYPE_INT64 :
                funcparamarray[pindex].val.i64[0] = 0;
                funcparamarray[pindex].val.i64[1] = 0;
                funcparamarray[pindex].val.i64[2] = 0;
                funcparamarray[pindex].val.i64[3] = 0;
                break;

            case FPTYPE_UINT64 :
                funcparamarray[pindex].val.ui64[0] = 0;
                funcparamarray[pindex].val.ui64[1] = 0;
                funcparamarray[pindex].val.ui64[2] = 0;
                funcparamarray[pindex].val.ui64[3] = 0;
                break;

            case FPTYPE_FLOAT64 :
                funcparamarray[pindex].val.f64[0] = 0.0;
                funcparamarray[pindex].val.f64[1] = 0.0;
                funcparamarray[pindex].val.f64[2] = 0.0;
                funcparamarray[pindex].val.f64[3] = 0.0;
                break;

            case FPTYPE_FLOAT32 :
                funcparamarray[pindex].val.f32[0] = 0.0;
                funcparamarray[pindex].val.f32[1] = 0.0;
                funcparamarray[pindex].val.f32[2] = 0.0;
                funcparamarray[pindex].val.f32[3] = 0.0;
                break;

            case FPTYPE_PID :
                funcparamarray[pindex].val.pid[0] = 0;
                funcparamarray[pindex].val.pid[1] = 0;
                break;

            case FPTYPE_TIMESPEC :
                funcparamarray[pindex].val.ts[0].tv_sec  = 0;
                funcparamarray[pindex].val.ts[0].tv_nsec = 0;
                funcparamarray[pindex].val.ts[1].tv_sec  = 0;
                funcparamarray[pindex].val.ts[1].tv_nsec = 0;
                break;

            case FPTYPE_FILENAME :
                SNPRINTF_CHECK(funcparamarray[pindex].val.string[0],
                               FUNCTION_PARAMETER_STRMAXLEN, "NULL");
                SNPRINTF_CHECK(funcparamarray[pindex].val.string[1],
                               FUNCTION_PARAMETER_STRMAXLEN, "NULL");
                break;

            case FPTYPE_FITSFILENAME :
                SNPRINTF_CHECK(funcparamarray[pindex].val.string[0],
                               FUNCTION_PARAMETER_STRMAXLEN, "NULL");
                SNPRINTF_CHECK(funcparamarray[pindex].val.string[1],
                               FUNCTION_PARAMETER_STRMAXLEN, "NULL");
                break;

            case FPTYPE_EXECFILENAME :
                SNPRINTF_CHECK(funcparamarray[pindex].val.string[0],
                               FUNCTION_PARAMETER_STRMAXLEN, "NULL");
                SNPRINTF_CHECK(funcparamarray[pindex].val.string[1],
                               FUNCTION_PARAMETER_STRMAXLEN, "NULL");
                break;

            case FPTYPE_DIRNAME :
                SNPRINTF_CHECK(funcparamarray[pindex].val.string[0],
                               FUNCTION_PARAMETER_STRMAXLEN, "NULL");
                SNPRINTF_CHECK(funcparamarray[pindex].val.string[1],
                               FUNCTION_PARAMETER_STRMAXLEN, "NULL");
                break;

            case FPTYPE_STREAMNAME :
                SNPRINTF_CHECK(funcparamarray[pindex].val.string[0],
                               FUNCTION_PARAMETER_STRMAXLEN, "NULL");
                SNPRINTF_CHECK(funcparamarray[pindex].val.string[1],
                               FUNCTION_PARAMETER_STRMAXLEN, "NULL");
                break;

            case FPTYPE_STRING :
                SNPRINTF_CHECK(funcparamarray[pindex].val.string[0],
                               FUNCTION_PARAMETER_STRMAXLEN, "NULL");
                SNPRINTF_CHECK(funcparamarray[pindex].val.string[1],
                               FUNCTION_PARAMETER_STRMAXLEN, "NULL");
                break;

            case FPTYPE_ONOFF :
                funcparamarray[pindex].fpflag &= ~FPFLAG_ONOFF; // initialize state to OFF
                funcparamarray[pindex].val.ui64[0] = 0;
                break;

            case FPTYPE_FPSNAME :
                SNPRINTF_CHECK(funcparamarray[pindex].val.string[0],
                               FUNCTION_PARAMETER_STRMAXLEN, "NULL");
                SNPRINTF_CHECK(funcparamarray[pindex].val.string[1],
                               FUNCTION_PARAMETER_STRMAXLEN, "NULL");
                break;
        }



        if(valueptr != NULL)  // allocate value requested by function call
        {
            int32_t  *valueptr_INT32;
            uint32_t *valueptr_UINT32;
            int64_t  *valueptr_INT64;
            uint64_t *valueptr_UINT64;
            double   *valueptr_FLOAT64;
            float    *valueptr_FLOAT32;
            struct timespec *valueptr_ts;

            switch(funcparamarray[pindex].type)
            {

                case FPTYPE_INT32 :
                    valueptr_INT32 = (int32_t *) valueptr;
                    funcparamarray[pindex].val.i32[0] = valueptr_INT32[0];
                    funcparamarray[pindex].val.i32[1] = valueptr_INT32[1];
                    funcparamarray[pindex].val.i32[2] = valueptr_INT32[2];
                    funcparamarray[pindex].val.i32[3] = valueptr_INT32[3];
                    funcparamarray[pindex].cnt0++;
                    break;

                case FPTYPE_UINT32 :
                    valueptr_UINT32 = (uint32_t *) valueptr;
                    funcparamarray[pindex].val.ui32[0] = valueptr_UINT32[0];
                    funcparamarray[pindex].val.ui32[1] = valueptr_UINT32[1];
                    funcparamarray[pindex].val.ui32[2] = valueptr_UINT32[2];
                    funcparamarray[pindex].val.ui32[3] = valueptr_UINT32[3];
                    funcparamarray[pindex].cnt0++;
                    break;

                case FPTYPE_INT64 :
                    valueptr_INT64 = (int64_t *) valueptr;
                    funcparamarray[pindex].val.i64[0] = valueptr_INT64[0];
                    funcparamarray[pindex].val.i64[1] = valueptr_INT64[1];
                    funcparamarray[pindex].val.i64[2] = valueptr_INT64[2];
                    funcparamarray[pindex].val.i64[3] = valueptr_INT64[3];
                    funcparamarray[pindex].cnt0++;
                    break;

                case FPTYPE_UINT64 :
                    valueptr_UINT64 = (uint64_t *) valueptr;
                    funcparamarray[pindex].val.ui64[0] = valueptr_UINT64[0];
                    funcparamarray[pindex].val.ui64[1] = valueptr_UINT64[1];
                    funcparamarray[pindex].val.ui64[2] = valueptr_UINT64[2];
                    funcparamarray[pindex].val.ui64[3] = valueptr_UINT64[3];
                    funcparamarray[pindex].cnt0++;
                    break;

                case FPTYPE_FLOAT64 :
                    valueptr_FLOAT64 = (double *) valueptr;
                    funcparamarray[pindex].val.f64[0] = valueptr_FLOAT64[0];
                    funcparamarray[pindex].val.f64[1] = valueptr_FLOAT64[1];
                    funcparamarray[pindex].val.f64[2] = valueptr_FLOAT64[2];
                    funcparamarray[pindex].val.f64[3] = valueptr_FLOAT64[3];
                    funcparamarray[pindex].cnt0++;
                    break;

                case FPTYPE_FLOAT32 :
                    valueptr_FLOAT32 = (float *) valueptr;
                    funcparamarray[pindex].val.f32[0] = valueptr_FLOAT32[0];
                    funcparamarray[pindex].val.f32[1] = valueptr_FLOAT32[1];
                    funcparamarray[pindex].val.f32[2] = valueptr_FLOAT32[2];
                    funcparamarray[pindex].val.f32[3] = valueptr_FLOAT32[3];
                    funcparamarray[pindex].cnt0++;
                    break;

                case FPTYPE_PID :
                    valueptr_INT64 = (int64_t *) valueptr;
                    funcparamarray[pindex].val.pid[0] = (pid_t)(*valueptr_INT64);
                    funcparamarray[pindex].cnt0++;
                    break;

                case FPTYPE_TIMESPEC:
                    valueptr_ts = (struct timespec *) valueptr;
                    funcparamarray[pindex].val.ts[0] = *valueptr_ts;
                    funcparamarray[pindex].cnt0++;
                    break;

                case FPTYPE_FILENAME :
                    strncpy(funcparamarray[pindex].val.string[0], (char *) valueptr,
                            FUNCTION_PARAMETER_STRMAXLEN - 1);
                    funcparamarray[pindex].cnt0++;
                    break;

                case FPTYPE_FITSFILENAME :
                    strncpy(funcparamarray[pindex].val.string[0], (char *) valueptr,
                            FUNCTION_PARAMETER_STRMAXLEN - 1);
                    funcparamarray[pindex].cnt0++;
                    break;

                case FPTYPE_EXECFILENAME :
                    strncpy(funcparamarray[pindex].val.string[0], (char *) valueptr,
                            FUNCTION_PARAMETER_STRMAXLEN - 1);
                    funcparamarray[pindex].cnt0++;
                    break;

                case FPTYPE_DIRNAME :
                    strncpy(funcparamarray[pindex].val.string[0], (char *) valueptr,
                            FUNCTION_PARAMETER_STRMAXLEN - 1);
                    funcparamarray[pindex].cnt0++;
                    break;

                case FPTYPE_STREAMNAME :
                    strncpy(funcparamarray[pindex].val.string[0], (char *) valueptr,
                            FUNCTION_PARAMETER_STRMAXLEN - 1);
                    funcparamarray[pindex].cnt0++;
                    break;

                case FPTYPE_STRING :
                    strncpy(funcparamarray[pindex].val.string[0], (char *) valueptr,
                            FUNCTION_PARAMETER_STRMAXLEN - 1);
                    funcparamarray[pindex].cnt0++;
                    break;

                case FPTYPE_ONOFF :
                    funcparamarray[pindex].val.ui64[0] = 0;
                    funcparamarray[pindex].cnt0++;
                    break;

                case FPTYPE_FPSNAME :
                    strncpy(funcparamarray[pindex].val.string[0], (char *) valueptr,
                            FUNCTION_PARAMETER_STRMAXLEN - 1);
                    funcparamarray[pindex].cnt0++;
                    break;
            }

            // RVAL = 2;  // default value entered
        }

    }



    if(pindexptr != NULL)
    {
        *pindexptr = pindex;
    }

    DEBUG_TRACE_FEXIT();

    return RETURN_SUCCESS;
}



