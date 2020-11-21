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

int function_parameter_add_entry(
    FUNCTION_PARAMETER_STRUCT *fps,
    const char                *keywordstring,
    const char                *descriptionstring,
    uint64_t                   type,
    uint64_t                   fpflag,
    void                      *valueptr
)
{
    long pindex = 0;
    char *pch;
    char tmpstring[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN * FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
    FUNCTION_PARAMETER *funcparamarray;

    funcparamarray = fps->parray;

    long NBparamMAX = -1;

    NBparamMAX = fps->md->NBparamMAX;





    // process keywordstring
    // if string starts with ".", insert fps name
    char keywordstringC[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN * FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
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
                FUNCTION_PARAMETER_KEYWORD_STRMAXLEN * FUNCTION_PARAMETER_KEYWORD_MAXLEVEL);
        strncpy(tmpstring, keywordstringC,
                FUNCTION_PARAMETER_KEYWORD_STRMAXLEN * FUNCTION_PARAMETER_KEYWORD_MAXLEVEL);
        funcparamarray[pindex].keywordlevel = 0;
        pch = strtok(tmpstring, ".");
        while(pch != NULL)
        {
            strncpy(funcparamarray[pindex].keyword[funcparamarray[pindex].keywordlevel],
                    pch, FUNCTION_PARAMETER_KEYWORD_STRMAXLEN);
            funcparamarray[pindex].keywordlevel++;
            pch = strtok(NULL, ".");
        }


        // Write description
        strncpy(funcparamarray[pindex].description, descriptionstring,
                FUNCTION_PARAMETER_DESCR_STRMAXLEN);

        // type
        funcparamarray[pindex].type = type;



        // Allocate value
        funcparamarray[pindex].cnt0 = 0; // not allocated

        // Default values
        switch(funcparamarray[pindex].type)
        {
        case FPTYPE_INT64 :
            funcparamarray[pindex].val.l[0] = 0;
            funcparamarray[pindex].val.l[1] = 0;
            funcparamarray[pindex].val.l[2] = 0;
            funcparamarray[pindex].val.l[3] = 0;
            break;

        case FPTYPE_FLOAT64 :
            funcparamarray[pindex].val.f[0] = 0.0;
            funcparamarray[pindex].val.f[1] = 0.0;
            funcparamarray[pindex].val.f[2] = 0.0;
            funcparamarray[pindex].val.f[3] = 0.0;
            break;

        case FPTYPE_FLOAT32 :
            funcparamarray[pindex].val.s[0] = 0.0;
            funcparamarray[pindex].val.s[1] = 0.0;
            funcparamarray[pindex].val.s[2] = 0.0;
            funcparamarray[pindex].val.s[3] = 0.0;
            break;

        case FPTYPE_PID :
            funcparamarray[pindex].val.pid[0] = 0;
            funcparamarray[pindex].val.pid[1] = 0;
            break;

        case FPTYPE_TIMESPEC :
            funcparamarray[pindex].val.ts[0].tv_sec = 0;
            funcparamarray[pindex].val.ts[0].tv_nsec = 0;
            funcparamarray[pindex].val.ts[1].tv_sec = 0;
            funcparamarray[pindex].val.ts[1].tv_nsec = 0;
            break;

        case FPTYPE_FILENAME :
            if(snprintf(funcparamarray[pindex].val.string[0], FUNCTION_PARAMETER_STRMAXLEN,
                        "NULL") < 0)
            {
                PRINT_ERROR("snprintf error");
            }
            if(snprintf(funcparamarray[pindex].val.string[1], FUNCTION_PARAMETER_STRMAXLEN,
                        "NULL") < 0)
            {
                PRINT_ERROR("snprintf error");
            }
            break;

        case FPTYPE_FITSFILENAME :
            if(snprintf(funcparamarray[pindex].val.string[0], FUNCTION_PARAMETER_STRMAXLEN,
                        "NULL") < 0)
            {
                PRINT_ERROR("snprintf error");
            }
            if(snprintf(funcparamarray[pindex].val.string[1], FUNCTION_PARAMETER_STRMAXLEN,
                        "NULL") < 0)
            {
                PRINT_ERROR("snprintf error");
            }
            break;

        case FPTYPE_EXECFILENAME :
            if(snprintf(funcparamarray[pindex].val.string[0], FUNCTION_PARAMETER_STRMAXLEN,
                        "NULL") < 0)
            {
                PRINT_ERROR("snprintf error");
            }
            if(snprintf(funcparamarray[pindex].val.string[1], FUNCTION_PARAMETER_STRMAXLEN,
                        "NULL") < 0)
            {
                PRINT_ERROR("snprintf error");
            }
            break;

        case FPTYPE_DIRNAME :
            if(snprintf(funcparamarray[pindex].val.string[0], FUNCTION_PARAMETER_STRMAXLEN,
                        "NULL") < 0)
            {
                PRINT_ERROR("snprintf error");
            }
            if(snprintf(funcparamarray[pindex].val.string[1], FUNCTION_PARAMETER_STRMAXLEN,
                        "NULL") < 0)
            {
                PRINT_ERROR("snprintf error");
            }
            break;

        case FPTYPE_STREAMNAME :
            if(snprintf(funcparamarray[pindex].val.string[0], FUNCTION_PARAMETER_STRMAXLEN,
                        "NULL") < 0)
            {
                PRINT_ERROR("snprintf error");
            }
            if(snprintf(funcparamarray[pindex].val.string[1], FUNCTION_PARAMETER_STRMAXLEN,
                        "NULL") < 0)
            {
                PRINT_ERROR("snprintf error");
            }
            break;

        case FPTYPE_STRING :
            if(snprintf(funcparamarray[pindex].val.string[0], FUNCTION_PARAMETER_STRMAXLEN,
                        "NULL") < 0)
            {
                PRINT_ERROR("snprintf error");
            }
            if(snprintf(funcparamarray[pindex].val.string[1], FUNCTION_PARAMETER_STRMAXLEN,
                        "NULL") < 0)
            {
                PRINT_ERROR("snprintf error");
            }
            break;

        case FPTYPE_ONOFF :
            funcparamarray[pindex].fpflag &= ~FPFLAG_ONOFF; // initialize state to OFF
            if(snprintf(funcparamarray[pindex].val.string[0], FUNCTION_PARAMETER_STRMAXLEN,
                        "OFF state") < 0)
            {
                PRINT_ERROR("snprintf error");
            }
            if(snprintf(funcparamarray[pindex].val.string[1], FUNCTION_PARAMETER_STRMAXLEN,
                        " ON state") < 0)
            {
                PRINT_ERROR("snprintf error");
            }
            break;

        case FPTYPE_FPSNAME :
            if(snprintf(funcparamarray[pindex].val.string[0], FUNCTION_PARAMETER_STRMAXLEN,
                        "NULL") < 0)
            {
                PRINT_ERROR("snprintf error");
            }
            if(snprintf(funcparamarray[pindex].val.string[1], FUNCTION_PARAMETER_STRMAXLEN,
                        "NULL") < 0)
            {
                PRINT_ERROR("snprintf error");
            }
            break;
        }



        if(valueptr != NULL)  // allocate value requested by function call
        {
            int64_t *valueptr_INT64;
            double  *valueptr_FLOAT64;
            float   *valueptr_FLOAT32;
            struct timespec *valueptr_ts;

            switch(funcparamarray[pindex].type)
            {

            case FPTYPE_INT64 :
                valueptr_INT64 = (int64_t *) valueptr;
                funcparamarray[pindex].val.l[0] = valueptr_INT64[0];
                funcparamarray[pindex].val.l[1] = valueptr_INT64[1];
                funcparamarray[pindex].val.l[2] = valueptr_INT64[2];
                funcparamarray[pindex].val.l[3] = valueptr_INT64[3];
                funcparamarray[pindex].cnt0++;
                break;

            case FPTYPE_FLOAT64 :
                valueptr_FLOAT64 = (double *) valueptr;
                funcparamarray[pindex].val.f[0] = valueptr_FLOAT64[0];
                funcparamarray[pindex].val.f[1] = valueptr_FLOAT64[1];
                funcparamarray[pindex].val.f[2] = valueptr_FLOAT64[2];
                funcparamarray[pindex].val.f[3] = valueptr_FLOAT64[3];
                funcparamarray[pindex].cnt0++;
                break;

            case FPTYPE_FLOAT32 :
                valueptr_FLOAT32 = (float *) valueptr;
                funcparamarray[pindex].val.s[0] = valueptr_FLOAT32[0];
                funcparamarray[pindex].val.s[1] = valueptr_FLOAT32[1];
                funcparamarray[pindex].val.s[2] = valueptr_FLOAT32[2];
                funcparamarray[pindex].val.s[3] = valueptr_FLOAT32[3];
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
                        FUNCTION_PARAMETER_STRMAXLEN);
                funcparamarray[pindex].cnt0++;
                break;

            case FPTYPE_FITSFILENAME :
                strncpy(funcparamarray[pindex].val.string[0], (char *) valueptr,
                        FUNCTION_PARAMETER_STRMAXLEN);
                funcparamarray[pindex].cnt0++;
                break;

            case FPTYPE_EXECFILENAME :
                strncpy(funcparamarray[pindex].val.string[0], (char *) valueptr,
                        FUNCTION_PARAMETER_STRMAXLEN);
                funcparamarray[pindex].cnt0++;
                break;

            case FPTYPE_DIRNAME :
                strncpy(funcparamarray[pindex].val.string[0], (char *) valueptr,
                        FUNCTION_PARAMETER_STRMAXLEN);
                funcparamarray[pindex].cnt0++;
                break;

            case FPTYPE_STREAMNAME :
                strncpy(funcparamarray[pindex].val.string[0], (char *) valueptr,
                        FUNCTION_PARAMETER_STRMAXLEN);
                funcparamarray[pindex].cnt0++;
                break;

            case FPTYPE_STRING :
                strncpy(funcparamarray[pindex].val.string[0], (char *) valueptr,
                        FUNCTION_PARAMETER_STRMAXLEN);
                funcparamarray[pindex].cnt0++;
                break;

            case FPTYPE_ONOFF : // already allocated through the status flag
                break;

            case FPTYPE_FPSNAME :
                strncpy(funcparamarray[pindex].val.string[0], (char *) valueptr,
                        FUNCTION_PARAMETER_STRMAXLEN);
                funcparamarray[pindex].cnt0++;
                break;
            }

            // RVAL = 2;  // default value entered
        }

    }





    /*

    	// READING PARAMETER FROM DISK


        // attempt to read value for filesystem
        char fname[200];
        FILE *fp;
        long tmpl;

        int RVAL = 0;
        // 0: parameter initialized to default value
        // 1: initialized using file value (read from disk)
        // 2: initialized to function argument value


        int index;
        // index = 0  : setval
        // index = 1  : minval
        // index = 2  : maxval


        for(index = 0; index < 3; index++)
        {
            switch(index)
            {
                case 0 :
                    functionparameter_GetFileName(fps, &funcparamarray[pindex], fname, "setval");
                    break;

                case 1 :
                    functionparameter_GetFileName(fps, &funcparamarray[pindex], fname, "minval");
                    break;

                case 2 :
                    functionparameter_GetFileName(fps, &funcparamarray[pindex], fname, "maxval");
                    break;

            }


            if((fp = fopen(fname, "r")) != NULL)
            {
                switch(funcparamarray[pindex].type)
                {

                    case FPTYPE_INT64 :
                        if(fscanf(fp, "%ld", &funcparamarray[pindex].val.l[index]) == 1)
                            if(index ==
                                    0)     // return value is set by setval, cnt0 tracks updates to setval, not to minval or maxval
                            {
                                RVAL = 1;
                                funcparamarray[pindex].cnt0++;
                            }
                        break;

                    case FPTYPE_FLOAT64 :
                        if(fscanf(fp, "%lf", &funcparamarray[pindex].val.f[index]) == 1)
                            if(index == 0)
                            {
                                RVAL = 1;
                                funcparamarray[pindex].cnt0++;
                            }
                        break;

                    case FPTYPE_FLOAT32 :
                        if(fscanf(fp, "%f", &funcparamarray[pindex].val.s[index]) == 1)
                            if(index == 0)
                            {
                                RVAL = 1;
                                funcparamarray[pindex].cnt0++;
                            }
                        break;

                    case FPTYPE_PID :
                        if(index == 0) // PID does not have min / max
                        {
                            if(fscanf(fp, "%d", &funcparamarray[pindex].val.pid[index]) == 1)
                            {
                                RVAL = 1;
                            }
                            funcparamarray[pindex].cnt0++;
                        }
                        break;

                    case FPTYPE_TIMESPEC :
                        if(fscanf(fp, "%ld %ld", &funcparamarray[pindex].val.ts[index].tv_sec,
                                  &funcparamarray[pindex].val.ts[index].tv_nsec) == 2)
                            if(index == 0)
                            {
                                RVAL = 1;
                                funcparamarray[pindex].cnt0++;
                            }
                        break;

                    case FPTYPE_FILENAME :
                        if(index == 0)    // FILENAME does not have min / max
                        {
                            if(fscanf(fp, "%s", funcparamarray[pindex].val.string[0]) == 1)
                            {
                                RVAL = 1;
                                funcparamarray[pindex].cnt0++;
                            }
                        }
                        break;

                    case FPTYPE_FITSFILENAME :
                        if(index == 0)    // FITSFILENAME does not have min / max
                        {
                            if(fscanf(fp, "%s", funcparamarray[pindex].val.string[0]) == 1)
                            {
                                RVAL = 1;
                                funcparamarray[pindex].cnt0++;
                            }
                        }
                        break;

                    case FPTYPE_EXECFILENAME :
                        if(index == 0)    // EXECFILENAME does not have min / max
                        {
                            if(fscanf(fp, "%s", funcparamarray[pindex].val.string[0]) == 1)
                            {
                                RVAL = 1;
                                funcparamarray[pindex].cnt0++;
                            }
                        }
                        break;


                    case FPTYPE_DIRNAME :
                        if(index == 0)    // DIRNAME does not have min / max
                        {
                            if(fscanf(fp, "%s", funcparamarray[pindex].val.string[0]) == 1)
                            {
                                RVAL = 1;
                                funcparamarray[pindex].cnt0++;
                            }
                        }
                        break;

                    case FPTYPE_STREAMNAME :
                        if(index == 0)    // STREAMNAME does not have min / max
                        {
                            if(fscanf(fp, "%s", funcparamarray[pindex].val.string[0]) == 1)
                            {
                                RVAL = 1;
                                funcparamarray[pindex].cnt0++;
                            }
                        }
                        break;

                    case FPTYPE_STRING :
                        if(index == 0)    // STRING does not have min / max
                        {
                            if(fscanf(fp, "%s", funcparamarray[pindex].val.string[0]) == 1)
                            {
                                RVAL = 1;
                                funcparamarray[pindex].cnt0++;
                            }
                        }
                        break;

                    case FPTYPE_ONOFF :
                        if(index == 0)
                        {
                            if(fscanf(fp, "%ld", &tmpl) == 1)
                            {
                                if(tmpl == 1)
                                {
                                    funcparamarray[pindex].fpflag |= FPFLAG_ONOFF;
                                }
                                else
                                {
                                    funcparamarray[pindex].fpflag &= ~FPFLAG_ONOFF;
                                }

                                funcparamarray[pindex].cnt0++;
                            }
                        }
                        break;


                    case FPTYPE_FPSNAME :
                        if(index == 0)    // FPSNAME does not have min / max
                        {
                            if(fscanf(fp, "%s", funcparamarray[pindex].val.string[0]) == 1)
                            {
                                RVAL = 1;
                                funcparamarray[pindex].cnt0++;
                            }
                        }
                        break;

                }
                fclose(fp);


            }

        }




    	// WRITING PARAMETER TO DISK
    	//

        if(RVAL == 0)
        {
            functionparameter_WriteParameterToDisk(fps, pindex, "setval",
                                                   "AddEntry created");
            if(funcparamarray[pindex].fpflag |= FPFLAG_MINLIMIT)
            {
                functionparameter_WriteParameterToDisk(fps, pindex, "minval",
                                                       "AddEntry created");
            }
            if(funcparamarray[pindex].fpflag |= FPFLAG_MAXLIMIT)
            {
                functionparameter_WriteParameterToDisk(fps, pindex, "maxval",
                                                       "AddEntry created");
            }
        }

        if(RVAL == 2)
        {
            functionparameter_WriteParameterToDisk(fps, pindex, "setval",
                                                   "AddEntry argument");
            if(funcparamarray[pindex].fpflag |= FPFLAG_MINLIMIT)
            {
                functionparameter_WriteParameterToDisk(fps, pindex, "minval",
                                                       "AddEntry argument");
            }
            if(funcparamarray[pindex].fpflag |= FPFLAG_MAXLIMIT)
            {
                functionparameter_WriteParameterToDisk(fps, pindex, "maxval",
                                                       "AddEntry argument");
            }
        }

        if(RVAL != 0)
        {
            functionparameter_WriteParameterToDisk(fps, pindex, "fpsname", "AddEntry");
            functionparameter_WriteParameterToDisk(fps, pindex, "fpsdir", "AddEntry");
            functionparameter_WriteParameterToDisk(fps, pindex, "status", "AddEntry");
        }
    */

    return pindex;
}



