/**
 * @file    loadmemstream.c
 * @brief   load memory stream
 *
 * Scan for source location, load stream of FITS file
 */

#include "CommandLineInterface/CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "COREMOD_iofits_common.h"
#include "loadfits.h"
#include "savefits.h"

extern COREMOD_IOFITS_DATA COREMOD_iofits_data;



// High-level load to stream


imageID COREMOD_IOFITS_LoadMemStream(
    const char *sname,
    uint64_t   *streamflag,
    uint32_t   *imLOC
)
{
    imageID ID = -1;


    // toggles to 1 if updating shared mem
    __attribute__((unused)) int updateSHAREMEM = 0;


    int updateCONFFITS = 0; // toggles to 1 if updating CONF FITS


    int MEMLOADREPORT = 0;
    if(FPFLAG_STREAM_MEMLOADREPORT & *streamflag)   // write report to disk
    {
        MEMLOADREPORT = 1;
    }
    MEMLOADREPORT = 1;// TMP


    *imLOC = STREAM_LOAD_SOURCE_NOTFOUND;
    printf("%s %d imLOC = %d\n", __FILE__, __LINE__, *imLOC);




    if(strcmp(sname, "NULL") == 0)   // don't bother looking for it
    {
        *imLOC = STREAM_LOAD_SOURCE_NULL;
        ID = -1;
    }


    int imLOCALMEM;
    if(*imLOC == STREAM_LOAD_SOURCE_NOTFOUND)   // still searching
    {
        // Does image exist in memory ?
        ID = image_ID(sname);
        if(ID == -1)
        {
            imLOCALMEM = 0;
            if(MEMLOADREPORT == 1)
            {
                char msg[STRINGMAXLEN_FPS_LOGMSG];
                SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                               "%s stream not in local memory", sname);
                functionparameter_outlog("LOADMEMSTREAM", msg);
            }
        }
        else
        {
            imLOCALMEM = 1;
            if(MEMLOADREPORT == 1)
            {
                char msg[STRINGMAXLEN_FPS_LOGMSG];
                SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                               "%s stream in local memory", sname);
                functionparameter_outlog("LOADMEMSTREAM", msg);
            }
        }


        //    printf("imLOC = %d\n", *imLOC);


        // FORCE_LOCALMEM
        if(FPFLAG_STREAM_LOAD_FORCE_LOCALMEM & *streamflag)
        {
            if(imLOCALMEM == 0)
            {
                *imLOC = STREAM_LOAD_SOURCE_EXITFAILURE; // fail
                if(MEMLOADREPORT)
                {
                    char msg[STRINGMAXLEN_FPS_LOGMSG];
                    SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                                   "%s EXITFAIL STREAM_LOAD_FORCE_LOCALMEM: Image does not exist in local memory",
                                   sname);
                    functionparameter_outlog("LOADMEMSTREAM", msg);
                }
            }
            else
            {
                *imLOC = STREAM_LOAD_SOURCE_LOCALMEM;
                if(MEMLOADREPORT == 1)
                {
                    char msg[STRINGMAXLEN_FPS_LOGMSG];
                    SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                                   "%s SUCCESS STREAM_LOAD_FORCE_LOCALMEM", sname);
                    functionparameter_outlog("LOADMEMSTREAM", msg);
                    sprintf(msg, "%s imLOC %u", sname, *imLOC);
                    functionparameter_outlog("LOADMEMSTREAM", msg);
                }
            }
        }
    }


    // FORCE_SHAREMEM
    if(*imLOC == STREAM_LOAD_SOURCE_NOTFOUND)   // still searching
    {
        if(FPFLAG_STREAM_LOAD_FORCE_SHAREMEM & *streamflag)
        {
            // search SHAREMEM
            ID = read_sharedmem_image(sname);
            if(ID == -1)
            {
                *imLOC = STREAM_LOAD_SOURCE_EXITFAILURE; // fail
                if(MEMLOADREPORT == 1)
                {
                    char msg[STRINGMAXLEN_FPS_LOGMSG];
                    SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                                   "%s EXITFAIL STREAM_LOAD_FORCE_SHAREDMEM: Image does not exist in shared memory",
                                   sname);
                    functionparameter_outlog("LOADMEMSTREAM", msg);
                }
            }
            else
            {
                *imLOC = STREAM_LOAD_SOURCE_SHAREMEM;
                if(MEMLOADREPORT == 1)
                {
                    char msg[STRINGMAXLEN_FPS_LOGMSG];
                    SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                                   "%s SUCCESS STREAM_LOAD_FORCE_SHAREDMEM", sname);
                    functionparameter_outlog("LOADMEMSTREAM", msg);
                    SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                                   "%s imLOC %u", sname, *imLOC);
                    functionparameter_outlog("LOADMEMSTREAM", msg);
                }
            }
        }
    }


    // FORCE_CONFFITS
    if(*imLOC == STREAM_LOAD_SOURCE_NOTFOUND)   // still searching
    {
        if(FPFLAG_STREAM_LOAD_FORCE_CONFFITS & *streamflag)
        {
            // search CONFFITS
            char fname[STRINGMAXLEN_FULLFILENAME];
            WRITE_FULLFILENAME(fname, "./conf/shmim.%s.fits", sname);
            load_fits(fname, sname, 0, &ID);
            if(ID == -1)
            {
                *imLOC = STREAM_LOAD_SOURCE_EXITFAILURE; // fail
                if(MEMLOADREPORT == 1)
                {
                    char msg[STRINGMAXLEN_FPS_LOGMSG];
                    SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                                   "%s EXITFAIL STREAM_LOAD_FORCE_CONFFITS: Image does not exist as conf FITS",
                                   sname);
                    functionparameter_outlog("LOADMEMSTREAM", msg);
                }
            }
            else
            {
                *imLOC = STREAM_LOAD_SOURCE_CONFFITS;
                if(MEMLOADREPORT == 1)
                {
                    char msg[STRINGMAXLEN_FPS_LOGMSG];
                    SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                                   "%s SUCCESS STREAM_LOAD_FORCE_CONFFITS", sname);
                    functionparameter_outlog("LOADMEMSTREAM", msg);
                    SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                                   "%s imLOC %u", sname, *imLOC);
                    functionparameter_outlog("LOADMEMSTREAM", msg);
                }
            }
        }
    }


    // FORCE_CONFNAME
    if(*imLOC == STREAM_LOAD_SOURCE_NOTFOUND)   // still searching
    {
        if(FPFLAG_STREAM_LOAD_FORCE_CONFNAME & *streamflag)
        {
            // search CONFNAME
            FILE *fp;
            char fname[STRINGMAXLEN_FULLFILENAME] = "";
            char streamfname[STRINGMAXLEN_FULLFILENAME] = "";
            int fscanfcnt = 0;

            WRITE_FULLFILENAME(fname, "./conf/shmim.%s.fname.txt", sname);

            fp = fopen(fname, "r");
            if(fp == NULL)
            {
                printf("ERROR: stream %s could not be loaded from CONF\n", sname);
                *imLOC = STREAM_LOAD_SOURCE_EXITFAILURE; // fail
                if(MEMLOADREPORT == 1)
                {
                    char msg[STRINGMAXLEN_FPS_LOGMSG];
                    SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                                   "%s EXITFAIL STREAM_LOAD_FORCE_CONFNAME: File %s does not exist", sname, fname);
                    functionparameter_outlog("LOADMEMSTREAM", msg);
                    SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                                   "%s imLOC %u", sname, *imLOC);
                    functionparameter_outlog("LOADMEMSTREAM", msg);
                }
            }
            else
            {
                fscanfcnt = fscanf(fp, "%s", streamfname);
                if(fscanfcnt == EOF)
                {
                    if(ferror(fp))
                    {
                        perror("fscanf");
                        *imLOC = STREAM_LOAD_SOURCE_EXITFAILURE; // fail
                        if(MEMLOADREPORT == 1)
                        {
                            char msg[STRINGMAXLEN_FPS_LOGMSG];
                            SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                                           "%s EXITFAIL STREAM_LOAD_FORCE_CONFNAME: fscanf cannot read stream fname",
                                           sname);
                            functionparameter_outlog("LOADMEMSTREAM", msg);
                        }
                    }
                    else
                    {
                        fprintf(stderr,
                                "Error: fscanf reached end of file, no matching characters, no matching failure\n");
                        *imLOC = STREAM_LOAD_SOURCE_EXITFAILURE; // fail
                        if(MEMLOADREPORT == 1)
                        {
                            char msg[STRINGMAXLEN_FPS_LOGMSG];
                            SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                                           "%s EXITFAIL STREAM_LOAD_FORCE_CONFNAME: fscanf reached end of file, no matching characters",
                                           sname);
                            functionparameter_outlog("LOADMEMSTREAM", msg);
                            SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                                           "%s imLOC %u", sname, *imLOC);
                            functionparameter_outlog("LOADMEMSTREAM", msg);
                        }
                    }
                }
                fclose(fp);
            }


            if(*imLOC == STREAM_LOAD_SOURCE_NOTFOUND)
            {
                if(fscanfcnt > 0)
                {
                    {
                        load_fits(streamfname, sname, 0, &ID);
                        if(ID == -1)
                        {
                            *imLOC = STREAM_LOAD_SOURCE_EXITFAILURE; // fail
                            if(MEMLOADREPORT == 1)
                            {
                                char msg[STRINGMAXLEN_FPS_LOGMSG];
                                SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                                               "%s EXITFAIL STREAM_LOAD_FORCE_CONFNAME: cannot load stream fname", sname);
                                functionparameter_outlog("LOADMEMSTREAM", msg);
                                SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                                               "%s imLOC %u", sname, *imLOC);
                                functionparameter_outlog("LOADMEMSTREAM", msg);
                            }
                        }
                        else
                        {
                            *imLOC = STREAM_LOAD_SOURCE_CONFNAME;
                            if(MEMLOADREPORT == 1)
                            {
                                char msg[STRINGMAXLEN_FPS_LOGMSG];
                                SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                                               "%s SUCCESS STREAM_LOAD_FORCE_CONFFITS", sname);
                                functionparameter_outlog("LOADMEMSTREAM", msg);
                                SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                                               "%s imLOC %u", sname, *imLOC);
                                functionparameter_outlog("LOADMEMSTREAM", msg);
                            }
                        }
                    }
                }
            }
        }
    }



    // SEARCH LOCALMEM
    if(*imLOC == STREAM_LOAD_SOURCE_NOTFOUND)   // still searching
    {
        //printf("imLOC = %d\n", *imLOC);
        if(!(FPFLAG_STREAM_LOAD_SKIPSEARCH_LOCALMEM & *streamflag))
        {
            if(MEMLOADREPORT == 1)
            {
                char msg[STRINGMAXLEN_FPS_LOGMSG];
                SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                               "%s Search LOCALMEM", sname);
                functionparameter_outlog("LOADMEMSTREAM", msg);
            }
            if(imLOCALMEM == 1)
            {
                *imLOC = STREAM_LOAD_SOURCE_LOCALMEM;
                if(MEMLOADREPORT == 1)
                {
                    char msg[STRINGMAXLEN_FPS_LOGMSG];
                    SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                                   "%s SUCCESS found image in LOCALMEM", sname);
                    functionparameter_outlog("LOADMEMSTREAM", msg);
                    SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG, "%s imLOC %u", sname, *imLOC);
                    functionparameter_outlog("LOADMEMSTREAM", msg);
                }
            }
            else
            {
                char msg[STRINGMAXLEN_FPS_LOGMSG];
                SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG, "%s localmem stream not found",
                               sname);
                functionparameter_outlog("LOADMEMSTREAM", msg);
            }
        }
        //printf("imLOC = %d\n", *imLOC);
    }


    // SEARCH SHAREMEM
    if(*imLOC == STREAM_LOAD_SOURCE_NOTFOUND)   // still searching
    {
        //printf("imLOC = %d\n", *imLOC);
        if(!(FPFLAG_STREAM_LOAD_SKIPSEARCH_SHAREMEM & *streamflag))
        {
            if(MEMLOADREPORT == 1)
            {
                char msg[STRINGMAXLEN_FPS_LOGMSG];
                SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                               "%s Search SHAREMEM", sname);
                functionparameter_outlog("LOADMEMSTREAM", msg);
            }
            ID = read_sharedmem_image(sname);
            if(ID != -1)
            {
                *imLOC = STREAM_LOAD_SOURCE_SHAREMEM;
                if(MEMLOADREPORT == 1)
                {
                    char msg[STRINGMAXLEN_FPS_LOGMSG];
                    SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                                   "%s SUCCESS found image in SHAREMEM", sname);
                    functionparameter_outlog("LOADMEMSTREAM", msg);
                    SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                                   "%s imLOC %u", sname, *imLOC);
                    functionparameter_outlog("LOADMEMSTREAM", msg);
                }
            }
            else
            {
                char msg[STRINGMAXLEN_FPS_LOGMSG];
                SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                               "%s sharedmem stream not found", sname);
                functionparameter_outlog("LOADMEMSTREAM", msg);
            }
        }
        //printf("imLOC = %d\n", *imLOC);
    }


    // SEARCH CONFFITS
    if(*imLOC == STREAM_LOAD_SOURCE_NOTFOUND)   // still searching
    {
        if(!(FPFLAG_STREAM_LOAD_SKIPSEARCH_CONFFITS & *streamflag))
        {
            if(MEMLOADREPORT == 1)
            {
                char msg[STRINGMAXLEN_FPS_LOGMSG];
                SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                               "%s Search CONFFITS", sname);
                functionparameter_outlog("LOADMEMSTREAM", msg);
            }
            //printf("imLOC = %d\n", *imLOC);
            char fname[STRINGMAXLEN_FULLFILENAME];
            WRITE_FULLFILENAME(fname, "./conf/shmim.%s.fits", sname);
            load_fits(fname, sname, 0, &ID);
            if(ID != -1)
            {
                *imLOC = STREAM_LOAD_SOURCE_CONFFITS;
                if(MEMLOADREPORT == 1)
                {
                    char msg[STRINGMAXLEN_FPS_LOGMSG];
                    SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                                   "%s SUCCESS found image in CONFFITS", sname);
                    functionparameter_outlog("LOADMEMSTREAM", msg);
                    SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                                   "%s imLOC %u", sname, *imLOC);
                    functionparameter_outlog("LOADMEMSTREAM", msg);
                }
            }
            else
            {
                char msg[STRINGMAXLEN_FPS_LOGMSG];
                SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                               "%s File %s not found", sname, fname);
                functionparameter_outlog("LOADMEMSTREAM", msg);
            }
            // printf("imLOC = %d\n", *imLOC);
        }
    }


    // SEARCH CONFNAME
    if(*imLOC == STREAM_LOAD_SOURCE_NOTFOUND)   // still searching
    {
        if(!(FPFLAG_STREAM_LOAD_SKIPSEARCH_CONFNAME & *streamflag))
        {
            if(MEMLOADREPORT == 1)
            {
                char msg[STRINGMAXLEN_FPS_LOGMSG];
                SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                               "%s Search CONFNAME", sname);
                functionparameter_outlog("LOADMEMSTREAM", msg);
            }

            //printf("imLOC = %d\n", *imLOC);
            FILE *fp;
            char fname[200] = "";
            char streamfname[200] = "";
            int fscanfcnt = 0;

            sprintf(fname, "./conf/shmim.%s.fname.txt", sname);

            fp = fopen(fname, "r");
            if(fp == NULL)
            {
                printf("ERROR: stream %s could not be loaded from CONF\n", sname);
                if(MEMLOADREPORT == 1)
                {
                    char msg[STRINGMAXLEN_FPS_LOGMSG];
                    SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                                   "%s Cannot find CONFNAME file %s", sname, fname);
                    functionparameter_outlog("LOADMEMSTREAM", msg);
                    // don't fail... keep going
                }
            }
            else
            {
                fscanfcnt = fscanf(fp, "%s", streamfname);
                if(fscanfcnt == EOF)
                {
                    if(ferror(fp))
                    {
                        perror("fscanf");
                        *imLOC = STREAM_LOAD_SOURCE_EXITFAILURE; // fail
                        if(MEMLOADREPORT == 1)
                        {
                            char msg[STRINGMAXLEN_FPS_LOGMSG];
                            SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                                           "%s EXITFAILURE fscanf error reading %s", sname,  fname);
                            functionparameter_outlog("LOADMEMSTREAM", msg);
                            SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                                           "%s imLOC %u", sname, *imLOC);
                            functionparameter_outlog("LOADMEMSTREAM", msg);
                        }
                    }
                    else
                    {
                        fprintf(stderr,
                                "Error: fscanf reached end of file, no matching characters, no matching failure\n");
                        *imLOC = STREAM_LOAD_SOURCE_EXITFAILURE; // fail
                        if(MEMLOADREPORT == 1)
                        {
                            char msg[STRINGMAXLEN_FPS_LOGMSG];
                            SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                                           "%s EXITFAILURE fscanf error reading %s. fscanf reached end of file, no matching characters",
                                           sname, fname);
                            functionparameter_outlog("LOADMEMSTREAM", msg);
                            SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                                           "%s imLOC %u", sname, *imLOC);
                            functionparameter_outlog("LOADMEMSTREAM", msg);
                        }
                    }
                }
                fclose(fp);
            }

            if(*imLOC == STREAM_LOAD_SOURCE_NOTFOUND)
            {
                if(fscanfcnt > 0)
                {
                    {
                        char msg[STRINGMAXLEN_FPS_LOGMSG];
                        SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                                       "%s LOADING %s", sname, streamfname);
                        functionparameter_outlog("LOADMEMSTREAM", msg);
                        load_fits(streamfname, sname, 0, &ID);
                        if(ID != -1)
                        {
                            *imLOC = STREAM_LOAD_SOURCE_CONFNAME;
                            updateCONFFITS = 1;
                            if(MEMLOADREPORT == 1)
                            {
                                SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG, "%s SUCCESS CONFNAME", sname);
                                functionparameter_outlog("LOADMEMSTREAM", msg)
                                ;
                                SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                                               "%s imLOC %u", sname, *imLOC);
                                functionparameter_outlog("LOADMEMSTREAM", msg);
                            }
                        }
                    }
                }
            }
            //printf("imLOC = %d\n", *imLOC);
        }
    }

    printf("%s %d imLOC = %d\n", __FILE__, __LINE__, *imLOC);//TBE

    // copy to shared memory
    if(*imLOC == STREAM_LOAD_SOURCE_LOCALMEM)
        if(FPFLAG_STREAM_LOAD_UPDATE_SHAREMEM & *streamflag)
        {
            if(MEMLOADREPORT == 1)
            {
                char msg[STRINGMAXLEN_FPS_LOGMSG];
                SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                               "%s copy to SHAREMEM", sname);
                functionparameter_outlog("LOADMEMSTREAM", msg);
            }
            copy_image_ID(sname, sname, 1);
        }



    // copy to conf FITS
    if((*imLOC != STREAM_LOAD_SOURCE_NOTFOUND)
            && (*imLOC != STREAM_LOAD_SOURCE_CONFFITS))
        if(FPFLAG_STREAM_LOAD_UPDATE_CONFFITS & *streamflag)
        {
            updateCONFFITS = 1;
        }
    if(updateCONFFITS == 1)
    {
        char fname[STRINGMAXLEN_FULLFILENAME];
        if(MEMLOADREPORT == 1)
        {
            char msg[STRINGMAXLEN_FPS_LOGMSG];
            SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG,
                           "%s copy to CONFFITS", sname);
            functionparameter_outlog("LOADMEMSTREAM", msg);
        }
        WRITE_FULLFILENAME(fname, "./conf/shmim.%s.fits", sname);
        save_fits(sname, fname);
    }



    if((int) *imLOC == STREAM_LOAD_SOURCE_EXITFAILURE)
    {
        *imLOC = STREAM_LOAD_SOURCE_NOTFOUND;
    }

    printf("%s %d imLOC = %d\n", __FILE__, __LINE__, *imLOC);//TBE

    if(MEMLOADREPORT == 1)
    {
        char msg[STRINGMAXLEN_FPS_LOGMSG];
        char locstring[100];

        switch(*imLOC)
        {

        case STREAM_LOAD_SOURCE_NOTFOUND :
            strcpy(locstring, STREAM_LOAD_SOURCE_NOTFOUND_STRING);
            break;

        case STREAM_LOAD_SOURCE_LOCALMEM :
            strcpy(locstring, STREAM_LOAD_SOURCE_LOCALMEM_STRING);
            break;

        case STREAM_LOAD_SOURCE_SHAREMEM :
            strcpy(locstring, STREAM_LOAD_SOURCE_SHAREMEM_STRING);
            break;

        case STREAM_LOAD_SOURCE_CONFFITS :
            strcpy(locstring, STREAM_LOAD_SOURCE_CONFFITS_STRING);
            break;

        case STREAM_LOAD_SOURCE_CONFNAME :
            strcpy(locstring, STREAM_LOAD_SOURCE_CONFNAME_STRING);
            break;

        case STREAM_LOAD_SOURCE_NULL :
            strcpy(locstring, STREAM_LOAD_SOURCE_NULL_STRING);
            break;

        case STREAM_LOAD_SOURCE_EXITFAILURE :
            strcpy(locstring, STREAM_LOAD_SOURCE_EXITFAILURE_STRING);
            break;

        default :
            strcpy(locstring, "unknown");
            break;
        }


        SNPRINTF_CHECK(msg, STRINGMAXLEN_FPS_LOGMSG, "%s FINAL imLOC %u %s", sname,
                       *imLOC, locstring);

        functionparameter_outlog("LOADMEMSTREAM", msg);
    }
    printf("%s %d imLOC = %d\n", __FILE__, __LINE__, *imLOC);//TBE

    return ID;
}



