/**
 * @file    fps_PrintParameterInfo.c
 * @brief   print FPS parameter status/values
 */

#include <limits.h> // CHAR_BIT

#include "CommandLineInterface/CLIcore.h"

#include "fps_GetTypeString.h"

#include "TUItools.h"

errno_t
functionparameter_PrintParameterInfo(FUNCTION_PARAMETER_STRUCT *fpsentry,
                                     int                        pindex)
{
    printf("%s\n", fpsentry->parray[pindex].description);
    printf("\n");

    printf("------------- FUNCTION PARAMETER STRUCTURE\n");
    printf("FPS name       : %s\n", fpsentry->md->name);
    printf("   %s ", fpsentry->md->pname);
    int i;
    for(i = 0; i < fpsentry->md->NBnameindex; i++)
    {
        printf(" [%s]", fpsentry->md->nameindexW[i]);
    }
    printf("\n\n");

    if(fpsentry->md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CHECKOK)
    {
        printf("[%ld] Scan OK\n", fpsentry->md->msgcnt);
    }
    else
    {
        int msgi;

        printf("%s [%ld] %d ERROR(s)\n",
               fpsentry->md->name,
               fpsentry->md->msgcnt,
               fpsentry->md->conferrcnt);
        for(msgi = 0; msgi < fpsentry->md->msgcnt; msgi++)
        {
            printf("%s [%3d] %s\n",
                   fpsentry->md->name,
                   fpsentry->md->msgpindex[msgi],
                   fpsentry->md->message[msgi]);
        }
    }

    //snprintf(fpsentry->md->message[fpsentry->md->msgcnt], FUNCTION_PARAMETER_STRUCT_MSG_SIZE, "cannot load stream");
    //			fpsentry->md->msgcnt++;

    printf("\n");

    printf("------------- FUNCTION PARAMETER \n");
    printf("[%d] Parameter name : %s\n",
           pindex,
           fpsentry->parray[pindex].keywordfull);

    char typestring[STRINGMAXLEN_FPSTYPE];
    functionparameter_GetTypeString(fpsentry->parray[pindex].type, typestring);
    printf("type: %s\n", typestring);

    printf("\n");
    printf("-- FLAG: ");

    // print binary flag
    TUI_printfw("FLAG : ");
    uint64_t mask = (uint64_t) 1 << (sizeof(uint64_t) * CHAR_BIT - 1);
    while(mask)
    {
        int digit = fpsentry->parray[pindex].fpflag & mask ? 1 : 0;
        if(digit == 1)
        {
            printf(AECBOLDHIGREEN);
            printf("%d", digit);
            printf(AECNORMAL);
        }
        else
        {
            printf("%d", digit);
        }
        mask >>= 1;
    }
    printf("\n");

    int flagstringlen = 32;

    if(fpsentry->parray[pindex].fpflag & FPFLAG_ACTIVE)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "ACTIVE");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "ACTIVE");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_USED)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "USED");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "USED");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_VISIBLE)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "VISIBLE");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "VISIBLE");
    }

    printf("%*s", flagstringlen, "---");

    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_WRITE)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "WRITE");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "WRITE");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_WRITECONF)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "WRITECONF");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "WRITECONF");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_WRITERUN)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "WRITERUN");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "WRITERUN");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_WRITESTATUS)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "WRITESTATUS");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "WRITESTATUS");
    }

    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_LOG)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "LOG");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "LOG");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_SAVEONCHANGE)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "SAVEONCHANGE");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "SAVEONCHANGE");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_SAVEONCLOSE)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "SAVEONCLOSE");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "SAVEONCLOSE");
    }

    printf("%*s", flagstringlen, "---");

    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_IMPORTED)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "IMPORTED");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "IMPORTED");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_FEEDBACK)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "FEEDBACK");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "FEEDBACK");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_ONOFF)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "ONOFF");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "ONOFF");
    }

    printf("%*s", flagstringlen, "---");

    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_CHECKINIT)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "CHECKINIT");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "CHECKINIT");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_MINLIMIT)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "MINLIMIT");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "MINLIMIT");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_MAXLIMIT)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "MAXLIMIT");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "MAXLIMIT");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_ERROR)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "ERROR");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "ERROR");
    }

    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_LOAD_FORCE_LOCALMEM)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_LOAD_FORCE_LOCALMEM");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_LOAD_FORCE_LOCALMEM");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_LOAD_FORCE_SHAREMEM)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_LOAD_FORCE_SHAREMEM");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_LOAD_FORCE_SHAREMEM");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_LOAD_FORCE_CONFFITS)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_LOAD_FORCE_CONFFITS");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_LOAD_FORCE_CONFFITS");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_LOAD_FORCE_CONFNAME)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_LOAD_FORCE_CONFNAME");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_LOAD_FORCE_CONFNAME");
    }

    printf("\n");

    if(fpsentry->parray[pindex].fpflag &
            FPFLAG_STREAM_LOAD_SKIPSEARCH_LOCALMEM)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_LOAD_SKIPSEARCH_LOCALMEM");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_LOAD_SKIPSEARCH_LOCALMEM");
    }

    if(fpsentry->parray[pindex].fpflag &
            FPFLAG_STREAM_LOAD_SKIPSEARCH_SHAREMEM)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_LOAD_SKIPSEARCH_SHAREMEM");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_LOAD_SKIPSEARCH_SHAREMEM");
    }

    if(fpsentry->parray[pindex].fpflag &
            FPFLAG_STREAM_LOAD_SKIPSEARCH_CONFFITS)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_LOAD_SKIPSEARCH_CONFFITS");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_LOAD_SKIPSEARCH_CONFFITS");
    }

    if(fpsentry->parray[pindex].fpflag &
            FPFLAG_STREAM_LOAD_SKIPSEARCH_CONFNAME)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_LOAD_SKIPSEARCH_CONFNAME");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_LOAD_SKIPSEARCH_CONFNAME");
    }

    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_LOAD_UPDATE_SHAREMEM)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_LOAD_UPDATE_SHAREMEM");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_LOAD_UPDATE_SHAREMEM");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_LOAD_UPDATE_CONFFITS)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_LOAD_UPDATE_CONFFITS");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_LOAD_UPDATE_CONFFITS");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_FILE_CONF_REQUIRED)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "FILE/FPS/STREAM_CONF_REQUIRED");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "FILE/FPS/STREAM_CONF_REQUIRED");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_FILE_RUN_REQUIRED)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "FILE/FPS/STREAM_RUN_REQUIRED");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "FILE/FPS/STREAM_RUN_REQUIRED");
    }

    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_ENFORCE_DATATYPE)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_ENFORCE_DATATYPE");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_ENFORCE_DATATYPE");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_UINT8)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_UINT8");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_UINT8");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_INT8)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_INT8");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_INT8");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_UINT16)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_UINT16");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_UINT16");
    }

    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_INT16)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_INT16");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_INT16");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_UINT32)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_UINT32");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_UINT32");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_INT32)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_INT32");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_INT32");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_UINT64)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_UINT64");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_UINT64");
    }

    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_INT64)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_INT64");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_INT64");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_HALF)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_HALF");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_HALF");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_FLOAT)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_FLOAT");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_FLOAT");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_DOUBLE)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_DOUBLE");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_DOUBLE");
    }

    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_ENFORCE_1D)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_ENFORCE_1D");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_ENFORCE_1D");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_ENFORCE_2D)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_ENFORCE_2D");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_ENFORCE_2D");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_ENFORCE_3D)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_ENFORCE_3D");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_ENFORCE_3D");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_ENFORCE_XSIZE)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_ENFORCE_XSIZE");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_ENFORCE_XSIZE");
    }

    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_ENFORCE_YSIZE)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_ENFORCE_YSIZE");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_ENFORCE_YSIZE");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_ENFORCE_ZSIZE)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_ENFORCE_ZSIZE");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_ENFORCE_ZSIZE");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_CHECKSTREAM)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "CHECKSTREAM");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "CHECKSTREAM");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_MEMLOADREPORT)
    {
        printf(AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_MEMLOADREPORT");
        printf(AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_MEMLOADREPORT");
    }

    printf("\n");
    printf("\n");
    printf("cnt0 = %ld\n", fpsentry->parray[pindex].cnt0);

    printf("\n");

    printf("Current value : ");

    if(fpsentry->parray[pindex].type == FPTYPE_UNDEF)
    {
        printf("  %s", "-undef-");
    }

    if(fpsentry->parray[pindex].type == FPTYPE_INT64)
    {
        printf("  %10d", (int) fpsentry->parray[pindex].val.i64[0]);
    }

    if(fpsentry->parray[pindex].type == FPTYPE_FLOAT64)
    {
        printf("  %10f", (float) fpsentry->parray[pindex].val.f64[0]);
    }

    if(fpsentry->parray[pindex].type == FPTYPE_FLOAT32)
    {
        printf("  %10f", (float) fpsentry->parray[pindex].val.f32[0]);
    }

    if(fpsentry->parray[pindex].type == FPTYPE_PID)
    {
        printf("  %10d", (int) fpsentry->parray[pindex].val.pid[0]);
    }

    if(fpsentry->parray[pindex].type == FPTYPE_TIMESPEC)
    {
        printf("  %10f",
               1.0 * fpsentry->parray[pindex].val.ts[0].tv_sec +
               1e-9 * fpsentry->parray[pindex].val.ts[0].tv_nsec);
    }

    if(fpsentry->parray[pindex].type == FPTYPE_FILENAME)
    {
        printf("  %10s", fpsentry->parray[pindex].val.string[0]);
    }

    if(fpsentry->parray[pindex].type == FPTYPE_FITSFILENAME)
    {
        printf("  %10s", fpsentry->parray[pindex].val.string[0]);
    }

    if(fpsentry->parray[pindex].type == FPTYPE_EXECFILENAME)
    {
        printf("  %10s", fpsentry->parray[pindex].val.string[0]);
    }

    if(fpsentry->parray[pindex].type == FPTYPE_DIRNAME)
    {
        printf("  %10s", fpsentry->parray[pindex].val.string[0]);
    }

    if(fpsentry->parray[pindex].type == FPTYPE_STREAMNAME)
    {
        printf("  %10s", fpsentry->parray[pindex].val.string[0]);
    }

    if(fpsentry->parray[pindex].type == FPTYPE_STRING)
    {
        printf("  %10s", fpsentry->parray[pindex].val.string[0]);
    }

    if(fpsentry->parray[pindex].type == FPTYPE_ONOFF)
    {
        if(fpsentry->parray[pindex].fpflag & FPFLAG_ONOFF)
        {
            printf("    ON  [ %s ]\n", fpsentry->parray[pindex].val.string[1]);
        }
        else
        {
            printf("   OFF  [ %s ]\n", fpsentry->parray[pindex].val.string[0]);
        }
    }

    if(fpsentry->parray[pindex].type == FPTYPE_FPSNAME)
    {
        printf("  %10s", fpsentry->parray[pindex].val.string[0]);
    }

    printf("\n");
    printf("\n");

    return RETURN_SUCCESS;
}
