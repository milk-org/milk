/**
 * @file    fps_PrintParameterInfo.c
 * @brief   print FPS parameter status/values
 */

#include <limits.h> // CHAR_BIT

#include <stdio.h>

#include "milkDebugTools.h"
#include "fps.h"
#define AECBOLDHIGREEN ""
#define AECNORMAL      ""
#define TUI_printfw(...) printf(__VA_ARGS__)


#include "fps_PrintParameterInfo.h"


errno_t
functionparameter_PrintParameterInfo(
    FUNCTION_PARAMETER_STRUCT *fpsentry,
    int                        pindex
)
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

    const char *display_keyword = fpsentry->parray[pindex].keywordfull;
    int prefix_len = strlen(fpsentry->md->name);
    if (strncmp(display_keyword, fpsentry->md->name, prefix_len) == 0 && display_keyword[prefix_len] == '.') {
        display_keyword += prefix_len + 1;
    }

    printf("[%d] Parameter name : %s\n",
           pindex,
           display_keyword);

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
            printf("%s", AECBOLDHIGREEN);
            printf("%d", digit);
            printf("%s", AECNORMAL);
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
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "ACTIVE");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "ACTIVE");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_USED)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "USED");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "USED");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_VISIBLE)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "VISIBLE");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "VISIBLE");
    }

    printf("%*s", flagstringlen, "---");

    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_WRITE)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "WRITE");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "WRITE");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_WRITECONF)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "WRITECONF");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "WRITECONF");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_WRITERUN)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "WRITERUN");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "WRITERUN");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_WRITESTATUS)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "WRITESTATUS");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "WRITESTATUS");
    }

    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_LOG)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "LOG");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "LOG");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_SAVEONCHANGE)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "SAVEONCHANGE");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "SAVEONCHANGE");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_SAVEONCLOSE)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "SAVEONCLOSE");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "SAVEONCLOSE");
    }

    printf("%*s", flagstringlen, "---");

    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_IMPORTED)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "IMPORTED");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "IMPORTED");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_FEEDBACK)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "FEEDBACK");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "FEEDBACK");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_ONOFF)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "ONOFF");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "ONOFF");
    }

    printf("%*s", flagstringlen, "---");

    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_CHECKINIT)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "CHECKINIT");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "CHECKINIT");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_MINLIMIT)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "MINLIMIT");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "MINLIMIT");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_MAXLIMIT)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "MAXLIMIT");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "MAXLIMIT");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_ERROR)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "ERROR");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "ERROR");
    }

    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_LOAD_FORCE_LOCALMEM)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_LOAD_FORCE_LOCALMEM");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_LOAD_FORCE_LOCALMEM");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_LOAD_FORCE_SHAREMEM)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_LOAD_FORCE_SHAREMEM");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_LOAD_FORCE_SHAREMEM");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_LOAD_FORCE_CONFFITS)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_LOAD_FORCE_CONFFITS");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_LOAD_FORCE_CONFFITS");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_LOAD_FORCE_CONFNAME)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_LOAD_FORCE_CONFNAME");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_LOAD_FORCE_CONFNAME");
    }

    printf("\n");

    if(fpsentry->parray[pindex].fpflag &
            FPFLAG_STREAM_LOAD_SKIPSEARCH_LOCALMEM)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_LOAD_SKIPSEARCH_LOCALMEM");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_LOAD_SKIPSEARCH_LOCALMEM");
    }

    if(fpsentry->parray[pindex].fpflag &
            FPFLAG_STREAM_LOAD_SKIPSEARCH_SHAREMEM)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_LOAD_SKIPSEARCH_SHAREMEM");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_LOAD_SKIPSEARCH_SHAREMEM");
    }

    if(fpsentry->parray[pindex].fpflag &
            FPFLAG_STREAM_LOAD_SKIPSEARCH_CONFFITS)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_LOAD_SKIPSEARCH_CONFFITS");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_LOAD_SKIPSEARCH_CONFFITS");
    }

    if(fpsentry->parray[pindex].fpflag &
            FPFLAG_STREAM_LOAD_SKIPSEARCH_CONFNAME)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_LOAD_SKIPSEARCH_CONFNAME");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_LOAD_SKIPSEARCH_CONFNAME");
    }

    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_LOAD_UPDATE_SHAREMEM)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_LOAD_UPDATE_SHAREMEM");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_LOAD_UPDATE_SHAREMEM");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_LOAD_UPDATE_CONFFITS)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_LOAD_UPDATE_CONFFITS");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_LOAD_UPDATE_CONFFITS");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_FILE_CONF_REQUIRED)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "FILE/FPS/STREAM_CONF_REQUIRED");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "FILE/FPS/STREAM_CONF_REQUIRED");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_FILE_RUN_REQUIRED)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "FILE/FPS/STREAM_RUN_REQUIRED");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "FILE/FPS/STREAM_RUN_REQUIRED");
    }

    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_ENFORCE_DATATYPE)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_ENFORCE_DATATYPE");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_ENFORCE_DATATYPE");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_UINT8)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_UINT8");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_UINT8");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_INT8)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_INT8");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_INT8");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_UINT16)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_UINT16");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_UINT16");
    }

    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_INT16)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_INT16");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_INT16");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_UINT32)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_UINT32");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_UINT32");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_INT32)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_INT32");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_INT32");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_UINT64)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_UINT64");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_UINT64");
    }

    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_INT64)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_INT64");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_INT64");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_HALF)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_HALF");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_HALF");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_FLOAT)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_FLOAT");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_FLOAT");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_TEST_DATATYPE_DOUBLE)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_DOUBLE");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_TEST_DATATYPE_DOUBLE");
    }

    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_ENFORCE_1D)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_ENFORCE_1D");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_ENFORCE_1D");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_ENFORCE_2D)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_ENFORCE_2D");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_ENFORCE_2D");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_ENFORCE_3D)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_ENFORCE_3D");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_ENFORCE_3D");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_ENFORCE_XSIZE)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_ENFORCE_XSIZE");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_ENFORCE_XSIZE");
    }

    printf("\n");

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_ENFORCE_YSIZE)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_ENFORCE_YSIZE");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_ENFORCE_YSIZE");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_ENFORCE_ZSIZE)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_ENFORCE_ZSIZE");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "STREAM_ENFORCE_ZSIZE");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_CHECKSTREAM)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "CHECKSTREAM");
        printf("%s", AECNORMAL);
    }
    else
    {
        printf("%*s", flagstringlen, "CHECKSTREAM");
    }

    if(fpsentry->parray[pindex].fpflag & FPFLAG_STREAM_MEMLOADREPORT)
    {
        printf("%s", AECBOLDHIGREEN);
        printf("%*s", flagstringlen, "STREAM_MEMLOADREPORT");
        printf("%s", AECNORMAL);
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

    if(fpsentry->parray[pindex].type == FPTYPE_INT32)
    {
        printf("  %10d", fpsentry->parray[pindex].val.i32[0]);
    }

    if(fpsentry->parray[pindex].type == FPTYPE_UINT32)
    {
        printf("  %10u", fpsentry->parray[pindex].val.ui32[0]);
    }

    if(fpsentry->parray[pindex].type == FPTYPE_INT64)
    {
        printf("  %10ld", (long) fpsentry->parray[pindex].val.i64[0]);
    }

    if(fpsentry->parray[pindex].type == FPTYPE_UINT64)
    {
        printf("  %10lu", (unsigned long) fpsentry->parray[pindex].val.ui64[0]);
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
        printf("  %10ld", (long) fpsentry->parray[pindex].val.pid[0]);
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

    if(fpsentry->parray[pindex].type == FPTYPE_PROCESS)
    {
        printf("  %10s", fpsentry->parray[pindex].val.string[0]);
    }

    if(fpsentry->parray[pindex].type == FPTYPE_STRING_NOT_STREAM)
    {
        printf("  %10s", fpsentry->parray[pindex].val.string[0]);
    }

    printf("\n");
    printf("\n");

    return RETURN_SUCCESS;
}
