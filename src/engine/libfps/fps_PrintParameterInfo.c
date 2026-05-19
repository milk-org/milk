/**
 * @file    fps_PrintParameterInfo.c
 * @brief   print FPS parameter status/values
 */

#include <limits.h> // CHAR_BIT


#define AECBOLDHIGREEN ""
#define AECNORMAL      ""
#define TUI_printfw(...) printf(__VA_ARGS__)


#include "fps_PrintParameterInfo.h"


errno_t
functionparameter_PrintParameterInfo(
    FPS *fpsentry,
    int                        pindex
)
{
    printf("%s\n", fpsentry->parray[pindex].description);
    printf("\n");

    printf("------------- FUNCTION PARAMETER STRUCTURE\n");
    printf("FPS name       : %s\n", fpsentry->md->name);
    printf("   %s ", fpsentry->md->pname);

    for(int i = 0; i < fpsentry->md->NBnameindex; i++)
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


        printf("%s [%ld] %d ERROR(s)\n",
               fpsentry->md->name,
               fpsentry->md->msgcnt,
               fpsentry->md->conferrcnt);
        for(int msgi = 0; msgi < fpsentry->md->msgcnt; msgi++)
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
    if(strncmp(display_keyword, fpsentry->md->name, prefix_len) == 0
            && display_keyword[prefix_len] == '.')
    {
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

#define PRINT_FPFLAG(FLAG_MACRO, STR_NAME) \
    if(fpsentry->parray[pindex].fpflag & (FLAG_MACRO)) \
    { \
        printf("%s%*s%s", AECBOLDHIGREEN, flagstringlen, STR_NAME, AECNORMAL); \
    } \
    else \
    { \
        printf("%*s", flagstringlen, STR_NAME); \
    }


    PRINT_FPFLAG(FPFLAG_ACTIVE, "ACTIVE");

    PRINT_FPFLAG(FPFLAG_USED, "USED");

    PRINT_FPFLAG(FPFLAG_VISIBLE, "VISIBLE");

    printf("%*s", flagstringlen, "---");

    printf("\n");

    PRINT_FPFLAG(FPFLAG_WRITE, "WRITE");

    PRINT_FPFLAG(FPFLAG_WRITECONF, "WRITECONF");

    PRINT_FPFLAG(FPFLAG_WRITERUN, "WRITERUN");

    PRINT_FPFLAG(FPFLAG_WRITESTATUS, "WRITESTATUS");

    printf("\n");

    PRINT_FPFLAG(FPFLAG_LOG, "LOG");

    PRINT_FPFLAG(FPFLAG_SAVEONCHANGE, "SAVEONCHANGE");

    PRINT_FPFLAG(FPFLAG_SAVEONCLOSE, "SAVEONCLOSE");

    printf("%*s", flagstringlen, "---");

    printf("\n");

    PRINT_FPFLAG(FPFLAG_IMPORTED, "IMPORTED");

    PRINT_FPFLAG(FPFLAG_FEEDBACK, "FEEDBACK");

    PRINT_FPFLAG(FPFLAG_ONOFF, "ONOFF");

    printf("%*s", flagstringlen, "---");

    printf("\n");

    PRINT_FPFLAG(FPFLAG_CHECKINIT, "CHECKINIT");

    PRINT_FPFLAG(FPFLAG_MINLIMIT, "MINLIMIT");

    PRINT_FPFLAG(FPFLAG_MAXLIMIT, "MAXLIMIT");

    PRINT_FPFLAG(FPFLAG_ERROR, "ERROR");

    printf("\n");

    PRINT_FPFLAG(FPFLAG_STREAM_LOAD_FORCE_LOCALMEM, "STREAM_LOAD_FORCE_LOCALMEM");

    PRINT_FPFLAG(FPFLAG_STREAM_LOAD_FORCE_SHAREMEM, "STREAM_LOAD_FORCE_SHAREMEM");

    PRINT_FPFLAG(FPFLAG_STREAM_LOAD_FORCE_CONFFITS, "STREAM_LOAD_FORCE_CONFFITS");

    PRINT_FPFLAG(FPFLAG_STREAM_LOAD_FORCE_CONFNAME, "STREAM_LOAD_FORCE_CONFNAME");

    printf("\n");

    PRINT_FPFLAG(FPFLAG_STREAM_LOAD_SKIPSEARCH_LOCALMEM, "STREAM_LOAD_SKIPSEARCH_LOCALMEM");

    PRINT_FPFLAG(FPFLAG_STREAM_LOAD_SKIPSEARCH_SHAREMEM, "STREAM_LOAD_SKIPSEARCH_SHAREMEM");

    PRINT_FPFLAG(FPFLAG_STREAM_LOAD_SKIPSEARCH_CONFFITS, "STREAM_LOAD_SKIPSEARCH_CONFFITS");

    PRINT_FPFLAG(FPFLAG_STREAM_LOAD_SKIPSEARCH_CONFNAME, "STREAM_LOAD_SKIPSEARCH_CONFNAME");

    printf("\n");

    PRINT_FPFLAG(FPFLAG_STREAM_LOAD_UPDATE_SHAREMEM, "STREAM_LOAD_UPDATE_SHAREMEM");

    PRINT_FPFLAG(FPFLAG_STREAM_LOAD_UPDATE_CONFFITS, "STREAM_LOAD_UPDATE_CONFFITS");

    PRINT_FPFLAG(FPFLAG_FILE_CONF_REQUIRED, "FILE/FPS/STREAM_CONF_REQUIRED");

    PRINT_FPFLAG(FPFLAG_FILE_RUN_REQUIRED, "FILE/FPS/STREAM_RUN_REQUIRED");

    printf("\n");

    PRINT_FPFLAG(FPFLAG_STREAM_ENFORCE_DATATYPE, "STREAM_ENFORCE_DATATYPE");

    PRINT_FPFLAG(FPFLAG_STREAM_TEST_DATATYPE_UINT8, "STREAM_TEST_DATATYPE_UINT8");

    PRINT_FPFLAG(FPFLAG_STREAM_TEST_DATATYPE_INT8, "STREAM_TEST_DATATYPE_INT8");

    PRINT_FPFLAG(FPFLAG_STREAM_TEST_DATATYPE_UINT16, "STREAM_TEST_DATATYPE_UINT16");

    printf("\n");

    PRINT_FPFLAG(FPFLAG_STREAM_TEST_DATATYPE_INT16, "STREAM_TEST_DATATYPE_INT16");

    PRINT_FPFLAG(FPFLAG_STREAM_TEST_DATATYPE_UINT32, "STREAM_TEST_DATATYPE_UINT32");

    PRINT_FPFLAG(FPFLAG_STREAM_TEST_DATATYPE_INT32, "STREAM_TEST_DATATYPE_INT32");

    PRINT_FPFLAG(FPFLAG_STREAM_TEST_DATATYPE_UINT64, "STREAM_TEST_DATATYPE_UINT64");

    printf("\n");

    PRINT_FPFLAG(FPFLAG_STREAM_TEST_DATATYPE_INT64, "STREAM_TEST_DATATYPE_INT64");

    PRINT_FPFLAG(FPFLAG_STREAM_TEST_DATATYPE_HALF, "STREAM_TEST_DATATYPE_HALF");

    PRINT_FPFLAG(FPFLAG_STREAM_TEST_DATATYPE_FLOAT, "STREAM_TEST_DATATYPE_FLOAT");

    PRINT_FPFLAG(FPFLAG_STREAM_TEST_DATATYPE_DOUBLE, "STREAM_TEST_DATATYPE_DOUBLE");

    printf("\n");

    PRINT_FPFLAG(FPFLAG_STREAM_ENFORCE_1D, "STREAM_ENFORCE_1D");

    PRINT_FPFLAG(FPFLAG_STREAM_ENFORCE_2D, "STREAM_ENFORCE_2D");

    PRINT_FPFLAG(FPFLAG_STREAM_ENFORCE_3D, "STREAM_ENFORCE_3D");

    PRINT_FPFLAG(FPFLAG_STREAM_ENFORCE_XSIZE, "STREAM_ENFORCE_XSIZE");

    printf("\n");

    PRINT_FPFLAG(FPFLAG_STREAM_ENFORCE_YSIZE, "STREAM_ENFORCE_YSIZE");

    PRINT_FPFLAG(FPFLAG_STREAM_ENFORCE_ZSIZE, "STREAM_ENFORCE_ZSIZE");

    PRINT_FPFLAG(FPFLAG_CHECKSTREAM, "CHECKSTREAM");

    PRINT_FPFLAG(FPFLAG_STREAM_MEMLOADREPORT, "STREAM_MEMLOADREPORT");

    printf("\n");
    printf("\n");
#undef PRINT_FPFLAG

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
