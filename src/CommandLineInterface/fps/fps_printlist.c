/**
 * @file    fps_printlist.c
 * @brief   print list of parameters
 */

#include "CommandLineInterface/CLIcore.h"

int function_parameter_printlist(FUNCTION_PARAMETER *funcparamarray,
                                 long                NBparamMAX)
{
    long pindex = 0;
    long pcnt   = 0;

    printf("\n");
    for(pindex = 0; pindex < NBparamMAX; pindex++)
    {
        if(funcparamarray[pindex].fpflag & FPFLAG_ACTIVE)
        {
            printf("Parameter %4ld : %s\n",
                   pindex,
                   funcparamarray[pindex].keywordfull);
            /*for(int kl=0; kl< funcparamarray[pindex].keywordlevel; kl++)
            	printf("  %s", funcparamarray[pindex].keyword[kl]);
            printf("\n");*/
            printf("    %s\n", funcparamarray[pindex].description);

            // STATUS FLAGS
            printf("    STATUS FLAGS (0x%02hhx) :",
                   (int) funcparamarray[pindex].fpflag);
            if(funcparamarray[pindex].fpflag & FPFLAG_ACTIVE)
            {
                printf(" ACTIVE");
            }
            if(funcparamarray[pindex].fpflag & FPFLAG_USED)
            {
                printf(" USED");
            }
            if(funcparamarray[pindex].fpflag & FPFLAG_VISIBLE)
            {
                printf(" VISIBLE");
            }
            if(funcparamarray[pindex].fpflag & FPFLAG_WRITE)
            {
                printf(" WRITE");
            }
            if(funcparamarray[pindex].fpflag & FPFLAG_WRITECONF)
            {
                printf(" WRITECONF");
            }
            if(funcparamarray[pindex].fpflag & FPFLAG_WRITERUN)
            {
                printf(" WRITERUN");
            }
            if(funcparamarray[pindex].fpflag & FPFLAG_LOG)
            {
                printf(" LOG");
            }
            if(funcparamarray[pindex].fpflag & FPFLAG_SAVEONCHANGE)
            {
                printf(" SAVEONCHANGE");
            }
            if(funcparamarray[pindex].fpflag & FPFLAG_SAVEONCLOSE)
            {
                printf(" SAVEONCLOSE");
            }
            if(funcparamarray[pindex].fpflag & FPFLAG_MINLIMIT)
            {
                printf(" MINLIMIT");
            }
            if(funcparamarray[pindex].fpflag & FPFLAG_MAXLIMIT)
            {
                printf(" MAXLIMIT");
            }
            if(funcparamarray[pindex].fpflag & FPFLAG_CHECKSTREAM)
            {
                printf(" CHECKSTREAM");
            }
            if(funcparamarray[pindex].fpflag & FPFLAG_IMPORTED)
            {
                printf(" IMPORTED");
            }
            if(funcparamarray[pindex].fpflag & FPFLAG_FEEDBACK)
            {
                printf(" FEEDBACK");
            }
            if(funcparamarray[pindex].fpflag & FPFLAG_ERROR)
            {
                printf(" ERROR");
            }
            if(funcparamarray[pindex].fpflag & FPFLAG_ONOFF)
            {
                printf(" ONOFF");
            }
            printf("\n");

            // DATA TYPE
            //			printf("    TYPE : 0x%02hhx\n", (int) funcparamarray[pindex].type);
            if(funcparamarray[pindex].type & FPTYPE_UNDEF)
            {
                printf("    TYPE = UNDEF\n");
            }
            if(funcparamarray[pindex].type & FPTYPE_INT64)
            {
                printf("    TYPE  = INT64\n");
                printf("    VALUE = %ld\n",
                       (long) funcparamarray[pindex].val.i64[0]);
            }
            if(funcparamarray[pindex].type & FPTYPE_FLOAT64)
            {
                printf("    TYPE = FLOAT64\n");
            }
            if(funcparamarray[pindex].type & FPTYPE_PID)
            {
                printf("    TYPE = PID\n");
            }
            if(funcparamarray[pindex].type & FPTYPE_TIMESPEC)
            {
                printf("    TYPE = TIMESPEC\n");
            }
            if(funcparamarray[pindex].type & FPTYPE_FILENAME)
            {
                printf("    TYPE = FILENAME\n");
            }
            if(funcparamarray[pindex].type & FPTYPE_DIRNAME)
            {
                printf("    TYPE = DIRNAME\n");
            }
            if(funcparamarray[pindex].type & FPTYPE_STREAMNAME)
            {
                printf("    TYPE = STREAMNAME\n");
            }
            if(funcparamarray[pindex].type & FPTYPE_STRING)
            {
                printf("    TYPE = STRING\n");
            }
            if(funcparamarray[pindex].type & FPTYPE_ONOFF)
            {
                printf("    TYPE = ONOFF\n");
            }
            if(funcparamarray[pindex].type & FPTYPE_FPSNAME)
            {
                printf("    TYPE = FPSNAME\n");
            }

            pcnt++;
        }
    }
    printf("\n");
    printf("%ld/%ld active parameters\n", pcnt, NBparamMAX);

    printf("\n");

    return 0;
}
