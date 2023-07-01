/**
 * @file    fps_userinputsetparamvalue.c
 * @brief   read user input to set parameter value
 */

#include <limits.h>
#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "fps_PrintParameterInfo.h"
#include "fps_WriteParameterToDisk.h"

#include "TUItools.h"




/** @brief Enter new value for parameter
 *
 *
 */
int functionparameter_UserInputSetParamValue(
    FUNCTION_PARAMETER_STRUCT *fpsentry,
    int pindex
)
{
    int  inputOK;
    int  strlenmax = 64;
    char buff[100];
    char c = -1;

    functionparameter_PrintParameterInfo(fpsentry, pindex);

    if(fpsentry->parray[pindex].fpflag & FPFLAG_WRITESTATUS)
    {
        inputOK = 0;
        fflush(stdout);


        int esc_toggle = 0;

        while(inputOK == 0)
        {
            printf("\n Update value (ESC + ENTER to abort) : ");
            fflush(stdout);

            int stringindex = 0;

            c = get_singlechar_block();

            // 10 : line feed
            // 27 : escape
            // 13 : carriage return

            while((c != 10) && (c != 13) && (stringindex < strlenmax - 1))
            {

                if(c == 27)
                {
                    esc_toggle = 1;
                }

                buff[stringindex] = c;
                if(c == 127)  // delete key
                {
                    putchar(0x8);
                    putchar(' ');
                    putchar(0x8);
                    stringindex--;
                }
                else
                {
                    putchar(c); // echo on screen for non-ncurses mode
                    fflush(stdout);
                    stringindex++;
                }
                if(stringindex < 0)
                {
                    stringindex = 0;
                }

                c = get_singlechar_block();
            }
            buff[stringindex] = '\0';
            inputOK           = 1;
        }



        if(esc_toggle == 0)  // update value if escape key has not been pressed
        {

            long   lval = 0;
            double fval = 0.0;
            char  *endptr;
            int    vOK = 1;

            switch(fpsentry->parray[pindex].type)
            {

            case FPTYPE_INT32:
                errno = 0; /* To distinguish success/failure after call */
                lval  = strtol(buff, &endptr, 10);

                /* Check for various possible errors */
                if((errno == ERANGE &&
                        (lval == LONG_MAX || lval == LONG_MIN)) ||
                        (errno != 0 && lval == 0))
                {
                    perror("strtol");
                    vOK = 0;
                    sleep(1);
                }

                if(endptr == buff)
                {
                    fprintf(stderr, "\nERROR: No digits were found\n");
                    vOK = 0;
                    sleep(1);
                }

                if(vOK == 1)
                {
                    fpsentry->parray[pindex].val.i32[0] = lval;
                    functionparameter_outlog("SETVAL", "%s  INT32  %ld", fpsentry->parray[pindex].keywordfull, lval);
                }
                break;

            case FPTYPE_UINT32:
                errno = 0; /* To distinguish success/failure after call */
                lval  = strtol(buff, &endptr, 10);

                /* Check for various possible errors */
                if((errno == ERANGE &&
                        (lval == LONG_MAX || lval == LONG_MIN)) ||
                        (errno != 0 && lval == 0))
                {
                    perror("strtol");
                    vOK = 0;
                    sleep(1);
                }

                if(endptr == buff)
                {
                    fprintf(stderr, "\nERROR: No digits were found\n");
                    vOK = 0;
                    sleep(1);
                }

                if(lval < 0)
                {
                    fprintf(stderr, "\nERROR: must be positive\n");
                    vOK = 0;
                }

                if(vOK == 1)
                {
                    fpsentry->parray[pindex].val.ui32[0] = lval;
                    functionparameter_outlog("SETVAL", "%s  UINT32  %ld", fpsentry->parray[pindex].keywordfull, lval);
                }
                break;

            case FPTYPE_INT64:
                errno = 0; /* To distinguish success/failure after call */
                lval  = strtol(buff, &endptr, 10);

                /* Check for various possible errors */
                if((errno == ERANGE &&
                        (lval == LONG_MAX || lval == LONG_MIN)) ||
                        (errno != 0 && lval == 0))
                {
                    perror("strtol");
                    vOK = 0;
                    sleep(1);
                }

                if(endptr == buff)
                {
                    fprintf(stderr, "\nERROR: No digits were found\n");
                    vOK = 0;
                    sleep(1);
                }

                if(vOK == 1)
                {
                    fpsentry->parray[pindex].val.i64[0] = lval;
                    functionparameter_outlog("SETVAL", "%s  INT64  %ld", fpsentry->parray[pindex].keywordfull, lval);
                }
                break;

            case FPTYPE_UINT64:
                errno = 0; /* To distinguish success/failure after call */
                lval  = strtol(buff, &endptr, 10);

                /* Check for various possible errors */
                if((errno == ERANGE &&
                        (lval == LONG_MAX || lval == LONG_MIN)) ||
                        (errno != 0 && lval == 0))
                {
                    perror("strtol");
                    vOK = 0;
                    sleep(1);
                }

                if(endptr == buff)
                {
                    fprintf(stderr, "\nERROR: No digits were found\n");
                    vOK = 0;
                    sleep(1);
                }

                if(lval < 0)
                {
                    fprintf(stderr, "\nERROR: must be positive\n");
                    vOK = 0;
                }

                if(vOK == 1)
                {
                    fpsentry->parray[pindex].val.ui64[0] = lval;
                    functionparameter_outlog("SETVAL", "%s  UINT64  %ld", fpsentry->parray[pindex].keywordfull, lval);
                }
                break;

            case FPTYPE_FLOAT64:
                errno = 0; /* To distinguish success/failure after call */
                fval  = strtod(buff, &endptr);

                /* Check for various possible errors */
                if((errno == ERANGE) || (errno != 0 && fval == 0.0))
                {
                    perror("strtod");
                    vOK = 0;
                    sleep(1);
                }

                if(endptr == buff)
                {
                    fprintf(stderr, "\nERROR: No digits were found\n");
                    vOK = 0;
                    sleep(1);
                }

                if(vOK == 1)
                {
                    fpsentry->parray[pindex].val.f64[0] = fval;
                    functionparameter_outlog("SETVAL", "%s  FLOAT64  %g", fpsentry->parray[pindex].keywordfull, fval);
                }
                break;

            case FPTYPE_FLOAT32:
                errno = 0; /* To distinguish success/failure after call */
                fval  = strtod(buff, &endptr);

                /* Check for various possible errors */
                if((errno == ERANGE) || (errno != 0 && fval == 0.0))
                {
                    perror("strtod");
                    vOK = 0;
                    sleep(1);
                }

                if(endptr == buff)
                {
                    fprintf(stderr, "\nERROR: No digits were found\n");
                    vOK = 0;
                    sleep(1);
                }

                if(vOK == 1)
                {
                    fpsentry->parray[pindex].val.f32[0] = fval;
                    functionparameter_outlog("SETVAL", "%s  FLOAT32  %g", fpsentry->parray[pindex].keywordfull, fval);
                }
                break;

            case FPTYPE_TIMESPEC:
                errno = 0; /* To distinguish success/failure after call */
                fval  = strtod(buff, &endptr);

                /* Check for various possible errors */
                if((errno == ERANGE) || (errno != 0 && fval == 0.0))
                {
                    perror("strtod");
                    vOK = 0;
                    sleep(1);
                }

                if(endptr == buff)
                {
                    fprintf(stderr, "\nERROR: No digits were found\n");
                    vOK = 0;
                    sleep(1);
                }

                if(vOK == 1)
                {
                    double finteger;
                    double fractional = modf(fval, &finteger);

                    fpsentry->parray[pindex].val.ts[0].tv_sec =
                        (long)(finteger + 0.1);
                    fpsentry->parray[pindex].val.ts[0].tv_nsec =
                        (long)(1.0e9 * fractional + 0.1);

                    functionparameter_outlog("SETVAL", "%s  TIMESPEC  %ld.%09ld", fpsentry->parray[pindex].keywordfull,
                                             fpsentry->parray[pindex].val.ts[0].tv_sec,
                                             fpsentry->parray[pindex].val.ts[0].tv_nsec );
                }
                break;

            case FPTYPE_PID:
                errno = 0; /* To distinguish success/failure after call */
                lval  = strtol(buff, &endptr, 10);

                /* Check for various possible errors */
                if((errno == ERANGE &&
                        (lval == LONG_MAX || lval == LONG_MIN)) ||
                        (errno != 0 && lval == 0))
                {
                    perror("strtol");
                    vOK = 0;
                    sleep(1);
                }

                if(endptr == buff)
                {
                    fprintf(stderr, "\nERROR: No digits were found\n");
                    vOK = 0;
                    sleep(1);
                }

                if(vOK == 1)
                {
                    fpsentry->parray[pindex].val.pid[0] = (pid_t) lval;
                    functionparameter_outlog("SETVAL", "%s  PID  %ld", fpsentry->parray[pindex].keywordfull, lval);
                }
                break;

            case FPTYPE_FILENAME:
                if(snprintf(fpsentry->parray[pindex].val.string[0],
                            FUNCTION_PARAMETER_STRMAXLEN,
                            "%s",
                            buff) < 0)
                {
                    PRINT_ERROR("snprintf error");
                }
                else
                {
                    functionparameter_outlog("SETVAL", "%s  FILENAME  %s", fpsentry->parray[pindex].keywordfull, buff);
                }
                break;

            case FPTYPE_FITSFILENAME:
                if(snprintf(fpsentry->parray[pindex].val.string[0],
                            FUNCTION_PARAMETER_STRMAXLEN,
                            "%s",
                            buff) < 0)
                {
                    PRINT_ERROR("snprintf error");
                }
                else
                {
                    functionparameter_outlog("SETVAL", "%s  FITSFILENAME  %s", fpsentry->parray[pindex].keywordfull, buff);
                }
                break;

            case FPTYPE_EXECFILENAME:
                if(snprintf(fpsentry->parray[pindex].val.string[0],
                            FUNCTION_PARAMETER_STRMAXLEN,
                            "%s",
                            buff) < 0)
                {
                    PRINT_ERROR("snprintf error");
                }
                else
                {
                    functionparameter_outlog("SETVAL", "%s  EXECFILENAME  %s", fpsentry->parray[pindex].keywordfull, buff);
                }
                break;

            case FPTYPE_DIRNAME:
                if(snprintf(fpsentry->parray[pindex].val.string[0],
                            FUNCTION_PARAMETER_STRMAXLEN,
                            "%s",
                            buff) < 0)
                {
                    PRINT_ERROR("snprintf error");
                }
                else
                {
                    functionparameter_outlog("SETVAL", "%s  DIRNAME  %s", fpsentry->parray[pindex].keywordfull, buff);
                }
                break;

            case FPTYPE_STREAMNAME:
                if(snprintf(fpsentry->parray[pindex].val.string[0],
                            FUNCTION_PARAMETER_STRMAXLEN,
                            "%s",
                            buff) < 0)
                {
                    PRINT_ERROR("snprintf error");
                }
                else
                {
                    functionparameter_outlog("SETVAL", "%s  STREAMNAME  %s", fpsentry->parray[pindex].keywordfull, buff);
                }
                break;

            case FPTYPE_STRING:
                if(snprintf(fpsentry->parray[pindex].val.string[0],
                            FUNCTION_PARAMETER_STRMAXLEN,
                            "%s",
                            buff) < 0)
                {
                    PRINT_ERROR("snprintf error");
                }
                else
                {
                    functionparameter_outlog("SETVAL", "%s  STRING  %s", fpsentry->parray[pindex].keywordfull, buff);
                }
                break;

            case FPTYPE_FPSNAME:
                if(snprintf(fpsentry->parray[pindex].val.string[0],
                            FUNCTION_PARAMETER_STRMAXLEN,
                            "%s",
                            buff) < 0)
                {
                    PRINT_ERROR("snprintf error");
                }
                else
                {
                    functionparameter_outlog("SETVAL", "%s  FPSNAME  %s", fpsentry->parray[pindex].keywordfull, buff);
                }
                break;
            }

            fpsentry->parray[pindex].cnt0++;

            // notify GUI
            fpsentry->md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;

            // Save to disk
            if(fpsentry->parray[pindex].fpflag & FPFLAG_SAVEONCHANGE)
            {
                functionparameter_WriteParameterToDisk(
                    fpsentry,
                    pindex,
                    "setval",
                    "UserInputSetParamValue");

                functionparameter_SaveFPS2disk(fpsentry);
            }
        }
    }
    else
    {
        printf("%s Value cannot be modified %s\n", AECBOLDHIRED, AECNORMAL);
        c = getchar();
    }

    return 0;
}
