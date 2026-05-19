/**
 * @file    fps_userinputsetparamvalue.c
 * @brief   read user input to set parameter value
 */


#include "fps.h"


#define AECBOLDHIRED ""
#define AECNORMAL    ""


/** @brief Enter new value for parameter
 *
 *
 */
int functionparameter_UserInputSetParamValue(
    FPS *fpsentry,
    int pindex)
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

            c = getchar();

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

                c = getchar();
            }
            buff[stringindex] = '\0';
            inputOK           = 1;
        }


        if(esc_toggle == 0)  // update value if escape key has not been pressed
        {

            if(functionparameter_SetParamValue_fromString(fpsentry, pindex, buff) != 0)
            {
                printf("\n%s Error: could not convert argument %s\n", AECBOLDHIRED, AECNORMAL);
                sleep(1);
            }
            else
            {
                if(strncmp(fpsentry->parray[pindex].keywordfull, ".procinfo.", 10) == 0)
                {
                    fpsentry->md->processinfo_change_cnt++;
                }

                // notify GUI
                fpsentry->md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;

                // Save to disk
                if(fpsentry->parray[pindex].fpflag & FPFLAG_SAVEONCHANGE)
                {
                    functionparameter_WriteParameterToDisk(
                        fpsentry, pindex, "setval", "UserInputSetParamValue");

                    functionparameter_SaveFPS2disk(fpsentry);
                }
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
