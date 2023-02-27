/**
 * @file    imagemon.c
 * @brief   image monitor
 */

#define NCURSES_WIDECHAR 1

#include <math.h>
#include <ncurses.h>
#include <curses.h>

#include <stdio.h>
#include <wchar.h>
#include <wctype.h>
#include <locale.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "print_header.h"
#include "streamtiming_stats.h"
#include "timediff.h"


#include "TUItools.h"




// screen size
static uint16_t wrow, wcol;


static uint64_t       cntlast;
static struct timespec tlast;



// Local variables pointers
static char  *instreamname;
static float *updatefrequency;

static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".insname",
        "input stream",
        "im1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &instreamname,
        NULL
    },
    {
        CLIARG_FLOAT32,
        ".frequ",
        "frequency [Hz]",
        "3.0",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &updatefrequency,
        NULL
    }
};

static CLICMDDATA CLIcmddata =
{
    "imgmon", "image monitor", CLICMD_FIELDS_DEFAULTS
};




// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}



errno_t info_image_monitor(const char *ID_name, float frequ);
errno_t printstatus(imageID ID);



static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();


    INSERT_TUI_SETUP

    // define screens
    static int NBTUIscreen = 3;

    TUIscreenarray[0].index = 1;
    TUIscreenarray[0].keych = 'h';
    strcpy(TUIscreenarray[0].name, "[h] Help");

    TUIscreenarray[1].index = 2;
    TUIscreenarray[1].keych = KEY_F(2);
    strcpy(TUIscreenarray[1].name, "[F2] summary");

    TUIscreenarray[2].index = 3;
    TUIscreenarray[2].keych = KEY_F(3);
    strcpy(TUIscreenarray[2].name, "[F3] timing");


    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT

    double pinfotdelay = 1.0 * processinfo->triggerdelay.tv_sec +
                         1.0e-9 * processinfo->triggerdelay.tv_nsec;
    int diplaycntinterval = (int)((1.0 / *updatefrequency) / pinfotdelay);
    int dispcnt           = 0;

    imageID ID        = image_ID(instreamname);
    int     TUIscreen = 2;
    int     sem       = -1;

    int timingbuffinit = 0;

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART

    {
        INSTERT_TUI_KEYCONTROLS


        if(TUIinputkch == ' ')
        {
            TUI_printfw("BUFFER INIT\n");
            // flush timing buffer
            timingbuffinit = 1;
        }

        if((dispcnt == 0) && (TUIpause == 0))
        {
            processinfo_WriteMessage(processinfo, "clear screen");
            erase();

            // Check for screen size change
            TUI_get_terminal_size(&wrow, &wcol);

            INSERT_TUI_SCREEN_MENU
            TUI_newline();

            if(TUIscreen == 1)
            {
                TUI_printfw("h / F2 / F3 : change screen\n");
                TUI_printfw("x : exit\n");
            }

            if(TUIscreen == 2)
            {
                processinfo->triggermode = PROCESSINFO_TRIGGERMODE_DELAY;
                processinfo_WriteMessage(processinfo, "printstatus start");
                printstatus(ID);
                processinfo_WriteMessage(processinfo, "printstatus end");
            }

            if(TUIscreen == 3)
            {
                processinfo->triggermode =
                    PROCESSINFO_TRIGGERMODE_IMMEDIATE; // DIIIIIIRTY

                if(sem == -1)
                {
                    int semdefault = 0;
                    sem = ImageStreamIO_getsemwaitindex(&data.image[ID],
                                                        semdefault);
                }
                long  NBtsamples     = 10000;
                float samplestimeout = pinfotdelay;
                TUI_printfw(
                    "Listening on semaphore %d, collecting %ld samples, "
                    "updating "
                    "every %.3f sec\n",
                    sem,
                    NBtsamples,
                    samplestimeout);
                TUI_printfw("Press SPACE to reset buffer\n");

                // Hack to avoid missing a large amount of frames while waiting for the processinfo trigger delay
                info_image_streamtiming_stats(
                    ID,
                    sem,
                    NBtsamples,
                    processinfo->triggerdelay.tv_sec +
                    1.0e-9 * processinfo->triggerdelay.tv_nsec,
                    timingbuffinit);
                timingbuffinit = 0;
            }
            else
            {
                sem = -1;
            }

            refresh();
        }

        if(++dispcnt > diplaycntinterval)
        {
            dispcnt = 0;
        }
    }

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    endwin();

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




INSERT_STD_FPSCLIfunctions




// Register function in CLI
errno_t
CLIADDCMD_info__imagemon()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}




errno_t printstatus(imageID ID)
{
    IMAGE *image = &data.image[ID];

    long          j;
    double        frequ;
    long          NBhistopt = 20;
    long         *vcnt;
    long          h;
    unsigned long cnt;
    long          i;

    int customcolor;

    float  minPV = 60000;
    float  maxPV = 0;
    float charval;
    double average;
    double imtotal;


    char line1[200];

    double tmp;
    double RMS = 0.0;

    static double RMS01 = 0.0;
    long   vcntmax;
    int    semval;

    DEBUG_TRACEPOINT("window size %3d %3d", wcol, wrow);

    uint8_t datatype;

    {
        // Image name, type and size
        char str[STRINGMAXLEN_DEFAULT];
        char str1[STRINGMAXLEN_DEFAULT];

        TUI_printfw("%s  ", image->name);
        datatype = image->md->datatype;
        sprintf(str,
                "%s [ %6ld",
                ImageStreamIO_typename(datatype),
                (long) image->md->size[0]);

        for(j = 1; j < image->md->naxis; j++)
        {
            WRITE_STRING(str1,
                         "%s x %6ld",
                         str,
                         (long) image->md->size[j]);

            strcpy(str, str1);
        }

        WRITE_STRING(str1, "%s]", str);
        strcpy(str, str1);

        TUI_printfw("%-28s\n", str);
    }

    TUI_printfw("[write %d] ", image->md->write);
    TUI_printfw("[status %2d] ", image->md->status);

    {
        // timing and counters

        struct timespec tnow;
        struct timespec tdiff;
        double          tdiffv;

        clock_gettime(CLOCK_REALTIME, &tnow);
        tdiff = info_time_diff(tlast, tnow);
        clock_gettime(CLOCK_REALTIME, &tlast);

        tdiffv  = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;
        frequ   = (image->md->cnt0 - cntlast) / tdiffv;
        cntlast = image->md->cnt0;

        TUI_printfw("[cnt0 %8d] [%6.2f Hz] ", image->md->cnt0, frequ);
        TUI_printfw("[cnt1 %8d]\n", image->md->cnt1);
    }



    if(1)
    {
        // semaphores, read / write
        TUI_printfw("[%3ld sems ", image->md->sem);
        for(int s = 0; s < image->md->sem; s++)
        {
            sem_getvalue(image->semptr[s], &semval);
            TUI_printfw(" %6d ", semval);
        }
        TUI_printfw("]\n");

        TUI_printfw("[ WRITE   ", image->md->sem);
        for(int s = 0; s < image->md->sem; s++)
        {
            TUI_printfw(" %6d ", image->semWritePID[s]);
        }
        TUI_printfw("]\n");

        TUI_printfw("[ READ    ", image->md->sem);
        for(int s = 0; s < image->md->sem; s++)
        {
            TUI_printfw(" %6d ", image->semReadPID[s]);
        }
        TUI_printfw("]\n");

        sem_getvalue(image->semlog, &semval);
        TUI_printfw(" [semlog % 3d] ", semval);

        TUI_printfw(" [circbuff %3d/%3d  %4ld]",
                    image->md->CBindex,
                    image->md->CBsize,
                    image->md->CBcycle);

        TUI_printfw("\n");
    }


    if(1)
    {
        // image stats

        average = arith_image_mean(image->name);

        imtotal = arith_image_total(image->name);

        if(datatype == _DATATYPE_FLOAT)
        {
            TUI_printfw("median %12g   ", arith_image_median(image->name));
        }

        TUI_printfw("average %12g    total = %12g\n",
                    imtotal / image->md->nelement,
                    imtotal);




        vcnt = (long *) malloc(sizeof(long) * NBhistopt);
        if(vcnt == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }
        for(h = 0; h < NBhistopt; h++)
        {
            vcnt[h] = 0;
        }



        if(datatype == _DATATYPE_FLOAT)
        {
            minPV = image->array.F[0];
            maxPV = minPV;

            for(unsigned long ii = 0; ii < image->md->nelement; ii++)
            {
                if(image->array.F[ii] < minPV)
                {
                    minPV = image->array.F[ii];
                }
                if(image->array.F[ii] > maxPV)
                {
                    maxPV = image->array.F[ii];
                }
                tmp = (1.0 * image->array.F[ii] - average);
                RMS += tmp * tmp;
                h = (long)(1.0 * NBhistopt *
                           ((float)(image->array.F[ii] - minPV)) /
                           (maxPV - minPV));
                if((h > -1) && (h < NBhistopt))
                {
                    vcnt[h]++;
                }
            }
        }


        if(datatype == _DATATYPE_DOUBLE)
        {
            minPV = image->array.D[0];
            maxPV = minPV;

            for(unsigned long ii = 0; ii < image->md->nelement; ii++)
            {
                if(image->array.D[ii] < minPV)
                {
                    minPV = image->array.D[ii];
                }
                if(image->array.D[ii] > maxPV)
                {
                    maxPV = image->array.D[ii];
                }
                tmp = (1.0 * image->array.D[ii] - average);
                RMS += tmp * tmp;
                h = (long)(1.0 * NBhistopt *
                           ((float)(image->array.D[ii] - minPV)) /
                           (maxPV - minPV));
                if((h > -1) && (h < NBhistopt))
                {
                    vcnt[h]++;
                }
            }
        }

        if(datatype == _DATATYPE_UINT8)
        {
            minPV = image->array.UI8[0];
            maxPV = minPV;

            for(unsigned long ii = 0; ii < image->md->nelement; ii++)
            {
                if(image->array.UI8[ii] < minPV)
                {
                    minPV = image->array.UI8[ii];
                }
                if(image->array.UI8[ii] > maxPV)
                {
                    maxPV = image->array.UI8[ii];
                }
                tmp = (1.0 * image->array.UI8[ii] - average);
                RMS += tmp * tmp;
                h = (long)(1.0 * NBhistopt *
                           ((float)(image->array.UI8[ii] - minPV)) /
                           (maxPV - minPV));
                if((h > -1) && (h < NBhistopt))
                {
                    vcnt[h]++;
                }
            }
        }

        if(datatype == _DATATYPE_UINT16)
        {
            minPV = image->array.UI16[0];
            maxPV = minPV;

            for(unsigned long ii = 0; ii < image->md->nelement; ii++)
            {
                if(image->array.UI16[ii] < minPV)
                {
                    minPV = image->array.UI16[ii];
                }
                if(image->array.UI16[ii] > maxPV)
                {
                    maxPV = image->array.UI16[ii];
                }
                tmp = (1.0 * image->array.UI16[ii] - average);
                RMS += tmp * tmp;
                h = (long)(1.0 * NBhistopt *
                           ((float)(image->array.UI16[ii] - minPV)) /
                           (maxPV - minPV));
                if((h > -1) && (h < NBhistopt))
                {
                    vcnt[h]++;
                }
            }
        }

        if(datatype == _DATATYPE_UINT32)
        {
            minPV = image->array.UI32[0];
            maxPV = minPV;

            for(unsigned long ii = 0; ii < image->md->nelement; ii++)
            {
                if(image->array.UI32[ii] < minPV)
                {
                    minPV = image->array.UI32[ii];
                }
                if(image->array.UI32[ii] > maxPV)
                {
                    maxPV = image->array.UI32[ii];
                }
                tmp = (1.0 * image->array.UI32[ii] - average);
                RMS += tmp * tmp;
                h = (long)(1.0 * NBhistopt *
                           ((float)(image->array.UI32[ii] - minPV)) /
                           (maxPV - minPV));
                if((h > -1) && (h < NBhistopt))
                {
                    vcnt[h]++;
                }
            }
        }

        if(datatype == _DATATYPE_UINT64)
        {
            minPV = image->array.UI64[0];
            maxPV = minPV;

            for(unsigned long ii = 0; ii < image->md->nelement; ii++)
            {
                if(image->array.UI64[ii] < minPV)
                {
                    minPV = image->array.UI64[ii];
                }
                if(image->array.UI64[ii] > maxPV)
                {
                    maxPV = image->array.UI64[ii];
                }
                tmp = (1.0 * image->array.UI64[ii] - average);
                RMS += tmp * tmp;
                h = (long)(1.0 * NBhistopt *
                           ((float)(image->array.UI64[ii] - minPV)) /
                           (maxPV - minPV));
                if((h > -1) && (h < NBhistopt))
                {
                    vcnt[h]++;
                }
            }
        }

        if(datatype == _DATATYPE_INT8)
        {
            minPV = image->array.SI8[0];
            maxPV = minPV;

            for(unsigned long ii = 0; ii < image->md->nelement; ii++)
            {
                if(image->array.SI8[ii] < minPV)
                {
                    minPV = image->array.SI8[ii];
                }
                if(image->array.SI8[ii] > maxPV)
                {
                    maxPV = image->array.SI8[ii];
                }
                tmp = (1.0 * image->array.SI8[ii] - average);
                RMS += tmp * tmp;
                h = (long)(1.0 * NBhistopt *
                           ((float)(image->array.SI8[ii] - minPV)) /
                           (maxPV - minPV));
                if((h > -1) && (h < NBhistopt))
                {
                    vcnt[h]++;
                }
            }
        }

        if(datatype == _DATATYPE_INT16)
        {
            minPV = image->array.SI16[0];
            maxPV = minPV;

            for(unsigned long ii = 0; ii < image->md->nelement; ii++)
            {
                if(image->array.SI16[ii] < minPV)
                {
                    minPV = image->array.SI16[ii];
                }
                if(image->array.SI16[ii] > maxPV)
                {
                    maxPV = image->array.SI16[ii];
                }
                tmp = (1.0 * image->array.SI16[ii] - average);
                RMS += tmp * tmp;
                h = (long)(1.0 * NBhistopt *
                           ((float)(image->array.SI16[ii] - minPV)) /
                           (maxPV - minPV));
                if((h > -1) && (h < NBhistopt))
                {
                    vcnt[h]++;
                }
            }
        }

        if(datatype == _DATATYPE_INT32)
        {
            minPV = image->array.SI32[0];
            maxPV = minPV;

            for(unsigned long ii = 0; ii < image->md->nelement; ii++)
            {
                if(image->array.SI32[ii] < minPV)
                {
                    minPV = image->array.SI32[ii];
                }
                if(image->array.SI32[ii] > maxPV)
                {
                    maxPV = image->array.SI32[ii];
                }
                tmp = (1.0 * image->array.SI32[ii] - average);
                RMS += tmp * tmp;
                h = (long)(1.0 * NBhistopt *
                           ((float)(image->array.SI32[ii] - minPV)) /
                           (maxPV - minPV));
                if((h > -1) && (h < NBhistopt))
                {
                    vcnt[h]++;
                }
            }
        }

        if(datatype == _DATATYPE_INT64)
        {
            minPV = image->array.SI64[0];
            maxPV = minPV;

            for(unsigned long ii = 0; ii < image->md->nelement; ii++)
            {
                if(image->array.SI64[ii] < minPV)
                {
                    minPV = image->array.SI64[ii];
                }
                if(image->array.SI64[ii] > maxPV)
                {
                    maxPV = image->array.SI64[ii];
                }
                tmp = (1.0 * image->array.SI64[ii] - average);
                RMS += tmp * tmp;
                h = (long)(1.0 * NBhistopt *
                           ((float)(image->array.SI64[ii] - minPV)) /
                           (maxPV - minPV));
                if((h > -1) && (h < NBhistopt))
                {
                    vcnt[h]++;
                }
            }
        }



        RMS   = sqrt(RMS / image->md->nelement);
        RMS01 = 0.9 * RMS01 + 0.1 * RMS; // wut

        TUI_printfw("RMS = %12.6g     ->  %12.6g\n", RMS, RMS01);


        // pix vales and histogram

        print_header(" PIXEL VALUES ", '-');
        TUI_printfw("min - max   :   %12.6e - %12.6e\n", minPV, maxPV);

        if(image->md->nelement > 25)
        {
            TUI_printfw("histogram %d levels\n", NBhistopt);
            vcntmax = 1; // initialize at one to avoid division by zero
            for(h = 0; h < NBhistopt; h++)
                if(vcnt[h] > vcntmax)
                {
                    vcntmax = vcnt[h];
                }

            for(h = 0; h < NBhistopt; h++)
            {

                customcolor = 1;
                if(h == NBhistopt - 1)
                {
                    customcolor = 2;
                }
                sprintf(line1,
                        "[%12.4e - %12.4e] %7ld",
                        (minPV + 1.0 * (maxPV - minPV) * h / NBhistopt),
                        (minPV + 1.0 * (maxPV - minPV) * (h + 1) / NBhistopt),
                        vcnt[h]);

                TUI_printfw("%s", line1);
                attron(COLOR_PAIR(customcolor));

                cnt = vcnt[h] * (wcol - 2 - strlen(line1)) / vcntmax;
                for(i = 0; i < cnt; ++i)
                {
                    TUI_printfw(" ");
                }
                attroff(COLOR_PAIR(customcolor));

                TUI_printfw("\n");
            }
        }
        else
        {
            if(image->md->datatype == _DATATYPE_FLOAT)
            {
                for(unsigned long ii = 0; ii < image->md->nelement; ii++)
                {
                    TUI_printfw("%3ld  %f\n", ii, image->array.F[ii]);
                }
            }

            if(image->md->datatype == _DATATYPE_DOUBLE)
            {
                for(unsigned long ii = 0; ii < image->md->nelement; ii++)
                {
                    TUI_printfw("%3ld  %f\n",
                                ii,
                                (float) image->array.D[ii]);
                }
            }

            if(image->md->datatype == _DATATYPE_UINT8)
            {
                for(unsigned long ii = 0; ii < image->md->nelement; ii++)
                {
                    TUI_printfw("%3ld  %5u\n", ii, image->array.UI8[ii]);
                }
            }

            if(image->md->datatype == _DATATYPE_UINT16)
            {
                for(unsigned long ii = 0; ii < image->md->nelement; ii++)
                {
                    TUI_printfw("%3ld  %5u\n", ii, image->array.UI16[ii]);
                }
            }

            if(image->md->datatype == _DATATYPE_UINT32)
            {
                for(unsigned long ii = 0; ii < image->md->nelement; ii++)
                {
                    TUI_printfw("%3ld  %5lu\n", ii, image->array.UI32[ii]);
                }
            }

            if(image->md->datatype == _DATATYPE_UINT64)
            {
                for(unsigned long ii = 0; ii < image->md->nelement; ii++)
                {
                    TUI_printfw("%3ld  %5lu\n", ii, image->array.UI64[ii]);
                }
            }

            if(image->md->datatype == _DATATYPE_INT8)
            {
                for(unsigned long ii = 0; ii < image->md->nelement; ii++)
                {
                    TUI_printfw("%3ld  %5d\n", ii, image->array.SI8[ii]);
                }
            }

            if(image->md->datatype == _DATATYPE_INT16)
            {
                for(unsigned long ii = 0; ii < image->md->nelement; ii++)
                {
                    TUI_printfw("%3ld  %5d\n", ii, image->array.SI16[ii]);
                }
            }

            if(image->md->datatype == _DATATYPE_INT32)
            {
                for(unsigned long ii = 0; ii < image->md->nelement; ii++)
                {
                    TUI_printfw("%3ld  %5ld\n",
                                ii,
                                (long) image->array.SI32[ii]);
                }
            }

            if(image->md->datatype == _DATATYPE_INT64)
            {
                for(unsigned long ii = 0; ii < image->md->nelement; ii++)
                {
                    TUI_printfw("%3ld  %5ld\n",
                                ii,
                                (long) image->array.SI64[ii]);
                }
            }

        }


        free(vcnt);
    }


    return RETURN_SUCCESS;
}
