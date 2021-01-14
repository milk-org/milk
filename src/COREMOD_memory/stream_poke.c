/**
 * @file    stream_poke.c
 * @brief   poke image stream
 */



#include "CommandLineInterface/CLIcore.h"
#include "CommandLineInterface/timeutils.h"

#include "image_ID.h"
#include "stream_sem.h"

#include "COREMOD_tools/COREMOD_tools.h"





// ==========================================
// forward declarations
// ==========================================

imageID COREMOD_MEMORY_streamPoke(
    const char *IDstream_name,
    long        usperiod
);


// ==========================================
// command line interface wrapper functions
// ==========================================

static errno_t COREMOD_MEMORY_streamPoke__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_LONG)
            == 0)
    {
        COREMOD_MEMORY_streamPoke(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.numl
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}




// ==========================================
// Register CLI command(s)
// ==========================================


errno_t stream_poke_addCLIcmd()
{
    RegisterCLIcommand(
        "streampoke",
        __FILE__,
        COREMOD_MEMORY_streamPoke__cli,
        "Poke image stream at regular interval",
        "<in stream> <poke period [us]>",
        "streampoke stream 100",
        "long COREMOD_MEMORY_streamPoke(const char *IDstream_name, long usperiod)");

    return RETURN_SUCCESS;
}








/**
 * ## Purpose
 *
 * Poke a stream at regular time interval\n
 * Does not change shared memory content\n
 *
 */
imageID COREMOD_MEMORY_streamPoke(
    const char *IDstream_name,
    long        usperiod
)
{
    imageID ID;
    long    twait1;
    struct  timespec t0;
    struct  timespec t1;
    double  tdiffv;
    struct  timespec tdiff;

    ID = image_ID(IDstream_name);



    PROCESSINFO *processinfo;
    if(data.processinfo == 1)
    {
        // CREATE PROCESSINFO ENTRY
        // see processtools.c in module CommandLineInterface for details
        //
        char pinfoname[200];
        sprintf(pinfoname, "streampoke-%s", IDstream_name);
        processinfo = processinfo_shm_create(pinfoname, 0);
        processinfo->loopstat = 0; // loop initialization

        strcpy(processinfo->source_FUNCTION, __FUNCTION__);
        strcpy(processinfo->source_FILE,     __FILE__);
        processinfo->source_LINE = __LINE__;

        char msgstring[200];
        sprintf(msgstring, "%s", IDstream_name);
        processinfo_WriteMessage(processinfo, msgstring);
    }

    if(data.processinfo == 1)
    {
        processinfo->loopstat = 1;    // loop running
    }
    int loopOK = 1;
    int loopCTRLexit = 0; // toggles to 1 when loop is set to exit cleanly
    long loopcnt = 0;


    while(loopOK == 1)
    {
        // processinfo control
        if(data.processinfo == 1)
        {
            while(processinfo->CTRLval == 1)   // pause
            {
                struct timespec treq, trem;
                treq.tv_sec = 0;
                treq.tv_nsec = 50000;
                nanosleep(&treq, &trem);
            }

            if(processinfo->CTRLval == 2) // single iteration
            {
                processinfo->CTRLval = 1;
            }

            if(processinfo->CTRLval == 3) // exit loop
            {
                loopCTRLexit = 1;
            }
        }


        clock_gettime(CLOCK_REALTIME, &t0);

        data.image[ID].md[0].write = 1;
        data.image[ID].md[0].cnt0++;
        data.image[ID].md[0].write = 0;
        COREMOD_MEMORY_image_set_sempost_byID(ID, -1);



        usleep(twait1);

        clock_gettime(CLOCK_REALTIME, &t1);
        tdiff = timespec_diff(t0, t1);
        tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

        if(tdiffv < 1.0e-6 * usperiod)
        {
            twait1 ++;
        }
        else
        {
            twait1 --;
        }

        if(twait1 < 0)
        {
            twait1 = 0;
        }
        if(twait1 > usperiod)
        {
            twait1 = usperiod;
        }


        if(loopCTRLexit == 1)
        {
            loopOK = 0;
            if(data.processinfo == 1)
            {
                struct timespec tstop;
                struct tm *tstoptm;
                char msgstring[200];

                clock_gettime(CLOCK_REALTIME, &tstop);
                tstoptm = gmtime(&tstop.tv_sec);

                sprintf(msgstring, "CTRLexit at %02d:%02d:%02d.%03d", tstoptm->tm_hour,
                        tstoptm->tm_min, tstoptm->tm_sec, (int)(0.000001 * (tstop.tv_nsec)));
                strncpy(processinfo->statusmsg, msgstring, 200);

                processinfo->loopstat = 3; // clean exit
            }
        }

        loopcnt++;
        if(data.processinfo == 1)
        {
            processinfo->loopcnt = loopcnt;
        }
    }


    return ID;
}


