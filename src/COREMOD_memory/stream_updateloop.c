/** @file stream_updateloop.c
 */
 
#include <sched.h>


#include "CommandLineInterface/CLIcore.h"
#include "CommandLineInterface/timeutils.h"

#include "image_ID.h"
#include "stream_sem.h"
#include "create_image.h"

#include "COREMOD_tools/COREMOD_tools.h"



// ==========================================
// Forward declaration(s)
// ==========================================



errno_t COREMOD_MEMORY_image_streamburst(
    const char *IDin_name,
    const char *IDout_name,
    long        periodus
);


imageID COREMOD_MEMORY_image_streamupdateloop(
    const char *IDinname,
    const char *IDoutname,
    long        usperiod,
    long        NBcubes,
    long        period,
    long        offsetus,
    const char *IDsync_name,
    int         semtrig,
    int         timingmode
);


imageID COREMOD_MEMORY_image_streamupdateloop_semtrig(
    const char *IDinname,
    const char *IDoutname,
    long        period,
    long        offsetus,
    const char *IDsync_name,
    int         semtrig,
    int         timingmode
);




// ==========================================
// Command line interface wrapper function(s)
// ==========================================


static errno_t COREMOD_MEMORY_image_streamburst__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_IMG)
            + CLI_checkarg(3, CLIARG_LONG)
            == 0)
    {
        COREMOD_MEMORY_image_streamburst(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.string,
            data.cmdargtoken[3].val.numl
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}



static errno_t COREMOD_MEMORY_image_streamupdateloop__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_STR)
            + CLI_checkarg(3, CLIARG_LONG)
            + CLI_checkarg(4, CLIARG_LONG)
            + CLI_checkarg(5, CLIARG_LONG)
            + CLI_checkarg(6, CLIARG_LONG)
            + CLI_checkarg(7, CLIARG_STR)
            + CLI_checkarg(8, CLIARG_LONG)
            + CLI_checkarg(9, CLIARG_LONG)
            == 0)
    {
        COREMOD_MEMORY_image_streamupdateloop(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.string,
            data.cmdargtoken[3].val.numl,
            data.cmdargtoken[4].val.numl,
            data.cmdargtoken[5].val.numl,
            data.cmdargtoken[6].val.numl,
            data.cmdargtoken[7].val.string,
            data.cmdargtoken[8].val.numl,
            data.cmdargtoken[9].val.numl
        );
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}



static errno_t COREMOD_MEMORY_image_streamupdateloop_semtrig__cli()
{
    if(0
            + CLI_checkarg(1, CLIARG_IMG)
            + CLI_checkarg(2, CLIARG_STR)
            + CLI_checkarg(3, CLIARG_LONG)
            + CLI_checkarg(4, CLIARG_LONG)
            + CLI_checkarg(5, CLIARG_STR)
            + CLI_checkarg(6, CLIARG_LONG)
            + CLI_checkarg(7, CLIARG_LONG)
            == 0)
    {
        COREMOD_MEMORY_image_streamupdateloop_semtrig(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.string,
            data.cmdargtoken[3].val.numl,
            data.cmdargtoken[4].val.numl,
            data.cmdargtoken[5].val.string,
            data.cmdargtoken[6].val.numl,
            data.cmdargtoken[7].val.numl
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

errno_t stream_updateloop_addCLIcmd()
{
	RegisterCLIcommand(
		"streamburst",
		__FILE__,
		COREMOD_MEMORY_image_streamburst__cli,
		"send burst of frames to stream",
		"<input cube> <output stream> <period[us]>",
		"streamburst inC outstream 1000",
		"errno_t COREMOD_MEMORY_image_streamburst(const char *IDin_name, const char *IDout_name, long periodus)");
	
    RegisterCLIcommand(
        "creaimstream",
        __FILE__,
        COREMOD_MEMORY_image_streamupdateloop__cli,
        "create 2D image stream from 3D cube",
        "<image3d in> <image2d out> <interval [us]> <NBcubes> <period> <offsetus> <sync stream name> <semtrig> <timing mode>",
        "creaimstream imcube imstream 1000 3 3 154 ircam1 3 0",
        "long COREMOD_MEMORY_image_streamupdateloop(const char *IDinname, const char *IDoutname, long usperiod, long NBcubes, long period, long offsetus, const char *IDsync_name, int semtrig, int timingmode)");

    RegisterCLIcommand(
        "creaimstreamstrig",
        __FILE__,
        COREMOD_MEMORY_image_streamupdateloop_semtrig__cli,
        "create 2D image stream from 3D cube, use other stream to synchronize",
        "<image3d in> <image2d out> <period [int]> <delay [us]> <sync stream> <sync sem index> <timing mode>",
        "creaimstreamstrig imcube outstream 3 152 streamsync 3 0",
        "long COREMOD_MEMORY_image_streamupdateloop_semtrig(const char *IDinname, const char *IDoutname, long period, long offsetus, const char *IDsync_name, int semtrig, int timingmode)");    

    return RETURN_SUCCESS;
}





/** @brief Send single burst of frames to stream
 *
 */

errno_t COREMOD_MEMORY_image_streamburst(
    const char *IDin_name,
    const char *IDout_name,
    long        periodus
)
{
    imageID IDin;
    imageID IDout;
    
    int        RT_priority = 80; //any number from 0-99
    struct     sched_param schedpar;
    
    char      *ptr0s; // source start 3D array ptr
    char      *ptr0; // source
    char      *ptr1; // dest
    long       framesize;

	struct timespec tim;
    
    schedpar.sched_priority = RT_priority;
    sched_setscheduler(0, SCHED_FIFO, &schedpar);


	IDin = image_ID(IDin_name);
	long naxis = data.image[IDin].md[0].naxis;
    uint32_t *arraysize;
    arraysize = (uint32_t *) malloc(sizeof(uint32_t) * 3);
    if(naxis != 3)
    {
        printf("ERROR: input image %s should be 3D\n", IDin_name);
        exit(0);
    }
    arraysize[0] = data.image[IDin].md[0].size[0];
    arraysize[1] = data.image[IDin].md[0].size[1];
    arraysize[2] = data.image[IDin].md[0].size[2];
	uint8_t datatype = data.image[IDin].md[0].datatype;
	int NBslice = arraysize[2];

	// check that IDout has same format
	IDout = image_ID(IDout_name);
	if(data.image[IDout].md[0].size[0] != data.image[IDin].md[0].size[0])
	{
		printf("ERROR: in and out have different size\n");
		return RETURN_FAILURE;
	}
	if(data.image[IDout].md[0].size[1] != data.image[IDin].md[0].size[1])
	{
		printf("ERROR: in and out have different size\n");
		return RETURN_FAILURE;
	}
	if(data.image[IDout].md[0].datatype != data.image[IDin].md[0].datatype)
	{
		printf("ERROR: in and out have different datatype\n");
		return RETURN_FAILURE;
	}	
	
	
    switch(datatype)
    {
    case _DATATYPE_INT8:
        ptr0s = (char *) data.image[IDin].array.SI8;
        ptr1 = (char *) data.image[IDout].array.SI8;
        framesize = data.image[IDin].md[0].size[0] *
                    data.image[IDin].md[0].size[1] * SIZEOF_DATATYPE_INT8;
        break;

    case _DATATYPE_UINT8:
        ptr0s = (char *) data.image[IDin].array.UI8;
        ptr1 = (char *) data.image[IDout].array.UI8;
        framesize = data.image[IDin].md[0].size[0] *
                    data.image[IDin].md[0].size[1] * SIZEOF_DATATYPE_UINT8;
        break;

    case _DATATYPE_INT16:
        ptr0s = (char *) data.image[IDin].array.SI16;
        ptr1 = (char *) data.image[IDout].array.SI16;
        framesize = data.image[IDin].md[0].size[0] *
                    data.image[IDin].md[0].size[1] * SIZEOF_DATATYPE_INT16;
        break;

    case _DATATYPE_UINT16:
        ptr0s = (char *) data.image[IDin].array.UI16;
        ptr1 = (char *) data.image[IDout].array.UI16;
        framesize = data.image[IDin].md[0].size[0] *
                    data.image[IDin].md[0].size[1] * SIZEOF_DATATYPE_UINT16;
        break;

    case _DATATYPE_INT32:
        ptr0s = (char *) data.image[IDin].array.SI32;
        ptr1 = (char *) data.image[IDout].array.SI32;
        framesize = data.image[IDin].md[0].size[0] *
                    data.image[IDin].md[0].size[1] * SIZEOF_DATATYPE_INT32;
        break;

    case _DATATYPE_UINT32:
        ptr0s = (char *) data.image[IDin].array.UI32;
        ptr1 = (char *) data.image[IDout].array.UI32;
        framesize = data.image[IDin].md[0].size[0] *
                    data.image[IDin].md[0].size[1] * SIZEOF_DATATYPE_UINT32;
        break;

    case _DATATYPE_INT64:
        ptr0s = (char *) data.image[IDin].array.SI64;
        ptr1 = (char *) data.image[IDout].array.SI64;
        framesize = data.image[IDin].md[0].size[0] *
                    data.image[IDin].md[0].size[1] * SIZEOF_DATATYPE_INT64;
        break;

    case _DATATYPE_UINT64:
        ptr0s = (char *) data.image[IDin].array.UI64;
        ptr1 = (char *) data.image[IDout].array.UI64;
        framesize = data.image[IDin].md[0].size[0] *
                    data.image[IDin].md[0].size[1] * SIZEOF_DATATYPE_UINT64;
        break;


    case _DATATYPE_FLOAT:
        ptr0s = (char *) data.image[IDin].array.F;
        ptr1 = (char *) data.image[IDout].array.F;
        framesize = data.image[IDin].md[0].size[0] *
                    data.image[IDin].md[0].size[1] * sizeof(float);
        break;

    case _DATATYPE_DOUBLE:
        ptr0s = (char *) data.image[IDin].array.D;
        ptr1 = (char *) data.image[IDout].array.D;
        framesize = data.image[IDin].md[0].size[0] *
                    data.image[IDin].md[0].size[1] * sizeof(double);
        break;

    }

	
   tim.tv_sec = 0;
   tim.tv_nsec = (long) (1000*periodus);
	

	for(int slice = 0; slice < NBslice; slice++)
	{
		if(nanosleep(&tim , NULL) < 0 )   
		{
			printf("Nano sleep system call failed \n");
		}

		
        ptr0 = ptr0s + slice * framesize;
        
        ptr0 = ptr0s + slice * framesize;
        data.image[IDout].md[0].write = 1;
        memcpy((void *) ptr1, (void *) ptr0, framesize);
        data.image[IDout].md[0].cnt1 = slice;
        data.image[IDout].md[0].cnt0++;
        data.image[IDout].md[0].write = 0;
        COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);
	}

    free(arraysize);

    return RETURN_SUCCESS;
}






/** @brief takes a 3Dimage(s) (circular buffer(s)) and writes slices to a 2D image with time interval specified in us
 *
 *
 * If NBcubes=1, then the circular buffer named IDinname is sent to IDoutname at a frequency of 1/usperiod MHz
 * If NBcubes>1, several circular buffers are used, named ("%S_%03ld", IDinname, cubeindex). Semaphore semtrig of image IDsync_name triggers switch between circular buffers, with a delay of offsetus. The number of consecutive sem posts required to advance to the next circular buffer is period
 *
 * @param IDinname      Name of DM circular buffer (appended by _000, _001 etc... if NBcubes>1)
 * @param IDoutname     Output channel stream
 * @param usperiod      Interval between consecutive frames [us]
 * @param NBcubes       Number of input circular buffers
 * @param period        If NBcubes>1: number of input triggers required to advance to next input buffer
 * @param offsetus      If NBcubes>1: time offset [us] between input trigger and input buffer switch
 * @param IDsync_name   If NBcubes>1: Stream used for synchronization
 * @param semtrig       If NBcubes>1: semaphore used for synchronization
 * @param timingmode    Not used
 *
 *
 */
imageID COREMOD_MEMORY_image_streamupdateloop(
    const char *IDinname,
    const char *IDoutname,
    long        usperiod,
    long        NBcubes,
    long        period,
    long        offsetus,
    const char *IDsync_name,
    int         semtrig,
    __attribute__((unused)) int         timingmode
)
{
    imageID   *IDin;
    long       cubeindex;
    char       imname[200];
    long       IDsync;
    unsigned long long  cntsync;
    long       pcnt = 0;
    long       offsetfr = 0;
    long       offsetfrcnt = 0;
    int        cntDelayMode = 0;

    imageID    IDout;
    long       kk;
    uint32_t  *arraysize;
    long       naxis;
    uint8_t    datatype;
    char      *ptr0s; // source start 3D array ptr
    char      *ptr0; // source
    char      *ptr1; // dest
    long       framesize;
//    int        semval;

    int        RT_priority = 80; //any number from 0-99
    struct     sched_param schedpar;

    long       twait1;
    struct     timespec t0;
    struct     timespec t1;
    double     tdiffv;
    struct     timespec tdiff;

    int        SyncSlice = 0;



    schedpar.sched_priority = RT_priority;
#ifndef __MACH__
    sched_setscheduler(0, SCHED_FIFO,
                       &schedpar); //other option is SCHED_RR, might be faster
#endif


    PROCESSINFO *processinfo;
    if(data.processinfo == 1)
    {
        // CREATE PROCESSINFO ENTRY
        // see processtools.c in module CommandLineInterface for details
        //
        char pinfoname[200];
        sprintf(pinfoname, "streamloop-%s", IDoutname);
        processinfo = processinfo_shm_create(pinfoname, 0);
        processinfo->loopstat = 0; // loop initialization

        strcpy(processinfo->source_FUNCTION, __FUNCTION__);
        strcpy(processinfo->source_FILE,     __FILE__);
        processinfo->source_LINE = __LINE__;

        char msgstring[200];
        sprintf(msgstring, "%s->%s", IDinname, IDoutname);
        processinfo_WriteMessage(processinfo, msgstring);
    }




    if(NBcubes < 1)
    {
        printf("ERROR: invalid number of input cubes, needs to be >0");
        return RETURN_FAILURE;
    }


    int sync_semwaitindex;
    IDin = (long *) malloc(sizeof(long) * NBcubes);
    SyncSlice = 0;
    if(NBcubes == 1)
    {
        IDin[0] = image_ID(IDinname);

        // in single cube mode, optional sync stream drives updates to next slice within cube
        IDsync = image_ID(IDsync_name);
        if(IDsync != -1)
        {
            SyncSlice = 1;
            sync_semwaitindex = ImageStreamIO_getsemwaitindex(&data.image[IDsync], semtrig);
        }
    }
    else
    {
        IDsync = image_ID(IDsync_name);
        sync_semwaitindex = ImageStreamIO_getsemwaitindex(&data.image[IDsync], semtrig);

        for(cubeindex = 0; cubeindex < NBcubes; cubeindex++)
        {
            sprintf(imname, "%s_%03ld", IDinname, cubeindex);
            IDin[cubeindex] = image_ID(imname);
        }
        offsetfr = (long)(0.5 + 1.0 * offsetus / usperiod);

        printf("FRAMES OFFSET = %ld\n", offsetfr);
    }



    printf("SyncSlice = %d\n", SyncSlice);

    printf("Creating / connecting to image stream ...\n");
    fflush(stdout);


    naxis = data.image[IDin[0]].md[0].naxis;
    arraysize = (uint32_t *) malloc(sizeof(uint32_t) * 3);
    if(naxis != 3)
    {
        printf("ERROR: input image %s should be 3D\n", IDinname);
        exit(0);
    }
    arraysize[0] = data.image[IDin[0]].md[0].size[0];
    arraysize[1] = data.image[IDin[0]].md[0].size[1];
    arraysize[2] = data.image[IDin[0]].md[0].size[2];



    datatype = data.image[IDin[0]].md[0].datatype;

    IDout = image_ID(IDoutname);
    if(IDout == -1)
    {
        IDout = create_image_ID(IDoutname, 2, arraysize, datatype, 1, 0);
        COREMOD_MEMORY_image_set_createsem(IDoutname, IMAGE_NB_SEMAPHORE);
    }

    cubeindex = 0;
    pcnt = 0;
    if(NBcubes > 1)
    {
        cntsync = data.image[IDsync].md[0].cnt0;
    }

    twait1 = usperiod;
    kk = 0;
    cntDelayMode = 0;



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
            while(processinfo->CTRLval == 1)  // pause
            {
                usleep(50);
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



        if(NBcubes > 1)
        {
            if(cntsync != data.image[IDsync].md[0].cnt0)
            {
                pcnt++;
                cntsync = data.image[IDsync].md[0].cnt0;
            }
            if(pcnt == period)
            {
                pcnt = 0;
                offsetfrcnt = 0;
                cntDelayMode = 1;
            }

            if(cntDelayMode == 1)
            {
                if(offsetfrcnt < offsetfr)
                {
                    offsetfrcnt++;
                }
                else
                {
                    cntDelayMode = 0;
                    cubeindex++;
                    kk = 0;
                }
            }
            if(cubeindex == NBcubes)
            {
                cubeindex = 0;
            }
        }


        switch(datatype)
        {

            case _DATATYPE_INT8:
                ptr0s = (char *) data.image[IDin[cubeindex]].array.SI8;
                ptr1 = (char *) data.image[IDout].array.SI8;
                framesize = data.image[IDin[cubeindex]].md[0].size[0] *
                            data.image[IDin[cubeindex]].md[0].size[1] * SIZEOF_DATATYPE_INT8;
                break;

            case _DATATYPE_UINT8:
                ptr0s = (char *) data.image[IDin[cubeindex]].array.UI8;
                ptr1 = (char *) data.image[IDout].array.UI8;
                framesize = data.image[IDin[cubeindex]].md[0].size[0] *
                            data.image[IDin[cubeindex]].md[0].size[1] * SIZEOF_DATATYPE_UINT8;
                break;

            case _DATATYPE_INT16:
                ptr0s = (char *) data.image[IDin[cubeindex]].array.SI16;
                ptr1 = (char *) data.image[IDout].array.SI16;
                framesize = data.image[IDin[cubeindex]].md[0].size[0] *
                            data.image[IDin[cubeindex]].md[0].size[1] * SIZEOF_DATATYPE_INT16;
                break;

            case _DATATYPE_UINT16:
                ptr0s = (char *) data.image[IDin[cubeindex]].array.UI16;
                ptr1 = (char *) data.image[IDout].array.UI16;
                framesize = data.image[IDin[cubeindex]].md[0].size[0] *
                            data.image[IDin[cubeindex]].md[0].size[1] * SIZEOF_DATATYPE_UINT16;
                break;

            case _DATATYPE_INT32:
                ptr0s = (char *) data.image[IDin[cubeindex]].array.SI32;
                ptr1 = (char *) data.image[IDout].array.SI32;
                framesize = data.image[IDin[cubeindex]].md[0].size[0] *
                            data.image[IDin[cubeindex]].md[0].size[1] * SIZEOF_DATATYPE_INT32;
                break;

            case _DATATYPE_UINT32:
                ptr0s = (char *) data.image[IDin[cubeindex]].array.UI32;
                ptr1 = (char *) data.image[IDout].array.UI32;
                framesize = data.image[IDin[cubeindex]].md[0].size[0] *
                            data.image[IDin[cubeindex]].md[0].size[1] * SIZEOF_DATATYPE_UINT32;
                break;

            case _DATATYPE_INT64:
                ptr0s = (char *) data.image[IDin[cubeindex]].array.SI64;
                ptr1 = (char *) data.image[IDout].array.SI64;
                framesize = data.image[IDin[cubeindex]].md[0].size[0] *
                            data.image[IDin[cubeindex]].md[0].size[1] * SIZEOF_DATATYPE_INT64;
                break;

            case _DATATYPE_UINT64:
                ptr0s = (char *) data.image[IDin[cubeindex]].array.UI64;
                ptr1 = (char *) data.image[IDout].array.UI64;
                framesize = data.image[IDin[cubeindex]].md[0].size[0] *
                            data.image[IDin[cubeindex]].md[0].size[1] * SIZEOF_DATATYPE_UINT64;
                break;


            case _DATATYPE_FLOAT:
                ptr0s = (char *) data.image[IDin[cubeindex]].array.F;
                ptr1 = (char *) data.image[IDout].array.F;
                framesize = data.image[IDin[cubeindex]].md[0].size[0] *
                            data.image[IDin[cubeindex]].md[0].size[1] * sizeof(float);
                break;

            case _DATATYPE_DOUBLE:
                ptr0s = (char *) data.image[IDin[cubeindex]].array.D;
                ptr1 = (char *) data.image[IDout].array.D;
                framesize = data.image[IDin[cubeindex]].md[0].size[0] *
                            data.image[IDin[cubeindex]].md[0].size[1] * sizeof(double);
                break;

        }




        clock_gettime(CLOCK_REALTIME, &t0);

        ptr0 = ptr0s + kk * framesize;
        data.image[IDout].md[0].write = 1;
        memcpy((void *) ptr1, (void *) ptr0, framesize);
        data.image[IDout].md[0].cnt1 = kk;
        data.image[IDout].md[0].cnt0++;
        data.image[IDout].md[0].write = 0;
        COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);

        kk++;
        if(kk == data.image[IDin[0]].md[0].size[2])
        {
            kk = 0;
        }



        if(SyncSlice == 0)
        {
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
        }
        else
        {
            sem_wait(data.image[IDsync].semptr[sync_semwaitindex]);
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

    free(IDin);

    return IDout;
}







// takes a 3Dimage (circular buffer) and writes slices to a 2D image synchronized with an image semaphore
imageID COREMOD_MEMORY_image_streamupdateloop_semtrig(
    const char *IDinname,
    const char *IDoutname,
    long        period,
    long        offsetus,
    const char *IDsync_name,
    int         semtrig,
    __attribute__((unused)) int         timingmode
)
{
    imageID    IDin;
    imageID    IDout;
    imageID    IDsync;

    long       kk;
    long       kk1;

    uint32_t  *arraysize;
    long       naxis;
    uint8_t    datatype;
    char      *ptr0s; // source start 3D array ptr
    char      *ptr0; // source
    char      *ptr1; // dest
    long       framesize;
//    int        semval;

    int        RT_priority = 80; //any number from 0-99
    struct     sched_param schedpar;


    schedpar.sched_priority = RT_priority;
#ifndef __MACH__
    sched_setscheduler(0, SCHED_FIFO,
                       &schedpar); //other option is SCHED_RR, might be faster
#endif


    printf("Creating / connecting to image stream ...\n");
    fflush(stdout);

    IDin = image_ID(IDinname);
    naxis = data.image[IDin].md[0].naxis;
    arraysize = (uint32_t *) malloc(sizeof(uint32_t) * 3);
    if(naxis != 3)
    {
        printf("ERROR: input image %s should be 3D\n", IDinname);
        exit(0);
    }
    arraysize[0] = data.image[IDin].md[0].size[0];
    arraysize[1] = data.image[IDin].md[0].size[1];
    arraysize[2] = data.image[IDin].md[0].size[2];





    datatype = data.image[IDin].md[0].datatype;

    IDout = image_ID(IDoutname);
    if(IDout == -1)
    {
        IDout = create_image_ID(IDoutname, 2, arraysize, datatype, 1, 0);
        COREMOD_MEMORY_image_set_createsem(IDoutname, IMAGE_NB_SEMAPHORE);
    }

    switch(datatype)
    {

        case _DATATYPE_INT8:
            ptr0s = (char *) data.image[IDin].array.SI8;
            ptr1 = (char *) data.image[IDout].array.SI8;
            framesize = data.image[IDin].md[0].size[0] * data.image[IDin].md[0].size[1] *
                        SIZEOF_DATATYPE_INT8;
            break;

        case _DATATYPE_UINT8:
            ptr0s = (char *) data.image[IDin].array.UI8;
            ptr1 = (char *) data.image[IDout].array.UI8;
            framesize = data.image[IDin].md[0].size[0] * data.image[IDin].md[0].size[1] *
                        SIZEOF_DATATYPE_UINT8;
            break;

        case _DATATYPE_INT16:
            ptr0s = (char *) data.image[IDin].array.SI16;
            ptr1 = (char *) data.image[IDout].array.SI16;
            framesize = data.image[IDin].md[0].size[0] * data.image[IDin].md[0].size[1] *
                        SIZEOF_DATATYPE_INT16;
            break;

        case _DATATYPE_UINT16:
            ptr0s = (char *) data.image[IDin].array.UI16;
            ptr1 = (char *) data.image[IDout].array.UI16;
            framesize = data.image[IDin].md[0].size[0] * data.image[IDin].md[0].size[1] *
                        SIZEOF_DATATYPE_UINT16;
            break;

        case _DATATYPE_INT32:
            ptr0s = (char *) data.image[IDin].array.SI32;
            ptr1 = (char *) data.image[IDout].array.SI32;
            framesize = data.image[IDin].md[0].size[0] * data.image[IDin].md[0].size[1] *
                        SIZEOF_DATATYPE_INT32;
            break;

        case _DATATYPE_UINT32:
            ptr0s = (char *) data.image[IDin].array.UI32;
            ptr1 = (char *) data.image[IDout].array.UI32;
            framesize = data.image[IDin].md[0].size[0] * data.image[IDin].md[0].size[1] *
                        SIZEOF_DATATYPE_UINT32;
            break;

        case _DATATYPE_INT64:
            ptr0s = (char *) data.image[IDin].array.SI64;
            ptr1 = (char *) data.image[IDout].array.SI64;
            framesize = data.image[IDin].md[0].size[0] * data.image[IDin].md[0].size[1] *
                        SIZEOF_DATATYPE_INT64;
            break;

        case _DATATYPE_UINT64:
            ptr0s = (char *) data.image[IDin].array.UI64;
            ptr1 = (char *) data.image[IDout].array.UI64;
            framesize = data.image[IDin].md[0].size[0] * data.image[IDin].md[0].size[1] *
                        SIZEOF_DATATYPE_UINT64;
            break;


        case _DATATYPE_FLOAT:
            ptr0s = (char *) data.image[IDin].array.F;
            ptr1 = (char *) data.image[IDout].array.F;
            framesize = data.image[IDin].md[0].size[0] * data.image[IDin].md[0].size[1] *
                        sizeof(float);
            break;

        case _DATATYPE_DOUBLE:
            ptr0s = (char *) data.image[IDin].array.D;
            ptr1 = (char *) data.image[IDout].array.D;
            framesize = data.image[IDin].md[0].size[0] * data.image[IDin].md[0].size[1] *
                        sizeof(double);
            break;
    }




    IDsync = image_ID(IDsync_name);

    kk = 0;
    kk1 = 0;

    int sync_semwaitindex;
    sync_semwaitindex = ImageStreamIO_getsemwaitindex(&data.image[IDin], semtrig);

    while(1)
    {
        sem_wait(data.image[IDsync].semptr[sync_semwaitindex]);

        kk++;
        if(kk == period) // UPDATE
        {
            kk = 0;
            kk1++;
            if(kk1 == data.image[IDin].md[0].size[2])
            {
                kk1 = 0;
            }
            usleep(offsetus);
            ptr0 = ptr0s + kk1 * framesize;
            data.image[IDout].md[0].write = 1;
            memcpy((void *) ptr1, (void *) ptr0, framesize);
            COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);
            data.image[IDout].md[0].cnt0++;
            data.image[IDout].md[0].write = 0;
        }
    }

    // release semaphore
    data.image[IDsync].semReadPID[sync_semwaitindex] = 0;

    return IDout;
}





