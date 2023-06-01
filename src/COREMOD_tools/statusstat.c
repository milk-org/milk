/**
 * @file statusstat.c
 */

#include <sched.h>
#include <time.h>

#include "CommandLineInterface/CLIcore.h"

#include "CommandLineInterface/timeutils.h"

#include "COREMOD_memory/COREMOD_memory.h"

// ==========================================
// Forward declaration(s)
// ==========================================

imageID COREMOD_TOOLS_statusStat(const char *IDstat_name, long indexmax);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

errno_t COREMOD_TOOLS_statusStat_cli()
{
    if(0 + CLI_checkarg(1, CLIARG_IMG) + CLI_checkarg(2, CLIARG_INT64) == 0)
    {
        COREMOD_TOOLS_statusStat(data.cmdargtoken[1].val.string,
                                 data.cmdargtoken[2].val.numl);

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

errno_t statusstat_addCLIcmd()
{

    RegisterCLIcommand("ctsmstats",
                       __FILE__,
                       COREMOD_TOOLS_statusStat_cli,
                       "monitors shared variable status",
                       "<imname> <NBstep>",
                       "ctsmstats imst 100000",
                       "long COREMOD_TOOLS_statusStat(const char *IDstat_name, "
                       "long indexmax)");

    return RETURN_SUCCESS;
}

//
// watch shared memory status image and perform timing statistics
//
imageID COREMOD_TOOLS_statusStat(const char *IDstat_name, long indexmax)
{
    imageID            IDout;
    int                RT_priority = 91; //any number from 0-99
    struct sched_param schedpar;
    float              usec0 = 50.0;
    float              usec1 = 150.0;
    long long          k;
    long long          NBkiter = 2000000000;
    imageID            IDstat;

    unsigned short st;

    struct timespec t1;
    struct timespec t2;
    struct timespec tdiff;
    double          tdisplay = 1.0; // interval
    double          tdiffv1  = 0.0;
    uint32_t       *sizearray;

    long cnttot;

    IDstat = image_ID(IDstat_name);

    sizearray = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    if(sizearray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    sizearray[0] = indexmax;
    sizearray[1] = 1;
    create_image_ID("statout", 2, sizearray, _DATATYPE_INT64, 0, 0, 0, &IDout);
    free(sizearray);

    for(st = 0; st < indexmax; st++)
    {
        data.image[IDout].array.SI64[st] = 0;
    }

    schedpar.sched_priority = RT_priority;
    sched_setscheduler(0, SCHED_FIFO, &schedpar);

    printf("Measuring status distribution \n");
    fflush(stdout);

    clock_gettime(CLOCK_MILK, &t1);
    for(k = 0; k < NBkiter; k++)
    {
        double tdiffv;

        usleep((long)(usec0 + usec1 * (1.0 * k / NBkiter)));
        st = data.image[IDstat].array.UI16[0];
        if(st < indexmax)
        {
            data.image[IDout].array.SI64[st]++;
        }

        clock_gettime(CLOCK_MILK, &t2);
        tdiff  = timespec_diff(t1, t2);
        tdiffv = 1.0 * tdiff.tv_sec + 1.0e-9 * tdiff.tv_nsec;

        if(tdiffv > tdiffv1)
        {
            tdiffv1 += tdisplay;
            printf("\n");
            printf("============== %10lld  %d  ==================\n", k, st);
            printf("\n");
            cnttot = 0;
            for(st = 0; st < indexmax; st++)
            {
                cnttot += data.image[IDout].array.SI64[st];
            }

            for(st = 0; st < indexmax; st++)
            {
                printf("STATUS  %5d    %20ld   %6.3f  \n",
                       st,
                       data.image[IDout].array.SI64[st],
                       100.0 * data.image[IDout].array.SI64[st] / cnttot);
            }
        }
    }

    printf("\n");
    for(st = 0; st < indexmax; st++)
    {
        printf("STATUS  %5d    %10ld\n", st, data.image[IDout].array.SI64[st]);
    }

    printf("\n");

    return (IDout);
}
