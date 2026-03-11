/**
 * @file statusstat.c
 * @brief Status statistics monitoring
 *
 * Uses FPS V2 framework.
 */

#include <sched.h>
#include <time.h>

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"

#include "timeutils.h"

#include "COREMOD_memory/COREMOD_memory.h"

/* forward decl */
imageID COREMOD_TOOLS_statusStat(
    const char *IDstat_name,
    long indexmax);


/* ================================================================
 *  PARAMS
 * ============================================================= */

static char p_imname[
    FUNCTION_PARAMETER_STRMAXLEN]
    = "imst";
static long long p_nbstep = 100000;

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "ctsmstats",
    .cmdkey      = "ctsmstats",
    .description =
        "monitors shared variable status"
};

#define FPS_PARAMS(X) \
    X(".imname", p_imname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "status image") \
    X(".nbstep", &p_nbstep, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "number of steps")

static FPS_CLI_BINDING my_bindings[] = {
    FPS_PARAMS(FPS_X_BINDING)
};

static const int nb_bindings =
    sizeof(my_bindings) /
    sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = {
    FPS_PARAMS(FPS_X_FARG)
};

static CLICMDDATA CLIcmddata = {
    "", "", CLICMD_FIELDS_DEFAULTS
};

static CMDSETTINGS cms = {0};

static __attribute__((constructor))
void init_cms(void)
{
    strncpy(CLIcmddata.key,
            FPS_app_info.cmdkey,
            sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description,
            FPS_app_info.description,
            sizeof(CLIcmddata.description)
            - 1);
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings = &cms;
    }
}

static errno_t compute_function()
{
    COREMOD_TOOLS_statusStat(
        p_imname, p_nbstep);
    return RETURN_SUCCESS;
}

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_COREMOD_tools__statusstat()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    int cmdi = RegisterCLIcmd(
        CLIcmddata, CLIfunction);
    CLIcmddata.cmdsettings =
        &data.cmd[cmdi].cmdsettings;
    return RETURN_SUCCESS;
}
#endif

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

    IDstat = image_ID(IDstat_name, dcimg, dcnimg);

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
        dcimg[IDout].array.SI64[st] = 0;
    }

    schedpar.sched_priority = RT_priority;
    sched_setscheduler(0, SCHED_FIFO, &schedpar);

    printf("Measuring status distribution \n");
    fflush(stdout);

    clock_gettime(CLOCK_MILK, &t1);
    for(k = 0; k < NBkiter; k++)
    {
        double tdiffv;

        usleep((long)(usec0 + usec1 * ((double) k / NBkiter)));
        st = dcimg[IDstat].array.UI16[0];
        if(st < indexmax)
        {
            dcimg[IDout].array.SI64[st]++;
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
                cnttot += dcimg[IDout].array.SI64[st];
            }

            for(st = 0; st < indexmax; st++)
            {
                printf("STATUS  %5d    %20ld   %6.3f  \n",
                       st,
                       dcimg[IDout].array.SI64[st],
                       100.0 * dcimg[IDout].array.SI64[st] / cnttot);
            }
        }
    }

    printf("\n");
    for(st = 0; st < indexmax; st++)
    {
        printf("STATUS  %5d    %10ld\n", st, dcimg[IDout].array.SI64[st]);
    }

    printf("\n");

    return (IDout);
}
