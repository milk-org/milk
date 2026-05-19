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
    long       indexmax);


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
        "monitors shared variable status",
    .description_long =
        "Collect and display statistics about process status values written to shared memory streams. Histograms the status values over time."
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

static const int __attribute__((unused)) nb_bindings =
    sizeof(my_bindings) /
    sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = {
    FPS_PARAMS(FPS_X_FARG)
};

static CLICMDDATA CLIcmddata = {
    "", "", CLICMD_FIELDS_DEFAULTS
};

FPS_CMDSETTINGS_INIT(main, CLIcmddata, FPS_app_info)

static MILK_HOT errno_t __attribute__((unused)) compute_function()
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

/**
 * Watch shared memory status image
 * and perform timing statistics.
 */
imageID COREMOD_TOOLS_statusStat(
    const char *IDstat_name,
    long       indexmax)
{
    int RT_priority = 91;
    struct sched_param schedpar;
    float usec0 = 50.0;
    float usec1 = 150.0;
    long long NBkiter = 2000000000;

    IMGID imgstat =
        imgid_make_from_name(
            IDstat_name);
    resolveIMGID(&imgstat,
                 ERRMODE_ABORT,
                 dcimg, dcnimg);

    IMGID imgout =
        imgid_make_from_name(
            "statout");
    imgout.mdt->naxis = 2;
    imgout.mdt->size[0] = indexmax;
    imgout.mdt->size[1] = 1;
    imgout.mdt->datatype =
        _DATATYPE_INT64;
    imgout.mdt->shared = 0;
    imgout.im = (IMAGE *) calloc(
        1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    for(unsigned short st = 0;
        st < indexmax; st++)
    {
        imgout.im->array.SI64[st] = 0;
    }

    schedpar.sched_priority =
        RT_priority;
    sched_setscheduler(
        0, SCHED_FIFO, &schedpar);

    printf(
        "Measuring status"
        " distribution \n");
    fflush(stdout);

    struct timespec t1;
    clock_gettime(CLOCK_MILK, &t1);
    double tdiffv1 = 0.0;
    double tdisplay = 1.0;

    for(long long k = 0;
        k < NBkiter; k++)
    {
        usleep(
            (long)(usec0
                   + usec1
                     * ((double) k
                        / NBkiter)));
        unsigned short st =
            imgstat.im->array.UI16[0];
        if(st < indexmax)
        {
            imgout.im
                ->array.SI64[st]++;
        }

        struct timespec t2;
        clock_gettime(CLOCK_MILK, &t2);
        struct timespec tdiff =
            timespec_diff(t1, t2);
        double tdiffv =
            1.0 * tdiff.tv_sec
            + 1.0e-9
              * tdiff.tv_nsec;

        if(tdiffv > tdiffv1)
        {
            tdiffv1 += tdisplay;
            printf("\n");
            printf(
                "=============="
                " %10lld  %d  "
                "==================\n",
                k, st);
            printf("\n");
            long cnttot = 0;
            for(st = 0;
                st < indexmax; st++)
            {
                cnttot +=
                    imgout.im
                        ->array
                        .SI64[st];
            }

            for(st = 0;
                st < indexmax; st++)
            {
                printf(
                    "STATUS  %5d"
                    "    %20ld"
                    "   %6.3f  \n",
                    st,
                    imgout.im
                        ->array
                        .SI64[st],
                    100.0
                    * imgout.im
                          ->array
                          .SI64[st]
                    / cnttot);
            }
        }
    }

    printf("\n");
    for(unsigned short st = 0;
        st < indexmax; st++)
    {
        printf(
            "STATUS  %5d"
            "    %10ld\n",
            st,
            imgout.im
                ->array.SI64[st]);
    }

    printf("\n");

    return imgout.ID;
}
