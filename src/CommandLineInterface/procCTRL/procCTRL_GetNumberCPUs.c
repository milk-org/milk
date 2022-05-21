#include "CLIcore.h"
#include <processtools.h>


#ifdef USE_HWLOC
#include <hwloc.h>
#endif


/**
 * ## Purpose
 *
 * detects the number of CPU and fill the cpuids
 *
 * ## Description
 *
 * populates cpuids array with the global system PU numbers in the physical order:
 * [PU0 of CPU0, PU1 of CPU0, ... PU0 of CPU1, PU1 of CPU1, ...]
 *
 */

int GetNumberCPUs(PROCINFOPROC *pinfop)
{
    int pu_index = 0;

#ifdef USE_HWLOC

    static int initStatus = 0;

    if (initStatus == 0)
    {
        initStatus             = 1;
        unsigned int     depth = 0;
        hwloc_topology_t topology;

        /* Allocate and initialize topology object. */
        hwloc_topology_init(&topology);

        /* ... Optionally, put detection configuration here to ignore
           some objects types, define a synthetic topology, etc....
           The default is to detect all the objects of the machine that
           the caller is allowed to access.  See Configure Topology
           Detection. */

        /* Perform the topology detection. */
        hwloc_topology_load(topology);

        depth               = hwloc_get_type_depth(topology, HWLOC_OBJ_PACKAGE);
        pinfop->NBcpusocket = hwloc_get_nbobjs_by_depth(topology, depth);

        depth          = hwloc_get_type_depth(topology, HWLOC_OBJ_PU);
        pinfop->NBcpus = hwloc_get_nbobjs_by_depth(topology, depth);

        hwloc_obj_t obj = hwloc_get_obj_by_depth(topology, depth, 0);
        do
        {
            pinfop->CPUids[pu_index] = obj->os_index;
            ++pu_index;
            obj = obj->next_cousin;
        } while (obj != NULL);

        hwloc_topology_destroy(topology);
    }

#else

    FILE *fpout;
    char  outstring[16];
    char  buf[100];

    //unsigned int tmp_index = 0;

    fpout = popen("getconf _NPROCESSORS_ONLN", "r");
    if (fpout == NULL)
    {
        printf("WARNING: cannot run command \"tmuxsessionname\"\n");
    }
    else
    {
        if (fgets(outstring, 16, fpout) == NULL)
        {
            printf("WARNING: fgets error\n");
        }
        pclose(fpout);
    }
    pinfop->NBcpus = atoi(outstring);

    fpout =
        popen("cat /proc/cpuinfo |grep \"physical id\" | awk '{ print $NF }'",
              "r");
    pu_index            = 0;
    pinfop->NBcpusocket = 1;
    while ((fgets(buf, sizeof(buf), fpout) != NULL) &&
           (pu_index < pinfop->NBcpus))
    {
        pinfop->CPUids[pu_index]  = pu_index;
        pinfop->CPUphys[pu_index] = atoi(buf);

        //printf("cpu %2d belongs to Physical CPU %d\n", pu_index, pinfop->CPUphys[pu_index] );
        if (pinfop->CPUphys[pu_index] + 1 > pinfop->NBcpusocket)
        {
            pinfop->NBcpusocket = pinfop->CPUphys[pu_index] + 1;
        }

        pu_index++;
    }

#endif

    return (pinfop->NBcpus);
}
