/**
 * @file    fps_add_RTsetting_entries.c
 * @brief   Add parameters to FPS for real-time process settings
 */


#include "CommandLineInterface/CLIcore.h"


#include "fps_GetParamIndex.h"


/** @brief Add parameters to FPS for real-time process settings
 * 
 * Adds standard set of parameters for integration with process info
 * and other milk conventions.
 *
 * This function is typically called within the FPCONF fps creation step.
 *
 */
errno_t fps_add_processinfo_entries(FUNCTION_PARAMETER_STRUCT *fps)
{

    //void *pNull = NULL;
    uint64_t FPFLAG;

    FPFLAG = FPFLAG_DEFAULT_INPUT | FPFLAG_MINLIMIT | FPFLAG_MAXLIMIT;
    FPFLAG &= ~FPFLAG_WRITERUN;

    // value = -1 indicates no RT priority
    long RTprio_default[4] = { -1, -1, 49, 20 };
    function_parameter_add_entry(fps, ".conf.procinfo.RTprio", "RTprio",
                                 FPTYPE_INT64, FPFLAG, &RTprio_default);

    // value = 0 indicates process will adjust to available nb cores
    long maxNBthread_default[4] = { 0, 0, 50, 1 };
    function_parameter_add_entry(fps, ".conf.procinfo.NBthread", "max NB threads",
                                 FPTYPE_INT64, FPFLAG, &maxNBthread_default);

	// taskset
    function_parameter_add_entry(fps, ".conf.taskset", "CPUs mask",
                                 FPTYPE_STRING, FPFLAG, "0-127");	

    // run time string
    function_parameter_add_entry(fps, ".conf.timestring", "runstart time string",
                                 FPTYPE_STRING, FPFLAG, "undef");


	// custom label
    function_parameter_add_entry(fps, ".conf.label", "custom label",
                                 FPTYPE_STRING, FPFLAG, "");	


	// output directory where results are saved
	//
//	char outdir[FPS_DIR_STRLENMAX];
//	snprintf(outdir, FPS_DIR_STRLENMAX, "fps.%s", fps->md->name);
    function_parameter_add_entry(fps, ".conf.datadir", "data directory",
                                 FPTYPE_DIRNAME, FPFLAG, fps->md->datadir);

	// input directory, FPS configuration files ready by FPSsync operation
	//
//	char confdir[FPS_DIR_STRLENMAX];
//	snprintf(confdir, FPS_DIR_STRLENMAX, "fpsconfdir-%s", fps->md->name);
    function_parameter_add_entry(fps, ".conf.confdir", "conf directory",
                                 FPTYPE_DIRNAME, FPFLAG, fps->md->confdir);



	// Where results are archived
	//	
    function_parameter_add_entry(fps, ".conf.archivedir", "archive directory",
                                 FPTYPE_DIRNAME, FPFLAG, NULL);


    return RETURN_SUCCESS;
}



errno_t fps_to_processinfo(FUNCTION_PARAMETER_STRUCT *fps, PROCESSINFO *procinfo)
{


    // set RT_priority if applicable
    long pindex = functionparameter_GetParamIndex(fps, ".conf.procinfo.RTprio");
    if(pindex > -1) {
        long RTprio = functionparameter_GetParamValue_INT64(fps, ".conf.procinfo.RTprio");
        procinfo->RT_priority = RTprio;
    }



    return RETURN_SUCCESS;
}
