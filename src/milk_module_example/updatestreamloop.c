/**
 * @file    updatestreamloop.c
 * @brief   simple procinfo+fps example
 *
 * A pre-existing stream is updated at regular intervals, using the
 * DELAY trigger mode.
 *
 * This is the first example in a series of procinfo/fps examples.
 * Subsequent example are more complex and demonstrate more powerful
 * features.
 *
 * The function parameter structure (FPS) is a structure holding function
 * parameters, and compatible with wide range of convenient tools to
 * for users to interact with functions.
 *
 * To enable FPS support with command line interface (CLI) wrapper, four
 * functions are required in a C program:
 * - FPSCONFfunction() configures parameters and setup the FPS
 * - FPSRUNfunction() executes the code
 * - FPSEXECfuncion() a self-contained execution function
 * - a CLI wrapper function
 *
 * There are two ways to run the code inside FPSRUNfunction():
 * - CLI-called  : function called inside CLI by typing command
 * - FPS-managed : function called by FPS configuration
 *
 * When CLI-called, command CLICMD_SHORTNAME is typed/entered in the CLI
 * (possibly preceeded by command namespace), which is processed by
 * FPSCLIfunction() with data.FPS_CMDCODE set to 0. Following argument
 * checks, FPSEXECfunction() is called to create the FPS and run the code.
 * The FPS is created to hold arguments, uniquely named <CLICMD_SHORTNAME>-<PID>
 *
 * When FPS-managed, FPSCONFfunction() manages parameters and execution.
 * The FPS name is then <CLICMD_SHORTNAME>-<ARG0>-<ARG1>...
 *
 * To use the FPS-managed interface, see script milk-fpsinit:
 * > milk-fpsinit -h
 *
 */

#include "CommandLineInterface/CLIcore.h"

// required for create_2Dimage_ID()
#include "COREMOD_memory/COREMOD_memory.h"

#include "COREMOD_tools/COREMOD_tools.h"

// required for timespec_diff()
#include "CommandLineInterface/timeutils.h"


static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG, ".in_name", "input stream", "ims1",
        CLICMDARG_FLAG_DEFAULT, FPTYPE_AUTO, FPFLAG_DEFAULT_INPUT,
        NULL
    },
    {
        CLIARG_LONG, ".delayus", "delay [us]", "2000",
        CLICMDARG_FLAG_DEFAULT, FPTYPE_AUTO, FPFLAG_DEFAULT_INPUT,
        NULL
    },
    {
        CLIARG_FLOAT, ".fpsonly", "test val", "1.2334",
        CLICMDARG_FLAG_NOCLI, FPTYPE_AUTO, FPFLAG_DEFAULT_INPUT,
        NULL
    }
};

static CLICMDDATA CLIcmddata =
{
    "streamupdate",
    "update stream",
    __FILE__, sizeof(farg) / sizeof(CLICMDARGDEF), farg,
    0,
    NULL
};

/** @brief FPCONF function for updatestreamloop
 */
static errno_t FPSCONFfunction()
{
    FPS_SETUP_INIT(data.FPS_name, data.FPS_CMDCODE);
    fps_add_processinfo_entries(&fps); // add processinfo std entries
    CMDargs_to_FPSparams_create(&fps); // add function arguments

    long fps_delayus = functionparameter_GetParamValue_INT64(&fps, ".delayus");

    FPS_CONFLOOP_START
    printf("delayus value inside conf loop: %ld\n", fps_delayus);
    FPS_CONFLOOP_END

    return RETURN_SUCCESS;
}



/** @brief Loop process code example
 *
 * ## Purpose
 *
 * Update stream at regular time interval.\n
 * This example demonstrates combined use of processinfo and fps structures.\n
 *
 * ## Details
 *
 */
static errno_t FPSRUNfunction()
{
    FPS_CONNECT(data.FPS_name, FPSCONNECT_RUN);

    /** ### GET FUNCTION PARAMETER VALUES
     *
     * Parameters are addressed by their tag name\n
     * These parameters are read once, before running the loop.\n
     *
     * FPS_GETPARAM... macros are wrapper to functionparameter_GetParamValue
     * and functionparameter_GetParamPtr functions, all defined in
     * fps_paramvalue.h.
     *
     * Each of the FPS_GETPARAM macro creates a variable with "_" prepended
     * to the first macro argument.
     *
     * Equivalent code without using macros:
     *
     *     char _IDin_name[200];
     *     strncpy(_IDin_name,  functionparameter_GetParamPtr_STRING(&fps, ".in_name"), FUNCTION_PARAMETER_STRMAXLEN);
     *     long _delayus = functionparameter_GetParamValue_INT64(&fps, ".delayus");
     */
    char *fps_IDin_name = functionparameter_GetParamPtr_STRING(&fps, ".in_name");
    long fps_delayus = functionparameter_GetParamValue_INT64(&fps, ".delayus");

    /** ### SET UP PROCESSINFO
     *
     * Equivalent code without using macros:
     *
     *     PROCESSINFO *processinfo;
     *
     *     char pinfodescr[200];
     *     sprintf(pinfodescr, "streamupdate %.10s", _IDin_name);
     *     processinfo = processinfo_setup(
                      data.FPS_name, // re-use fpsname as processinfo name
                      pinfodescr,    // description
                      "startup",     // message on startup
                      __FUNCTION__, __FILE__, __LINE__
                  );

    *     // OPTIONAL SETTINGS
    *     // Measure timing
    *     // processinfo->MeasureTiming = 1;
    *     // RT_priority, 0-99. Larger number = higher priority. If <0, ignore
    *     // processinfo->RT_priority = 20;
    *
    *     fps_to_processinfo(&fps, processinfo);
    *
    *
    *     int processloopOK = 1;
    */
    FPSPROCINFOLOOP_RUNINIT("streamupdate %.10s", fps_IDin_name);

    /** ### OPTIONAL: TESTING CONDITION FOR LOOP ENTRY
     *
     * Pre-loop testing, anything that would prevent loop from starting should issue message\n
     * Set processloopOK to 0 if tests fail.
     */
    imageID IDin = image_ID(fps_IDin_name);

    /** ### Specify input trigger
     *
     * In this example, a simple delay is inserted between trigger\n
     * This is done by calling a macro.\n
     *
     * Equivalent code without macro:
     *
     *     processinfo_waitoninputstream_init(processinfo, -1, PROCESSINFO_TRIGGERMODE_DELAY, -1);
     *     processinfo->triggerdelay.tv_sec = 0;
     *     processinfo->triggerdelay.tv_nsec = (long)(_delayus * 1000);
     *     while(processinfo->triggerdelay.tv_nsec > 1000000000)
     *     {
     *         processinfo->triggerdelay.tv_nsec -= 1000000000;
     *         processinfo->triggerdelay.tv_sec += 1;
     *     }
     */
    PROCINFO_TRIGGER_DELAYUS(fps_delayus);

    /** ### START LOOP
     *
     * Notify processinfo that we are entering loop\n
     * Start loop and handle timing and control hooks\n
     *
     * Equivalent code:
     *
     *     processinfo_loopstart(processinfo);
     *     while(processloopOK == 1) {
     *         processloopOK = processinfo_loopstep(processinfo);
     *         processinfo_waitoninputstream(processinfo);
     *         processinfo_exec_start(processinfo);
     *         if(processinfo_compute_status(processinfo) == 1) {
     */
    PROCINFOLOOP_START

    DEBUG_TRACEPOINT("Starting loop iteration");

    DEBUG_TRACEPOINT("Update stream \"%s\" %ld", fps_IDin_name, IDin);
    processinfo_update_output_stream(processinfo, IDin);
    /** ### END LOOP
     *
     * Equivalent non-macro code:
     *
     *     }
     *     processinfo_exec_end(processinfo);
     *     }
     *     processinfo_cleanExit(processinfo);
     */
    DEBUG_TRACEPOINT("Ending loop iteration");
    PROCINFOLOOP_END

    DEBUG_TRACEPOINT("Ending loop computation");

    function_parameter_RUNexit(&fps);

    return RETURN_SUCCESS;
}

/** @brief Self-contained execution function
 *
 * ## Purpose
 *
 * This function is called to perform all required steps :
 * - set up FPS
 * - call RUN function
 *
 * The CONF function loop will not be running.
 *
 *
 * ## Details
 *
 */
static errno_t FPSEXECfunction()
{
    FUNCTION_PARAMETER_STRUCT fps;
    sprintf(data.FPS_name, "%s-%06ld", CLIcmddata.key, (long)getpid());

    // initialize and create FPS
    data.FPS_CMDCODE = FPSCMDCODE_FPSINIT;
    FPSCONFfunction();

    // connect to newly created FPS
    function_parameter_struct_connect(data.FPS_name, &fps, FPSCONNECT_SIMPLE);

    /** ### WRITE FUNCTION PARAMETERS TO FPS
     */
    CLIargs_to_FPSparams_setval(farg, CLIcmddata.nbarg, &fps);
    function_parameter_struct_disconnect(&fps);

    /** EXECUTE RUN FUNCTION
     */
    FPSRUNfunction();

    return RETURN_SUCCESS;
}

// =====================================================================
// Command line interface wrapper function(s)
// =====================================================================

static errno_t FPSCLIfunction(void)
{
    // Try FPS implementation

    // Set data.fpsname, providing default value as first arg, and set data.FPS_CMDCODE value.
    //    Default FPS name will be used if CLI process has NOT been named.
    //    See code in function_parameter.h for detailed rules.
    function_parameter_getFPSargs_from_CLIfunc(CLIcmddata.key);

    if(data.FPS_CMDCODE != 0)  // use FPS implementation
    {
        // set pointers to CONF and RUN functions
        data.FPS_CONFfunc = FPSCONFfunction;
        data.FPS_RUNfunc = FPSRUNfunction;
        function_parameter_execFPScmd();
        return RETURN_SUCCESS;
    }

    // call self-contained execution function - all parameters specified at function launch
    if(CLI_checkarg_array(farg, CLIcmddata.nbarg) == RETURN_SUCCESS)
    {
        FPSEXECfunction();
        return RETURN_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

// =====================================================================
// Register CLI command(s)
// =====================================================================

errno_t FPSCLIADDCMD_milk_module_example__updatestreamloop()
{
    RegisterCLIcmd(CLIcmddata, FPSCLIfunction);

    return RETURN_SUCCESS;
}
