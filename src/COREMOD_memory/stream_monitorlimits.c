/** @file stream_monitorlimits.c
 *
 * Monitors stream to fit within limits.
 *
 * Can take actions if limits are exceeded.
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"
#include "image_ID.h"
#include "stream_sem.h"
#include "create_image.h"
#include "delete_image.h"

#include "COREMOD_tools/COREMOD_tools.h"







// ==========================================
// Forward declaration(s)
// ==========================================



errno_t stream_monitorlimits_FPCONF();

errno_t stream_monitorlimits_RUN();


errno_t stream_monitorlimits(
    const char *instreamname
);




// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t COREMOD_MEMORY_stream_monitorlimits__cli()
{
    // Try FPS implementation

    // Set data.fpsname, providing default value as first arg, and set data.FPS_CMDCODE value.
    // Default FPS name will be used if CLI process has NOT been named.
    // See code in function_parameter.c for detailed rules.
    function_parameter_getFPSargs_from_CLIfunc("streammlim");

    if(data.FPS_CMDCODE != 0) {	// use FPS implementation
        // set pointers to CONF and RUN functions
        data.FPS_CONFfunc = stream_monitorlimits_FPCONF;
        data.FPS_RUNfunc  = stream_monitorlimits_RUN;
        function_parameter_execFPScmd();
        return RETURN_SUCCESS;
    }


    // call non FPS implementation - all parameters specified at function launch
    if(
        CLI_checkarg(1, CLIARG_IMG)
        == 0) {
        stream_monitorlimits(
            data.cmdargtoken[1].val.string
        );

        return RETURN_SUCCESS;
    } else {
        return CLICMD_INVALID_ARG;
    }
}



// ==========================================
// Register CLI command(s)
// ==========================================

errno_t stream_monitorlimits_addCLIcmd()
{

    RegisterCLIcommand(
        "streammlim",
        __FILE__,
        COREMOD_MEMORY_stream_monitorlimits__cli,
        "monitor stream values for safety",
        "FPS function",
        "streammlim",
        "COREMOD_MEMORY_stream_monitorlimits_RUN");

    return RETURN_SUCCESS;
}








/**
 * @brief Manages configuration parameters for stream_monitorlimits
 *
 * ## Purpose
 *
 * Initializes configuration parameters structure\n
 *
 * ## Arguments
 *
 * @param[in]
 * char*		fpsname
 * 				name of function parameter structure
 *
 * @param[in]
 * uint32_t		CMDmode
 * 				Command mode
 *
 *
 */
errno_t stream_monitorlimits_FPCONF()
{
    FPS_SETUP_INIT(data.FPS_name, data.FPS_CMDCODE);
    uint64_t FPFLAG;

    FPFLAG = FPFLAG_DEFAULT_INPUT;
    FPFLAG &= ~FPFLAG_WRITERUN;

    FPS_ADDPARAM_STREAM_IN(streaminname,   ".in_sname",  "input stream", NULL);

    long dtus_default[4] = { 50, 1, 1000000000, 50 };
    long fp_dtus    = function_parameter_add_entry(&fps, ".dtus",
                      "Loop period [us]", FPTYPE_INT64, FPFLAG, &dtus_default);
    (void) fp_dtus; // suppresses unused parameter compiler warning



    // Limits

    FPFLAG = FPFLAG_DEFAULT_INPUT;
    FPFLAG |= FPFLAG_WRITERUN;

    long fpi_minON = function_parameter_add_entry(&fps, ".minON", "min toggle",
                     FPTYPE_ONOFF, FPFLAG, NULL);
    (void) fpi_minON;

    long fpi_minVal = function_parameter_add_entry(&fps, ".minVal", "min value",
                      FPTYPE_FLOAT32, FPFLAG, NULL);
    (void) fpi_minVal;




    long fpi_maxON = function_parameter_add_entry(&fps, ".maxON", "max toggle",
                     FPTYPE_ONOFF, FPFLAG_DEFAULT_INPUT, NULL);
    (void) fpi_maxON;

    long fpi_maxVal = function_parameter_add_entry(&fps, ".minVal", "min value",
                      FPTYPE_FLOAT32, FPFLAG, NULL);
    (void) fpi_maxVal;




    // start function parameter conf loop, defined in function_parameter.h
    FPS_CONFLOOP_START


    // stop function parameter conf loop, defined in function_parameter.h
    FPS_CONFLOOP_END

    printf("CONF EXIT CONDITION MET\n");


    return RETURN_SUCCESS;
}














/**
 * @brief Delay image stream by time offset
 *
 * IDout_name is a time-delayed copy of IDin_name
 *
 */

errno_t stream_monitorlimits_RUN()
{

    // ===========================
    /// ### CONNECT TO FPS
    // ===========================
    FPS_CONNECT( data.FPS_name, FPSCONNECT_RUN );

    // ===============================
    /// ### GET FUNCTION PARAMETER VALUES
    // ===============================
    // parameters are addressed by their tag name
    // These parameters are read once, before running the loop
    //
    char IDin_name[FUNCTION_PARAMETER_STRMAXLEN];
    strncpy(IDin_name,  functionparameter_GetParamPtr_STRING(&fps, ".in_sname"),
            FUNCTION_PARAMETER_STRMAXLEN-1);

    long dtus    = functionparameter_GetParamValue_INT64(&fps, ".dtus");




    // ===========================
    /// ### processinfo support
    // ===========================
    PROCESSINFO *processinfo;

    processinfo = processinfo_setup(
                      data.FPS_name,	             // re-use fpsname as processinfo name
                      "monitor stream limits",     // description
                      "starting monitor",          // message on startup
                      __FUNCTION__, __FILE__, __LINE__
                  );


    // =============================================
    /// ### OPTIONAL: TESTING CONDITION FOR LOOP ENTRY
    // =============================================
    // Pre-loop testing, anything that would prevent loop from starting should issue message
    int loopOK = 1;



    // Specify input stream trigger
    imageID IDin = image_ID(IDin_name);

    processinfo_waitoninputstream_init(processinfo, IDin,
                                       PROCESSINFO_TRIGGERMODE_DELAY, -1);
    processinfo->triggerdelay.tv_sec = 0;
    processinfo->triggerdelay.tv_nsec = (long) (dtus*1000);
    while(processinfo->triggerdelay.tv_nsec > 1000000000)
    {
        processinfo->triggerdelay.tv_nsec -= 1000000000;
        processinfo->triggerdelay.tv_sec += 1;
    }


    // ===========================
    /// ### START LOOP
    // ===========================

    // Notify processinfo that we are entering loop
    processinfo_loopstart(processinfo);


    while(loopOK == 1)
    {
        loopOK = processinfo_loopstep(processinfo);

        processinfo_waitoninputstream(processinfo);


        processinfo_exec_start(processinfo);

        if(processinfo_compute_status(processinfo) == 1)
        {

        }

        // process signals, increment loop counter
        processinfo_exec_end(processinfo);
    }

    // ==================================
    /// ### ENDING LOOP
    // ==================================
    processinfo_cleanExit(processinfo);
    function_parameter_RUNexit(&fps);


    return RETURN_SUCCESS;
}









errno_t stream_monitorlimits(
    const char *instreamname
)
{
    FUNCTION_PARAMETER_STRUCT fps;

    // create and initialize FPS
    // specify name
    sprintf(data.FPS_name, "%s-%s", "streammlim", instreamname);
    data.FPS_CMDCODE = FPSCMDCODE_FPSINIT;
    stream_monitorlimits_FPCONF();

    // initialize parameters
    function_parameter_struct_connect(data.FPS_name, &fps, FPSCONNECT_SIMPLE);
    functionparameter_SetParamValue_STRING(&fps, "in_sname", instreamname);
    function_parameter_struct_disconnect(&fps);

    // run
    stream_monitorlimits_RUN();

    return RETURN_SUCCESS;
}






