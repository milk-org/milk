# Function Parameter Structure (FPS) {#page_FunctionParameterStructure}

@note This file: ./src/CommandLineInterface/doc/FunctionParameterStructure.md

[TOC]




---


**The function  parameter structure (FPS) exposes a function's internal variables for read and/or write. It is stored in shared memory, in /MILK_SHM_DIR/fpsname.fps.shm.**


Steps to run FPS-enabled processes:

	$ vim fpslist.txt                    # Edit file, listing functions and corresponding FPS names that will be used
	$ milk-fpsinit -e cacao -C           # create all (-C) FPS shared memory structure(s) and tmux sessions. Use cacao (-e) to launch commands
	$ milk-fpsCTRL                       # FPS control tool, scan ALL FPSs (-m option force match with fpscmd/fpslist.txt)
---


# 1. Overview and background {#page_FunctionParameterStructure_Overview}

## 1.1. Main elements

FPS-enabled functions have the following elements:

- The shared memory FPS: /tmp/fpsname.fps.shm
- A configuration process that manages the FPS entries
- A run process (the function itself)



## 1.2. FPS name

<fpsname> consists of a root name (string), and a series of optional integers. Note that the number of digits matters and is part of the name:

	<fpsname> = <fpsnameroot>.<opt0>.<opt1>...

Examples:

	myfps                        # simple name, no optional integers
	myfps-000000                 # optional integer 000000
	myfps-000000-white-000002    # 3 optional args

@warning The FPS name does not need to match the process or function name. FPS name is specified in the CLI function as described in @ref page_FunctionParameterStructure_WritingCLIfunc.



## 1.3. FPS-related entities


name                                  | Type           | Description        | Origin
--------------------------------------|----------------|--------------------|---------------------------------
/MILK_SHM_DIR/fpsname.fps.shm                | shared memory  | FP structure       | Created by FPS init function 
fpsname:run                        | tmux session window 2  | FPS control terminal    | Set up by milk-fpsinit
fpsname:conf                        | tmux session window 3  | where CONF runs    | Set up by milk-fpsinit
fpsname:run                         | tmux session   | where RUN runs     | Set up by milk-fpsinit
/MILK_SHM_DIR/fpslog.tstamp.pid.FPSTYPE-fpsname                | ASCII file  | log files       | Created by FPS processes
/MILK_SHM_DIR/fpslog.FPSTYPE-fpsname                | sym link  | log file       | points to latest FPS log file
./fpsconf/fpsname/...               | ASCII file     | parameter value    | OPTIONAL




---




# 2. FPS user interface {#page_FunctionParameterStructure_UserInterface}


Main steps to enable FPS-enabled function for fpsCTRL:

- **Define which FPSs to enable**: Add entry in ./fpslist.txt
- **Set up FPSs entities**: Run milk-fpsinit 
- **Control and monitor FPSs**: start milk-fpsCTRL

These steps should ideally performed by a setup script.

## 2.1. Define which FPSs to enable with a `fpslist.txt` file {#page_FunctionParameterStructure_WritingFPSCMDscripts}


The user-provided `fpslist.txt` file lists the functions and corresponding FPS names that will be in use:

~~~
# List of FPS-enabled function
# Column 1: root name used to name FPS
# Column 2: CLI command
# Column(s) 3: optional arguments, string

fpsrootname0	CLIcommand0
fpsrootname1	CLIcommand1		optarg0  optarg1
~~~

## 2.2. Set up FPSs entities

FPS are built by

	$ milk-fpsinit
	
	

## 2.3. milk-fpsCTRL tool {#page_FunctionParameterStructure_fpsCTRL}

The FPS control tool is started from the command line :

	$ milk-fpsCTRL

A fifo is set up by milk-fpsCTRL to receive commands to start/stop the conf and run processes. Commands can also be issued directly from the milk-fpsCTRL GUI.


### 2.3.1. Starting a conf process

To start a run process, the user issues a command to the fifo :

	echo "confstart fpsname-01" >> ${MILK_SHM_DIR}/${LOOPNAME}_fpsCTRL.fifo

The steps triggered by this command are :

- Command is processed by fifo command interpreter functionparameter_FPSprocess_cmdline()
- The fifo command interpreter resolves the fps entry, and runs functionparameter_CONFstart(fps, fpsindex)
- Pre-configured function fpsconfstart is executed within the tmux session :run window :
	- fps CLI command is launched with argument "_CONFSTART_" :
		- function_parameter_getFPSname_from_CLIfunc() called, sets data.FPS_CMDCODE = FPSCMDCODE_CONFSTART


### 2.2.2. Stopping a conf process

To stop a conf process, the user issues a command to the fifo :

	echo "confstop fpsname-01" >> ${MILK_SHM_DIR}/${LOOPNAME}_fpsCTRL.fifo

The steps triggered by this command are :

- Command is processed by fifo command interpreter functionparameter_FPSprocess_cmdline()
- The fifo command interpreter resolves the fps entry, and runs functionparameter_CONFstop(fps, fpsindex)
	- FUNCTION_PARAMETER_STRUCT_SIGNAL_CONFRUN flag is set to 0
	

### 2.2.3. Starting a run process

To start a run process, the user issues a command to the fifo :

	echo "confstart fpsname-01" >> ${MILK_SHM_DIR}/${LOOPNAME}_fpsCTRL.fifo

The steps triggered by this command are :

- Command is processed by fifo command interpreter functionparameter_FPSprocess_cmdline()
- The fifo command interpreter resolves the fps entry, and runs functionparameter_RUNstart(fps, fpsindex)
- Pre-configured function fpsrunstart is executed within the tmux session :run window :
	- fps CLI command is launched with argument "_RUNSTART_" :
		- function_parameter_getFPSname_from_CLIfunc() called, sets data.FPS_CMDCODE = FPSCMDCODE_RUNSTART


### 2.3.4. Stopping a run process

To stop a run process, the user issues a command to the fifo :

	echo "runstop fpsname-01" >> ${MILK_SHM_DIR}/${LOOPNAME}_fpsCTRL.fifo

The steps triggered by this command are :

- Command is processed by fifo command interpreter functionparameter_FPSprocess_cmdline()
- The fifo command interpreter resolves the fps entry, and runs functionparameter_RUNstop(fps, fpsindex)
- Pre-configured function fpsrunstop is executed within the tmux session :ctrl window :
	- fps CLI command is launched with argument "_RUNSTOP_" :
		- function_parameter_getFPSname_from_CLIfunc() called, sets data.FPS_CMDCODE = FPSCMDCODE_RUNSTOP
		- function_parameter_execFPScmd() called, calls conf process (function data.FPS_CONFfunc)
		- function_parameter_FPCONFsetup, called within conf process, configures variables for run stop
	- CTRL-c sent to run tmux session to ensure exit
---


# 3. Writing the CLI function (in <module>.c file)  {#page_FunctionParameterStructure_WritingCLIfunc}


A single CLI function, named <functionname>_cli, will take the following arguments:

- arg1: A command code
- arg2+: Optional arguments 

The command code is a string, and will determine the action to be executed:

- `_FPSINIT_`  : Initialize FPS for the function, look for parameter values on filesystem
- `_CONFSTART_` : Start the FPS configuration process
- `_CONFSTOP_`  : Stop the FPS configuration process
- `_RUNSTART_`  : Start the run process
- `_RUNSTOP_`   : Stop the run process
 
 
 
@note Why Optional arguments to CLI function ?
@note Multiple instances of a C function may need to be running, each with its own FPS. Optional arguments provides a mechanism to differentiate the FPSs. They are appended to the FPS name following a dash. Optional arguments can be a number (usually integer) or a string.
 

Example source code below.

~~~~{.c}

errno_t ExampleFunction__cli()
{
    // Try FPS implementation

    // Set data.fpsname, providing default value as first arg, and set data.FPS_CMDCODE value.
    // Default FPS name will be used if CLI process has NOT been named.
    // See code in function_parameter.c for detailed rules.

    function_parameter_getFPSargs_from_CLIfunc("measlinRM");

	if(data.FPS_CMDCODE != 0) {	// use FPS implementation
		// set pointers to CONF and RUN functions
		data.FPS_CONFfunc = ExampleFunction_FPCONF;
		data.FPS_RUNfunc  = ExampleFunction_RUN;
		function_parameter_execFPScmd();
		return RETURN_SUCCESS;
	}


    // call non FPS implementation - all parameters specified at function launch
    if(
        CLI_checkarg(1, CLIARG_FLOAT) +
        CLI_checkarg(2, CLIARG_LONG)
        == 0) {
        ExampleFunction(
            data.cmdargtoken[1].val.numf,
            data.cmdargtoken[2].val.numl
        );

        return RETURN_SUCCESS;
    } else {
        return CLICMD_INVALID_ARG;
    }
}
~~~~



---


# 4. Writing function prototypes (in <module>.h) {#page_FunctionParameterStructure_WritingPrototypes}



~~~~{.c}
errno_t ExampleFunction_FPCONF();
errno_t ExampleFunction_RUN();
errno_t ExampleFunction(long arg0num, long arg1num, long arg2num, long arg3num);
~~~~ 



---

# 5. Writing CONF function (in source .c file) {#page_FunctionParameterStructure_WritingCONFfunc}

Check function_parameters.h for full list of flags.

~~~~{.c} 


//
// manages configuration parameters
// initializes configuration parameters structure
//
errno_t ExampleFunction_FPCONF()
{
    // ===========================
    // SETUP FPS
    // ===========================
    FPS_SETUP_INIT(data.FPS_name, data.FPS_CMDCODE); // macro in function_parameter.h
    fps_add_processinfo_entries(&fps); // include real-time settings

    // ==============================================
    // ========= ALLOCATE FPS ENTRIES ===============
    // ==============================================

    uint64_t FPFLAG;

    // Entries are added one by one with function_parameter_add_entry()
    // For each entry, we record the function parameter index (fpi_) returned by the function so that parameters can conveniently be accesses in the "LOGIC" section
    // Arguments:
    //   - pointer to fps
    //   - parameter path for GUI
    //   - description
    //   - type
    //   - flags
    //   - initialization pointer. If pNull, then the variable is not initialized
    //
    // Check CommandLineInterface/function_parameters.h for full list of flags.

    long fpi_param01 = function_parameter_add_entry(&fps, ".param01", "First parameter",
                       FPTYPE_INT64, FPFLAG_DEFAULT_INPUT, NULL;

    // This parameter will be intitialized to a value of 5, min-max range from 0 to 10, and current value 5
    int64_t param02default[4] = { 5, 0, 10, 5 };
    FPFLAG = FPFLAG_DEFAULT_INPUT | FPFLAG_MINLIMIT | FPFLAG_MAXLIMIT;  // required to enforce the min and max limits
    FPFLAG &= ~FPFLAG_WRITECONF;  // Don't allow parameter to be written during configuration
    FPFLAG &= ~FPFLAG_WRITERUN;   // Don't allow parameter to be written during run
    long fpi_param02 = function_parameter_add_entry(&fps, ".param02", "Second parameter",
                       FPTYPE_INT64, FPFLAG, &param02default);

    // if parameter type = FPTYPE_FLOAT32, make sure default is declared as float[4]
    // if parameter type = FPTYPE_FLOAT64, make sure default is declared as double[4]
    float gaindefault[4] = { 0.01, 0.0, 1.0, 0.01 };
    FPFLAG = FPFLAG_DEFAULT_INPUT | FPFLAG_MINLIMIT | FPFLAG_MAXLIMIT;  // required to enforce the min and max limits
    long fpi_gain = function_parameter_add_entry(&fps, ".gain", "gain value",
                    FPTYPE_FLOAT32, FPFLAG, &gaindefault);


    // This parameter is a ON / OFF toggle
    long fpi_gainset = function_parameter_add_entry(&fps, ".option.gainwrite", "gain can be changed",
                       FPTYPE_ONOFF, FPFLAG_DEFAULT_INPUT, NULL);


    // stream that needs to be loaded on startup
    FPFLAG = FPFLAG_DEFAULT_INPUT_STREAM;
    long fpi_streamname_wfs       = function_parameter_add_entry(&fps, ".sn_wfs",  "WFS stream name",
                                    FPTYPE_STREAMNAME, FPFLAG, NULL);


    // Output file name
    long fpi_filename_out1          = function_parameter_add_entry(&fps, ".out.fname_out1", "output file 1",
                                      FPTYPE_FILENAME, FPFLAG_DEFAULT_OUTPUT, NULL);




    // Macros examples
    // see function_parameters.h

    FPS_ADDPARAM_STREAM_IN  (stream_inname,        ".in_name",     "input stream", NULL);
    FPS_ADDPARAM_STREAM_OUT (stream_outname,       ".out_name",    "output stream");

    long timeavemode_default[4] = { 0, 0, 3, 0 };
    FPS_ADDPARAM_INT64_IN  (
        option_timeavemode,
        ".option.timeavemode",
        "Enable time window averaging (>0)",
        &timeavemode_default);

    double avedt_default[4] = { 0.001, 0.0001, 1.0, 0.001};
    FPS_ADDPARAM_FLT64_IN  (
        option_avedt,
        ".option.avedt",
        "Averaging time window width",
        &avedt_default);

    // status
    FPS_ADDPARAM_INT64_OUT (zsize,        ".status.zsize",     "cube size");
    FPS_ADDPARAM_INT64_OUT (framelog,     ".status.framelag",  "lag in frame unit");
    FPS_ADDPARAM_INT64_OUT (kkin,         ".status.kkin",      "input cube slice index");
    FPS_ADDPARAM_INT64_OUT (kkout,        ".status.kkout",     "output cube slice index");




    // ==============================================
    // ======== START FPS CONF LOOP =================
    // ==============================================
    FPS_CONFLOOP_START  // macro in function_parameter.h

    // here goes the logic
    if ( fps.parray[fpi_gainset].fpflag & FPFLAG_ONOFF )  // ON state
    {
        fps.parray[fpi_gain].fpflag |= FPFLAG_WRITERUN;
        fps.parray[fpi_gain].fpflag |= FPFLAG_USED;
        fps.parray[fpi_gain].fpflag |= FPFLAG_VISIBLE;
    }
    else // OFF state
    {

        fps.parray[fpi_gain].fpflag &= ~FPFLAG_WRITERUN;
        fps.parray[fpi_gain].fpflag &= ~FPFLAG_USED;
        fps.parray[fpi_gain].fpflag &= ~FPFLAG_VISIBLE;
    }



    // ==============================================
    // ======== STOP FPS CONF LOOP ==================
    // ==============================================
    FPS_CONFLOOP_END  // macro in function_parameter.h


    return RETURN_SUCCESS;
}




~~~~





---



# 6. Writing RUN function (in source .c file) {#page_FunctionParameterStructure_WritingRUNfunc}


The RUN function will connect to the FPS and execute the run loop. 

## 6.1. A simple _RUN example {#page_FunctionParameterStructure_WritingRUNfunc_simple}


~~~~{.c}
//
// run loop process
//
errno_t ExampleFunction_RUN()
{
	// ===========================
	// CONNECT TO FPS
	// ===========================
	FPS_CONNECT(data.FPS_name, FPSCONNECT_RUN );


	// ===============================
	// GET FUNCTION PARAMETER VALUES
	// ===============================

	// parameters are addressed by their tag name

	// These parameters are read once, before running the loop
	//
	int param01 = functionparameter_GetParamValue_INT64(&fps, ".param01");
	int param02 = functionparameter_GetParamValue_INT64(&fps, ".param02");


    // This parameter is a ON / OFF toggle
	int gainwrite = functionparameter_GetParamValue_ONOFF(&fps, ".option.gainwrite");

	// This parameter value will be tracked during loop run, so we create a pointer for it
	// The corresponding function is functionparameter_GetParamPtr_<TYPE>
	//
	float *gain = functionparameter_GetParamPtr_FLOAT32(&fps, ".status.loopcnt");

	char imsname[FUNCTION_PARAMETER_STRMAXLEN];
	strncpy(imsname, functionparameter_GetParamPtr_STRING(&fps, ".option.imname"), FUNCTION_PARAMETER_STRMAXLEN);

	// connect to WFS image
    	long IDim = read_sharedmem_image(imsname);

	// ===============================
	// RUN LOOP
	// ===============================

	int loopOK = 1;
	while( loopOK == 1 )
	{
		// here we compute what we need...
		//

		// Note that some mechanism is required to set loopOK to 0 when MyFunction_Stop() is called
		// This can use a separate shared memory path
	}

	function_parameter_RUNexit( &fps );
	return RETURN_SUCCESS;
}
~~~~


## 6.2. Non-FPS fallback function

~~~{.c}
errno_t ExampleFunction(
    long arg0num,
    long arg1num,
    long arg2num,
    long arg3num
) {
    long pindex = (long) getpid();  // index used to differentiate multiple calls to function
    // if we don't have anything more informative, we use PID

    FUNCTION_PARAMETER_STRUCT fps;

    // create FPS
    sprintf(data.FPS_name, "exfunc-%06ld", pindex);
    data.FPS_CMDMODE = FPSCMDCODE_FPSINIT;
    ExampleFunction_FPCONF();

    function_parameter_struct_connect(data.FPS_name, &fps, FPSCONNECT_SIMPLE);

    functionparameter_SetParamValue_INT64(&fps, ".arg0", arg0);
    functionparameter_SetParamValue_INT64(&fps, ".arg1", arg1);
    functionparameter_SetParamValue_INT64(&fps, ".arg2", arg2);
    functionparameter_SetParamValue_INT64(&fps, ".arg3", arg3);

    function_parameter_struct_disconnect(&fps);

    ExampleFunction_RUN();

    return RETURN_SUCCESS;
}



~~~




## 6.3. RUN function with FPS and processinfo {#page_FunctionParameterStructure_WritingRUNfunc_processinfo}


In this example, the loop process supports both FPS and processinfo.
This is the preferred way to code a loop process.


### 6.3.1. Simple example making extensive use of macros (concept - to be implemented)

~~~~{.c}
errno_t MyFunction_RUN()
{
    // ========= INITIALIZATION
    FPSPROCINFOLOOP_INIT("computes something", "add image1 to image2");

    // ========= GET VALUES FROM FPS
    int param01 = functionparameter_GetParamValue_INT64(&fps, ".param01");

    // ========= Specify input stream trigger by name
    PROCESSINFO_SET_SEMSTREAMWAIT("inputstreamname");

    // ========= Identifiers for output streams
    imageID IDout = image_ID("output");




    PROCINFOLOOP_START;


    //
    // computation ....
    //

    data.image[IDout].md[0].write = 1;
    // write into output image

    // Post semaphore(s) and counter(s), notify system that outputs have been written
    processinfo_update_output_stream(processinfo, IDout);



    PROCINFOLOOP_END;

}
~~~~

### 6.3.2. Verbose example with customization

The example also shows using FPS to set the process realtime priority.

~~~~{.c}

/** @brief Loop process code example
 *
 * ## Purpose
 *
 * This example demonstrates use of processinfo and fps structures.\n
 *
 *
 * All function parameters are held inside the function parameter structure (FPS).\n
 * 
 *
 * ## Details
 *
 */

errno_t MyFunction_RUN()
{
	// ==================================================
	// ### Connect to FUNCTION PARAMETER STRUCTURE (FPS)
	// ==================================================
	FPS_CONNECT( data.FPS_name, FPSCONNECT_RUN );

    // Write time string (optional, requires .out.timestring entry)
    char timestring[100];
    mkUTtimestring_millisec(timestring);
    functionparameter_SetParamValue_STRING(
        &fps,
        ".out.timestring",
        timestring);

	// ===================================
	// ### GET VALUES FROM FPS
	// ===================================
	// parameters are addressed by their tag name

	// These parameters are read once, before running the loop
	//
	int param01 = functionparameter_GetParamValue_INT64(&fps, ".param01");
	int param02 = functionparameter_GetParamValue_INT64(&fps, ".param02");

	char nameim[FUNCTION_PARAMETER_STRMAXLEN+1];
	strncpy(nameim, functionparameter_GetParamPtr_STRING(&fps, ".option.nameim"), FUNCTION_PARAMETER_STRMAXLEN);


	// This parameter value will be tracked during loop run, so we create a pointer for it
	// The corresponding function is functionparameter_GetParamPtr_<TYPE>
	//
	float *gain = functionparameter_GetParamPtr_FLOAT32(&fps, ".ctrl.gain");




	// ===========================
	// ### processinfo support 
	// ===========================

    PROCESSINFO *processinfo;

    processinfo = processinfo_setup(
        data.FPS_name,	             // re-use fpsname as processinfo name
        "computes something",        // description
        "add image1 to image2",      // message on startup
        __FUNCTION__, __FILE__, __LINE__
        );

	// OPTIONAL SETTINGS

    // Measure timing
    processinfo->MeasureTiming = 1;

    // RT_priority, 0-99. Larger number = higher priority. If <0, ignore
    processinfo->RT_priority = 20;

    // max number of iterations. -1 if infinite
    processinfo->loopcntMax = 1000;


    // apply relevant FPS entries to PROCINFO
    // see macro code for details
    fps_to_processinfo(&fps, processinfo);


    int loopOK = 1;


    // =============================================
    // OPTIONAL: TESTING CONDITION FOR LOOP ENTRY
    // =============================================
    // Pre-loop testing, anything that would prevent loop from starting should issue message
    int loopOK = 1;
    if(.... error condition ....)
    {
        // exit function with ERROR status
        processinfo_error(processinfo, "ERROR: no WFS reference");
        return RETURN_FAILURE;
    }


    // Specify input stream trigger
    IDin = image_ID("inputstream");
    processinfo_waitoninputstream_init(processinfo, IDin,
		PROCESSINFO_TRIGGERMODE_SEMAPHORE, -1);

    // Identifiers for output streams
    imageID IDout0 = image_ID("output0");
    imageID IDout1 = image_ID("output1");

	// ===========================
	// ### Start loop
	// ===========================

    // Notify processinfo that we are entering loop
    processinfo_loopstart(processinfo);

    while(loopOK==1)
    {
	    loopOK = processinfo_loopstep(processinfo);
        processinfo_waitoninputstream(processinfo);

        processinfo_exec_start(processinfo);
        if(processinfo_compute_status(processinfo)==1)
        {
            //
		    // computation ....
		    //

            data.image[IDout0].md[0].write = 1;
            data.image[IDout1].md[0].write = 1;
            // write into output images

            // Post semaphore(s) and counter(s), notify system that outputs have been written
            processinfo_update_output_stream(processinfo, IDout0);
            processinfo_update_output_stream(processinfo, IDout1);
        }

        // process signals, increment loop counter
        processinfo_exec_end(processinfo);

        // OPTIONAL: MESSAGE WHILE LOOP RUNNING
        processinfo_WriteMessage(processinfo, "loop running fine");
	}


	// ==================================
	// ### ENDING LOOP
	// ==================================

    processinfo_cleanExit(processinfo);


	// ==================================
	// ### SAVING RESULTS (OPTIONAL)
	// ==================================

    // Conventions
    //
    // Everything saved in directory fps.md->outdir, which by default is
    // fp.md->fpsdirectory/out-fpsname, but can also be set by user parameter conf.outdir.
    // conf.outdir is relative to fps.md->workdir
    //

    // save FPS content
    //
    functionparameter_SaveFPS2disk(&fps);

    EXECUTE_SYSTEM_COMMAND("cd %s", fps.md->outdir);


    // Add files to loglist for archieval into file loglist.dat
    // Example loglist.dat entry / line :
    // YYYYMMDDTHHMMSSssssss rawfilename logname extension
    // Will log the file as ${MILKDATALOGDIR}/logname.YYYYMMDDTHHMMSSssssss.extension

    EXECUTE_SYSTEM_COMMAND("touch loglist.dat");

    save_fits(output0, "output0.fits");
    EXECUTE_SYSTEM_COMMAND("echo \"%s output0.fits output0 fits\" >> loglist.dat", fps.md->runpidstarttime);

    save_fits(output1, "output1.fits");
    EXECUTE_SYSTEM_COMMAND("echo \"%s output1.fits output1 fits\" >> loglist.dat", fps.md->runpidstarttime);

    EXECUTE_SYSTEM_COMMAND("echo \"%s somedata.dat somedata dat\" >> loglist.dat", fps.md->runpidstarttime);


    // Create archiving script that will copy files to directory datadir (usually a sym link)
    functionparameter_write_archivescript(&fps);

    EXECUTE_SYSTEM_COMMAND("cd %s", fps.md->workdir);


    // optional: run exec scripts that take FPSname as argument




	function_parameter_RUNexit( &fps  );

	return RETURN_SUCCESS;
}
~~~~



----





