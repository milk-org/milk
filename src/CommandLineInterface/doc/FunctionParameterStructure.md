# Function Parameter Structure (FPS) {#page_FunctionParameterStructure}

@note This file: ./src/CommandLineInterface/doc/FunctionParameterStructure.md

[TOC]




---


**The function  parameter structure (FPS) exposes a function's internal variables for read and/or write. It is stored in shared memory, in /tmp/<fpsname>.fps.shm.**


Steps to run FPS-enabled processes:

	$ vim fpslist.txt               # Edit file, listing functions and corresponding FPS names that will be used
	$ fpsmkcmd                      # create FPS scripts in `./fpscmd/`
	$ ./fpscmd/fpsinitscript        # create FPS shared memory structure(s)
	$ ./fpscmd/fpsconfstartscript   # start FPS configuration process(es)
	$ fpsCTRL -m _ALL               # FPS control tool, scan ALL FPSs (-m: force match with fpscmd/fpslist.txt)
	Type 'P' to im(P)ort configuration


---


# 1. Overview and background {#page_FunctionParameterStructure_Overview}

## 1.1. Main elements

FPS-enabled functions have the following elements:
- The shared memory FPS: /tmp/<fpsname>.fps.shm
- A configuration process that manages the FPS entries
- A run process (the function itself)



## 1.2. FPS name

<fpsname> consists of a root name (string), and a series of optional integers. Note that the number of digits matters and is part of the name:

	<fpsname> = <fpsnameroot>.<opt0>.<opt1>...

Examples:

	myfps                        # simple name, no optional integers
	myfps-000000                 # optional integer 000000
	myfps-000043-000020-000002   # 3 optional integers
	
@warning The FPS name does not need to match the process or function name. FPS name is specified in the CLI function as described in @ref page_FunctionParameterStructure_WritingCLIfunc.



## 1.3. FPS-related entities

name                                  | Type           | Description        | Origin
--------------------------------------|----------------|--------------------|---------------------------------
/tmp/<fpsname>.fps.shm                | shared memory  | FP structure       | Created by FPS init function 
./fpscmd/<fpsnameroot>-confinit       | script         | Initialize FPS     | Built by fpsmkcmd, can be user-edited
./fpscmd/<fpsnameroot>-confstart      | script         | Start CONF process | Built by fpsmkcmd, can be user-edited
./fpscmd/<fpsnameroot>-runstart       | script         | Start RUN process  | Built by fpsmkcmd, can be user-edited
./fpscmd/<fpsnameroot>-runstop        | script         | Stop RUN process   | Built by fpsmkcmd, can be user-edited
<fpsname>-conf                        | tmux session   | where CONF runs    | Set up by fpsCTRL
<fpsname>-run                         | tmux session   | where RUN runs     | Set up by fpsCTRL
./fpsconf/<fpsname>/...               | ASCII file     | parameter value    | <TBD>


---



	



# 2. FPS user interface {#page_FunctionParameterStructure_UserInterface}


Main steps to enable FPS-enabled function for fpsCTRL:

- Add entry in ./fpslist.txt
- Run milk-fpsmkcmd 
- Run ./fpscmd/<fpsname>-confinit

These steps should ideally performed by a setup script.

## 2.1. Building command scripts from a `fpslist.txt` file {#page_FunctionParameterStructure_WritingFPSCMDscripts}


The user-provided `fpslist.txt` file lists the functions and corresponding FPS names that will be in use:

~~~
# List of FPS-enabled function
# Column 1: root name used to name FPS
# Column 2: CLI command
# Column(s) 3+: optional arguments, integers, 6 digits

fpsrootname0	CLIcommand0
fpsrootname1	CLIcommand1		optarg00	optarg01
~~~

FPS command scripts are built by

	$ fpsmkcmd
	
The command will create the FPS command scripts in directory `./fpscmd/`, which are then called by the @ref page_FunctionParameterStructure_fpsCTRL to control the CONF and RUN processes.

	

## 2.2. fpsCTRL tool {#page_FunctionParameterStructure_fpsCTRL}

The FPS control tool is started from the command line :

	$ fpsCTRL



---


# 3. Writing the CLI function (in <module>.c file)  {#page_FunctionParameterStructure_WritingCLIfunc}


A single CLI function, named <functionname>_cli, will take the following arguments:
- arg1: A command code
- arg2: Optional argument 

The command code is a string, and will determine the action to be executed:
- `_FPSINIT_`  : Initialize FPS for the function
- `_CONFSTART_` : Start the FPS configuration process
- `_CONFSTOP_`  : Stop the FPS configuration process
- `_RUNSTART_`  : Start the run process
- `_RUNSTOP_`   : Stop the run process
 
 
 
@note Why Optional argument to CLI function ?
@note Multiple instances of a C function may need to be running, each with its own FPS. An optional argument provides a mechanism to differentiate the FPSs. It is appended to the FPS name following a dash. The optional argument can be a number (usually integer) or a string.
 

Example source code below, assuming one optional long type argument. 

~~~~{.c}

errno_t MyFunction_cli() {
    int stringmaxlen = 200;
    char fpsname[stringmaxlen];

    // First, we try to execute function through FPS interface
    if(CLI_checkarg(1, 5) == 0) { // check that first arg is string
        unsigned int OptionalArg00 = data.cmdargtoken[2].val.numl;

        // Set FPS interface name
        // By convention, if there are optional arguments, they should be appended to the fps name
        //
        if(data.processnameflag == 0) {
            // the process has not been named with -n CLI option
            // name fps to something different than the process name
            // by appending user-provided string if available
             if(strlen(data.cmdargtoken[2].val.string)>0)
                snprintf(fpsname, stringmaxlen, "myfunc-%s", data.cmdargtoken[2].val.string);
            else
                sprintf(fpsname, "myfunc");
        } else { 
            // Automatically set fps name to be process name up to first instance of character '.'
            // This is the preferred option
            strcpy(fpsname, data.processname0);
        }

        if(strcmp(data.cmdargtoken[1].val.string, "_FPSFINIT_") == 0) {  // Initialize FPS and conf process
            printf("Function parameters configure\n");
            MyFunction_FPCONF(fpsname, CMDCODE_FPSINIT);
            return RETURN_SUCCESS;
        }

        if(strcmp(data.cmdargtoken[1].val.string, "_CONFSTART_") == 0) {  // Start conf process
            printf("Function parameters configure\n");
            MyFunction_FPCONF(fpsname, CMDCODE_CONFSTART);
            return RETURN_SUCCESS;
        }

        if(strcmp(data.cmdargtoken[1].val.string, "_CONFSTOP_") == 0) { // Stop conf process
            printf("Function parameters configure\n");
            MyFunction_FPCONF(fpsname, CMDCODE_CONFSTOP);
            return RETURN_SUCCESS;
        }

        if(strcmp(data.cmdargtoken[1].val.string, "_RUNSTART_") == 0) { // Run process
            printf("Run function\n");
            MyFunction_RUN(fpsname);
            return RETURN_SUCCESS;
        }

        if(strcmp(data.cmdargtoken[1].val.string, "_RUNSTOP_") == 0) { // Stop process
            printf("Run function\n");
            MyFunction_STOP();
            return RETURN_SUCCESS;
        }
    }

    // non FPS implementation - all parameters specified at function launch
    if(CLI_checkarg(1, 2) + CLI_checkarg(2, 2) + CLI_checkarg(3, 2) + CLI_checkarg(4, 2) == 0) {
        MyFunction(data.cmdargtoken[1].val.numl, data.cmdargtoken[2].val.numl, data.cmdargtoken[3].val.numl, ata.cmdargtoken[4].val.numl);
            return RETURN_SUCCESS;
        } else {
            return RETURN_FAILURE;
        }
}
~~~~



---


# 4. Writing function prototypes (in <module>.h) {#page_FunctionParameterStructure_WritingPrototypes}



~~~~{.c}
errno_t MyFunction_FPCONF(char *fpsname, uint32_t CMDmode, long optarg00);
errno_t MyFunction_RUN(char *fpsname);
errno_t MyFunction(long arg0num, long arg1num, long arg2num, long arg3num);
~~~~ 



---

# 5. Writing CONF function (in source .c file) {#page_FunctionParameterStructure_WritingCONFfunc}

Check function_parameters.h for full list of flags.

~~~~{.c} 


//
// manages configuration parameters
// initializes configuration parameters structure
//
errno_t MyFunction_FPCONF(
    char *fpsname,
    uint32_t CMDmode,
    long optarg00
)
{
    // ===========================
    // SETUP FPS
    // ===========================
    FPS_SETUP_INIT(fpsname, CMDmode); // macro in function_parameter.h


    // ==============================================
    // ========= ALLOCATE FPS ENTRIES ===============
    // ==============================================

    void *pNull = NULL;
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
                       FPTYPE_INT64, FPFLAG_DEFAULT_INPUT, pNull);

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
                       FPTYPE_ONOFF, FPFLAG_DEFAULT_INPUT, pNull);


    // stream that needs to be loaded on startup
    FPFLAG = FPFLAG_DEFAULT_INPUT_STREAM;
    long fpi_streamname_wfs       = function_parameter_add_entry(&fps, ".sn_wfs",  "WFS stream name",
                                    FPTYPE_STREAMNAME, FPFLAG, pNull);


    // Output file name
    long fpi_filename_out1          = function_parameter_add_entry(&fps, ".out.fname_out1", "output file 1",
                                      FPTYPE_FILENAME, FPFLAG_DEFAULT_OUTPUT, pNull);




    // Macros examples
    // see function_parameters.h

    FPS_ADDPARAM_STREAM_IN  (stream_inname,        ".in_name",     "input stream");
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
errno_t MyFunction_RUN(
    char *fpsname
)
{
	// ===========================
	// CONNECT TO FPS
	// ===========================
	FPS_CONNECT( fpsname, FPSCONNECT_RUN );

	
	
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
long MyFunction(
    long arg0num, 
    long arg1num, 
    long arg2num, 
    long arg3num
) {
    char fpsname[200];
    
    long pindex = (long) getpid();  // index used to differentiate multiple calls to function
    // if we don't have anything more informative, we use PID
    
    FUNCTION_PARAMETER_STRUCT fps;

    // create FPS
    sprintf(fpsname, "myfunc-%06ld", pindex);
    MyFunction_FPCONF(fpsname, CMDCODE_FPSINIT, DMindex);

    function_parameter_struct_connect(fpsname, &fps, FPSCONNECT_SIMPLE);

    functionparameter_SetParamValue_INT64(&fps, ".arg0", arg0);
    functionparameter_SetParamValue_INT64(&fps, ".arg1", arg1);
    functionparameter_SetParamValue_INT64(&fps, ".arg2", arg2);
    functionparameter_SetParamValue_INT64(&fps, ".arg3", arg3);

    function_parameter_struct_disconnect(&fps);

    MyFunction_RUN(fpsname);

    return(IDout);
}



~~~




## 6.3. RUN function with FPS and processinfo {#page_FunctionParameterStructure_WritingRUNfunc_processinfo}


In this example, the loop process supports both FPS and processinfo.
This is the preferred way to code a loop process.

The example also shows using FPS to set the process realtime priority.



~~~~{.c}

/* \@brief Loop process code example
 *
 * ## Purpose
 *
 * This example demonstrates use of processinfo and fps structures.\n
 *
 * ## Arguments
 *
 * @param[in]
 * char		fpsname*
 * 			name of function parameter structure
 *
 * All function parameters are held inside the function parameter structure (FPS).\n
 * 
 *
 * ## Details
 *
 */

errno_t MyFunction_RUN(
    char *fpsname
)
{
	// ===========================
	// ### Connect to FPS 
	// ===========================
	FPS_CONNECT( fpsname, FPSCONNECT_RUN );
	
	
	// ===========================	
	// ### GET FUNCTION PARAMETER VALUES
	// ===========================
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
        fpsname,	             // re-use fpsname as processinfo name
        "computes something",    // description
        "add image1 to image2",  // message on startup
        __FUNCTION__, __FILE__, __LINE__
        );

	// OPTIONAL SETTINGS
    processinfo->MeasureTiming = 1; // Measure timing 
    processinfo->RT_priority = 20;  // RT_priority, 0-99. Larger number = higher priority. If <0, ignore
    processinfo->loopcntMax = 1000; // max number of iterations. -1 if infinite


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




	// ===========================
	// ### Start loop
	// ===========================

    
    processinfo_loopstart(processinfo); // Notify processinfo that we are entering loop

    while(loopOK==1)
    {
	    loopOK = processinfo_loopstep(processinfo);
     
     
        //
        // Semaphore wait goes here  
        // computation starts here
       
        
        
        processinfo_exec_start(processinfo);    
        if(processinfo_compute_status(processinfo)==1)
        {
            //
		    // computation ....
		    //
        }
  
  
        // Post semaphore(s) and counter(s) 
        // computation done
        
        // process signals, increment loop counter
        processinfo_exec_end(processinfo);
    
    
        // OPTIONAL: MESSAGE WHILE LOOP RUNNING
        processinfo_WriteMessage(processinfo, "loop running fine");		
	}


	// ==================================
	// ### ENDING LOOP
	// ==================================

     processinfo_cleanExit(processinfo);
	function_parameter_RUNexit( &fps  );
    
	return RETURN_SUCCESS;
}
~~~~



----





