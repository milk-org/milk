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

<fpsname> consists of a root name (string), and a series of optional integers, each printed on two digits:

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
- `_CONFINIT_`  : Initialize FPS for the function
- `_CONFSTART_` : Start the FPS configuration process
- `_CONFSTOP_`  : Stop the FPS configuration process
- `_RUNSTART_`  : Start the run process
- `_RUNSTOP_`   : Stop the run process
 
 
 
@note Why Optional argument to CLI function ?
@note Multiple instances of a C function may need to be running, each with its own FPS. An optional argument provides a mechanism to differentiate the FPSs. It is appended to the FPS name following a dash. The optional argument can be a number (usually integer) or a string.
 

Example source code below, assuming one optional long type argument:

~~~~{.c}

errno_t MyFunction_cli() {
    char fpsname[200];

    // First, we try to execute function through FPS interface
    if(CLI_checkarg(1, 5) + CLI_checkarg(2, 2) == 0) { // check that first arg is string, second arg is int
        unsigned int OptionalArg00 = data.cmdargtoken[2].val.numl;

        // Set FPS interface name
        // By convention, if there are optional arguments, they should be appended to the fps name
        //
        if(data.processnameflag == 0) { // name fps to something different than the process name
            sprintf(fpsname, "myfunc-%06u", OptionalArg00);
        } else { // Automatically set fps name to be process name up to first instance of character '.'
            strcpy(fpsname, data.processname0);
        }

        if(strcmp(data.cmdargtoken[1].val.string, "_CONFINIT_") == 0) {  // Initialize FPS and conf process
            printf("Function parameters configure\n");
            MyFunction_FPCONF(fpsname, CMDCODE_CONFINIT, OptionalArg00);
            return RETURN_SUCCESS;
        }

        if(strcmp(data.cmdargtoken[1].val.string, "_CONFSTART_") == 0) {  // Start conf process
            printf("Function parameters configure\n");
            MyFunction_FPCONF(fpsname, CMDCODE_CONFSTART, OptionalArg00);
            return RETURN_SUCCESS;
        }

        if(strcmp(data.cmdargtoken[1].val.string, "_CONFSTOP_") == 0) { // Stop conf process
            printf("Function parameters configure\n");
            MyFunction_FPCONF(fpsname, CMDCODE_CONFSTOP, OptionalArg00);
            return RETURN_SUCCESS;
        }

        if(strcmp(data.cmdargtoken[1].val.string, "_RUNSTART_") == 0) { // Run process
            printf("Run function\n");
            MyFunction_RUN(fpsname);
            return RETURN_SUCCESS;
        }

        if(strcmp(data.cmdargtoken[1].val.string, "_RUNSTOP_") == 0) { // Stop process
            printf("Run function\n");
            MyFunction_STOP(OptionalArg00);
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
	uint16_t loopstatus;
	

	// ===========================
	// SETUP FPS
	// ===========================
	
    FUNCTION_PARAMETER_STRUCT fps = function_parameter_FPCONFsetup(fpsname, CMDmode, &loopstatus);
    if( loopstatus == 0 ) // stop fps
        return 0;




	// ===========================
	// ALLOCATE FPS ENTRIES	
	// ===========================
	
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
     
    long fpi_param01 = function_parameter_add_entry(&fps, ".param01", "First parameter", FPTYPE_INT64, FPFLAG_DEFAULT_INPUT, pNull);
    
    // This parameter will be intitialized to a value of 5, min-max range from 0 to 10, and current value 5
    long param02default[4] = { 5, 0, 10, 5 };
    FPFLAG = FPFLAG_DEFAULT_INPUT | FPFLAG_MINLIMIT | FPFLAG_MAXLIMIT;  // required to enforce the min and max limits
    FPFLAG &= ~FPFLAG_WRITECONF;  // Don't allow parameter to be written during configuration
    FPFLAG &= ~FPFLAG_WRITERUN;   // Don't allow parameter to be written during run
    long fpi_param02 = function_parameter_add_entry(&fps, ".param02", "Second parameter", FPTYPE_INT64, FPFLAG, &param02default);
    
    // if parameter type = FPTYPE_FLOAT32, make sure default is declared as float[4]
    // if parameter type = FPTYPE_FLOAT64, make sure default is declared as double[4]
    float gaindefault[4] = { 0.01, 0.0, 1.0, 0.01 };
    FPFLAG = FPFLAG_DEFAULT_INPUT | FPFLAG_MINLIMIT | FPFLAG_MAXLIMIT;  // required to enforce the min and max limits
    long fpi_gain = function_parameter_add_entry(&fps, ".gain", "gain value", FPTYPE_FLOAT32, FPFLAG, &gaindefault);


	// This parameter is a ON / OFF toggle
	long fpi_gainset = function_parameter_add_entry(&fps, ".option.gainwrite", "gain can be changed", FPTYPE_ONOFF, FPFLAG_DEFAULT_INPUT, pNull);

	
	// stream that needs to be loaded on startup
	FPFLAG = FPFLAG_DEFAULT_INPUT_STREAM;
	long fp_streamname_wfs       = function_parameter_add_entry(&fps, ".sn_wfs",  "WFS stream name",
                                     FPTYPE_STREAMNAME, FPFLAG, pNull);



	// =====================================
	// PARAMETER LOGIC AND UPDATE LOOP
	// =====================================

	while ( loopstatus == 1 )
	{
		if( function_parameter_FPCONFloopstep(&fps, CMDmode, &loopstatus) == 1) // Apply logic if update is needed
		{
			// here goes the logic
			if ( fps.parray[fpi_gainset].status & FPFLAG_ONOFF )  // ON state
                {
                    fps.parray[fpi_gain].status |= FPFLAG_WRITERUN;
                    fps.parray[fpi_gain].status |= FPFLAG_USED;
                    fps.parray[fpi_gain].status |= FPFLAG_VISIBLE;
                }
                else // OFF state
                {
                    fps.parray[fpi_gain].status &= ~FPFLAG_WRITERUN;
                    fps.parray[fpi_gain].status &= ~FPFLAG_USED;
                    fps.parray[fpi_gain].status &= ~FPFLAG_VISIBLE;
                }
                
            functionparameter_CheckParametersAll(&fps);  // check all parameter values
		}		

	}

	function_parameter_FPCONFexit( &fps );

    return RETURN_SUCCESS;
}
~~~~	





---



# 6. Writing RUN function (in source .c file) {#page_FunctionParameterStructure_WritingRUNfunc}


The RUN function will connect to the FPS and execute the run loop. 

## 6.1. A simple example {#page_FunctionParameterStructure_WritingRUNfunc_simple}


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
	
	FUNCTION_PARAMETER_STRUCT fps;
	
	if(function_parameter_struct_connect(fpsname, &fps, FPSCONNECT_RUN) == -1)
	{
		printf("ERROR: fps \"%s\" does not exist -> running without FPS interface\n", fpsname);
		return RETURN_FAILURE;
	}

	
	
	// ===============================
	// GET FUNCTION PARAMETER VALUES
	// ===============================

	// parameters are addressed by their tag name
		
	// These parameters are read once, before running the loop
	//
	int param01 = functionparameter_GetParamValue_INT64(&fps, ".param01");
	int param02 = functionparameter_GetParamValue_INT64(&fps, ".param02");


	// This parameter value will be tracked during loop run, so we create a pointer for it
	// The corresponding function is functionparameter_GetParamPtr_<TYPE>
	//
	float *gain = functionparameter_GetParamPtr_FLOAT32(&fps, ".status.loopcnt");




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
    MyFunction_FPCONF(fpsname, CMDCODE_CONFINIT, DMindex);

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
//
// run loop process
//
errno_t MyFunction_RUN(
    char *fpsname
)
{
	// ===========================
	// Connect to FPS 
	// ===========================

	FUNCTION_PARAMETER_STRUCT fps;
	if(function_parameter_struct_connect(fpsname, &fps, FPSCONNECT_RUN) == -1)
	{
		printf("ERROR: fps \"%s\" does not exist -> running without FPS interface\n", fpsname);
		return RETURN_FAILURE;
	}
	
	// ===========================	
	// GET FUNCTION PARAMETER VALUES
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
	// processinfo support 
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

	// ===========================
	// Start loop
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
	// ENDING LOOP
	// ==================================

    processinfo_cleanExit(processinfo);




	return RETURN_SUCCESS;
}
~~~~



----




