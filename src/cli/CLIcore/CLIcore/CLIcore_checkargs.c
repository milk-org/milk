/**
 * @file CLIcore_checkargs.c
 *
 * @brief Check CLI command line arguments
 *
 */

#include <stdio.h>

#include "CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"
#include "fps_globals.h"

// keep processing if 1
static int argcheck_process_flag = 1;


// toggles to 1 if function help called
static int functionhelp_called = 0;


static const char* CLIargtype_to_string(uint32_t type)
{
    switch(type)
    {
        case FPTYPE_FLOAT32: return "FLOAT32";
        case FPTYPE_FLOAT64: return "FLOAT64";
        case FPTYPE_ONOFF: return "ONOFF";
        case FPTYPE_INT32: return "INT32";
        case FPTYPE_UINT32: return "UINT32";
        case FPTYPE_INT64: return "INT64";
        case FPTYPE_UINT64: return "UINT64";
        case FPTYPE_STRING_NOT_STREAM: return "STR_NOT_IMG";
        case FPTYPE_STREAMNAME: return "STREAM";
        case FPTYPE_STRING: return "STRING";
        case FPTYPE_FILENAME: return "FILENAME";
        case FPTYPE_FITSFILENAME: return "FITSFILE";
        case FPTYPE_FPSNAME: return "FPSNAME";
        case FPTYPE_EXECFILENAME: return "EXECFILE";
        case FPTYPE_DIRNAME: return "DIRNAME";
        case FPTYPE_PID: return "PID";
        case FPTYPE_TIMESPEC: return "TIMESPEC";
        case CLIARG_MISSING: return "MISSING";
        default: return "UNKNOWN";
    }
}

static const char* CMDARGTOKEN_type_to_string(uint32_t type)
{
    switch(type)
    {
        case 0: return "UNSOLVED_TOKEN";
        case 1: return "FLOAT_TOKEN";
        case 2: return "INT_TOKEN";
        case 3: return "STRING_TOKEN";
        case 4: return "IMG_TOKEN";
        case 5: return "CMD_TOKEN";
        case 6: return "RAWSTRING_TOKEN";
        default: return CLIargtype_to_string(type);
    }
}


/**
 * @brief check that input CLI argument matches required function argument type
 *
 * @param CLIargnum    CLI argument / token index
 * @param funcargtype   function argument type
 * @param errmsg    error message printing flag (1 if printing errors)
 * @return int
 */
static int CLI_checkarg0(
    int       CLIargnum,
    uint32_t  funcargtype,
    int       errmsg
)
{
    DEBUG_TRACE_FSTART();

    int      rval = 2; // Default to 'wrong type'
    uint32_t ftype = funcargtype;
    int      ttype = data.cmdargtoken[CLIargnum].type;

    if(strcmp(data.cmdargtoken[CLIargnum].val.string, "?") == 0)
    {
        argcheck_process_flag = 0; // stop processing arguments, will call help
        help_command(data.cmdargtoken[0].val.string);
        snprintf(data.cmdargtoken[CLIargnum].val.string,
                 STRINGMAXLEN_CMDARGTOKEN_VAL,
                 " "); // avoid re-running help
        functionhelp_called = 1;
        DEBUG_TRACE_FEXIT();
        return 1;
    }

    // Normalization and conversion
    if(ttype == CMDARGTOKEN_TYPE_FLOAT) {
       data.cmdargtoken[CLIargnum].val.numl = (long)(data.cmdargtoken[CLIargnum].val.numf + 0.5);
    }
    if(ttype == CMDARGTOKEN_TYPE_LONG) {
       data.cmdargtoken[CLIargnum].val.numf = (double)data.cmdargtoken[CLIargnum].val.numl;
    }

    // Special conversion for ONOFF
    if (ftype == FPTYPE_ONOFF) {
       if (strcasecmp(data.cmdargtoken[CLIargnum].val.string, "on") == 0) {
          data.cmdargtoken[CLIargnum].val.numl = 1;
          data.cmdargtoken[CLIargnum].val.numf = 1.0;
          rval = 0;
       }
       else if (strcasecmp(data.cmdargtoken[CLIargnum].val.string, "off") == 0) {
          data.cmdargtoken[CLIargnum].val.numl = 0;
          data.cmdargtoken[CLIargnum].val.numf = 0.0;
          rval = 0;
       }
    }

    // Type matching logic
    if (rval == 2) {
       if (ftype == FPTYPE_FLOAT32 || ftype == FPTYPE_FLOAT64) {
           if (ttype == CMDARGTOKEN_TYPE_FLOAT || ttype == (CLIARG_FLOAT32 & 0x0000FFFF) ||
               ttype == CMDARGTOKEN_TYPE_LONG || ttype == (CLIARG_INT64 & 0x0000FFFF) ||
               ttype == 6) {
              
               if (ttype == 6) {
                   data.cmdargtoken[CLIargnum].val.numf = atof(data.cmdargtoken[CLIargnum].val.string);
                   data.cmdargtoken[CLIargnum].val.numl = (long)data.cmdargtoken[CLIargnum].val.numf;
               }
               rval = 0;
           }
       }
       else if (ftype == FPTYPE_INT32 || ftype == FPTYPE_INT64 || ftype == FPTYPE_UINT32 || ftype == FPTYPE_UINT64 || ftype == FPTYPE_ONOFF || ftype == FPTYPE_PID || ftype == FPTYPE_TIMESPEC) {
           if (ttype == CMDARGTOKEN_TYPE_LONG || ttype == (CLIARG_INT64 & 0x0000FFFF) ||
               ttype == CMDARGTOKEN_TYPE_FLOAT || ttype == (CLIARG_FLOAT32 & 0x0000FFFF) ||
               ttype == 6) {
              
               if (ttype == 6) {
                   data.cmdargtoken[CLIargnum].val.numl = atol(data.cmdargtoken[CLIargnum].val.string);
                   data.cmdargtoken[CLIargnum].val.numf = (double)data.cmdargtoken[CLIargnum].val.numl;
               }
               rval = 0;
           }
       }
       else if (ftype == FPTYPE_STREAMNAME) {
           if (ttype == CMDARGTOKEN_TYPE_EXISTINGIMAGE || ftype == FPTYPE_STREAMNAME || 
               ttype == CMDARGTOKEN_TYPE_STRING || ftype == FPTYPE_STRING || 
               ttype == 6) {
              rval = 0;
           }
       }
       else if (ftype == FPTYPE_STRING || ftype == FPTYPE_STRING_NOT_STREAM || ftype == FPTYPE_FILENAME || ftype == FPTYPE_FITSFILENAME || ftype == FPTYPE_FPSNAME || ftype == FPTYPE_DIRNAME || ftype == FPTYPE_EXECFILENAME) {
           if (ttype == CMDARGTOKEN_TYPE_STRING || ftype == FPTYPE_STRING || 
               ttype == CMDARGTOKEN_TYPE_EXISTINGIMAGE || ftype == FPTYPE_STREAMNAME || 
               ttype == CLIARG_STR_NOT_IMG || ttype == 6) {
              rval = 0;
           }
       }
    }
    
    // Check if it's a variable if not already resolved
    if (rval == 2) {
       imageID IDv = variable_ID(data.cmdargtoken[CLIargnum].val.string);
       if (IDv != -1) {
          if (ftype == FPTYPE_FLOAT32 || ftype == FPTYPE_FLOAT64) {
             data.cmdargtoken[CLIargnum].val.numf = (double) dcvar[IDv].value.f;
             data.cmdargtoken[CLIargnum].val.numl = (long) data.cmdargtoken[CLIargnum].val.numf;
             data.cmdargtoken[CLIargnum].type = CLIARG_FLOAT64;
             rval = 0;
          }
          else if (ftype == FPTYPE_INT32 || ftype == FPTYPE_INT64 || ftype == FPTYPE_UINT32 || ftype == FPTYPE_UINT64 || ftype == FPTYPE_ONOFF) {
             data.cmdargtoken[CLIargnum].val.numl = (long) dcvar[IDv].value.l;
             data.cmdargtoken[CLIargnum].val.numf = (double) data.cmdargtoken[CLIargnum].val.numl;
             data.cmdargtoken[CLIargnum].type = CLIARG_INT64;
             rval = 0;
          }
       }
    }

    // Final result and error reporting
    if(rval == 2)
    {
        if(errmsg == 1)
        {
            printf("arg %d: wrong arg type (expected %s but got %s)\n",
                   CLIargnum - 1,
                   CLIargtype_to_string(funcargtype),
                   CMDARGTOKEN_type_to_string(data.cmdargtoken[CLIargnum].type));
        }
        rval = 1;
    }

    DEBUG_TRACE_FEXIT();
    return rval;
}


/**
 * @brief Check that input CLI argument matches required argument type
 *
 * @param CLIargnum
 * @param funcargtype
 * @return int
 */
int CLI_checkarg(int CLIargnum, uint32_t funcargtype)
{
    DEBUG_TRACE_FSTART();

    int rval;

    if(CLIargnum == 1)
    {
        argcheck_process_flag = 1;
    }

    if(argcheck_process_flag == 1)
    {
        rval = CLI_checkarg0(CLIargnum, funcargtype, 1);
    }
    else
    {
        rval = 1;
    }

    DEBUG_TRACE_FEXIT();
    return rval;
}

/**
 * @brief Check that input CLI argument matches required argument type - do not print error message
 *
 * @param CLIargnum
 * @param funcargtype
 * @return int
 */
int CLI_checkarg_noerrmsg(int CLIargnum, uint32_t funcargtype)
{
    DEBUG_TRACE_FSTART();

    int rval;

    if(CLIargnum == 1)
    {
        argcheck_process_flag = 1;
    }

    if(argcheck_process_flag == 1)
    {
        rval = CLI_checkarg0(CLIargnum, funcargtype, 0);
    }
    else
    {
        rval = 1;
    }

    DEBUG_TRACE_FEXIT();
    return rval;
}


/** @brief Check array of command line (CLI) arguments
 *
 * Use list of arguments in fpscliarg[].
 * Skip arguments that have CLICMDARG_FLAG_NOCLI flag.
 *
 * CLIarg keep count of argument position in CLI call
 *
 */
errno_t CLI_checkarg_array(
    CLICMDARGDEF fpscliarg[],
    int nbarg
)
{
    DEBUG_TRACE_FSTART();

    // initialize arg check
    argcheck_process_flag = 1;

    // Sync current values from FPS to CLI argdata BEFORE processing
    if(dcfpsptr != NULL)
    {
        for(int arg = 0; arg < nbarg; arg++)
        {
            long pindex = functionparameter_GetParamIndex(dcfpsptr,
                fpscliarg[arg].fpstag);
            if(pindex != -1)
            {
                uint32_t ptype = dcfpsptr->parray[pindex].type;
                switch(ptype)
                {
                    case FPTYPE_FLOAT32:
                        data.cmd[data.cmdindex].argdata[arg].val.f32 = dcfpsptr->parray[pindex].val.f32[0];
                        break;
                    case FPTYPE_FLOAT64:
                        data.cmd[data.cmdindex].argdata[arg].val.f64 = dcfpsptr->parray[pindex].val.f64[0];
                        break;
                    case FPTYPE_INT32:
                        data.cmd[data.cmdindex].argdata[arg].val.i32 = dcfpsptr->parray[pindex].val.i32[0];
                        break;
                    case FPTYPE_UINT32:
                        data.cmd[data.cmdindex].argdata[arg].val.ui32 = dcfpsptr->parray[pindex].val.ui32[0];
                        break;
                    case FPTYPE_INT64:
                        data.cmd[data.cmdindex].argdata[arg].val.i64 = dcfpsptr->parray[pindex].val.i64[0];
                        break;
                    case FPTYPE_UINT64:
                        data.cmd[data.cmdindex].argdata[arg].val.ui64 = dcfpsptr->parray[pindex].val.ui64[0];
                        break;
                    case FPTYPE_ONOFF:
                        data.cmd[data.cmdindex].argdata[arg].val.i64 = dcfpsptr->parray[pindex].val.i32[0];
                        break;
                    case FPTYPE_PID:
                        data.cmd[data.cmdindex].argdata[arg].val.i64 = (int64_t)dcfpsptr->parray[pindex].val.pid[0];
                        break;
                    case FPTYPE_TIMESPEC:
                        data.cmd[data.cmdindex].argdata[arg].val.f64 = (double)dcfpsptr->parray[pindex].val.ts[0].tv_sec + (double)dcfpsptr->parray[pindex].val.ts[0].tv_nsec * 1e-9;
                        break;
                    case FPTYPE_STRING:
                    case FPTYPE_STREAMNAME:
                    case FPTYPE_DIRNAME:
                    case FPTYPE_FILENAME:
                    case FPTYPE_FITSFILENAME:
                    case FPTYPE_EXECFILENAME:
                    case FPTYPE_FPSNAME:
                    case FPTYPE_PROCESS:
                    case FPTYPE_STRING_NOT_STREAM:
                        strncpy(data.cmd[data.cmdindex].argdata[arg].val.s,
                            dcfpsptr->parray[pindex].val.string[0],
                            STRINGMAXLEN_CLICMDARG - 1);
                        break;
                }
            }
        }
    }

    int argindexmatch = -1;
    // check if CLI argument 1 is one of the function parameters keys
    // if it is, set argindexmatch to the function parameter index
    for(int arg = 0; arg < nbarg; arg++)
    {
        if(strcmp(data.cmdargtoken[1].val.string, fpscliarg[arg].fpstag) == 0)
        {
            argindexmatch = arg;
        }
    }

    // if CLI arg 1 is not in the static list, check if it's a tag in the FPS
    if(argindexmatch == -1 && dcfpsptr != NULL)
    {
        if(data.cmdargtoken[1].val.string[0] == '.')
        {
            char *fpstag = data.cmdargtoken[1].val.string;
            if(fpstag[1] == '.') // Handle ".." prefix
            {
                fpstag++;
            }

            long pindex = functionparameter_GetParamIndex(dcfpsptr, fpstag);
            if(pindex != -1)
            {
                if(data.cmdargtoken[2].type == CLIARG_MISSING)
                {
                    printf("Setting parameter %s : input missing\n",
                           data.cmdargtoken[1].val.string);
                    DEBUG_TRACE_FEXIT();
                    return RETURN_CLICHECKARGARRAY_FAILURE;
                }

                // Update the parameter in FPS
                uint32_t ptype = dcfpsptr->parray[pindex].type;
                switch(ptype)
                {
                    case FPTYPE_INT64:
                        functionparameter_SetParamValue_INT64(dcfpsptr,
                            fpstag,
                            data.cmdargtoken[2].val.numl);
                        break;
                    case FPTYPE_UINT64:
                        functionparameter_SetParamValue_UINT64(dcfpsptr,
                            fpstag,
                            data.cmdargtoken[2].val.numl);
                        break;
                    case FPTYPE_INT32:
                        functionparameter_SetParamValue_INT32(dcfpsptr,
                            fpstag,
                            data.cmdargtoken[2].val.numl);
                        break;
                    case FPTYPE_UINT32:
                        functionparameter_SetParamValue_UINT32(dcfpsptr,
                            fpstag,
                            data.cmdargtoken[2].val.numl);
                        break;
                    case FPTYPE_FLOAT64:
                        functionparameter_SetParamValue_FLOAT64(dcfpsptr,
                            fpstag,
                            data.cmdargtoken[2].val.numf);
                        break;
                    case FPTYPE_FLOAT32:
                        functionparameter_SetParamValue_FLOAT32(dcfpsptr,
                            fpstag,
                            (float)data.cmdargtoken[2].val.numf);
                        break;
                    case FPTYPE_PID:
                        functionparameter_SetParamValue_INT64(dcfpsptr,
                            fpstag,
                            (int64_t)data.cmdargtoken[2].val.numl);
                        break;
                    case FPTYPE_TIMESPEC:
                        functionparameter_SetParamValue_TIMESPEC(dcfpsptr,
                            fpstag,
                            data.cmdargtoken[2].val.numf);
                        break;
                    case FPTYPE_STRING:
                    case FPTYPE_STREAMNAME:
                    case FPTYPE_DIRNAME:
                    case FPTYPE_FILENAME:
                    case FPTYPE_FITSFILENAME:
                    case FPTYPE_EXECFILENAME:
                    case FPTYPE_FPSNAME:
                    case FPTYPE_PROCESS:
                    case FPTYPE_STRING_NOT_STREAM:
                        functionparameter_SetParamValue_STRING(dcfpsptr,
                            fpstag,
                            data.cmdargtoken[2].val.string);
                        break;
                    case FPTYPE_ONOFF:
                        functionparameter_SetParamValue_ONOFF(dcfpsptr,
                            fpstag,
                            (int)data.cmdargtoken[2].val.numl);
                        break;
                }

                char valstr[STRINGMAXLEN_FPSCLIARG_TAG];
                functionparameter_GetParamValueString(&dcfpsptr->parray[pindex],
                                                      valstr,
                                                      STRINGMAXLEN_FPSCLIARG_TAG);

                printf("Parameter %s value updated to %s in FPS\n",
                       data.cmdargtoken[1].val.string,
                       valstr);
                DEBUG_TRACE_FEXIT();
                return RETURN_CLICHECKARGARRAY_FUNCPARAMSET;
            }
        }
    }

    // if CLI arg 1 is a function parameter, set function parameter to value entered in CLI arg 2
    if(argindexmatch != -1)
    {
        if(data.cmdargtoken[2].type == CLIARG_MISSING)
        {
            printf("Setting arg %s : input missing\n",
                   fpscliarg[argindexmatch].fpstag);
            DEBUG_TRACE_FEXIT();
            return RETURN_CLICHECKARGARRAY_FAILURE;
        }
        //printf("match to arg %s\n", fpscliarg[argindexmatch].fpstag); //TEST

        DEBUG_TRACEPOINT("calling CLI_checkarg");
        if(CLI_checkarg(2, fpscliarg[argindexmatch].type) == 0)
        {
            int cmdi = data.cmdindex;
            switch(fpscliarg[argindexmatch].type)  // & 0x0000FFFF)
            {
                case CLIARG_FLOAT32:
                    data.cmd[cmdi].argdata[argindexmatch].val.f32 =
                        data.cmdargtoken[2].val.numf;
                    break;
                case CLIARG_FLOAT64:
                    data.cmd[cmdi].argdata[argindexmatch].val.f64 =
                        data.cmdargtoken[2].val.numf;
                    break;
                case CLIARG_INT32:
                    data.cmd[cmdi].argdata[argindexmatch].val.i32 =
                        data.cmdargtoken[2].val.numl;
                    break;
                case CLIARG_INT64:
                    data.cmd[cmdi].argdata[argindexmatch].val.i64 =
                        data.cmdargtoken[2].val.numl;
                    break;
                case CLIARG_UINT32:
                    data.cmd[cmdi].argdata[argindexmatch].val.ui32 =
                        data.cmdargtoken[2].val.numl;
                    break;
                case CLIARG_UINT64:
                    data.cmd[cmdi].argdata[argindexmatch].val.ui64 =
                        data.cmdargtoken[2].val.numl;
                    break;
                case CLIARG_ONOFF:
                    data.cmd[cmdi].argdata[argindexmatch].val.i64 =
                        data.cmdargtoken[2].val.numl;
                    break;
                case CLIARG_STR_NOT_IMG:
                case CLIARG_IMG:
                case CLIARG_STR:
                case FPTYPE_FILENAME:
                case FPTYPE_FITSFILENAME:
                case FPTYPE_FPSNAME:
                case FPTYPE_DIRNAME:
                case FPTYPE_EXECFILENAME:
                case FPTYPE_PROCESS:
                    strncpy(data.cmd[cmdi].argdata[argindexmatch].val.s,
                            data.cmdargtoken[2].val.string,
                            STRINGMAXLEN_CLICMDARG - 1);
                    break;
                case FPTYPE_PID:
                    data.cmd[cmdi].argdata[argindexmatch].val.i64 = data.cmdargtoken[2].val.numl;
                    break;
                case FPTYPE_TIMESPEC:
                    data.cmd[cmdi].argdata[argindexmatch].val.f64 = data.cmdargtoken[2].val.numf;
                    break;
            }

            // Sync updated value back to FPS
            if(dcfpsptr != NULL)
            {
                long pindex = functionparameter_GetParamIndex(dcfpsptr,
                    fpscliarg[argindexmatch].fpstag);
                if(pindex != -1)
                {
                    uint32_t ptype = dcfpsptr->parray[pindex].type;
                    switch(ptype)
                    {
                        case FPTYPE_INT64:
                            functionparameter_SetParamValue_INT64(dcfpsptr,
                                fpscliarg[argindexmatch].fpstag,
                                data.cmdargtoken[2].val.numl);
                            break;
                        case FPTYPE_UINT64:
                            functionparameter_SetParamValue_UINT64(dcfpsptr,
                                fpscliarg[argindexmatch].fpstag,
                                data.cmdargtoken[2].val.numl);
                            break;
                        case FPTYPE_INT32:
                            functionparameter_SetParamValue_INT32(dcfpsptr,
                                fpscliarg[argindexmatch].fpstag,
                                data.cmdargtoken[2].val.numl);
                            break;
                        case FPTYPE_UINT32:
                            functionparameter_SetParamValue_UINT32(dcfpsptr,
                                fpscliarg[argindexmatch].fpstag,
                                data.cmdargtoken[2].val.numl);
                            break;
                        case FPTYPE_FLOAT64:
                            functionparameter_SetParamValue_FLOAT64(dcfpsptr,
                                fpscliarg[argindexmatch].fpstag,
                                data.cmdargtoken[2].val.numf);
                            break;
                        case FPTYPE_FLOAT32:
                            functionparameter_SetParamValue_FLOAT32(dcfpsptr,
                                fpscliarg[argindexmatch].fpstag,
                                (float)data.cmdargtoken[2].val.numf);
                            break;
                        case FPTYPE_PID:
                            functionparameter_SetParamValue_INT64(dcfpsptr,
                                fpscliarg[argindexmatch].fpstag,
                                (int64_t)data.cmdargtoken[2].val.numl);
                            break;
                        case FPTYPE_TIMESPEC:
                            functionparameter_SetParamValue_TIMESPEC(dcfpsptr,
                                fpscliarg[argindexmatch].fpstag,
                                data.cmdargtoken[2].val.numf);
                            break;
                        case FPTYPE_STRING:
                        case FPTYPE_STREAMNAME:
                        case FPTYPE_DIRNAME:
                        case FPTYPE_FILENAME:
                        case FPTYPE_FITSFILENAME:
                        case FPTYPE_EXECFILENAME:
                        case FPTYPE_FPSNAME:
                        case FPTYPE_PROCESS:
                        case FPTYPE_STRING_NOT_STREAM:
                            functionparameter_SetParamValue_STRING(dcfpsptr,
                                fpscliarg[argindexmatch].fpstag,
                                data.cmdargtoken[2].val.string);
                            break;
                        case FPTYPE_ONOFF:
                            functionparameter_SetParamValue_ONOFF(dcfpsptr,
                                fpscliarg[argindexmatch].fpstag,
                                (int)data.cmdargtoken[2].val.numl);
                            break;
                    }
                }
            }
        }
        else
        {
            printf("Setting arg %s : Wrong type\n",
                   fpscliarg[argindexmatch].fpstag);
            DEBUG_TRACE_FEXIT();
            return RETURN_CLICHECKARGARRAY_FAILURE;
        }

        if((dcfpsptr != NULL) && (dcfpsptr->parray != NULL))
        {
            char valstr[STRINGMAXLEN_FPSCLIARG_TAG];
            long fpsi = -1;
            if(fpscliarg[argindexmatch].indexptr != NULL)
            {
                fpsi = *fpscliarg[argindexmatch].indexptr;
            }
            if(fpsi == -1) // If index not yet known, look it up
            {
                fpsi = functionparameter_GetParamIndex(
                           dcfpsptr,
                           fpscliarg[argindexmatch].fpstag);
            }

            if((fpsi >= 0) && (fpsi < dcfpsptr->md->NBparamMAX))
            {
                functionparameter_GetParamValueString(&dcfpsptr->parray[fpsi],
                                                      valstr,
                                                      STRINGMAXLEN_FPSCLIARG_TAG);
                printf("Argument %s value updated to %s\n",
                       fpscliarg[argindexmatch].fpstag,
                       valstr);
            }
            else
            {
                printf("Argument %s value updated to %s\n",
                       fpscliarg[argindexmatch].fpstag,
                       data.cmdargtoken[2].val.string);
            }
        }
        else
        {
            printf("Argument %s value updated to %s\n",
                   fpscliarg[argindexmatch].fpstag,
                   data.cmdargtoken[2].val.string);
        }

        //printf("arg 1: [%d] %s %f %ld\n", data.cmdargtoken[2].type, data.cmdargtoken[2].val.string, data.cmdargtoken[2].val.numf, data.cmdargtoken[2].val.numl);
        DEBUG_TRACE_FEXIT();
        return RETURN_CLICHECKARGARRAY_FUNCPARAMSET;
    }

    //printf("arg 1: %s %f %ld\n", data.cmdargtoken[2].val.string);

    int nberr  = 0;
    int CLIarg = 0; // index of argument in CLI call
    int printed_default_warning = 0;

    for(int arg = 0; arg < nbarg; arg++)
    {
        char argtypestring[16];
        switch(fpscliarg[arg].type)
        {
            case CLIARG_FLOAT32:
                strcpy(argtypestring, "FLT32");
                break;
            case CLIARG_FLOAT64:
                strcpy(argtypestring, "FLT64");
                break;
            case CLIARG_ONOFF:
                strcpy(argtypestring, "ONOFF");
                break;
            case CLIARG_INT32:
                strcpy(argtypestring, "INT32");
                break;
            case CLIARG_UINT32:
                strcpy(argtypestring, "UINT32");
                break;
            case CLIARG_INT64:
                strcpy(argtypestring, "INT64");
                break;
            case CLIARG_UINT64:
                strcpy(argtypestring, "UINT64");
                break;
            case CLIARG_STR_NOT_IMG:
                strcpy(argtypestring, "STRnIMG");
                break;
            case CLIARG_IMG:
                strcpy(argtypestring, "STREAM");
                break;
            case CLIARG_STR:
                strcpy(argtypestring, "STRING");
                break;
        }

        if(fpscliarg[arg].fpflag & FPFLAG_PRIMARY_CLI_INPUT)
        {
            int cmdi = data.cmdindex;

            DEBUG_TRACEPOINT("  arg %d  CLI %2d  [%7s]  %s",
                             arg,
                             CLIarg,
                             argtypestring,
                             fpscliarg[arg].fpstag);

            if(CLIarg + 1 >= data.cmdNBarg) // Missing mandatory argument
            {
                if((data.cmdNBarg == 1) && (dcfpsptr != NULL))
                {
                    // Allow no argument call if FPS is connected
                    if(printed_default_warning == 0)
                    {
                        printf("\033[36mCommand entered without arguments. Adopting current values:\033[0m\n");
                        printed_default_warning = 1;
                    }

                    printf("  %-15s = ", fpscliarg[arg].fpstag);
                    switch(fpscliarg[arg].type)
                    {
                        case CLIARG_FLOAT32:
                            printf("%f\n", data.cmd[cmdi].argdata[arg].val.f32);
                            break;
                        case CLIARG_FLOAT64:
                            printf("%lf\n", data.cmd[cmdi].argdata[arg].val.f64);
                            break;
                        case CLIARG_INT32:
                            printf("%d\n", data.cmd[cmdi].argdata[arg].val.i32);
                            break;
                        case CLIARG_INT64:
                        case CLIARG_ONOFF:
                        case FPTYPE_PID:
                            printf("%ld\n", data.cmd[cmdi].argdata[arg].val.i64);
                            break;
                        case CLIARG_UINT32:
                            printf("%u\n", data.cmd[cmdi].argdata[arg].val.ui32);
                            break;
                        case CLIARG_UINT64:
                            printf("%lu\n", data.cmd[cmdi].argdata[arg].val.ui64);
                            break;
                        case FPTYPE_TIMESPEC:
                            printf("%lf\n", data.cmd[cmdi].argdata[arg].val.f64);
                            break;
                        case CLIARG_STR_NOT_IMG:
                        case CLIARG_IMG:
                        case CLIARG_STR:
                        case FPTYPE_FILENAME:
                        case FPTYPE_FITSFILENAME:
                        case FPTYPE_FPSNAME:
                        case FPTYPE_DIRNAME:
                        case FPTYPE_EXECFILENAME:
                        case FPTYPE_PROCESS:
                            printf("\"%s\"\n", data.cmd[cmdi].argdata[arg].val.s);
                            break;
                        default:
                            printf("?\n");
                            break;
                    }

                    continue;
                }
                else
                {
                    printf("Error: Missing mandatory argument %d (%s: %s)\n",
                           CLIarg,
                           fpscliarg[arg].fpstag,
                           fpscliarg[arg].descr);
                    help_command(data.cmd[data.cmdindex].key);
                    argcheck_process_flag = 0;
                    DEBUG_TRACE_FEXIT();
                    return RETURN_CLICHECKARGARRAY_FAILURE;
                }
            }

            if(strcmp(data.cmdargtoken[CLIarg + 1].val.string, ".") == 0)
            {
                // if arg token starts with "."
                DEBUG_TRACEPOINT("ADOPTING DEFAULT/LAST VALUE");
                switch(fpscliarg[arg].type)  // & 0x0000FFFF)
                {

                    case CLIARG_FLOAT32: // if desired type is float single precision
                        data.cmdargtoken[CLIarg + 1].val.numf =
                            data.cmd[cmdi].argdata[arg].val.f32;
                        data.cmdargtoken[CLIarg + 1].type = CLIARG_FLOAT32;
                        break;

                    case CLIARG_FLOAT64: // if desired type is float double precision
                        data.cmdargtoken[CLIarg + 1].val.numf =
                            data.cmd[cmdi].argdata[arg].val.f64;
                        data.cmdargtoken[CLIarg + 1].type = CLIARG_FLOAT64;
                        break;

                    case CLIARG_INT32:
                        data.cmdargtoken[CLIarg + 1].val.numl =
                            data.cmd[cmdi].argdata[arg].val.i32;
                        data.cmdargtoken[CLIarg + 1].type = CLIARG_INT32;
                        break;

                    case CLIARG_INT64:
                        data.cmdargtoken[CLIarg + 1].val.numl =
                            data.cmd[cmdi].argdata[arg].val.i64;
                        data.cmdargtoken[CLIarg + 1].type = CLIARG_INT64;
                        break;

                    case CLIARG_UINT32:
                        data.cmdargtoken[CLIarg + 1].val.numl =
                            data.cmd[cmdi].argdata[arg].val.ui32;
                        data.cmdargtoken[CLIarg + 1].type = CLIARG_UINT32;
                        break;

                    case CLIARG_UINT64:
                        data.cmdargtoken[CLIarg + 1].val.numl =
                            data.cmd[cmdi].argdata[arg].val.ui64;
                        data.cmdargtoken[CLIarg + 1].type = CLIARG_UINT64;
                        break;

                    case CLIARG_STR_NOT_IMG: // if desired is string not image
                        strncpy(data.cmdargtoken[CLIarg + 1].val.string,
                                data.cmd[cmdi].argdata[arg].val.s,
                                STRINGMAXLEN_CMDARGTOKEN_VAL - 1);
                        data.cmdargtoken[CLIarg + 1].type = CLIARG_STR_NOT_IMG;
                        break;

                    case CLIARG_IMG: // should be image
                        strncpy(data.cmdargtoken[CLIarg + 1].val.string,
                                data.cmd[cmdi].argdata[arg].val.s,
                                STRINGMAXLEN_CMDARGTOKEN_VAL - 1);
                        if(image_ID(data.cmd[cmdi].argdata[arg].val.s, dcimg, dcnimg) != -1)
                        {
                            // if image exists
                            data.cmdargtoken[CLIarg + 1].type = CLIARG_IMG;
                        }
                        else
                        {
                            data.cmdargtoken[CLIarg + 1].type = CLIARG_STR_NOT_IMG;
                        }
                        //printf("arg %d IMG        : %s\n", CLIarg+1, data.cmdargtoken[CLIarg+1].val.string);
                        break;

                    case CLIARG_STR:
                        strncpy(data.cmdargtoken[CLIarg + 1].val.string,
                                data.cmd[cmdi].argdata[arg].val.s,
                                STRINGMAXLEN_CMDARGTOKEN_VAL - 1);
                        data.cmdargtoken[CLIarg + 1].type = CLIARG_STR;
                        break;
                }
            }

            DEBUG_TRACEPOINT("calling CLI_checkarg");
            if(CLI_checkarg(CLIarg + 1, fpscliarg[arg].type) == 0)
            {
                DEBUG_TRACEPOINT("successful parsing, update default to last");
                switch(fpscliarg[arg].type)  // & 0x0000FFFF)
                {
                    case CLIARG_FLOAT32:
                        data.cmd[cmdi].argdata[arg].val.f32 =
                            data.cmdargtoken[CLIarg + 1].val.numf;
                        break;
                    case CLIARG_FLOAT64:
                        data.cmd[cmdi].argdata[arg].val.f64 =
                            data.cmdargtoken[CLIarg + 1].val.numf;
                        break;
                    case CLIARG_INT32:
                        data.cmd[cmdi].argdata[arg].val.i32 =
                            data.cmdargtoken[CLIarg + 1].val.numl;
                        break;
                    case CLIARG_INT64:
                        data.cmd[cmdi].argdata[arg].val.i64 =
                            data.cmdargtoken[CLIarg + 1].val.numl;
                        break;
                    case CLIARG_UINT32:
                        data.cmd[cmdi].argdata[arg].val.ui32 =
                            data.cmdargtoken[CLIarg + 1].val.numl;
                        break;
                    case CLIARG_UINT64:
                        data.cmd[cmdi].argdata[arg].val.ui64 =
                            data.cmdargtoken[CLIarg + 1].val.numl;
                        break;
                    case FPTYPE_PID:
                        data.cmd[cmdi].argdata[arg].val.i64 =
                            data.cmdargtoken[CLIarg + 1].val.numl;
                        break;
                    case FPTYPE_TIMESPEC:
                        data.cmd[cmdi].argdata[arg].val.f64 =
                            data.cmdargtoken[CLIarg + 1].val.numf;
                        break;
                    case CLIARG_STR_NOT_IMG:
                    case CLIARG_IMG:
                    case CLIARG_STR:
                    case FPTYPE_FILENAME:
                    case FPTYPE_FITSFILENAME:
                    case FPTYPE_FPSNAME:
                    case FPTYPE_DIRNAME:
                    case FPTYPE_EXECFILENAME:
                    case FPTYPE_PROCESS:
                        strncpy(data.cmd[cmdi].argdata[arg].val.s,
                                data.cmdargtoken[CLIarg + 1].val.string,
                                STRINGMAXLEN_CLICMDARG - 1);
                        break;
                }
            }
            else
            {
                if(functionhelp_called == 1)
                {
                    DEBUG_TRACE_FEXIT();
                    return RETURN_CLICHECKARGARRAY_HELP;
                }
                nberr++;
            }
            CLIarg++;
        }
        else
        {
            DEBUG_TRACEPOINT("argument not part of CLI");
            DEBUG_TRACEPOINT("  arg %d  IGNORED [%7s]  %s",
                             arg,
                             argtypestring,
                             fpscliarg[arg].fpstag);
        }
    }

    DEBUG_TRACEPOINT("Number of arg error(s): %d / %d", nberr, CLIarg);

    if(nberr == 0)
    {
        DEBUG_TRACE_FEXIT();
        return RETURN_CLICHECKARGARRAY_SUCCESS;
    }
    else
    {
        DEBUG_TRACE_FEXIT();
        return RETURN_CLICHECKARGARRAY_FAILURE;
    }
}


/** @brief Build FPS content from FPSCLIARG list
 *
 * All CLI arguments converted to FPS parameters
 *
 */
int CLIargs_to_FPSparams_setval(CLICMDARGDEF               fpscliarg[],
                                int                        nbarg,
                                FUNCTION_PARAMETER_STRUCT *fps)
{
    DEBUG_TRACE_FSTART();

    int NBarg_processed = 0;
    int cmdi = data.cmdindex;

    for(int arg = 0; arg < nbarg; arg++)
    {
        // if argument is part of FPS
        switch(fpscliarg[arg].type)
        {
                case CLIARG_FLOAT32:
                    functionparameter_SetParamValue_FLOAT32(
                        fps,
                        fpscliarg[arg].fpstag,
                        data.cmd[cmdi].argdata[arg].val.f32);
                    NBarg_processed++;
                    break;

                case CLIARG_FLOAT64:
                    functionparameter_SetParamValue_FLOAT64(
                        fps,
                        fpscliarg[arg].fpstag,
                        data.cmd[cmdi].argdata[arg].val.f64);
                    NBarg_processed++;
                    break;

                case CLIARG_ONOFF:
                    functionparameter_SetParamValue_ONOFF(
                        fps,
                        fpscliarg[arg].fpstag,
                        (int) data.cmd[cmdi].argdata[arg].val.i64);
                    NBarg_processed++;
                    break;

                case CLIARG_INT32:
                    functionparameter_SetParamValue_INT32(
                        fps,
                        fpscliarg[arg].fpstag,
                        data.cmd[cmdi].argdata[arg].val.i32);
                    NBarg_processed++;
                    break;

                case CLIARG_UINT32:
                    functionparameter_SetParamValue_UINT32(
                        fps,
                        fpscliarg[arg].fpstag,
                        data.cmd[cmdi].argdata[arg].val.ui32);
                    NBarg_processed++;
                    break;

                case CLIARG_INT64:
                    functionparameter_SetParamValue_INT64(
                        fps,
                        fpscliarg[arg].fpstag,
                        data.cmd[cmdi].argdata[arg].val.i64);
                    NBarg_processed++;
                    break;

                case CLIARG_UINT64:
                    functionparameter_SetParamValue_UINT64(
                        fps,
                        fpscliarg[arg].fpstag,
                        data.cmd[cmdi].argdata[arg].val.ui64);
                    NBarg_processed++;
                    break;
                
                case FPTYPE_PID:
                    functionparameter_SetParamValue_INT64(
                        fps,
                        fpscliarg[arg].fpstag,
                        data.cmd[cmdi].argdata[arg].val.i64);
                    NBarg_processed++;
                    break;
                    
                case FPTYPE_TIMESPEC:
                    functionparameter_SetParamValue_TIMESPEC(
                        fps,
                        fpscliarg[arg].fpstag,
                        data.cmd[cmdi].argdata[arg].val.f64);
                    NBarg_processed++;
                    break;

                case CLIARG_STR_NOT_IMG:
                case CLIARG_IMG:
                case CLIARG_STR:
                case FPTYPE_FILENAME:
                case FPTYPE_FITSFILENAME:
                case FPTYPE_FPSNAME:
                case FPTYPE_DIRNAME:
                case FPTYPE_EXECFILENAME:
                case FPTYPE_PROCESS:
                    functionparameter_SetParamValue_STRING(
                        fps,
                        fpscliarg[arg].fpstag,
                        data.cmd[cmdi].argdata[arg].val.s);
                    NBarg_processed++;
                    break;
            }
    }

    DEBUG_TRACE_FEXIT();
    return NBarg_processed;
}


/** @brief Build FPS from command args
 */
int CMDargs_to_FPSparams_create(
    FUNCTION_PARAMETER_STRUCT *fps
)
{
    DEBUG_TRACE_FSTART();

    int  NBarg_processed = 0;
    long fpi             = 0;


    for(int argi = 0; argi < data.cmd[data.cmdindex].nbparam; argi++)
    {
        // if argument is part of FPS
        long tmpvall = 0;

        switch(data.cmd[data.cmdindex].argdata[argi].type)
        {
                // float point types

                case FPTYPE_FLOAT32:
                {
                    float tmpf = data.cmd[data.cmdindex].argdata[argi].val.f32;
                    function_parameter_add_entry(
                        fps,
                        data.cmd[data.cmdindex].argdata[argi].fpstag,
                        data.cmd[data.cmdindex].argdata[argi].descr,
                        FPTYPE_FLOAT32,
                        data.cmd[data.cmdindex].argdata[argi].fpflag,
                        &tmpf,
                        NULL);
                    NBarg_processed++;
                }
                break;

                case FPTYPE_FLOAT64:
                {
                    double tmplf = data.cmd[data.cmdindex].argdata[argi].val.f64;
                    function_parameter_add_entry(
                        fps,
                        data.cmd[data.cmdindex].argdata[argi].fpstag,
                        data.cmd[data.cmdindex].argdata[argi].descr,
                        FPTYPE_FLOAT64,
                        data.cmd[data.cmdindex].argdata[argi].fpflag,
                        &tmplf,
                        NULL);
                    NBarg_processed++;
                }
                break;

                // integer typtes

                case FPTYPE_ONOFF: // default to INT64
                {
                    tmpvall = data.cmd[data.cmdindex].argdata[argi].val.ui64;
                    function_parameter_add_entry(
                        fps,
                        data.cmd[data.cmdindex].argdata[argi].fpstag,
                        data.cmd[data.cmdindex].argdata[argi].descr,
                        FPTYPE_ONOFF,
                        data.cmd[data.cmdindex].argdata[argi].fpflag,
                        &tmpvall,
                        NULL);
                    NBarg_processed++;
                }
                break;

                case FPTYPE_INT32:
                {
                    int32_t tmpi32 = data.cmd[data.cmdindex].argdata[argi].val.i32;
                    function_parameter_add_entry(
                        fps,
                        data.cmd[data.cmdindex].argdata[argi].fpstag,
                        data.cmd[data.cmdindex].argdata[argi].descr,
                        FPTYPE_INT32,
                        data.cmd[data.cmdindex].argdata[argi].fpflag,
                        &tmpi32,
                        NULL);
                    NBarg_processed++;
                }
                break;

                case FPTYPE_UINT32:
                {
                    uint32_t tmpui32 =
                        data.cmd[data.cmdindex].argdata[argi].val.ui32;
                    function_parameter_add_entry(
                        fps,
                        data.cmd[data.cmdindex].argdata[argi].fpstag,
                        data.cmd[data.cmdindex].argdata[argi].descr,
                        FPTYPE_UINT32,
                        data.cmd[data.cmdindex].argdata[argi].fpflag,
                        &tmpui32,
                        NULL);
                    NBarg_processed++;
                }
                break;

                case FPTYPE_INT64:
                {
                    int64_t tmpi64 = data.cmd[data.cmdindex].argdata[argi].val.i64;
                    function_parameter_add_entry(
                        fps,
                        data.cmd[data.cmdindex].argdata[argi].fpstag,
                        data.cmd[data.cmdindex].argdata[argi].descr,
                        FPTYPE_INT64,
                        data.cmd[data.cmdindex].argdata[argi].fpflag,
                        &tmpi64,
                        NULL);
                    NBarg_processed++;
                }
                break;

                case FPTYPE_UINT64:
                {
                    uint64_t tmpui64 =
                        data.cmd[data.cmdindex].argdata[argi].val.ui64;
                    function_parameter_add_entry(
                        fps,
                        data.cmd[data.cmdindex].argdata[argi].fpstag,
                        data.cmd[data.cmdindex].argdata[argi].descr,
                        FPTYPE_UINT64,
                        data.cmd[data.cmdindex].argdata[argi].fpflag,
                        &tmpui64,
                        NULL);
                    NBarg_processed++;
                }
                break;

                case FPTYPE_STRING_NOT_STREAM:
                case FPTYPE_STRING:
                    function_parameter_add_entry(
                        fps,
                        data.cmd[data.cmdindex].argdata[argi].fpstag,
                        data.cmd[data.cmdindex].argdata[argi].descr,
                        FPTYPE_STRING,
                        data.cmd[data.cmdindex].argdata[argi].fpflag,
                        data.cmd[data.cmdindex].argdata[argi].val.s,
                        NULL);
                    NBarg_processed++;
                    break;

                case FPTYPE_STREAMNAME:
                    function_parameter_add_entry(
                        fps,
                        data.cmd[data.cmdindex].argdata[argi].fpstag,
                        data.cmd[data.cmdindex].argdata[argi].descr,
                        FPTYPE_STREAMNAME,
                        data.cmd[data.cmdindex].argdata[argi].fpflag,
                        data.cmd[data.cmdindex].argdata[argi].val.s,
                        NULL);
                    NBarg_processed++;
                    break;

                case FPTYPE_FILENAME:
                    function_parameter_add_entry(
                        fps,
                        data.cmd[data.cmdindex].argdata[argi].fpstag,
                        data.cmd[data.cmdindex].argdata[argi].descr,
                        FPTYPE_FILENAME,
                        data.cmd[data.cmdindex].argdata[argi].fpflag,
                        data.cmd[data.cmdindex].argdata[argi].val.s,
                        NULL);
                    NBarg_processed++;
                    break;

                case FPTYPE_FITSFILENAME:
                    function_parameter_add_entry(
                        fps,
                        data.cmd[data.cmdindex].argdata[argi].fpstag,
                        data.cmd[data.cmdindex].argdata[argi].descr,
                        FPTYPE_FITSFILENAME,
                        data.cmd[data.cmdindex].argdata[argi].fpflag,
                        data.cmd[data.cmdindex].argdata[argi].val.s,
                        NULL);
                    NBarg_processed++;
                    break;

                case FPTYPE_FPSNAME:
                    //printf("ADDING FPS ENTRY %s at index %d\n", data.cmd[data.cmdindex].argdata[argi].fpstag, argi);
                    function_parameter_add_entry(
                        fps,
                        data.cmd[data.cmdindex].argdata[argi].fpstag,
                        data.cmd[data.cmdindex].argdata[argi].descr,
                        FPTYPE_FPSNAME,
                        FPFLAG_DEFAULT_INPUT | FPFLAG_FPS_RUN_REQUIRED,
                        data.cmd[data.cmdindex].argdata[argi].val.s,
                        &fpi);
                    //printf("fpi = %ld\n", fpi);
                    fps->parray[fpi].info.fps.FPSNBparamMAX = 0;
                    NBarg_processed++;
                    break;
            }
    }

    DEBUG_TRACE_FEXIT();
    return NBarg_processed;
}


/** @brief get FPS pointer to function argument/parameter
 */
void *get_farg_ptr(
    char *tag,
    long *fpsi
)
{
    DEBUG_TRACE_FSTART();

    void *ptr = NULL;

    DEBUG_TRACEPOINT("looking for pointer %s", tag);
    DEBUG_TRACEPOINT("FPS_CMDCODE = %d", dcfpscode);

    if(dcfpscode != 0)
    {
        // look for pointer in FPS
        // We use INT64 type as default
        ptr = (void *) functionparameter_GetParamPtr_generic(dcfpsptr,
                tag,
                fpsi);
    }
    else
    {
        for(int argi = 0; argi < data.cmd[data.cmdindex].nbparam; argi++)
        {
            if(strcmp(data.cmd[data.cmdindex].argdata[argi].fpstag, tag) == 0)
            {
                ptr = (void *)(&data.cmd[data.cmdindex].argdata[argi].val);
                break;
            }
        }
    }
    DEBUG_TRACEPOINT("found pointer");

    DEBUG_TRACE_FEXIT();
    return ptr;
}

/** @brief get FPS arguments from command line function call
 *
 * This function is intended to be used when running from the CLI.
 * For standalone applications, arguments are parsed in the main() function.
 *
 * This function has been moved from libfps to CLIcore_checkargs.c
 * to have native access to CLIcore data structures.
 */
errno_t function_parameter_getFPSargs_from_CLIfunc(char *fpsname_default)
{
    DEBUG_TRACE_FSTART();

    int  INIT_MODE __attribute__((unused)) = 0;
    dcfpscode = 0;
    char FPS_name[STRINGMAXLEN_FPS_NAME];
    int FPS_name_set = 0;

    // Initialize FPS_name to default
    if (fpsname_default != NULL) {
        strncpy(FPS_name, fpsname_default, STRINGMAXLEN_FPS_NAME - 1);
        FPS_name[STRINGMAXLEN_FPS_NAME - 1] = '\0';
    } else {
        FPS_name[0] = '\0';
    }

    // Check if the command itself has colon-separated syntax: cmdkey:fpsname:action
    char *cmd_str = data.cmdargtoken[0].val.string;
    char *first_colon = strchr(cmd_str, ':');
    if (first_colon != NULL) {
        char *second_colon = strchr(first_colon + 1, ':');
        
        // Extract FPS name if present between first and second colon, or after first colon
        if (first_colon[1] != '\0' && first_colon[1] != ':') {
            size_t fps_len = second_colon ? (size_t)(second_colon - (first_colon + 1)) : strlen(first_colon + 1);
            if (fps_len > 0 && fps_len < STRINGMAXLEN_FPS_NAME) {
                strncpy(FPS_name, first_colon + 1, fps_len);
                FPS_name[fps_len] = '\0';
                FPS_name_set = 1;
            }
        }


        // Extract action if present after second colon
        if (second_colon != NULL && second_colon[1] != '\0') {
            char *action = second_colon + 1;
            if (strcmp(action, "init") == 0) {
                dcfpscode = FPSCMDCODE_FPSINIT;
                data.cmd[data.cmdindex].cmdsettings.flags &= ~CLICMDFLAG_PROCINFO;
            } else if (strcmp(action, "initp") == 0) {
                dcfpscode = FPSCMDCODE_FPSINIT;
                data.cmd[data.cmdindex].cmdsettings.flags |= CLICMDFLAG_PROCINFO;
            } else if (strcmp(action, "?") == 0) {
                // Ignore the command so it doesn't run, but allow FPS args check
                dcfpscode = FPSCMDCODE_IGNORE;
                // Print the FPS parameters by connecting to it
                FUNCTION_PARAMETER_STRUCT tmp_fps;
                if (function_parameter_struct_connect(FPS_name, &tmp_fps, FPSCONNECT_SIMPLE) == -1) {
                    printf("FPS %s does not exist.\n", FPS_name);
                } else {
                    function_parameter_print_info(&tmp_fps, 0, 0);
                    function_parameter_struct_disconnect(&tmp_fps);
                }
            }
        }
    }

    strncpy(dcfpsname, FPS_name, STRINGMAXLEN_FPS_NAME - 1);
    dcfpsname[STRINGMAXLEN_FPS_NAME - 1] = '\0';

    // Read FPS interface from args
    if (dcfpscode == 0) {
        // set to 0 as default if we didn't extract an action above
        dcfpscode = 0;
    }


    int printinfo __attribute__((unused)) = 0;
    // if using FPS implementation, FPSCMDCODE will be set to != 0
    DEBUG_TRACEPOINT("pre-processing CLI arg");

    // by default, pre-process argument
    int argpreprocess = 1;

    if (data.cmdNBarg < 2) {
        return RETURN_SUCCESS;
    }

    switch(data.cmdargtoken[1].type)
    {
        case CLIARG_FLOAT32:
        case CLIARG_FLOAT64:
        case CLIARG_INT32:
        case CLIARG_UINT32:
        case CLIARG_INT64:
        case CLIARG_UINT64:
            argpreprocess = 0;
            break;
    }

    if(argpreprocess == 1)
    {
        // modify function attribute

        if(strcmp(data.cmdargtoken[1].val.string, "..procinfo") == 0)
        {
            if(data.cmdargtoken[2].val.numl == 0)
            {
                printf("Command %ld: updating PROCINFO mode OFF\n",
                       data.cmdindex);
                data.cmd[data.cmdindex].cmdsettings.flags &=
                    ~CLICMDFLAG_PROCINFO;
            }
            else
            {
                printf("Command %ld: updating PROCINFO mode ON\n",
                       data.cmdindex);
                data.cmd[data.cmdindex].cmdsettings.flags |=
                    CLICMDFLAG_PROCINFO;
            }
            dcfpscode = FPSCMDCODE_IGNORE;
            return RETURN_SUCCESS;
        }

        if(strcmp(data.cmdargtoken[1].val.string, "..RTprio") == 0)
        {
            printf("Command %ld: updating RTprio to value %ld\n",
                   data.cmdindex,
                   data.cmdargtoken[2].val.numl);
            data.cmd[data.cmdindex].cmdsettings.RT_priority =
                data.cmdargtoken[2].val.numl;
            dcfpscode = FPSCMDCODE_IGNORE;
            return RETURN_SUCCESS;
        }

        if(strcmp(data.cmdargtoken[1].val.string, "..loopcntMax") == 0)
        {
            printf("Command %ld: updating loopcntMax to value %ld\n",
                   data.cmdindex,
                   data.cmdargtoken[2].val.numl);
            data.cmd[data.cmdindex].cmdsettings.procinfo_loopcntMax =
                data.cmdargtoken[2].val.numl;
            dcfpscode = FPSCMDCODE_IGNORE;
            return RETURN_SUCCESS;
        }

        if(strcmp(data.cmdargtoken[1].val.string, "..triggermode") == 0)
        {
            printf("Command %ld: updating triggermode to value %ld\n",
                   data.cmdindex,
                   data.cmdargtoken[2].val.numl);
            data.cmd[data.cmdindex].cmdsettings.triggermode =
                data.cmdargtoken[2].val.numl;
            dcfpscode = FPSCMDCODE_IGNORE;
            return RETURN_SUCCESS;
        }

        if(strcmp(data.cmdargtoken[1].val.string, "..triggersname") == 0)
        {
            printf("Command %ld: updating triggerstreamname to value %s\n",
                   data.cmdindex,
                   data.cmdargtoken[2].val.string);
            strcpy(data.cmd[data.cmdindex].cmdsettings.triggerstreamname,
                   data.cmdargtoken[2].val.string);
            dcfpscode = FPSCMDCODE_IGNORE;
            return RETURN_SUCCESS;
        }

        if(strcmp(data.cmdargtoken[1].val.string, "..semindexrequested") == 0)
        {
            printf("Command %ld: updating semindexrequested to value %ld\n",
                   data.cmdindex,
                   data.cmdargtoken[2].val.numl);
            data.cmd[data.cmdindex].cmdsettings.semindexrequested =
                   data.cmdargtoken[2].val.numl;
            dcfpscode = FPSCMDCODE_IGNORE;
            return RETURN_SUCCESS;
        }


        if(strcmp(data.cmdargtoken[1].val.string, "..triggerdelay") == 0)
        {
            double x = 0.0;
            switch(data.cmdargtoken[2].type)
            {
                case CMDARGTOKEN_TYPE_FLOAT:
                    x = data.cmdargtoken[2].val.numf;
                    break;

                case CMDARGTOKEN_TYPE_LONG:
                    x = data.cmdargtoken[2].val.numl;
                    break;

                default:
                    printf(
                        "wrong argument type, should be float or int "
                        "-> setting to zero\n");
            }
            printf("Command %ld: updating triggerdelay to value %f\n",
                   data.cmdindex,
                   x);
            x += 0.5e-9;
            long x_sec  = (long) x;
            long x_nsec = (x - x_sec) * 1000000000L;

            data.cmd[data.cmdindex].cmdsettings.triggerdelay.tv_sec  = x_sec;
            data.cmd[data.cmdindex].cmdsettings.triggerdelay.tv_nsec = x_nsec;
            dcfpscode = FPSCMDCODE_IGNORE;
            return RETURN_SUCCESS;
        }

        if(strcmp(data.cmdargtoken[1].val.string, "..triggertimeout") == 0)
        {
            printf("Command %ld: updating triggertimeout to value %f\n",
                   data.cmdindex,
                   data.cmdargtoken[2].val.numf);
            double x = data.cmdargtoken[2].val.numf;
            x += 0.5e-9;
            long x_sec  = (long) x;
            long x_nsec = (x - x_sec) * 1000000000L;

            data.cmd[data.cmdindex].cmdsettings.triggertimeout.tv_sec  = x_sec;
            data.cmd[data.cmdindex].cmdsettings.triggertimeout.tv_nsec = x_nsec;
            dcfpscode = FPSCMDCODE_IGNORE;
            return RETURN_SUCCESS;
        }

        // check if recognized FPSCMDCODE
        if(strcmp(data.cmdargtoken[1].val.string,
                  "_FPSINIT_") == 0) // Initialize FPS
        {
            dcfpscode = FPSCMDCODE_FPSINIT;
        }
        else if(strcmp(data.cmdargtoken[1].val.string,
                       "_CONFSTART_") == 0) // Start conf process
        {
            dcfpscode = FPSCMDCODE_CONFSTART;
        }
        else if(strcmp(data.cmdargtoken[1].val.string,
                       "_CONFSTOP_") == 0) // Stop conf process
        {
            dcfpscode = FPSCMDCODE_CONFSTOP;
        }
        else if(strcmp(data.cmdargtoken[1].val.string,
                       "_RUNSTART_") == 0) // Run process
        {
            dcfpscode = FPSCMDCODE_RUNSTART;
        }
        else if(strcmp(data.cmdargtoken[1].val.string,
                       "_RUNSTOP_") == 0) // Stop process
        {
            dcfpscode = FPSCMDCODE_RUNSTOP;
        }
        else if(strcmp(data.cmdargtoken[1].val.string,
                       "_TMUXSTART_") == 0) // Start tmux session
        {
            dcfpscode = FPSCMDCODE_TMUXSTART;
        }
        else if(strcmp(data.cmdargtoken[1].val.string,
                       "_TMUXSTOP_") == 0) // Stop tmux session
        {
            dcfpscode = FPSCMDCODE_TMUXSTOP;
        }

    }


    // if recognized FPSCMDCODE, use FPS implementation
    if((dcfpscode != 0) && (dcfpscode != FPSCMDCODE_IGNORE))
    {
        // ===============================
        //     SET FPS INTERFACE NAME
        // ===============================

        // if main CLI process has been named with -n option, then use the process name to construct fpsname
        if(FPS_name_set == 0)
        {
            if(data.processnameflag == 1)
            {
                // Automatically set fps name to be process name up to first instance of character '.'
                strcpy(FPS_name, data.processname0);
            }
            else // otherwise, construct name as follows
            {
                // Adopt default name for fpsname
                int slen = snprintf(FPS_name,
                                    STRINGMAXLEN_FPS_NAME,
                                    "%s",
                                    fpsname_default);
                if(slen < 1)
                {
                    PRINT_ERROR("snprintf wrote <1 char");
                    abort(); // can't handle this error any other way
                }
                if(slen >= STRINGMAXLEN_FPS_NAME)
                {
                    PRINT_ERROR(
                        "snprintf string truncation.\n"
                        "Full string  : %s\n"
                        "Truncated to : %s",
                        fpsname_default,
                        FPS_name);
                    abort(); // can't handle this error any other way
                }
            }
        }

    }

    // Always keep dcfpsname updated
    strncpy(dcfpsname, FPS_name, STRINGMAXLEN_FPS_NAME - 1);
    dcfpsname[STRINGMAXLEN_FPS_NAME - 1] = '\0';

    // if recognized FPSCMDCODE, use FPS implementation
    if((dcfpscode != 0) && (dcfpscode != FPSCMDCODE_IGNORE))
    {
        // By convention, if there are optional arguments,
        // they should be appended to the default fps name
        //
        int argindex = 2; // start at arg #2
        while(data.cmdargtoken[argindex].type != CMDARGTOKEN_TYPE_UNSOLVED && strlen(data.cmdargtoken[argindex].val.string) > 0)
        {
            if (data.cmdargtoken[1].type == CMDARGTOKEN_TYPE_UNSOLVED || 
                (strcmp(data.cmdargtoken[1].val.string, "_FPSINIT_") != 0 &&
                 strcmp(data.cmdargtoken[1].val.string, "_CONFSTART_") != 0 &&
                 strcmp(data.cmdargtoken[1].val.string, "_CONFSTOP_") != 0 &&
                 strcmp(data.cmdargtoken[1].val.string, "_RUNSTART_") != 0 &&
                 strcmp(data.cmdargtoken[1].val.string, "_RUNSTOP_") != 0 &&
                 strcmp(data.cmdargtoken[1].val.string, "_TMUXSTART_") != 0 &&
                 strcmp(data.cmdargtoken[1].val.string, "_TMUXSTOP_") != 0)) {
                break; // Don't append regular arguments to the FPS name
            }

            char fpsname1[STRINGMAXLEN_FPS_NAME];

            int slen = snprintf(fpsname1,
                                STRINGMAXLEN_FPS_NAME,
                                "%s-%s",
                                FPS_name,
                                data.cmdargtoken[argindex].val.string);
            if(slen < 1)
            {
                PRINT_ERROR("snprintf wrote <1 char");
                abort(); // can't handle this error any other way
            }
            if(slen >= STRINGMAXLEN_FPS_NAME)
            {
                PRINT_ERROR(
                    "snprintf string truncation.\n"
                    "Full string  : %s-%s\n"
                    "Truncated to : %s",
                    FPS_name,
                    data.cmdargtoken[argindex].val.string,
                    fpsname1);
                abort(); // can't handle this error any other way
            }

            strncpy(FPS_name,
                    fpsname1,
                    STRINGMAXLEN_FPS_NAME - 1);
            argindex++;
        }
    }

    return RETURN_SUCCESS;
}
