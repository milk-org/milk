/**
 * @file CLIcore_checkargs.c
 *
 * @brief Command line argument validation and type checking
 *
 * Architecture Overview:
 * This file verifies that the arguments provided to a CLI command match the
 * expected parameters defined by the module's `FPS_CMDDEF` table. It ensures
 * type safety for CLI interactions by checking scalars, arrays, FPS parameter
 * formats, existing files, and shared memory streams before the underlying
 * C function is invoked.
 */

#include <stdio.h>

#include "CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"
#include "fps_globals.h"

// keep processing if 1
static int argcheck_process_flag = 1;


// toggles to 1 if function help called
static int functionhelp_called = 0;


/**
 * @brief Convert a CLI argument type enum to string.
 */
static const char *CLIargtype_to_string(uint32_t type)
{
    switch(type)
    {
    case FPTYPE_FLOAT32:
        return "FLOAT32";
    case FPTYPE_FLOAT64:
        return "FLOAT64";
    case FPTYPE_ONOFF:
        return "ONOFF";
    case FPTYPE_INT32:
        return "INT32";
    case FPTYPE_UINT32:
        return "UINT32";
    case FPTYPE_INT64:
        return "INT64";
    case FPTYPE_UINT64:
        return "UINT64";
    case FPTYPE_STRING_NOT_STREAM:
        return "STR_NOT_IMG";
    case FPTYPE_STREAMNAME:
        return "STREAM";
    case FPTYPE_STRING:
        return "STRING";
    case FPTYPE_FILENAME:
        return "FILENAME";
    case FPTYPE_FITSFILENAME:
        return "FITSFILE";
    case FPTYPE_FPSNAME:
        return "FPSNAME";
    case FPTYPE_EXECFILENAME:
        return "EXECFILE";
    case FPTYPE_DIRNAME:
        return "DIRNAME";
    case FPTYPE_PID:
        return "PID";
    case FPTYPE_TIMESPEC:
        return "TIMESPEC";
    case CLIARG_MISSING:
        return "MISSING";
    default:
        return "UNKNOWN";
    }
}

/**
 * @brief Convert a command argument token type to string.
 */
static const char *CMDARGTOKEN_type_to_string(uint32_t type)
{
    switch(type)
    {
    case 0:
        return "UNSOLVED_TOKEN";
    case 1:
        return "FLOAT_TOKEN";
    case 2:
        return "INT_TOKEN";
    case 3:
        return "STRING_TOKEN";
    case 4:
        return "IMG_TOKEN";
    case 5:
        return "CMD_TOKEN";
    case 6:
        return "RAWSTRING_TOKEN";
    default:
        return CLIargtype_to_string(type);
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
    int      CLIargnum,
    uint32_t funcargtype,
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
    if(ttype == CMDARGTOKEN_TYPE_FLOAT)
    {
        data.cmdargtoken[CLIargnum].val.numl = (long)(data.cmdargtoken[CLIargnum].val.numf + 0.5);
    }
    if(ttype == CMDARGTOKEN_TYPE_LONG)
    {
        data.cmdargtoken[CLIargnum].val.numf = (double)data.cmdargtoken[CLIargnum].val.numl;
    }

    // Special conversion for ONOFF
    if(ftype == FPTYPE_ONOFF)
    {
        if(strcasecmp(data.cmdargtoken[CLIargnum].val.string, "on") == 0)
        {
            data.cmdargtoken[CLIargnum].val.numl = 1;
            data.cmdargtoken[CLIargnum].val.numf = 1.0;
            rval = 0;
        }
        else if(strcasecmp(data.cmdargtoken[CLIargnum].val.string, "off") == 0)
        {
            data.cmdargtoken[CLIargnum].val.numl = 0;
            data.cmdargtoken[CLIargnum].val.numf = 0.0;
            rval = 0;
        }
    }

    // Type matching logic
    if(rval == 2)
    {
        if(ftype == FPTYPE_FLOAT32 || ftype == FPTYPE_FLOAT64)
        {
            if(ttype == CMDARGTOKEN_TYPE_FLOAT || ttype == (CLIARG_FLOAT32 & 0x0000FFFF) ||
                    ttype == CMDARGTOKEN_TYPE_LONG || ttype == (CLIARG_INT64 & 0x0000FFFF) ||
                    ttype == 6)
            {

                if(ttype == 6)
                {
                    data.cmdargtoken[CLIargnum].val.numf = atof(data.cmdargtoken[CLIargnum].val.string);
                    data.cmdargtoken[CLIargnum].val.numl = (long)data.cmdargtoken[CLIargnum].val.numf;
                }
                rval = 0;
            }
        }
        else if(ftype == FPTYPE_INT32 || ftype == FPTYPE_INT64 || ftype == FPTYPE_UINT32
                || ftype == FPTYPE_UINT64 || ftype == FPTYPE_ONOFF || ftype == FPTYPE_PID
                || ftype == FPTYPE_TIMESPEC)
        {
            if(ttype == CMDARGTOKEN_TYPE_LONG || ttype == (CLIARG_INT64 & 0x0000FFFF) ||
                    ttype == CMDARGTOKEN_TYPE_FLOAT || ttype == (CLIARG_FLOAT32 & 0x0000FFFF) ||
                    ttype == 6)
            {

                if(ttype == 6)
                {
                    data.cmdargtoken[CLIargnum].val.numl = atol(data.cmdargtoken[CLIargnum].val.string);
                    data.cmdargtoken[CLIargnum].val.numf = (double)data.cmdargtoken[CLIargnum].val.numl;
                }
                rval = 0;
            }
        }
        else if(ftype == FPTYPE_STREAMNAME)
        {
            if(ttype == CMDARGTOKEN_TYPE_EXISTINGIMAGE || ftype == FPTYPE_STREAMNAME ||
                    ttype == CMDARGTOKEN_TYPE_STRING || ftype == FPTYPE_STRING ||
                    ttype == 6)
            {
                rval = 0;
            }
        }
        else if(ftype == FPTYPE_STRING || ftype == FPTYPE_STRING_NOT_STREAM || ftype == FPTYPE_FILENAME
                || ftype == FPTYPE_FITSFILENAME || ftype == FPTYPE_FPSNAME || ftype == FPTYPE_DIRNAME
                || ftype == FPTYPE_EXECFILENAME)
        {
            if(ttype == CMDARGTOKEN_TYPE_STRING || ftype == FPTYPE_STRING ||
                    ttype == CMDARGTOKEN_TYPE_EXISTINGIMAGE || ftype == FPTYPE_STREAMNAME ||
                    ttype == CLIARG_STR_NOT_IMG || ttype == 6)
            {
                rval = 0;
            }
        }
    }

    // Check if it's a variable if not already resolved
    if(rval == 2)
    {
        imageID IDv = variable_ID(data.cmdargtoken[CLIargnum].val.string);
        if(IDv != -1)
        {
            if(ftype == FPTYPE_FLOAT32 || ftype == FPTYPE_FLOAT64)
            {
                data.cmdargtoken[CLIargnum].val.numf = (double) dcvar[IDv].value.f;
                data.cmdargtoken[CLIargnum].val.numl = (long) data.cmdargtoken[CLIargnum].val.numf;
                data.cmdargtoken[CLIargnum].type = CLIARG_FLOAT64;
                rval = 0;
            }
            else if(ftype == FPTYPE_INT32 || ftype == FPTYPE_INT64 || ftype == FPTYPE_UINT32
                    || ftype == FPTYPE_UINT64 || ftype == FPTYPE_ONOFF)
            {
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
            printf(
                "\033[1;31mERROR\033[0m"
                " arg %d: wrong type"
                " (expected %s, got %s)\n",
                CLIargnum - 1,
                CLIargtype_to_string(funcargtype),
                CMDARGTOKEN_type_to_string(
                    data.cmdargtoken[CLIargnum]
                    .type));
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
/**
 * @brief Read one FPS parameter value into argdata.
 *
 * Dispatches on FPS parameter type and copies the
 * value from the FPS shared memory into the
 * corresponding argdata field.
 *
 * @param ptype  FPS parameter type
 * @param fp     Pointer to the FPS parameter entry
 * @param ad     Pointer to the argdata slot
 */
static void sync_fps_to_argdata(
    uint32_t ptype,
    FPS_PARAM *fp,
    CLICMDARGDATA      *ad
)
{
    switch(ptype)
    {
    case FPTYPE_FLOAT32:
        ad->val.f32 = fp->val.f32[0];
        break;
    case FPTYPE_FLOAT64:
        ad->val.f64 = fp->val.f64[0];
        break;
    case FPTYPE_INT32:
        ad->val.i32 = fp->val.i32[0];
        break;
    case FPTYPE_UINT32:
        ad->val.ui32 = fp->val.ui32[0];
        break;
    case FPTYPE_INT64:
        ad->val.i64 = fp->val.i64[0];
        break;
    case FPTYPE_UINT64:
        ad->val.ui64 = fp->val.ui64[0];
        break;
    case FPTYPE_ONOFF:
        ad->val.i64 = fp->val.i32[0];
        break;
    case FPTYPE_PID:
        ad->val.i64 =
            (int64_t) fp->val.pid[0];
        break;
    case FPTYPE_TIMESPEC:
        ad->val.f64 =
            (double) fp->val.ts[0].tv_sec
            + (double) fp->val.ts[0]
            .tv_nsec * 1e-9;
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
        strncpy(
            ad->val.s,
            fp->val.string[0],
            STRINGMAXLEN_CLICMDARG - 1);
        break;
    }
}


/**
 * @brief Write a CLI token value into an FPS parameter.
 *
 * Dispatches on FPS parameter type and calls the
 * appropriate functionparameter_SetParamValue_*
 * function using the numeric or string value from
 * the CLI token.
 *
 * @param fps     Pointer to the FPS struct
 * @param fpstag  Parameter tag/key
 * @param ptype   FPS parameter type
 * @param numl    Integer value from CLI token
 * @param numf    Float value from CLI token
 * @param str     String value from CLI token
 */
static void set_fps_from_clitoken(
    FPS        *fps,
    const char *fpstag,
    uint32_t   ptype,
    long       numl,
    double     numf,
    const char *str
)
{
    switch(ptype)
    {
    case FPTYPE_INT64:
        functionparameter_SetParamValue_INT64(
            fps, fpstag, numl);
        break;
    case FPTYPE_UINT64:
        functionparameter_SetParamValue_UINT64(
            fps, fpstag, numl);
        break;
    case FPTYPE_INT32:
        functionparameter_SetParamValue_INT32(
            fps, fpstag, numl);
        break;
    case FPTYPE_UINT32:
        functionparameter_SetParamValue_UINT32(
            fps, fpstag, numl);
        break;
    case FPTYPE_FLOAT64:
        functionparameter_SetParamValue_FLOAT64(
            fps, fpstag, numf);
        break;
    case FPTYPE_FLOAT32:
        functionparameter_SetParamValue_FLOAT32(
            fps, fpstag, (float) numf);
        break;
    case FPTYPE_PID:
        functionparameter_SetParamValue_INT64(
            fps, fpstag, (int64_t) numl);
        break;
    case FPTYPE_TIMESPEC:
        functionparameter_SetParamValue_TIMESPEC(
            fps, fpstag, numf);
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
        functionparameter_SetParamValue_STRING(
            fps, fpstag, str);
        break;
    case FPTYPE_ONOFF:
        functionparameter_SetParamValue_ONOFF(
            fps, fpstag, (int) numl);
        break;
    }
}


/**
 * @brief Validate an array of CLI arguments.
 *
 * Checks that each argument matches its expected
 * type and range.
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
                sync_fps_to_argdata(
                    ptype,
                    &dcfpsptr->parray[pindex],
                    &data.cmd[data.cmdindex]
                    .argdata[arg]
                );
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
                    printf(
                        "\n\033[1;31mERROR\033[0m"
                        " Setting parameter %s :"
                        " input missing\n",
                        data.cmdargtoken[1]
                        .val.string);
                    help_command(
                        data.cmd[data.cmdindex]
                        .key);
                    DEBUG_TRACE_FEXIT();
                    return
                        RETURN_CLICHECKARGARRAY_FAILURE;
                }

                // Update the parameter in FPS
                uint32_t ptype = dcfpsptr->parray[pindex].type;
                set_fps_from_clitoken(
                    dcfpsptr, fpstag, ptype,
                    data.cmdargtoken[2].val.numl,
                    data.cmdargtoken[2].val.numf,
                    data.cmdargtoken[2].val.string
                );

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
            printf(
                "\n\033[1;31mERROR\033[0m"
                " Setting arg %s :"
                " input missing\n",
                fpscliarg[argindexmatch].fpstag);
            help_command(
                data.cmd[data.cmdindex].key);
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
                    set_fps_from_clitoken(
                        dcfpsptr,
                        fpscliarg[argindexmatch]
                        .fpstag,
                        dcfpsptr->parray[pindex]
                        .type,
                        data.cmdargtoken[2]
                        .val.numl,
                        data.cmdargtoken[2]
                        .val.numf,
                        data.cmdargtoken[2]
                        .val.string
                    );
                }
            }
        }
        else
        {
            printf(
                "\n\033[1;31mERROR\033[0m"
                " Setting arg %s :"
                " wrong type\n",
                fpscliarg[argindexmatch].fpstag);
            help_command(
                data.cmd[data.cmdindex].key);
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
            snprintf(argtypestring,
                     sizeof(argtypestring),
                     "FLT32");
            break;
        case CLIARG_FLOAT64:
            snprintf(argtypestring,
                     sizeof(argtypestring),
                     "FLT64");
            break;
        case CLIARG_ONOFF:
            snprintf(argtypestring,
                     sizeof(argtypestring),
                     "ONOFF");
            break;
        case CLIARG_INT32:
            snprintf(argtypestring,
                     sizeof(argtypestring),
                     "INT32");
            break;
        case CLIARG_UINT32:
            snprintf(argtypestring,
                     sizeof(argtypestring),
                     "UINT32");
            break;
        case CLIARG_INT64:
            snprintf(argtypestring,
                     sizeof(argtypestring),
                     "INT64");
            break;
        case CLIARG_UINT64:
            snprintf(argtypestring,
                     sizeof(argtypestring),
                     "UINT64");
            break;
        case CLIARG_STR_NOT_IMG:
            snprintf(argtypestring,
                     sizeof(argtypestring),
                     "STRnIMG");
            break;
        case CLIARG_IMG:
            snprintf(argtypestring,
                     sizeof(argtypestring),
                     "STREAM");
            break;
        case CLIARG_STR:
            snprintf(argtypestring,
                     sizeof(argtypestring),
                     "STRING");
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
                    printf(
                        "\n\033[1;31mERROR\033[0m"
                        " Missing mandatory"
                        " argument %d (%s: %s)\n",
                        CLIarg,
                        fpscliarg[arg].fpstag,
                        fpscliarg[arg].descr);
                    help_command(
                        data.cmd[data.cmdindex]
                        .key);
                    argcheck_process_flag = 0;
                    DEBUG_TRACE_FEXIT();
                    return
                        RETURN_CLICHECKARGARRAY_FAILURE;
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
        printf(
            "\n\033[1;31mERROR\033[0m"
            " %d argument(s) have wrong"
            " type\n",
            nberr);
        help_command(
            data.cmd[data.cmdindex].key);
        DEBUG_TRACE_FEXIT();
        return RETURN_CLICHECKARGARRAY_FAILURE;
    }
}
