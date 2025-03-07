#ifndef PYFPS_H
#define PYFPS_H

#include <iostream>

extern "C"
{
#include "CLIcore.h"
#include "CLIcore/CLIcore_datainit.h"
#include "fps/fps_CONFstart.h"
#include "fps/fps_CONFstop.h"
#include "fps/fps_RUNstart.h"
#include "fps/fps_RUNstop.h"
#include "fps/fps_tmux.h"
#include "processtools.h"
}

/**
 * @brief This code is a part of octopus module from COSMIC project
 *
 * @author: Florian Ferreira <florian.ferreira@obspm.fr>
 *
 */

enum FPS_status : uint32_t
{
    CONF     = FUNCTION_PARAMETER_STRUCT_STATUS_CONF,
    RUN      = FUNCTION_PARAMETER_STRUCT_STATUS_RUN,
    CMDCONF  = FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF,
    CMDRUN   = FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN,
    RUNLOOP  = FUNCTION_PARAMETER_STRUCT_STATUS_RUNLOOP,
    CHECKOK  = FUNCTION_PARAMETER_STRUCT_STATUS_CHECKOK,
    TMUXCONF = FUNCTION_PARAMETER_STRUCT_STATUS_TMUXCONF,
    TMUXRUN  = FUNCTION_PARAMETER_STRUCT_STATUS_TMUXRUN,
    TMUXCTRL = FUNCTION_PARAMETER_STRUCT_STATUS_TMUXCTRL
};

enum FPS_type : uint32_t
{
    AUTO         = FPTYPE_AUTO,
    UNDEF        = FPTYPE_UNDEF,
    INT32        = FPTYPE_INT32,
    UINT32       = FPTYPE_UINT32,
    INT64        = FPTYPE_INT64,
    UINT64       = FPTYPE_UINT64,
    FLOAT32      = FPTYPE_FLOAT32,
    FLOAT64      = FPTYPE_FLOAT64,
    PID          = FPTYPE_PID,
    TIMESPEC     = FPTYPE_TIMESPEC,
    FILENAME     = FPTYPE_FILENAME,
    FITSFILENAME = FPTYPE_FITSFILENAME,
    EXECFILENAME = FPTYPE_EXECFILENAME,
    DIRNAME      = FPTYPE_DIRNAME,
    STREAMNAME   = FPTYPE_STREAMNAME,
    STRING       = FPTYPE_STRING,
    ONOFF        = FPTYPE_ONOFF,
    PROCESS      = FPTYPE_PROCESS,
    FPSNAME      = FPTYPE_FPSNAME,
};

enum FPS_flags : uint64_t
{
    // There are many more
    DEFAULT_INPUT         = FPFLAG_DEFAULT_INPUT,
    DEFAULT_OUTPUT        = FPFLAG_DEFAULT_OUTPUT,
    DEFAULT_INPUT_STREAM  = FPFLAG_DEFAULT_INPUT_STREAM,
    DEFAULT_OUTPUT_STREAM = FPFLAG_DEFAULT_OUTPUT_STREAM,
    DEFAULT_STATUS        = FPFLAG_DEFAULT_STATUS,
};

class pyFps
{
    std::string name_;
    FUNCTION_PARAMETER_STRUCT fps_;
    std::map<std::string, FPS_type> keys_;

    int read_keys()
    {
        int k = 0;
        while((k < fps_.md->NBparamMAX) &&
                (fps_.parray[k].keywordfull[0] != '\0'))
        {
            int offset = strlen(fps_.parray[k].keyword[0]) + 1;
            char *key = fps_.parray[k].keywordfull + offset;
            keys_[key] = static_cast<FPS_type>(fps_.parray[k].type);
            k++;
        }

        return EXIT_SUCCESS;
    };

  public:
    /**
     * @brief Read / connect to existing shared memory FPS

     * @param name : the name of the shared memory file to connect
     */
    pyFps(std::string name)
        : pyFps(name, false, FUNCTION_PARAMETER_NBPARAM_DEFAULT) {};

    /**
     * @brief Read or create an shared FPS
     *
     * @param fps_name : the name of the shared memory file to connect
     * @param create : flag for creating of shared memory identifier
     * @param NBparamMAX : Max number of parameters
     */
    pyFps(std::string fps_name, bool create, int NBparamMAX) : name_(fps_name)
    {
        fps_.md = nullptr;
        fps_.parray = nullptr;  // array of function parameters

        // these variables are local to each process
        fps_.localstatus = 0;  // 1 if conf loop should be active
        fps_.SMfd = -1;
        fps_.CMDmode = 0;

        fps_.NBparam = 0;        // number of parameters in array
        fps_.NBparamActive = 0;  // number of active parameters

        if(create)
        {
            create_and_connect(NBparamMAX);
        }
        else
        {
            if(connect() == -1)
            {
                throw std::runtime_error("FPS does not exist");
            }
            read_keys();
        }
        //std::cout << "FPS connected" << std::endl;
    }

    /**
     * @brief Destroy the py Fps object
     *
     */
    ~pyFps()
    {
        if(fps_.SMfd != -1)
        {
            function_parameter_struct_disconnect(&fps_);
        }
    }

    FUNCTION_PARAMETER_STRUCT *operator->()
    {
        return &fps_;
    }
    operator FUNCTION_PARAMETER_STRUCT *()
    {
        return &fps_;
    }
    const FUNCTION_PARAMETER_STRUCT_MD *md() const
    {
        return fps_.md;
    }
    const std::map<std::string, FPS_type> &keys()
    {
        return keys_;
    }
    const FPS_type keys(const std::string &key)
    {
        return keys_[key];
    }

    /**
     * @brief Create a and connect object
     *
     * @param NBparamMAX
     * @return int
     */
    int create_and_connect(int NBparamMAX)
    {
        if(connect() == -1)
        {
            std::cout << "Creating FPS...";
            function_parameter_struct_create(NBparamMAX, name_.c_str());
            std::cout << "Done" << std::endl;
            connect();
        }
        else
        {
            connect();
            read_keys();
        }
        return EXIT_SUCCESS;
    }

    /**
     * @brief Connect to existing shared memory FPS

     * @param name : the name of the shared memory file to connect
     */
    int connect()
    {
        return function_parameter_struct_connect(name_.c_str(), &fps_,
                FPSCONNECT_SIMPLE);
    };

    /**
     * @brief Release the mmaped file and fd.
     */
    int disconnect()
    {
        return function_parameter_struct_disconnect(&fps_);
    }

    /**
     * @brief Add parameter to database with default settings
     *
     * If entry already exists, do not modify it
     *
     * @param entry_name : entry name
     * @param entry_desc : entry description
     * @param fptype : entry type ("int","double","float","string")
     * @return int
     */
    int add_entry(std::string entry_name, std::string entry_desc,
                  uint32_t fptype)
    {
        keys_[entry_name] = static_cast<FPS_type>(fptype);
        // We need to prepend a . to entry to abide by FPS conventions
        // See fps_add_entry.c
        std::string prepended = "." + entry_name;
        return function_parameter_add_entry(&fps_, prepended.c_str(),
                                            entry_desc.c_str(), fptype,
                                            FPFLAG_DEFAULT_INPUT, nullptr, nullptr);
    }

    /**
     * @brief Add parameter to database with generic settings
     *
     * If entry already exists, do not modify it
     *
     * @param entry_name : entry name
     * @param entry_desc : entry description
     * @param fptype : entry type ("int","double","float","string")
     * @param fpflag : entry flags
     * @return int
     */
    int add_entry_w_flags(std::string entry_name, std::string entry_desc,
                          uint32_t fptype, uint64_t fpflag)
    {
        keys_[entry_name] = static_cast<FPS_type>(fptype);
        std::string prepended = "." + entry_name;
        return function_parameter_add_entry(&fps_, prepended.c_str(),
                                            entry_desc.c_str(), fptype,
                                            fpflag, nullptr, nullptr);
    }

    /**
     * @brief Get the status object
     *
     * @return int
     */
    int get_status()
    {
        return fps_.md->status;
    }

    /**
     * @brief Set the status object
     *
     * @param status
     * @return int
     */
    int set_status(int status)
    {
        fps_.md->status = status;
        return EXIT_SUCCESS;
    }

    /**
     * @brief Get the levelKeys object
     *
     * @param level
     * @return std::vector<std::string>
     */
    std::vector<std::string> get_levelKeys(int level)
    {
        std::vector<std::string> levelKeys = std::vector<std::string>();
        int k = 0;
        while(fps_.parray[k].keywordfull[0] != '\0' && k < fps_.md->NBparamMAX)
        {
            std::string tmp = fps_.parray[k].keyword[level];
            auto exist = std::find(levelKeys.begin(), levelKeys.end(), tmp);
            if(exist == levelKeys.end())
            {
                levelKeys.push_back(tmp);
            }
            k++;
        }

        return levelKeys;
    }

    void signal_update()
    {
        fps_.md->signal |=
            FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE; // notify GUI loop to update
    }

    errno_t CONFstart()
    {
        return functionparameter_CONFstart(&fps_);
    }

    errno_t CONFstop()
    {
        return functionparameter_CONFstop(&fps_);
    }

    errno_t FPCONFexit()
    {
        return function_parameter_FPCONFexit(&fps_);
    }

    // errno_t  FPCONFsetup(){
    //   return function_parameter_FPCONFsetup(&fps_);
    // }

    errno_t FPCONFloopstep()
    {
        return function_parameter_FPCONFloopstep(&fps_);
    }

    errno_t RUNstart()
    {
        return functionparameter_RUNstart(&fps_);
    }

    errno_t RUNstop()
    {
        return functionparameter_RUNstop(&fps_);
    }

    errno_t RUNexit()
    {
        return function_parameter_RUNexit(&fps_);
    }

    errno_t TMUXstart()
    {
        return functionparameter_FPS_tmux_init(&fps_);
    }

    errno_t TMUXstop()
    {
        return functionparameter_FPS_tmux_kill(&fps_);
    }
};

#endif
