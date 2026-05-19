
#ifndef PYPROCESSINFO_H
#define PYPROCESSINFO_H

extern "C"
{
#include "CLIcore.h"
#include "CLIcore/CLIcore_datainit.h"
#include "processinfo.h"
#include "processtools.h"
}

class pyProcessInfo
{
    PROCESSINFO *m_pinfo;
    int          m_fd;

  public:
    /**
    * @brief Construct a empty Process Info object
    *
    */
    pyProcessInfo() : m_pinfo(nullptr) {}

    /**
    * @brief Construct a new Process Info object
    *
    * @param pname : name of the Process Info object (human-readable)
    * @param CTRLval : control value to be externally written.
    *                 - 0: run                     (default)
    *                 - 1: pause
    *                 - 2: increment single step (will go back to 1)
    *                 - 3: exit loop
    */
    pyProcessInfo(char *pname, int CTRLval)
    {
        create(pname, CTRLval);
    }

    /**
    * @brief Destroy the Process Info object
    *
    */
    ~pyProcessInfo()
    {
        if(m_pinfo != nullptr)
        {
            processinfo_cleanExit(m_pinfo);
            m_pinfo = nullptr;
        }
    }

    PROCESSINFO *operator->()
    {
        return m_pinfo;
    }

    /**
    * @brief Create Process Info object in shared memory
    *
    * The structure holds real-time information about a process, so its status can be monitored and controlled
    * See structure PROCESSINFO in CLLIcore.h for details
    *
    * @param pname : name of the Process Info object (human-readable)
    * @param CTRLval : control value to be externally written.
    *                 - 0: run                     (default)
    *                 - 1: pause
    *                 - 2: increment single step (will go back to 1)
    *                 - 3: exit loop
    * @return int : error code
    */
    int create(char *pname, int CTRLval)
    {
        // if (m_pinfo != nullptr) {
        //   processinfo_cleanExit(m_pinfo);
        // }
        m_pinfo           = processinfo_shm_create(pname, CTRLval);
        m_pinfo->loopstat = 0; // loop initialization
        strcpy(m_pinfo->source_FUNCTION, "\0");
        strcpy(m_pinfo->source_FILE, "\0");
        m_pinfo->source_LINE = 0;
        strcpy(m_pinfo->description, "\0");
        writeMessage("initialization done");
        return EXIT_SUCCESS;
    }

    /**
    * @brief Link an existing Process Info object in shared memory
    *
    * The structure holds real-time information about a process, so its status can be monitored and controlled
    * See structure PROCESSINFO in CLLIcore.h for details
    *
    * @param pname : name of the Process Info object (human-readable)
    * @return int : error code
    */
    int link(char *pname)
    {
        m_pinfo = processinfo_shm_link(pname, &m_fd);
        return EXIT_SUCCESS;
    }

    /**
    * @brief Close an existing Process Info object in shared memory
    *
    * @param pname : name of the Process Info object (human-readable)
    * @return int : error code
    */
    int close(char *pname)
    {
        processinfo_shm_close(m_pinfo, m_fd);
        return EXIT_SUCCESS;
    }

    /**
    * @brief Send a signal to a Process Info object
    *
    * @param sig : signal to send
    * @return int : error code
    */
    int sigexit(int sig)
    {
        if(m_pinfo != nullptr)
        {
            int ret = processinfo_SIGexit(m_pinfo, sig);
            m_pinfo = nullptr;
            return ret;
        }
        return EXIT_FAILURE;
    }

    /**
    * @brief Write a message into a Process Info object
    *
    * @param message : message to write
    * @return int : error code
    */
    int writeMessage(const char *message)
    {
        if(m_pinfo != nullptr)
        {
            return processinfo_WriteMessage(m_pinfo, message);
        }
        return EXIT_FAILURE;
    };

    /**
    * @brief Define the start of the process (timing purpose)
    *
    * @return int : error code
    */
    int exec_start()
    {
        if((m_pinfo != nullptr) && (m_pinfo->MeasureTiming == 1))
        {
            return processinfo_exec_start(m_pinfo);
        }
        return EXIT_FAILURE;
    };

    /**
    * @brief Define the end of the process (timing purpose)
    *
    * @return int : error code
    */
    int exec_end()
    {
        if((m_pinfo != nullptr) && (m_pinfo->MeasureTiming == 1))
        {
            return processinfo_exec_end(m_pinfo);
        }
        return EXIT_FAILURE;
    };
};

#endif
