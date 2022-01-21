#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pyFps.hpp"
#include "pyProcessInfo.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

extern "C"
{
    DATA __attribute__((used)) data;
}

int fps_value_to_key(pyFps             &cls,
                     const std::string &key,
                     const FPS_type     fps_type,
                     py::object         value)
{
    switch (fps_type)
    {
    case FPS_type::INT32:
    case FPS_type::UINT32:
    case FPS_type::INT64:
    case FPS_type::UINT64:
        return functionparameter_SetParamValue_INT64(cls,
                                                     key.c_str(),
                                                     py::int_(value));
    case FPS_type::FLOAT32:
        return functionparameter_SetParamValue_FLOAT32(cls,
                                                       key.c_str(),
                                                       py::float_(value));
    case FPS_type::FLOAT64:
        return functionparameter_SetParamValue_FLOAT64(cls,
                                                       key.c_str(),
                                                       py::float_(value));
    case FPS_type::STRING:
        return functionparameter_SetParamValue_STRING(
            cls,
            key.c_str(),
            std::string(py::str(value)).c_str());
    default:
        return EXIT_FAILURE;
    }
}

py::object
fps_value_from_key(pyFps &cls, const std::string &key, const FPS_type fps_type)
{
    switch (fps_type)
    {
    case FPS_type::INT32:
    case FPS_type::UINT32:
    case FPS_type::INT64:
    case FPS_type::UINT64:
        return py::int_(
            functionparameter_GetParamValue_INT64(cls, key.c_str()));
    case FPS_type::FLOAT32:
        return py::float_(
            functionparameter_GetParamValue_FLOAT32(cls, key.c_str()));
    case FPS_type::FLOAT64:
        return py::float_(
            functionparameter_GetParamValue_FLOAT64(cls, key.c_str()));
    case FPS_type::STRING:
        return py::str(functionparameter_GetParamPtr_STRING(cls, key.c_str()));
    default:
        return py::none();
    }
}

py::dict fps_to_dict(pyFps &cls)
{
    py::dict fps_dict;
    for (auto &key : cls.keys())
    {
        fps_dict[py::str(key.first)] =
            fps_value_from_key(cls, key.first, key.second);
    }
    return fps_dict;
}

PYBIND11_MODULE(CacaoProcessTools, m)
{
    m.doc() = "CacaoProcessTools library module";

    CLI_data_init();
    //   m.attr("data") = &data;

    m.def("processCTRL",
          &processinfo_CTRLscreen,
          R"pbdoc(Open the process control monitor
)pbdoc");

    m.def("streamCTRL",
          &streamCTRL_CTRLscreen,
          R"pbdoc(Open the stream control monitor
)pbdoc");

    m.def("fparamCTRL",
          &functionparameter_CTRLscreen,
          R"pbdoc(Open the function parameter monitor
)pbdoc");

    py::enum_<FPS_status>(m, "FPS_status")
        .value("CONF", FPS_status::CONF)
        .value("RUN", FPS_status::RUN)
        .value("CMDCONF", FPS_status::CMDCONF)
        .value("CMDRUN", FPS_status::CMDRUN)
        .value("RUNLOOP", FPS_status::RUNLOOP)
        .value("CHECKOK", FPS_status::CHECKOK)
        .value("TMUXCONF", FPS_status::TMUXCONF)
        .value("TMUXRUN", FPS_status::TMUXRUN)
        .value("TMUXCTRL", FPS_status::TMUXCTRL)
        .export_values();

    py::enum_<FPS_type>(m, "FPS_type")
        .value("AUTO", FPS_type::AUTO)
        .value("UNDEF", FPS_type::UNDEF)
        .value("INT32", FPS_type::INT32)
        .value("UINT32", FPS_type::UINT32)
        .value("INT64", FPS_type::INT64)
        .value("UINT64", FPS_type::UINT64)
        .value("FLOAT32", FPS_type::FLOAT32)
        .value("FLOAT64", FPS_type::FLOAT64)
        .value("PID", FPS_type::PID)
        .value("TIMESPEC", FPS_type::TIMESPEC)
        .value("FILENAME", FPS_type::FILENAME)
        .value("FITSFILENAME", FPS_type::FITSFILENAME)
        .value("EXECFILENAME", FPS_type::EXECFILENAME)
        .value("DIRNAME", FPS_type::DIRNAME)
        .value("STREAMNAME", FPS_type::STREAMNAME)
        .value("STRING", FPS_type::STRING)
        .value("ONOFF", FPS_type::ONOFF)
        .value("PROCESS", FPS_type::PROCESS)
        .value("FPSNAME", FPS_type::FPSNAME)
        .export_values();

    py::class_<timespec>(m, "timespec")
        .def(py::init<time_t, long>())
        .def_readwrite("tv_sec", &timespec::tv_sec)
        .def_readwrite("tv_nsec", &timespec::tv_nsec);

    py::class_<pyProcessInfo>(m, "processinfo")
        .def(py::init<>(),
             R"pbdoc(Construct a empty Process Info object
)pbdoc")

        .def(py::init<char *, int>(),
             R"pbdoc(Construct a new Process Info object

Parameters:
    pname : name of the Process Info object (human-readable)
    CTRLval : control value to be externally written.
             - 0: run                     (default)
             - 1: pause
             - 2: increment single step (will go back to 1)
             - 3: exit loop
)pbdoc",
             py::arg("name"),
             py::arg("CTRLval"))

        .def("create",
             &pyProcessInfo::create,
             R"pbdoc(Create Process Info object in shared memory

Parameters:
    pname : name of the Process Info object (human-readable)
    CTRLval : control value to be externally written.
             - 0: run                     (default)
             - 1: pause
             - 2: increment single step (will go back to 1)
             - 3: exit loop
Return:
    ret : error code
)pbdoc",
             py::arg("name"),
             py::arg("CTRLval"))

        .def("link",
             &pyProcessInfo::link,
             R"pbdoc(Link an existing Process Info object in shared memory

Parameters:
    pname : name of the Process Info object (human-readable)
Return:
    ret : error code
)pbdoc",
             py::arg("name"))

        .def("close",
             &pyProcessInfo::close,
             R"pbdoc(Close an existing Process Info object in shared memory

Parameters:
    pname : name of the Process Info object (human-readable)
Return:
    ret : error code
)pbdoc",
             py::arg("name"))

        .def("sigexit",
             &pyProcessInfo::sigexit,
             R"pbdoc(Send a signal to a Process Info object

Parameters:
    sig : signal to send
)pbdoc",
             py::arg("sig"))

        .def("writeMessage",
             &pyProcessInfo::writeMessage,
             R"pbdoc(Write a message into a Process Info object

Parameters:
    message : message to write
Return:
    ret : error code
)pbdoc",
             py::arg("message"))

        .def("exec_start",
             &pyProcessInfo::exec_start,
             R"pbdoc(Define the start of the process (timing purpose)

Return:
    ret : error code
)pbdoc")

        .def("exec_end",
             &pyProcessInfo::exec_end,
             R"pbdoc(Define the end of the process (timing purpose)

Return:
    ret : error code
)pbdoc")

        .def_property(
            "name",
            [](pyProcessInfo &p) {
                return std::string(p->name);
            },
            [](pyProcessInfo &p, std::string name) {
                strncpy(p->name, name.c_str(), sizeof(p->name));
            },
            "Name of the Process Info object")

        .def_property(
            "source_FUNCTION",
            [](pyProcessInfo &p) {
                return std::string(p->source_FUNCTION);
            },
            [](pyProcessInfo &p, std::string source_FUNCTION) {
                strncpy(p->source_FUNCTION,
                        source_FUNCTION.c_str(),
                        sizeof(p->source_FUNCTION));
            })

        .def_property(
            "source_FILE",
            [](pyProcessInfo &p) {
                return std::string(p->source_FILE);
            },
            [](pyProcessInfo &p, std::string source_FILE) {
                strncpy(p->source_FILE,
                        source_FILE.c_str(),
                        sizeof(p->source_FILE));
            })
        .def_property(
            "source_LINE",
            [](pyProcessInfo &p) {
                return p->source_LINE;
            },
            [](pyProcessInfo &p, int source_LINE) {
                p->source_LINE = source_LINE;
            })
        .def_property(
            "PID",
            [](pyProcessInfo &p) {
                return p->PID;
            },
            [](pyProcessInfo &p, int PID) {
                p->PID = PID;
            })

        .def_property(
            "createtime",
            [](pyProcessInfo &p) {
                return p->createtime;
            },
            [](pyProcessInfo &p, timespec createtime) {
                p->createtime = createtime;
            })
        .def_property(
            "loopcnt",
            [](pyProcessInfo &p) {
                return p->loopcnt;
            },
            [](pyProcessInfo &p, int loopcnt) {
                p->loopcnt = loopcnt;
            })
        .def_property(
            "CTRLval",
            [](pyProcessInfo &p) {
                return p->CTRLval;
            },
            [](pyProcessInfo &p, int CTRLval) {
                p->CTRLval = CTRLval;
            })
        .def_property(
            "tmuxname",
            [](pyProcessInfo &p) {
                return std::string(p->tmuxname);
            },
            [](pyProcessInfo &p, std::string name) {
                strncpy(p->tmuxname, name.c_str(), sizeof(p->tmuxname));
            })
        .def_property(
            "loopstat",
            [](pyProcessInfo &p) {
                return p->loopstat;
            },
            [](pyProcessInfo &p, int loopstat) {
                p->loopstat = loopstat;
            })
        .def_property(
            "statusmsg",
            [](pyProcessInfo &p) {
                return std::string(p->statusmsg);
            },
            [](pyProcessInfo &p, std::string name) {
                strncpy(p->statusmsg, name.c_str(), sizeof(p->statusmsg));
            })
        .def_property(
            "statuscode",
            [](pyProcessInfo &p) {
                return p->statuscode;
            },
            [](pyProcessInfo &p, int statuscode) {
                p->statuscode = statuscode;
            })
        // .def_readwrite("logFile", &pyProcessInfo::m_pinfo::logFile)
        .def_property(
            "description",
            [](pyProcessInfo &p) {
                return std::string(p->description);
            },
            [](pyProcessInfo &p, std::string name) {
                strncpy(p->description, name.c_str(), sizeof(p->description));
            });

    py::class_<pyFps>(m, "fps")
        // read-only constructor
        .def(py::init<std::string>(),
             R"pbdoc(Read / connect to existing shared memory FPS
Parameters:
    name     [in]:  the name of the shared memory file to connect
)pbdoc",
             py::arg("name"))

        // read/write constructor
        .def(py::init<std::string, bool, int>(),
             R"pbdoc(Read / connect to existing shared memory FPS
Parameters:
    name     [in]:  the name of the shared memory file to connect
    create   [in]:  flag for creating of shared memory identifier
    NBparamMAX   [in]:  Max number of parameters
)pbdoc",
             py::arg("name"),
             py::arg("create"),
             py::arg("NBparamMAX") = FUNCTION_PARAMETER_NBPARAM_DEFAULT)

        .def("asdict", &fps_to_dict)

        .def("__getitem__",
             [](pyFps &cls, const std::string &key) {
                 return fps_value_from_key(cls, key, cls.keys(key));
             })

        .def("__setitem__",
             [](pyFps &cls, const std::string &key, py::object value) {
                 return fps_value_to_key(cls, key, cls.keys(key), value);
             })

        .def("md", &pyFps::md, py::return_value_policy::reference)

        .def("add_entry",
             &pyFps::add_entry,
             R"pbdoc(Add parameter to database with default settings

If entry already exists, do not modify it

Parameters:
    name     [in]:  the name of the shared memory file to connect
    create   [in]:  flag for creating of shared memory identifier
    NBparamMAX   [in]:  Max number of parameters
)pbdoc",
             py::arg("entry_name"),
             py::arg("entry_desc"),
             py::arg("fptype"))

        .def("get_status", &pyFps::get_status)
        .def("set_status", &pyFps::set_status)

        .def("get_levelKeys", &pyFps::get_levelKeys)

        .def(
            "get_param_value_int",
            [](pyFps &cls, std::string key) {
                return functionparameter_GetParamValue_INT64(cls, key.c_str());
            },
            R"pbdoc(Get the int64 value of the FPS key

Parameters:
    key     [in]: Parameter name
Return:
    ret      [out]: parameter value
)pbdoc",
            py::arg("key"))

        .def(
            "get_param_value_float",
            [](pyFps &cls, std::string key) {
                return functionparameter_GetParamValue_FLOAT32(cls,
                                                               key.c_str());
            },
            R"pbdoc(Get the float32 value of the FPS key

Parameters:
    key     [in]: Parameter name
Return:
    ret      [out]: parameter value
)pbdoc",
            py::arg("key"))
        .def(
            "get_param_value_double",
            [](pyFps &cls, std::string key) {
                return functionparameter_GetParamValue_FLOAT64(cls,
                                                               key.c_str());
            },
            R"pbdoc(Get the float64 value of the FPS key

Parameters:
    key     [in]: Parameter name
Return:
    ret      [out]: parameter value
)pbdoc",
            py::arg("key"))
        .def(
            "get_param_value_string",
            [](pyFps &cls, std::string key) {
                return std::string(
                    functionparameter_GetParamPtr_STRING(cls, key.c_str()));
            },
            R"pbdoc(Get the string value of the FPS key

Parameters:
    key     [in]: Parameter name
Return:
    ret      [out]: parameter value
)pbdoc",
            py::arg("key"))

        .def(
            "set_param_value_int",
            [](pyFps &cls, std::string key, std::string value) {
                return functionparameter_SetParamValue_INT64(cls,
                                                             key.c_str(),
                                                             std::stol(value));
            },
            R"pbdoc(Set the int64 value of the FPS key

Parameters:
    key     [in]: Parameter name
    value   [in]: Parameter value
Return:
    ret      [out]: error code
)pbdoc",
            py::arg("key"),
            py::arg("value"))
        .def(
            "set_param_value_float",
            [](pyFps &cls, std::string key, std::string value) {
                return functionparameter_SetParamValue_FLOAT32(
                    cls,
                    key.c_str(),
                    std::stol(value));
            },
            R"pbdoc(Set the float32 value of the FPS key

Parameters:
    key     [in]: Parameter name
    value   [in]: Parameter value
Return:
    ret      [out]: error code
)pbdoc",
            py::arg("key"),
            py::arg("value"))
        .def(
            "set_param_value_double",
            [](pyFps &cls, std::string key, std::string value) {
                return functionparameter_SetParamValue_FLOAT64(
                    cls,
                    key.c_str(),
                    std::stol(value));
            },
            R"pbdoc(Set the float64 value of the FPS key

Parameters:
    key     [in]: Parameter name
    value   [in]: Parameter value
Return:
    ret      [out]: error code
)pbdoc",
            py::arg("key"),
            py::arg("value"))
        .def(
            "set_param_value_string",
            [](pyFps &cls, std::string key, std::string value) {
                return functionparameter_SetParamValue_STRING(cls,
                                                              key.c_str(),
                                                              value.c_str());
            },
            R"pbdoc(Set the string value of the FPS key

Parameters:
    key     [in]: Parameter name
    value   [in]: Parameter value
Return:
    ret      [out]: error code
)pbdoc",
            py::arg("key"),
            py::arg("value"))

        .def_property_readonly("keys",
                               [](pyFps &cls) {
                                   return cls.keys();
                               })

        .def("CONFstart",
             &pyFps::CONFstart,
             R"pbdoc(FPS start CONF process

Requires setup performed by milk-fpsinit, which performs the following setup
- creates the FPS shared memory
- create up tmux sessions
- create function fpsrunstart, fpsrunstop, fpsconfstart and fpsconfstop

Return:
    ret      [out]: error code
)pbdoc")

        .def("CONFstop",
             &pyFps::CONFstop,
             R"pbdoc(FPS stop CONF process

Return:
    ret      [out]: error code
)pbdoc")

        .def("FPCONFexit",
             &pyFps::FPCONFexit,
             R"pbdoc(FPS exit FPCONF process

      Return:
          ret      [out]: error code
      )pbdoc")

        //   .def("FPCONFsetup", &pyFps::FPCONFsetup,
        //        R"pbdoc(FPS setup FPCONF process

        //   Return:
        //       ret      [out]: error code
        //   )pbdoc")

        .def("FPCONFloopstep",
             &pyFps::FPCONFloopstep,
             R"pbdoc(FPS loop step FPCONF process

      Return:
          ret      [out]: error code
      )pbdoc")

        .def("RUNstart",
             &pyFps::RUNstart,
             R"pbdoc(FPS start RUN process

Requires setup performed by milk-fpsinit, which performs the following setup
- creates the FPS shared memory
- create up tmux sessions
- create function fpsrunstart, fpsrunstop, fpsconfstart and fpsconfstop

Return:
    ret      [out]: error code
)pbdoc")

        .def("RUNstop",
             &pyFps::RUNstop,
             R"pbdoc(FPS stop RUN process

 Run pre-set function fpsrunstop in tmux ctrl window

Return:
    ret      [out]: error code
)pbdoc")

        .def("RUNexit",
             &pyFps::RUNexit,
             R"pbdoc(FPS exit RUN process

Return:
    ret      [out]: error code
)pbdoc")
        // Test if CONF process is running

        .def_property_readonly(
            "CONFrunning",
            [](pyFps &cls) {
                pid_t pid = cls->md->confpid;
                if ((getpgid(pid) >= 0) && (pid > 0))
                {
                    return 1;
                }
                else // PID not active
                {
                    if (cls->md->status &
                        FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF)
                    {
                        // not clean exit
                        return -1;
                    }
                    else
                    {
                        return 0;
                    }
                }
            },
            R"pbdoc(Test if CONF process is running

Return:
    ret      [out]: CONF process is running or not
)pbdoc")

        .def_property_readonly(
            "RUNrunning",
            [](pyFps &cls) {
                pid_t pid = cls->md->runpid;
                if ((getpgid(pid) >= 0) && (pid > 0))
                {
                    return 1;
                }
                else // PID not active
                {
                    if (cls->md->status &
                        FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN)
                    {
                        // not clean exit
                        return -1;
                    }
                    else
                    {
                        return 0;
                    }
                }
            },
            R"pbdoc(Test if RUN process is running

Return:
    ret      [out]: RUN process is running or not
)pbdoc")

        ;
    py::class_<FUNCTION_PARAMETER_STRUCT_MD>(m, "FPS_md")
        // .def(py::init([]() {
        //     return std::unique_ptr<FUNCTION_PARAMETER_STRUCT_MD>(new
        //     FUNCTION_PARAMETER_STRUCT_MD());
        // }))
        .def_readonly("name", &FUNCTION_PARAMETER_STRUCT_MD::name)
        .def_readonly("description", &FUNCTION_PARAMETER_STRUCT_MD::description)
        .def_readonly("workdir", &FUNCTION_PARAMETER_STRUCT_MD::workdir)
        .def_readonly("datadir", &FUNCTION_PARAMETER_STRUCT_MD::datadir)
        .def_readonly("confdir", &FUNCTION_PARAMETER_STRUCT_MD::confdir)
        .def_readonly("sourcefname", &FUNCTION_PARAMETER_STRUCT_MD::sourcefname)
        .def_readonly("sourceline", &FUNCTION_PARAMETER_STRUCT_MD::sourceline)
        .def_readonly("pname", &FUNCTION_PARAMETER_STRUCT_MD::pname)
        .def_readonly("callprogname",
                      &FUNCTION_PARAMETER_STRUCT_MD::callprogname)
        //   .def_readonly("nameindexW",
        //  &FUNCTION_PARAMETER_STRUCT_MD::nameindexW)
        .def_readonly("NBnameindex", &FUNCTION_PARAMETER_STRUCT_MD::NBnameindex)
        .def_readonly("confpid", &FUNCTION_PARAMETER_STRUCT_MD::confpid)
        .def_readonly("confpidstarttime",
                      &FUNCTION_PARAMETER_STRUCT_MD::confpidstarttime)
        .def_readonly("runpid", &FUNCTION_PARAMETER_STRUCT_MD::runpid)
        .def_readonly("runpidstarttime",
                      &FUNCTION_PARAMETER_STRUCT_MD::runpidstarttime)
        .def_readonly("signal", &FUNCTION_PARAMETER_STRUCT_MD::signal)
        .def_readonly("confwaitus", &FUNCTION_PARAMETER_STRUCT_MD::confwaitus)
        .def_readonly("status", &FUNCTION_PARAMETER_STRUCT_MD::status)
        .def_readonly("NBparamMAX", &FUNCTION_PARAMETER_STRUCT_MD::NBparamMAX)
        //   .def_readonly("message", &FUNCTION_PARAMETER_STRUCT_MD::message)
        .def_readonly("msgpindex", &FUNCTION_PARAMETER_STRUCT_MD::msgpindex)
        .def_readonly("msgcode", &FUNCTION_PARAMETER_STRUCT_MD::msgcode)
        .def_readonly("msgcnt", &FUNCTION_PARAMETER_STRUCT_MD::msgcnt)
        .def_readonly("conferrcnt", &FUNCTION_PARAMETER_STRUCT_MD::conferrcnt);
}
