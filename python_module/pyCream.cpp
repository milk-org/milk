#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pyProcessInfo.hpp"
#include "pyFps.hpp"

namespace py = pybind11;

extern "C" {
DATA __attribute__((used)) data;
}

PYBIND11_MODULE(pyCream, m) {
  m.doc() = "pyCream library module";

  CLI_data_init();
//   m.attr("data") = &data;

  m.def("processCTRL", &processinfo_CTRLscreen,
        R"pbdoc(Open the process control monitor
)pbdoc");

  m.def("streamCTRL", &streamCTRL_CTRLscreen,
        R"pbdoc(Open the stream control monitor
)pbdoc");

  m.def("fparamCTRL", &functionparameter_CTRLscreen,
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
           py::arg("name"), py::arg("CTRLval"))

      .def("create", &pyProcessInfo::create,
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
           py::arg("name"), py::arg("CTRLval"))

      .def("link", &pyProcessInfo::link,
           R"pbdoc(Link an existing Process Info object in shared memory

Parameters:
    pname : name of the Process Info object (human-readable)
Return:
    ret : error code
)pbdoc",
           py::arg("name"))

      .def("close", &pyProcessInfo::close,
           R"pbdoc(Close an existing Process Info object in shared memory

Parameters:
    pname : name of the Process Info object (human-readable)
Return:
    ret : error code
)pbdoc",
           py::arg("name"))

      .def("sigexit", &pyProcessInfo::sigexit,
           R"pbdoc(Send a signal to a Process Info object

Parameters:
    sig : signal to send
)pbdoc",
           py::arg("sig"))

      .def("writeMessage", &pyProcessInfo::writeMessage,
           R"pbdoc(Write a message into a Process Info object

Parameters:
    message : message to write
Return:
    ret : error code
)pbdoc",
           py::arg("message"))

      .def("exec_start", &pyProcessInfo::exec_start,
           R"pbdoc(Define the start of the process (timing purpose)

Return:
    ret : error code
)pbdoc")

      .def("exec_end", &pyProcessInfo::exec_end,
           R"pbdoc(Define the end of the process (timing purpose)

Return:
    ret : error code
)pbdoc")

      .def_property(
          "name", [](pyProcessInfo &p) { return std::string(p->name); },
          [](pyProcessInfo &p, std::string name) {
            strncpy(p->name, name.c_str(), sizeof(p->name));
          },
          "Name of the Process Info object")

      .def_property(
          "source_FUNCTION",
          [](pyProcessInfo &p) { return std::string(p->source_FUNCTION); },
          [](pyProcessInfo &p, std::string source_FUNCTION) {
            strncpy(p->source_FUNCTION, source_FUNCTION.c_str(),
                    sizeof(p->source_FUNCTION));
          })

      .def_property(
          "source_FILE",
          [](pyProcessInfo &p) { return std::string(p->source_FILE); },
          [](pyProcessInfo &p, std::string source_FILE) {
            strncpy(p->source_FILE, source_FILE.c_str(),
                    sizeof(p->source_FILE));
          })
      .def_property(
          "source_LINE", [](pyProcessInfo &p) { return p->source_LINE; },
          [](pyProcessInfo &p, int source_LINE) {
            p->source_LINE = source_LINE;
          })
      .def_property(
          "PID", [](pyProcessInfo &p) { return p->PID; },
          [](pyProcessInfo &p, int PID) { p->PID = PID; })

      .def_property(
          "createtime", [](pyProcessInfo &p) { return p->createtime; },
          [](pyProcessInfo &p, timespec createtime) {
            p->createtime = createtime;
          })
      .def_property(
          "loopcnt", [](pyProcessInfo &p) { return p->loopcnt; },
          [](pyProcessInfo &p, int loopcnt) { p->loopcnt = loopcnt; })
      .def_property(
          "CTRLval", [](pyProcessInfo &p) { return p->CTRLval; },
          [](pyProcessInfo &p, int CTRLval) { p->CTRLval = CTRLval; })
      .def_property(
          "tmuxname", [](pyProcessInfo &p) { return std::string(p->tmuxname); },
          [](pyProcessInfo &p, std::string name) {
            strncpy(p->tmuxname, name.c_str(), sizeof(p->tmuxname));
          })
      .def_property(
          "loopstat", [](pyProcessInfo &p) { return p->loopstat; },
          [](pyProcessInfo &p, int loopstat) { p->loopstat = loopstat; })
      .def_property(
          "statusmsg",
          [](pyProcessInfo &p) { return std::string(p->statusmsg); },
          [](pyProcessInfo &p, std::string name) {
            strncpy(p->statusmsg, name.c_str(), sizeof(p->statusmsg));
          })
      .def_property(
          "statuscode", [](pyProcessInfo &p) { return p->statuscode; },
          [](pyProcessInfo &p, int statuscode) { p->statuscode = statuscode; })
      // .def_readwrite("logFile", &pyProcessInfo::m_pinfo::logFile)
      .def_property(
          "description",
          [](pyProcessInfo &p) { return std::string(p->description); },
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
           py::arg("name"), py::arg("create"),
           py::arg("NBparamMAX") = FUNCTION_PARAMETER_NBPARAM_DEFAULT)

      .def("md", &pyFps::md, py::return_value_policy::reference)

      .def("add_entry", &pyFps::add_entry,
           R"pbdoc(Add parameter to database with default settings

If entry already exists, do not modify it

Parameters:
    name     [in]:  the name of the shared memory file to connect
    create   [in]:  flag for creating of shared memory identifier
    NBparamMAX   [in]:  Max number of parameters
)pbdoc",
           py::arg("entry_name"), py::arg("entry_desc"), py::arg("fptype"))

      .def("get_status", &pyFps::get_status)
      .def("set_status", &pyFps::set_status)

      .def("get_levelKeys", &pyFps::get_levelKeys)

      .def(
          "get_param_value_int",
          [](pyFps &cls, std::string key) {
            return functionparameter_GetParamValue_INT64(&cls.fps, key.c_str());
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
            return functionparameter_GetParamValue_FLOAT32(&cls.fps,
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
            return functionparameter_GetParamValue_FLOAT64(&cls.fps,
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
                functionparameter_GetParamPtr_STRING(&cls.fps, key.c_str()));
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
            return functionparameter_SetParamValue_INT64(&cls.fps, key.c_str(),
                                                         std::stol(value));
          },
          R"pbdoc(Set the int64 value of the FPS key

Parameters:
    key     [in]: Parameter name
    value   [in]: Parameter value
Return:
    ret      [out]: error code
)pbdoc",
          py::arg("key"), py::arg("value"))
      .def(
          "set_param_value_float",
          [](pyFps &cls, std::string key, std::string value) {
            return functionparameter_SetParamValue_FLOAT32(
                &cls.fps, key.c_str(), std::stol(value));
          },
          R"pbdoc(Set the float32 value of the FPS key

Parameters:
    key     [in]: Parameter name
    value   [in]: Parameter value
Return:
    ret      [out]: error code
)pbdoc",
          py::arg("key"), py::arg("value"))
      .def(
          "set_param_value_double",
          [](pyFps &cls, std::string key, std::string value) {
            return functionparameter_SetParamValue_FLOAT64(
                &cls.fps, key.c_str(), std::stol(value));
          },
          R"pbdoc(Set the float64 value of the FPS key

Parameters:
    key     [in]: Parameter name
    value   [in]: Parameter value
Return:
    ret      [out]: error code
)pbdoc",
          py::arg("key"), py::arg("value"))
      .def(
          "set_param_value_string",
          [](pyFps &cls, std::string key, std::string value) {
            return functionparameter_SetParamValue_STRING(&cls.fps, key.c_str(),
                                                          value.c_str());
          },
          R"pbdoc(Set the string value of the FPS key

Parameters:
    key     [in]: Parameter name
    value   [in]: Parameter value
Return:
    ret      [out]: error code
)pbdoc",
          py::arg("key"), py::arg("value"))

      .def_property_readonly("keys", [](const pyFps &cls) { return cls.keys; })

      .def(
          "CONFstart",
          [](pyFps &cls) { return functionparameter_CONFstart(&cls.fps); },
          R"pbdoc(FPS start CONF process

Requires setup performed by milk-fpsinit, which performs the following setup
- creates the FPS shared memory
- create up tmux sessions
- create function fpsrunstart, fpsrunstop, fpsconfstart and fpsconfstop

Return:
    ret      [out]: error code
)pbdoc")

      .def(
          "CONFstop",
          [](pyFps &cls) { return functionparameter_CONFstop(&cls.fps); },
          R"pbdoc(FPS stop CONF process

Return:
    ret      [out]: error code
)pbdoc")

//       .def(
//           "CONFupdate",
//           [](pyFps &cls) { return functionparameter_CONFupdate(&cls.fps); },
//           R"pbdoc(FPS update CONF process

// Return:
//     ret      [out]: error code
// )pbdoc")

      .def(
          "RUNstart",
          [](pyFps &cls) { return functionparameter_RUNstart(&cls.fps); },
          R"pbdoc(FPS start RUN process

Requires setup performed by milk-fpsinit, which performs the following setup
- creates the FPS shared memory
- create up tmux sessions
- create function fpsrunstart, fpsrunstop, fpsconfstart and fpsconfstop

Return:
    ret      [out]: error code
)pbdoc")

      .def(
          "RUNstop",
          [](pyFps &cls) { return functionparameter_RUNstop(&cls.fps); },
          R"pbdoc(FPS stop RUN process

 Run pre-set function fpsrunstop in tmux ctrl window

Return:
    ret      [out]: error code
)pbdoc")

      .def(
          "RUNexit",
          [](pyFps &cls) { return function_parameter_RUNexit(&cls.fps); },
          R"pbdoc(FPS exit RUN process

Return:
    ret      [out]: error code
)pbdoc")
      // Test if CONF process is running

      .def_property_readonly(
          "CONFrunning",
          [](pyFps &cls) {
            pid_t pid = cls->md->confpid;
            if ((getpgid(pid) >= 0) && (pid > 0)) {
              return 1;
            } else  // PID not active
            {
              if (cls->md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF) {
                // not clean exit
                return -1;
              } else {
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
            if ((getpgid(pid) >= 0) && (pid > 0)) {
              return 1;
            } else  // PID not active
            {
              if (cls->md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN) {
                // not clean exit
                return -1;
              } else {
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
      .def_readonly("callprogname", &FUNCTION_PARAMETER_STRUCT_MD::callprogname)
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
