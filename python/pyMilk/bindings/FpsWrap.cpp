#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>

#include <fstream>
#include <string>
#include <sys/types.h>
#include <unistd.h>

#include "pyFps.hpp"

namespace nb = nanobind;
using namespace nanobind::literals;

namespace
{
// getpgid() succeeds for zombie processes (PID still valid until reaped by
// its parent), so a finished-but-unreaped process would otherwise be
// reported as still running. Check /proc/<pid>/stat for zombie state 'Z'.
bool pid_is_running(pid_t pid)
{
    if ((pid <= 0) || (getpgid(pid) < 0))
    {
        return false;
    }

    std::ifstream stat_file("/proc/" + std::to_string(pid) + "/stat");
    std::string   line;
    if (!stat_file || !std::getline(stat_file, line))
    {
        // Can't inspect state (e.g. non-Linux or /proc unavailable): fall
        // back to the getpgid()-only result.
        return true;
    }

    // Format is "pid (comm) state ...". comm may contain ')' so search from
    // the end for the closing parenthesis.
    const auto closing = line.rfind(')');
    return (closing == std::string::npos) || (line[closing + 2] != 'Z');
}
} // namespace


struct timespec double_to_timespec(double value)
{
    struct timespec tspec;
    tspec.tv_sec  = static_cast<time_t>(value);
    tspec.tv_nsec = static_cast<long>((value - tspec.tv_sec) * 1e9);
    return tspec;
}

double timespec_to_double(struct timespec tspec)
{
    return static_cast<double>(tspec.tv_sec) + static_cast<double>(tspec.tv_nsec) * 1e-9;
}


int fps_value_to_key(pyFps &cls, const std::string &key, const FPS_type fps_type, nb::object value)
{
    FPS_type switch_fps_type = fps_type == FPS_type::AUTO ? cls.keys(key) : fps_type;

    switch (switch_fps_type)
    {
    case FPS_type::ONOFF:
        return functionparameter_SetParamValue_ONOFF(cls, key.c_str(), nb::cast<bool>(value));
    case FPS_type::INT32:
    case FPS_type::PID:
    case FPS_type::INT64:
        return functionparameter_SetParamValue_INT64(cls, key.c_str(), nb::cast<int64_t>(value));
    case FPS_type::UINT32:
    case FPS_type::UINT64:
        return functionparameter_SetParamValue_UINT64(cls, key.c_str(), nb::cast<uint64_t>(value));
    case FPS_type::FLOAT32:
        return functionparameter_SetParamValue_FLOAT32(cls, key.c_str(), nb::cast<float>(value));
    case FPS_type::FLOAT64:
        return functionparameter_SetParamValue_FLOAT64(cls, key.c_str(), nb::cast<double>(value));
    case FPS_type::STRING:
    case FPS_type::STRING_NOT_STREAM:
    case FPS_type::STREAMNAME:
    case FPS_type::DIRNAME:
    case FPS_type::EXECFILENAME:
    case FPS_type::FILENAME:
    case FPS_type::FITSFILENAME:
    case FPS_type::PROCESS:
    case FPS_type::FPSNAME:
        return functionparameter_SetParamValue_STRING(cls, key.c_str(),
                                                      nb::cast<std::string>(value).c_str());
    case FPS_type::TIMESPEC:
        return functionparameter_SetParamValue_TIMESPEC(
            cls, key.c_str(), double_to_timespec(nb::cast<double>(value)));
    default:
        return EXIT_FAILURE;
    }
}

nb::object fps_value_from_key(pyFps &cls, const std::string &key, const FPS_type fps_type)
{
    switch (fps_type)
    {
    case FPS_type::ONOFF:
        return nb::bool_(functionparameter_GetParamValue_ONOFF(cls, key.c_str()));
    case FPS_type::
        INT32: // TODO this is actally probably wrong ? since the underlying values stored are arrays, so reading a uint32 as a uint64 will concatenate them
    case FPS_type::PID:
    case FPS_type::INT64:
        return nb::int_(functionparameter_GetParamValue_INT64(cls, key.c_str()));
    case FPS_type::UINT32:
    case FPS_type::UINT64:
        return nb::int_(functionparameter_GetParamValue_UINT64(cls, key.c_str()));
    case FPS_type::FLOAT32:
        return nb::float_(functionparameter_GetParamValue_FLOAT32(cls, key.c_str()));
    case FPS_type::FLOAT64:
        return nb::float_(functionparameter_GetParamValue_FLOAT64(cls, key.c_str()));
    case FPS_type::STRING:
    case FPS_type::STRING_NOT_STREAM:
    case FPS_type::STREAMNAME:
    case FPS_type::DIRNAME:
    case FPS_type::EXECFILENAME:
    case FPS_type::FILENAME:
    case FPS_type::FITSFILENAME:
    case FPS_type::PROCESS:
    case FPS_type::FPSNAME:
        return nb::str(functionparameter_GetParamPtr_STRING(cls, key.c_str()));
    case FPS_type::TIMESPEC:
        return nb::float_(
            timespec_to_double(functionparameter_GetParamValue_TIMESPEC(cls, key.c_str())));
    default:
        return nb::none();
    }
}

nb::dict fps_to_dict(pyFps &cls)
{
    nb::dict fps_dict;
    for (auto &key : cls.keys())
    {
        fps_dict[nb::str(key.first.c_str())] = fps_value_from_key(cls, key.first, key.second);
    }
    return fps_dict;
}

NB_MODULE(FpsWrap, m)
{
    m.doc() = "FpsWrap library module";

    nb::enum_<FPS_status>(m, "FpsStatus")
        .value("CONF", FPS_status::CONF)
        .value("RUN", FPS_status::RUN)
        .value("CMDCONF", FPS_status::CMDCONF)
        .value("CMDRUN", FPS_status::CMDRUN)
        .value("RUNLOOP", FPS_status::RUNLOOP)
        .value("CHECKOK", FPS_status::CHECKOK)
        .value("TMUXCONF", FPS_status::TMUXCONF)
        .value("TMUXRUN", FPS_status::TMUXRUN)
        .value("TMUXCTRL", FPS_status::TMUXCTRL);

    nb::enum_<FPS_type>(m, "FpsType")
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
        .value("STRING_NOT_STREAM", FPS_type::STRING_NOT_STREAM);

    auto fps_flags = nb::enum_<FPS_flags>(m, "FpsFlag");
#define X(name) fps_flags.value(#name, FPS_flags::name);
    FPS_FLAGS_LIST(X)
#undef X

    if (!nb::type<timespec>().is_valid())
    { // Possible multiple registration warning
        nb::class_<timespec>(m, "timespec")
            .def(nb::init<time_t, long>())
            .def_rw("tv_sec", &timespec::tv_sec)
            .def_rw("tv_nsec", &timespec::tv_nsec);
    }

    nb::class_<pyFps>(m, "fps")
        // read-only constructor
        .def(nb::init<std::string>(),
             R"pbdoc(Read / connect to existing shared memory FPS
Parameters:
    name     [in]:  the name of the shared memory file to connect
)pbdoc",
             nb::arg("name"))

        // read/write constructor
        .def(nb::init<std::string, bool, int>(),
             R"pbdoc(Read / connect to existing shared memory FPS
Parameters:
    name     [in]:  the name of the shared memory file to connect
    create   [in]:  flag for creating of shared memory identifier
    NBparamMAX   [in]:  Max number of parameters
)pbdoc",
             nb::arg("name"), nb::arg("create"),
             nb::arg("NBparamMAX") = FUNCTION_PARAMETER_NBPARAM_DEFAULT)

        .def("asdict", &fps_to_dict)

        .def("__getitem__", [](pyFps &cls, const std::string &key)
             { return fps_value_from_key(cls, key, cls.keys(key)); })

        .def("__setitem__", [](pyFps &cls, const std::string &key, nb::object value)
             { return fps_value_to_key(cls, key, cls.keys(key), value); })

        .def("md", &pyFps::md, nb::rv_policy::reference)

        .def("add_entry", &pyFps::add_entry,
             R"pbdoc(Add parameter to database with default settings

If entry already exists, do not modify it

Parameters:
    name     [in]:  the name of the shared memory file to connect
    comment  [in]:  description
    fptype   [in]:  datatype
)pbdoc",
             nb::arg("entry_name"), nb::arg("entry_desc"), nb::arg("fptype"))

        .def("add_entry", &pyFps::add_entry_w_flags,
             R"pbdoc(Add parameter to database with custom flags

If entry already exists, do not modify it

Parameters:
    name     [in]:  the name of the shared memory file to connect
    comment  [in]:  description
    fptype   [in]:  datatype
    fpflag   [in]:  fps flags
)pbdoc",
             nb::arg("entry_name"), nb::arg("entry_desc"), nb::arg("fptype"), nb::arg("fpflag"))

        .def("get_status", &pyFps::get_status)
        .def("set_status", &pyFps::set_status)
        .def("disconnect", &pyFps::disconnect)

        .def("get_levelKeys", &pyFps::get_levelKeys)

        .def_prop_ro("keys", [](pyFps &cls) { return cls.keys(); })
        .def("signal_update", &pyFps::signal_update,
             R"pbdoc(Send update signal to FPS
)pbdoc")
        .def("CONFstart", &pyFps::CONFstart,
             R"pbdoc(FPS start CONF process

Requires setup performed by milk-fpsinit, which performs the following setup
- creates the FPS shared memory
- create up tmux sessions
- create function fpsrunstart, fpsrunstop, fpsconfstart and fpsconfstop

Return:
    ret      [out]: error code
)pbdoc")

        .def("CONFstop", &pyFps::CONFstop,
             R"pbdoc(FPS stop CONF process

Return:
    ret      [out]: error code
)pbdoc")

        .def("FPCONFexit", &pyFps::FPCONFexit,
             R"pbdoc(FPS exit FPCONF process

      Return:
          ret      [out]: error code
      )pbdoc")

        //   .def("FPCONFsetup", &pyFps::FPCONFsetup,
        //        R"pbdoc(FPS setup FPCONF process

        //   Return:
        //       ret      [out]: error code
        //   )pbdoc")

        .def("FPCONFloopstep", &pyFps::FPCONFloopstep,
             R"pbdoc(FPS loop step FPCONF process

      Return:
          ret      [out]: error code
      )pbdoc")

        .def("RUNstart", &pyFps::RUNstart,
             R"pbdoc(FPS start RUN process

Requires setup performed by milk-fpsinit, which performs the following setup
- creates the FPS shared memory
- create up tmux sessions
- create function fpsrunstart, fpsrunstop, fpsconfstart and fpsconfstop

Return:
    ret      [out]: error code
)pbdoc")

        .def("RUNstop", &pyFps::RUNstop,
             R"pbdoc(FPS stop RUN process

 Run pre-set function fpsrunstop in tmux ctrl window

Return:
    ret      [out]: error code
)pbdoc")

        .def("RUNexit", &pyFps::RUNexit,
             R"pbdoc(FPS exit RUN process

Return:
    ret      [out]: error code
)pbdoc")

        .def("TMUXstart", &pyFps::TMUXstart,
             R"pbdoc(Initialize tmux session for FPS

Creates a tmux session with ctrl, conf and run windows.

Return:
    ret      [out]: error code
)pbdoc")

        .def("TMUXstop", &pyFps::TMUXstop,
             R"pbdoc(Kill tmux session for FPS

Sends SIGINT to each window then kills the session.

Return:
    ret      [out]: error code
)pbdoc")
        // Test if CONF process is running

        .def_prop_ro(
            "CONFrunning",
            [](pyFps &cls)
            {
                pid_t pid = cls->md->confpid;
                if (pid_is_running(pid) && cls->md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CONF)
                {
                    return 1;
                }
                else // PID not active
                {
                    if (cls->md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF)
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

        .def_prop_ro(
            "RUNrunning",
            [](pyFps &cls)
            {
                pid_t pid = cls->md->runpid;
                if (pid_is_running(pid) &&
                    cls->md->status &
                        FUNCTION_PARAMETER_STRUCT_STATUS_RUNLOOP) // PID is live + completed loop setup.
                {
                    return 1;
                }
                else // PID not active
                {
                    if (cls->md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN)
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
    nb::class_<FUNCTION_PARAMETER_STRUCT_MD>(m, "FPS_md")
        .def_ro("name", &FUNCTION_PARAMETER_STRUCT_MD::name)
        .def_ro("description", &FUNCTION_PARAMETER_STRUCT_MD::description)
        .def_ro("kwarray", &FUNCTION_PARAMETER_STRUCT_MD::keywordarray)
        .def_ro("workdir", &FUNCTION_PARAMETER_STRUCT_MD::workdir)
        .def_ro("datadir", &FUNCTION_PARAMETER_STRUCT_MD::datadir)
        .def_ro("confdir", &FUNCTION_PARAMETER_STRUCT_MD::confdir)
        .def_ro("sourcefname", &FUNCTION_PARAMETER_STRUCT_MD::sourcefname)
        .def_ro("sourceline", &FUNCTION_PARAMETER_STRUCT_MD::sourceline)
        .def_ro("pname", &FUNCTION_PARAMETER_STRUCT_MD::pname)
        .def_ro("callprogname", &FUNCTION_PARAMETER_STRUCT_MD::callprogname)
        //   .def_ro("nameindexW",
        //  &FUNCTION_PARAMETER_STRUCT_MD::nameindexW)
        .def_ro("NBnameindex", &FUNCTION_PARAMETER_STRUCT_MD::NBnameindex)
        .def_ro("confpid", &FUNCTION_PARAMETER_STRUCT_MD::confpid)
        .def_ro("confpidstarttime", &FUNCTION_PARAMETER_STRUCT_MD::confpidstarttime)
        .def_ro("runpid", &FUNCTION_PARAMETER_STRUCT_MD::runpid)
        .def_ro("runpidstarttime", &FUNCTION_PARAMETER_STRUCT_MD::runpidstarttime)
        .def_ro("signal", &FUNCTION_PARAMETER_STRUCT_MD::signal)
        .def_ro("confwaitus", &FUNCTION_PARAMETER_STRUCT_MD::confwaitus)
        .def_ro("status", &FUNCTION_PARAMETER_STRUCT_MD::status)
        .def_ro("NBparamMAX", &FUNCTION_PARAMETER_STRUCT_MD::NBparamMAX)
        //   .def_ro("message", &FUNCTION_PARAMETER_STRUCT_MD::message)
        .def_ro("msgpindex", &FUNCTION_PARAMETER_STRUCT_MD::msgpindex)
        .def_ro("msgcode", &FUNCTION_PARAMETER_STRUCT_MD::msgcode)
        .def_ro("msgcnt", &FUNCTION_PARAMETER_STRUCT_MD::msgcnt)
        .def_ro("conferrcnt", &FUNCTION_PARAMETER_STRUCT_MD::conferrcnt);
}
