#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/map.h>

#include <string>
#include <cstring>

#include "ProcessInfo-engine/processinfo.h"
//#include "ProcessInfo-engine/pinfo_struct_api.h"
//#include "ProcessInfo-engine/pinfo_exec.h"

#include "pyProcessInfo.hpp"

namespace nb = nanobind;
using namespace nanobind::literals;

using TimespecArray = nb::ndarray<nb::numpy, double, nb::shape<PROCESSINFO_NBtimer>>;

// Copy of a timespec buffer in shared memory, converted to seconds (double).
static TimespecArray timespec_array_to_double(const timespec *buf)
{
    double *data = new double[PROCESSINFO_NBtimer];
    for (size_t i = 0; i < PROCESSINFO_NBtimer; ++i)
    {
        data[i] = buf[i].tv_sec + 1e-9 * buf[i].tv_nsec;
    }
    nb::capsule owner(data, [](void *p) noexcept { delete[] (double *) p; });
    return TimespecArray(data, { (size_t) PROCESSINFO_NBtimer }, owner);
}

NB_MODULE(ProcessInfoWrap, m)
{
    m.doc() = "ProcessInfoWrap library module";

    m.attr("PROCESSINFO_NBtimer") = PROCESSINFO_NBtimer;

    if (!nb::type<timespec>().is_valid())
    { // Possible multiple registration warning
        nb::class_<timespec>(m, "timespec")
            .def(nb::init<time_t, long>())
            .def_rw("tv_sec", &timespec::tv_sec)
            .def_rw("tv_nsec", &timespec::tv_nsec);
    }

    nb::class_<pyProcessInfo>(m, "processinfo")
        .def(nb::init<>(),
             R"pbdoc(Construct a empty Process Info object
)pbdoc")

        .def(nb::init<const char *, int>(),
             R"pbdoc(Construct a new Process Info object

Parameters:
    pname : name of the Process Info object (human-readable)
    CTRLval : control value to be externally written.
             - 0: run                     (default)
             - 1: pause
             - 2: increment single step (will go back to 1)
             - 3: exit loop
)pbdoc",
             nb::arg("name"), nb::arg("CTRLval"))

        .def("create", &pyProcessInfo::create,
             R"pbdoc(Create Process Info object in shared memory

Parameters:
    pname : name of the Process Info object (human-readable)
    CTRLval : control value to be externally written on startup.
             - 0: run                     (default)
             - 1: pause
             - 2: increment single step (will go back to 1)
             - 3: exit loop
Return:
    ret : error code
)pbdoc",
             nb::arg("name"), nb::arg("CTRLval"))

        .def("link", &pyProcessInfo::link,
             R"pbdoc(Link an existing Process Info object in shared memory

Parameters:
    pname : name of the Process Info object (human-readable)
Return:
    ret : error code
)pbdoc",
             nb::arg("name"))
        .def("clean_exit", &pyProcessInfo::clean_exit,
             R"pbdoc(Send CTRLval signal to tell process to exit gracefully.

Return:
    ret: error code
)pbdoc")
        .def("sigexit", &pyProcessInfo::sigexit,
             R"pbdoc(Send a signal to a Process Info object

Parameters:
    sig : signal to send
)pbdoc",
             nb::arg("sig"))

        .def("writeMessage", &pyProcessInfo::writeMessage,
             R"pbdoc(Write a message into a Process Info object

Parameters:
    message : message to write
Return:
    ret : error code
)pbdoc",
             nb::arg("message"))

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

        .def_prop_rw(
            "name", [](pyProcessInfo &p) { return std::string(p->name); },
            [](pyProcessInfo &p, std::string name)
            { strncpy(p->name, name.c_str(), sizeof(p->name)); }, "Name of the Process Info object")

        .def_prop_rw(
            "source_FUNCTION", [](pyProcessInfo &p) { return std::string(p->source_FUNCTION); },
            [](pyProcessInfo &p, std::string source_FUNCTION)
            { strncpy(p->source_FUNCTION, source_FUNCTION.c_str(), sizeof(p->source_FUNCTION)); })

        .def_prop_rw(
            "source_FILE", [](pyProcessInfo &p) { return std::string(p->source_FILE); },
            [](pyProcessInfo &p, std::string source_FILE)
            { strncpy(p->source_FILE, source_FILE.c_str(), sizeof(p->source_FILE)); })
        .def_prop_rw(
            "source_LINE", [](pyProcessInfo &p) { return p->source_LINE; },
            [](pyProcessInfo &p, int source_LINE) { p->source_LINE = source_LINE; })
        .def_prop_rw(
            "PID", [](pyProcessInfo &p) { return p->PID; },
            [](pyProcessInfo &p, int PID) { p->PID = PID; })

        .def_prop_rw(
            "createtime", [](pyProcessInfo &p) { return p->createtime; },
            [](pyProcessInfo &p, timespec createtime) { p->createtime = createtime; })
        .def_prop_rw(
            "loopcnt", [](pyProcessInfo &p) { return p->loopcnt; },
            [](pyProcessInfo &p, int loopcnt) { p->loopcnt = loopcnt; })
        .def_prop_rw(
            "CTRLval", [](pyProcessInfo &p) { return p->CTRLval; },
            [](pyProcessInfo &p, int CTRLval) { p->CTRLval = CTRLval; })
        .def_prop_rw(
            "tmuxname", [](pyProcessInfo &p) { return std::string(p->tmuxname); },
            [](pyProcessInfo &p, std::string name)
            { strncpy(p->tmuxname, name.c_str(), sizeof(p->tmuxname)); })
        .def_prop_rw(
            "loopstat", [](pyProcessInfo &p) { return p->loopstat; },
            [](pyProcessInfo &p, int loopstat) { p->loopstat = loopstat; })
        .def_prop_rw(
            "statusmsg", [](pyProcessInfo &p) { return std::string(p->statusmsg); },
            [](pyProcessInfo &p, std::string name)
            { strncpy(p->statusmsg, name.c_str(), sizeof(p->statusmsg)); })
        .def_prop_rw(
            "statuscode", [](pyProcessInfo &p) { return p->statuscode; },
            [](pyProcessInfo &p, int statuscode) { p->statuscode = statuscode; })
        .def_prop_rw(
            "timerindex", [](pyProcessInfo &p) { return p->timerindex; },
            [](pyProcessInfo &p, int timerindex) { p->timerindex = timerindex; },
            "Last written index in the timing circular buffers")
        .def_prop_ro(
            "texecstart", [](pyProcessInfo &p) { return timespec_array_to_double(p->texecstart); },
            "Copy of the task start times circular buffer, as a float64 "
            "array of seconds, shape (PROCESSINFO_NBtimer,)",
            nb::rv_policy::move)
        .def_prop_ro(
            "texecend", [](pyProcessInfo &p) { return timespec_array_to_double(p->texecend); },
            "Copy of the task end times circular buffer, as a float64 "
            "array of seconds, shape (PROCESSINFO_NBtimer,)",
            nb::rv_policy::move)
        // .def_readwrite("logFile", &pyProcessInfo::m_pinfo::logFile)
        .def_prop_rw(
            "description", [](pyProcessInfo &p) { return std::string(p->description); },
            [](pyProcessInfo &p, std::string name)
            { strncpy(p->description, name.c_str(), sizeof(p->description)); });
}
