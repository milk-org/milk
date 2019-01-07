#include <wyrm>
#include "processtools.h"
#include "streamCTRL.h"

class pyPROCESSINFO {
  PROCESSINFO *m_pinfo;

 public:
  pyPROCESSINFO() : m_pinfo(nullptr) {}

  pyPROCESSINFO(char *pname, int CTRLval) { create(pname, CTRLval); }

  ~pyPROCESSINFO() {
    // if (m_pinfo != nullptr) {
    //   processinfo_cleanExit(m_pinfo);
    //   m_pinfo = nullptr;
    // }
  }

  PROCESSINFO *operator->() { return m_pinfo; }

  int create(char *pname, int CTRLval) {
    // if (m_pinfo != nullptr) {
    //   processinfo_cleanExit(m_pinfo);
    // }
    m_pinfo = processinfo_shm_create(pname, CTRLval);
    m_pinfo->loopstat = 0;  // loop initialization
    strcpy(m_pinfo->source_FUNCTION, "\0");
    strcpy(m_pinfo->source_FILE, "\0");
    m_pinfo->source_LINE = 0;
    strcpy(m_pinfo->description, "\0");
    writeMessage("initialization done");
    return EXIT_SUCCESS;
  }

  int sigexit(int sig) {
    if (m_pinfo != nullptr) {
      int ret = processinfo_SIGexit(m_pinfo, sig);
      m_pinfo = nullptr;
      return ret;
    }
    return EXIT_FAILURE;
  }

  int writeMessage(char *message) {
    if (m_pinfo != nullptr) {
      return processinfo_WriteMessage(m_pinfo, message);
    }
    return EXIT_FAILURE;
  };
};

namespace py = pybind11;

PYBIND11_MODULE(CacaoProcessTools, m) {
  m.doc() = "CacaoProcessTools library module";

  m.def("processCTRL", &processinfo_CTRLscreen);
  m.def("streamCTRL", &streamCTRL_CTRLscreen);

  py::class_<timespec>(m, "timespec")
      .def(py::init<time_t, long>())
      .def_readwrite("tv_sec", &timespec::tv_sec)
      .def_readwrite("tv_nsec", &timespec::tv_nsec);

  py::class_<pyPROCESSINFO>(m, "processinfo")
      .def(py::init<>())
      .def(py::init<char *, int>())
      .def("create", &pyPROCESSINFO::create)
      .def("sigexit", &pyPROCESSINFO::sigexit)
      .def("writeMessage", &pyPROCESSINFO::writeMessage)
      .def_property("name",
                    [](pyPROCESSINFO &p) { return std::string(p->name); },
                    [](pyPROCESSINFO &p, std::string name) {
                      strncpy(p->name, name.c_str(), sizeof(p->name));
                    })
      .def_property(
          "source_FUNCTION",
          [](pyPROCESSINFO &p) { return std::string(p->source_FUNCTION); },
          [](pyPROCESSINFO &p, std::string source_FUNCTION) {
            strncpy(p->source_FUNCTION, source_FUNCTION.c_str(),
                    sizeof(p->source_FUNCTION));
          })
      .def_property(
          "source_FILE",
          [](pyPROCESSINFO &p) { return std::string(p->source_FILE); },
          [](pyPROCESSINFO &p, std::string source_FILE) {
            strncpy(p->source_FILE, source_FILE.c_str(),
                    sizeof(p->source_FILE));
          })
      .def_property("source_LINE",
                    [](pyPROCESSINFO &p) { return p->source_LINE; },
                    [](pyPROCESSINFO &p, int source_LINE) {
                      p->source_LINE = source_LINE;
                    })
      .def_property("PID", [](pyPROCESSINFO &p) { return p->PID; },
                    [](pyPROCESSINFO &p, int PID) { p->PID = PID; })

      .def_property("createtime",
                    [](pyPROCESSINFO &p) { return p->createtime; },
                    [](pyPROCESSINFO &p, timespec createtime) {
                      p->createtime = createtime;
                    })
      .def_property("loopcnt", [](pyPROCESSINFO &p) { return p->loopcnt; },
                    [](pyPROCESSINFO &p, int loopcnt) { p->loopcnt = loopcnt; })
      .def_property("CTRLval", [](pyPROCESSINFO &p) { return p->CTRLval; },
                    [](pyPROCESSINFO &p, int CTRLval) { p->CTRLval = CTRLval; })
      .def_property("tmuxname",
                    [](pyPROCESSINFO &p) { return std::string(p->tmuxname); },
                    [](pyPROCESSINFO &p, std::string name) {
                      strncpy(p->tmuxname, name.c_str(), sizeof(p->tmuxname));
                    })
      .def_property(
          "loopstat", [](pyPROCESSINFO &p) { return p->loopstat; },
          [](pyPROCESSINFO &p, int loopstat) { p->loopstat = loopstat; })
      .def_property("statusmsg",
                    [](pyPROCESSINFO &p) { return std::string(p->statusmsg); },
                    [](pyPROCESSINFO &p, std::string name) {
                      strncpy(p->statusmsg, name.c_str(), sizeof(p->statusmsg));
                    })
      .def_property(
          "statuscode", [](pyPROCESSINFO &p) { return p->statuscode; },
          [](pyPROCESSINFO &p, int statuscode) { p->statuscode = statuscode; })
      // .def_readwrite("logFile", &pyPROCESSINFO::m_pinfo::logFile)
      .def_property(
          "description",
          [](pyPROCESSINFO &p) { return std::string(p->description); },
          [](pyPROCESSINFO &p, std::string name) {
            strncpy(p->description, name.c_str(), sizeof(p->description));
          });

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
