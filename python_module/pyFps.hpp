#ifndef PYFPS_H
#define PYFPS_H

extern "C" {
#include "processtools.h"

#include "fps_CONFstart.h"
#include "fps_CONFstop.h"

#include "fps_RUNstart.h"
#include "fps_RUNstop.h"

}

/**
 * @brief This code is a part of octopus module from COSMIC project
 *
 * @author: Florian Ferreira <florian.ferreira@obspm.fr>
 *
 */


enum FPS_status : uint16_t {
  CONF = FUNCTION_PARAMETER_STRUCT_STATUS_CONF,
  RUN = FUNCTION_PARAMETER_STRUCT_STATUS_RUN,
  CMDCONF = FUNCTION_PARAMETER_STRUCT_STATUS_CMDCONF,
  CMDRUN = FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN,
  RUNLOOP = FUNCTION_PARAMETER_STRUCT_STATUS_RUNLOOP,
  CHECKOK = FUNCTION_PARAMETER_STRUCT_STATUS_CHECKOK,
  TMUXCONF = FUNCTION_PARAMETER_STRUCT_STATUS_TMUXCONF,
  TMUXRUN = FUNCTION_PARAMETER_STRUCT_STATUS_TMUXRUN,
  TMUXCTRL = FUNCTION_PARAMETER_STRUCT_STATUS_TMUXCTRL
};

class pyFps {
 public:
  std::string name;
  FUNCTION_PARAMETER_STRUCT fps;
  std::map<std::string, std::string> keys;
  std::map<std::string, const int> fptype_map{{"int", FPTYPE_INT64},
                                              {"double", FPTYPE_FLOAT64},
                                              {"float", FPTYPE_FLOAT32},
                                              {"string", FPTYPE_STRING}};

 public:
  /**
   * @brief Read / connect to existing shared memory FPS

   * @param name : the name of the shared memory file to connect
   */
  pyFps(std::string name)
      : pyFps(name, false, FUNCTION_PARAMETER_NBPARAM_DEFAULT){};

  /**
   * @brief Read or create an shared FPS
   *
   * @param fps_name : the name of the shared memory file to connect
   * @param create : flag for creating of shared memory identifier
   * @param NBparamMAX : Max number of parameters
   */
  pyFps(std::string fps_name, bool create, int NBparamMAX) {
    this->name = fps_name;
    if (create) {
      this->create_and_connect(NBparamMAX);
    } else {
      while (this->connect() == -1) {
        sleep(0.01);
      }
      this->read_keys();
    }
    std::cout << "FPS connected" << std::endl;
  }

  /**
   * @brief Destroy the py Fps object
   *
   */
  ~pyFps() = default;

  FUNCTION_PARAMETER_STRUCT *operator->() { return &fps; }

  FUNCTION_PARAMETER_STRUCT_MD *md() {return fps.md;}

  /**
   * @brief Create a and connect object
   *
   * @param NBparamMAX
   * @return int
   */
  int create_and_connect(int NBparamMAX) {
    if (this->connect() == -1) {
      std::cout << "Creating FPS...";
      function_parameter_struct_create(NBparamMAX, this->name.c_str());
      std::cout << "Done" << std::endl;
      this->connect();
    } else {
      this->connect();
      this->read_keys();
    }
    return EXIT_SUCCESS;
  }

  /**
   * @brief Connect to existing shared memory FPS

   * @param name : the name of the shared memory file to connect
   */
  int connect() {
    return function_parameter_struct_connect(this->name.c_str(), &this->fps,
                                             FPSCONNECT_SIMPLE);
  };

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
                std::string fptype) {
    this->keys[entry_name] = fptype;
    return function_parameter_add_entry(
        &this->fps, entry_name.c_str(), entry_desc.c_str(),
        this->fptype_map[fptype], FPFLAG_DEFAULT_INPUT, nullptr);
  }

  /**
   * @brief Get the status object
   *
   * @return int
   */
  int get_status() { return this->fps.md->status; }

  /**
   * @brief Set the status object
   *
   * @param status
   * @return int
   */
  int set_status(int status) {
    this->fps.md->status = status;
    return EXIT_SUCCESS;
  }
  int read_keys() {
    int k = 0;
    while ((k < this->fps.md->NBparamMAX) &&
           (this->fps.parray[k].keywordfull[0] != '\0')) {
      for (auto const &fptype : fptype_map) {
        if (fptype.second == this->fps.parray[k].type) {
          this->keys[this->fps.parray[k].keywordfull] = fptype.first;
          break;
        }
      }
      k++;
    }

    return EXIT_SUCCESS;
  };

  /**
   * @brief Get the levelKeys object
   *
   * @param level
   * @return std::vector<std::string>
   */
  std::vector<std::string> get_levelKeys(int level) {
    std::vector<std::string> levelKeys = std::vector<std::string>();
    int k = 0;
    while (this->fps.parray[k].keywordfull[0] != '\0' &&
           k < this->fps.md->NBparamMAX) {
      std::string tmp = this->fps.parray[k].keyword[level];
      auto exist = std::find(levelKeys.begin(), levelKeys.end(), tmp);
      if (exist == levelKeys.end()) {
        levelKeys.push_back(tmp);
      }
      k++;
    }

    return levelKeys;
  }
};

#endif