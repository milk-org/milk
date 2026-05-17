# Install script for directory: /home/oguyon/src/milk/plugins/cacao-src/AOloopControl

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/oguyon/local_milk/milk-1.03.00")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libcacaoAOloopControl.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libcacaoAOloopControl.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libcacaoAOloopControl.so"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/lib:/usr/local/include")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/oguyon/src/milk/build_local/plugins/cacao-src/AOloopControl/libcacaoAOloopControl.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libcacaoAOloopControl.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libcacaoAOloopControl.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libcacaoAOloopControl.so"
         OLD_RPATH "/usr/local/lib:/usr/local/include:/home/oguyon/src/milk/build_local/plugins/milk-extra-src/linopt_imtools:/home/oguyon/src/milk/build_local/src/cli/CLIcore:/home/oguyon/src/milk/build_local/src/cli/libmilkscript:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/lib:/usr/local/include")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libcacaoAOloopControl.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/cacao-src/AOloopControl/CMakeFiles/cacaoAOloopControl.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE PROGRAM FILES
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-setup"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-cli"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacaofuncs-log"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-check-cacaovars"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-commands"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-fpsctrl"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-fpsctrl-log"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-fpsctrl-logprocess"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-fpsctrl-TUI"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-loops"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-NULL"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-calib-archive"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-calib-apply"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-calib-archivecurrent"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-calib-streamload"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-exec"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao.tmux"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-fpsconfstart"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-fpsconfstop"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-fpslistadd"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-fpslistadd-ACQWFS"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-fpslistadd-MAPWFS"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-fpslistadd-MFILTTEST"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-fpslistadd-MVMGPU_CMODEVAL2DM"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-fpslistadd-MVMGPU_CMODEVALOFFLOAD2DM"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-fpslistadd-MVMGPU_DM2MVAL"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-fpslistadd-MVMGPU_WFS2CMODEVAL"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-fpslistadd-MVMGPU_ZPO"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-fpslistadd-MODALCTRL_STATS"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-fpslistadd-MODALFILTERING"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-fpslistadd-MVMGPU_ZPO"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-fpslistadd-OLMODEVAL2DM"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-fpslistadd-SIMMVMGPU"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-fpslistadd-WFSMODEVAL2DM"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-fpslistadd-WFSCAMSIM"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-fpslistadd-ZONALFILTERING"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-fpsrunstart"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-fpsrunstop"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-modalstatsTUI"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-takedark"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-task-manager"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-msglogCLI"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-msglogCTRL"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-logstreamsFITS"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-loop-deploy"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-fps-deploy-v2"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-setDMnolink"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-wfsref-setflat"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacaotask-check"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacaotask-INITSETUP"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacaotask-GETSIMCONFFILES"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacaotask-TESTCONFIG"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacaotask-CACAOSETUP"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacaotask-STARTDMCOMB"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacaotask-STARTSTREAMDELAY"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-000-dm"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-001-dmsim"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-002-simwfs"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-003-wfsmapping"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-005-takedark"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-020-mlat"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-020-mlatset"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-021-mlatshow"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-025-acqWFS"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-026-takeref"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-028-mkZFmodes"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-030-acqlinResp"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-031-RMHdecode"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-032-RMmkmask"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-033-RM-mksynthetic"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-034-RMzonal2modal"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-039-compstrCM"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-040-compfCM"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-041-loadCM"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-042-loadCM"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-041-saveCM"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-045-compCM-byblocks"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-050-wfs2cmval"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-060-mfilt"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-061-setmgains"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-062-setmmults"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-063-setmlimits"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-065-zonalfiltering"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-070-cmval2dm"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-071-zpo"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-080-testOL"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-100-DMturb"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-120-mstat"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-130-mkPF"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-140-applyPF"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-200-mfiltturbrec"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/aorun/cacao-aorun-300-wfsrefoptpsf"
    "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/scripts/cacao-fpsexec-list"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/AOloopControl" TYPE FILE FILES "/home/oguyon/src/milk/plugins/cacao-src/AOloopControl/AOloopControl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/cacao-fpsexec-cacao-zonalfilter" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/cacao-fpsexec-cacao-zonalfilter")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/cacao-fpsexec-cacao-zonalfilter"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/cacao-src/AOloopControl/cacao-fpsexec-cacao-zonalfilter")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/cacao-fpsexec-cacao-zonalfilter" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/cacao-fpsexec-cacao-zonalfilter")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/cacao-fpsexec-cacao-zonalfilter"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/cacao-fpsexec-cacao-zonalfilter")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/cacao-src/AOloopControl/CMakeFiles/cacao-fpsexec-cacao-zonalfilter.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/cacao-fpsexec-mfilt" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/cacao-fpsexec-mfilt")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/cacao-fpsexec-mfilt"
         RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/oguyon/src/milk/build_local/plugins/cacao-src/AOloopControl/cacao-fpsexec-mfilt")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/cacao-fpsexec-mfilt" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/cacao-fpsexec-mfilt")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/cacao-fpsexec-mfilt"
         OLD_RPATH "/usr/local/include:/usr/local/lib:/home/oguyon/src/milk/build_local/src/engine/libfps:/home/oguyon/src/milk/build_local/src/engine/libfpsseq:/home/oguyon/src/milk/build_local/src/engine/libmilkdata:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_arith:/home/oguyon/src/milk/build_local/src/coremods/COREMOD_iofits:/home/oguyon/src/milk/build_local/src/engine/libprocessinfo:/home/oguyon/src/milk/build_local/src/engine/ImageStreamIO:"
         NEW_RPATH "/home/oguyon/local_milk/milk-1.03.00/lib:/usr/local/include:/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/cacao-fpsexec-mfilt")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/oguyon/src/milk/build_local/plugins/cacao-src/AOloopControl/CMakeFiles/cacao-fpsexec-mfilt.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/oguyon/src/milk/build_local/plugins/cacao-src/AOloopControl/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
