# CMake generated Testfile for 
# Source directory: /home/oguyon/src/milk/src/coremods/COREMOD_tools
# Build directory: /home/oguyon/src/milk/build_local/src/coremods/COREMOD_tools
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(CLIfunc-COREMOD_tools-module "milk-exec" "-T" "m?")
set_tests_properties(CLIfunc-COREMOD_tools-module PROPERTIES  LABELS "CLImodule" PASS_REGULAR_EXPRESSION "COREMOD_tools" TIMEOUT "1" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/coremods/COREMOD_tools/CMakeLists.txt;36;add_test;/home/oguyon/src/milk/src/coremods/COREMOD_tools/CMakeLists.txt;0;")
add_test(CLIfunc-COREMOD_tools-rtprio "milk-exec" "-T" "cmd? tools.rtprio")
set_tests_properties(CLIfunc-COREMOD_tools-rtprio PROPERTIES  LABELS "CLIfunc" PASS_REGULAR_EXPRESSION "src:" TIMEOUT "1" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/coremods/COREMOD_tools/CMakeLists.txt;49;add_test;/home/oguyon/src/milk/src/coremods/COREMOD_tools/CMakeLists.txt;0;")
add_test(CLIfunc-COREMOD_tools-tsetpmove "milk-exec" "-T" "cmd? tools.tsetpmove")
set_tests_properties(CLIfunc-COREMOD_tools-tsetpmove PROPERTIES  LABELS "CLIfunc" PASS_REGULAR_EXPRESSION "src:" TIMEOUT "1" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/coremods/COREMOD_tools/CMakeLists.txt;49;add_test;/home/oguyon/src/milk/src/coremods/COREMOD_tools/CMakeLists.txt;0;")
add_test(CLIfunc-COREMOD_tools-tsetpmoveext "milk-exec" "-T" "cmd? tools.tsetpmoveext")
set_tests_properties(CLIfunc-COREMOD_tools-tsetpmoveext PROPERTIES  LABELS "CLIfunc" PASS_REGULAR_EXPRESSION "src:" TIMEOUT "1" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/coremods/COREMOD_tools/CMakeLists.txt;49;add_test;/home/oguyon/src/milk/src/coremods/COREMOD_tools/CMakeLists.txt;0;")
add_test(CLIfunc-COREMOD_tools-csetpmove "milk-exec" "-T" "cmd? tools.csetpmove")
set_tests_properties(CLIfunc-COREMOD_tools-csetpmove PROPERTIES  LABELS "CLIfunc" PASS_REGULAR_EXPRESSION "src:" TIMEOUT "1" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/coremods/COREMOD_tools/CMakeLists.txt;49;add_test;/home/oguyon/src/milk/src/coremods/COREMOD_tools/CMakeLists.txt;0;")
add_test(CLIfunc-COREMOD_tools-csetandprioext "milk-exec" "-T" "cmd? tools.csetandprioext")
set_tests_properties(CLIfunc-COREMOD_tools-csetandprioext PROPERTIES  LABELS "CLIfunc" PASS_REGULAR_EXPRESSION "src:" TIMEOUT "1" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/coremods/COREMOD_tools/CMakeLists.txt;49;add_test;/home/oguyon/src/milk/src/coremods/COREMOD_tools/CMakeLists.txt;0;")
add_test(CLIfunc-COREMOD_tools-writef2file "milk-exec" "-T" "cmd? tools.writef2file")
set_tests_properties(CLIfunc-COREMOD_tools-writef2file PROPERTIES  LABELS "CLIfunc" PASS_REGULAR_EXPRESSION "src:" TIMEOUT "1" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/coremods/COREMOD_tools/CMakeLists.txt;49;add_test;/home/oguyon/src/milk/src/coremods/COREMOD_tools/CMakeLists.txt;0;")
add_test(CLIfunc-COREMOD_tools-dispim3d "milk-exec" "-T" "cmd? tools.dispim3d")
set_tests_properties(CLIfunc-COREMOD_tools-dispim3d PROPERTIES  LABELS "CLIfunc" PASS_REGULAR_EXPRESSION "src:" TIMEOUT "1" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/coremods/COREMOD_tools/CMakeLists.txt;49;add_test;/home/oguyon/src/milk/src/coremods/COREMOD_tools/CMakeLists.txt;0;")
add_test(CLIfunc-COREMOD_tools-ctsmstats "milk-exec" "-T" "cmd? tools.ctsmstats")
set_tests_properties(CLIfunc-COREMOD_tools-ctsmstats PROPERTIES  LABELS "CLIfunc" PASS_REGULAR_EXPRESSION "src:" TIMEOUT "1" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/coremods/COREMOD_tools/CMakeLists.txt;49;add_test;/home/oguyon/src/milk/src/coremods/COREMOD_tools/CMakeLists.txt;0;")
