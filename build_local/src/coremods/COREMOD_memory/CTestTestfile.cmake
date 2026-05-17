# CMake generated Testfile for 
# Source directory: /home/oguyon/src/milk/src/coremods/COREMOD_memory
# Build directory: /home/oguyon/src/milk/build_local/src/coremods/COREMOD_memory
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(CLIfunc-COREMOD_memory-module "milk-exec" "-T" "m?")
set_tests_properties(CLIfunc-COREMOD_memory-module PROPERTIES  LABELS "CLImodule" PASS_REGULAR_EXPRESSION "COREMOD_memory" TIMEOUT "1" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/coremods/COREMOD_memory/CMakeLists.txt;141;add_test;/home/oguyon/src/milk/src/coremods/COREMOD_memory/CMakeLists.txt;0;")
add_test(CLIfunc-COREMOD_memory-mk2Dim "milk-exec" "-T" "cmd? mem.mk2Dim")
set_tests_properties(CLIfunc-COREMOD_memory-mk2Dim PROPERTIES  LABELS "CLIfunc" PASS_REGULAR_EXPRESSION "src:" TIMEOUT "1" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/coremods/COREMOD_memory/CMakeLists.txt;154;add_test;/home/oguyon/src/milk/src/coremods/COREMOD_memory/CMakeLists.txt;0;")
add_test(CLIfunc-COREMOD_memory-mk3Dim "milk-exec" "-T" "cmd? mem.mk3Dim")
set_tests_properties(CLIfunc-COREMOD_memory-mk3Dim PROPERTIES  LABELS "CLIfunc" PASS_REGULAR_EXPRESSION "src:" TIMEOUT "1" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/coremods/COREMOD_memory/CMakeLists.txt;154;add_test;/home/oguyon/src/milk/src/coremods/COREMOD_memory/CMakeLists.txt;0;")
add_test(CLIfunc-COREMOD_memory-listim "milk-exec" "-T" "cmd? mem.listim")
set_tests_properties(CLIfunc-COREMOD_memory-listim PROPERTIES  LABELS "CLIfunc" PASS_REGULAR_EXPRESSION "src:" TIMEOUT "1" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/coremods/COREMOD_memory/CMakeLists.txt;154;add_test;/home/oguyon/src/milk/src/coremods/COREMOD_memory/CMakeLists.txt;0;")
add_test(CLIfunc-COREMOD_memory-rmall "milk-exec" "-T" "cmd? mem.rmall")
set_tests_properties(CLIfunc-COREMOD_memory-rmall PROPERTIES  LABELS "CLIfunc" PASS_REGULAR_EXPRESSION "src:" TIMEOUT "1" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/coremods/COREMOD_memory/CMakeLists.txt;154;add_test;/home/oguyon/src/milk/src/coremods/COREMOD_memory/CMakeLists.txt;0;")
add_test(milksemloopspeed "/home/oguyon/src/milk/src/coremods/COREMOD_memory/scripts/milk-semloopspeed" "-D" "123456")
set_tests_properties(milksemloopspeed PROPERTIES  LABELS "perf" PASS_REGULAR_EXPRESSION "cnt0 = 123457" TIMEOUT "5" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/coremods/COREMOD_memory/CMakeLists.txt;164;add_test;/home/oguyon/src/milk/src/coremods/COREMOD_memory/CMakeLists.txt;0;")
