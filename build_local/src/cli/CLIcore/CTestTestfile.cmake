# CMake generated Testfile for 
# Source directory: /home/oguyon/src/milk/src/cli/CLIcore
# Build directory: /home/oguyon/src/milk/build_local/src/cli/CLIcore
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(milklistim "milk-exec" "mem.listim")
set_tests_properties(milklistim PROPERTIES  LABELS "CLI" PASS_REGULAR_EXPRESSION "0 image" TIMEOUT "1" _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/cli/CLIcore/CMakeLists.txt;337;add_test;/home/oguyon/src/milk/src/cli/CLIcore/CMakeLists.txt;0;")
