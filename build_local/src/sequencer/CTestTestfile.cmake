# CMake generated Testfile for 
# Source directory: /home/oguyon/src/milk/src/sequencer
# Build directory: /home/oguyon/src/milk/build_local/src/sequencer
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(sequencer_cmd_help_seq_list "milk-exec" "-T" "cmd? seq.list")
set_tests_properties(sequencer_cmd_help_seq_list PROPERTIES  _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/sequencer/CMakeLists.txt;38;add_test;/home/oguyon/src/milk/src/sequencer/CMakeLists.txt;0;")
add_test(sequencer_cmd_help_seq_start "milk-exec" "-T" "cmd? seq.start")
set_tests_properties(sequencer_cmd_help_seq_start PROPERTIES  _BACKTRACE_TRIPLES "/home/oguyon/src/milk/src/sequencer/CMakeLists.txt;44;add_test;/home/oguyon/src/milk/src/sequencer/CMakeLists.txt;0;")
