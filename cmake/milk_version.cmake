# milk version and shared-memory structure versions (see milk_config.h).
# Included by the root CMakeLists and by src/engine/CMakeLists.txt (pyMilk BUNDLED builds).
set(VERSION_MAJOR 1)
set(VERSION_MINOR 03)
set(VERSION_PATCH 00)

# Shared-memory structure versions, checked on connect (see milk_config.h)
set(PROCESSINFO_VERSION "2.00")
set(FUNCTION_PARAMETER_STRUCT_VERSION "2.00")
