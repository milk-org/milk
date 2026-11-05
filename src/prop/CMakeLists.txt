set(LIBNAME "milkprop")

set(SOURCEFILES
    milk_shm_prop.c
    milk_shm_prop_stream.c
    milk_shm_prop_fps.c
    milk_shm_prop_proc.c
    milk_shm_cache.c)

set(INCLUDEFILES
    milk_shm_prop.h
    milk_shm_cache.h
    milk_entity_ns.h)

add_library(${LIBNAME} SHARED ${SOURCEFILES})

if(CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
  target_compile_options(${LIBNAME} PRIVATE -fopenmp-simd)
endif()

target_include_directories(
  ${LIBNAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
         $<INSTALL_INTERFACE:include/milkprop>
         $<INSTALL_INTERFACE:include>
  PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine>
          $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/ImageStreamIO>
          $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libmilkcommon>
          $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libmilkdata>
          $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libprocessinfo>
          $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>
          $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libcomputeunit>)

target_link_libraries(
  ${LIBNAME}
  PUBLIC ImageStreamIO
         milkcommon
         milkdata
         milkprocessinfo
         milkfps
         milkcomputeunit
         dl
         rt
         m
         ${CMAKE_THREAD_LIBS_INIT})

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/milkprop)
install(FILES ${INCLUDEFILES} DESTINATION include)
