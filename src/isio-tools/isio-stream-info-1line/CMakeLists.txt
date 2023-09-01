cmake_minimum_required(VERSION 3.5)

project (isio-stream-info)

find_package(PkgConfig REQUIRED)

pkg_check_modules(ISIO ImageStreamIO)

if(${ISIO_FOUND})
  add_executable(${PROJECT_NAME} isio-stream-info.c)
  target_include_directories(${PROJECT_NAME} PRIVATE ${ISIO_INCLUDE_DIRS})
  target_link_directories(${PROJECT_NAME} PRIVATE ${ISIO_LIBRARY_DIRS})
  target_link_libraries(${PROJECT_NAME} PRIVATE m ${ISIO_LIBRARIES})
  install(TARGETS ${PROJECT_NAME} DESTINATION bin)
endif()
