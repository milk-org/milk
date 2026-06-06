# ~~~
# have_cuda.cmake
#
# Scans for CUDA library
#
# Provides    HAVE_CUDA symbol (unset / TRUE)
# May disable USE_CUDA  symbol (OFF / ON)
# ~~~

check_language(CUDA) # -> CMAKE_CUDA_COMPILER
find_package(CUDAToolkit 9.0 REQUIRED) # -> CUDA_FOUND

if(CMAKE_CUDA_COMPILER) # <- check_language
  enable_language(CUDA)
  set(HAVE_CUDA TRUE)
elseif(CUDA_FOUND) # <- find_package
  set(CMAKE_CUDA_COMPILER
      "${CUDAToolkit_NVCC_EXECUTABLE}"
      CACHE FILEPATH "CUDA compiler" FORCE)
  enable_language(CUDA)
  set(HAVE_CUDA TRUE)
endif()

if(HAVE_CUDA)
  add_compile_definitions(HAVE_CUDA)
  include_directories(${CUDAToolkit_INCLUDE_DIRS})
else()
  message(WARNING "No CUDA support found -- USE_CUDA is deactivated")
  set(USE_CUDA OFF)
endif()

function(apply_cuda_to_target target)
  if(NOT HAVE_CUDA)
    message(WARNING "apply_cuda_to_target: CUDA not found, skipping ${target}")
    return()
  endif()
  target_link_libraries(${target} PRIVATE CUDA::cudart CUDA::toolkit
                                          CUDA::cublas CUDA::cusolver)
endfunction()
