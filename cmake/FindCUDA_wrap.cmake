# - Wrapper around FindCUDA
cmake_minimum_required(VERSION 3.3)

if (MINGW)
  # Cuda doesn't work with mingw at all
  set (CUDA_FOUND FALSE)
elseif (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} VERSION_LESS 2.8)
  # FindCuda is included with CMake 2.8
  set (CUDA_FOUND FALSE)
else ()
  # GCS 2011.03.16
  # Make nvcc less whiny
  if (CMAKE_COMPILER_IS_GNUCC)
    set (CUDA_PROPAGATE_HOST_FLAGS CACHE BOOL OFF)
  endif ()

  # GCS 2012-05-11:  We need to propagate cxx flags to nvcc, but
  # the flag -ftest-coverage causes nvcc to barf, so exclude that one
  if (CMAKE_COMPILER_IS_GNUCC)
    string (REPLACE "-ftest-coverage" "" TMP "${CMAKE_CXX_FLAGS}")
    string (REPLACE "-ftemplate-depth-50" "" TMP "${TMP}")
    string (REPLACE " " "," TMP "${TMP}")
    set (CUDA_CXX_FLAGS ${CUDA_CXX_FLAGS} ${TMP})
  endif ()
  find_package (CUDA QUIET)
endif ()

# GCS 2012-09-25 - Seems this is needed too
if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC")
endif ()

# SR Remove warning with shared libs and MSVC
if(MSVC)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DRTK_EXPORTS")
endif()

set (CUDA_FOUND ${CUDA_FOUND} CACHE BOOL "Did we find cuda?")
mark_as_advanced(CUDA_FOUND)

if (CUDA_FOUND)
  cuda_include_directories (${CMAKE_CURRENT_SOURCE_DIR})
endif ()

if("${CUDA_VERSION}" VERSION_LESS 9.0)
 set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
     -gencode arch=compute_20,code=sm_20)
endif()
if("${CUDA_VERSION}" VERSION_LESS 11.0)
 set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
     -gencode arch=compute_30,code=sm_30)
endif()
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
     -Wno-deprecated-gpu-targets
     -gencode arch=compute_35,code=sm_35
     -gencode arch=compute_35,code=compute_35
     )

if(NOT "-std=c++${CMAKE_CXX_STANDARD}" IN_LIST CUDA_NVCC_FLAGS)
  list(APPEND CUDA_NVCC_FLAGS "-std=c++${CMAKE_CXX_STANDARD}")
endif()

if(CUDA_FOUND)
  try_run(RUN_RESULT_VAR COMPILE_RESULT_VAR
         ${CMAKE_BINARY_DIR}
         ${CMAKE_CURRENT_LIST_DIR}/has_cuda_gpu.cxx
         CMAKE_FLAGS
             -DINCLUDE_DIRECTORIES:STRING=${CUDA_TOOLKIT_INCLUDE}
             -DLINK_LIBRARIES:STRING=${CUDA_CUDART_LIBRARY}
         COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT_VAR
         RUN_OUTPUT_VARIABLE RUN_OUTPUT_VAR)
    # COMPILE_RESULT_VAR is TRUE when compile succeeds
    # RUN_RESULT_VAR is zero when a GPU is found
    if(COMPILE_RESULT_VAR AND NOT RUN_RESULT_VAR)
        set(CUDA_HAVE_GPU TRUE CACHE BOOL "Whether CUDA-capable GPU is present")
    else()
        set(CUDA_HAVE_GPU FALSE CACHE BOOL "Whether CUDA-capable GPU is present")
    endif()
    mark_as_advanced(CUDA_HAVE_GPU)
endif()
