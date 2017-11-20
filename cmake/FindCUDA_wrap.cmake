# - Wrapper around FindCUDA

if (MINGW)
  # Cuda doesn't work with mingw at all
  set (CUDA_FOUND FALSE)
elseif (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} LESS 2.8)
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

  # GCS 2012-05-07: Workaround for poor, troubled FindCUDA
  set (CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE FALSE)
  find_package (CUDA QUIET)
endif ()

# GCS 2012-09-25 - Seems this is needed too
if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
	#  set (CUDA_CXX_FLAGS "${CUDA_CXX_FLAGS},-fPIC")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC")
endif ()


set (CUDA_FOUND ${CUDA_FOUND} CACHE BOOL "Did we find cuda?")
mark_as_advanced(CUDA_FOUND)

if(CUDA_FOUND)
  if(${CUDA_VERSION} LESS 3.2)
    message("CUDA version ${CUDA_VERSION} found, too old for RTK")
    set(CUDA_FOUND FALSE)
  endif()
endif()

if (CUDA_FOUND)
  cuda_include_directories (${CMAKE_CURRENT_SOURCE_DIR})
endif ()

# JAS 08.25.2010
#   Check to make sure nvcc has gcc-4.3 for compiling.
#   This script will modify CUDA_NVCC_FLAGS if system default is not gcc-4.3
include (nvcc-check)

if("${CUDA_VERSION}" LESS 6.5)
	#  set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
	#        -gencode arch=compute_10,code=sm_10
	#        -gencode arch=compute_11,code=sm_11
	#        -gencode arch=compute_12,code=sm_12
	#        -gencode arch=compute_13,code=sm_13
	#      )
endif ()

if("${CUDA_VERSION}" LESS 5.0)
 set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
     -gencode arch=compute_20,code=sm_20
     -gencode arch=compute_20,code=compute_20
    )
elseif("${CUDA_VERSION}" LESS 8.0)
 set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
     -gencode arch=compute_20,code=sm_20
     -gencode arch=compute_30,code=sm_30
     -gencode arch=compute_35,code=sm_35
     -gencode arch=compute_35,code=compute_35
     )
elseif("${CUDA_VERSION}" LESS 9.0)
 set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
     -Wno-deprecated-gpu-targets
     -gencode arch=compute_20,code=sm_20
     -gencode arch=compute_30,code=sm_30
     -gencode arch=compute_35,code=sm_35
     -gencode arch=compute_35,code=compute_35
     )
else()
 set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
     -gencode arch=compute_30,code=sm_30
     -gencode arch=compute_35,code=sm_35
     -gencode arch=compute_35,code=compute_35
     )
endif()

if(CUDA_FOUND)
  try_run(RUN_RESULT_VAR COMPILE_RESULT_VAR
         ${CMAKE_BINARY_DIR} 
         ${CMAKE_CURRENT_LIST_DIR}/has_cuda_gpu.c
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
