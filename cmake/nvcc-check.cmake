######################################################################
# Based on a script by James Shackleford in Plastimatch.org
# Modified by Simon Rit
#
# Currently, nvcc works with gcc-4.3 and gcc-4.4 for CUDA 4.x
#
# This script:
#   * Checks the version of default gcc
#   * If CUDA 4.x, we look for gcc-4.4
#   * If not found or CUDA<4, we look for gcc-4.3 on the system
#   * If found we tell nvcc to use it
#   * If not found, fatal error
######################################################################

# This function searches for gcc version x.y (arguments 2 and 3 of the macro)
# Return the result in argument 1, empty if not found
FUNCTION(FIND_GCC GCC_PATH GCC_MAJOR GCC_MINOR)

  # Search for gcc-x.y
  FIND_PROGRAM(EXACT_GCC "gcc-${GCC_MAJOR}.${GCC_MINOR}")
  IF(EXACT_GCC)
    SET(GCC_PATH "${EXACT_GCC}" PARENT_SCOPE)
  ELSE(EXACT_GCC)
    # gcc-x.y not found, check default gcc
    FIND_PROGRAM(DEFAULT_GCC gcc)
    EXEC_PROGRAM(${DEFAULT_GCC} ARGS "-dumpversion" OUTPUT_VARIABLE GCCVER)

    # Get major and minor revs
    STRING(REGEX REPLACE "([0-9]+).[0-9]+" "\\1" GCCVER_MAJOR "${GCCVER}")
    STRING(REGEX REPLACE "[0-9]+.([0-9]+)" "\\1" GCCVER_MINOR "${GCCVER}")

    # Check that adequate
    IF("${GCCVER_MAJOR}" MATCHES "${GCC_MAJOR}" AND "${GCCVER_MINOR}" MATCHES "${GCC_MINOR}")
      SET(GCC_PATH ${DEFAULT_GCC} PARENT_SCOPE)
    ENDIF()
  ENDIF(EXACT_GCC)

ENDFUNCTION(FIND_GCC)

# Main code dealing with each version of cuda
IF(CUDA_FOUND)
  IF(CMAKE_SYSTEM_NAME MATCHES "Linux" AND CMAKE_COMPILER_IS_GNUCC)
    SET(GCC_PATH "")
    IF(CUDA_VERSION_MAJOR MATCHES "4")
      FIND_GCC(GCC_PATH "4" "4")
    ENDIF(CUDA_VERSION_MAJOR MATCHES "4")
    IF(NOT GCC_PATH OR CUDA_VERSION_MAJOR LESS "3")
      FIND_GCC(GCC_PATH "4" "3")
    ENDIF(NOT GCC_PATH OR CUDA_VERSION_MAJOR LESS "3")

    IF(GCC_PATH)
      MESSAGE(STATUS "nvcc-check: Found adequate gcc (${GCC_PATH})... telling nvcc to use it!")
      SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --compiler-bindir=${GCC_PATH})
    ELSE(GCC_PATH)
      MESSAGE(FATAL_ERROR "nvcc-check: Please install adequate gcc for cuda (gcc-4.3 or gcc-4.4 for Cuda 4).\nNote that gcc-4.x can be installed side-by-side with your current version of gcc.\n")
    ENDIF(GCC_PATH)
  ENDIF(CMAKE_SYSTEM_NAME MATCHES "Linux" AND CMAKE_COMPILER_IS_GNUCC)


  IF(CMAKE_SYSTEM_NAME MATCHES "Linux" OR "APPLE")
      # For CUDA 3.2: surface_functions.h does some non-compliant things...
      #               so we tell g++ to ignore them when called via nvcc
      #               by passing the -fpermissive flag through the nvcc
      #               build trajectory.  Unfortunately, nvcc will also
      #               blindly pass this flag to gcc, even though it is not
      #               valid... resulting in TONS of warnings.  So, we go
      #               version checking again, this time nvcc...
      # Get the nvcc version number

      IF(CUDA_VERSION_MAJOR MATCHES "3")
          IF(CUDA_VERSION_MINOR MATCHES "2")
              SET (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --compiler-options='-fpermissive')
              MESSAGE(STATUS "nvcc-check: CUDA 3.2 exception: CUDA_NVCC_FLAGS set to \"${CUDA_NVCC_FLAGS}\"")
          ENDIF()
      ENDIF()
  ENDIF()
ENDIF(CUDA_FOUND)
