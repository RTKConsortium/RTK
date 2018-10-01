######################################################################
# Based on a script by James Shackleford in Plastimatch.org
# Modified by Simon Rit
#
# Currently, nvcc works with gcc-4.3 and gcc-4.4 for CUDA 4.x
#
# This script:
#   * Checks the version of default gcc
#   * If CUDA 4.x, we look for gcc-4.4 or gcc44
#   * If not found or CUDA<4, we look for gcc-4.3 or gcc43 on the system
#   * If found we tell nvcc to use it
#   * If not found, fatal error
######################################################################

# This function searches for gcc version x.y (arguments 2 and 3 of the macro)
# Return the result in argument 1, empty if not found
function(FIND_GCC GCC_PATH GCC_MAJOR GCC_MINOR)
  # Search for gcc-x.y
  find_program(EXACT_GCC "gcc-${GCC_MAJOR}.${GCC_MINOR}")
  if(EXACT_GCC)
    set(GCC_PATH "${EXACT_GCC}" PARENT_SCOPE)
  else()
    # Search for gccxy
    find_program(EXACT_GCC2 "gcc${GCC_MAJOR}${GCC_MINOR}")
    if(EXACT_GCC2)
      set(GCC_PATH "${EXACT_GCC2}" PARENT_SCOPE)
    else()
      # gcc-x.y not found, check default gcc
      find_program(DEFAULT_GCC gcc)
      exec_program(${DEFAULT_GCC} ARGS "-dumpversion" OUTPUT_VARIABLE GCCVER)

      # Get major and minor revs
      string(REGEX REPLACE "^([0-9]+)\\..*" "\\1" GCCVER_MAJOR "${GCCVER}")
      string(REGEX REPLACE "^[0-9]+\\.([0-9]+).*" "\\1" GCCVER_MINOR "${GCCVER}")

      # Check that adequate
      if("${GCCVER_MAJOR}" MATCHES "${GCC_MAJOR}" AND "${GCCVER_MINOR}" MATCHES "${GCC_MINOR}")
        set(GCC_PATH ${DEFAULT_GCC} PARENT_SCOPE)
      endif()
    endif()
  endif()
endfunction()

# Main code dealing with each version of cuda
if(CUDA_FOUND)
  if((CMAKE_SYSTEM_NAME MATCHES "Linux" AND CMAKE_COMPILER_IS_GNUCC) OR APPLE)
    # Compatible gcc can be checked in host_config.h
    set(GCC_PATH "")
    if(${CUDA_VERSION} VERSION_GREATER "8.99")
      FIND_GCC(GCC_PATH "6" "4")
    endif()
    if(NOT GCC_PATH )
      FIND_GCC(GCC_PATH "4" "9")
    endif()
    if(NOT GCC_PATH )
      FIND_GCC(GCC_PATH "4" "8")
    endif()
    if(NOT GCC_PATH )
      FIND_GCC(GCC_PATH "4" "7")
    endif()
    if(NOT GCC_PATH )
      FIND_GCC(GCC_PATH "4" "6")
    endif()
    if(NOT GCC_PATH )
      FIND_GCC(GCC_PATH "4" "5")
    endif()
    if(NOT GCC_PATH )
      FIND_GCC(GCC_PATH "4" "4")
    endif()
    if(NOT GCC_PATH)
      FIND_GCC(GCC_PATH "4" "3")
    endif()
    if(NOT GCC_PATH)
      FIND_GCC(GCC_PATH "4" "2")
    endif()
    if(NOT GCC_PATH)
      FIND_GCC(GCC_PATH "3" "4")
    endif()

    if(GCC_PATH)
      if(NOT APPLE)
        message(STATUS "nvcc-check: Found adequate gcc (${GCC_PATH})... telling nvcc to use it!")
        #Only append option if not already done
        list (FIND CUDA_NVCC_FLAGS "--compiler-bindir" _index)
        if (${_index} EQUAL -1)
          list(APPEND CUDA_NVCC_FLAGS --compiler-bindir ${GCC_PATH})
        endif()
      endif()
    else()
      message(WARNING "nvcc-check: Please install adequate gcc for cuda.\nNote that gcc-4.x can be installed side-by-side with your current version of gcc.\n")
      set(CUDA_FOUND OFF)
    endif()
  endif()
endif()
