#
# CMake Module to check for TR1 and C++11 features
#
# SRTK_HAS_CXX11_STATIC_ASSERT    - True if "static_assert" keyword is supported
# SRTK_HAS_CXX11_FUNCTIONAL        - True if functional header has C++11 features
# SRTK_HAS_CXX11_TYPE_TRAITS
# SRTK_HAS_CXX11_UNORDERED_MAP
#
# SRTK_HAS_TR1_SUB_INCLUDE
#
# SRTK_HAS_TR1_FUNCTIONAL
# SRTK_HAS_TR1_TYPE_TRAITS
# SRTK_HAS_TR1_UNORDERED_MAP

#
# Function to do wrap try compiles on the aggregate cxx test file1
#
function(srtkCXX11Test VARIABLE)
  # use the hash of the depended cxx flags in the variable name to
  # cache the results.
  string(MD5 cmake_cxx_flags_hash "${CMAKE_CXX_FLAGS}")
  set(cache_var "${VARIABLE}_${hash_cmake_cxx_flags_hash}")
  if("${cache_var}" MATCHES "^${cache_var}$")
    message(STATUS "Performing Test ${VARIABLE}")
    set(requred_definitions "${CMAKE_REQUIRED_DEFINITIONS} -D${VARIABLE}")
    try_compile(${VARIABLE}
      ${SimpleRTK_BINARY_DIR}/CMakeTmp
      ${SimpleRTK_SOURCE_DIR}/CMake/srtk_check_cxx11.cxx
      CMAKE_FLAGS
      -DCOMPILE_DEFINITIONS:STRING=${requred_definitions}
      OUTPUT_VARIABLE output)
    if(${VARIABLE})
      set(${cache_var} 1 CACHE INTERNAL "SimpleRTK test ${FUNCTION}")
      message(STATUS "Performing Test ${VARIABLE} - Success")
    else()
      message(STATUS "Performing Test ${VARIABLE} - Failed")
      set(${cache_var} 0 CACHE INTERNAL "SimpleRTKTest ${FUNCTION}")
      file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
                 "Performing Test ${VARIABLE} failed with the following output:\n"
                 "${OUTPUT}\n")
    endif()
  else()
    set("${VARIABLE}" "${cache_var}")
  endif()
endfunction()

#
# Check for CXX11 Features
#
srtkCXX11Test(SRTK_HAS_CXX11_STATIC_ASSERT)
srtkCXX11Test(SRTK_HAS_CXX11_FUNCTIONAL)
srtkCXX11Test(SRTK_HAS_CXX11_TYPE_TRAITS)
srtkCXX11Test(SRTK_HAS_CXX11_UNORDERED_MAP)

# Microsoft Visual Studio 2008sp1,2010,2012 don't need tr1 as a path
# prefix to tr1 headers, while libc++
srtkCXX11Test(SRTK_HAS_TR1_SUB_INCLUDE)

set(CMAKE_REQUIRED_DEFINITIONS_SAVE ${CMAKE_REQUIRED_DEFINITIONS})
if (SRTK_HAS_TR1_SUB_INCLUDE)
  list(APPEND CMAKE_REQUIRED_DEFINITIONS "-DHAS_TR1_SUB_INCLUDE")
endif()

#
# Check for TR1 features
#
srtkCXX11Test(SRTK_HAS_TR1_FUNCTIONAL)
srtkCXX11Test(SRTK_HAS_TR1_TYPE_TRAITS)
srtkCXX11Test(SRTK_HAS_TR1_UNORDERED_MAP)

# restore varaible
set(CMAKE_REQUIRED_DEFINITIONS ${CMAKE_REQUIRED_DEFINITIONS_SAVE})


if ( (NOT SRTK_HAS_TR1_FUNCTIONAL AND NOT SRTK_HAS_CXX11_FUNCTIONAL)
    OR
    (NOT SRTK_HAS_TR1_TYPE_TRAITS AND NOT SRTK_HAS_CXX11_TYPE_TRAITS) )
  message( FATAL_ERROR
    "SimpleRTK requires usage of C++11 or C++ Technical Report 1 (TR1).\n"
    "It may be available as an optional download for your compiler or difference CXX_FLAGS."
    "Please see the FAQs for details."
    "http://www.itk.org/Wiki/SimpleRTK/FAQ#Do_I_need_to_download_an_option_package_for_TR1_support.3F\n" )
endif ()
