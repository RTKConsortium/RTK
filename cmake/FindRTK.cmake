# - Find an RTK installation or build tree.

# When RTK is found, the RTKConfig.cmake file is sourced to setup the
# location and configuration of RTK.  Please read this file, or
# RTKConfig.cmake.in from the RTK source tree for the full list of
# definitions.  Of particular interest is RTK_USE_FILE, a CMake source file
# that can be included to set the include directories, library directories,
# and preprocessor macros.  In addition to the variables read from
# RTKConfig.cmake, this find module also defines
#  RTK_DIR  - The directory containing RTKConfig.cmake.  
#             This is either the root of the build tree, 
#             or the lib/InsightToolkit directory.  
#             This is the only cache entry.
#   
#  RTK_FOUND - Whether RTK was found.  If this is true, 
#              RTK_DIR is okay.
#

SET(RTK_DIR_STRING "directory containing RTKConfig.cmake.  This is either the root of the build tree, or PREFIX/lib for an installation.")

# Search only if the location is not already known.
IF(NOT RTK_DIR)
  # Get the system search path as a list.
  IF(UNIX)
    STRING(REGEX MATCHALL "[^:]+" RTK_DIR_SEARCH1 "$ENV{PATH}")
  ELSE(UNIX)
    STRING(REGEX REPLACE "\\\\" "/" RTK_DIR_SEARCH1 "$ENV{PATH}")
  ENDIF(UNIX)
  STRING(REGEX REPLACE "/;" ";" RTK_DIR_SEARCH2 ${RTK_DIR_SEARCH1})

  # Construct a set of paths relative to the system search path.
  SET(RTK_DIR_SEARCH "")
  FOREACH(dir ${RTK_DIR_SEARCH2})
    SET(RTK_DIR_SEARCH ${RTK_DIR_SEARCH} "${dir}/../lib")
  ENDFOREACH(dir)

  #
  # Look for an installation or build tree.
  #
  FIND_PATH(RTK_DIR RTKConfig.cmake
    # Look for an environment variable RTK_DIR.
    $ENV{RTK_DIR}

    # Look in places relative to the system executable search path.
    ${RTK_DIR_SEARCH}

    # Look in standard UNIX install locations.
    /usr/local/lib
    /usr/lib

    # Read from the CMakeSetup registry entries.  It is likely that
    # RTK will have been recently built.
    [HKEY_CURRENT_USER\\Software\\Kitware\\CMakeSetup\\Settings\\StartPath;WhereBuild1]
    [HKEY_CURRENT_USER\\Software\\Kitware\\CMakeSetup\\Settings\\StartPath;WhereBuild2]
    [HKEY_CURRENT_USER\\Software\\Kitware\\CMakeSetup\\Settings\\StartPath;WhereBuild3]
    [HKEY_CURRENT_USER\\Software\\Kitware\\CMakeSetup\\Settings\\StartPath;WhereBuild4]
    [HKEY_CURRENT_USER\\Software\\Kitware\\CMakeSetup\\Settings\\StartPath;WhereBuild5]
    [HKEY_CURRENT_USER\\Software\\Kitware\\CMakeSetup\\Settings\\StartPath;WhereBuild6]
    [HKEY_CURRENT_USER\\Software\\Kitware\\CMakeSetup\\Settings\\StartPath;WhereBuild7]
    [HKEY_CURRENT_USER\\Software\\Kitware\\CMakeSetup\\Settings\\StartPath;WhereBuild8]
    [HKEY_CURRENT_USER\\Software\\Kitware\\CMakeSetup\\Settings\\StartPath;WhereBuild9]
    [HKEY_CURRENT_USER\\Software\\Kitware\\CMakeSetup\\Settings\\StartPath;WhereBuild10]

    # Help the user find it if we cannot.
    DOC "The ${RTK_DIR_STRING}"
  )
ENDIF(NOT RTK_DIR)

# If RTK was found, load the configuration file to get the rest of the
# settings.
IF(RTK_DIR)
  SET(RTK_FOUND 1)
  INCLUDE(${RTK_DIR}/RTKConfig.cmake)
ELSE(RTK_DIR)
  SET(RTK_FOUND 0)
  IF(RTK_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Please set RTK_DIR to the ${RTK_DIR_STRING}")
  ENDIF(RTK_FIND_REQUIRED)
ENDIF(RTK_DIR)
