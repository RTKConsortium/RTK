# client maintainer: simon.rit@creatis.insa-lyon.fr
set(CTEST_SITE "russula.clb")
set(CTEST_BUILD_NAME "Linux-64bit-intel")
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(dashboard_binary_name "RTK_lin64_intel")
set(ENV{ITK_DIR} "/home/srit/src/itk/lin64-intel")

# Set intel compiler
set(ENV{CC}  /opt/intel/bin/icc)   # C compiler
set(ENV{CXX} /opt/intel/bin/icpc)  # C++ compiler

include("${CTEST_SCRIPT_DIRECTORY}/rtk_common.cmake")

