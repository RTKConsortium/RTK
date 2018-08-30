# client maintainer: simon.rit@creatis.insa-lyon.fr
set(CTEST_SITE "russula.clb")
set(CTEST_BUILD_NAME "Linux-64bit-gcc-cuda-valgrind")
set(CTEST_BUILD_CONFIGURATION RelWithDebInfo)
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(dashboard_binary_name "RTK_lin64_gcc_nocuda_system_itk4_valgrind")
set(ENV{ITK_DIR} "/home/srit/src/itk/lin64-RelWithDebInfo")
set(CTEST_BUILD_FLAGS -j16)
set(ENV{CUDA_BIN_PATH} "/home/srit/Download/cuda80/bin")

set(ENV{VALGRIND_LIB} "/usr/lib64/valgrind")
set(CTEST_MEMORYCHECK_COMMAND /usr/bin/valgrind)
set(CTEST_MEMORYCHECK_COMMAND_OPTIONS "--gen-suppressions=all --child-silent-after-fork=yes -q --leak-check=yes --show-reachable=yes --num-callers=50 -v")
set(CTEST_MEMORYCHECK_SUPPRESSIONS_FILE "${CTEST_SCRIPT_DIRECTORY}/RTK.supp")
set(CTEST_CUSTOM_MEMCHECK_IGNORE "RTKInDoxygenGroup")
set(CONFIGURE_OPTIONS -DFAST_TESTS_NO_CHECKS=TRUE)
set(dashboard_do_memcheck true)

include("${CTEST_SCRIPT_DIRECTORY}/rtk_common.cmake")

