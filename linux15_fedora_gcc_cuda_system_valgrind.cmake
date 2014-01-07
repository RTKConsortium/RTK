# client maintainer: simon.rit@creatis.insa-lyon.fr
set(CTEST_SITE "linux15.dg.creatis.insa-lyon.fr")
set(CTEST_BUILD_NAME "Linux-64bit-gcc-cuda-system-valgrind")
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_DASHBOARD_ROOT "/tmp/RTK_dashboard")
set(dashboard_binary_name "RTK_lin64_gcc_cuda_system-valgrind")
set(ENV{ITK_DIR} "/home/srit/src/itk/lin64-dg")
set(ENV{CUDA_BIN_PATH} "/usr/lib64/cuda/bin")
set(ENV{CUDA_LIB_PATH} "/usr/lib64/nvidia-304xx")
set(CTEST_BUILD_FLAGS -j8)

# OpenCL
set(CONFIGURE_OPTIONS
   -DOPENCL_LIBRARIES:PATH=/usr/lib64/nvidia-304xx/libOpenCL.so.1
   -DOPENCL_INCLUDE_DIRS:PATH=/usr/include/cuda
  )

SET(ENV{VALGRIND_LIB} "/usr/lib64/valgrind")
SET(CTEST_MEMORYCHECK_COMMAND /usr/bin/valgrind)
SET(CTEST_MEMORYCHECK_COMMAND_OPTIONS "--gen-suppressions=all --child-silent-after-fork=yes -q --leak-check=yes --show-reachable=yes --workaround-gcc296-bugs=yes --num-callers=50 -v")
SET(CTEST_MEMORYCHECK_SUPPRESSIONS_FILE "${CTEST_DASHBOARD_ROOT}/RTK/cmake/RTK.supp")
SET(CONFIGURE_OPTIONS ${CONFIGURE_OPTIONS}
                      -DFAST_TESTS_NO_CHECKS=TRUE)
set(dashboard_do_memcheck true)

include("${CTEST_SCRIPT_DIRECTORY}/rtk_common.cmake")

