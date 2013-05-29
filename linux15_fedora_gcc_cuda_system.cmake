# client maintainer: simon.rit@creatis.insa-lyon.fr
set(CTEST_SITE "linux15.dg.creatis.insa-lyon.fr")
set(CTEST_BUILD_NAME "Linux-64bit-gcc-cuda-system")
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_DASHBOARD_ROOT "/tmp/RTK_dashboard")
set(dashboard_binary_name "RTK_lin64_gcc_cuda_system")
set(ENV{ITK_DIR} "/home/srit/src/itk/lin64-dg")
set(ENV{CUDA_BIN_PATH} "/usr/lib64/cuda/bin")
set(CTEST_BUILD_FLAGS -j8)

# OpenCL
set(CONFIGURE_OPTIONS
   -DOPENCL_LIBRARIES:PATH=/usr/lib64/nvidia/libOpenCL.so.1
   -DOPENCL_INCLUDE_DIRS:PATH=/usr/include/cuda
  )

# Coverage
SET(CTEST_COVERAGE_COMMAND "/usr/bin/gcov")
SET(ENV{CXXFLAGS} "-g -O0 -fprofile-arcs -ftest-coverage -Wno-deprecated -W -Wall")
SET(ENV{CFLAGS} "-g -O0 -fprofile-arcs -ftest-coverage -W -Wall")
SET(ENV{LDFLAGS} "-fprofile-arcs -ftest-coverage")
set(CTEST_CUSTOM_COVERAGE_EXCLUDE
    ${CTEST_CUSTOM_COVERAGE_EXCLUDE} # keep current exclude expressions
    "/cmake/"
    "/utilities/"
    )
set(CTEST_EXTRA_COVERAGE_GLOB "/code/*.h" "/code/*.cxx" "/code/*.txx")
set(dashboard_do_coverage true)

include("${CTEST_SCRIPT_DIRECTORY}/rtk_common.cmake")

