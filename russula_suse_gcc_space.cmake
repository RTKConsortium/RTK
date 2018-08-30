# client maintainer: simon.rit@creatis.insa-lyon.fr
set(CTEST_SITE "russula.clb")
set(CTEST_BUILD_NAME "Linux 64bit-gcc")
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(dashboard_binary_name "RTK lin64_gcc")
set(dashboard_source_name "RTK sources")
set(dashboard_git_branch "SpacesInSourceAndBinaryDirectories")
set(ENV{ITK_DIR} "/home/srit/src/itk/lin64")
set(CTEST_BUILD_FLAGS -j16)
set(ENV{CUDA_BIN_PATH} "/home/srit/Download/cuda80/bin")
include("${CTEST_SCRIPT_DIRECTORY}/rtk_common.cmake")

