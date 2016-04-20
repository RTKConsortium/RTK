# client maintainer: simon.rit@creatis.insa-lyon.fr
set(CTEST_SITE "russula.clb")
set(CTEST_BUILD_NAME "Linux-64bit-Doxygen")
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_UPDATE_COMMAND "git")
set(dashboard_root_name "dashboard_tests")
get_filename_component(CTEST_DASHBOARD_ROOT "${CTEST_SCRIPT_DIRECTORY}/../${dashboard_root_name}" ABSOLUTE)
set(CTEST_SOURCE_DIRECTORY ${CTEST_DASHBOARD_ROOT}/RTK)
set(CTEST_BINARY_DIRECTORY ${CTEST_DASHBOARD_ROOT}/RTK-Doxygen)
set(CTEST_NOTES_FILES "${CTEST_SCRIPT_DIRECTORY}/${CTEST_SCRIPT_NAME}")
set(ENV{ITK_DIR} "/home/srit/src/itk/lin64")
set(ENV{CUDA_BIN_PATH} "/home/srit/Download/cuda55/bin")

file(WRITE ${CTEST_BINARY_DIRECTORY}/CTestCustom.cmake
  "set(CTEST_CUSTOM_WARNING_EXCEPTION ${CTEST_CUSTOM_WARNING_EXCEPTION}
  \"warning: Duplicate anchor RegistrationMetrics found\"
  \"rtkDigisensGeometryXMLFileReader.cxx:49: warning: member with no name found.\"
  \"warning: member ThreadedGenerateData belongs to two different groups. The second one found here will be ignored.\"
  \"rtkTimeProbesCollectorBase.cxx:26: warning: no uniquely matching class member found for\"
  )")
CTEST_READ_CUSTOM_FILES("${CTEST_BINARY_DIRECTORY}")

set(CTEST_TEST_TIMEOUT "200")
ctest_empty_binary_directory(${CTEST_BINARY_DIRECTORY})

ctest_start(Nightly)
ctest_update()

file(WRITE ${CTEST_BINARY_DIRECTORY}/CMakeCache.txt "
SITE:STRING=${CTEST_SITE}
BUILDNAME:STRING=${CTEST_BUILD_NAME}
CTEST_TEST_CTEST:BOOL=${CTEST_TEST_CTEST}
DART_TESTING_TIMEOUT:STRING=${CTEST_TEST_TIMEOUT}
BUILD_DOXYGEN:BOOL=ON
BUILD_DOCUMENTATION:BOOL=ON
")

ctest_configure(OPTIONS "${cfg_options}")
ctest_build(BUILD ${CTEST_BINARY_DIRECTORY} TARGET Documentation)
ctest_submit()


