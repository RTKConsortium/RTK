# client maintainer: simon.rit@creatis.insa-lyon.fr
set(CTEST_SITE "shiitake.clb")
set(CTEST_BUILD_NAME "Linux-64bit-Style")
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_UPDATE_COMMAND "git")
  set(dashboard_root_name "dashboard_tests")
get_filename_component(CTEST_DASHBOARD_ROOT "${CTEST_SCRIPT_DIRECTORY}/../${dashboard_root_name}" ABSOLUTE)
set(CTEST_SOURCE_DIRECTORY ${CTEST_DASHBOARD_ROOT}/RTK)
set(CTEST_BINARY_DIRECTORY ${CTEST_DASHBOARD_ROOT}/RTK-Style)
set(CTEST_NOTES_FILES "${CTEST_SCRIPT_DIRECTORY}/${CTEST_SCRIPT_NAME}")
set(ENV{ITK_DIR} "/home/srit/src/itk/lin64")
set(ENV{CUDA_BIN_PATH} "/home/srit/Download/cuda55/bin")

set(CTEST_TEST_TIMEOUT "60")
ctest_empty_binary_directory(${CTEST_BINARY_DIRECTORY})

ctest_start(Nightly)
#ctest_start(Experimental)
ctest_update()

set(KWSTYLE "/home/srit/src/kwstyle/lin64/bin/KWStyle")

file(WRITE ${CTEST_BINARY_DIRECTORY}/CMakeCache.txt "
SITE:STRING=${CTEST_SITE}
BUILDNAME:STRING=${CTEST_BUILD_NAME}
CTEST_TEST_CTEST:BOOL=${CTEST_TEST_CTEST}
DART_TESTING_TIMEOUT:STRING=${CTEST_TEST_TIMEOUT}
RTK_USE_KWSTYLE:BOOL=ON
KWSTYLE_EXECUTABLE:FILEPATH=${KWSTYLE}
")

ctest_configure(OPTIONS "${cfg_options}")
# Runs the Style Check
macro(ctest_style)
  execute_process(
    COMMAND ${KWSTYLE} -lesshtml -xml ${CTEST_BINARY_DIRECTORY}/cmake/KWStyle/RTK.kws.xml -dart . -1 1 -D ${CTEST_BINARY_DIRECTORY}/cmake/KWStyle/RTKFiles.txt
              -o ${CTEST_SOURCE_DIRECTORY}/cmake/KWStyle/RTKOverwrite.txt
    WORKING_DIRECTORY ${CTEST_BINARY_DIRECTORY}
    )
endmacro()

ctest_style()
ctest_submit()


