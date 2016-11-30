include_directories(${ITK_INCLUDE_DIRS})

find_package( GTest REQUIRED )

set ( SimpleRTKUnitTestSourceBase
  SimpleRTKTestHarness.cxx
  srtkImageCompare.cxx
)
add_library ( SimpleRTKUnitTestBase STATIC ${SimpleRTKUnitTestSourceBase} )
add_dependencies( SimpleRTKUnitTestBase  BasicFiltersSourceCode )
# Find tests generated in the binary directory
# To add a new file, just add it to this list.  Any GoogleTests inside will be automatically
# added to ctest.
set ( SimpleRTKUnitTestSource
  srtkImageTests.cxx
  srtkCommonTests.cxx
  srtkImageIOTests.cxx
  srtkBasicFiltersTests.cxx
  )

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/PythonTests)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/LuaTests)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/TclTests)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/RTests)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/RubyTests)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/JavaTests/org/rtk/simple/testing)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/CSharpTests)

# Adjust Python to run in the virtualenv
set( PythonVirtualenvHome ${SimpleRTK_BINARY_DIR}/Testing/Installation/PythonVirtualenv )

add_executable(SimpleRTKUnitTestDriver0 SimpleRTKUnitTestDriver.cxx ${SimpleRTKUnitTestSource})
target_link_libraries ( SimpleRTKUnitTestDriver0 ${GTEST_LIBRARIES} SimpleRTKUnitTestBase ${SimpleRTK_LIBRARIES} )

#
# Glob for necessary template files up front, before the foreach loop over
# the filters:
#
file ( GLOB CXX_TEMPLATE_FILES "*Template*.cxx.in" )
file ( GLOB LUA_TEMPLATE_FILES "*Template*.lua.in" )
file ( GLOB PYTHON_TEMPLATE_FILES "*Template*py.in" )
file ( GLOB TCL_TEMPLATE_FILES "*Template*.tcl.in" )
file ( GLOB R_TEMPLATE_FILES "*Template*.R.in" )
file ( GLOB RUBY_TEMPLATE_FILES "*Template*rb.in" )
file ( GLOB JAVA_TEMPLATE_FILES "*Template*.java.in" )
file ( GLOB CSHARP_TEMPLATE_FILES "*Template*.cs.in" )

#
# Generate code for the tests in different languages
#
# Break all these source files into a bunch of separate executables
# base on every n files in the list.
set ( _stride 50 )
set ( _exec_i 1 ) # exec 0 is the manual tests
set ( _i 0 )

# Disable the generated fileter for now
#foreach ( FILTERNAME ${GENERATED_FILTER_LIST} )
foreach(FILTERNAME ${NOTHING})
  if ( NOT _i LESS _stride )
    add_executable(SimpleRTKUnitTestDriver${_exec_i} SimpleRTKUnitTestDriver.cxx ${GENERATED_TEST_SOURCE})
    target_link_libraries ( SimpleRTKUnitTestDriver${_exec_i} ${GTEST_LIBRARIES} SimpleRTKUnitTestBase ${SimpleRTK_LIBRARIES} )
    math(EXPR _exec_i "${_exec_i}+1")
    set(GENERATED_TEST_SOURCE "")
    set ( _i 0 )
  endif ()

  # Set up some convenience variables
  set (filter_json_file ${SimpleRTK_SOURCE_DIR}/Code/BasicFilters/json/${FILTERNAME}.json)
  set (template_expansion_script ${SimpleRTK_SOURCE_DIR}/Utilities/ExpandTemplate.lua)
  set (template_include_dir ${SimpleRTK_SOURCE_DIR}/TemplateComponents)

  set(OUTPUT_TEST_FILENAME "${SimpleRTK_BINARY_DIR}/Testing/srtk${FILTERNAME}Test.cxx")
  add_custom_command (
    OUTPUT  ${OUTPUT_TEST_FILENAME}
    COMMAND ${CMAKE_COMMAND} -E remove -f "${OUTPUT_TEST_FILENAME}"
    COMMAND lua ${template_expansion_script} test ${filter_json_file} ${SimpleRTK_SOURCE_DIR}/Testing/srtk ${template_include_dir} TestTemplate.cxx.in "${OUTPUT_TEST_FILENAME}"
    DEPENDS ${filter_json_file} ${CXX_TEMPLATE_FILES}
    )
  add_test( NAME BasicFilters.${FILTERNAME} COMMAND SimpleRTKUnitTestDriver${_exec_i} --gtest_filter=BasicFilters.${FILTERNAME}:BasicFilters.${FILTERNAME}_* )
  set ( GENERATED_TEST_SOURCE ${GENERATED_TEST_SOURCE} ${OUTPUT_TEST_FILENAME})

  if ( WRAP_LUA )
    set(OUTPUT_TEST_FILENAME "${SimpleRTK_BINARY_DIR}/Testing/LuaTests/${FILTERNAME}Test.lua")
    add_custom_command (
      OUTPUT "${OUTPUT_TEST_FILENAME}"
      COMMAND ${CMAKE_COMMAND} -E remove -f "${OUTPUT_TEST_FILENAME}"
      COMMAND lua ${template_expansion_script} test ${filter_json_file} ${SimpleRTK_SOURCE_DIR}/Testing/Lua ${template_include_dir} TestTemplate.lua.in "${OUTPUT_TEST_FILENAME}"
      DEPENDS ${filter_json_file} ${LUA_TEMPLATE_FILES}
      )
    add_test( NAME Lua.${FILTERNAME} COMMAND SimpleRTKUnitTestDriver${_exec_i} --gtest_filter=Lua.${FILTERNAME} )
    set ( WRAPPED_GENERATED_TEST_SOURCE ${WRAPPED_GENERATED_TEST_SOURCE} ${OUTPUT_TEST_FILENAME})
  endif()

  if ( WRAP_PYTHON )
    set(OUTPUT_TEST_FILENAME "${SimpleRTK_BINARY_DIR}/Testing/PythonTests/${FILTERNAME}Test.py")
    add_custom_command (
      OUTPUT  "${OUTPUT_TEST_FILENAME}"
      COMMAND ${CMAKE_COMMAND} -E remove -f "${OUTPUT_TEST_FILENAME}"
      COMMAND lua ${template_expansion_script} test ${filter_json_file} ${SimpleRTK_SOURCE_DIR}/Testing/Python ${template_include_dir} TestTemplate.py.in "${OUTPUT_TEST_FILENAME}"
      DEPENDS ${filter_json_file} ${PYTHON_TEMPLATE_FILES}
      )
    add_test( NAME Python.${FILTERNAME} COMMAND SimpleRTKUnitTestDriver${_exec_i} --gtest_filter=Python.${FILTERNAME} )
    set ( WRAPPED_GENERATED_TEST_SOURCE ${WRAPPED_GENERATED_TEST_SOURCE} ${OUTPUT_TEST_FILENAME})
  endif()

  if ( WRAP_TCL )
    set(OUTPUT_TEST_FILENAME "${SimpleRTK_BINARY_DIR}/Testing/TclTests/${FILTERNAME}Test.tcl")
    add_custom_command (
      OUTPUT  "${OUTPUT_TEST_FILENAME}"
      COMMAND ${CMAKE_COMMAND} -E remove -f "${OUTPUT_TEST_FILENAME}"
      COMMAND lua ${template_expansion_script} test ${filter_json_file} ${SimpleRTK_SOURCE_DIR}/Testing/Tcl ${template_include_dir} TestTemplate.tcl.in "${OUTPUT_TEST_FILENAME}"
      DEPENDS ${filter_json_file} ${TCL_TEMPLATE_FILES}
      )
    add_test( NAME Tcl.${FILTERNAME} COMMAND SimpleRTKUnitTestDriver${_exec_i} --gtest_filter=Tcl.${FILTERNAME} )
    set ( WRAPPED_GENERATED_TEST_SOURCE ${WRAPPED_GENERATED_TEST_SOURCE} ${OUTPUT_TEST_FILENAME})
  endif()

  if ( WRAP_R )
    set(OUTPUT_TEST_FILENAME "${SimpleRTK_BINARY_DIR}/Testing/RTests/${FILTERNAME}Test.R")
    add_custom_command (
      OUTPUT  "${OUTPUT_TEST_FILENAME}"
      COMMAND ${CMAKE_COMMAND} -E remove -f "${OUTPUT_TEST_FILENAME}"
      COMMAND lua ${template_expansion_script} test ${filter_json_file} ${SimpleRTK_SOURCE_DIR}/Testing/R ${template_include_dir} TestTemplate.R.in "${OUTPUT_TEST_FILENAME}"
      DEPENDS ${filter_json_file} ${R_TEMPLATE_FILES}
      )
    add_test( NAME R.${FILTERNAME} COMMAND SimpleRTKUnitTestDriver${_exec_i} --gtest_filter=R.${FILTERNAME} )
    set ( WRAPPED_GENERATED_TEST_SOURCE ${WRAPPED_GENERATED_TEST_SOURCE} ${OUTPUT_TEST_FILENAME})
  endif()

  if ( WRAP_RUBY )
    set(OUTPUT_TEST_FILENAME "${SimpleRTK_BINARY_DIR}/Testing/RubyTests/${FILTERNAME}Test.rb")
    add_custom_command (
      OUTPUT  "${OUTPUT_TEST_FILENAME}"
      COMMAND ${CMAKE_COMMAND} -E remove -f "${OUTPUT_TEST_FILENAME}"
      COMMAND lua ${template_expansion_script} test ${filter_json_file} ${SimpleRTK_SOURCE_DIR}/Testing/Ruby ${template_include_dir} TestTemplate.rb.in "${OUTPUT_TEST_FILENAME}"
      DEPENDS ${filter_json_file} ${RUBY_TEMPLATE_FILES}
      )
    add_test( NAME Ruby.${FILTERNAME} COMMAND SimpleRTKUnitTestDriver${_exec_i} --gtest_filter=Ruby.${FILTERNAME} )
    set ( WRAPPED_GENERATED_TEST_SOURCE ${WRAPPED_GENERATED_TEST_SOURCE} ${OUTPUT_TEST_FILENAME})
  endif()

  # Java Tests
  if ( WRAP_JAVA )
    set(OUTPUT_TEST_FILENAME "${SimpleRTK_BINARY_DIR}/Testing/JavaTests/org/rtk/simple/testing/${FILTERNAME}Test.java")
    add_custom_command (
      OUTPUT  "${OUTPUT_TEST_FILENAME}"
      COMMAND ${CMAKE_COMMAND} -E remove -f "${OUTPUT_TEST_FILENAME}"
      COMMAND lua ${template_expansion_script} test ${filter_json_file} ${SimpleRTK_SOURCE_DIR}/Testing/Java ${template_include_dir} TestTemplate.java.in "${OUTPUT_TEST_FILENAME}"
      COMMAND ${Java_JAVAC_EXECUTABLE} -classpath ${SimpleRTK_BINARY_DIR}/Wrapping/${JAR_FILE} ${SimpleRTK_BINARY_DIR}/Testing/JavaTests/org/rtk/simple/testing/${FILTERNAME}Test.java
      DEPENDS ${filter_json_file} ${JAVA_TEMPLATE_FILES} ${SWIG_MODULE_SimpleRTKJava_TARGET_NAME}
      )
    add_test( NAME Java.${FILTERNAME} COMMAND SimpleRTKUnitTestDriver${_exec_i} --gtest_filter=Java.${FILTERNAME} )
    set ( WRAPPED_GENERATED_TEST_SOURCE ${WRAPPED_GENERATED_TEST_SOURCE} ${OUTPUT_TEST_FILENAME})
  endif()

  # C# Tests
  if ( WRAP_CSHARP )
    set( OUTPUT_TEST_FILENAME "${SimpleRTK_BINARY_DIR}/Testing/CSharpTests/Test${FILTERNAME}.cs" )
    if ( WIN32 )
      string( REPLACE "/" "\\" OUTPUT_TEST_FILENAME_SAFE ${OUTPUT_TEST_FILENAME} )
    else ()
      string( REPLACE "\\" "/" OUTPUT_TEST_FILENAME_SAFE ${OUTPUT_TEST_FILENAME} )
    endif ()
    add_custom_command (
      OUTPUT "${OUTPUT_TEST_FILENAME}"
      COMMAND ${CMAKE_COMMAND} -E remove -f "${OUTPUT_TEST_FILENAME}"
      COMMAND lua ${template_expansion_script} test ${filter_json_file} ${SimpleRTK_SOURCE_DIR}/Testing/CSharp ${template_include_dir} TestTemplate.cs.in "${OUTPUT_TEST_FILENAME}"
      COMMAND ${CSHARP_COMPILER}
        /t:exe /platform:${CSHARP_PLATFORM}
        /lib:${CSHARP_BINARY_DIRECTORY} /reference:System.dll /reference:SimpleRTKCSharpManaged.dll
        /out:${CSHARP_BINARY_DIRECTORY}/Test${FILTERNAME}.exe ${OUTPUT_TEST_FILENAME_SAFE}
      DEPENDS ${filter_json_file} ${CSHARP_TEMPLATE_FILES}
      )
    add_test( NAME CSharp.${FILTERNAME} COMMAND SimpleRTKUnitTestDriver${_exec_i} --gtest_filter=CSharp.${FILTERNAME} )
    set ( WRAPPED_GENERATED_TEST_SOURCE ${WRAPPED_GENERATED_TEST_SOURCE} ${OUTPUT_TEST_FILENAME})
  endif()

  math(EXPR _i "${_i}+1")

endforeach()

add_executable(SimpleRTKUnitTestDriver${_exec_i} SimpleRTKUnitTestDriver.cxx ${GENERATED_TEST_SOURCE})
target_link_libraries ( SimpleRTKUnitTestDriver${_exec_i} ${GTEST_LIBRARIES} SimpleRTKUnitTestBase ${SimpleRTK_LIBRARIES} )

# Build Google Test
find_package(Threads)
if (CMAKE_USE_PTHREADS_INIT)  # The pthreads library is available.
  set(cxx_base_flags "${cxx_base_flags} -DGTEST_HAS_PTHREAD=1")
endif()

# Test data directory
set(TEST_HARNESS_TEMP_DIRECTORY ${SimpleRTK_BINARY_DIR}/Testing/Temporary)
set(TEST_HARNESS_DATA_DIRECTORY ${SimpleRTK_BINARY_DIR}/ExternalData/Testing/Data)

# Set some variables
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SimpleRTKTestHarnessPaths.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/SimpleRTKTestHarnessPaths.h ESCAPE_QUOTES)

include_directories ( 
${GTEST_INCLUDE_DIRS} 
${CMAKE_CURRENT_SOURCE_DIR} 
${CMAKE_CURRENT_BINARY_DIR}
${SimpleRTK_INCLUDE_DIRS} 
)

add_custom_target(WrappedGeneratedTests ALL
  DEPENDS ${WRAPPED_GENERATED_TEST_SOURCE}
)

# Add org.rtk.simple.jar dependency if necessary
if( WRAP_JAVA )
  add_dependencies(WrappedGeneratedTests org_rtk_simple_jar)
endif()

# Add C# dependency if necessary
if( WRAP_CSHARP )
  add_dependencies(WrappedGeneratedTests SimpleRTKCSharpManaged)
endif()

add_executable( srtkSystemInformationTest srtkSystemInformationTest.cxx )
target_link_libraries( srtkSystemInformationTest ${SimpleRTK_LIBRARIES} )
add_test( NAME srtkSystemInformaionTest COMMAND srtkSystemInformationTest ${CMAKE_BINARY_DIR} )


# Add all the tests by parsing the source code
# This macro searches for GoogleTest macros and adds them as test automatically
macro(ADD_GOOGLE_TESTS target)
  # Add the generated tests
  set ( AllTestsHits "" )
  set ( LongTestsHits "" )

  foreach ( source ${ARGN} )
    file(READ "${source}" contents)

    # Find all test and long test lists
    string(REGEX MATCHALL "TEST_?F?\\(([A-Za-z_0-9 ,_]+)\\) /\\* Long \\*/" LongTests ${contents})
    string(REGEX MATCHALL "TEST_?F?\\(([A-Za-z_0-9 ,_]+)\\)" AllTests ${contents})

    # Convert the C++ code into a short test name
    foreach(hit ${AllTests})
      string(REGEX REPLACE ".*\\( *( *[A-Za-z_0-9]+)[, ]*([A-Za-z_0-9]+) *\\).*" "\\1.\\2" test_name ${hit})
      set ( AllTestsHits ${AllTestsHits} ${test_name} )
    endforeach()
    foreach(hit ${LongTests})
      string(REGEX REPLACE ".*\\(( *[A-Za-z_0-9]+)[, ]*([A-Za-z_0-9]+) *\\).*" "\\1.\\2" test_name ${hit})
      set ( LongTestsHits ${LongTestsHits} ${test_name} )
    endforeach()
  endforeach()

  # If we are not running the long tests, remove them from the AllTests list
  if ( NOT RUN_LONG_TESTS )
    foreach ( test ${LongTestsHits} )
      list ( REMOVE_ITEM AllTestsHits ${test} )
    endforeach()
  endif ()

  list ( SORT AllTestsHits )
  list ( REMOVE_DUPLICATES AllTestsHits )
  # Add all the remaining tests to CTest's test list
  foreach(hit ${AllTestsHits})
    # Take the first item in gtest list as the name
    string( REGEX MATCH "[^:]+" name ${hit} )
    add_test( NAME ${name} COMMAND ${target} --gtest_filter=${hit} )
  endforeach()
endmacro()

add_google_tests(SimpleRTKUnitTestDriver0 ${SimpleRTKUnitTestSource})


#######################################################
#
# Tests which use the itkTestDriver
#
#######################################################

set( RTK_TEST_DRIVER  "rtkTestDriver" )

#
# CXX Examples
#
#if ( ${BUILD_EXAMPLES} )
#  add_test(NAME CXX.Example.SimpleGaussian
#    COMMAND "${RTK_TEST_DRIVER}"
#    --compare-MD5
#    "${TEST_HARNESS_TEMP_DIRECTORY}/CXX.SimpleGaussian.nrrd"
#    5ef06ed0d5d8952c770348699b70f23e
#    $<TARGET_FILE:SimpleGaussian>
#    "${TEST_HARNESS_DATA_DIRECTORY}/Input/RA-Short.nrrd"
#    2.0
#    "${TEST_HARNESS_TEMP_DIRECTORY}/CXX.SimpleGaussian.nrrd"
#    )
#endif()

if ( WRAP_CSHARP )

  #
  # This is a function which compiles the program and set up the
  # enviroment for executing CSharp examples and tests.
  #
  function(add_csharp_test name csharp_file)

    if ( NOT ${BUILD_EXAMPLES} AND "${name}" MATCHES "^Example." )
      return()
    endif()

    # the root is with out extension or path
    get_filename_component( CSHARP_EXECUTABLE ${csharp_file} NAME_WE )

    # make sure the executable has CSharp in it
    if ( NOT CSHARP_EXECUTABLE MATCHES "CSharp" )
      set( CSHARP_EXECUTABLE "CSharp${CSHARP_EXECUTABLE}" )
    endif()

    # add the target to compile the test
    csharp_add_executable(
        "${CSHARP_EXECUTABLE}"
        SimpleRTKCSharpManaged.dll
        ${csharp_file}
    )

    # because each executable is it's own target we actually don't
    # need to make a target depend on this list
    list( APPEND compiled_csharp_tests "${CSHARP_BINARY_DIRECTORY}/${CSHARP_EXECUTABLE}.exe")

    add_dependencies("${CSHARP_EXECUTABLE}" SimpleRTKCSharpManaged)

    # the interpreter is set to "" when none is needed
    if( CSHARP_INTERPRETER )
      add_test(NAME CSharp.${name}
        COMMAND "${RTK_TEST_DRIVER}"
        "${CSHARP_INTERPRETER}"
        "${CSHARP_BINARY_DIRECTORY}/${CSHARP_EXECUTABLE}.exe"
        ${ARGN}
        )
    else ()
      add_test(NAME CSharp.${name}
        COMMAND "${RTK_TEST_DRIVER}"
        "${CSHARP_BINARY_DIRECTORY}/${CSHARP_EXECUTABLE}.exe"
        ${ARGN}
        )
    endif()

  endfunction()

  #
  # CSharp Examples
  #
  add_csharp_test( Example.SimpleGaussian
    "${SimpleRTK_SOURCE_DIR}/Examples/RTKFirstReconstruction.cs"
#    --compare
#      "${TEST_HARNESS_TEMP_DIRECTORY}/CSharp.SimpleGaussian.nrrd"
#      "${TEST_HARNESS_DATA_DIRECTORY}/Baseline/Example_SimpleGaussian_2.0.nrrd"
    "${TEST_HARNESS_TEMP_DIRECTORY}/CSharp.RTKFirstReconstruction.mha" )
  #
  # CSharp Tests
  #
  add_csharp_test( Test.ImageTest
    "${SimpleRTK_SOURCE_DIR}/Testing/CSharpImageTest.cs" )

endif ()


if ( WRAP_JAVA )

  #
  # This is a function which set up the enviroment for executing JAVA examples and tests
  #
  function(add_java_test name java_file)

    if ( NOT ${BUILD_EXAMPLES} AND "${name}" MATCHES "^Example." )
      return()
    endif()

    # the root is with out extension or path, it is also assumed to the the name of the main class
    get_filename_component( _java_class ${java_file} NAME_WE )
    set( _java_file_class "${_java_class}.class" )

    if(WIN32)
      set( _JAVA_LIBRARY_PATH "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<CONFIGURATION>" )
      # Note: on windows this is a semi-colon separated list
      set( _JAVA_CLASSPATH "${SimpleRTK_BINARY_DIR}/Wrapping/${JAR_FILE};${CMAKE_CURRENT_BINARY_DIR}" )
    else()
      set( _JAVA_LIBRARY_PATH "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" )
      set( _JAVA_CLASSPATH "${SimpleRTK_BINARY_DIR}/Wrapping/${JAR_FILE}:${CMAKE_CURRENT_BINARY_DIR}" )
    endif()

    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${_java_file_class}"
      COMMAND "${Java_JAVAC_EXECUTABLE}"
      ARGS -classpath "${_JAVA_CLASSPATH}"
      -d "${CMAKE_CURRENT_BINARY_DIR}"
      "${java_file}"
      DEPENDS ${java_file} ${SWIG_MODULE_SimpleRTKJava_TARGET_NAME} org_rtk_simple_jar
      )

    list( APPEND compiled_java_tests "${CMAKE_CURRENT_BINARY_DIR}/${_java_file_class}")

    # set in parent scope to accumulate list during function execution
    set( compiled_java_tests ${compiled_java_tests} PARENT_SCOPE)

    add_test(NAME Java.${name}
      COMMAND "${RTK_TEST_DRIVER}"
      "${Java_JAVA_EXECUTABLE}"
      "-Djava.library.path=${_JAVA_LIBRARY_PATH}"
      "-classpath" "${_JAVA_CLASSPATH}"
      "${_java_class}"
      ${ARGN}
      )
  endfunction()

  add_java_test( ProcessObjectTest
    "${SimpleRTK_SOURCE_DIR}/Testing/srtkProcessObjectTest.java" )

  #
  # JAVA Examples
  #
  add_java_test( Example.SimpleGaussian
    "${SimpleRTK_SOURCE_DIR}/Examples/SimpleGaussian.java"
    --compare
      "${TEST_HARNESS_TEMP_DIRECTORY}/Java.SimpleGaussian.nrrd"
      "${TEST_HARNESS_DATA_DIRECTORY}/Baseline/Example_SimpleGaussian_2.0.nrrd"
    "${TEST_HARNESS_DATA_DIRECTORY}/Input/RA-Short.nrrd"
    "2.0"
    "${TEST_HARNESS_TEMP_DIRECTORY}/Java.SimpleGaussian.nrrd" )

  add_java_test( Example.FilterProgressReporting
    "${SimpleRTK_SOURCE_DIR}/Examples/FilterProgressReporting.java"
    "${TEST_HARNESS_DATA_DIRECTORY}/Input/RA-Short.nrrd"
    "2.0"
    "${TEST_HARNESS_TEMP_DIRECTORY}/Java.FilterProgressReporting.nrrd"
    )
  set_tests_properties( Java.Example.FilterProgressReporting
    PROPERTIES PASS_REGULAR_EXPRESSION "DiscreteGaussianImageFilter Progress" )

# This test currently does not compile or work
#
#  add_java_test( Example.ImageConnection
#    "${SimpleRTK_SOURCE_DIR}/Examples/ImageConnection.java" )


  # after all add_java_tests are called, form a target so all the code
  # is built
  add_custom_target( CompileJavaTests ALL DEPENDS ${compiled_java_tests} )

endif ()


if ( WRAP_PYTHON )

  #
  # This is a function which set up the enviroment for executing python examples and tests
  #
  function(add_python_test name)

    if ( NOT ${BUILD_EXAMPLES} AND "${name}" MATCHES "^Example." )
      return()
    endif()

    set(command "${VIRTUAL_PYTHON_EXECUTABLE}")

    # add extra command which may be needed on some systems
    if(CMAKE_OSX_ARCHITECTURES)
      list(GET CMAKE_OSX_ARCHITECTURES 0 test_arch)
      set(command arch -${test_arch} ${command})
    endif()

    add_test(NAME Python.${name}
      COMMAND "${RTK_TEST_DRIVER}"
      --add-before-env SITK_NOSHOW "YES"
      ${command}
      ${ARGN}
      )
  endfunction()

  #
  # Python Examples
  #
  add_python_test( Example.SimpleGaussian
    "${SimpleRTK_SOURCE_DIR}/Examples/SimpleGaussian.py"
    --compare
      "${TEST_HARNESS_TEMP_DIRECTORY}/Python.SimpleGaussian.nrrd"
      "${TEST_HARNESS_DATA_DIRECTORY}/Baseline/Example_SimpleGaussian_2.0.nrrd"
    "${TEST_HARNESS_DATA_DIRECTORY}/Input/RA-Short.nrrd"
    "2.0"
    "${TEST_HARNESS_TEMP_DIRECTORY}/Python.SimpleGaussian.nrrd" )

  add_python_test( Example.SimpleGaussianProcedural
    "${SimpleRTK_SOURCE_DIR}/Examples/SimpleGaussianProcedural.py"
    --compare
      "${TEST_HARNESS_TEMP_DIRECTORY}/Python.SimpleGaussianProcedural.nrrd"
      "${TEST_HARNESS_DATA_DIRECTORY}/Baseline/Example_SimpleGaussian_2.0.nrrd"
    "${TEST_HARNESS_DATA_DIRECTORY}/Input/RA-Short.nrrd"
    "2.0"
    "${TEST_HARNESS_TEMP_DIRECTORY}/Python.SimpleGaussianProcedural.nrrd" )

  add_python_test( Example.ImageCreateAndSet
    "${SimpleRTK_SOURCE_DIR}/Examples/ImageCreateAndSet.py" )

  add_python_test( Example.CannyEdge
    "${SimpleRTK_SOURCE_DIR}/Examples/CannyEdge.py"
    "${TEST_HARNESS_DATA_DIRECTORY}/Input/OAS1_0001_MR1_mpr-1_anon.nrrd" )

  add_python_test( Example.ExtractSlice
    "${SimpleRTK_SOURCE_DIR}/Examples/ExtractSlice.py"
    "${TEST_HARNESS_DATA_DIRECTORY}/Input/OAS1_0001_MR1_mpr-1_anon.nrrd"
    "30"
    "${TEST_HARNESS_TEMP_DIRECTORY}/Python.ExtractSlice.nrrd")

  add_python_test( Example.BoarderSegmentation
    "${SimpleRTK_SOURCE_DIR}/Examples/BoarderSegmentation.py"
    "${TEST_HARNESS_DATA_DIRECTORY}/Input/OAS1_0001_MR1_mpr-1_anon.nrrd"
    "${TEST_HARNESS_TEMP_DIRECTORY}/Python.BoarderSegmentation.nrrd"
    "5000"
    "150" )

 add_python_test( Example.N4BiasFieldCorrection
    "${SimpleRTK_SOURCE_DIR}/Examples/N4BiasFieldCorrection.py"
    "${TEST_HARNESS_DATA_DIRECTORY}/Input/OAS1_0001_MR1_mpr-1_anon.nrrd"
    "${TEST_HARNESS_TEMP_DIRECTORY}/Python.Example.N4BiasFieldCorrection.nrrd"
    "1" )

  add_python_test( Example.ReadAndShow
    "${SimpleRTK_SOURCE_DIR}/Examples/ReadAndShow.py"
    "${TEST_HARNESS_DATA_DIRECTORY}/Input/VM1111Shrink-RGB.png" )

  add_python_test( Example.GeodesicActiceContourSegmentation
    "${SimpleRTK_SOURCE_DIR}/Examples/GeodesicActiceContourSegmentation.py"
    "${TEST_HARNESS_DATA_DIRECTORY}/Input/cthead1.png"
    "${TEST_HARNESS_TEMP_DIRECTORY}/Python.Example.GeodesicActiceContourSegmentation.nrrd"
    .3 10 .9 50 55
    )

  add_python_test( Example.FFTConvolution
    "${SimpleRTK_SOURCE_DIR}/Examples/FFTConvolution.py"
    "${TEST_HARNESS_DATA_DIRECTORY}/Input/cthead1-Float.mha"
    "${TEST_HARNESS_DATA_DIRECTORY}/Input/Gaussian_1.5.nrrd"
    "${TEST_HARNESS_TEMP_DIRECTORY}/Python.Examples.FFTConvolution.nrrd"
    )

  add_python_test( Example.FilterProgressReporting
    "${SimpleRTK_SOURCE_DIR}/Examples/FilterProgressReporting.py"
    "${TEST_HARNESS_DATA_DIRECTORY}/Input/RA-Short.nrrd"
    "2.0"
    "${TEST_HARNESS_TEMP_DIRECTORY}/PythonFilterProgressReporting.nrrd"
    )
  set_tests_properties( Python.Example.FilterProgressReporting
    PROPERTIES PASS_REGULAR_EXPRESSION "DiscreteGaussianImageFilter Progress" )

  add_python_test( Example.Segmentation.ConnectedThresholdImageFilter1
    "${SimpleRTK_SOURCE_DIR}/Examples/Segmentation/ConnectedThresholdImageFilter.py"
    --compare-MD5
      "${TEST_HARNESS_TEMP_DIRECTORY}/PythonSegmentationConnectedThreshold1.png"
      "d4ff9c512f9c18a84729677de800db0d"
    "${TEST_HARNESS_DATA_DIRECTORY}/Input/BrainProtonDensitySlice.png"
    "${TEST_HARNESS_TEMP_DIRECTORY}/PythonSegmentationConnectedThreshold1.png"
    150 180
    60 116
    )

  add_python_test( Example.Segmentation.ConnectedThresholdImageFilter2
    "${SimpleRTK_SOURCE_DIR}/Examples/Segmentation/ConnectedThresholdImageFilter.py"
    --compare-MD5
      "${TEST_HARNESS_TEMP_DIRECTORY}/PythonSegmentationConnectedThreshold2.png"
      "a9695df04cde5b9c5481d8c253a5b3aa"
    "${TEST_HARNESS_DATA_DIRECTORY}/Input/BrainProtonDensitySlice.png"
    "${TEST_HARNESS_TEMP_DIRECTORY}/PythonSegmentationConnectedThreshold2.png"
    210 250
    81 112
    )

  add_python_test( Example.Segmentation.NeighborhoodConnectedImageFilter1
    "${SimpleRTK_SOURCE_DIR}/Examples/Segmentation/NeighborhoodConnectedImageFilter.py"
    --compare-MD5
      "${TEST_HARNESS_TEMP_DIRECTORY}/PythonSegmentationNeighborhoodConnected1.png"
      "311d9d4b492e1ea625e3b0b295814dce"
    "${TEST_HARNESS_DATA_DIRECTORY}/Input/BrainProtonDensitySlice.png"
    "${TEST_HARNESS_TEMP_DIRECTORY}/PythonSegmentationNeighborhoodConnected1.png"
    150 180
    60 116
    )


  #
  # Python Tests
  #
  add_python_test( Test.ImageTests
    "${SimpleRTK_SOURCE_DIR}/Testing/srtkImageTests.py" )
  add_python_test( Test.ImageIndexing
    "${SimpleRTK_SOURCE_DIR}/Testing/ImageIndexingTest.py" )

  add_python_test( Test.IOTest
    "${SimpleRTK_SOURCE_DIR}/Testing/IOTest.py"
    "${TEST_HARNESS_TEMP_DIRECTORY}")

  # Numpy test.
  add_python_test( Test.Numpy
    "${SimpleRTK_SOURCE_DIR}/Testing/srtkNumpyArrayConversionTest.py"
  )

 add_python_test( Test.ProcessObject
    "${SimpleRTK_SOURCE_DIR}/Testing/srtkProcessObjectTest.py"
  )



endif()


if ( WRAP_LUA )

  #
  # This is a function which set up the enviroment for executing lua examples and tests
  #
  function(add_lua_test name)

    if ( NOT ${BUILD_EXAMPLES} AND "${name}" MATCHES "^Example." )
      return()
    endif()

    set(command "$<TARGET_FILE:SimpleRTKLua>")

    # add extra command which may be needed on some systems
    if(CMAKE_OSX_ARCHITECTURES)
      list(GET CMAKE_OSX_ARCHITECTURES 0 test_arch)
      set(command arch -${test_arch} ${command})
    endif()

    add_test(NAME Lua.${name}
      COMMAND "${RTK_TEST_DRIVER}"
      ${command}
      ${ARGN}
      )
  endfunction()


  #
  # Lua Examples
  #
  add_lua_test( Example.SimpleGaussian
    "${SimpleRTK_SOURCE_DIR}/Examples/SimpleGaussian.lua"
    --compare
      "${TEST_HARNESS_TEMP_DIRECTORY}/Lua.SimpleGaussian.nrrd"
      "${TEST_HARNESS_DATA_DIRECTORY}/Baseline/Example_SimpleGaussian_2.0.nrrd"
    "${TEST_HARNESS_DATA_DIRECTORY}/Input/RA-Short.nrrd"
    "2.0"
    "${TEST_HARNESS_TEMP_DIRECTORY}/Lua.SimpleGaussian.nrrd" )

endif ()

if ( WRAP_RUBY )

  #
  # This is a function which set up the enviroment for executing ruby examples and tests
  #
  function(add_ruby_test name)

    if ( NOT ${BUILD_EXAMPLES} AND "${name}" MATCHES "^Example." )
      return()
    endif()

    set(command "${RUBY_EXECUTABLE}")

    # add extra command which may be needed on some systems
    if(CMAKE_OSX_ARCHITECTURES)
      list(GET CMAKE_OSX_ARCHITECTURES 0 test_arch)
      set(command arch -${test_arch} ${command})
    endif()

    add_test(NAME Ruby.${name}
      COMMAND "${RTK_TEST_DRIVER}"
      --add-before-env RUBYLIB "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<CONFIGURATION>"
      --add-before-env RUBYLIB "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
      ${command}
      ${ARGN}
      )
  endfunction()


  #
  # Ruby Examples
  #
  add_ruby_test( Example.SimpleGaussian
    "${SimpleRTK_SOURCE_DIR}/Examples/SimpleGaussian.rb"
    --compare
      "${TEST_HARNESS_TEMP_DIRECTORY}/Ruby.SimpleGaussian.nrrd"
      "${TEST_HARNESS_DATA_DIRECTORY}/Baseline/Example_SimpleGaussian_2.0.nrrd"
    "${TEST_HARNESS_DATA_DIRECTORY}/Input/RA-Short.nrrd"
    "2.0"
    "${TEST_HARNESS_TEMP_DIRECTORY}/Ruby.SimpleGaussian.nrrd" )

endif ()


if ( WRAP_TCL )

  #
  # This is a function which set up the enviroment for executing TCL examples and tests
  #
  function(add_tcl_test name)

    if ( NOT ${BUILD_EXAMPLES} AND "${name}" MATCHES "^Example." )
      return()
    endif()


    set(command "$<TARGET_FILE:SimpleRTKTclsh>")

    # add extra command which may be needed on some systems
    if(CMAKE_OSX_ARCHITECTURES)
      list(GET CMAKE_OSX_ARCHITECTURES 0 test_arch)
      set(command arch -${test_arch} ${command})
    endif()

    add_test(NAME Tcl.${name}
      COMMAND "${RTK_TEST_DRIVER}"
      ${command}
      ${ARGN}
      )
  endfunction()


  #
  # TCL Examples
  #
  add_tcl_test( Example.SimpleGaussian
    "${SimpleRTK_SOURCE_DIR}/Examples/SimpleGaussian.tcl"
    --compare
      "${TEST_HARNESS_TEMP_DIRECTORY}/Tcl.SimpleGaussian.nrrd"
      "${TEST_HARNESS_DATA_DIRECTORY}/Baseline/Example_SimpleGaussian_2.0.nrrd"
    "${TEST_HARNESS_DATA_DIRECTORY}/Input/RA-Short.nrrd"
    "2.0"
    "${TEST_HARNESS_TEMP_DIRECTORY}/Tcl.SimpleGaussian.nrrd" )

endif ()

if ( WRAP_R )

  #
  # This is a function which set up the enviroment for executing R examples and tests
  #
  function(add_r_test name)

    if ( NOT ${BUILD_EXAMPLES} AND "${name}" MATCHES "^Example." )
      return()
    endif()

    set(command "${R_COMMAND}")

    # add extra command which may be needed on some systems
    if(CMAKE_OSX_ARCHITECTURES)
      list(GET CMAKE_OSX_ARCHITECTURES 0 test_arch)
      set(command arch -${test_arch} ${command})
    endif()

    add_test(NAME R.${name}
      COMMAND
      ${command}
      ${ARGN}
      )

    set_property(TEST R.${name}
                 PROPERTY ENVIRONMENT R_LIBS=${SimpleRTK_BINARY_DIR}/Wrapping/RLib/
                 )
  endfunction()


  #
  # R tests and Examples
  #
  add_r_test( Arithmetic
    "--file=${SimpleRTK_SOURCE_DIR}/Testing/RArithmeticTest.R"
     )

endif ()
