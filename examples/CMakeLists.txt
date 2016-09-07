set(RTK_DIR ${RTK_BINARY_DIR})

add_subdirectory(HelloWorld)
add_subdirectory(FirstReconstruction)

# Testing of the examples
if(BUILD_TESTING)
  add_test(HelloWorldTest ${EXECUTABLE_OUTPUT_PATH}/HelloWorld)
  set_tests_properties (HelloWorldTest
    PROPERTIES 
    PASS_REGULAR_EXPRESSION "RTK Hello World!"
    )
	
  # This test can take a long time (especially in CPU)	
  if(NOT FAST_TESTS_NO_CHECKS)	
    add_test(FirstReconstructionTest ${EXECUTABLE_OUTPUT_PATH}/FirstReconstruction)
    add_test(FirstReconstructionCheckTest ${EXECUTABLE_OUTPUT_PATH}/rtkcheckimagequality output.mha)
    set_tests_properties(FirstReconstructionCheckTest PROPERTIES DEPENDS FirstReconstructionTest)
  endif()
endif()

